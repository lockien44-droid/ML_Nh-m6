"""Run the Lab04 experiment: python -m Lab04.src.pipeline."""
from .evaluate import predict
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import platform
from time import perf_counter
import joblib
import pandas as pd
import sklearn
import torch
import yaml
from .preprocessing import LAB, FEATURES, TARGET, load_data, prepare
from .tuning import select_models
from .report import generate
from .evaluate import metrics


def run_pipeline(csv_path=None, config_path=None, output_root=None, quick=False):
    config = yaml.safe_load(Path(config_path or LAB / 'src/config.yaml').read_text(encoding='utf-8'))
    if quick:
        config['mlp'] = [dict(hidden=[32,16], learning_rate=.001, epochs=5, batch_size=64, patience=3)]
        config['decision_tree'] = config['decision_tree'][:1]
        config['random_forest'] = [dict(n_estimators=20, max_depth=12, min_samples_leaf=2)]
    torch.set_num_threads(1)
    now = lambda: datetime.now(timezone(timedelta(hours=7)))
    root = Path(output_root or LAB / 'results')
    base = now().strftime('%Y%m%d_%H%M%S') + ('_mlp_quick' if quick else '_mlp_comparison')
    sequence = 1
    while True:
        run = root / (base if sequence == 1 else f'{base}_{sequence:02d}')
        try:
            run.mkdir(parents=True)
            break
        except FileExistsError:
            sequence += 1
    started = perf_counter()
    manifest = dict(run_id=run.name, status='running', started_at=now().isoformat(), quick=quick,
        timezone='Asia/Ho_Chi_Minh', target=TARGET, target_unit='100000 USD', fit_split='train', refit=False,
        versions=dict(python=platform.python_version(), torch=str(torch.__version__), sklearn=sklearn.__version__))
    def save():
        (run / 'run.json').write_text(json.dumps(manifest,indent=2)+'\n', encoding='utf-8')
    def log(stage):
        print(stage, flush=True)
        with (run / 'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(stage=stage,time=now().isoformat()))+'\n')
    save()
    print(f'Run: {run}', flush=True)
    try:
        (run / 'config.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
        log('preprocessing')
        frame, path, duplicates = load_data(csv_path)
        manifest.update(dataset=str(path.resolve()), dataset_sha256=sha256(path.read_bytes()).hexdigest(),
                        duplicates_removed=duplicates, rows=len(frame), features=FEATURES)
        frames, arrays, scaler = prepare(frame, config, run / 'preprocessing')
        manifest['split_counts'] = {name:len(part) for name,part in frames.items()}
        model_dir, evaluation = run / 'models', run / 'evaluation'
        model_dir.mkdir(); evaluation.mkdir()
        joblib.dump(scaler, model_dir / 'scaler.joblib')
        (model_dir / 'schema.json').write_text(json.dumps(dict(features=FEATURES,target=TARGET,target_unit='100000 USD'),indent=2),encoding='utf-8')
        log('validation_selection')
        models, choices = select_models(arrays, config, evaluation)
        (evaluation / 'selection.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
        for name, model in models.items():
            if name == 'mlp':
                torch.save(dict(state_dict=model.state_dict(), input_dim=len(FEATURES), hidden=choices[name]['params']['hidden']), model_dir / 'mlp.pt')
            else:
                joblib.dump(model, model_dir / f'{name}.joblib')
        log('final_test_evaluation')
        X, y = arrays['test']
        rows = []
        for name, model in models.items():
            if name == 'mlp':
                predicted = predict(model, X)
            else:
                predicted = model.predict(X)
            rows.append(dict(model=name, **metrics(y,predicted)))
            pd.DataFrame(dict(row_id=frames['test'].row_id.to_numpy(),actual=y,predicted=predicted,residual=y-predicted)).to_csv(evaluation / f'{name}_predictions.csv',index=False)
        # A training-mean predictor provides an interpretable naive reference.
        import numpy as np
        rows.append(dict(model='mean_baseline', **metrics(y,np.full(len(y), arrays['train'][1].mean()))))
        pd.DataFrame(dict(row_id=frames['test'].row_id.to_numpy(), actual=y, predicted=arrays['train'][1].mean())).to_csv(evaluation / 'mean_baseline_predictions.csv',index=False)
        pd.DataFrame(rows).to_csv(evaluation / 'metrics.csv',index=False)
        log('report')
        generate(run)
        manifest['status'] = 'success'
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest.update(finished_at=now().isoformat(),duration_seconds=perf_counter()-started)
        save()
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', help='Local CSV with California Housing feature and target names')
    parser.add_argument('--config')
    parser.add_argument('--quick', action='store_true', help='Smoke run only; five epochs, not final experiment')
    args = parser.parse_args()
    run_pipeline(args.data,args.config,quick=args.quick)


if __name__ == '__main__':
    main()
