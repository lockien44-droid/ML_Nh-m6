"""Coordinate cleaning, validation selection, per-fit tracking and final test evaluation.

training/training_summary.csv records every fit; tuning/ records validation decisions;
evaluation/<model>/<variant> records the final test, never threshold selection.
"""
import argparse
import json
from pathlib import Path
from shutil import copytree
from importlib.metadata import version
import platform

import joblib
import pandas as pd
import yaml
from scipy.sparse import isspmatrix_csr, save_npz

from .evaluate import evaluate, save_errors
from .preprocessing import DATA_DIR, LAB_DIR, PREPROCESSING_DIR, main as clean_data, split_clean_data
from .experiment_tracking import training_run
from .eda import save_preprocessing_figures
from .tuning import select_models, model_scores


def train_run(run_dir, log, model_name=None):
    config = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text(encoding='utf-8'))
    config.setdefault('split', dict(test_size=.2, random_state=42))
    config['split']['stratified'] = True
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    prep = run_dir / 'preprocessing'
    log('eda')
    from .eda import save_eda
    source = DATA_DIR / 'raw/spam_ham_dataset.csv'
    save_eda(pd.read_csv(source),source,run_dir / 'figures/eda')
    log('eda','success')
    log('preprocessing')
    clean_data(prep)
    split_clean_data(prep, config['split'])
    train,test = (pd.read_csv(prep / f'{s}_clean.csv',keep_default_na=False) for s in ('train','test'))
    log('validation_baseline_and_selection')
    selection = select_models(train,run_dir / 'tuning',model_name)
    vectorizer_id = selection.vectorizer_id
    X_test = selection.vectorizer.transform(test.text_clean)
    feature_dir = prep / 'tfidf' / vectorizer_id
    feature_dir.mkdir(parents=True,exist_ok=True)
    for name,X,frame in [('subtrain',selection.X,selection.subtrain),
                         ('validation',selection.V,selection.validation),('test',X_test,test)]:
        if not isspmatrix_csr(X) or len(frame)!=X.shape[0]:
            raise ValueError(f'CSR features do not align with {name} labels.')
        save_npz(feature_dir / f'X_{name}.npz',X)
        frame[['email_id','label_num']].to_csv(feature_dir / f'y_{name}.csv',index=False)
    pd.DataFrame({'feature_name':selection.vectorizer.get_feature_names_out()}).to_csv(feature_dir / 'feature_names.csv',index=False)
    metadata = dict(vectorizer_id=vectorizer_id,fit_split='subtrain',refit=False,
        vectorizer_path=f'models/vectorizers/{vectorizer_id}.joblib',
        feature_names='feature_names.csv',split_rows={s:f'../../{s}_clean.csv' for s in ['subtrain','validation','test']})
    (feature_dir / 'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    save_preprocessing_figures(run_dir,selection.X,X_test,selection.subtrain.label_num.to_numpy(),
        test.label_num.to_numpy(),X_validation=selection.V,y_validation=selection.validation.label_num.to_numpy())
    log('save_selected_models')
    entries=[]
    for variant,models,fits in [('baseline',selection.baselines,selection.baseline_fits),
                                ('tuned',selection.models,selection.selected_fits)]:
        destination=run_dir / 'models' / variant
        destination.mkdir(parents=True,exist_ok=True)
        for name,model in models.items():
            path=destination / f'{name}.joblib'
            joblib.dump(model,path)
            entries.append(dict(model=name,variant=variant,fit_id=fits[name],vectorizer_id=vectorizer_id,
                training_path='training/training_summary.csv',
                training_history_path=f'training/{name}/history.csv' if name!='naive_bayes' else None,
                model_path=path.relative_to(run_dir).as_posix(),
                vectorizer_path=f'models/vectorizers/{vectorizer_id}.joblib',
                threshold=getattr(model,'threshold',None),fit_split='subtrain',refit=False))
    model_info=dict(schema_version=2,training_layout='compact',run_id=run_dir.name,models=entries)
    (run_dir / 'models/model_info.json').write_text(json.dumps(model_info,indent=2),encoding='utf-8')
    manifest=json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    manifest.update(schema_version=2,training_layout='compact',selected_fits=selection.selected_fits,
                    vectorizer_ids=[vectorizer_id],selection_metric='validation_macro_f1',
                    resolved_config=config,python_version=platform.python_version(),
                    library_versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib']})
    (run_dir / 'run.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    log('final_test_evaluation')
    rows=[]
    for variant,models,fits in [('baseline',selection.baselines,selection.baseline_fits),
                                ('tuned',selection.models,selection.selected_fits)]:
        for name,model in models.items():
            output=run_dir / 'evaluation' / name / variant
            predicted=model.predict(X_test)
            score=evaluate(test.label_num.to_numpy(),predicted,test.email_id,output)
            scores,score_type=model_scores(model,X_test)
            predictions=pd.read_csv(output / 'predictions.csv')
            predictions['score']=scores
            predictions['score_type']=score_type
            predictions['threshold']=getattr(model,'threshold',None)
            predictions.to_csv(output / 'predictions.csv',index=False)
            # Test-specific naming prevents confusion with validation metrics.
            (output / 'metrics.json').rename(output / 'test_metrics.json')
            (output / 'test_metrics.json').write_text(json.dumps(dict(split='test',fit_id=fits[name],
                vectorizer_id=vectorizer_id,**score),indent=2),encoding='utf-8')
            save_errors(model,X_test,test,output)
            rows.append(dict(model=name,variant=variant,fit_id=fits[name],vectorizer_id=vectorizer_id,split='test',**score))
            print(f"{variant}/{name}: accuracy={score['accuracy']:.2%}, macro F1={score['macro_f1']:.2%}")
    comparison=run_dir / 'evaluation/comparison'
    comparison.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(comparison / 'test_summary.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['logistic_regression','svm','naive_bayes'])
    args=parser.parse_args()
    with training_run(LAB_DIR / 'results',DATA_DIR / 'raw/spam_ham_dataset.csv') as (run,log):
        train_run(run,log,args.model)
        from .eda import generate
        log('reporting')
        generate(run,allow_running=True)
        log('reporting','success')
        if args.model is None:
            log('publish_compatibility_outputs')
            copytree(run / 'preprocessing',PREPROCESSING_DIR,dirs_exist_ok=True)
            # Publish matching models and vectorizers; predict resolves paths through model_info.
            copytree(run / 'models',LAB_DIR / 'models',dirs_exist_ok=True)
            (LAB_DIR / 'models/run_id.txt').write_text(run.name+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
