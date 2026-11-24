"""Classify one unseen email using saved Lab01 artifacts."""

import argparse
import csv
from datetime import datetime, timezone
import json
import re

import joblib

from .preprocessing import LAB_DIR, clean_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('logistic_regression', 'svm', 'naive_bayes'), default='logistic_regression')
    parser.add_argument('--text', required=True)
    parser.add_argument('--run-id', help='Use saved models from this experiment instead of Lab01/models.')
    args = parser.parse_args()
    cleaned = clean_text(args.text)
    if not cleaned:
        parser.error('Email phải còn nội dung sau khi làm sạch.')

    directory = LAB_DIR / 'models'
    if args.run_id:
        if not re.fullmatch(r'(?:\d{8}_\d{6}_baseline_tuned(?:_\d{2,})?|\d{8}T\d{6}Z-[0-9a-f]{8})', args.run_id):
            parser.error('Invalid run ID.')
        candidates = [LAB_DIR / root / args.run_id for root in
                      ('results', 'results_runs', 'results/runs')]
        run = next((path for path in candidates if (path / 'run.json').is_file()), None)
        if run is None:
            parser.error(f'Run not found: {args.run_id}')
        manifest = json.loads((run / 'run.json').read_text(encoding='utf-8'))
        if manifest.get('status') != 'success':
            parser.error('Prediction requires a successful run.')
        directory = run / 'models'
    marker = directory / 'run_id.txt'
    run_id = args.run_id or (marker.read_text(encoding='utf-8').strip() if marker.exists() else '')
    info_path = directory / 'model_info.json'
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding='utf-8'))
        entry = next((row for row in info['models'] if row['model']==args.model and row['variant']=='tuned'), None)
        if entry is None:
            parser.error(f'Model {args.model} was not trained in this run.')
        root = directory.parent
        model_path = root / entry['model_path']
        vectorizer_path = root / entry['vectorizer_path']
        if any(not path.resolve().is_relative_to(directory.resolve()) for path in (model_path, vectorizer_path)):
            parser.error('Artifact paths must stay inside the model directory.')
    else:
        # Historical runs used a flat models directory.
        model_path = directory / f'{args.model}.joblib'
        vectorizer_path = directory / 'tfidf_vectorizer.joblib'
    for path in (model_path,vectorizer_path):
        if not path.is_file():
            parser.error(f'Missing saved artifact: {path}')
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    X = vectorizer.transform([cleaned])
    label = int(model.predict(X)[0])
    result = {'Run_ID': run_id, 'model': args.model, 'label': 'spam' if label else 'ham'}
    if args.model == 'logistic_regression':
        result['spam_probability'] = float(model.predict_proba(X)[0])
        result['threshold'] = getattr(model, 'threshold', 0.5)
    elif args.model == 'svm':
        result['decision_score'] = float(model.decision_function(X)[0])
        result['threshold'] = getattr(model, 'threshold', 0.0)
    reports = LAB_DIR / "reports"
    reports.mkdir(exist_ok=True)
    history = reports / "inference_history.csv"
    fields = ["timestamp_utc", "Run_ID", "model", "label", "spam_probability",
              "decision_score", "threshold"]
    with history.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                         "Run_ID": run_id, **result})
    print(json.dumps(result))


if __name__ == '__main__':
    main()
