"""Classify one unseen email using saved Lab01 artifacts."""

import argparse
import csv
from datetime import datetime, timezone
import json

import joblib

from .preprocessing import LAB_DIR, clean_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('logistic_regression', 'svm', 'naive_bayes'), default='logistic_regression')
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    cleaned = clean_text(args.text)
    if not cleaned:
        parser.error('Email phải còn nội dung sau khi làm sạch.')

    directory = LAB_DIR / 'models'
    model = joblib.load(directory / f'{args.model}.joblib')
    vectorizer = joblib.load(directory / 'tfidf_vectorizer.joblib')
    X = vectorizer.transform([cleaned])
    label = int(model.predict(X)[0])
    result = {'model': args.model, 'label': 'spam' if label else 'ham'}
    if args.model == 'logistic_regression':
        result['spam_probability'] = float(model.predict_proba(X)[0])
    elif args.model == 'svm':
        result['decision_score'] = float(model.decision_function(X)[0])
        result['threshold'] = getattr(model, 'threshold', 0.0)
    reports = LAB_DIR / "reports"
    reports.mkdir(exist_ok=True)
    history = reports / "inference_history.csv"
    marker = directory / "run_id.txt"
    run_id = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
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
