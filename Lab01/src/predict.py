"""Classify one unseen email using saved Lab01 artifacts."""

import argparse
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
    print(json.dumps(result))


if __name__ == '__main__':
    main()
