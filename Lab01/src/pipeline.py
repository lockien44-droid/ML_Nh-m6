"""Run the Lab01 preprocessing, training, evaluation and reporting pipeline."""

from shutil import copytree
import json

import joblib
import pandas as pd
from scipy.sparse import isspmatrix_csr, save_npz

from .evaluate import evaluate
from .preprocessing import DATA_DIR, LAB_DIR, PREPROCESSING_DIR, main as clean_data, split_clean_data
from .run_report import save_preprocessing_figures, training_run
from .tuning import select_models


def train_run(run_dir, log):
    result_dir = run_dir / "evaluation"
    model_dir = run_dir / "models"
    prep = run_dir / "preprocessing"
    from .report import save_eda

    log("eda")
    source = DATA_DIR / "raw" / "spam_ham_dataset.csv"
    save_eda(pd.read_csv(source), source, run_dir / "figures" / "eda")
    log("eda", "success")
    log("preprocessing")
    clean_data(prep)
    split_clean_data(prep)
    train, test = (pd.read_csv(prep / f"{split}_clean.csv", keep_default_na=False)
                   for split in ("train", "test"))
    comparison_dir = result_dir / "comparison"
    log("validation_baseline_and_selection")
    selection = select_models(train, comparison_dir)
    vectorizer = selection.vectorizer
    models, tuned_models = selection.baselines, selection.models
    X_train, X_validation = selection.X, selection.V
    # Transform only: never fit again after selecting a threshold.
    X_test = vectorizer.transform(test["text_clean"])
    y_train = selection.subtrain['label_num'].to_numpy()
    y_validation = selection.validation['label_num'].to_numpy()
    y_test, email_ids = test['label_num'].to_numpy(), test['email_id']
    feature_dir = prep / '09_tfidf'
    feature_dir.mkdir(parents=True, exist_ok=True)
    selection.subtrain.to_csv(prep / 'subtrain_clean.csv', index=False)
    selection.validation.to_csv(prep / 'validation_clean.csv', index=False)
    for name, features, frame in (
        ('train', X_train, selection.subtrain),
        ('validation', X_validation, selection.validation), ('test', X_test, test)):
        if not isspmatrix_csr(features) or features.shape[0] != len(frame):
            raise ValueError(f"CSR rows and labels do not align for {name}.")
        save_npz(feature_dir / f'X_{name}.npz', features)
        frame[['email_id', 'label_num']].to_csv(feature_dir / f'y_{name}.csv', index=False)
    pd.DataFrame({'feature_name': vectorizer.get_feature_names_out()}).to_csv(
        feature_dir / 'feature_names.csv', index=False)
    joblib.dump(vectorizer, feature_dir / 'tfidf_vectorizer.joblib')
    (feature_dir / 'metadata.json').write_text(json.dumps(dict(
        fit_split='subtrain', X_train_rows='subtrain_clean.csv', refit=False), indent=2) + '\n', encoding='utf-8')
    save_preprocessing_figures(run_dir, X_train, X_test, y_train, y_test,
                               X_validation=X_validation, y_validation=y_validation)
    log("save_selected_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / 'tfidf_vectorizer.joblib')
    baseline_model_dir = model_dir / 'baseline'
    baseline_model_dir.mkdir(parents=True, exist_ok=True)
    for group, group_models in (("baseline", models), ("tuned", tuned_models)):
        destination = baseline_model_dir if group == 'baseline' else model_dir
        for name, model in group_models.items():
            joblib.dump(model, destination / f'{name}.joblib')

    log("final_test_evaluation")
    baseline, comparison = [], []
    for group, group_models in (("baseline", models), ("tuned", tuned_models)):
        for name, model in group_models.items():
            predicted = model.predict(X_test)
            output = comparison_dir / "baseline" / name if group == "baseline" else result_dir / name
            metrics = evaluate(y_test, predicted, email_ids, output)
            row = {"model": name, "split": "test", **metrics}
            if group == "baseline":
                baseline.append({**row, "predicted_spam": int(predicted.sum()),
                                 "predicted_ham": int(len(predicted) - predicted.sum())})
            else:
                comparison.append(row)
            print(f"{group}/{name}: accuracy={metrics['accuracy']:.2%}, macro F1={metrics['macro_f1']:.2%}")
    pd.DataFrame(baseline).to_csv(comparison_dir / "baseline.csv", index=False)
    pd.DataFrame(comparison).to_csv(comparison_dir / "metrics.csv", index=False)
    training_frame = pd.DataFrame(selection.training)
    training_frame.loc[training_frame['variant'].eq('tuned')].to_csv(
        comparison_dir / "training.csv", index=False)
    training_frame.loc[training_frame['variant'].eq('baseline')].to_csv(
        comparison_dir / "baseline_training.csv", index=False)
    pd.DataFrame([{**row, "variant": variant}
                  for variant, rows in (("baseline", baseline), ("tuned", comparison))
                  for row in rows]).to_csv(comparison_dir / "before_after.csv", index=False)
    log("publish_compatibility_outputs")
    copytree(prep, PREPROCESSING_DIR, dirs_exist_ok=True)
    copytree(model_dir, LAB_DIR / "models", dirs_exist_ok=True)
    (LAB_DIR / "models" / "run_id.txt").write_text(run_dir.name + "\n", encoding="utf-8")


def main():
    dataset = DATA_DIR / "raw" / "spam_ham_dataset.csv"
    with training_run(LAB_DIR / "results", dataset) as (run_dir, log):
        train_run(run_dir, log)
    from .report import generate

    generate(run_dir)


if __name__ == "__main__":
    main()
