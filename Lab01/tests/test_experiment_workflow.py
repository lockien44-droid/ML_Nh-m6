"""Regression checks for joint selection and preserving fitted artifacts."""

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from Lab01.algorithms import LinearSVM, LogisticRegression, MultinomialNaiveBayes
from Lab01.src import pipeline, report, tuning, run_report
from Lab01.src.run_report import training_run


class ExperimentWorkflowTests(unittest.TestCase):
    def test_vietnam_run_names_collision_and_report_lookup(self):
        instant = datetime(2026, 11, 12, 19, 41, 22, tzinfo=timezone.utc)
        local = instant.astimezone(run_report.VIETNAM_TIME)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'input.csv'
            dataset.write_text('example', encoding='utf-8')
            with patch.object(run_report, 'vietnam_now', return_value=local):
                with training_run(root / 'results_runs', dataset) as (first, _):
                    self.assertEqual(first.name, '20261113_024122_baseline_tuned')
                with training_run(root / 'results_runs', dataset) as (second, _):
                    self.assertEqual(second.name, first.name + '_02')
            manifest = json.loads((first / 'run.json').read_text())
            self.assertTrue(manifest['started_at'].endswith('+07:00'))
            self.assertEqual(manifest['timezone'], 'Asia/Ho_Chi_Minh')
            self.assertEqual(manifest['status'], 'success')
            for run_id, expected in [
                (first.name, root / 'results_runs' / first.name),
                (second.name, root / 'results_runs' / second.name),
                ('20261106T172203Z-a2e0309c', root / 'legacy' / '20261106T172203Z-a2e0309c'),
            ]:
                with patch.object(report, 'RUNS', root / 'results_runs'), \
                     patch.object(report, 'LEGACY_RUNS', root / 'legacy'), \
                     patch.object(report, 'generate') as generate, \
                     patch('sys.argv', ['report', '--run-id', run_id]):
                    report.main()
                    generate.assert_called_once_with(expected)

    def test_svm_selects_configuration_after_threshold_search(self):
        fitted = []

        class Candidate:
            def __init__(self, **params):
                self.params = params

            def fit(self, X, y):
                fitted.append(self)
                return self

            def decision_function(self, X):
                # C=1 loses at threshold zero but wins after threshold selection.
                return np.array([1., 2., 3., 4.]) if self.params['C'] == 1 else np.array([-1., 1., -1., 1.])

        with TemporaryDirectory() as tmp, patch.object(tuning, 'LinearSVM', Candidate):
            model, chosen = tuning.select_svm(None, None, None, np.array([0, 0, 1, 1]),
                Path(tmp), [dict(C=1), dict(C=2)])
            self.assertEqual(len(fitted), 10)
            self.assertTrue(any(model is item for item in fitted))
            self.assertEqual(chosen['C'], 1)
            self.assertEqual(chosen['macro_f1'], 1.)
            self.assertEqual(model.threshold, 3.)

    def test_real_data_artifacts_and_report_never_refit_selected_models(self):
        # Shorten optimization only for this data-flow test; use real preprocessing,
        # selection, serialization, test evaluation and report generation.
        stage = []
        selected = []
        original_select = tuning.select_models

        def short_fit(original):
            def fit(model, X, y, **kwargs):
                self.assertNotIn('save_selected_models', stage)
                steps = model.n_iterations
                model.n_iterations = min(steps, 3)
                try:
                    return original(model, X, y, **kwargs)
                finally:
                    model.n_iterations = steps
            return fit

        def capture(*args):
            result = original_select(*args)
            selected.append(result)
            return result

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            published, reports = root / 'published', root / 'reports'
            published.mkdir()
            reports.mkdir()
            dataset = pipeline.DATA_DIR / 'raw/spam_ham_dataset.csv'
            with patch.object(pipeline, 'LAB_DIR', published), \
                 patch.object(pipeline, 'PREPROCESSING_DIR', published / 'preprocessing'), \
                 patch.object(pipeline, 'select_models', capture), \
                 patch.object(LogisticRegression, 'fit', short_fit(LogisticRegression.fit)), \
                 patch.object(LinearSVM, 'fit', short_fit(LinearSVM.fit)):
                with training_run(root / 'runs', dataset) as (run, log):
                    def record(name):
                        stage.append(name)
                        log(name)
                    pipeline.train_run(run, record)

            selection = selected[0]
            vectorizer = joblib.load(run / 'models/tfidf_vectorizer.joblib')
            self.assertEqual(vectorizer.vocabulary_, selection.vectorizer.vocabulary_)
            np.testing.assert_array_equal(vectorizer.idf_, selection.vectorizer.idf_)
            V = load_npz(run / 'preprocessing/09_tfidf/X_validation.npz')
            comparison = run / 'evaluation/comparison'
            for variant, models in [('baseline', selection.baselines), ('tuned', selection.models)]:
                for name, model in models.items():
                    folder = run / 'models/baseline' if variant == 'baseline' else run / 'models'
                    restored = joblib.load(folder / f'{name}.joblib')
                    np.testing.assert_array_equal(restored.predict(V), model.predict(selection.V))
                    source = comparison / ('baseline_validation' if variant == 'baseline' else 'selected_validation') / name
                    predictions = pd.read_csv(source / 'predictions.csv')
                    np.testing.assert_array_equal(predictions.predicted, restored.predict(V))
                    if name != 'naive_bayes':
                        np.testing.assert_array_equal(restored.weights, model.weights)
                        self.assertEqual(restored.bias, model.bias)
                        self.assertEqual(restored.threshold, model.threshold)
            y_train = pd.read_csv(run / 'preprocessing/09_tfidf/y_train.csv')
            self.assertEqual(y_train.email_id.tolist(), selection.subtrain.email_id.tolist())
            self.assertEqual(len(pd.read_csv(comparison / 'before_after.csv')), 6)
            metadata = json.loads((comparison / 'selection.json').read_text())
            self.assertFalse(metadata['refit'])
            svm_rows = pd.read_csv(comparison / 'svm_study/validation.csv')
            self.assertEqual(svm_rows.candidate_id.nunique(), 35)
            self.assertEqual(metadata['selected']['svm']['threshold'], selection.models['svm'].threshold)

            def forbidden_fit(*args, **kwargs):
                raise AssertionError('Report must not train any model')

            with patch.object(report, 'REPORTS', reports), \
                 patch.object(LogisticRegression, 'fit', forbidden_fit), \
                 patch.object(LinearSVM, 'fit', forbidden_fit), \
                 patch.object(MultinomialNaiveBayes, 'fit', forbidden_fit):
                report.generate(run)
            self.assertTrue((reports / f'report_{run.name}.pdf').exists())
            saved = pd.read_csv(comparison / 'logistic_threshold_validation.csv')
            self.assertEqual(len(report.threshold_comparison(run)), int(saved.selected_configuration.sum()))


if __name__ == '__main__':
    unittest.main()
