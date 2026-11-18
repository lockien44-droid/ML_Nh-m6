# Lab01 training report - 20261118_212649_baseline_tuned

Models and TF-IDF fitted on subtrain; configurations and thresholds selected on validation; no refit before test. LR threshold comparison uses saved validation results.

| Model | Accuracy | Macro F1 | Precision | Recall | Spam F1 | FN | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 99.18% | 99.02% | 98.29% | 98.97% | 98.63% | 3 | 5 |
| Naive Bayes | 98.26% | 97.93% | 96.28% | 97.94% | 97.10% | 6 | 11 |
| Linear SVM | 98.46% | 98.18% | 96.00% | 98.97% | 97.46% | 3 | 12 |

![email_length_distribution](figures/20261118_212649_baseline_tuned/eda/email_length_distribution.png)

![label_distribution](figures/20261118_212649_baseline_tuned/eda/label_distribution.png)

![missing_values](figures/20261118_212649_baseline_tuned/eda/missing_values.png)

![overview](figures/20261118_212649_baseline_tuned/eda/overview.png)

![confusion_matrix_all](figures/20261118_212649_baseline_tuned/evaluation/confusion_matrix_all.png)

![confusion_matrix_lr](figures/20261118_212649_baseline_tuned/evaluation/confusion_matrix_lr.png)

![confusion_matrix_nb](figures/20261118_212649_baseline_tuned/evaluation/confusion_matrix_nb.png)

![confusion_matrix_svm](figures/20261118_212649_baseline_tuned/evaluation/confusion_matrix_svm.png)

![metrics_comparison](figures/20261118_212649_baseline_tuned/evaluation/metrics_comparison.png)

![roc_curve_all](figures/20261118_212649_baseline_tuned/evaluation/roc_curve_all.png)

![tpr_fpr_comparison](figures/20261118_212649_baseline_tuned/evaluation/tpr_fpr_comparison.png)

![cleaning_comparison](figures/20261118_212649_baseline_tuned/preprocessing/cleaning_comparison.png)

![preprocessing_steps](figures/20261118_212649_baseline_tuned/preprocessing/preprocessing_steps.png)

![tfidf_sparsity](figures/20261118_212649_baseline_tuned/preprocessing/tfidf_sparsity.png)

![tfidf_top_features](figures/20261118_212649_baseline_tuned/preprocessing/tfidf_top_features.png)

![train_test_distribution](figures/20261118_212649_baseline_tuned/preprocessing/train_test_distribution.png)

![training_configuration](figures/20261118_212649_baseline_tuned/training/training_configuration.png)

![training_dataset_overview](figures/20261118_212649_baseline_tuned/training/training_dataset_overview.png)

![training_time_comparison](figures/20261118_212649_baseline_tuned/training/training_time_comparison.png)
