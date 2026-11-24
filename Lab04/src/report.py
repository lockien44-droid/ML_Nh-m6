"""Plot saved results without refitting models."""
import json
import pandas as pd
from matplotlib.figure import Figure


def generate(run):
    figures = run / 'figures'
    figures.mkdir(exist_ok=True)
    history = pd.read_csv(run / 'evaluation/mlp_history.csv')
    fig = Figure(figsize=(8,4), layout='tight')
    ax = fig.subplots()
    ax.plot(history.epoch, history.train_mse, label='Train')
    ax.plot(history.epoch, history.validation_mse, label='Validation')
    chosen = json.loads((run / 'evaluation/selection.json').read_text())['mlp']['best_epoch']
    ax.axvline(chosen, linestyle='--', color='grey', label=f'Selected epoch {chosen}')
    ax.set(xlabel='Epoch', ylabel='MSE (100,000 USD)^2', title='MLP learning curve')
    ax.legend()
    fig.savefig(figures / 'learning_curve.png', dpi=150)
    frame = pd.read_csv(run / 'evaluation/metrics.csv')
    for name in frame.model:
        predictions = pd.read_csv(run / f'evaluation/{name}_predictions.csv')
        fig = Figure(figsize=(10,4), layout='tight')
        left, right = fig.subplots(1,2)
        left.scatter(predictions.actual, predictions.predicted, s=4, alpha=.3)
        low = min(predictions.actual.min(), predictions.predicted.min())
        high = max(predictions.actual.max(), predictions.predicted.max())
        left.plot([low, high], [low, high], '--', color='grey')
        left.set(xlabel='Actual (100,000 USD)', ylabel='Predicted (100,000 USD)', title=name)
        right.scatter(predictions.predicted, predictions.actual-predictions.predicted, s=4, alpha=.3)
        right.axhline(0, color='grey')
        right.set(xlabel='Predicted', ylabel='Actual - predicted', title='Residuals')
        fig.savefig(figures / f'{name}_predictions.png', dpi=150)
    lines = ['# Lab04 — PyTorch MLP regression', '',
        'California Housing: median house value by census block group, unit 100,000 USD. Random train/validation/test split; scaler fitted on train only. Models retained after validation selection, no refit. MLP uses the checkpoint with minimum validation MSE.', '',
        '| Model | MSE | RMSE | MAE | R2 |', '|---|---:|---:|---:|---:|']
    for row in frame.itertuples():
        lines.append(f'| {row.model} | {row.mse:.5f} | {row.rmse:.5f} | {row.mae:.5f} | {row.r2:.5f} |')
    lines += ['', 'RMSE/MAE have unit 100,000 USD; MSE has the squared unit. These are historical area-level observations, not current individual-house valuations. Random splits do not measure geographic or future-time generalization. Target values are capped in the source dataset. One seed and a small search do not establish a universal model ranking.', '', '![Learning curve](figures/learning_curve.png)']
    for name in frame.model:
        lines += ['', f'![{name}](figures/{name}_predictions.png)']
    (run / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
