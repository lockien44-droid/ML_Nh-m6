"""Choose all configurations using validation only; keep fitted winners."""
from time import perf_counter
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from .train import fit_mlp
from .evaluate import metrics


def select_models(arrays, config, directory):
    X, y = arrays['train']
    V, actual = arrays['validation']
    rows, models, choices = [], {}, {}
    settings = dict(mlp=config['mlp'], linear_regression=[{}], decision_tree=config['decision_tree'], random_forest=config['random_forest'])
    for name, candidates in settings.items():
        best = float('inf')
        for index, params in enumerate(candidates):
            started = perf_counter()
            epoch = None
            if name == 'mlp':
                model, history, epoch = fit_mlp(X, y, V, actual, params, config['seed'])
                history.to_csv(directory / f'mlp_candidate_{index}_history.csv', index=False)
            else:
                cls = dict(linear_regression=LinearRegression, decision_tree=DecisionTreeRegressor, random_forest=RandomForestRegressor)[name]
                kwargs = {} if name == 'linear_regression' else dict(random_state=config['seed'])
                model = cls(**params, **kwargs).fit(X, y)
            seconds = perf_counter() - started
            result = metrics(actual, model.predict(V))
            rows.append(dict(model=name, candidate=index, fit_seconds=seconds, **result))
            if result['mse'] < best:
                best = result['mse']
                models[name] = model
                choices[name] = dict(candidate=index, params=params, best_epoch=epoch, fit_seconds=seconds, validation=result)
        if name == 'mlp':
            pd.read_csv(directory / f"mlp_candidate_{choices[name]['candidate']}_history.csv").to_csv(directory / 'mlp_history.csv', index=False)
    pd.DataFrame(rows).to_csv(directory / 'validation_metrics.csv', index=False)
    return models, choices
