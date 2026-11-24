"""Predict from a CSV using artifacts of an explicit completed run."""
import argparse
import json
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
import torch
from .preprocessing import LAB
from .model import MLP


def predict_csv(run, input_path, model_name='mlp'):
    manifest = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'success':
        raise ValueError('Run must be successful before inference.')
    directory = run / 'models'
    schema = json.loads((directory / 'schema.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(input_path)
    X = frame[schema['features']].apply(pd.to_numeric,errors='raise')
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('Input features must be finite.')
    X = joblib.load(directory / 'scaler.joblib').transform(X).astype('float32')
    if model_name == 'mlp':
        checkpoint = torch.load(directory / 'mlp.pt', map_location='cpu', weights_only=True)
        model = MLP(checkpoint['input_dim'], checkpoint['hidden'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = joblib.load(directory / f'{model_name}.joblib')
    result = frame.copy()
    result['prediction_100000_usd'] = model.predict(X)
    result['prediction_usd'] = result['prediction_100000_usd'] * 100000
    result['run_id'] = run.name
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--model',default='mlp',choices=['mlp','linear_regression','decision_tree','random_forest'])
    args=parser.parse_args()
    if not re.fullmatch(r'\d{8}_\d{6}_mlp_(?:comparison|quick)(?:_\d{2,})?',args.run_id):
        parser.error('Invalid run ID.')
    destination=Path(args.output)
    if destination.resolve() == Path(args.input).resolve() or destination.exists():
        parser.error('Output must be a new file, different from the input.')
    result=predict_csv(LAB / 'results' / args.run_id,args.input,args.model)
    destination.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(destination,index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
