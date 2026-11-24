"""California Housing loading and train-only preprocessing."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LAB = Path(__file__).resolve().parents[1]
FEATURES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
TARGET = 'MedHouseVal'


def load_data(csv_path=None):
    path = Path(csv_path) if csv_path else LAB / 'Data/raw/california_housing.csv'
    if not path.exists():
        if csv_path:
            raise FileNotFoundError(path)
        data = fetch_california_housing(data_home=str(LAB / 'Data/cache'), as_frame=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.frame.to_csv(path, index=False)
    frame = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    frame = frame[FEATURES + [TARGET]].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame.to_numpy()).all() or len(frame) < 20:
        raise ValueError('Need at least 20 rows with finite numeric values.')
    frame.insert(0, 'row_id', np.arange(len(frame)))
    duplicates = int(frame.duplicated(FEATURES + [TARGET]).sum())
    frame = frame.drop_duplicates(FEATURES + [TARGET])
    # Conflicting duplicate inputs could otherwise occur in multiple splits.
    if frame.duplicated(FEATURES).any():
        raise ValueError('Repeated feature rows have different targets; resolve before splitting.')
    return frame, path, duplicates


def prepare(frame, config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    train, test = train_test_split(frame, test_size=config['test_fraction'], random_state=config['seed'])
    train, validation = train_test_split(train, test_size=config['validation_fraction'], random_state=config['seed'] + 1)
    scaler = StandardScaler().fit(train[FEATURES])
    frames = dict(train=train, validation=validation, test=test)
    arrays = {}
    for name, part in frames.items():
        part.to_csv(directory / f'{name}.csv', index=False)
        arrays[name] = (scaler.transform(part[FEATURES]).astype('float32'), part[TARGET].to_numpy(dtype='float32'))
    np.savez_compressed(directory / 'scaled.npz', **{f'{name}_{kind}': pair[i] for name, pair in arrays.items() for i, kind in enumerate(['X','y'])})
    return frames, arrays, scaler
