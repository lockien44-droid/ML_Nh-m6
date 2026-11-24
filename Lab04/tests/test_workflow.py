import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import joblib
import numpy as np
import pandas as pd
import yaml
from Lab04.src.preprocessing import FEATURES, TARGET, load_data
from Lab04.src.pipeline import run_pipeline
from Lab04.src.predict import predict_csv


class WorkflowTests(unittest.TestCase):
    def test_no_leakage_and_saved_predictions(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(rng.normal(size=(120,8)),columns=FEATURES)
        frame[TARGET] = 2 + frame.MedInc * .3 + frame.HouseAge ** 2 * .05
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'input.csv'; frame.to_csv(path,index=False)
            config=dict(seed=42,test_fraction=.2,validation_fraction=.2,
                mlp=[dict(hidden=[32,16],learning_rate=.01,epochs=4,batch_size=16,patience=2)],
                decision_tree=[dict(max_depth=3)],random_forest=[dict(n_estimators=5,max_depth=3)])
            cfg=root/'config.yaml'; cfg.write_text(yaml.safe_dump(config))
            run=run_pipeline(path,cfg,root/'results')
            manifest=json.loads((run/'run.json').read_text())
            self.assertEqual(manifest['status'],'success')
            splits={s:pd.read_csv(run/f'preprocessing/{s}.csv') for s in ['train','validation','test']}
            ids=[set(splits[s].row_id) for s in splits]
            self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
            self.assertEqual(len(set.union(*ids)),len(frame))
            scaler=joblib.load(run/'models/scaler.joblib')
            np.testing.assert_allclose(scaler.mean_,splits['train'][FEATURES].mean())
            test=run/'preprocessing/test.csv'
            # Checkpoints/scaler must reproduce predictions with no training at inference.
            for name in ['mlp','linear_regression','decision_tree','random_forest']:
                expected=pd.read_csv(run/f'evaluation/{name}_predictions.csv').predicted
                actual=predict_csv(run,test,name).prediction_100000_usd
                np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-5)
            self.assertTrue((run/'figures/learning_curve.png').exists())
            self.assertTrue((run/'report.md').exists())
            selection=json.loads((run/'evaluation/selection.json').read_text())
            hist=pd.read_csv(run/'evaluation/mlp_history.csv')
            self.assertEqual(selection['mlp']['best_epoch'],int(hist.loc[hist.validation_mse.idxmin(),'epoch']))

    def test_reject_nonfinite_input(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.csv'
            frame=pd.DataFrame(np.zeros((25,8)),columns=FEATURES)
            frame[TARGET]=1
            frame.loc[0,'MedInc']=np.inf
            frame.to_csv(path,index=False)
            with self.assertRaises(ValueError):
                load_data(path)


if __name__ == '__main__':
    unittest.main()
