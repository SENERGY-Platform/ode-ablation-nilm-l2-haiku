"""Training, on Ray.

Separate from op.py because the two run in different places: op.py runs in the
operator's own process for every message, while this runs distributed and rarely.

The history of each input topic is bucketed to 15 minutes on Ray, the bucket
tables are joined and turned into causal features (features.py), and a gradient
boosted regressor is fitted to the compressor sub-meter with an absolute-error
objective, so the model is trained on what the evaluation scores.
"""

import datetime

import numpy as np
import pandas as pd
import ray
from mlflow.pyfunc import PythonModel
from sklearn.ensemble import HistGradientBoostingRegressor

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger

from features import FEATURES, bucketize, bucket_table, feature_frame

# How much history one training pass reads. The device history starts in
# January 2025, so a year before the training end is what is there.
TRAINING_WINDOW = datetime.timedelta(days=365)

# The last 30 days of training history are held out to report a validation
# error; the registered model is then fitted on all of it.
VALIDATION_BUCKETS = 96 * 30

PARAMS = dict(
    loss="absolute_error",
    max_iter=400,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    random_state=0,
)


class OdeAblationNilmL2HaikuModel(PythonModel):
    """The model MLflow registers and op.py later loads.

    predict() receives the feature row infer() builds. Keep it serialisable:
    MLflow stores this object.
    """

    def __init__(self, model, features) -> None:
        self.model = model
        self.features = list(features)

    def predict(self, context, model_input=None, params=None):
        payload = model_input if model_input is not None else context
        frame = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame([payload])
        frame = frame.reindex(columns=self.features).apply(pd.to_numeric, errors="coerce")
        return np.asarray(self.model.predict(frame.astype(float)), dtype=float)


@ray.remote
def _bucket(dataset):
    """Bucket one topic's history. The topic is named by its columns."""
    frame = dataset.to_pandas()
    columns = set(frame.columns)
    if "main_kwh" in columns:
        kind = "main"
    elif "ev_a" in columns:
        kind = "ev"
    elif "temp_c" in columns:
        kind = "weather"
    elif "target_w" in columns:
        kind = "target"
    else:
        raise ValueError(f"unrecognised topic columns {sorted(columns)}")
    return bucketize(frame, kind)


@ray.remote
def _fit(X, y, n_val):
    """Fit on everything before the validation tail, report that tail, then refit on all."""
    train_X, train_y = X.iloc[:-n_val], y.iloc[:-n_val]
    val_X, val_y = X.iloc[-n_val:], y.iloc[-n_val:]

    probe = HistGradientBoostingRegressor(**PARAMS).fit(train_X, train_y)
    val_pred = probe.predict(val_X)
    metrics = {
        "val_mae_w": float(np.mean(np.abs(val_pred - val_y.to_numpy()))),
        "baseline_mae_w": float(np.mean(np.abs(val_y.to_numpy() - float(train_y.median())))),
        "train_rows": int(len(train_y)),
        "val_rows": int(len(val_y)),
    }
    final = HistGradientBoostingRegressor(**PARAMS).fit(X, y)
    return final, metrics


def train_model(logger: TrainMlflowLogger):
    """Read the history, bucket it, fit, and hand back a model for MLflow to register."""
    with logger.trace("read history"):
        inputs = provide_historic_data(TRAINING_WINDOW)
        targets = provide_historic_data(TRAINING_WINDOW, role="training")
    if not inputs or not targets:
        return None

    with logger.trace("bucket"):
        parts = ray.get([_bucket.remote(ref) for ref in inputs + targets])

    table = bucket_table(parts)
    if "main_kwh" not in table or "target_w" not in table or table["target_w"].notna().sum() == 0:
        return None

    features = feature_frame(table)
    mask = table["target_w"].notna()
    X = features[mask]
    y = table["target_w"][mask]

    n_val = min(VALIDATION_BUCKETS, len(X) // 5)
    with logger.trace("fit"):
        model, metrics = ray.get(_fit.remote(X, y, n_val))

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "bucket_minutes": 15,
        "features": ",".join(FEATURES),
        "objective": "absolute_error",
    })
    logger.log_metrics(metrics)
    return OdeAblationNilmL2HaikuModel(model=model, features=FEATURES)
