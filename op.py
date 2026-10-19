"""The operator: what it infers per message, and when it retrains.

MLOperator is the machine-learning half of Operator Lib. It loads the model
registered under this pipeline and operator from MLflow, calls infer() for every
message that matches a selector, and calls train() when there is no model yet or
when need_retraining() says so. Training runs on Ray.

infer() keeps a 15-minute bucket state. Each main-meter, EV or weather message
updates the bucket it falls into. When a message arrives in bucket b, the bucket
b-15min is complete, so its features are built from the state and a prediction
is emitted with result time b-15min. The output is therefore a 15-minute-delayed
estimate of the compressor's mean power for the bucket, not a forecast.
"""

import datetime
import typing

import numpy as np
import pandas as pd
from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from features import BUCKET_MINUTES, HISTORY_BUCKETS, feature_frame
from training import train_model

UTC = datetime.timezone.utc
BUCKET = datetime.timedelta(minutes=BUCKET_MINUTES)
EPOCH = datetime.datetime(1970, 1, 1, tzinfo=UTC)


class CustomConfig(Config):
    """Deployment configuration, typed.

    The base Config already carries mlflow_url, ray_url and ts_conn. Anything
    added here arrives from the operator's deployment config under the same name,
    with this value as the default.
    """

    # Retrain at most this often, in seconds. A day, so a deployment does not
    # spend its life training.
    retrain_after_s = 86400


def _utc(value: datetime.datetime) -> datetime.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _bucket_start(value: datetime.datetime) -> datetime.datetime:
    elapsed = value - EPOCH
    return EPOCH + (elapsed // BUCKET) * BUCKET


def _number(value) -> typing.Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


class Operator(MLOperator):
    configType = CustomConfig

    # The args are the mapping destinations the pipeline is configured with; the
    # selector name is what infer() receives, so each input is told apart by it.
    selectors = [
        Selector({"name": "main", "args": ["main_kwh"]}),
        Selector({"name": "ev", "args": ["ev_a", "ev_b", "ev_c"]}),
        Selector({"name": "weather", "args": ["temp_c"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns, so anything set after it is missing
        # during the replay.
        self.trained_at: typing.Optional[datetime.datetime] = None
        self._buckets: typing.Dict[datetime.datetime, dict] = {}
        self._last_emitted: typing.Optional[datetime.datetime] = None
        super().init(*args, **kwargs)

    def _table(self, last: datetime.datetime) -> pd.DataFrame:
        """The bucket table ending at `last`, on the same grid training uses."""
        keys = [last - BUCKET * i for i in range(HISTORY_BUCKETS - 1, -1, -1)]
        main, ev, temp = [], [], []
        for key in keys:
            record = self._buckets.get(key)
            if record is None:
                main.append(np.nan)
                ev.append(np.nan)
                temp.append(np.nan)
                continue
            main.append(np.nan if record["main_kwh"] is None else record["main_kwh"])
            ev.append(record["ev_sum"] / record["ev_n"] if record["ev_n"] else np.nan)
            temp.append(np.nan if record["temp_c"] is None else record["temp_c"])
        index = pd.DatetimeIndex([key.replace(tzinfo=None) for key in keys])
        return pd.DataFrame({"main_kwh": main, "ev_w": ev, "temp_c": temp}, index=index)

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        at = _utc(timestamp)
        current = _bucket_start(at)
        record = self._buckets.setdefault(
            current, {"main_kwh": None, "ev_sum": 0.0, "ev_n": 0, "temp_c": None})

        if selector == "main":
            value = _number(data.get("main_kwh"))
            if value is not None:
                record["main_kwh"] = value
        elif selector == "ev":
            phases = [_number(data.get(name)) for name in ("ev_a", "ev_b", "ev_c")]
            if all(phase is not None for phase in phases):
                record["ev_sum"] += sum(phases)
                record["ev_n"] += 1
        elif selector == "weather":
            value = _number(data.get("temp_c"))
            if value is not None:
                record["temp_c"] = value

        completed = current - BUCKET
        if model is None or (self._last_emitted is not None and completed <= self._last_emitted):
            return None, None, None

        features = feature_frame(self._table(completed)).iloc[[-1]]
        prediction = float(np.asarray(model.predict(features), dtype=float).ravel()[0])
        self._last_emitted = completed

        # Buckets older than the history the features need are no longer read.
        horizon = completed - BUCKET * HISTORY_BUCKETS
        for key in [key for key in self._buckets if key < horizon]:
            del self._buckets[key]

        return completed, {"prediction": prediction}, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        """Called when there is no model, or when need_retraining() said so."""
        self.trained_at = datetime.datetime.now(UTC)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        """Called after every inference, so it has to be cheap."""
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(UTC) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
