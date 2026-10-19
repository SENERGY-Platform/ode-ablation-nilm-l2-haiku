"""Feature construction shared by training (training.py) and inference (op.py).

Both paths reduce the streams to one row per 15-minute bucket and build the
features from that table with causal operations only, so a bucket's features
are the same whether they come from the training table or from the streaming
state that infer() keeps.

Per bucket:
  main_kwh  the last reading of the factored main-meter energy counter (kWh)
  ev_w      the mean EV socket power, summed over its three phases (W)
  temp_c    the last outdoor temperature reading, forward-filled (degC)
  target_w  the mean compressor sub-meter power (W), training only
"""

import pandas as pd

BUCKET_MINUTES = 15
HOURS_PER_BUCKET = BUCKET_MINUTES / 60
# The streaming state keeps this many buckets; the longest lag used is 96.
HISTORY_BUCKETS = 500

FEATURES = [
    "main_w",
    "main_w_lag1",
    "main_w_lag2",
    "main_w_lag4",
    "main_w_roll4",
    "main_w_day",
    "ev_w",
    "ev_w_lag1",
    "temp_c",
    "hour",
    "weekday",
]


def bucketize(frame, kind):
    """Reduce one topic's rows to one value per 15-minute bucket.

    kind is "main", "ev", "weather" or "target". Within a bucket, the main
    counter takes its last reading, the EV total and the target take their
    mean, and the temperature takes its last reading.
    """
    df = frame.copy()
    times = pd.to_datetime(df["time"])
    if getattr(times.dt, "tz", None) is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    df["time"] = times
    df = df.sort_values("time", kind="stable")
    bucket = df["time"].dt.floor(f"{BUCKET_MINUTES}min")

    if kind == "main":
        series = pd.to_numeric(df["main_kwh"], errors="coerce").groupby(bucket).last()
        name = "main_kwh"
    elif kind == "ev":
        phases = df[["ev_a", "ev_b", "ev_c"]].apply(pd.to_numeric, errors="coerce")
        total = phases.sum(axis=1, min_count=3)
        series = total.groupby(bucket).mean()
        name = "ev_w"
    elif kind == "weather":
        series = pd.to_numeric(df["temp_c"], errors="coerce").groupby(bucket).last()
        name = "temp_c"
    elif kind == "target":
        series = pd.to_numeric(df["target_w"], errors="coerce").groupby(bucket).mean()
        name = "target_w"
    else:
        raise ValueError(f"unknown bucket kind {kind!r}")
    return series.rename(name).to_frame()


def bucket_table(parts):
    """Join per-kind bucket frames onto one regular 15-minute grid."""
    table = pd.concat(parts, axis=1)
    grid = pd.date_range(table.index.min(), table.index.max(), freq=f"{BUCKET_MINUTES}min")
    return table.reindex(grid)


def feature_frame(table):
    """Causal features from a bucket table with main_kwh, ev_w and temp_c.

    Every feature at bucket k uses buckets at or before k only. The table
    must sit on a regular grid; a bucket with no data is NaN.
    """
    index = table.index
    temp = table["temp_c"].ffill()
    main_w = table["main_kwh"].diff() * (1000.0 / HOURS_PER_BUCKET)
    ev = table["ev_w"]

    f = pd.DataFrame(index=index)
    f["main_w"] = main_w
    f["main_w_lag1"] = main_w.shift(1)
    f["main_w_lag2"] = main_w.shift(2)
    f["main_w_lag4"] = main_w.shift(4)
    f["main_w_roll4"] = main_w.rolling(4, min_periods=2).mean()
    f["main_w_day"] = main_w.shift(96)
    f["ev_w"] = ev
    f["ev_w_lag1"] = ev.shift(1)
    f["temp_c"] = temp
    f["hour"] = index.hour + index.minute / 60.0
    f["weekday"] = index.dayofweek
    return f[FEATURES]
