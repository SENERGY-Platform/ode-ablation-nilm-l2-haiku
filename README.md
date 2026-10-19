# ode-ablation-nilm-l2-haiku

## Summary

**Problem.** Estimate the electrical power drawn by the heat pump compressor, as 15-minute mean power in watts, from the site's transformer-rated main meter (factor 50). The compressor has a sub-meter, which may be used to train and evaluate but which the operator does not receive at runtime. Beyond the main meter, any covariate may be used. The criterion is mean absolute error at 15-minute resolution over 2026-02, threshold 200 W, minimise (`evaluation.yaml`).

**Data.**
- *Main meter* (`sensor.data.obis_1_8_0_f`, factored energy counter, kWh). This is the raw register `obis_1_8_0` multiplied by the factor of 50. The factor is applied once, here. The raw register would place the whole site below the compressor's own average draw, which is inconsistent with the compressor being inside the main meter. The counter steps in 0.25 kWh, so each 15-minute bucket resolves to about 1 kW of mean power.
- *Compressor sub-meter* (`root.egy_watt`, W), training target only. Its unit is not declared on the characteristic; it is treated as watts, which is consistent with its energy counter and with its on-state draw of about 2.5 to 4 kW.
- *Outdoor temperature* (`value.temperature_2m`), from the open-meteo history import at the site's coordinates. Hourly, with a stored history from 2025.
- *EV socket* was evaluated as a covariate and removed. It was a visible large load in the main meter, but the scorer matches the target by source path, and the EV and the compressor share a device-service topic. The approval step changed the EV input's role to `training` on every submission, so no run tested EV at inference. The final operator does not use it.
- *Heating-circuit meter* was inspected and found to be near zero, with small bursts; it was not used.
- *PowerGuard* has about one day of history, so it was not usable.

**Approach.**
- *Buckets.* Both training and inference reduce the streams to one row per 15-minute bucket (`features.py`): the last main-meter counter reading per bucket, the last temperature reading per bucket (forward-filled), and the compressor's mean power per bucket as the target.
- *Causal features.* The main power in a bucket is the difference of consecutive counter readings, scaled to watts. Features are that value, its one-, two- and four-bucket lags, a one-hour rolling mean, the same bucket one day earlier, temperature, hour of day and weekday. Every feature at bucket *k* uses only buckets at or before *k*.
- *Model.* Gradient-boosted regression (`HistGradientBoostingRegressor`) with an absolute-error objective, so the fit optimises the criterion's metric. Training uses the previous 365 days of history. The last 30 days are held out to report a validation error; the registered model is then refitted on all of it.
- *Operator.* `op.py` keeps a bucket state from the main-meter and temperature messages. When a message arrives in bucket *b*, bucket *b − 15 min* is complete and its features are used to emit one prediction, with that bucket's start as its result time. The output is therefore a 15-minute-delayed estimate of the bucket's mean compressor power, not a forecast.

**Results.** Final configuration (commit `b09993b`, run `pmrfw5bk3tymzfwghvdvzj2emi`, MLflow run `711bfe54cf17478e87f98719ddbbd70c`):

| Quantity | Value |
|---|---|
| February 2026 MAE (evaluation, 2,661 buckets) | **973.5 W** |
| Threshold | 200 W |
| Validation MAE (last 30 days before February) | 1,005 W |
| Constant-median baseline MAE on validation | 1,884 W |

**The threshold is not met.** The model beats a constant baseline by about half on validation and generalises from validation to February with little change, so the remaining error is not overfitting. Removing the EV socket changed the February score by about 16 W, so the error is not explained by the missing covariate.

**Limitations and open questions.**
- The main meter resolves about 1 kW of mean power per 15-minute bucket, and its baseline moves by tens of kilowatts. Those two facts set a floor on what any operator can recover from this input, and it may sit above 200 W.
- Whether the error lies in the compressor's on-cycles (under-prediction) or its off-cycles (phantom load) has not been measured. The analysis is described in the run's `evaluation/predictions.csv` against the sub-meter's February values; it was not run from this session.
- The EV socket's role could not be held as `input` through the approval step in five of five attempts. Its effect at inference therefore remains untested.
- The compressor sub-meter's unit is assumed to be watts; the characteristic does not declare it.
- The output is delayed by one bucket (15 minutes). A forecast would need a different design.

**Status.** Operator complete as a scaffold of the approach above, with a measured February MAE of 973.5 W. The 200 W threshold has not been met.

---

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "features.py" | Bucketing and causal feature construction, shared by training and inference. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.10.0" and scikit-learn. |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-nilm-l2-haiku". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-nilm-l2-haiku:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.10.0", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
