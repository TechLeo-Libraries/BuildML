"""Real M4 hourly fixed-origin evaluation; public holdouts, baselines, per-series metrics."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.public_data import public_file
from buildml import Session
from buildml.forecasting.predict import generate_forecast

SOURCE = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/6c1067e5a57161249b17289a565178dc7a3fb3ca/Dataset"


def read_series(path: Path) -> dict[str, np.ndarray]:
    """Retain original series IDs and omit only empty trailing cells."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        next(rows)
        return {r[0]: np.asarray([float(v) for v in r[1:] if v.strip()], dtype=float) for r in rows}


def scores(actual: np.ndarray, prediction: np.ndarray, train: np.ndarray, season: int) -> dict:
    """Report sMAPE and seasonal MASE without substituting a scale for constant series."""
    if actual.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Forecasts must be finite and aligned with every holdout timestamp")
    absolute = np.abs(actual - prediction)
    denominator = np.abs(actual) + np.abs(prediction)
    smape = float(np.mean(np.divide(200 * absolute, denominator,
                                   out=np.zeros_like(absolute), where=denominator > 0)))
    scale = float(np.mean(np.abs(train[season:] - train[:-season])))
    return {"mae": float(absolute.mean()), "smape": smape,
            "mase": float(absolute.mean() / scale) if scale > 0 else None}


def main() -> int:
    """Evaluate fixed hyperparameters without consulting official test targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path("artifacts/public-data/m4"))
    parser.add_argument("--series", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=["naive", "seasonal_naive", "lag_ridge"],
                        choices=["naive", "seasonal_naive", "lag_ridge", "lag_hgb", "ets", "arima", "sarimax"])
    parser.add_argument("--out", type=Path, default=Path("artifacts/benchmarks/m4-real.json"))
    args = parser.parse_args()
    sources = [public_file(f"{SOURCE}/{part}/Hourly-{part.lower()}.csv",
                          args.cache / f"hourly-{part.lower()}.csv", download=args.download)
               for part in ("Train", "Test")]
    training, testing = [read_series(Path(s["path"])) for s in sources]
    ids = sorted(set(training) & set(testing))
    if not 1 <= args.series <= len(ids):
        raise ValueError(f"--series must be between 1 and {len(ids)}")
    selected = sorted(np.random.default_rng(args.seed).choice(ids, args.series, replace=False))
    results = []
    for series_id in selected:
        values, official = training[series_id], testing[series_id]
        # Two historical origins plus the official competition holdout.
        horizon = len(official)
        for origin in ("historical_2", "historical_1", "official_test"):
            offset = {"historical_2": 2 * horizon, "historical_1": horizon, "official_test": 0}[origin]
            history = values[:-offset] if offset else values
            actual = values[-offset:-offset + horizon] if offset > horizon else (
                values[-horizon:] if offset else official)
            frame = pd.DataFrame({"time": pd.date_range("2000-01-01", periods=len(history), freq="h"),
                                  "value": history})
            # Session requires test membership. This unlabeled sentinel is never fitted or scored.
            frame.loc[len(frame)] = [frame["time"].iloc[-1] + pd.Timedelta(hours=1), np.nan]
            direct = np.resize(history[-24:], horizon)
            for method in dict.fromkeys(args.methods):
                started = time.perf_counter()
                session = Session.ingest(frame).set_roles({"time": "time", "value": "target"})
                # All available origin history is fitting data; no test values ingested.
                session.inject_split(train_indices=list(range(len(history))), test_indices=[len(history)])
                session.forecast.fit(method=method, horizon=horizon, seasonal_period=24,
                                     lags=[1, 2, 24, 48, 168])
                prediction = np.asarray(generate_forecast(session.forecast_plan,
                                                         horizon=horizon).predictions)
                if method == "seasonal_naive" and not np.allclose(prediction, direct):
                    raise AssertionError("BuildML seasonal-naive differs from independent baseline")
                results.append({"series": str(series_id), "origin": origin, "method": method,
                                "train_rows": len(history), "horizon": horizon,
                                "elapsed_seconds": time.perf_counter() - started,
                                **scores(actual, prediction, history, 24)})
            results.append({"series": str(series_id), "origin": origin,
                            "method": "independent_seasonal_naive", **scores(actual, direct, history, 24)})
    summary = {}
    for method in sorted({r["method"] for r in results}):
        rows = [r for r in results if r["method"] == method]
        summary[method] = {metric: float(np.mean([r[metric] for r in rows if r[metric] is not None]))
                           for metric in ("mae", "smape", "mase")}
    payload = {"dataset": "M4 Hourly", "sources": sources, "seed": args.seed,
               "methods": list(dict.fromkeys(args.methods)),
               "series_ids": [str(v) for v in selected], "n_available_series": len(ids),
               "protocol": "Fixed-origin recursive forecasts; two historical origins plus official test; no test tuning",
               "season": 24, "results": results, "summary": summary,
               "limitations": ["Hourly subset only; no claim of overall M4 superiority.",
                               "Point errors averaged by series/origin; no pooled scale comparisons."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
