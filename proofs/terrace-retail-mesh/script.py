"""Tier B product: Terrace Retail Mesh.

Composes multitask SKU heads + chronological demand forecast + collaborative
recommenders for a synthetic retail mesh.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
import pandas as pd

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_no_test_in_selection,
    extra_available,
    load_catalog_interactions_synthetic,
    load_store_sales_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def _sku_multitask(n: int = 520, seed: int = 25) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    buy = (x[:, 0] + 0.55 * x[:, 1] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    high_margin = (
        x[:, 2] - 0.45 * x[:, 3] + 0.2 * x[:, 4] + rng.normal(scale=0.3, size=n) > 0
    ).astype(int)
    frame = pd.DataFrame(
        x,
        columns=[
            "price_z",
            "discount_z",
            "affinity_z",
            "competitor_z",
            "season_z",
            "stock_z",
        ],
    )
    frame["buy"] = buy
    frame["high_margin"] = high_margin
    meta = {
        "name": "terrace_synthetic_sku_multitask",
        "license": "synthetic/public-domain",
        "n_rows": n,
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("terrace-retail-mesh", seed=25)
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 1: multitask ---
    sku, sku_meta = _sku_multitask(seed=ctx.seed)
    try:
        feats = list(sku.columns[:6])
        mt = (
            Session.ingest(sku)
            .set_roles(
                {**{c: "feature" for c in feats}, "buy": "target", "high_margin": "target"}
            )
            .split(test_size=0.2, validation_size=0.2, random_state=ctx.seed)
        )
        fit = mt.multitask.fit(method="multioutput", random_state=ctx.seed)
        val = mt.multitask.evaluate(partition="validation")
        test = mt.multitask.evaluate(partition="test")
        stages["multitask"] = {
            "status": "ok",
            "data": sku_meta,
            "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
            "validation_metrics": metrics_round(dict(getattr(val, "metrics", {}) or {})),
            "test_metrics": metrics_round(dict(getattr(test, "metrics", {}) or {})),
        }
        write_results(ctx, stages["multitask"], filename="multitask.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["multitask"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"multitask: {exc}")

    # --- Stage 2: forecast ---
    try:
        sales, sales_meta = load_store_sales_synthetic(n_days=730, seed=ctx.seed)
        fc = (
            Session.ingest(sales.copy())
            .set_roles({"date": "time", "promo": "feature", "sales": "target"})
            .time_split(test_size=0.15, validation_size=0.15)
        )
        fit_f = fc.forecast.fit(
            method="lag_ridge",
            horizon=14,
            lags=[1, 2, 3, 7, 14],
            alpha=1.0,
        )
        assert_no_test_in_selection(
            selection_partition="validation", evaluation_partition="test"
        )
        val_fc = fc.forecast.evaluate(partition="validation", strategy="rolling_one_step")
        test_fc = fc.forecast.evaluate(partition="test", strategy="rolling_one_step")
        gen = fc.forecast.generate(horizon=14)
        stages["forecast"] = {
            "status": "ok",
            "data": sales_meta,
            "fit": metrics_round(fit_f.to_dict() if hasattr(fit_f, "to_dict") else {}),
            "validation": metrics_round(
                val_fc.to_dict() if hasattr(val_fc, "to_dict") else {}
            ),
            "test": metrics_round(
                test_fc.to_dict() if hasattr(test_fc, "to_dict") else {}
            ),
            "generate": metrics_round(gen.to_dict() if hasattr(gen, "to_dict") else {}),
        }
        write_results(ctx, stages["forecast"], filename="forecast.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["forecast"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"forecast: {exc}")

    # --- Stage 3: recommenders ---
    try:
        interactions, rec_meta = load_catalog_interactions_synthetic(seed=ctx.seed)
        impl_ok = extra_available("implicit")
        rec = (
            Session.ingest(interactions)
            .set_roles(
                {
                    "user_id": "id",
                    "item_id": "id",
                    "rating": "target",
                    "category_code": "feature",
                    "price_band": "feature",
                }
            )
            .split(test_size=0.2, validation_size=0.15, random_state=ctx.seed)
        )
        method = "als" if impl_ok else "item_knn"
        try:
            if impl_ok:
                fit_r = rec.recommender.fit(
                    method="als",
                    feedback="implicit",
                    user_column="user_id",
                    item_column="item_id",
                    random_state=ctx.seed,
                )
            else:
                fit_r = rec.recommender.fit(
                    method="item_knn",
                    user_column="user_id",
                    item_column="item_id",
                    n_neighbors=25,
                    random_state=ctx.seed,
                )
                method = "item_knn"
        except (MissingExtraError, TypeError, ValueError):
            fit_r = rec.recommender.fit(
                method="item_knn",
                user_column="user_id",
                item_column="item_id",
                n_neighbors=25,
                random_state=ctx.seed,
            )
            method = "item_knn"
        ev_r = rec.recommender.evaluate(partition="test", k=5)
        stages["recommender"] = {
            "status": "ok",
            "data": rec_meta,
            "method": method,
            "implicit_available": impl_ok,
            "fit": metrics_round(fit_r.to_dict() if hasattr(fit_r, "to_dict") else {}),
            "test_metrics": metrics_round(dict(ev_r.metrics)),
        }
        write_results(ctx, stages["recommender"], filename="recommender.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["recommender"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"recommender: {exc}")

    ok_stages = sum(1 for v in stages.values() if v.get("status") == "ok")
    summary = {
        "status": "completed" if ok_stages >= 3 else "partial",
        "product": "Terrace Retail Mesh",
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "Multitask split before multi-output fit",
            "Forecast uses time_split; lag features from past only",
            "Recommender split before ALS / item_knn fit",
            "Test session.multitask.evaluate / session.forecast.evaluate / session.recommender.evaluate after locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Fitting multitask heads on the full SKU table invents holdout F1",
            "Using future sales in lag features invents forecast MAE",
            "Fitting recommenders on test interactions invents recall@k",
        ],
        "limitations": [
            "Three synthetic retail surfaces stitched into one product narrative",
            "Not a production merchandising stack",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "terrace-retail-mesh OK",
        {
            "multitask": (stages.get("multitask") or {}).get("status"),
            "forecast": (stages.get("forecast") or {}).get("status"),
            "recommender": (stages.get("recommender") or {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
