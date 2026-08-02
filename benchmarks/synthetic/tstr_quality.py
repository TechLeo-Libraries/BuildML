"""TSTR quality benchmark: native copula baseline vs optional SDV backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from buildml import Session
from buildml.synthetic.catalog import synthetic_capability_matrix
from buildml.synthetic.extras import sdv_available


def _classification_frame(n: int = 420, seed: int = 0) -> pd.DataFrame:
    x, y = make_classification(
        n_samples=n,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.65, 0.35],
        random_state=seed,
    )
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    frame["y"] = y
    frame["grp"] = pd.Series(y).map({0: "A", 1: "B"})
    return frame


def _run_method(
    *,
    backend: str | None,
    method: str,
    epochs: int = 40,
    batch_size: int = 128,
) -> dict[str, object]:
    session = (
        Session.ingest(_classification_frame())
        .set_roles(
            {
                **{c: "feature" for c in [f"f{i}" for i in range(8)]},
                "grp": "feature",
                "y": "target",
            }
        )
        .split(test_size=0.25, validation_size=0.15, random_state=0)
    )
    fit_kwargs: dict[str, object] = {
        "method": method,
        "random_state": 0,
    }
    if backend is not None:
        fit_kwargs["backend"] = backend
    if method in {"ctgan", "tvae", "copulagan"}:
        fit_kwargs["epochs"] = epochs
        fit_kwargs["batch_size"] = batch_size

    fit = session.fit_synthesizer(**fit_kwargs)  # type: ignore[arg-type]
    tstr = session.evaluate_synthetic(mode="tstr", partition="test", random_state=1)
    fid = session.evaluate_synthetic(
        mode="fidelity",
        partition="test",
        eval_backend="auto",
        random_state=1,
    )
    return {
        "backend": getattr(fit, "backend", backend or "native"),
        "method": method,
        "n_rows_fitted": fit.n_rows,
        "tstr_score": tstr.metrics.get("score"),
        "tstr_gap_vs_trtr": tstr.metrics.get("tstr_gap_vs_trtr"),
        "mean_ks": fid.metrics.get("mean_ks"),
        "corr_l1": fid.metrics.get("corr_l1"),
        "sdmetrics_overall": fid.metrics.get("sdmetrics_overall"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="BuildML synthetic TSTR quality benchmark (copula baseline vs SDV)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/synthetic/results/tstr_quality.json"),
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)

    runs: list[dict[str, object]] = []
    try:
        runs.append(_run_method(backend="native", method="gaussian_copula"))
    except (OSError, ImportError, RuntimeError) as exc:
        # sdmetrics/torch DLL failures must not hard-fail the native baseline.
        runs.append(
            {
                "backend": "native",
                "method": "gaussian_copula",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    if sdv_available():
        for method in ("ctgan", "tvae", "copulagan"):
            try:
                runs.append(
                    _run_method(
                        backend="sdv",
                        method=method,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — optional SDV / torch failures
                runs.append(
                    {
                        "backend": "sdv",
                        "method": method,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    else:
        print(
            "SDV unavailable/unusable — skipping ctgan/tvae/copulagan "
            "(install buildml[synthetic-industry] on a host with working torch).",
            file=sys.stderr,
        )

    baseline = next((r for r in runs if r.get("method") == "gaussian_copula"), {})
    baseline_score = baseline.get("tstr_score")
    for row in runs:
        score = row.get("tstr_score")
        if score is not None and baseline_score is not None:
            row["tstr_delta_vs_copula"] = float(score) - float(baseline_score)

    try:
        matrix = synthetic_capability_matrix()
    except Exception as exc:  # noqa: BLE001
        matrix = {"error": f"{type(exc).__name__}: {exc}"}

    payload = {
        "capability_matrix": matrix,
        "runs": runs,
        "baseline_method": "gaussian_copula",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(runs)} run(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
