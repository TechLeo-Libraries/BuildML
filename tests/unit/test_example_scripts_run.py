"""Core example scripts must exit 0 on a clean install."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"

# Fast core loops. Extra-gated scripts (torch, tda, graph) skip themselves.
CORE_SMOKE: tuple[str, ...] = (
    "classical_loan_loop.py",
    "breast_cancer_classical_loop.py",
    "leakage_cv_recipe.py",
    "cbr_knn_loop.py",
    "fairness_observational_loop.py",
    "forecast_lag_loop.py",
    "timeseries_analyze_loop.py",
    "anomaly_iforest_loop.py",
    "symbolic_rules_loop.py",
    "unsupervised_cluster_loop.py",
    "recommender_item_knn_loop.py",
    "decision_threshold_loop.py",
    "synthetic_copula_loop.py",
)


@pytest.mark.parametrize("script", CORE_SMOKE)
def test_core_example_exits_zero(script: str) -> None:
    path = EXAMPLES / script
    assert path.is_file(), path
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script} exited {result.returncode}\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    assert "Traceback" not in result.stderr
