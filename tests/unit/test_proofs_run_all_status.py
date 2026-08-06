"""Harness status classification for proof CI smoke discipline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_RUN_ALL = _REPO / "proofs" / "_lib" / "run_all.py"


def _load_run_all():
    # Load the module file directly to avoid proofs._lib package __init__
    # torch probes that can hard-crash on broken Windows wheels.
    spec = importlib.util.spec_from_file_location("buildml_proofs_run_all", _RUN_ALL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_run_all = _load_run_all()
CI_SMOKE_TIER_A = _run_all.CI_SMOKE_TIER_A
_classify_process_ok = _run_all._classify_process_ok


def test_smoke_list_is_substantially_broader_than_legacy_eight() -> None:
    assert len(CI_SMOKE_TIER_A) >= 16
    assert "loan-approval-classical" in CI_SMOKE_TIER_A
    assert "loan-fairness-observational" in CI_SMOKE_TIER_A
    assert "stream-fraud-online" in CI_SMOKE_TIER_A
    assert "eda-industry-adaptability" in CI_SMOKE_TIER_A


def test_classify_completed_ok() -> None:
    assert (
        _classify_process_ok(
            process_status="ok",
            result_status="completed",
            allow_skip=False,
        )
        == "ok"
    )


def test_classify_skip_is_unexpected_under_smoke() -> None:
    assert (
        _classify_process_ok(
            process_status="ok",
            result_status="skipped_missing_extra",
            allow_skip=False,
        )
        == "unexpected_skip"
    )


def test_classify_skip_allowed_when_flag_set() -> None:
    assert (
        _classify_process_ok(
            process_status="ok",
            result_status="skipped_missing_extra",
            allow_skip=True,
        )
        == "skipped"
    )


def test_classify_process_error_unchanged() -> None:
    assert (
        _classify_process_ok(
            process_status="error",
            result_status="completed",
            allow_skip=False,
        )
        == "error"
    )
