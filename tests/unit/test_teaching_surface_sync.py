"""CI-facing teaching-surface sync: Session ↔ index ↔ catalog ↔ AI tools."""

from __future__ import annotations

import json
from pathlib import Path

from buildml.explain.catalog import OPERATION_CATALOG
from buildml.explain.sync import (
    OPERATION_INDEX_PATH,
    REQUIRED_AI_TOOL_SESSION_METHODS,
    check_teaching_surface,
    public_session_operations,
    write_operation_index,
)

PHASE_C_OPS = (
    "rag_generate",
    "make_text_torch_loaders",
    "cross_validate_torch",
    "fit_torch",
    "make_torch_loaders",
    "evaluate_torch",
)


def test_teaching_surface_sync_passes() -> None:
    report = check_teaching_surface()
    assert report.ok, "\n".join(report.errors)


def test_generated_operation_index_is_checked_in() -> None:
    assert OPERATION_INDEX_PATH.is_file()
    payload = json.loads(OPERATION_INDEX_PATH.read_text(encoding="utf-8"))
    assert payload["n_operations"] == len(public_session_operations())
    assert set(payload["operations"]) == set(public_session_operations())


def test_phase_c_ops_are_cataloged_with_concepts() -> None:
    for name in PHASE_C_OPS:
        assert name in OPERATION_CATALOG
        spec = OPERATION_CATALOG[name]
        assert spec.concept_links
        assert len(spec.definition) >= 30
        assert len(spec.purpose) >= 30
    generate = OPERATION_CATALOG["rag_generate"]
    blob = " ".join(
        (
            generate.definition,
            generate.purpose,
            *generate.mechanism,
            *generate.result_reading,
        )
    ).lower()
    assert "citation" in blob


def test_required_ai_tools_cover_phase_c_session_methods() -> None:
    assert set(PHASE_C_OPS) <= REQUIRED_AI_TOOL_SESSION_METHODS


def test_write_operation_index_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "operation_index.json"
    write_operation_index(path)
    first = path.read_text(encoding="utf-8")
    write_operation_index(path)
    second = path.read_text(encoding="utf-8")
    assert first == second
