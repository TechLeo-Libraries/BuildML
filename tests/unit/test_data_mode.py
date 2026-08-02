"""DataMode honesty: no decorative out-of-core fit mode."""

from __future__ import annotations

import pytest

from buildml.core.types import DataMode, coerce_data_mode


def test_data_mode_has_memory_and_lazy_only() -> None:
    assert {m.value for m in DataMode} == {"memory", "lazy"}
    assert not hasattr(DataMode, "OUT_OF_CORE")


def test_legacy_out_of_core_coerces_to_lazy() -> None:
    assert coerce_data_mode("out_of_core") is DataMode.LAZY
    assert coerce_data_mode(DataMode.LAZY) is DataMode.LAZY


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        coerce_data_mode("streaming")
