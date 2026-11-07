"""Hardware availability must not turn malformed device requests into CPU runs."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from buildml.core.errors import ValidationError
from buildml.dl.metrics import resolve_device


def _hardware(monkeypatch, *, cuda=False, mps=False, count=1):
    capabilities = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: count),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    monkeypatch.setattr("buildml.dl.metrics.require_torch", lambda **kwargs: capabilities)


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("requested", ["cuda:typo", "cuda:1.5", "cuda:", "cuda:0:1"])
def test_malformed_cuda_index_is_refused_on_every_hardware_profile(available, requested, monkeypatch):
    _hardware(monkeypatch, cuda=available)
    with pytest.raises(ValidationError, match="integer"):
        resolve_device(requested)


@pytest.mark.parametrize("requested,cuda,mps,count,resolved,warning", [
    ("auto", True, True, 1, "cuda", False),
    ("auto", False, True, 0, "mps", False),
    ("auto", False, False, 0, "cpu", False),
    ("cpu", True, True, 1, "cpu", False),
    ("cuda", False, False, 0, "cpu", True),
    ("mps", False, False, 0, "cpu", True),
    ("mps", False, True, 0, "mps", False),
    ("cuda:0", True, False, 1, "cuda:0", False),
    ("cuda:4", True, False, 1, "cpu", True),
    ("cuda:-1", True, False, 1, "cpu", True),
    ("cuda:0", False, False, 0, "cpu", True),
])
def test_selection_and_fallback_disclosures_match_capabilities(
    requested, cuda, mps, count, resolved, warning, monkeypatch
):
    _hardware(monkeypatch, cuda=cuda, mps=mps, count=count)
    spec = resolve_device(requested)
    assert spec.requested == requested
    assert spec.resolved == resolved
    assert bool(spec.fallback_warning) is warning
