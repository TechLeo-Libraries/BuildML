"""Narrow contracts for warnings from intentionally supported legacy formats."""

import re
import warnings

import pytest


@pytest.fixture
def torchscript_deprecation_contract():
    """Validate known TorchScript notices; leave unrelated warnings visible.

    Older supported Torch versions do not emit these deprecations. TorchScript
    tests deliberately exercise the existing serialized format on both versions.
    """
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        yield
    for notice in captured:
        if issubclass(notice.category, DeprecationWarning) and str(
            notice.message
        ).startswith("`torch.jit."):
            assert re.fullmatch(
                r"`torch\.jit\.(trace|trace_method|script|load)` is deprecated\. "
                r"Please switch to `torch\.(compile` or `torch\.)?export`\.",
                str(notice.message),
            ), f"Unexpected TorchScript migration notice: {notice.message}"
        else:
            warnings.warn_explicit(
                notice.message, notice.category, notice.filename, notice.lineno
            )
