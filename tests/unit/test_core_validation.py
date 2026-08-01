import pytest

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.core.types import ColumnRole
from buildml.core.validation import validate_role_name


def test_validate_role_name_accepts_enum_and_string() -> None:
    assert validate_role_name("target") is ColumnRole.TARGET
    assert validate_role_name(ColumnRole.FEATURE) is ColumnRole.FEATURE


def test_validate_role_name_rejects_unknown() -> None:
    with pytest.raises(ValidationError, match="Unknown column role"):
        validate_role_name("label")


def test_missing_extra_error_mentions_install_hint() -> None:
    err = MissingExtraError("polars", "Polars engine")
    assert "buildml[polars]" in str(err)
    assert err.extra == "polars"
