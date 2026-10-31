"""Industry rule feature identifiers map to the fitted design columns."""
import pytest

from buildml.core.errors import ValidationError
from buildml.symbolic.adapters.imodels_rules import _parse_imodels_rule_string


@pytest.mark.parametrize("alias", ["X0", "X_0", "feature_0"])
def test_exported_feature_index_maps_to_input_column(alias):
    predicates = _parse_imodels_rule_string(f"{alias} <= 1.5 and X1 > -2e-3", ["age", "income"])
    assert [(p.column, p.op, p.value) for p in predicates] == [
        ("age", "<=", 1.5), ("income", ">", -0.002)
    ]


@pytest.mark.parametrize("feature", ["X2", "feature_99", "unknown"])
def test_invalid_exported_feature_is_rejected(feature):
    with pytest.raises(ValidationError):
        _parse_imodels_rule_string(f"{feature} > 0", ["age", "income"])


def test_actual_feature_name_is_preserved():
    predicate, = _parse_imodels_rule_string("age >= 18", ["age", "income"])
    assert (predicate.column, predicate.op, predicate.value) == ("age", ">=", 18)


@pytest.mark.parametrize("representation", ["frame", "dict", "object"])
def test_zero_rule_coefficient_is_not_replaced_by_one(representation):
    from types import SimpleNamespace

    import pandas as pd

    from buildml.symbolic.adapters.imodels_rules import _normalize_imodels_rules

    row = {"rule": "X0 > 0", "coef": 0.0}
    if representation == "frame":
        raw = pd.DataFrame([row])
    elif representation == "dict":
        raw = [row]
    else:
        raw = [SimpleNamespace(**row)]
    assert _normalize_imodels_rules(raw, max_rules=8) == [(0, "X0 > 0", 0.0)]
