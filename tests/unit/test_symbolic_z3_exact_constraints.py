"""Real solver checks against exact scalar predicates and unsupported-input contracts."""

import operator
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.symbolic.adapters.z3_verify import verify_rule_constraints
from buildml.symbolic.rules import Predicate, Rule, RuleKnowledgeBase

pytest.importorskip("z3")


def _knowledge(*predicates):
    return RuleKnowledgeBase(rules=(Rule("bounds", tuple(predicates), None, kind="constraint"),))


@pytest.mark.parametrize(
    "op, compare",
    [
        ("<", operator.lt),
        ("<=", operator.le),
        (">", operator.gt),
        (">=", operator.ge),
        ("==", operator.eq),
        ("!=", operator.ne),
    ],
)
@pytest.mark.parametrize(
    "point,bound",
    [
        (2, 3),
        (9007199254740992, 9007199254740993),
        (9007199254740993, 9007199254740992),
        (True, 1),
        (np.bool_(True), 1),
    ],
)
def test_numeric_sat_matches_exact_scalar_comparison(op, compare, point, bound):
    result = verify_rule_constraints(
        _knowledge(Predicate("x", "==", point), Predicate("x", op, bound)), ["x"]
    )
    expected = bool(compare(point, bound))
    assert result.satisfiable is expected
    assert result.status == ("sat" if expected else "unsat")
    assert result.n_checked == 2
    assert result.n_hard_constraints == 1


@pytest.mark.parametrize(
    "lower,upper",
    [
        (Fraction(1, 3), Fraction(1, 3) + Fraction(1, 10**20)),
        (Decimal("0.10000000000000000001"), Decimal("0.10000000000000000002")),
    ],
)
def test_nonempty_exact_intervals_are_not_rounded_to_empty(lower, upper):
    assert Fraction(lower) < (Fraction(lower) + Fraction(upper)) / 2 < Fraction(upper)
    result = verify_rule_constraints(
        _knowledge(Predicate("x", ">", lower), Predicate("x", "<", upper)), ["x"]
    )
    assert result.status == "sat"
    assert result.satisfiable is True
    assert result.n_checked == 2


@pytest.mark.parametrize(
    "predicate",
    [
        Predicate("x", "==", "red"),
        Predicate("x", "in", [1, 2]),
        Predicate("x", "not_in", [1, 2]),
        Predicate("x", "isna"),
        Predicate("x", "==", float("nan")),
        Predicate("x", "==", float("inf")),
        Predicate("missing", ">", 2),
    ],
)
def test_unencodable_predicates_are_skipped_without_a_conversion_crash(predicate):
    result = verify_rule_constraints(_knowledge(predicate), ["x"])
    assert result.status == "skipped_non_numeric"
    assert result.satisfiable is None
    assert result.n_checked == 0


def test_partial_numeric_sat_discloses_unchecked_predicates():
    result = verify_rule_constraints(
        _knowledge(Predicate("x", ">=", 0), Predicate("category", "==", "red")),
        ["x", "category"],
    )
    assert result.status == "sat"
    assert result.n_checked == 1
    assert any("partial" in warning.lower() and "1" in warning for warning in result.warnings)
    assert "sat" == result.to_dict()["status"]


def test_partial_verification_warning_reaches_public_fit_without_changing_rules():
    frame = pd.DataFrame({"x": [-2, -1, 1, 2] * 12, "target": [0, 0, 1, 1] * 12})
    session = (
        Session.ingest(frame)
        .set_roles({"x": "feature", "target": "target"})
        .split(test_size=0.25, validation_size=0.25, random_state=7)
    )
    fit = session.symbolic.fit(
        source="declared",
        backend="sklearn",
        default_consequent=0,
        rules=[
            {
                "if": [
                    {"column": "x", "op": ">=", "value": 0},
                    {"column": "x", "op": "in", "value": [1, 2]},
                ],
                "then": 1,
                "kind": "constraint",
            }
        ],
        verify_constraints=True,
    )
    assert any("partial" in warning.lower() for warning in fit.warnings)
    predicted = session.symbolic.predict(partition="test")
    assert predicted.predictions == tuple(
        int(value in (1, 2)) for value in session.partition("test").x
    )
