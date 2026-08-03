"""Optional Z3 constraint verification for hard symbolic rule sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from buildml.symbolic.extras import z3_available
from buildml.symbolic.rules import RuleKnowledgeBase


@dataclass(slots=True)
class ConstraintVerificationResult:
    """Outcome of a lightweight Z3 satisfiability check on rule antecedents."""

    status: str
    satisfiable: bool | None
    n_hard_constraints: int
    n_checked: int
    disclosures: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise Z3 constraint verification outcome for fit disclosures.

        Attached to fit results when ``verify_constraints`` is enabled so
        walkthrough panels can show satisfiability without rerunning Z3.

        Returns
        -------
        dict[str, Any]
            Status, satisfiability flag, counts, disclosures, and warnings.
        """
        return {
            "status": self.status,
            "satisfiable": self.satisfiable,
            "n_hard_constraints": self.n_hard_constraints,
            "n_checked": self.n_checked,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


def verify_rule_constraints(
    knowledge_base: RuleKnowledgeBase,
    columns: list[str],
) -> ConstraintVerificationResult:
    """Check whether hard constraint rule antecedents are jointly satisfiable.

    Encodes numeric predicates from hard rules into a lightweight Z3 solver.
    Skips when Z3 is missing, no hard constraints exist, or predicates are
    non-numeric.

    Parameters
    ----------
    knowledge_base:
        Compiled rules including constraint or hard rules to verify.
    columns:
        Train frame columns available as real-valued Z3 variables.

    Returns
    -------
    ConstraintVerificationResult
        SAT/unsat outcome, counts, disclosures, and optional warnings.

    Notes
    -----
    Honesty: lightweight SAT check on numeric column bounds: not a full SMT
    product or complete rule-set consistency prover.
    """
    hard_rules = [
        r
        for r in knowledge_base.rules
        if r.kind == "constraint" or r.hardness == "hard"
    ]
    disclosures = [
        "Z3 constraint verification is optional lite SAT on numeric predicates: "
        "not a complete symbolic AI verifier.",
    ]
    if not hard_rules:
        return ConstraintVerificationResult(
            status="skipped_no_hard_constraints",
            satisfiable=None,
            n_hard_constraints=0,
            n_checked=0,
            disclosures=tuple(disclosures + ["No hard constraints to verify."]),
        )
    if not z3_available():
        return ConstraintVerificationResult(
            status="skipped_z3_missing",
            satisfiable=None,
            n_hard_constraints=len(hard_rules),
            n_checked=0,
            disclosures=tuple(
                disclosures
                + [
                    "Install buildml[symbolic-industry] with z3-solver for "
                    "constraint verification.",
                ]
            ),
        )

    import z3

    solver = z3.Solver()
    reals: dict[str, Any] = {}
    for col in columns:
        reals[col] = z3.Real(col)

    n_checked = 0
    for rule in hard_rules:
        if not rule.antecedents:
            continue
        expr = None
        for pred in rule.antecedents:
            if pred.column not in reals:
                continue
            var = reals[pred.column]
            rhs = z3.RealVal(float(pred.value)) if pred.value is not None else None
            if rhs is None:
                continue
            if pred.op == "<":
                clause = var < rhs
            elif pred.op == "<=":
                clause = var <= rhs
            elif pred.op == ">":
                clause = var > rhs
            elif pred.op == ">=":
                clause = var >= rhs
            elif pred.op == "==":
                clause = var == rhs
            elif pred.op == "!=":
                clause = var != rhs
            else:
                continue
            expr = clause if expr is None else z3.And(expr, clause)
            n_checked += 1
        if expr is not None:
            solver.add(expr)

    if n_checked == 0:
        return ConstraintVerificationResult(
            status="skipped_non_numeric",
            satisfiable=None,
            n_hard_constraints=len(hard_rules),
            n_checked=0,
            disclosures=tuple(
                disclosures
                + ["Hard constraints use non-numeric ops Z3 lite path cannot encode."]
            ),
        )

    result = solver.check()
    sat = result == z3.sat
    status = "sat" if sat else "unsat"
    warnings: list[str] = []
    if not sat:
        warnings.append(
            "Hard constraint antecedents are jointly unsatisfiable: review rule set."
        )
    return ConstraintVerificationResult(
        status=status,
        satisfiable=sat,
        n_hard_constraints=len(hard_rules),
        n_checked=n_checked,
        disclosures=tuple(
            disclosures
            + [f"Z3 checked {n_checked} numeric predicate(s) across hard rules."]
        ),
        warnings=tuple(warnings),
    )
