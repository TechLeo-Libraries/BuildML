"""Tier B product: Folio Claims NLP.

Composes NLP text classification over claim notes + CBR case memory +
symbolic guardrails for synthetic P&C claim routing / escalation.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from proofs._lib.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import numpy as np
import pandas as pd

from buildml import Session
from buildml.core.errors import MissingExtraError
from proofs._lib import (
    assert_no_test_in_selection,
    load_support_tickets_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def _claim_cases(seed: int = 29) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n = 300
    severity = rng.integers(1, 6, size=n).astype(float)
    vehicle_age = rng.integers(0, 20, size=n).astype(float)
    prior_claims = rng.poisson(0.7, size=n).astype(float)
    urban = rng.binomial(1, 0.55, size=n).astype(float)
    deductible = rng.choice([250.0, 500.0, 1000.0], size=n)
    logit = (
        -2.0
        + 0.55 * severity
        + 0.06 * vehicle_age
        + 0.4 * prior_claims
        + 0.7 * urban
        - 0.0004 * deductible
        + rng.normal(0, 0.35, size=n)
    )
    escalate = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    frame = pd.DataFrame(
        {
            "severity": severity,
            "vehicle_age": vehicle_age,
            "prior_claims": prior_claims,
            "urban": urban,
            "deductible": deductible,
            "escalate": escalate,
            "claim_id": [f"cl-{i}" for i in range(n)],
        }
    )
    meta = {
        "name": "folio_synthetic_claim_cases",
        "license": "synthetic/public-domain",
        "n_rows": n,
        "positive_rate": float(escalate.mean()),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("folio-claims-nlp", seed=29)
    # Reuse ticket-style text as claim-note narrative (queue → claim desk)
    tickets, ticket_meta = load_support_tickets_synthetic(n=720, seed=ctx.seed)
    tickets = tickets.rename(columns={"ticket_id": "claim_note_id", "queue": "desk"})
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 1: NLP text classifier ---
    try:
        session = (
            Session.ingest(tickets.copy())
            .set_roles(
                {
                    "claim_note_id": "id",
                    "body": "feature",
                    "channel": "feature",
                    "desk": "target",
                }
            )
            .split(
                test_size=0.2,
                validation_size=0.2,
                stratify=True,
                random_state=ctx.seed,
            )
        )
        profile = session.nlp.profile_corpus(
            text_column="body",
            near_duplicate_threshold=0.9,
            detect_languages=True,
        )
        fit = session.nlp.fit_classifier(
            text_column="body",
            vectorizer="tfidf",
            estimator="logistic",
            ngram_range=(1, 2),
            min_df=2,
            class_weight="balanced",
            random_state=ctx.seed,
        )
        validation = session.nlp.evaluate(partition="validation")
        assert_no_test_in_selection(selection_partition="validation")
        test = session.nlp.evaluate(partition="test")
        topics = session.nlp.fit_topics(
            method="nmf",
            n_topics=4,
            text_column="body",
            min_df=3,
            max_df=0.9,
            stopword_language="en",
            random_state=ctx.seed,
        )
        stages["nlp"] = {
            "status": "ok",
            "data": ticket_meta,
            "profile": metrics_round(
                profile.to_dict() if hasattr(profile, "to_dict") else {}
            ),
            "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
            "validation_metrics": metrics_round(dict(validation.metrics)),
            "test_metrics": metrics_round(dict(test.metrics)),
            "topics": metrics_round(
                topics.to_dict() if hasattr(topics, "to_dict") else {}
            ),
        }
        write_results(ctx, stages["nlp"], filename="nlp.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["nlp"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"nlp: {exc}")

    # --- Stage 2: CBR case memory ---
    cases, case_meta = _claim_cases(seed=ctx.seed)
    try:
        session = (
            Session.ingest(cases.copy())
            .set_roles(
                {
                    "severity": "feature",
                    "vehicle_age": "feature",
                    "prior_claims": "feature",
                    "urban": "feature",
                    "deductible": "feature",
                    "escalate": "target",
                    "claim_id": "id",
                }
            )
            .split(
                test_size=0.2,
                validation_size=0.2,
                stratify=True,
                random_state=ctx.seed,
            )
            .scale(method="standard")
        )
        fit_c = session.cbr.fit(
            task="classification",
            metric="euclidean",
            reuse="distance_weighted",
            k=5,
            random_state=ctx.seed,
        )
        ev_c = session.cbr.evaluate(partition="test")
        plan_c = session.split_plan
        assert plan_c is not None
        stages["cbr"] = {
            "status": "ok",
            "data": case_meta,
            "fit": metrics_round(fit_c.to_dict() if hasattr(fit_c, "to_dict") else {}),
            "test_metrics": metrics_round(dict(getattr(ev_c, "metrics", {}) or {})),
            "split_counts": {
                "train": len(plan_c.train_indices),
                "validation": len(plan_c.validation_indices),
                "test": len(plan_c.test_indices),
            },
        }
        write_results(ctx, stages["cbr"], filename="cbr.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["cbr"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"cbr: {exc}")
        session = None
        plan_c = None

    # --- Stage 3: symbolic guardrails ---
    try:
        if session is None or plan_c is None:
            raise ValueError("CBR stage unavailable")
        sym = (
            Session.ingest(cases.copy())
            .set_roles(
                {
                    "severity": "feature",
                    "vehicle_age": "feature",
                    "prior_claims": "feature",
                    "urban": "feature",
                    "deductible": "feature",
                    "escalate": "target",
                    "claim_id": "id",
                }
            )
            .inject_split(
                train_indices=list(plan_c.train_indices),
                validation_indices=list(plan_c.validation_indices),
                test_indices=list(plan_c.test_indices),
            )
        )
        assert_no_test_in_selection(
            selection_partition="train", evaluation_partition="test"
        )
        try:
            fit_s = sym.symbolic.fit(
                source="decision_tree", max_depth=3, random_state=ctx.seed
            )
        except TypeError:
            fit_s = sym.symbolic.fit(method="decision_tree", random_state=ctx.seed)
        ev_s = sym.symbolic.evaluate(partition="test")
        stages["symbolic"] = {
            "status": "ok",
            "fit": metrics_round(fit_s.to_dict() if hasattr(fit_s, "to_dict") else {}),
            "test_metrics": metrics_round(dict(getattr(ev_s, "metrics", {}) or {})),
        }
        write_results(ctx, stages["symbolic"], filename="symbolic.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["symbolic"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"symbolic: {exc}")

    ok_stages = sum(1 for v in stages.values() if v.get("status") == "ok")
    summary = {
        "status": "completed" if ok_stages >= 3 else "partial",
        "product": "Folio Claims NLP",
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "NLP stratified split before TF-IDF / topics fit",
            "CBR case memory built from train cases only",
            "Symbolic rules induced on the same train split as CBR",
            "Test text / CBR / symbolic eval after each stage locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Fitting the text vectorizer on test notes invents desk accuracy",
            "Putting test claims into CBR memory makes escalate accuracy meaningless",
            "Inducing guardrail rules on the full book looks more 'compliant' than production",
        ],
        "limitations": [
            "Claim notes reuse synthetic ticket language — not a real P&C extract",
            "Product proof, not a claims-ops certification",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "folio-claims-nlp OK",
        {
            "nlp": (stages.get("nlp") or {}).get("status"),
            "cbr": (stages.get("cbr") or {}).get("status"),
            "symbolic": (stages.get("symbolic") or {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
