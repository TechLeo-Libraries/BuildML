"""Tier B product: Zenith Support OS.

Composes RAG retrieval over a support KB + NLP ticket routing + an
active-learning budget loop with a simulated oracle. Leakage discipline
throughout.
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
from buildml.data.dataset import Dataset
from buildml.ingest.detect import schema_from_dataframe
from buildml.rag.generate import EchoGroundedProvider
from proofs._lib import (
    assert_no_test_in_selection,
    extra_available,
    load_support_kb_corpus,
    load_support_tickets_synthetic,
    metrics_round,
    new_proof_context,
    write_results,
)


def _mask_train_labels(session: Session, fraction: float, seed: int, target: str):
    rng = np.random.default_rng(seed)
    full = session.to_pandas().copy()
    truth = full[target].to_numpy().copy()
    train_idx = list(session.split_plan.train_indices)
    n_blank = max(1, int(fraction * len(train_idx)))
    blank = rng.choice(train_idx, size=n_blank, replace=False)
    full.loc[blank, target] = np.nan
    session._dataset = Dataset.from_transformed(
        session.dataset,
        full,
        schema=schema_from_dataframe(full),
        roles=dict(session.dataset.roles),
    )
    return truth


def _al_pool(n_per: int = 180, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    x0 = rng.normal([-1.1, -1.0, 0.0, 0.2], 0.55, size=(n_per, 4))
    x1 = rng.normal([1.1, 1.0, 0.1, -0.2], 0.55, size=(n_per, 4))
    frame = pd.DataFrame(np.vstack([x0, x1]), columns=[f"f{i}" for i in range(4)])
    frame["label"] = [0] * n_per + [1] * n_per
    meta = {
        "name": "zenith_synthetic_label_pool",
        "license": "synthetic/public-domain",
        "n_rows": int(len(frame)),
    }
    return frame, meta


def main() -> None:
    ctx = new_proof_context("zenith-support-os", seed=13)
    stages: dict = {}
    skip_notes: list[str] = []

    # --- Stage 1: RAG ---
    docs, judgments = load_support_kb_corpus()
    st_ok = extra_available("sentence_transformers")
    try:
        rag = Session()
        rag.rag_ingest_corpus(docs)
        rag.rag_chunk(size=180, overlap=40)
        embed_backend = "hashing"
        try:
            if st_ok:
                rag.rag_embed_and_index(embedder="auto")
                embed_backend = "sentence_transformers_or_auto"
            else:
                rag.rag_embed_and_index(embedder="hashing")
        except (MissingExtraError, TypeError, ValueError):
            rag.rag_embed_and_index(embedder="hashing")
            embed_backend = "hashing"
        sample = rag.rag_retrieve("forgot password reset link expired", k=3, mode="hybrid")
        answer = rag.rag_generate(
            "How do I reset a forgotten password?",
            provider=EchoGroundedProvider(),
            k=3,
        )
        metrics = rag.rag_evaluate(judgments, k=3)
        stages["rag"] = {
            "status": "ok",
            "embed_backend": embed_backend,
            "sentence_transformers_available": st_ok,
            "sample_top_docs": [h.doc_id for h in sample.hits],
            "generate_citations": [c.doc_id for c in answer.citations],
            "retrieval_metrics": {
                "recall_at_k": float(metrics.recall_at_k),
                "mrr": float(metrics.mrr),
                "ndcg_at_k": float(metrics.ndcg_at_k),
            },
        }
        write_results(ctx, stages["rag"], filename="rag.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["rag"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"rag: {exc}")

    # --- Stage 2: NLP ticket routing ---
    try:
        tickets, ticket_meta = load_support_tickets_synthetic(n=720, seed=ctx.seed)
        nlp = (
            Session.ingest(tickets)
            .set_roles(
                {
                    "ticket_id": "id",
                    "body": "feature",
                    "channel": "feature",
                    "queue": "target",
                }
            )
            .split(
                test_size=0.2,
                validation_size=0.2,
                stratify=True,
                random_state=ctx.seed,
            )
        )
        profile = nlp.profile_text_corpus(
            text_column="body",
            near_duplicate_threshold=0.9,
            detect_languages=True,
        )
        fit = nlp.fit_text_classifier(
            text_column="body",
            vectorizer="tfidf",
            estimator="logistic",
            ngram_range=(1, 2),
            min_df=2,
            class_weight="balanced",
            random_state=ctx.seed,
        )
        validation = nlp.evaluate_text_classifier(partition="validation")
        assert_no_test_in_selection(selection_partition="validation")
        test = nlp.evaluate_text_classifier(partition="test")
        stages["nlp"] = {
            "status": "ok",
            "data": ticket_meta,
            "profile": metrics_round(
                profile.to_dict() if hasattr(profile, "to_dict") else {}
            ),
            "fit": metrics_round(fit.to_dict() if hasattr(fit, "to_dict") else {}),
            "validation_metrics": metrics_round(dict(validation.metrics)),
            "test_metrics": metrics_round(dict(test.metrics)),
        }
        write_results(ctx, stages["nlp"], filename="nlp.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["nlp"] = {"status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
        skip_notes.append(f"nlp: {exc}")

    # --- Stage 3: active learning budget ---
    try:
        pool, pool_meta = _al_pool(seed=ctx.seed)
        al = (
            Session.ingest(pool.copy())
            .set_roles({**{f"f{i}": "feature" for i in range(4)}, "label": "target"})
            .split(
                test_size=0.25,
                validation_size=0.15,
                stratify=True,
                random_state=ctx.seed,
            )
            .scale(method="standard")
        )
        truth = _mask_train_labels(al, fraction=0.85, seed=ctx.seed, target="label")
        fit_al = al.fit_active_learner(strategy="margin", batch_size=8, label_budget=32)
        curve = []
        for round_i in range(4):
            q = al.suggest_query(batch_size=8)
            if not q.indices:
                break
            human = [int(truth[i]) for i in q.indices]
            labeled = al.label_rows(indices=q.indices, labels=human)
            curve.append(
                {
                    "round": round_i,
                    "n_newly_labeled": int(labeled.n_newly_labeled),
                    "n_labeled_now": int(labeled.n_labeled_now),
                    "budget_remaining": int(labeled.budget_remaining),
                }
            )
        al_test = al.evaluate_active_learning(partition="test")
        stages["active_learning"] = {
            "status": "ok",
            "data": pool_meta,
            "fit": {
                "strategy": fit_al.strategy,
                "n_labeled_train": int(fit_al.n_labeled_train),
                "n_unlabeled_pool": int(fit_al.n_unlabeled_pool),
            },
            "label_curve": curve,
            "test_metrics": metrics_round(dict(al_test.metrics)),
        }
        write_results(ctx, stages["active_learning"], filename="active_learning.json")
    except (MissingExtraError, TypeError, ValueError) as exc:
        stages["active_learning"] = {
            "status": "skipped",
            "error": f"{type(exc).__name__}: {exc}",
        }
        skip_notes.append(f"active_learning: {exc}")

    ok_stages = sum(1 for v in stages.values() if v.get("status") == "ok")
    summary = {
        "status": "completed" if ok_stages >= 3 else "partial",
        "product": "Zenith Support OS",
        "stages": {k: {"status": v.get("status")} for k, v in stages.items()},
        "stage_details": stages,
        "skip_notes": skip_notes,
        "leakage_controls": [
            "RAG corpus = KB articles only; judgments never indexed as answers",
            "NLP stratified split before TF-IDF fit; validation for selection",
            "Active-learning queries drawn from train unlabeled pool only",
            "Test evaluate_text_classifier / evaluate_active_learning after locks",
        ],
        "what_fails_if_leakage_ignored": [
            "Indexing judgment answers into RAG inflates recall@k",
            "Fitting the text vectorizer on test tickets invents queue accuracy",
            "Querying the test pool for labels makes active-learning curves meaningless",
        ],
        "limitations": [
            "Synthetic KB + tickets — not a live helpdesk",
            "EchoGroundedProvider is offline scaffolding, not a production LLM",
            "Simulated oracle for active learning — not a real annotation UI",
        ],
    }
    write_results(ctx, summary, filename="summary.json")
    write_results(ctx, summary, filename="results.json")
    print(
        "zenith-support-os OK",
        {
            "rag": (stages.get("rag") or {}).get("status"),
            "nlp": (stages.get("nlp") or {}).get("status"),
            "active_learning": (stages.get("active_learning") or {}).get("status"),
            "skips": skip_notes,
        },
    )


if __name__ == "__main__":
    main()
