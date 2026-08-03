"""NLP representation trade-off benchmark.

Measures what each document representation actually costs and buys on one fixed
corpus and one fixed split: holdout accuracy, fit and score latency, vocabulary
size, and whether token attribution remains possible. Bag-of-n-grams variants
always run. The dense backends run only when their extra is installed, and are
recorded as skipped-with-a-reason otherwise, because a benchmark that silently
omits a configuration is worse than one that says why.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from buildml import Session
from buildml.core.errors import BuildMLError

_POOLS: dict[str, tuple[str, ...]] = {
    "billing": (
        "Invoice INV-{ref} charged the annual fee twice on the same card.",
        "The renewal quote said one figure but the invoice came to almost double.",
        "A proration credit never appeared against invoice INV-{ref}.",
        "Finance flagged the duplicate charge during monthly reconciliation.",
    ),
    "shipping": (
        "Order ORD-{ref} was promised for the 3rd and arrived nine days late.",
        "Two of the four cartons in shipment ORD-{ref} were crushed in transit.",
        "Tracking for ORD-{ref} has not updated since it left the depot.",
        "A customs hold added four days that nobody notified us about.",
    ),
    "account": (
        "Single sign-on stopped working for the whole workspace this morning.",
        "The onboarding portal rejects the invite link for every new hire.",
        "Password resets arrive but the link has already expired on arrival.",
        "Role permissions reverted to read-only after the last release.",
    ),
    "hardware": (
        "The hinge on unit HW-{ref} snapped within a week of light use.",
        "Unit HW-{ref} overheats and shuts down under a normal workload.",
        "The display on unit HW-{ref} flickers whenever the lid is moved.",
        "Battery life on unit HW-{ref} dropped to under an hour after a month.",
    ),
}

_GENERIC: tuple[str, ...] = (
    "Following up on the case our team raised earlier this week.",
    "This is the third time we are writing about the same problem.",
    "The last update we received simply asked us to wait.",
    "Please confirm who is handling this and by when.",
)

# Configurations share one corpus and one split, so differences are attributable
# to the representation rather than to the data.
_CONFIGS: tuple[dict[str, object], ...] = (
    {
        "name": "tfidf_word_1_1",
        "kwargs": {"vectorizer": "tfidf", "analyzer": "word", "ngram_range": (1, 1)},
    },
    {
        "name": "tfidf_word_1_2",
        "kwargs": {"vectorizer": "tfidf", "analyzer": "word", "ngram_range": (1, 2)},
    },
    {
        "name": "count_word_1_2",
        "kwargs": {"vectorizer": "count", "analyzer": "word", "ngram_range": (1, 2)},
    },
    {
        "name": "tfidf_char_wb_3_5",
        "kwargs": {
            "vectorizer": "tfidf",
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
        },
    },
    {
        "name": "hashing_word_1_2",
        "kwargs": {
            "vectorizer": "hashing",
            "analyzer": "word",
            "ngram_range": (1, 2),
            "n_hash_features": 2**16,
        },
    },
    {
        "name": "complement_nb_tfidf",
        "kwargs": {
            "vectorizer": "tfidf",
            "analyzer": "word",
            "ngram_range": (1, 2),
            "estimator": "complement_nb",
        },
    },
    {
        "name": "embedding_minilm",
        "kwargs": {"backend": "embedding"},
        "optional": True,
    },
    {
        "name": "transformer_pooled",
        "kwargs": {"backend": "transformer"},
        "optional": True,
    },
)


def _corpus(n: int = 600, seed: int = 0, ambiguous_rate: float = 0.18) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    queues = list(_POOLS)
    rows: list[dict[str, object]] = []
    for index in range(n):
        queue = queues[index % len(queues)]
        ref = int(rng.integers(10_000, 99_999))
        if rng.random() < ambiguous_rate:
            parts = [str(rng.choice(_GENERIC)) for _ in range(3)]
        else:
            pool = _POOLS[queue]
            parts = [
                str(rng.choice(pool)).format(ref=ref),
                str(rng.choice(pool)).format(ref=ref),
                str(rng.choice(_GENERIC)),
            ]
        rows.append({"body": " ".join(parts), "queue": queue})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _session(frame: pd.DataFrame) -> Session:
    return (
        Session.ingest(frame.copy())
        .set_roles({"body": "feature", "queue": "target"})
        .split(test_size=0.25, validation_size=0.2, random_state=0, stratify=True)
    )


def _run_config(frame: pd.DataFrame, config: dict[str, object]) -> dict[str, object]:
    name = str(config["name"])
    kwargs = dict(config["kwargs"])  # type: ignore[arg-type]
    session = _session(frame)
    try:
        t0 = time.perf_counter()
        fit = session.fit_text_classifier(min_df=2, random_state=0, **kwargs)
        fit_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        ev = session.evaluate_text_classifier(partition="test")
        score_seconds = time.perf_counter() - t1
    except BuildMLError as exc:
        return {
            "config": name,
            "skipped": True,
            "reason": f"{type(exc).__name__}: {exc}",
            "optional": bool(config.get("optional")),
        }

    attribution_works = True
    attribution_reason = "exact linear coefficient x feature value"
    try:
        session.interpret_text_prediction(partition="test", max_documents=1)
    except BuildMLError as exc:
        attribution_works = False
        attribution_reason = str(exc).split(".")[0]

    return {
        "config": name,
        "backend": fit.backend,
        "estimator": fit.estimator,
        "n_features": fit.n_features,
        "vocabulary_size": fit.vocabulary_size,
        "train_score": round(float(fit.train_score or 0.0), 6),
        "test_accuracy": round(float(ev.metrics.get("accuracy", 0.0)), 6),
        "test_balanced_accuracy": round(
            float(ev.metrics.get("balanced_accuracy", 0.0)), 6
        ),
        "test_f1_macro": round(float(ev.metrics.get("f1_macro", 0.0)), 6),
        "test_log_loss": (
            None
            if "log_loss" not in ev.metrics
            else round(float(ev.metrics["log_loss"]), 6)
        ),
        "holdout_oov_token_rate": (
            None if ev.oov_rate is None else round(float(ev.oov_rate), 6)
        ),
        "fit_seconds": round(fit_seconds, 4),
        "score_seconds": round(score_seconds, 4),
        "token_attribution_available": attribution_works,
        "token_attribution_note": attribution_reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="NLP representation trade-off benchmark"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/nlp/results/representation_tradeoff.json"),
    )
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Attempt the embedding / transformer backends (downloads model weights).",
    )
    args = parser.parse_args(argv)

    from buildml.nlp.catalog import nlp_capability_matrix

    matrix = nlp_capability_matrix()
    frame = _corpus(n=args.n)

    runs: list[dict[str, object]] = []
    for config in _CONFIGS:
        optional = bool(config.get("optional"))
        if optional and not args.include_optional:
            runs.append(
                {
                    "config": str(config["name"]),
                    "skipped": True,
                    "optional": True,
                    "reason": (
                        "Dense backends download model weights; pass "
                        "--include-optional to run them."
                    ),
                }
            )
            continue
        runs.append(_run_config(frame, config))

    results = {
        "corpus": {
            "n_rows": int(len(frame)),
            "classes": sorted(frame["queue"].unique().tolist()),
            "text_column": "body",
            "ambiguous_rate": 0.18,
            "note": (
                "One fixed corpus and one fixed stratified split for every "
                "configuration, so differences are attributable to the "
                "representation. An 18% share of tickets is queue-agnostic, "
                "which caps achievable accuracy near 0.87."
            ),
        },
        "availability": {
            "default_backend": matrix["default_backend_when_installed"],
            "sentence_transformers_present": matrix["sentence_transformers_present"],
            "transformers_present": matrix["transformers_present"],
            "nltk_present": matrix["nltk_present"],
            "langdetect_present": matrix["langdetect_present"],
            "spacy_model_present": matrix["spacy_model_present"],
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

    baseline = next(
        (run for run in runs if run.get("config") == "tfidf_word_1_2"), None
    )
    if baseline is None or baseline.get("skipped"):
        print("FAIL: the tfidf_word_1_2 baseline did not run.", file=sys.stderr)
        return 1
    accuracy = float(baseline["test_accuracy"])
    if accuracy < 0.75:
        print(
            f"FAIL: tfidf_word_1_2 test accuracy {accuracy} below floor 0.75",
            file=sys.stderr,
        )
        return 1
    hashing = next((run for run in runs if run.get("config") == "hashing_word_1_2"), None)
    if hashing and not hashing.get("skipped"):
        if hashing["token_attribution_available"] is not False:
            print(
                "FAIL: hashing must refuse token attribution (no invertible vocabulary).",
                file=sys.stderr,
            )
            return 1
    print("NLP representation_tradeoff benchmark passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
