# ruff: noqa: E501
"""Case-based reasoning concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

CBR_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="cbr-case-memory",
            title="Tabular case memory (train-only)",
            summary="CBR stores train rows as cases (features + solution/label) and never builds memory from Session test.",
            definition=(
                "A CaseBase is a collection of cases. Each case has numeric "
                "(and optional categorical) features plus a solution "
                "(class label or regression outcome). fit_cbr builds the "
                "memory from Session train only."
            ),
            intuition=(
                "Remember past solved examples; for a new query, find similar "
                "past cases and reuse their answers."
            ),
            formal_idea=(
                "Case memory M = {(x_i, y_i)} from train. Query q → "
                "neighbors N_k(q) ⊂ M under distance d."
            ),
            why_it_matters=(
                "Train-only memory preserves holdout honesty.",
                "Traces show which cases influenced each prediction.",
            ),
            how_buildml_uses=(
                "Session.fit_cbr → retrieve_cases / predict_cbr / evaluate_cbr.",
            ),
            interpretation_rules=(
                "Inspect CaseTrace.neighbor_case_ids, distances, weights.",
                "train_score is in-sample (often includes self as nearest).",
            ),
            assumptions=("Non-null features/targets on train; split present.",),
            failure_modes=(
                "Empty train; null features; k larger than memory (capped).",
            ),
            anti_patterns=(
                "Building the case base from the full dataset before split.",
                "Calling CBR 'tabular RAG'.",
            ),
            worked_example_pattern=(
                "fit_cbr(metric='euclidean', k=5) → "
                "predict_cbr(return_traces=True).",
            ),
            related_concepts=(
                "cbr-retrieve-reuse",
                "cbr-vs-rag",
                "cbr-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="cbr-retrieve-reuse",
            title="Retrieve, reuse, adapt, retain",
            summary="Classic CBR cycle lite: kNN retrieve → vote/average/local model → optional offset adapt → optional leakage-safe retain.",
            definition=(
                "Retrieve: k nearest under euclidean / manhattan / cosine / "
                "mixed (Gower-style). Reuse: majority or distance-weighted "
                "vote (classification); distance-weighted mean, local mean, "
                "or local Ridge (regression). Adapt='offset' is a lite blend. "
                "retain_cbr appends labeled cases with disclosure and refuses "
                "validation/test indices."
            ),
            intuition=(
                "Find neighbors, combine their solutions, optionally keep new "
                "labeled examples: without contaminating holdout."
            ),
            formal_idea=(
                "ŷ(q) = reuse({y_i : i ∈ N_k(q)}, weights=1/(d+ε)). "
                "Retain adds (x', y') only if index ∉ holdout."
            ),
            why_it_matters=(
                "Reuse mode and metric are disclosed on the plan.",
                "Retain without disclosure or holdout checks is leakage.",
            ),
            how_buildml_uses=(
                "predict_cbr applies reuse; retain_cbr updates memory.",
            ),
            interpretation_rules=(
                "Read plan.metric / plan.reuse / CaseTrace notes.",
                "Retain requires source_disclosure.",
            ),
            assumptions=("Distance is meaningful after scale/encode.",),
            failure_modes=(
                "Unscaled features dominating L2; unknown categoricals → -1.",
            ),
            anti_patterns=(
                "Retaining Session test rows into the case base.",
                "Treating in-sample train_score as holdout accuracy.",
            ),
            worked_example_pattern=(
                "fit_cbr(reuse='distance_weighted') → evaluate_cbr → "
                "retain_cbr(labeled_frame=..., source_disclosure=...).",
            ),
            related_concepts=(
                "cbr-case-memory",
                "cbr-vs-rag",
                "cbr-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="cbr-vs-rag",
            title="CBR ≠ RAG",
            summary="RAG retrieves text chunks for generation; CBR retrieves tabular cases to reuse solutions for supervised-style tasks.",
            definition=(
                "RAG (buildml.rag) indexes a document corpus, retrieves "
                "chunks (dense/BM25/hybrid), and optionally generates an "
                "answer with citations. CBR indexes train rows as cases and "
                "reuses labels/outcomes: no LLM generation path."
            ),
            intuition=(
                "Both say 'find similar things', but RAG is for documents/"
                "answers and CBR is for labeled tabular episodes."
            ),
            formal_idea=(
                "RAG: query → documents → (optional) LLM. "
                "CBR: query features → cases → ŷ via reuse."
            ),
            why_it_matters=(
                "Different Session APIs, metrics, bundles, and leakage models.",
                "Do not call CBR 'tabular RAG'.",
            ),
            how_buildml_uses=(
                "Separate packages: buildml.cbr vs buildml.rag; separate "
                "Session state (_cbr_plan vs _rag_*).",
            ),
            interpretation_rules=(
                "Use RAG for text grounding; use CBR for case→solution.",
            ),
            assumptions=("Product surfaces stay distinct.",),
            failure_modes=(
                "Sharing RAG index state with CBR case memory.",
            ),
            anti_patterns=(
                "Routing CBR through rag_retrieve / rag_generate.",
                "Requiring buildml[rag] extras for CBR.",
            ),
            worked_example_pattern=(
                "fit_cbr(...) for tabular kNN reasoning; "
                "rag_ingest_corpus(...) for document RAG.",
            ),
            related_concepts=(
                "cbr-case-memory",
                "cbr-retrieve-reuse",
                "rag-retrieve",
                "leakage-boundary",
            ),
        ),
        _note(
            key="cbr-bundle-boundary",
            title="CBR bundle vs Session checkpoint",
            summary="buildml.cbr_bundle.v1 stores CbrPlan; Session checkpoints do not embed the case memory.",
            definition=(
                "save_cbr_bundle writes meta.json + cbr_plan.joblib. "
                "checkpoint_save stores workflow state without the case base."
            ),
            intuition=(
                "Reload the learner with load_cbr_bundle; reload the workflow "
                "with checkpoint_load."
            ),
            formal_idea=(
                "Artifact separation: learner bundle ⊥ Session checkpoint."
            ),
            why_it_matters=(
                "Prevents silent loss of case memory across restarts.",
            ),
            how_buildml_uses=(
                "Session.save_cbr_bundle / load_cbr_bundle.",
            ),
            interpretation_rules=(
                "Confirm meta.json format == buildml.cbr_bundle.v1.",
            ),
            assumptions=("Writable path; matching feature columns on reload.",),
            failure_modes=("Expecting checkpoint_load to restore CbrPlan.",),
            anti_patterns=(
                "Treating RAG bundles and CBR bundles as interchangeable.",
            ),
            worked_example_pattern=(
                "save_cbr_bundle(path) → new Session → load_cbr_bundle(path).",
            ),
            related_concepts=(
                "cbr-case-memory",
                "checkpoint-boundary",
                "leakage-boundary",
            ),
        ),
    )
}
