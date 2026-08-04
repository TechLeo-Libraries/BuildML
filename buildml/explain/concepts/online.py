# ruff: noqa: E501
"""Online / continual learning concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

ONLINE_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="online-partial-fit",
            title="Online / continual partial_fit updates",
            summary="Incremental sklearn partial_fit on train chunks: not a distributed streaming platform.",
            definition=(
                "Online / continual learning in BuildML warm-starts an estimator "
                "that supports partial_fit, then updates it on subsequent train "
                "chunks (or role-aligned user frames). Holdout partitions are "
                "evaluation-only."
            ),
            intuition=(
                "Teach the model a page at a time from the practice binder: "
                "never rewrite the whole notebook silently, and never peek at "
                "the exam for 'updates'."
            ),
            formal_idea=(
                "θ_{t+1} = partial_fit(θ_t, X_chunk, y_chunk); chunks ⊂ train; "
                "holdout H disjoint from update stream."
            ),
            why_it_matters=(
                "Silent full refits pretend to be online and invalidate claims.",
                "Using validation/test for updates is leakage.",
            ),
            how_buildml_uses=(
                "session.online.fit → session.online.partial_fit → session.online.evaluate.",
                "allow_refit_fallback defaults False; when True, refits are disclosed.",
            ),
            interpretation_rules=(
                "Read n_seen_rows, n_updates, update_mode, and used_refit_fallback.",
                "update_mode='refit_fallback' means this was not incremental.",
            ),
            assumptions=("Estimator supports partial_fit, or fallback is explicitly allowed.",),
            failure_modes=(
                "Choosing HistGradientBoosting and expecting silent online updates.",
                "Feeding holdout rows as 'the next chunk'.",
            ),
            anti_patterns=("Calling .fit on all data each round while claiming online learning.",),
            worked_example_pattern=(
                "session.online.fit(chunk_size=40) → session.online.partial_fit() × k → session.online.evaluate('validation').",
            ),
            related_concepts=("online-class-discovery", "online-bundle-boundary"),
        ),
        _note(
            key="online-class-discovery",
            title="Classifier classes= on first partial_fit",
            summary="Classifiers need the full label vocabulary on init: explicit classes= or train-target discovery (labels only).",
            definition=(
                "Sklearn classifiers require classes= on the first partial_fit. "
                "BuildML accepts an explicit vocabulary or discovers it from the "
                "full train target column before streaming chunks (features from "
                "unseen rows are not used until their chunk is applied)."
            ),
            intuition=(
                "Tell the grader which answer choices exist before grading the "
                "first page: you do not need to read future essays to know the "
                "letter options."
            ),
            formal_idea=(
                "C = classes(y_train) or user-provided; first partial_fit(X0, y0, classes=C)."
            ),
            why_it_matters=(
                "Missing classes cause hard failures when a later chunk introduces a label.",
            ),
            how_buildml_uses=(
                "session.online.fit(classes=...) or automatic train-target discovery with disclosure.",
            ),
            interpretation_rules=(
                "Read plan.classes_ and the class-discovery disclosure on session.online.fit.",
            ),
            assumptions=("Train targets span (or classes= lists) every label the stream may emit.",),
            failure_modes=("Omitting a rare class that appears only in a late chunk.",),
            anti_patterns=("Expanding classes silently mid-stream without a declared contract.",),
            worked_example_pattern=(
                "session.online.fit(estimator='sgd_classifier', classes=[0, 1])",
            ),
            related_concepts=("online-partial-fit", "leakage-boundary"),
        ),
        _note(
            key="online-drift-disclose",
            title="Optional chunk drift disclosure",
            summary="Lightweight mean-shift notes vs the init chunk; not a full drift product.",
            definition=(
                "When drift_disclose=True, each update compares chunk feature "
                "means to the init-chunk means and flags large relative shifts. "
                "Richer train/test drift screens remain on Session.eda()."
            ),
            intuition=(
                "A sticky note that the new page looks unusually bright: not a "
                "full laboratory assay of distribution shift."
            ),
            formal_idea=(
                "flag if |μ_chunk − μ_init| / max(|μ_init|, ε) ≥ 0.5 for a feature."
            ),
            why_it_matters=(
                "Catching gross feature shifts mid-stream prevents silent quality cliffs.",
            ),
            how_buildml_uses=(
                "session.online.partial_fit returns drift_notes; walkthrough surfaces them.",
            ),
            interpretation_rules=(
                "Treat drift_notes as disclosure, not automatic remediation.",
            ),
            assumptions=("Init means were recorded at session.online.fit.",),
            failure_modes=("Treating the lite screen as a production drift monitor.",),
            anti_patterns=("Building a streaming platform claim from mean-shift notes.",),
            worked_example_pattern=(
                "u = session.online.partial_fit(); print(u.drift_notes)",
            ),
            related_concepts=("online-partial-fit", "online-bundle-boundary"),
        ),
        _note(
            key="online-river-streaming",
            title="River streaming backend (buildml[online-industry])",
            summary="Incremental River estimators with ADWIN/Page-Hinkley drift on prediction error: not a Kafka product.",
            definition=(
                "The industry online backend wraps River classifiers/regressors with "
                "partial_fit-style updates on train chunks and optional ADWIN or "
                "Page-Hinkley drift detectors on holdout prediction error."
            ),
            intuition=(
                "A lightweight stream processor inside your notebook: each chunk updates "
                "the model once, and drift notes fire when recent errors diverge."
            ),
            formal_idea=(
                "θ ← River.partial_fit(X_chunk, y_chunk); drift ← ADWIN(error_stream)."
            ),
            why_it_matters=(
                "Sklearn partial_fit alone misses streaming-specific drift tooling.",
                "Catalog defaults to River when installed: read backend on the plan.",
            ),
            how_buildml_uses=(
                "session.online.fit(backend='industry', estimator='river_logistic'|...).",
                "Drift disclosures attached to session.online.partial_fit / session.online.evaluate.",
            ),
            interpretation_rules=(
                "Read backend, estimator, drift_notes, and update_mode on results.",
            ),
            assumptions=("River extra installed; classification vs regression estimator matches task.",),
            failure_modes=("Expecting River to run without buildml[online-industry].",),
            anti_patterns=("Claiming a full streaming platform from River adapters.",),
            worked_example_pattern=(
                "session.online.fit(backend='industry') → session.online.partial_fit() → session.online.evaluate('validation').",
            ),
            related_concepts=("online-drift-disclose", "online-partial-fit"),
        ),
        _note(
            key="online-torch-continual",
            title="Torch replay / EWC continual learning (buildml[torch])",
            summary="Tabular MLP with replay buffer or EWC penalty: classification-only incremental path.",
            definition=(
                "Torch online backends train a small tabular MLP with either experience "
                "replay (replay_mlp) or elastic weight consolidation (ewc_mlp) across "
                "train chunks. Regression is not supported on this path."
            ),
            intuition=(
                "Replay keeps a scrapbook of past examples so new chunks do not erase "
                "old patterns; EWC penalizes moving weights that mattered before."
            ),
            formal_idea=(
                "Replay: L = CE(f_θ(X_new), y_new) + CE(f_θ(X_mem), y_mem). "
                "EWC: L += λ Σ F_i (θ_i - θ*_i)²."
            ),
            why_it_matters=(
                "Neural continual learning differs from sklearn partial_fit semantics.",
            ),
            how_buildml_uses=(
                "session.online.fit(backend='torch', estimator='replay_mlp'|'ewc_mlp').",
            ),
            interpretation_rules=("Read n_updates, n_seen_rows, and backend on OnlinePlan.",),
            assumptions=("Torch installed; classification task.",),
            failure_modes=("Calling replay_mlp for regression: refused at resolve time.",),
            anti_patterns=("Treating replay MLP as full lifelong learning at scale.",),
            worked_example_pattern=(
                "session.online.fit(backend='torch', estimator='ewc_mlp') → session.online.partial_fit().",
            ),
            related_concepts=("online-partial-fit", "online-bundle-boundary"),
        ),
        _note(
            key="online-bundle-boundary",
            title="Online-learning bundle boundary",
            summary="buildml.online_bundle.v1 stores OnlinePlan (estimator + cursor + update history); Session checkpoints do not embed it.",
            definition=(
                "An online bundle persists the incremental estimator, class "
                "vocabulary, train cursor, seen indices, and update ledger. "
                "Session checkpoints persist data/roles/splits/history: not "
                "OnlinePlan weights."
            ),
            intuition=(
                "Saving the binder resume is not the same as saving the running "
                "incremental model and its update diary."
            ),
            formal_idea=(
                "Artifacts are complementary: checkpoint_load ↛ online learner; "
                "session.online.load_bundle ↛ dataset rows."
            ),
            why_it_matters=("Mixing artifacts causes silent missing-learner failures.",),
            how_buildml_uses=("session.online.save_bundle / session.online.load_bundle.",),
            interpretation_rules=("Read meta.json format buildml.online_bundle.v1.",),
            assumptions=("Feature contract and target role still match at load time.",),
            failure_modes=("Expecting checkpoint_load to restore OnlinePlan.",),
            anti_patterns=("Treating active-learning bundles as online plans.",),
            worked_example_pattern=(
                "session.online.save_bundle(path); other.online.load_bundle(path).",
            ),
            related_concepts=("online-partial-fit", "activelearning-bundle-boundary"),
        ),
    )
}
