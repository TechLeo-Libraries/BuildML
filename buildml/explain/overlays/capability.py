# ruff: noqa: E501, F401
"""Capability-matrix overlays for every domain that publishes one.

These operations share one contract exactly: they are zero-argument static
introspection helpers that report what a domain can actually do on *this*
machine — which backends are installed, which methods each backend supports,
and which extra to install for anything missing. They are grouped here rather
than scattered across the domain overlays because the teaching content differs
only in the domain name and the honest limits of that domain.
"""

from __future__ import annotations

from buildml.explain.overlays._common import OperationKind, _operation
from buildml.explain.schemas import OperationSpec


def _matrix(
    name: str,
    domain: str,
    subject: str,
    *,
    honesty: str,
    next_steps: tuple[str, ...],
    concepts: tuple[str, ...],
    extra_mechanism: tuple[str, ...] = (),
) -> OperationSpec:
    """Build the shared teaching overlay for one domain's capability matrix."""
    return _operation(
        name,
        OperationKind.INSPECT,
        f"Report which {subject} are available for {domain} on this installation.",
        f"Find out what {domain} can actually do here before writing a fit call that will fail.",
        f"{domain} pre-flight introspection; safe to call at any point.",
        (
            "Probe optional dependencies without importing heavy stacks eagerly.",
            f"List {subject} and mark each as available or missing.",
            "Name the extras to install for anything unavailable.",
            *extra_mechanism,
        ),
        parameters=(),
        inputs=("Nothing — this is a static method and reads no dataset.",),
        outputs=("A plain dict describing backends, methods, availability, and install hints.",),
        prerequisites=(),
        ordering=(f"Before the {domain} fit call, especially when choosing a backend.",),
        alternatives=(
            "Reading the guides, which describe the same matrix but cannot know what is installed here.",
        ),
        rationale=(
            "Silent fallbacks hide which implementation actually ran, so BuildML publishes availability instead of guessing.",
        ),
        assumptions=("Optional extras are importable if installed in the active environment.",),
        failures=(
            "None by design: a missing optional dependency is reported as unavailable rather than raised.",
        ),
        leakage=("Read-only introspection; no dataset is touched.",),
        anti_patterns=(
            "Hard-coding a backend name in shared code without checking availability first.",
            "Treating the matrix as a benchmark: it reports availability, not quality.",
        ),
        state_changes=("None — no Session state and no history record.",),
        result_reading=(
            "An entry marked unavailable names the extra to install; the default backend is the best available one.",
            honesty,
        ),
        next_steps=next_steps,
        concepts=concepts,
    )


_OPERATIONS: tuple[OperationSpec, ...] = (
    _matrix(
        "unsupervised_capability_matrix",
        "clustering and dimensionality reduction",
        "clustering / reduction backends and methods",
        honesty="Availability says nothing about whether clusters are real; read evaluate_clusters for that.",
        next_steps=("fit_clusters; reduce_dimensions; evaluate_clusters.",),
        concepts=("cluster-validity-not-truth", "principal-components"),
    ),
    _matrix(
        "forecast_capability_matrix",
        "forecasting",
        "forecasting backends and model families",
        honesty="A model being available does not make it appropriate for your seasonality or horizon.",
        next_steps=("fit_forecast; evaluate_forecast.",),
        concepts=("forecast-univariate-vs-exog", "forecast-metric-limits"),
    ),
    _matrix(
        "timeseries_capability_matrix",
        "time-series analysis",
        "decomposition, stationarity, and changepoint backends",
        honesty="Diagnostics describe the series you gave them; they do not certify that a model will work.",
        next_steps=("analyze_timeseries; ts_decompose; ts_diagnostics.",),
        concepts=("forecast-temporal-leakage", "forecast-lag-features"),
    ),
    _matrix(
        "anomaly_capability_matrix",
        "anomaly detection",
        "detector backends, methods, and threshold policies",
        honesty="Detector availability is unrelated to whether your alert rate is affordable.",
        next_steps=("fit_anomaly; tune_anomaly_threshold; evaluate_anomaly.",),
        concepts=("anomaly-novelty-vs-unsupervised", "anomaly-threshold-alert-rate"),
    ),
    _matrix(
        "ssl_capability_matrix",
        "self-supervised learning",
        "pretext backends and encoder families",
        honesty="A pretext task being available does not mean its representation helps your downstream head.",
        next_steps=("fit_ssl_pretext; finetune_ssl_head; evaluate_ssl.",),
        concepts=("ssl-pretext-then-head", "ssl-vs-backbone-transfer"),
    ),
    _matrix(
        "rag_capability_matrix",
        "retrieval-augmented generation",
        "embedding, index, and retrieval stacks",
        honesty="Retrieval availability says nothing about answer quality; read the citation coverage in rag_generate.",
        next_steps=("rag_ingest_corpus; rag_embed_and_index; rag_retrieve.",),
        concepts=("rag-chunk-index-boundary", "rag-retrieval-metrics"),
    ),
    _matrix(
        "symbolic_capability_matrix",
        "symbolic and neuro-symbolic learning",
        "rule-induction backends and hybrid modes",
        honesty="Rule readability is not rule correctness; always read the holdout evaluation.",
        next_steps=("fit_symbolic; fit_neuro_symbolic; evaluate_symbolic.",),
        concepts=("symbolic-rules", "neuro-symbolic-hybrid"),
    ),
    _matrix(
        "cbr_capability_matrix",
        "case-based reasoning",
        "retrieval backends and distance metrics",
        honesty="An approximate-nearest-neighbour backend trades exactness for speed, which the matrix states rather than hides.",
        next_steps=("fit_cbr; retrieve_cases; predict_cbr.",),
        concepts=("cbr-case-memory", "cbr-vs-rag"),
    ),
    _matrix(
        "nlp_capability_matrix",
        "natural-language processing",
        "document representation backends and task surfaces",
        honesty="The core bag-of-n-grams path is always available; embedding, transformer, langdetect, NLTK, and spaCy paths are optional and named as such.",
        next_steps=(
            "profile_text_corpus; fit_text_classifier; fit_topics; analyze_sentiment.",
        ),
        concepts=("nlp-document-representation", "nlp-vs-rag", "nlp-token-attribution"),
        extra_mechanism=(
            "Report which task surfaces are unsupervised and therefore claim no quality metric.",
        ),
    ),
    _matrix(
        "tda_capability_matrix",
        "topological data analysis",
        "persistent-homology backends and vectorizations",
        honesty="The native backend and giotto-tda can disagree on diagram details; the matrix says which one will run.",
        next_steps=("fit_tda; transform_tda; evaluate_tda.",),
        concepts=("tda-persistent-homology", "tda-extra-boundary"),
    ),
    _matrix(
        "ranking_capability_matrix",
        "learning to rank",
        "LTR backends and pointwise / pairwise / listwise methods",
        honesty="Ranking metrics depend on query grouping, so an available ranker still needs an honest group split.",
        next_steps=("fit_ranker; rank; evaluate_ranker.",),
        concepts=("ltr-tabular-ranking", "ltr-ranking-metrics"),
    ),
    _matrix(
        "decision_capability_matrix",
        "decision policies",
        "solver backends and decision helper families",
        honesty="An available solver does not validate your cost matrix; the costs are your claim, not the solver's.",
        next_steps=("fit_decision_policy; apply_decisions; evaluate_decisions.",),
        concepts=("decision-cost-matrix", "decision-operating-point"),
    ),
    _matrix(
        "optimize_capability_matrix",
        "decision policies (alias of decision_capability_matrix)",
        "solver backends and decision helper families",
        honesty="This is the same payload as decision_capability_matrix, kept for the optimize-oriented naming.",
        next_steps=("fit_decision_policy; evolutionary_search; apply_decisions.",),
        concepts=("decision-allocation", "decision-cost-matrix"),
    ),
    _matrix(
        "rl_capability_matrix",
        "imitation learning and reinforcement learning",
        "imitation backends, RL modes, and tabular/deep algorithms",
        honesty=(
            "tabular_q being listed does not mean your env observation space fits "
            "in the discretized table — read n_bins and MAX_TABULAR_STATES."
        ),
        next_steps=(
            "fit_imitation; fit_rl(mode='tabular_q', algorithm='q_learning'); act_rl.",
        ),
        concepts=(
            "rl-tabular-q-learning",
            "rl-contextual-bandit",
            "imitation-behavioral-cloning",
        ),
        extra_mechanism=(
            "Publish non_goals (batch offline RL, MuJoCo/robotics) explicitly.",
        ),
    ),
    _matrix(
        "causal_capability_matrix",
        "causal inference",
        "causal backends and estimators",
        honesty="An available estimator does not validate your identification assumptions.",
        next_steps=("declare_causal_assumptions; fit_causal; evaluate_causal.",),
        concepts=("causal-assumptions", "causal-ate-backdoor"),
    ),
    _matrix(
        "federated_capability_matrix",
        "federated learning",
        "Flower / simulation backends and aggregation modes",
        honesty="Simulation availability is not a privacy guarantee for real deployments.",
        next_steps=("fit_federated; evaluate_federated.",),
        concepts=("federated-simulation", "federated-fedavg"),
    ),
    _matrix(
        "graph_capability_matrix",
        "graph machine learning",
        "classical topology + GCN / PyG backends",
        honesty="Classical path requires NetworkX; dense GCN guard limits node count.",
        next_steps=("set_graph; fit_graph; evaluate_graph.",),
        concepts=("graph-inductive-transductive", "graph-gcn"),
    ),
    _matrix(
        "kg_capability_matrix",
        "knowledge graphs",
        "embedding / link-prediction backends",
        honesty="Link-prediction metrics depend on negative sampling and graph split honesty.",
        next_steps=("fit_kg; evaluate_kg; predict_links.",),
        concepts=("kg-link-prediction", "kg-transe-distmult"),
    ),
    _matrix(
        "metalearning_capability_matrix",
        "meta-learning",
        "MAML / industry adaptation backends",
        honesty="Few-shot gains on synthetic tasks may not transfer to your task distribution.",
        next_steps=("fit_metalearning; adapt_to_task; evaluate_metalearning.",),
        concepts=("metalearning-maml", "metalearning-prototypical"),
    ),
    _matrix(
        "multitask_capability_matrix",
        "multi-task learning",
        "shared-trunk / multi-head backends",
        honesty="Shared representation helps only when tasks are related — the matrix cannot judge that.",
        next_steps=("fit_multitask; evaluate_multitask.",),
        concepts=("multitask-multi-output", "multitask-chain"),
    ),
    _matrix(
        "online_capability_matrix",
        "online / incremental learning",
        "partial_fit backends and drift policies",
        honesty="Online updates on shuffled data violate the streaming assumption.",
        next_steps=("fit_online; partial_fit_online; evaluate_online.",),
        concepts=("online-partial-fit", "online-drift-disclose"),
    ),
    _matrix(
        "probabilistic_capability_matrix",
        "probabilistic prediction",
        "interval / distribution backends",
        honesty="Calibrated intervals on train do not guarantee calibration on drifted holdout.",
        next_steps=("fit_probabilistic; predict_interval; evaluate_probabilistic.",),
        concepts=("probabilistic-uncertainty", "probabilistic-split-conformal"),
    ),
    _matrix(
        "recommender_capability_matrix",
        "recommender systems",
        "CF / implicit / hybrid backends",
        honesty="Implicit-feedback paths refuse explicit-only rating semantics.",
        next_steps=("fit_recommender; recommend; evaluate_recommender.",),
        concepts=("recommender-collaborative-filtering", "recommender-cold-start"),
    ),
    _matrix(
        "semisupervised_capability_matrix",
        "semi-supervised learning",
        "pseudo-label and self-training backends",
        honesty="Pseudo-label quality is bounded by the seed model — monitor holdout drift.",
        next_steps=("fit_semisupervised; evaluate_semisupervised.",),
        concepts=("semisupervised-label-missingness", "semisupervised-train-only-fit"),
    ),
    _matrix(
        "activelearning_capability_matrix",
        "active learning",
        "query strategies and uncertainty backends",
        honesty="Query budget on train does not estimate full-label ceiling without disclosure.",
        next_steps=("fit_active_learner; suggest_query; evaluate_active_learning.",),
        concepts=("activelearning-uncertainty", "activelearning-train-pool"),
    ),
    _matrix(
        "automl_capability_matrix",
        "AutoML search",
        "search backends and estimator spaces",
        honesty="Search finds strong configs on the given split — not proof of production readiness.",
        next_steps=("run_automl; evaluate_automl.",),
        concepts=("automl-beyond-hpo", "automl-selection-honesty"),
    ),
    _matrix(
        "ensemble_capability_matrix",
        "ensemble learning",
        "voting / stacking / blending strategies",
        honesty=(
            "Ensembles are core sklearn plus in-tree blending — availability is always "
            "True with core deps; quality still depends on base learners and the split."
        ),
        next_steps=("fit_voting; fit_stacking; fit_blending; evaluate_ensemble.",),
        concepts=("ensemble-voting-vs-single-tree", "leakage-boundary"),
    ),
    _matrix(
        "synthetic_capability_matrix",
        "synthetic data generation",
        "generator backends and evaluation paths",
        honesty="Fidelity availability is not privacy: read the privacy limits before sharing synthetic rows.",
        next_steps=("fit_synthesizer; sample_synthetic; evaluate_synthetic.",),
        concepts=("synthetic-fidelity-vs-tstr", "synthetic-privacy-limits"),
    ),
    _matrix(
        "dl_capability_matrix",
        "deep learning (Torch)",
        "Torch trainer / export / multimodal backends",
        honesty="Availability of Torch does not imply a good architecture for your data.",
        next_steps=("make_torch_loaders; fit_torch; evaluate_torch.",),
        concepts=("early-stopping-partition", "leakage-boundary", "reproducibility"),
    ),
)
