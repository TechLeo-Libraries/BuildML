# ruff: noqa: E501
"""Prerequisite registry: how each precondition is checked, named, and satisfied.

One prerequisite key answers three questions, and every consumer needs a
different one:

* :func:`probe`: is it satisfied right now, and what should the expert message
  say? Used by the workflow resolver.
* ``PROVIDERS``: which operations establish it? Used to build prerequisite
  chains and remedies.
* ``plain_prerequisite``: how do you say it to someone who has never used
  BuildML? Used by the beginner primer.

Keeping all three in one table is what stops them drifting apart, which is how
the resolver previously ended up unable to evaluate a third of the catalog's
prerequisite keys.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

from buildml.core.types import ColumnRole

PROVIDERS: dict[str, tuple[str, ...]] = {
    "dataset": ("ingest", "checkpoint_load", "reattach"),
    "roles": ("set_roles",),
    "split": ("split", "inject_split", "group_split", "time_split"),
    "fit": (
        "fit",
        "compare_models",
        "load_model",
        "load_pipeline",
        "grid_search",
        "randomized_search",
        "optuna_search",
        "evolutionary_search",
        "fit_voting",
        "fit_stacking",
        "fit_blending",
        "load_ensemble_bundle",
        "run_automl",
        "load_automl_bundle",
    ),
    "ensemble-plan": (
        "fit_voting",
        "fit_stacking",
        "fit_blending",
        "load_ensemble_bundle",
    ),
    "automl-plan": ("run_automl", "load_automl_bundle"),
    "forecast-plan": ("fit_forecast", "load_forecast_bundle"),
    "anomaly-plan": ("fit_anomaly", "load_anomaly_bundle"),
    "semisupervised-plan": ("fit_semisupervised", "load_semisupervised_bundle"),
    "ssl-plan": ("fit_ssl_pretext", "load_ssl_bundle"),
    "ssl-head": ("finetune_ssl_head",),
    "activelearning-plan": ("fit_active_learner", "load_active_learning_bundle"),
    "online-plan": ("fit_online", "load_online_bundle"),
    "multitask-plan": ("fit_multitask", "load_multitask_bundle"),
    "metalearning-plan": ("fit_metalearning", "load_metalearning_bundle"),
    "federated-plan": ("fit_federated", "load_federated_bundle"),
    "probabilistic-plan": ("fit_probabilistic", "load_probabilistic_bundle"),
    "causal-assumptions": ("declare_causal_assumptions", "fit_causal", "load_causal_bundle"),
    "causal-plan": ("fit_causal", "load_causal_bundle"),
    "graph-spec": ("set_graph", "load_graph_bundle"),
    "graph-plan": ("fit_graph", "load_graph_bundle"),
    "symbolic-plan": ("fit_symbolic", "load_symbolic_bundle"),
    "neuro-symbolic-plan": ("fit_neuro_symbolic", "load_symbolic_bundle"),
    "cbr-plan": ("fit_cbr", "load_cbr_bundle"),
    "imitation-plan": ("fit_imitation", "load_imitation_bundle"),
    "rl-plan": ("fit_rl", "load_rl_bundle"),
    "kg-plan": ("fit_kg", "load_kg_bundle"),
    "tda-plan": ("fit_tda", "load_tda_bundle"),
    "ranker-plan": ("fit_ranker", "load_ranker_bundle"),
    "recommender-plan": ("fit_recommender", "load_recommender_bundle"),
    "decision-plan": ("fit_decision_policy", "load_decision_bundle"),
    "synthesizer-plan": ("fit_synthesizer", "load_synthetic_bundle"),
    "nlp-text-column": ("set_roles", "ingest"),
    "nlp-text-plan": ("fit_text_classifier", "load_nlp_bundle"),
    "nlp-topic-plan": ("fit_topics", "load_nlp_bundle"),
    "fit_torch": ("fit_torch", "load_torch_bundle", "fit_torch_ddp"),
    "rag-corpus": ("rag_ingest_corpus",),
    "rag-index": ("rag_embed_and_index", "load_rag_bundle", "rag_upsert"),
    "cluster-plan": ("fit_clusters", "load_unsupervised_bundle"),
    "ai-provider": ("ai_configure",),
    "torch-extra": (),
    "rag-extra": (),
    "nlp-extra": (),
    "ai-extra": (),
    "viz-extra": (),
    "dashboard-extra": (),
}
"""Prerequisite key → the operations that can establish it."""


@dataclass(frozen=True, slots=True)
class PrerequisiteProbe:
    """How one prerequisite is checked and described.

    ``attributes`` is an *any-of* check against private Session state, which is
    how nearly every domain plan is detected. ``module`` covers optional
    dependency groups. ``custom`` exists for the handful of preconditions that
    are genuinely structural rather than a single attribute.
    """

    key: str
    plain: str
    label: str
    attributes: tuple[str, ...] = ()
    module: str | None = None
    remedy: str = ""
    custom: Callable[[Any], tuple[bool, str]] | None = field(default=None, compare=False)

    def check(self, session: Any) -> tuple[bool, str]:
        """Evaluate this precondition against a live session.

        Three strategies cover the whole catalog: a custom callable for anything
        with real logic, an importable module name for optional-extra gates, and
        otherwise a list of session attributes where any one being set means the
        state exists.

        Parameters
        ----------
        session:
            The session to inspect. Read with ``getattr`` defaults throughout,
            so a partially built session cannot raise here.

        Returns
        -------
        tuple of (bool, str)
            Whether the precondition holds, and the sentence explaining that
            verdict to an expert reader. :func:`plain_prerequisite` gives the
            beginner phrasing of the same condition.
        """
        if self.custom is not None:
            return self.custom(session)
        if self.module is not None:
            present = find_spec(self.module) is not None
            return present, (
                f"{self.label} are installed."
                if present
                else f"{self.label} are not installed. {self.remedy}".strip()
            )
        present = any(getattr(session, name, None) is not None for name in self.attributes)
        return present, (
            f"{self.label} is attached."
            if present
            else f"No {self.label} is attached. {self.remedy}".strip()
        )


def _check_dataset(session: Any) -> tuple[bool, str]:
    present = getattr(session, "_dataset", None) is not None
    return present, (
        "A materialized dataset is attached."
        if present
        else "No materialized dataset is attached."
    )


def _check_roles(session: Any) -> tuple[bool, str]:
    dataset = getattr(session, "_dataset", None)
    roles = {} if dataset is None else dataset.roles
    has_feature = ColumnRole.FEATURE in roles.values()
    has_target = ColumnRole.TARGET in roles.values()
    missing = [
        label
        for label, present in (("feature role", has_feature), ("target role", has_target))
        if not present
    ]
    return (
        has_feature and has_target,
        "Feature and target roles are assigned."
        if not missing
        else f"Missing {', '.join(missing)}.",
    )


def _check_split(session: Any) -> tuple[bool, str]:
    present = getattr(session, "_split_plan", None) is not None
    return present, (
        "A train/evaluation split exists." if present else "No train/evaluation split exists."
    )


def _check_text_column(session: Any) -> tuple[bool, str]:
    """Report whether the dataset actually carries a string-like column.

    BuildML has no ``text`` column role: the documents are chosen by
    ``text_column=`` at call time. The honest check is therefore whether any
    string-like column exists, and whether the choice is unambiguous.
    """
    if getattr(session, "_nlp_text_plan", None) is not None:
        return True, "A fitted text plan already resolved its text column."
    dataset = getattr(session, "_dataset", None)
    if dataset is None:
        return False, "No dataset is attached, so no text column can be resolved."
    from buildml.nlp.features import candidate_text_columns

    candidates = candidate_text_columns(dataset._ensure_pandas())
    if not candidates:
        return False, "No string-like column exists on the dataset."
    if len(candidates) == 1:
        return True, f"One string-like column is unambiguous: {candidates[0]!r}."
    return True, (
        f"Several string-like columns exist ({', '.join(candidates[:4])}); pass "
        "text_column= so the choice is yours rather than inferred."
    )


def _plan(key: str, plain: str, label: str, attribute: str, remedy: str) -> PrerequisiteProbe:
    return PrerequisiteProbe(
        key=key, plain=plain, label=label, attributes=(attribute,), remedy=remedy
    )


_PROBES: tuple[PrerequisiteProbe, ...] = (
    PrerequisiteProbe(
        key="dataset",
        plain="Your data has to be loaded into the session first.",
        label="dataset",
        custom=_check_dataset,
    ),
    PrerequisiteProbe(
        key="roles",
        plain="BuildML has to know which column is the answer you want and which columns it may learn from.",
        label="roles",
        custom=_check_roles,
    ),
    PrerequisiteProbe(
        key="split",
        plain="Some rows must already be set aside for testing, so this step cannot peek at them.",
        label="split",
        custom=_check_split,
    ),
    _plan(
        "fit",
        "A model has to be trained (or loaded) before you can use it.",
        "active fitted estimator",
        "_fit_result",
        "Run fit, compare_models, or load_model first.",
    ),
    _plan(
        "fit_torch",
        "A neural network has to be trained (or loaded) first.",
        "active Torch trainer",
        "_dl_train_result",
        "Run fit_torch or load_torch_bundle first.",
    ),
    _plan(
        "cluster-plan",
        "Clusters have to be found on the training rows before anything can be assigned to them.",
        "train-fitted ClusterPlan",
        "_cluster_plan",
        "Run fit_clusters or load_unsupervised_bundle first.",
    ),
    _plan(
        "ensemble-plan",
        "The combined model has to be built before you can use or score it.",
        "train-fitted EnsemblePlan",
        "_ensemble_plan",
        "Run fit_voting, fit_stacking, fit_blending, or load_ensemble_bundle first.",
    ),
    _plan(
        "automl-plan",
        "The automatic search has to run before there is a chosen model to talk about.",
        "train-selected AutoMLPlan",
        "_automl_plan",
        "Run run_automl or load_automl_bundle first.",
    ),
    _plan(
        "forecast-plan",
        "A forecasting model has to be fitted on the history before it can predict the future.",
        "train-fitted ForecastPlan",
        "_forecast_plan",
        "Run fit_forecast or load_forecast_bundle first.",
    ),
    _plan(
        "anomaly-plan",
        "The detector has to learn what normal looks like before it can flag anything.",
        "train-fitted AnomalyPlan",
        "_anomaly_plan",
        "Run fit_anomaly or load_anomaly_bundle first.",
    ),
    _plan(
        "semisupervised-plan",
        "The semi-supervised learner has to be fitted before it can label anything.",
        "train-fitted SemiSupervisedPlan",
        "_semisupervised_plan",
        "Run fit_semisupervised or load_semisupervised_bundle first.",
    ),
    _plan(
        "ssl-plan",
        "The pretext task has to be trained before there is a representation to reuse.",
        "train-fitted SelfSupervisedPlan",
        "_ssl_plan",
        "Run fit_ssl_pretext or load_ssl_bundle first.",
    ),
    _plan(
        "ssl-head",
        "A prediction head has to be fine-tuned on top of the learned representation.",
        "SSLHeadPlan",
        "_ssl_head_plan",
        "Run finetune_ssl_head first.",
    ),
    _plan(
        "activelearning-plan",
        "The active learner has to be started before it can suggest what to label next.",
        "train-fitted ActiveLearningPlan",
        "_activelearning_plan",
        "Run fit_active_learner or load_active_learning_bundle first.",
    ),
    _plan(
        "online-plan",
        "The streaming model has to be warm-started before it can take updates.",
        "warm-started OnlinePlan",
        "_online_plan",
        "Run fit_online or load_online_bundle first.",
    ),
    _plan(
        "multitask-plan",
        "The multi-target model has to be fitted before it can predict several answers at once.",
        "train-fitted MultiTaskPlan",
        "_multitask_plan",
        "Run fit_multitask or load_multitask_bundle first.",
    ),
    _plan(
        "metalearning-plan",
        "The meta-learner has to be trained across tasks before it can adapt to a new one.",
        "train-fitted MetaLearningPlan",
        "_metalearning_plan",
        "Run fit_metalearning or load_metalearning_bundle first.",
    ),
    _plan(
        "federated-plan",
        "The federated rounds have to run before there is a shared global model.",
        "train-fitted FederatedPlan",
        "_federated_plan",
        "Run fit_federated or load_federated_bundle first.",
    ),
    _plan(
        "probabilistic-plan",
        "The uncertainty model has to be fitted before it can produce intervals.",
        "train-fitted ProbabilisticPlan",
        "_probabilistic_plan",
        "Run fit_probabilistic or load_probabilistic_bundle first.",
    ),
    PrerequisiteProbe(
        key="causal-assumptions",
        plain="You have to write down what you are assuming about cause and effect; the data cannot tell you.",
        label="validated CausalAssumptions",
        attributes=("_causal_assumptions", "_causal_plan"),
        remedy="Run declare_causal_assumptions, or pass assumptions= into fit_causal. Exploratory analysis is not a substitute.",
    ),
    _plan(
        "causal-plan",
        "The causal estimator has to be fitted before an effect can be reported.",
        "train-fitted CausalPlan",
        "_causal_plan",
        "Run fit_causal or load_causal_bundle first.",
    ),
    PrerequisiteProbe(
        key="graph-spec",
        plain="You have to attach the list of connections between your rows.",
        label="GraphSpec edge list",
        attributes=("_graph_spec", "_graph_plan"),
        remedy="Run set_graph(edges, node_id_col=...) first.",
    ),
    _plan(
        "graph-plan",
        "The graph model has to be fitted before it can predict anything about nodes.",
        "train-fitted GraphPlan",
        "_graph_plan",
        "Run fit_graph or load_graph_bundle first.",
    ),
    _plan(
        "symbolic-plan",
        "The rule list has to exist before it can decide anything.",
        "train-fitted SymbolicPlan",
        "_symbolic_plan",
        "Run fit_symbolic or load_symbolic_bundle first.",
    ),
    _plan(
        "neuro-symbolic-plan",
        "The combined rules-plus-model has to be fitted first.",
        "train-fitted NeuroSymbolicPlan",
        "_neuro_symbolic_plan",
        "Run fit_neuro_symbolic or load_symbolic_bundle first.",
    ),
    _plan(
        "cbr-plan",
        "The memory of past solved cases has to be built first.",
        "train-fitted CbrPlan (case memory)",
        "_cbr_plan",
        "Run fit_cbr or load_cbr_bundle first.",
    ),
    _plan(
        "imitation-plan",
        "The policy has to be cloned from recorded expert decisions first.",
        "train-fitted ImitationPlan (behavioral cloning)",
        "_imitation_plan",
        "Run fit_imitation or load_imitation_bundle first.",
    ),
    _plan(
        "rl-plan",
        "The decision policy has to be trained before it can choose actions.",
        "fitted RlPlan (bandit or gym policy)",
        "_rl_plan",
        "Run fit_rl or load_rl_bundle first.",
    ),
    _plan(
        "kg-plan",
        "The knowledge-graph embeddings have to be trained before facts can be scored.",
        "train-fitted knowledge-graph plan",
        "_kg_plan",
        "Run fit_kg or load_kg_bundle first.",
    ),
    _plan(
        "tda-plan",
        "The shape-extraction pipeline has to be fitted on the training rows first.",
        "train-fitted TdaPlan",
        "_tda_plan",
        "Run fit_tda or load_tda_bundle first.",
    ),
    _plan(
        "ranker-plan",
        "The ranking model has to be fitted before it can order anything.",
        "train-fitted ranker plan",
        "_ranker_plan",
        "Run fit_ranker or load_ranker_bundle first.",
    ),
    _plan(
        "recommender-plan",
        "The recommender has to learn from past interactions before it can suggest items.",
        "train-fitted recommender plan",
        "_recommender_plan",
        "Run fit_recommender or load_recommender_bundle first.",
    ),
    _plan(
        "decision-plan",
        "The decision policy has to be built before allocations can be produced.",
        "fitted decision plan",
        "_decision_plan",
        "Run fit_decision_policy or load_decision_bundle first.",
    ),
    _plan(
        "synthesizer-plan",
        "The generator has to learn the shape of your training data before it can invent rows.",
        "train-fitted SynthesizerPlan",
        "_synthesizer_plan",
        "Run fit_synthesizer or load_synthetic_bundle first.",
    ),
    PrerequisiteProbe(
        key="nlp-text-column",
        plain="There has to be a column of actual text to work on.",
        label="text column",
        custom=_check_text_column,
    ),
    _plan(
        "nlp-text-plan",
        "The text model has to be fitted before it can classify or be interpreted.",
        "fitted NlpTextPlan",
        "_nlp_text_plan",
        "Run fit_text_classifier or load_nlp_bundle first.",
    ),
    _plan(
        "nlp-topic-plan",
        "Topics have to be fitted on the training documents before anything can be assigned to them.",
        "fitted topic plan",
        "_nlp_topic_plan",
        "Run fit_topics or load_nlp_bundle first.",
    ),
    _plan(
        "rag-corpus",
        "The documents have to be loaded before they can be chunked or indexed.",
        "RAG corpus",
        "_rag_corpus",
        "Run rag_ingest_corpus first.",
    ),
    PrerequisiteProbe(
        key="rag-index",
        plain="The documents have to be indexed before anything can be retrieved from them.",
        label="active RAG index",
        attributes=("_rag_index_result", "_rag_index"),
        remedy="Run rag_embed_and_index, rag_upsert, or load_rag_bundle first.",
    ),
    _plan(
        "ai-provider",
        "An AI provider has to be configured before any language-model call can be made.",
        "configured AI provider",
        "_ai_provider",
        "Run ai_configure first.",
    ),
    PrerequisiteProbe(
        key="torch-extra",
        plain="Deep-learning support is an optional install that has to be present.",
        label="Torch dependencies",
        module="torch",
        remedy="Install buildml[torch] before Torch Session methods.",
    ),
    PrerequisiteProbe(
        key="rag-extra",
        plain="Retrieval support is an optional install that has to be present.",
        label="RAG dependencies",
        module="sentence_transformers",
        remedy="Install buildml[rag] before RAG Session methods.",
    ),
    PrerequisiteProbe(
        key="nlp-extra",
        plain="The heavier text stack is an optional install; the core text path does not need it.",
        label="optional NLP dependencies",
        module="sentence_transformers",
        remedy="Install buildml[nlp] for embedding and transformer text backends.",
    ),
    PrerequisiteProbe(
        key="ai-extra",
        plain="Language-model support is an optional install that has to be present.",
        label="AI dependencies",
        module="openai",
        remedy="Install buildml[ai] before AI Session methods.",
    ),
    PrerequisiteProbe(
        key="viz-extra",
        plain="Plotting is an optional install; everything else still works without it.",
        label="visualization dependencies",
        module="matplotlib",
        remedy="Install buildml[viz] only when plots are requested.",
    ),
    PrerequisiteProbe(
        key="dashboard-extra",
        plain="The local dashboard is an optional install.",
        label="dashboard dependencies",
        module="fastapi",
        remedy="Install buildml[dashboard] before calling Session.eda_app(...).",
    ),
)

PREREQUISITES: dict[str, PrerequisiteProbe] = {probe.key: probe for probe in _PROBES}


def probe(session: Any, key: str) -> tuple[bool, str]:
    """Ask whether one precondition currently holds on a session.

    This is what turns a catalog prerequisite into a workflow status. An unknown
    key reports as unsatisfied with a diagnostic rather than raising, so a single
    stale catalog entry degrades one row of the workflow view instead of
    breaking the whole resolution.

    Parameters
    ----------
    session:
        The session to inspect.
    key:
        A prerequisite key from a catalog entry, such as ``'split'`` or
        ``'torch-extra'``.

    Returns
    -------
    tuple of (bool, str)
        Whether it holds, and the expert-facing sentence explaining why.

    See Also
    --------
    plain_prerequisite : The same condition, phrased for a beginner.
    providers_for : The operations that would satisfy it.
    """
    entry = PREREQUISITES.get(key)
    if entry is None:
        return False, f"Unknown prerequisite '{key}'."
    return entry.check(session)


def plain_prerequisite(key: str) -> str:
    """Say what a precondition means to someone who has not used BuildML.

    The expert message from :func:`probe` names session attributes and plan
    objects. This is the same condition stated as something a newcomer can act
    on, and it is what the beginner primer renders.

    Parameters
    ----------
    key:
        A prerequisite key from a catalog entry.

    Returns
    -------
    str
        A complete sentence. Unknown keys get an honest placeholder rather than
        an error, since a missing phrasing should not break an explanation.

    Examples
    --------
    >>> from buildml.explain import plain_prerequisite
    >>> plain_prerequisite("split")
    'Some rows must already be set aside for testing, so this step cannot peek at them.'
    """
    entry = PREREQUISITES.get(key)
    if entry is not None:
        return entry.plain
    return f"An internal condition named '{key}' has to hold."


def providers_for(key: str) -> tuple[str, ...]:
    """List the operations that would satisfy a precondition.

    Telling someone a step is blocked is only half an answer; this supplies the
    other half. It backs the remedies in the workflow resolver, the prerequisite
    chains in dry-run reports, and the "establish it with…" clause in beginner
    primers.

    Parameters
    ----------
    key:
        A prerequisite key from a catalog entry.

    Returns
    -------
    tuple of str
        Operation names, most direct first. Empty when the condition is not
        established by a BuildML call: an optional extra, for instance, which
        is satisfied by installing a package.
    """
    return PROVIDERS.get(key, ())


__all__ = [
    "PREREQUISITES",
    "PROVIDERS",
    "PrerequisiteProbe",
    "plain_prerequisite",
    "probe",
    "providers_for",
]
