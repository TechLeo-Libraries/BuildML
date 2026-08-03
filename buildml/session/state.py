"""Workflow state helpers and history recording for Session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from buildml.checkpoint.validate import ReattachResult
from buildml.core.results import IngestReport
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.eda.report import EDAReport
from buildml.explain.history import (
    make_operation_record,
    prior_state,
    session_state,
)
from buildml.model.compare import ModelComparison
from buildml.model.diagnostics import DiagnosticReport
from buildml.model.plot_boards import PlotBoardReport
from buildml.model.selection import CVScoreResult, NestedCVResult, SearchResult
from buildml.model.supervised import FitResult
from buildml.pipeline.card import ModelCard
from buildml.preprocess.binning import BinningPlan
from buildml.preprocess.custom import CustomTransformPlan
from buildml.preprocess.dates import DateFeaturePlan
from buildml.preprocess.encode import EncodePlan
from buildml.preprocess.imbalance import ResamplePlan
from buildml.preprocess.impute import SimpleImputePlan
from buildml.preprocess.outliers import OutlierPlan
from buildml.preprocess.reduce import ReducePlan
from buildml.preprocess.result import PreprocessResult
from buildml.preprocess.scale import ScalePlan
from buildml.preprocess.select import FeatureSelectPlan
from buildml.preprocess.text import TextFeaturePlan
from buildml.session.audit import DryRunReport, HistorySummary
from buildml.session.walkthrough import WorkflowWalkthroughReport

# Fit-capable Session-global plans that poison fold-eval rows when applied before CV.
FIT_CAPABLE_PLAN_KEYS = (
    "impute_plan",
    "encode_plan",
    "scale_plan",
    "outlier_plan",
    "binning_plan",
    "feature_select_plan",
    "text_plan",
    "reduce_plan",
    "custom_plan",
    "date_plan",
    "resample_plan",
)


@dataclass(slots=True)
class WorkflowState:
    """Typed container for Session-owned workflow slots.

    Session keeps flat ``_`` attributes for compatibility with explain/AI hooks
    that read ``session._dataset`` / ``session._fit_result``. This dataclass
    documents ownership and backs plan restore/clear helpers.
    """

    dataset: Dataset | None = None
    ingest_report: IngestReport | None = None
    split_plan: SplitPlan | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    reattach_result: ReattachResult | None = None

    impute_plan: SimpleImputePlan | None = None
    encode_plan: EncodePlan | None = None
    scale_plan: ScalePlan | None = None
    outlier_plan: OutlierPlan | None = None
    binning_plan: BinningPlan | None = None
    feature_select_plan: FeatureSelectPlan | None = None
    text_plan: TextFeaturePlan | None = None
    reduce_plan: ReducePlan | None = None
    custom_plan: CustomTransformPlan | None = None
    date_plan: DateFeaturePlan | None = None
    resample_plan: ResamplePlan | None = None
    last_preprocess: PreprocessResult | None = None

    fit_result: FitResult | None = None
    last_comparison: ModelComparison | None = None
    last_diagnostic: DiagnosticReport | None = None
    last_plot_board: PlotBoardReport | None = None
    last_cv: CVScoreResult | None = None
    last_nested_cv: NestedCVResult | None = None
    last_search: SearchResult | None = None
    model_card: ModelCard | None = None

    last_eda: EDAReport | None = None
    eda_app_handle: Any | None = None
    last_walkthrough: WorkflowWalkthroughReport | None = None
    last_dry_run: DryRunReport | None = None
    last_history_summary: HistorySummary | None = None

    torch_loaders: Any | None = None
    dl_train_result: Any | None = None
    rag_corpus: Any | None = None
    rag_chunks: Any | None = None
    rag_index: Any | None = None
    rag_index_result: Any | None = None
    rag_retrieve_result: Any | None = None
    rag_eval_result: Any | None = None

    ai_provider: Any | None = None
    ai_egress_config: Any | None = None
    ai_transcript: Any | None = None
    ai_result: Any | None = None
    ai_advisor_result: Any | None = None
    ai_executor_result: Any | None = None
    ai_registry: Any | None = None
    ai_max_iterations: int = 10
    ai_budget_tracker: Any | None = None
    ai_plan_result: Any | None = None

    def plan_objects(self) -> dict[str, Any]:
        """Return Session-global preprocess plan slots keyed by attribute name.

        Used by restore/clear helpers and leakage checks that need a single
        mapping of fit-capable plans without touching private ``_`` attributes.

        Returns
        -------
        dict[str, Any]
            Mapping from plan attribute names to live plan objects or ``None``.
        """
        return {
            "impute_plan": self.impute_plan,
            "encode_plan": self.encode_plan,
            "scale_plan": self.scale_plan,
            "date_plan": self.date_plan,
            "outlier_plan": self.outlier_plan,
            "binning_plan": self.binning_plan,
            "feature_select_plan": self.feature_select_plan,
            "text_plan": self.text_plan,
            "reduce_plan": self.reduce_plan,
            "custom_plan": self.custom_plan,
            "resample_plan": self.resample_plan,
        }

    def preprocess_summary(self) -> dict[str, Any]:
        """Return JSON-safe summaries of attached preprocess plans.

        Each plan's :meth:`~object.to_dict` output is included when present so
        walkthrough and checkpoint surfaces can show what was fitted on the
        Session train partition.

        Returns
        -------
        dict[str, Any]
            Mapping from preprocess step names to plan dicts or ``None``.
        """
        return {
            "impute": None if self.impute_plan is None else self.impute_plan.to_dict(),
            "encode": None if self.encode_plan is None else self.encode_plan.to_dict(),
            "scale": None if self.scale_plan is None else self.scale_plan.to_dict(),
            "dates": None if self.date_plan is None else self.date_plan.to_dict(),
            "outliers": None if self.outlier_plan is None else self.outlier_plan.to_dict(),
            "binning": None if self.binning_plan is None else self.binning_plan.to_dict(),
            "feature_select": (
                None if self.feature_select_plan is None else self.feature_select_plan.to_dict()
            ),
            "text": None if self.text_plan is None else self.text_plan.to_dict(),
            "reduce": None if self.reduce_plan is None else self.reduce_plan.to_dict(),
            "custom": None if self.custom_plan is None else self.custom_plan.to_dict(),
            "resample": None if self.resample_plan is None else self.resample_plan.to_dict(),
        }

    def restore_plans(self, plans: dict[str, Any] | None) -> None:
        """Restore preprocess plan slots from a checkpoint or bundle payload.

        Missing keys leave the corresponding slot unchanged; ``None`` values
        clear individual plans without touching unrelated Session state.

        Parameters
        ----------
        plans:
            Mapping of plan attribute names to deserialized plan objects, or
            ``None`` to no-op.
        """
        payload = plans or {}
        self.impute_plan = payload.get("impute_plan")
        self.encode_plan = payload.get("encode_plan")
        self.scale_plan = payload.get("scale_plan")
        self.date_plan = payload.get("date_plan")
        self.outlier_plan = payload.get("outlier_plan")
        self.binning_plan = payload.get("binning_plan")
        self.feature_select_plan = payload.get("feature_select_plan")
        self.text_plan = payload.get("text_plan")
        self.reduce_plan = payload.get("reduce_plan")
        self.custom_plan = payload.get("custom_plan")
        self.resample_plan = payload.get("resample_plan")

    def clear_plans(self) -> None:
        """Clear every Session-global preprocess plan slot on this state object.

        Does not reset dataset, split, fit results, or history: only the
        train-global transform plans that can poison fold-local evaluation.
        """
        self.impute_plan = None
        self.encode_plan = None
        self.scale_plan = None
        self.date_plan = None
        self.outlier_plan = None
        self.binning_plan = None
        self.feature_select_plan = None
        self.text_plan = None
        self.reduce_plan = None
        self.custom_plan = None
        self.resample_plan = None

    def session_preprocess_applied(self) -> bool:
        """Report whether any fit-capable Session-global preprocess plan exists.

        When ``True``, CV and search operations may need fold-local
        :class:`~buildml.preprocess.fold.PreprocessRecipe` overrides or an
        explicit opt-in before scores are trustworthy.

        Returns
        -------
        bool
            ``True`` when at least one plan in :data:`FIT_CAPABLE_PLAN_KEYS`
            is attached.
        """
        plans = self.plan_objects()
        return any(plans.get(key) is not None for key in FIT_CAPABLE_PLAN_KEYS)


def session_preprocess_applied(session: Any) -> bool:
    """Report whether Session-level train-global preprocess plans exist.

    Module-level helper for explain hooks and audit checks that read flat
    ``session._`` plan attributes rather than a :class:`WorkflowState` instance.

    Parameters
    ----------
    session:
        Live :class:`~buildml.session.session.Session` whose ``_*_plan``
        attributes are inspected.

    Returns
    -------
    bool
        ``True`` when any fit-capable preprocess plan slot is non-``None``.
    """
    return any(plan is not None for plan in plan_objects(session).values())


def plan_objects(session: Any) -> dict[str, Any]:
    """Return Session-global preprocess plan slots from flat Session attributes.

    Mirrors :meth:`WorkflowState.plan_objects` for callers that operate on the
    live Session object rather than a typed state container.

    Parameters
    ----------
    session:
        Live :class:`~buildml.session.session.Session` exposing ``_*_plan``
        attributes.

    Returns
    -------
    dict[str, Any]
        Mapping from plan attribute names to live plan objects or ``None``.
    """
    return {
        "impute_plan": session._impute_plan,
        "encode_plan": session._encode_plan,
        "scale_plan": session._scale_plan,
        "date_plan": session._date_plan,
        "outlier_plan": session._outlier_plan,
        "binning_plan": session._binning_plan,
        "feature_select_plan": session._feature_select_plan,
        "text_plan": session._text_plan,
        "reduce_plan": session._reduce_plan,
        "custom_plan": session._custom_plan,
        "resample_plan": session._resample_plan,
    }


def preprocess_summary(session: Any) -> dict[str, Any]:
    """Return JSON-safe summaries of Session-attached preprocess plans.

    Convenience wrapper around flat ``session._`` plan attributes for explain
    hooks and reporting surfaces.

    Parameters
    ----------
    session:
        Live :class:`~buildml.session.session.Session` whose preprocess plan
        slots are serialized.

    Returns
    -------
    dict[str, Any]
        Mapping from preprocess step names to plan dicts or ``None``.
    """
    return {
        "impute": None if session._impute_plan is None else session._impute_plan.to_dict(),
        "encode": None if session._encode_plan is None else session._encode_plan.to_dict(),
        "scale": None if session._scale_plan is None else session._scale_plan.to_dict(),
        "dates": None if session._date_plan is None else session._date_plan.to_dict(),
        "outliers": None if session._outlier_plan is None else session._outlier_plan.to_dict(),
        "binning": None if session._binning_plan is None else session._binning_plan.to_dict(),
        "feature_select": (
            None if session._feature_select_plan is None else session._feature_select_plan.to_dict()
        ),
        "text": None if session._text_plan is None else session._text_plan.to_dict(),
        "reduce": None if session._reduce_plan is None else session._reduce_plan.to_dict(),
        "custom": None if session._custom_plan is None else session._custom_plan.to_dict(),
        "resample": None if session._resample_plan is None else session._resample_plan.to_dict(),
    }


def restore_plans(session: Any, plans: dict[str, Any] | None) -> None:
    """Restore Session preprocess plan slots from a checkpoint payload.

    Writes directly to ``session._*`` attributes so reattached bundles restore
    the same train-global transform state recorded at save time.

    Parameters
    ----------
    session:
        Live :class:`~buildml.session.session.Session` whose plan slots are
        updated in place.
    plans:
        Mapping of plan attribute names to deserialized plan objects, or
        ``None`` to no-op.
    """
    payload = plans or {}
    session._impute_plan = payload.get("impute_plan")
    session._encode_plan = payload.get("encode_plan")
    session._scale_plan = payload.get("scale_plan")
    session._date_plan = payload.get("date_plan")
    session._outlier_plan = payload.get("outlier_plan")
    session._binning_plan = payload.get("binning_plan")
    session._feature_select_plan = payload.get("feature_select_plan")
    session._text_plan = payload.get("text_plan")
    session._reduce_plan = payload.get("reduce_plan")
    session._custom_plan = payload.get("custom_plan")
    session._resample_plan = payload.get("resample_plan")


def clear_plans(session: Any) -> None:
    """Clear every Session-global preprocess plan slot on the live Session.

    Does not reset dataset, split, fit results, or history: only the
    train-global transform plans that can poison fold-local evaluation.

    Parameters
    ----------
    session:
        Live :class:`~buildml.session.session.Session` whose ``_*_plan``
        attributes are set to ``None``.
    """
    session._impute_plan = None
    session._encode_plan = None
    session._scale_plan = None
    session._date_plan = None
    session._outlier_plan = None
    session._binning_plan = None
    session._feature_select_plan = None
    session._text_plan = None
    session._reduce_plan = None
    session._custom_plan = None
    session._resample_plan = None


def record(
    session: Any,
    action: str,
    details: dict[str, Any] | None = None,
    *,
    decision_origin: Literal["automatic", "recommended", "explicit"] = "explicit",
    warnings: list[str] | tuple[str, ...] = (),
    result_summary: dict[str, Any] | None = None,
) -> None:
    """Append one versioned operation record to Session history.

    Captures before/after workflow snapshots, decision origin, warnings, and an
    optional result summary so explain, audit, and walkthrough surfaces can
    replay what changed without re-executing the operation.

    Parameters
    ----------
    session:
        Live :class:`~buildml.session.session.Session` whose ``_history`` list
        receives the new record.
    action:
        Public operation identifier stored as ``operation_id``.
    details:
        Parameter mapping persisted with the record, or ``None`` for empty
        parameters.
    decision_origin:
        Whether the call was ``explicit``, ``recommended``, or ``automatic``.
    warnings:
        Leakage, scope, or policy warnings to surface in audit summaries.
    result_summary:
        Compact result metadata when the operation produced a structured
        outcome worth replaying offline.
    """
    before = prior_state(session._history)
    after = session_state(session)
    session._history.append(
        make_operation_record(
            sequence=len(session._history) + 1,
            operation_id=action,
            parameters=details,
            decision_origin=decision_origin,
            before=before,
            after=after,
            warnings=warnings,
            result_summary=result_summary,
        )
    )
