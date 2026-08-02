"""Shared imports for Session orchestration modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from buildml.checkpoint.bundle import load_checkpoint, save_checkpoint
from buildml.checkpoint.validate import ReattachResult
from buildml.core.errors import ValidationError
from buildml.core.results import IngestReport
from buildml.core.types import ColumnRole, DataMode, EngineName, coerce_data_mode
from buildml.data.dataset import Dataset
from buildml.data.engines.prep import MaterializePrepResult, prepare_design_frame
from buildml.data.splits import (
    PartitionName,
    SplitPlan,
    assert_fit_partition,
    create_group_split,
    create_split,
    create_time_split,
    frame_for_partition,
    inject_partitions,
)
from buildml.eda.profile import explore_dataset
from buildml.eda.report import EDAReport
from buildml.explain.history import (
    make_operation_record,
    normalize_history,
    prior_state,
    session_state,
)
from buildml.explain.resolver import explain as explain_session
from buildml.explain.resolver import resolve_workflow
from buildml.explain.schemas import WorkflowStep
from buildml.ingest.pipeline import ingest as ingest_source
from buildml.model.compare import ModelComparison, compare_estimators
from buildml.model.diagnostics import (
    DiagnosticReport,
    calibration_report,
    learning_curve_report,
    permutation_importance_report,
    segment_error_report,
    threshold_report,
)
from buildml.model.plot_boards import PlotBoardReport, build_eval_plot_board
from buildml.model.selection import (
    CVScoreResult,
    NestedCVResult,
    SearchResult,
)
from buildml.model.selection import (
    cv_score as run_cv_score,
)
from buildml.model.selection import (
    evolutionary_search as run_evolutionary_search,
)
from buildml.model.selection import (
    grid_search as run_grid_search,
)
from buildml.model.selection import (
    nested_cv_score as run_nested_cv_score,
)
from buildml.model.selection import (
    optuna_search as run_optuna_search,
)
from buildml.model.selection import (
    randomized_search as run_randomized_search,
)
from buildml.model.supervised import (
    EvaluateResult,
    FitResult,
    evaluate_estimator,
    fit_estimator,
    materialize_partition_design,
    predict_estimator,
)
from buildml.pipeline.bundle import load_pipeline_bundle, save_pipeline_bundle
from buildml.pipeline.card import ModelCard
from buildml.pipeline.persist import load_fit_result, save_fit_result
from buildml.pipeline.score import PipelinePredictResult
from buildml.pipeline.score import predict_from_pipeline as run_predict_from_pipeline
from buildml.preprocess.apply import ApplyPlansResult
from buildml.preprocess.apply import apply_preprocess_plans as run_apply_preprocess_plans
from buildml.preprocess.binning import BinningPlan, fit_binning, transform_binning
from buildml.preprocess.columns import drop_columns as drop_columns_transform
from buildml.preprocess.custom import (
    CustomTransformPlan,
    CustomTransformSpec,
    fit_custom_transform,
    transform_custom,
)
from buildml.preprocess.custom import (
    list_transforms as list_registered_transforms,
)
from buildml.preprocess.custom import (
    register_transform as register_custom_transform,
)
from buildml.preprocess.dates import DateFeaturePlan, extract_date_features
from buildml.preprocess.encode import EncodePlan, fit_encoder, transform_encoder
from buildml.preprocess.fold import PreprocessRecipe
from buildml.preprocess.imbalance import (
    ResamplePlan,
    list_resample_strategies,
    resample_train,
)
from buildml.preprocess.impute import (
    SimpleImputePlan,
    fit_simple_imputer,
    transform_simple_imputer,
)
from buildml.preprocess.outliers import OutlierPlan, apply_outlier_plan, fit_outlier_plan
from buildml.preprocess.reduce import ReducePlan, fit_reducer, transform_reducer
from buildml.preprocess.result import PreprocessResult
from buildml.preprocess.scale import ScalePlan, fit_scaler, transform_scaler
from buildml.preprocess.select import (
    FeatureSelectPlan,
    fit_feature_selector,
    transform_feature_selector,
)
from buildml.preprocess.text import TextFeaturePlan, fit_text_features, transform_text_features
from buildml.session.audit import DryRunReport, HistorySummary
from buildml.session.audit import dry_run_session as run_dry_run
from buildml.session.audit import summarize_history as build_history_summary
from buildml.session.walkthrough import WorkflowWalkthroughReport, build_walkthrough


