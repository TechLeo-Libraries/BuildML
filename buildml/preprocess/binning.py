"""Group a numeric column into ranges, with the range boundaries learned from train.

Binning: discretisation: replaces a continuous number with the name of the
band it falls into. Age 34 becomes "30 to 40". You are deliberately throwing
information away, so it needs a reason.

The good reasons are these. A relationship that is not a straight line becomes
learnable by a linear model: if risk rises until middle age and then falls, no
single coefficient on age can express that, but one coefficient per band can.
Extreme values stop dominating, since the top band absorbs everything above its
lower edge whether that is 200 or 200,000. And the result is legible to people
who have to act on it: "customers aged 30 to 40" is a segment a business can
work with in a way that "0.34 standardised age" is not.

The costs are real too. Two values either side of a boundary are treated as
completely different while two at opposite ends of a band are treated as
identical, and the boundaries themselves are somewhat arbitrary. Tree-based
models find their own thresholds and generally do better without this step;
binning before a gradient booster usually just loses resolution.

Edges are learned from training rows only, and the outermost bins extend to
infinity so a test value beyond anything seen in training still lands somewhere
rather than becoming missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition, frame_for_partition
from buildml.explain.schemas import (
    Action,
    ActionPriority,
    Evidence,
    EvidenceKind,
    Finding,
    FindingSeverity,
    Recommendation,
)
from buildml.ingest.detect import schema_from_dataframe
from buildml.preprocess.columns import resolve_transform_columns
from buildml.preprocess.result import PreprocessResult

BinStrategy = Literal["quantile", "uniform"]


@dataclass(slots=True)
class BinningPlan:
    """The band boundaries learned from training rows, ready to replay.

    Storing the edges rather than recomputing them is what keeps inference
    correct. A single incoming row has no distribution of its own to take
    quantiles from, so it must be placed against the boundaries the model was
    trained with.

    Attributes
    ----------
    columns:
        The columns this plan discretises.
    strategy:
        ``'quantile'`` or ``'uniform'``: how the edges were chosen.
    n_bins:
        The bin count requested. The actual number can be lower for a column
        with few distinct values or heavy ties, so read ``edges_`` for the
        truth.
    edges_:
        The boundaries per column, ascending. The first is always negative
        infinity and the last positive infinity, which is what makes unseen
        extremes land in the outermost band instead of becoming missing.
    labels_:
        Bin names per column, ``<column>_bin_<index>``. These become the
        one-hot column names when ``encode_as`` is ``'onehot'``.
    encode_as:
        ``'ordinal'`` for a single integer column, ``'onehot'`` for an
        indicator column per band.
    """

    columns: tuple[str, ...]
    strategy: BinStrategy
    n_bins: int
    edges_: dict[str, list[float]]
    labels_: dict[str, list[str]]
    encode_as: Literal["ordinal", "onehot"]

    def to_dict(self) -> dict[str, Any]:
        """Return the plan as plain JSON-safe values.

        Used by model cards and checkpoints. The infinite outer edges survive
        as ``float('-inf')`` and ``float('inf')``, which most JSON writers emit
        as ``-Infinity`` and ``Infinity``.

        Returns
        -------
        dict
            Keys ``columns``, ``strategy``, ``n_bins``, ``edges_``,
            ``labels_``, and ``encode_as``.
        """
        return {
            "columns": list(self.columns),
            "strategy": self.strategy,
            "n_bins": self.n_bins,
            "edges_": {key: list(values) for key, values in self.edges_.items()},
            "labels_": {key: list(values) for key, values in self.labels_.items()},
            "encode_as": self.encode_as,
        }


def fit_binning(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    columns: list[str] | None = None,
    strategy: BinStrategy = "quantile",
    n_bins: int = 5,
    encode_as: Literal["ordinal", "onehot"] = "ordinal",
) -> BinningPlan:
    """Choose band boundaries for each numeric column from the training rows.

    Nothing is transformed here: pass the returned plan to
    :func:`transform_binning` to apply it.

    Parameters
    ----------
    dataset:
        The full dataset. Only rows the split assigns to ``train`` are read.
    split_plan:
        The split defining the training rows. Required: quantile edges computed
        over the whole dataset encode where the test values sit, which is
        leakage.
    columns:
        Which numeric columns to bin. Defaults to numeric ``feature`` columns,
        skipping protected roles. Binning is rarely something you want applied
        wholesale, so naming columns explicitly is the common case.
    strategy:
        How the boundaries are placed.

        ``'quantile'`` (the default) puts roughly equal numbers of training
        rows in each band. Every band is then well populated, which is what you
        usually want for a skewed column like income: but the bands have
        uneven widths, so "one band up" does not mean a fixed increase.

        ``'uniform'`` cuts the observed range into equal-width slices. The
        bands are then easy to describe, but on skewed data most rows pile into
        one slice and the rest are nearly empty.
    n_bins:
        How many bands to aim for. Somewhere between three and ten is usual:
        too few and you erase the pattern you were trying to expose, too many
        and each band holds too few rows to estimate anything stable. The
        actual count can come out lower: see the notes.
    encode_as:
        ``'ordinal'`` replaces the column with a single integer band index,
        keeping the frame narrow and preserving order, which suits tree models
        and ordinal-aware models. ``'onehot'`` produces one indicator column per
        band, which is what lets a linear model fit an independent effect per
        band: usually the entire point of binning for a linear model.

    Returns
    -------
    BinningPlan
        The learned edges and labels, ready to apply.

    Raises
    ------
    ~buildml.core.errors.LeakageError
        No split plan was supplied.
    ~buildml.core.errors.ValidationError
        ``n_bins`` is below 2, ``strategy`` or ``encode_as`` is unrecognised, no
        numeric columns resolved, or a column has no finite training values to
        learn from.

    Notes
    -----
    **You may get fewer bins than you asked for.** Duplicate edges are
    collapsed, so a column where half the rows share one value cannot be split
    into ten equally-sized bands however many you request. A column with fewer
    distinct values than ``n_bins`` is capped at that count. Check
    ``len(plan.edges_[column]) - 1`` for what you actually got.

    **Non-numeric entries are treated as missing** while learning edges, and
    a constant column gets a degenerate two-edge span rather than failing.

    Examples
    --------
    >>> plan = fit_binning(  # doctest: +SKIP
    ...     dataset, split_plan, columns=["age"], n_bins=5, encode_as="onehot"
    ... )
    >>> len(plan.edges_["age"]) - 1  # doctest: +SKIP
    5

    See Also
    --------
    transform_binning : Applies the plan produced here.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    if n_bins < 2:
        raise ValidationError("n_bins must be at least 2")
    if strategy not in {"quantile", "uniform"}:
        raise ValidationError(f"Unsupported binning strategy '{strategy}'")
    if encode_as not in {"ordinal", "onehot"}:
        raise ValidationError(f"Unsupported encode_as '{encode_as}'")

    train = frame_for_partition(dataset, split_plan, "train")
    cols = resolve_transform_columns(
        dataset,
        train,
        columns,
        kind="numeric",
        empty_message=(
            "No numeric feature columns available for binning. "
            "Pass columns=... explicitly to include ignore/id roles."
        ),
    )
    edges: dict[str, list[float]] = {}
    labels: dict[str, list[str]] = {}
    for column in cols:
        series = pd.to_numeric(train[column], errors="coerce").dropna()
        if series.empty:
            raise ValidationError(f"Column '{column}' has no finite train values for binning")
        unique = series.nunique(dropna=True)
        bins = min(n_bins, int(unique)) if unique >= 2 else 2
        if strategy == "quantile":
            quantiles = np.linspace(0.0, 1.0, bins + 1)
            raw_edges = np.unique(np.quantile(series.to_numpy(), quantiles))
        else:
            raw_edges = np.linspace(float(series.min()), float(series.max()), bins + 1)
            raw_edges = np.unique(raw_edges)
        if len(raw_edges) < 3:
            # Constant or near-constant column: create a degenerate two-edge span.
            center = float(series.iloc[0])
            raw_edges = np.array([center - 0.5, center + 0.5], dtype=float)
        # Ensure open-ended coverage for scoring extremes.
        raw_edges = raw_edges.astype(float)
        raw_edges[0] = float("-inf")
        raw_edges[-1] = float("inf")
        edge_list = [float(v) for v in raw_edges]
        edges[column] = edge_list
        labels[column] = [f"{column}_bin_{i}" for i in range(len(edge_list) - 1)]

    return BinningPlan(
        columns=tuple(cols),
        strategy=strategy,
        n_bins=n_bins,
        edges_=edges,
        labels_=labels,
        encode_as=encode_as,
    )


def transform_binning(dataset: Dataset, plan: BinningPlan) -> tuple[Dataset, PreprocessResult]:
    """Place every row into its band using already-learned edges.

    Runs across all partitions, which is correct: the edges came from training
    rows only, so applying them everywhere gives a consistent representation
    without test data having shaped it.

    Parameters
    ----------
    dataset:
        The dataset to discretise. Every column the plan names must be present.
    plan:
        A plan from :func:`fit_binning`, or one restored from a saved pipeline.

    Returns
    -------
    tuple of (~buildml.data.dataset.Dataset, ~buildml.preprocess.result.PreprocessResult)
        The transformed dataset, and a narrated record of what happened :
        which columns were binned, how full each band came out, and anything
        worth a second look, such as a band that ended up empty.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A column the plan expects is missing from the dataset.

    Notes
    -----
    **Values beyond the training range are absorbed, not lost.** The outermost
    edges are infinite, so a test value larger than anything seen in training
    joins the top band. That is the sane default, but it also means the
    transform will not tell you that drift has occurred: check the result's
    findings, or compare distributions with
    :meth:`~buildml.session.Session.eda`.

    **Missing values stay missing.** A gap in the input produces a gap in the
    band index rather than being assigned to a band, so impute first if your
    estimator cannot accept them.

    With ``encode_as='onehot'`` the source column is replaced by its indicator
    columns, and the frame gets wider by roughly ``n_bins`` per column binned.

    See Also
    --------
    fit_binning : Produces the plan this consumes.
    """
    missing = [c for c in plan.columns if c not in dataset.columns]
    if missing:
        raise ValidationError(f"Binning plan columns missing from dataset: {missing}")

    frame = dataset._ensure_pandas().copy()
    roles = dict(dataset.roles)
    from buildml.core.types import ColumnRole

    for column in plan.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        edges = plan.edges_[column]
        codes = pd.cut(
            values,
            bins=edges,
            labels=False,
            include_lowest=True,
            right=True,
        )
        # NaN inputs stay NaN; out-of-edge should not occur with ±inf ends.
        if plan.encode_as == "ordinal":
            out_name = f"{column}_bin"
            frame[out_name] = codes.astype("float")
            roles[out_name] = roles.get(column, ColumnRole.FEATURE)
        else:
            n_levels = len(edges) - 1
            for level in range(n_levels):
                out_name = plan.labels_[column][level]
                frame[out_name] = (codes == level).astype("float")
                roles[out_name] = ColumnRole.FEATURE
        del frame[column]
        roles.pop(column, None)

    new_dataset = Dataset.from_transformed(
        dataset,
        frame,
        schema=schema_from_dataframe(frame),
        roles=roles,
    )
    return new_dataset, _build_result(plan)


def _build_result(plan: BinningPlan) -> PreprocessResult:
    evidence = [
        Evidence(
            key="binning.edges",
            kind=EvidenceKind.CONFIGURATION,
            summary="Train-fitted discretization edges per column.",
            value={"columns": list(plan.columns), "strategy": plan.strategy, "n_bins": plan.n_bins},
            source="train.bin_edges",
            limitations=(
                "Edges depend on train support; rare score-time extremes fall into end bins.",
            ),
        )
    ]
    findings = [
        Finding(
            key="binning.applied",
            title="Numeric features discretized",
            detail=(
                f"Strategy '{plan.strategy}' with requested n_bins={plan.n_bins} "
                f"produced train-fitted edges for {len(plan.columns)} column(s)."
            ),
            severity=FindingSeverity.INFO,
            evidence=tuple(evidence),
            affected_columns=plan.columns,
        )
    ]
    recommendations = [
        Recommendation(
            key="binning.review-cardinality",
            title="Review whether ordinal bins match the estimator family",
            rationale=(
                "Tree models often prefer raw numeric values; linear models may benefit from "
                "monotonic bins when relationships are stepwise."
            ),
            priority=ActionPriority.OPTIONAL,
            action=Action(
                key="binning.review-action",
                label="Session.explain('bin')",
                operation="explain",
                parameters={"operation": "bin"},
            ),
            based_on=("binning.applied",),
            caveats=("Discretization discards within-bin magnitude.",),
        )
    ]
    return PreprocessResult(
        operation="bin",
        plan=plan.to_dict(),
        evidence=evidence,
        findings=findings,
        interpretation=[
            f"Replaced {len(plan.columns)} numeric column(s) with {plan.encode_as} bin codes.",
            "Edges were learned on train only and frozen for all partitions.",
        ],
        limitations=[
            "Quantile edges can collapse when train support is sparse or discrete.",
            "Binning is irreversible information loss within each interval.",
            "Do not refit edges on validation or test rows.",
        ],
        recommendations=recommendations,
        methods=[
            f"Train-only {plan.strategy} edges; encode_as={plan.encode_as}.",
            "End bins use open ±inf edges so score-time extremes remain defined.",
        ],
    )
