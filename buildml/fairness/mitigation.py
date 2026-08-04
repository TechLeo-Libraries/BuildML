"""Opt-in post-hoc fairness helpers (thresholds / reweighing).

These tools return **suggestions** (per-group thresholds or sample weights).
They never mutate a Session model, never silently rewrite predictions, and
are **not** legal certification or proof that bias has been removed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from buildml.core.errors import ValidationError
from buildml.fairness.metrics import _as_bool_pred, group_selection_rates

ThresholdTarget = Literal["demographic_parity", "equal_opportunity"]


@dataclass(slots=True)
class GroupThresholdSuggestion:
    """Per-group decision thresholds suggested for post-hoc equalization."""

    target: ThresholdTarget
    positive_label: Any
    global_threshold: float
    thresholds_by_group: dict[str, float]
    achieved_selection_rate_by_group: dict[str, float]
    achieved_tpr_by_group: dict[str, float]
    n_rows: int
    support_by_group: dict[str, int]
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization."""
        return asdict(self)


@dataclass(slots=True)
class ReweighingSuggestion:
    """Kamiran–Calders-style sample weights for optional train rebalancing.

    Weights are returned for the caller to pass into a future ``fit`` /
    estimator ``sample_weight`` argument. BuildML does **not** auto-apply them.
    """

    positive_label: Any
    weights: np.ndarray
    weight_table: dict[str, dict[str, float]] = field(default_factory=dict)
    n_rows: int = 0
    support_by_group_label: dict[str, int] = field(default_factory=dict)
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization (weights included as a list)."""
        payload = asdict(self)
        payload["weights"] = [float(w) for w in np.asarray(self.weights).tolist()]
        return payload


def suggest_group_thresholds(
    y_true: Any,
    y_score: Any,
    sensitive: Any,
    *,
    positive_label: Any = 1,
    target: ThresholdTarget = "demographic_parity",
    grid_size: int = 101,
) -> GroupThresholdSuggestion:
    """Suggest per-group score thresholds to equalize selection rate or TPR.

    Operates on **scores/probabilities already produced** by a fitted model.
    Callers must apply the returned thresholds themselves (e.g. in a decision
    policy). Retuning thresholds on the same test rows you will report is a
    leakage / honesty risk — prefer validation for selection.

    Parameters
    ----------
    y_true:
        Ground-truth labels (needed for equal-opportunity / support checks).
    y_score:
        Positive-class scores in ``[0, 1]`` (or any comparable ranking).
    sensitive:
        Group keys (already composed if intersectional).
    positive_label:
        Positive class encoding for TPR computation.
    target:
        ``demographic_parity`` equalizes selection rates toward the global
        mean rate; ``equal_opportunity`` equalizes TPR toward the global TPR.
    grid_size:
        Threshold grid resolution over ``[0, 1]``.

    Returns
    -------
    GroupThresholdSuggestion
        Per-group thresholds plus achieved rates and honesty disclosures.

    Raises
    ------
    ValidationError
        On length mismatch, empty inputs, or unknown target.
    """
    if target not in ("demographic_parity", "equal_opportunity"):
        raise ValidationError(
            f"Unknown threshold target {target!r}; "
            "use 'demographic_parity' or 'equal_opportunity'."
        )
    yt = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    sens = np.asarray(sensitive)
    if len(yt) != len(scores) or len(yt) != len(sens):
        raise ValidationError("y_true, y_score, and sensitive must have equal length.")
    if len(yt) == 0:
        raise ValidationError("Threshold suggestion requires at least one row.")
    if grid_size < 5:
        raise ValidationError("grid_size must be >= 5.")
    if not np.all(np.isfinite(scores)):
        raise ValidationError("y_score must be finite.")

    yt_pos = _as_bool_pred(yt, positive_label)
    grid = np.linspace(0.0, 1.0, int(grid_size))
    # Global reference using midpoint default 0.5 for baseline rate targets.
    global_threshold = 0.5
    global_sel = float(np.mean(scores >= global_threshold))
    global_tpr = (
        float(np.mean(scores[yt_pos] >= global_threshold))
        if int(yt_pos.sum())
        else float("nan")
    )
    target_value = global_sel if target == "demographic_parity" else global_tpr
    if target_value != target_value:
        raise ValidationError(
            "Cannot equalize opportunity: no positives in y_true for the "
            "declared positive_label."
        )

    thresholds: dict[str, float] = {}
    achieved_sel: dict[str, float] = {}
    achieved_tpr: dict[str, float] = {}
    support: dict[str, int] = {}
    warnings: list[str] = []

    for group in sorted({str(g) for g in sens}):
        mask = np.asarray([str(g) == group for g in sens])
        n = int(mask.sum())
        support[group] = n
        if n == 0:
            thresholds[group] = global_threshold
            achieved_sel[group] = float("nan")
            achieved_tpr[group] = float("nan")
            continue
        g_scores = scores[mask]
        g_yt = yt_pos[mask]
        best_t = global_threshold
        best_err = float("inf")
        for t in grid:
            pred = g_scores >= t
            if target == "demographic_parity":
                metric = float(pred.mean())
            else:
                pos = int(g_yt.sum())
                if pos == 0:
                    metric = float("nan")
                else:
                    metric = float(pred[g_yt].mean())
            if metric != metric:
                continue
            err = abs(metric - target_value)
            if err < best_err or (err == best_err and t < best_t):
                best_err = err
                best_t = float(t)
        thresholds[group] = best_t
        pred_final = g_scores >= best_t
        achieved_sel[group] = float(pred_final.mean())
        pos_n = int(g_yt.sum())
        achieved_tpr[group] = (
            float(pred_final[g_yt].mean()) if pos_n else float("nan")
        )
        if n < 30:
            warnings.append(
                f"Group {group!r} has support {n} < 30; threshold is unstable."
            )
        if target == "equal_opportunity" and pos_n == 0:
            warnings.append(
                f"Group {group!r} has no positives; equal-opportunity threshold "
                "falls back toward the global reference."
            )

    disclosures = (
        "Opt-in post-hoc threshold equalization: not automatic mitigation and "
        "not legal certification.",
        "Thresholds are suggestions derived from the provided scores; BuildML "
        "does not rewrite Session predictions.",
        "Selecting thresholds on the same partition you report creates "
        "optimistic bias — prefer validation for selection, test for reporting.",
        f"Target={target!r}; reference uses global threshold={global_threshold}.",
    )
    return GroupThresholdSuggestion(
        target=target,
        positive_label=positive_label,
        global_threshold=global_threshold,
        thresholds_by_group=thresholds,
        achieved_selection_rate_by_group=achieved_sel,
        achieved_tpr_by_group=achieved_tpr,
        n_rows=int(len(yt)),
        support_by_group=support,
        disclosures=disclosures,
        warnings=tuple(warnings),
    )


def suggest_reweighing_weights(
    y_true: Any,
    sensitive: Any,
    *,
    positive_label: Any = 1,
) -> ReweighingSuggestion:
    """Suggest Kamiran–Calders reweighing sample weights for train-time use.

    For each (group, label) cell, weight is
    ``P(group) * P(label) / P(group, label)`` (with Laplace-smoothed empty
    cells refused via ValidationError when a cell is empty).

    Parameters
    ----------
    y_true, sensitive:
        Aligned label and group arrays (typically **train** rows).
    positive_label:
        Used only for disclosure / table labeling of the positive class;
        weights are computed for every observed label value.

    Returns
    -------
    ReweighingSuggestion
        Per-row weights plus the ``(group, label)`` weight table.

    Raises
    ------
    ValidationError
        On length mismatch, empty inputs, or empty (group, label) cells.
    """
    yt = np.asarray(y_true)
    sens = np.asarray(sensitive)
    if len(yt) != len(sens):
        raise ValidationError("y_true and sensitive must have equal length.")
    n = len(yt)
    if n == 0:
        raise ValidationError("Reweighing requires at least one row.")

    groups = sorted({str(g) for g in sens})
    labels = sorted({_label_token(v) for v in yt}, key=str)
    # Joint and marginal counts
    joint: dict[tuple[str, str], int] = {}
    group_counts: dict[str, int] = {g: 0 for g in groups}
    label_counts: dict[str, int] = {lab: 0 for lab in labels}
    for g_raw, y_raw in zip(sens, yt, strict=True):
        g = str(g_raw)
        lab = _label_token(y_raw)
        joint[(g, lab)] = joint.get((g, lab), 0) + 1
        group_counts[g] = group_counts.get(g, 0) + 1
        label_counts[lab] = label_counts.get(lab, 0) + 1

    empty_cells = [
        (g, lab)
        for g in groups
        for lab in labels
        if joint.get((g, lab), 0) == 0
    ]
    warnings: list[str] = []
    if empty_cells:
        # Allow missing cells by skipping them in the table; rows with those
        # combinations cannot exist. Warn if a group lacks the positive label.
        pos_tok = _label_token(positive_label)
        for g, lab in empty_cells:
            if lab == pos_tok:
                warnings.append(
                    f"Group {g!r} has zero rows with label {lab!r}; "
                    "reweighing cannot invent missing (group, label) mass."
                )

    weight_table: dict[str, dict[str, float]] = {g: {} for g in groups}
    for g in groups:
        for lab in labels:
            n_gl = joint.get((g, lab), 0)
            if n_gl == 0:
                continue
            # W = P(S) P(Y) / P(S,Y)
            w = (group_counts[g] / n) * (label_counts[lab] / n) / (n_gl / n)
            weight_table[g][lab] = float(w)

    weights = np.empty(n, dtype=float)
    support: dict[str, int] = {}
    for i, (g_raw, y_raw) in enumerate(zip(sens, yt, strict=True)):
        g = str(g_raw)
        lab = _label_token(y_raw)
        key = f"{g}::{lab}"
        support[key] = support.get(key, 0) + 1
        weights[i] = weight_table[g][lab]

    # Normalize mean weight to 1 for estimator friendliness.
    mean_w = float(weights.mean())
    if mean_w <= 0:
        raise ValidationError("Computed reweighing weights have non-positive mean.")
    weights = weights / mean_w

    disclosures = (
        "Opt-in Kamiran–Calders-style reweighing: returns sample weights only; "
        "BuildML does not auto-fit with them.",
        "Reweighing is a post-hoc statistical adjustment, not proof of fairness "
        "or legal compliance.",
        "Apply weights on train (or a declared training protocol); never pretend "
        "holdout metrics were produced without the chosen weighting policy.",
        f"positive_label={positive_label!r} is recorded for disclosure; "
        "weights cover all observed labels.",
    )
    return ReweighingSuggestion(
        positive_label=positive_label,
        weights=weights,
        weight_table=weight_table,
        n_rows=n,
        support_by_group_label=support,
        disclosures=disclosures,
        warnings=tuple(warnings),
    )


def apply_group_thresholds(
    y_score: Any,
    sensitive: Any,
    thresholds_by_group: dict[str, float],
    *,
    positive_label: Any = 1,
    negative_label: Any = 0,
) -> np.ndarray:
    """Apply per-group thresholds to scores → hard labels (helper, opt-in).

    This is an explicit transform for experiments; it does not touch Session
    state.
    """
    scores = np.asarray(y_score, dtype=float)
    sens = np.asarray(sensitive)
    if len(scores) != len(sens):
        raise ValidationError("y_score and sensitive must have equal length.")
    out = np.empty(len(scores), dtype=object)
    default_t = 0.5
    for i, (score, g_raw) in enumerate(zip(scores, sens, strict=True)):
        g = str(g_raw)
        t = float(thresholds_by_group.get(g, default_t))
        out[i] = positive_label if score >= t else negative_label
    return out


def selection_rates_from_scores(
    y_score: Any,
    sensitive: Any,
    thresholds_by_group: dict[str, float],
    *,
    positive_label: Any = 1,
    negative_label: Any = 0,
) -> dict[str, float]:
    """Convenience: selection rates after applying group thresholds."""
    y_pred = apply_group_thresholds(
        y_score,
        sensitive,
        thresholds_by_group,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    return group_selection_rates(y_pred, np.asarray(sensitive), positive_label=positive_label)


def _label_token(value: Any) -> str:
    return str(value)
