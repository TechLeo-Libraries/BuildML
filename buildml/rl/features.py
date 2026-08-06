"""Turn dataset columns into the arrays a policy can learn from.

Every function in this package eventually needs the same three things: a numeric
matrix of state or context features, integer codes for discrete actions, and
metrics comparing chosen actions against known ones. Doing that consistently is
what this module is for: if column resolution differed between
:mod:`~buildml.rl.imitation` and :mod:`~buildml.rl.fit`, a plan fitted by one
could not be scored by the other.

Two conventions run throughout. **Nulls are refused rather than filled.** A
missing state feature has no neutral value, and a missing action is not an
action; silently substituting zero would produce a policy trained on situations
that never occurred. Impute deliberately in :mod:`buildml.preprocess` if you
need to.

**Actions round-trip through string codes.** Encoding stringifies before
assigning integers, and decoding attempts to restore the original type. This
keeps mixed-type action columns workable, at the cost that an action of ``1``
and an action of ``"1"`` become the same arm.

See Also
--------
buildml.semisupervised.features : Where the column resolution logic lives.
buildml.rl.fit : The main consumer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.semisupervised.features import (
    matrix_from_frame as _matrix_from_frame,
)
from buildml.semisupervised.features import (
    resolve_semisupervised_columns,
)

__all__ = [
    "matrix_from_frame",
    "resolve_rl_columns",
    "infer_imitation_task",
    "encode_discrete_actions",
    "decode_discrete_actions",
    "continuous_actions",
    "classification_metrics",
    "regression_metrics",
    "softmax",
]


def matrix_from_frame(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Stack the named columns into a float matrix a policy can read.

    Policies work on arrays, not frames. This is the single conversion point,
    so that column order and null handling are identical everywhere in the
    package.

    Parameters
    ----------
    frame:
        The rows to convert.
    columns:
        Which columns to take, in order. That order becomes the column order of
        the matrix and must match at scoring time.

    Returns
    -------
    numpy.ndarray
        A ``(n_rows, n_columns)`` float array.

    Raises
    ------
    ValidationError
        If a column is absent, non-numeric, or contains nulls. Nulls are
        refused rather than filled: a state feature has no neutral value, and
        substituting one would train the policy on situations that never
        occurred.
    """
    try:
        return _matrix_from_frame(frame, columns)
    except ValidationError as exc:
        msg = str(exc).replace("Semi-supervised learning", "Imitation / RL")
        raise ValidationError(msg) from exc


def resolve_rl_columns(
    dataset: Dataset,
    frame: pd.DataFrame,
    columns: list[str] | None,
    *,
    reduce_plan: Any | None = None,
    prefer_reduce_components: bool = True,
    target_column: str,
    exclude_columns: Sequence[str] = (),
) -> tuple[list[str], bool, list[str]]:
    """Decide which columns describe the situation the policy acts on.

    State and context features are everything *except* the things that are not
    part of the situation: the action taken, the reward observed, and the
    target. Getting this wrong is the most direct route to a useless policy: a
    bandit that can see the reward column will learn to read it rather than to
    predict it.

    Parameters
    ----------
    dataset:
        Consulted for roles and any attached dimensionality reduction.
    frame:
        The partition whose columns are being resolved.
    columns:
        An explicit list, or ``None`` to infer from the dataset's usable
        numeric columns.
    reduce_plan:
        An explicit reduction plan, overriding whatever is attached.
    prefer_reduce_components:
        When ``True`` and a reduction is available, its components are used in
        place of the raw columns.
    target_column:
        The Dataset target, excluded from features.
    exclude_columns:
        Additional columns to drop: in practice the action and reward columns.

    Returns
    -------
    list of str
        The resolved feature columns, in matrix order.
    bool
        Whether reduction components were used rather than raw columns.
    list of str
        Disclosures describing how the columns were chosen, for the plan's
        record.

    Raises
    ------
    ValidationError
        If nothing usable remains after the exclusions. This normally means the
        dataset has only the action and reward columns, leaving no context for
        a policy to condition on.
    """
    cols, used_reduce, disclosures = resolve_semisupervised_columns(
        dataset,
        frame,
        columns,
        reduce_plan=reduce_plan,
        prefer_reduce_components=prefer_reduce_components,
        target_column=target_column,
    )
    exclude = {str(c) for c in exclude_columns}
    filtered = [c for c in cols if c not in exclude]
    if not filtered:
        raise ValidationError(
            "No usable feature/context columns remain after excluding "
            f"{sorted(exclude)}."
        )
    out = [
        note.replace("semi-supervised", "imitation / reinforcement learning")
        for note in disclosures
    ]
    if exclude:
        out.append(
            f"Excluded non-state columns from the design matrix: {sorted(exclude)}."
        )
    return filtered, used_reduce, out


def infer_imitation_task(action: pd.Series) -> str:
    """Guess whether actions are choices from a menu or points on a scale.

    Discrete actions ("which of four offers") make cloning a classification
    problem; continuous ones ("what steering angle") make it regression. The
    distinction is not always visible in the dtype, because discrete actions are
    routinely stored as integers.

    Parameters
    ----------
    action:
        The action column.

    Returns
    -------
    str
        ``'classification'`` or ``'regression'``.

    Notes
    -----
    The rule: non-numeric and boolean columns are classification. Integer
    columns with at most 20 distinct values are classification, on the reasoning
    that a genuinely continuous quantity rarely takes so few values. Float
    columns are classification only if they hold at most 8 values, all drawn
    from 0–7: the shape of a category that lost its integer dtype somewhere.
    Everything else is regression.

    **The 20-value threshold is a heuristic and will occasionally be wrong.** A
    discrete action space with 30 arms is read as regression; a small-integer
    count that really is continuous is read as classification. Pass ``task=``
    explicitly to :func:`~buildml.rl.imitation.fit_imitation` when you know.
    """
    if pd.api.types.is_numeric_dtype(action) and not pd.api.types.is_bool_dtype(action):
        nunique = int(action.nunique(dropna=True))
        # Small integer cardinalities → discrete actions (classification BC).
        if pd.api.types.is_integer_dtype(action) and nunique <= 20:
            return "classification"
        if nunique <= 8 and set(np.unique(action.dropna().to_numpy())).issubset(
            {0, 1, 2, 3, 4, 5, 6, 7}
        ):
            return "classification"
        return "regression"
    return "classification"


def encode_discrete_actions(
    y: pd.Series,
    *,
    classes: Sequence[Any] | None = None,
) -> tuple[np.ndarray, Any, tuple[Any, ...]]:
    """Number the distinct actions so a model can predict them.

    Models predict integers, not labels. This assigns each distinct action a
    code and hands back the encoder needed to undo it.

    Parameters
    ----------
    y:
        The action column.
    classes:
        The action vocabulary to encode against. Pass the plan's ``arms_`` when
        encoding a holdout partition, so codes line up with what the policy was
        fitted on. Left ``None``, the vocabulary is learned from ``y``, which is
        only correct at fit time.

    Returns
    -------
    numpy.ndarray
        Integer codes, one per row.
    sklearn.preprocessing.LabelEncoder
        The fitted encoder, stored on the plan for decoding later.
    tuple
        The action vocabulary in code order.

    Raises
    ------
    ValidationError
        If any action is null. A missing action is not an action, and there is
        no defensible substitute.

    Notes
    -----
    **Actions are stringified before encoding.** That is what lets a mixed-type
    action column work, but it also means an action of ``1`` and an action of
    ``"1"`` collapse into one arm.

    **Passing ``classes`` explicitly raises on unseen actions**, which is the
    intended behaviour: a holdout partition containing an action the policy was
    never fitted on cannot be scored against it.

    See Also
    --------
    decode_discrete_actions : The inverse.
    """
    from sklearn.preprocessing import LabelEncoder

    if y.isna().any():
        raise ValidationError(
            "Imitation / bandit discrete actions require non-null train values."
        )
    values = y.astype(str)
    encoder = LabelEncoder()
    if classes is not None:
        encoder.fit([str(c) for c in classes])
        codes = encoder.transform(values)
    else:
        codes = encoder.fit_transform(values)
    return np.asarray(codes, dtype=int), encoder, tuple(encoder.classes_)


def decode_discrete_actions(pred_codes: np.ndarray, label_encoder: Any) -> list[Any]:
    """Turn predicted codes back into recognisable actions.

    A user who supplied actions ``'offer_a'``/``'offer_b'`` should get those
    back, not ``0``/``1``.

    Parameters
    ----------
    pred_codes:
        Integer codes predicted by a policy.
    label_encoder:
        The encoder from :func:`encode_discrete_actions`, carried on the plan.

    Returns
    -------
    list
        The decoded actions.

    Notes
    -----
    **The original type is recovered heuristically, not preserved.** Encoding
    stringified everything, so decoding parses the strings back: digits become
    integers, strings containing a dot become floats where possible, everything
    else stays a string. An action column of integers round-trips to integers,
    which is what almost all callers expect. An exotic type: a Decimal, a
    timestamp: comes back as its string form.

    See Also
    --------
    encode_discrete_actions : The forward direction.
    """
    codes = np.asarray(pred_codes).astype(int)
    decoded = label_encoder.inverse_transform(codes)
    out: list[Any] = []
    for value in decoded:
        text = str(value)
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            out.append(int(text))
        else:
            try:
                out.append(float(text) if "." in text else text)
            except ValueError:
                out.append(text)
    return out


def continuous_actions(y: pd.Series) -> np.ndarray:
    """Read a continuous action column as a float array.

    The regression counterpart to encoding: continuous actions need no
    vocabulary, only a check that they are numeric and complete.

    Parameters
    ----------
    y:
        The action column, for regression cloning.

    Returns
    -------
    numpy.ndarray
        The actions as floats.

    Raises
    ------
    ValidationError
        If the column is non-numeric, or contains nulls. A missing action
        cannot be imputed to a mean without inventing a decision nobody made.

    See Also
    --------
    encode_discrete_actions : The discrete counterpart.
    """
    if y.isna().any():
        raise ValidationError(
            "Imitation regression requires non-null numeric action values."
        )
    if not pd.api.types.is_numeric_dtype(y):
        raise ValidationError(
            "Imitation regression requires a numeric action column."
        )
    return y.to_numpy(dtype=float)


def classification_metrics(
    y_true: Sequence[Any], y_pred: Sequence[Any]
) -> dict[str, float]:
    """Score how often chosen actions match demonstrated ones.

    Discrete imitation is judged as a classification problem, so the metrics
    are the classification ones: but what they measure is agreement with a
    demonstrator, not correctness.

    Parameters
    ----------
    y_true:
        The demonstrated actions.
    y_pred:
        The policy's actions, in the same row order.

    Returns
    -------
    dict
        ``accuracy``, the fraction of rows that match, and ``macro_f1``, the
        unweighted mean of per-action F1.

    Notes
    -----
    **Read ``macro_f1`` when the action distribution is skewed.** Accuracy
    counts rows, so a policy that always picks the dominant action inherits its
    frequency as a score. Macro F1 counts actions equally, so ignoring a rare
    action costs the same as ignoring a common one.

    Both are computed on the string forms of the actions, matching how they were
    encoded.

    See Also
    --------
    regression_metrics : The continuous-action counterpart.
    """
    from sklearn.metrics import accuracy_score, f1_score

    yt = [str(v) for v in y_true]
    yp = [str(v) for v in y_pred]
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Score how close chosen actions are to demonstrated ones.

    Continuous imitation is judged as a regression problem: the question is not
    whether the policy matched the demonstrator exactly, but by how much it
    missed.

    Parameters
    ----------
    y_true:
        The demonstrated actions.
    y_pred:
        The policy's actions, in the same row order.

    Returns
    -------
    dict
        ``rmse``, ``mae``, and ``r2``.

    Notes
    -----
    **MAE and RMSE answer different questions.** MAE is the typical error in the
    action's own units. RMSE squares before averaging, so a few large deviations
    move it much more than many small ones: read it when occasionally acting
    far off is much worse than routinely acting slightly off, which for a policy
    it usually is.

    ``r2`` is the fraction of action variance the policy reproduces. Negative
    values mean it does worse than always predicting the mean action.

    See Also
    --------
    classification_metrics : The discrete-action counterpart.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }


def softmax(logits: np.ndarray, *, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Turn arbitrary scores into probabilities that sum to one.

    Used wherever a policy has to convert action scores into a distribution it
    can sample from.

    Parameters
    ----------
    logits:
        The raw scores. Any real values; no scale is assumed.
    temperature:
        How sharply the largest score dominates. At 1.0 the result is the
        standard softmax. Below 1.0 the distribution concentrates on the best
        action, approaching a greedy choice. Above 1.0 it flattens toward
        uniform, which is more exploratory. Values at or below zero are clamped
        to a tiny positive number rather than dividing by zero.
    axis:
        The axis to normalise over. The default normalises the last axis, so a
        ``(n_rows, n_actions)`` array gives one distribution per row.

    Returns
    -------
    numpy.ndarray
        Non-negative values summing to 1.0 along ``axis``.

    Notes
    -----
    The maximum is subtracted before exponentiating. Without that step,
    moderately large logits overflow to infinity and the result becomes ``NaN``
   : a policy failure that surfaces only on the inputs where scores happen to
    be large.

    Examples
    --------
    >>> import numpy as np
    >>> np.round(softmax(np.array([1.0, 2.0, 3.0])), 3)
    array([0.09 , 0.245, 0.665])
    >>> np.round(softmax(np.array([1.0, 2.0, 3.0]), temperature=0.1), 3)
    array([0., 0., 1.])
    """
    t = max(float(temperature), 1e-8)
    z = np.asarray(logits, dtype=float) / t
    z = z - np.max(z, axis=axis, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=axis, keepdims=True)
