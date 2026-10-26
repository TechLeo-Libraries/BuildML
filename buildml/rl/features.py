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

**Action identities are retained beside string encoding keys.** Discrete
predictions preserve the observed scalar values, including numeric-looking
strings. Distinct typed actions sharing a key are refused before fitting.
Older encoders without identity metadata return their recorded strings exactly;
refit those policies to recover the original types.

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
    label_encoder: Any | None = None,
) -> tuple[np.ndarray, Any, tuple[Any, ...]]:
    """Encode action identities and retain their original scalar values.

    Fitting records the training vocabulary. Evaluation reuses that encoder
    so logged actions cannot silently acquire different meanings or codes.

    Parameters
    ----------
    y:
        Non-null discrete actions in row order.
    classes:
        Optional fixed encoding vocabulary. Learn it from train when omitted.
    label_encoder:
        Saved train encoder when scoring a fitted policy. New encoders also
        check that observed action types match the fitted vocabulary.

    Returns
    -------
    tuple
        Integer action codes, encoder and string keys in code order. Original
        scalar identities are carried by the encoder for prediction and bundles.

    Raises
    ------
    ValidationError
        If actions are missing, ambiguous, unseen or inconsistent with saved
        identities. Numeric-looking string actions are never parsed as numbers.
    """
    from sklearn.preprocessing import LabelEncoder

    observed = _action_vocabulary(y)
    values = [_action_key(value) for value in y]
    encoder = label_encoder
    if encoder is None:
        encoder = LabelEncoder()
        fixed = observed if classes is None else _action_vocabulary(pd.Series(list(classes), dtype=object))
        encoder.fit(list(fixed))
        encoder.buildml_original_classes_ = tuple(fixed[key] for key in encoder.classes_)
    original = getattr(encoder, "buildml_original_classes_", None)
    if original is not None:
        fitted = dict(zip(encoder.classes_, original, strict=True))
        for key, value in observed.items():
            if key in fitted and (type(fitted[key]) is not type(value) or fitted[key] != value):
                raise ValidationError(f"Observed action {value!r} differs from fitted action identity for key {key!r}.")
    try:
        codes = encoder.transform(values)
    except ValueError as exc:
        raise ValidationError(f"Discrete actions contain an unknown action: {exc}") from exc
    return np.asarray(codes, dtype=int), encoder, tuple(encoder.classes_)


def _action_key(value: Any) -> str:
    """Return a stable encoding key after normalizing NumPy scalar wrappers."""
    return str(value.item() if isinstance(value, np.generic) else value)


def _action_vocabulary(y: pd.Series) -> dict[str, Any]:
    """Retain observed scalar actions and refuse nulls or ambiguous keys."""
    if y.isna().any():
        raise ValidationError("Imitation / bandit discrete actions require non-null values.")
    vocabulary: dict[str, Any] = {}
    for value in y:
        value = value.item() if isinstance(value, np.generic) else value
        key = str(value)
        if key in vocabulary:
            previous = vocabulary[key]
            if type(previous) is not type(value) or previous != value:
                raise ValidationError(f"Ambiguous discrete actions share encoding key {key!r}; use distinct action names.")
        vocabulary[key] = value
    return vocabulary


def decode_discrete_actions(pred_codes: np.ndarray, label_encoder: Any) -> list[Any]:
    """Restore recorded scalar action identities from integer policy codes.

    Both behavioral cloning and contextual bandits use this boundary before
    returning actions to callers or estimating rewards from logged decisions.

    Parameters
    ----------
    pred_codes:
        Integer predictions from the fitted policy.
    label_encoder:
        Encoder saved with the policy, including original identities when known.

    Returns
    -------
    list
        Original scalar actions. Legacy encoders without identity metadata
        return their recorded strings exactly; their original types are unknown.
    """
    codes = np.asarray(pred_codes).astype(int)
    decoded = label_encoder.inverse_transform(codes)
    original = getattr(label_encoder, "buildml_original_classes_", None)
    if original is not None:
        return [original[int(code)] for code in codes]
    return list(decoded)


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

    yt = [_action_key(v) for v in y_true]
    yp = [_action_key(v) for v in y_pred]
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
    >>> from buildml.rl.features import softmax
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
