"""The stored cases, and the distance functions that decide which ones are near.

Case-based reasoning keeps its training rows rather than summarising them, and
this module holds both halves of that: the objects that store a case and the
arithmetic that measures how far apart two cases are.

Distance is where the method succeeds or fails, and it is worth being concrete
about why. A trained model learns which features matter; a case-based reasoner
is told, implicitly, through the feature set and the scaling. Include a column
that has nothing to do with the outcome and it still contributes to every
distance, pushing genuinely similar cases apart. Leave a large-scale column
unscaled and it drowns out the rest.

Four metrics are available. Euclidean is straight-line distance and the usual
choice for continuous features on a comparable scale. Manhattan sums absolute
differences, which lets one very different dimension count less. Cosine compares
direction rather than position, useful when the magnitude of a vector is an
artefact. Mixed is a Gower-style combination for data with both numeric and
categorical columns, since neither of the others has any sensible notion of the
distance between "red" and "blue".

Every transform used in distance — the standardisation, the numeric ranges, the
categorical vocabularies — is fitted on training cases and reused unchanged.
Refitting on holdout rows would let the evaluation set influence what "similar"
means, and the resulting score would flatter the model.

See Also
--------
buildml.cbr.types.CbrConfig : Choosing a metric and feature set.
buildml.cbr.retrieve.retrieve_cases : Finding neighbours with these functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from buildml.core.errors import ValidationError


@dataclass(slots=True)
class Case:
    """One remembered episode — what the situation was and how it turned out.

    The atom of case memory. Unlike a training row consumed and discarded during
    fitting, a case persists and can be shown to a user as the reason for a
    prediction: *this is the past case your query resembled, and this is what
    happened to it.*

    Attributes
    ----------
    case_id:
        Stable identifier, used to trace a prediction back to its evidence.
    row_index:
        The originating index in the source frame, so a case can be joined back
        to columns the case base does not carry.
    solution:
        What happened: the label for classification, the value for regression.
    numeric_features:
        Numeric feature values, in the case base's column order.
    categorical_features:
        Categorical values, in the case base's column order.
    source:
        ``'train'`` for a case from the original fit, ``'retained'`` for one
        added later. Worth checking — a case base that has grown mostly through
        retention no longer reflects the data it was evaluated on.
    disclosures:
        Facts about how this case was constructed or admitted.

    See Also
    --------
    CaseBase : The collection these live in.
    CaseTrace : The record of which cases drove a prediction.
    """

    case_id: str
    row_index: Any
    solution: Any
    numeric_features: tuple[float, ...] = ()
    categorical_features: tuple[Any, ...] = ()
    source: str = "train"  # train | retained
    disclosures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the case as a JSON-safe mapping.

        For logging a case alongside a prediction, or writing case memory out
        for inspection.

        Returns
        -------
        dict
            Every field, with feature tuples as lists.
        """
        return {
            "case_id": self.case_id,
            "row_index": self.row_index,
            "solution": self.solution,
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "source": self.source,
            "disclosures": list(self.disclosures),
        }


@dataclass(slots=True)
class CaseTrace:
    """The full account of why one query got the prediction it did.

    This is what case-based reasoning offers that most methods cannot: not an
    importance score or a coefficient, but the actual past cases behind the
    answer, how far each was, how much each counted, and what each of them
    resolved to. A user can read the neighbours and judge for themselves whether
    the analogy holds.

    Attributes
    ----------
    query_index:
        Which row this explains.
    neighbor_case_ids:
        The retrieved cases, nearest first.
    neighbor_row_indices:
        Their original row indices, for joining back to source data.
    distances:
        How far each neighbour was, in the configured metric. Scale is
        metric-dependent and only comparable within one query.
    weights:
        How much each neighbour counted. Equal under majority voting, inverse to
        distance under distance weighting.
    neighbor_solutions:
        What each neighbour resolved to. Disagreement here is the honest signal
        that the prediction is uncertain.
    prediction:
        The combined answer.
    reuse_mode:
        How the neighbours were combined.
    adapt_mode:
        Any post-reuse adjustment applied.
    notes:
        Anything unusual: ties broken, neighbours dropped, adaptation skipped.

    Notes
    -----
    **Read the distances before trusting the prediction.** Neighbours are
    returned whether or not they are close, so a query unlike anything in memory
    still gets ``k`` of them and a confident-looking answer. Large distances
    relative to typical ones mean extrapolation.

    **Disagreeing neighbours are informative, not a defect.** Three of five
    saying one thing is a genuinely uncertain prediction, and the trace shows it
    where a bare label would not.

    See Also
    --------
    buildml.cbr.results.CbrPredictResult : Where traces are returned.
    """

    query_index: Any
    neighbor_case_ids: tuple[str, ...]
    neighbor_row_indices: tuple[Any, ...]
    distances: tuple[float, ...]
    weights: tuple[float, ...]
    neighbor_solutions: tuple[Any, ...]
    prediction: Any
    reuse_mode: str
    adapt_mode: str = "none"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the explanation as a JSON-safe mapping.

        The form to log or serve when a prediction has to be justified to
        someone who will not be reading Python objects.

        Returns
        -------
        dict
            Every field, with neighbour tuples as lists in nearest-first order.
        """
        return {
            "query_index": self.query_index,
            "neighbor_case_ids": list(self.neighbor_case_ids),
            "neighbor_row_indices": list(self.neighbor_row_indices),
            "distances": list(self.distances),
            "weights": list(self.weights),
            "neighbor_solutions": list(self.neighbor_solutions),
            "prediction": self.prediction,
            "reuse_mode": self.reuse_mode,
            "adapt_mode": self.adapt_mode,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class CaseBase:
    """The remembered cases plus everything needed to measure distance to them.

    Both the cases and the fitted machinery that makes them searchable: the
    feature matrices, the standardisation and range statistics, the categorical
    vocabularies, and whatever index the chosen backend built.

    Every one of those transforms is fitted once, on training cases, and reused
    unchanged. That is the discipline that keeps a holdout score meaningful —
    refitting the scaler when new rows arrive would let those rows influence
    what "similar" means, and the evaluation would be measuring itself.

    This is tabular case memory for supervised problems. It is not a text
    corpus, not a vector database, and not a general cognitive CBR framework;
    for retrieval over documents see :mod:`buildml.rag`.

    Attributes
    ----------
    cases:
        The remembered episodes, aligned row-for-row with the matrices.
    numeric_matrix:
        Numeric features, one row per case.
    categorical_matrix:
        Integer-coded categorical features.
    numeric_columns, categorical_columns:
        Column names, defining the order every query must match.
    metric:
        The distance function these artefacts were prepared for.
    numeric_mean_, numeric_scale_:
        Train-fitted standardisation, or ``None`` when disabled.
    numeric_ranges_:
        Train-fitted per-feature ranges for the mixed metric.
    cat_vocabularies_:
        Train-observed categories per column. A value absent from training codes
        to ``-1``, which the mixed metric treats as different from everything.
    search_matrix_:
        The matrix actually searched, which may be reduced or embedded rather
        than the raw numeric features.
    ann_index_, ann_library_:
        The approximate index and the library that built it, when using the
        industry backend.
    text_embedder_id_:
        The sentence-transformer behind text features, when used.
    torch_encoder_:
        The learned metric encoder, when using the torch backend.
    disclosures:
        Statements about how memory was built, including backend fallbacks.
    n_retained:
        How many cases were added after the initial fit. Compare against
        ``n_cases``: a memory that is mostly retained no longer resembles the
        one that was evaluated.

    See Also
    --------
    Case : One entry.
    buildml.cbr.retain.retain_cbr : Adding cases after the fit.
    """

    cases: tuple[Case, ...]
    numeric_matrix: np.ndarray = field(repr=False)
    categorical_matrix: np.ndarray = field(repr=False)  # object/int codes
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    metric: str
    # Train-fit distance transforms (never refit on holdout).
    numeric_mean_: np.ndarray | None = field(repr=False, default=None)
    numeric_scale_: np.ndarray | None = field(repr=False, default=None)
    numeric_ranges_: np.ndarray | None = field(repr=False, default=None)
    cat_vocabularies_: tuple[tuple[Any, ...], ...] = ()
    # Industry / embedding / torch retrieval artifacts (train-fit; never refit).
    search_matrix_: np.ndarray | None = field(repr=False, default=None)
    ann_index_: Any = field(repr=False, default=None)
    ann_library_: str | None = None
    text_embedder_id_: str | None = None
    torch_encoder_: Any = field(repr=False, default=None)
    disclosures: tuple[str, ...] = ()
    n_retained: int = 0

    @property
    def n_cases(self) -> int:
        """How many cases are remembered, train and retained together."""
        return len(self.cases)

    def to_dict(self) -> dict[str, Any]:
        """Return a summary of the memory, with only the first few cases.

        Deliberately partial. The full memory can be very large and the
        matrices do not serialise usefully, so this carries the shape — counts,
        metric, columns — plus a five-case preview for a sanity check.

        Returns
        -------
        dict
            ``n_cases``, ``n_retained``, ``metric``, the column lists,
            ``disclosures``, and ``cases_preview``.

        Notes
        -----
        **This is not a saved case base.** The matrices, fitted statistics, and
        any index are absent, so nothing here can be queried. Use
        :func:`~buildml.cbr.checkpoint.save_cbr_bundle` to persist a working
        memory.
        """
        return {
            "n_cases": self.n_cases,
            "n_retained": self.n_retained,
            "metric": self.metric,
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "disclosures": list(self.disclosures),
            "cases_preview": [c.to_dict() for c in self.cases[:5]],
        }


def pairwise_distances(
    query: np.ndarray,
    memory: np.ndarray,
    *,
    metric: str,
    query_cat: np.ndarray | None = None,
    memory_cat: np.ndarray | None = None,
    numeric_ranges: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Measure how far each query is from every case in memory.

    The operation the whole method rests on. Four metrics are available and they
    encode genuinely different ideas of similarity, so the choice is a modelling
    decision rather than a detail:

    ``euclidean``
        Straight-line distance. The default, and right when features are
        continuous and on comparable scales.
    ``manhattan``
        Sum of absolute differences. One wildly different dimension moves the
        distance less than it would under Euclidean, which is useful when
        features are somewhat independent and outliers are expected.
    ``cosine``
        Angle rather than position, so two cases with the same profile at
        different magnitudes count as identical. Use when magnitude is an
        artefact of scale rather than a signal.
    ``mixed``
        Gower-style: range-normalised numeric differences and a categorical
        mismatch rate, averaged in proportion to how many columns are of each
        kind. The only option when categorical features must contribute.

    Parameters
    ----------
    query:
        Numeric features, ``(n_query, n_num)`` or ``(n_num,)``.
    memory:
        Case numeric features, ``(n_cases, n_num)``.
    metric:
        ``'euclidean'``, ``'manhattan'``, ``'cosine'``, or ``'mixed'``.
    query_cat:
        Integer-coded query categoricals. Required for ``'mixed'``, ignored
        otherwise.
    memory_cat:
        Integer-coded case categoricals, using the same codes as ``query_cat``.
        Required for ``'mixed'``, ignored otherwise.
    numeric_ranges:
        Train-fitted per-feature ranges for the mixed metric's numeric term.
        Defaults to ones, which leaves numeric differences unnormalised and lets
        a wide-ranging column dominate.
    eps:
        Floor for norms and ranges, preventing division by zero on a
        zero-magnitude vector or a constant feature.

    Returns
    -------
    numpy.ndarray
        Distances, shape ``(n_query, n_cases)``. Smaller is more similar.

    Raises
    ------
    ValidationError
        If query and memory widths differ, the metric is unrecognised, or
        ``'mixed'`` was requested without categorical codes.

    Notes
    -----
    **Memory grows as ``n_query × n_cases × n_features``** for the non-cosine
    metrics, which build a full difference array. Large batches against large
    memories should be chunked.

    **Distances are only comparable within one metric and one scaling.** A
    Euclidean 2.5 and a mixed 0.4 say nothing about each other, and neither
    survives a change to the feature set.

    **The mixed metric weights by column count, not importance.** Nine numeric
    columns and one categorical means the categorical contributes a tenth,
    however decisive it is.

    Examples
    --------
    One query against three cases:

    >>> import numpy as np
    >>> memory = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    >>> pairwise_distances(np.array([0.0, 0.0]), memory, metric="euclidean")
    array([[0., 5., 1.]])

    Manhattan makes the diagonal case closer than Euclidean does not:

    >>> pairwise_distances(np.array([0.0, 0.0]), memory, metric="manhattan")
    array([[0., 7., 1.]])

    Cosine ignores magnitude, so a scaled-up case is identical:

    >>> ray = np.array([[1.0, 1.0], [10.0, 10.0]])
    >>> np.round(pairwise_distances(np.array([2.0, 2.0]), ray, metric="cosine"), 6)
    array([[0., 0.]])

    See Also
    --------
    top_k_indices : Selecting the nearest.
    distance_weights : Turning distances into influence.
    """
    q = np.atleast_2d(np.asarray(query, dtype=float))
    m = np.asarray(memory, dtype=float)
    if q.shape[1] != m.shape[1]:
        raise ValidationError(
            f"Query numeric width {q.shape[1]} != case memory width {m.shape[1]}."
        )
    key = str(metric).lower().replace("-", "_")
    if key == "euclidean":
        # (n_q, n_m)
        diff = q[:, None, :] - m[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))
    if key == "manhattan":
        diff = q[:, None, :] - m[None, :, :]
        return np.sum(np.abs(diff), axis=-1)
    if key == "cosine":
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        mn = np.linalg.norm(m, axis=1, keepdims=True)
        qn = np.maximum(qn, eps)
        mn = np.maximum(mn, eps)
        sim = (q / qn) @ (m / mn).T
        return 1.0 - sim
    if key == "mixed":
        if query_cat is None or memory_cat is None:
            raise ValidationError(
                "metric='mixed' requires categorical codes for query and memory."
            )
        qc = np.atleast_2d(np.asarray(query_cat))
        mc = np.asarray(memory_cat)
        n_num = m.shape[1]
        n_cat = mc.shape[1] if mc.ndim == 2 else 0
        n_parts = n_num + n_cat
        if n_parts == 0:
            raise ValidationError("mixed metric needs at least one feature.")
        # Numeric: range-normalized absolute difference.
        if n_num > 0:
            ranges = (
                np.asarray(numeric_ranges, dtype=float)
                if numeric_ranges is not None
                else np.ones(n_num, dtype=float)
            )
            ranges = np.maximum(ranges, eps)
            diff = np.abs(q[:, None, :] - m[None, :, :]) / ranges[None, None, :]
            num_term = np.mean(np.clip(diff, 0.0, 1.0), axis=-1)
        else:
            num_term = np.zeros((q.shape[0], m.shape[0]), dtype=float)
        if n_cat > 0:
            # Mismatch rate over categorical columns.
            mism = (qc[:, None, :] != mc[None, :, :]).astype(float)
            cat_term = np.mean(mism, axis=-1)
        else:
            cat_term = np.zeros_like(num_term)
        # Weighted average by feature count (Gower).
        w_num = n_num / n_parts
        w_cat = n_cat / n_parts
        return w_num * num_term + w_cat * cat_term
    raise ValidationError(
        f"Unknown CBR metric {metric!r}; expected euclidean, manhattan, "
        "cosine, or mixed."
    )


def top_k_indices(distances: np.ndarray, k: int) -> np.ndarray:
    """Pick the ``k`` nearest cases, breaking ties the same way every time.

    Uses a partial selection to find the ``k`` smallest without sorting the
    whole array, then sorts just that shortlist. At realistic memory sizes the
    difference is substantial, and the final stable sort means equal distances
    always come back in memory order rather than in whatever order the
    partition happened to produce.

    Parameters
    ----------
    distances:
        A one-dimensional distance vector for a single query.
    k:
        How many neighbours to take. Clamped to the number of cases, so asking
        for more than exist returns all of them rather than raising.

    Returns
    -------
    numpy.ndarray
        Indices into the case base, nearest first.

    Raises
    ------
    ValidationError
        If ``distances`` is not one-dimensional, memory is empty, or ``k`` is
        less than one.

    Notes
    -----
    **Clamping ``k`` is deliberate.** A small case base should still answer,
    with fewer neighbours, rather than failing — but note that ``k`` neighbours
    out of ``k`` total cases is the whole memory, and the prediction is then a
    global average rather than a local one.

    **Ties break by index, which is stable but arbitrary.** Duplicate cases at
    identical distance are selected in memory order.

    Examples
    --------
    >>> import numpy as np
    >>> top_k_indices(np.array([0.5, 0.1, 0.9, 0.3]), 2).tolist()
    [1, 3]

    Tied distances come back in memory order:

    >>> top_k_indices(np.array([0.2, 0.2, 0.2]), 3).tolist()
    [0, 1, 2]

    Asking for more neighbours than exist returns all of them:

    >>> len(top_k_indices(np.array([0.4, 0.1]), 10))
    2
    """
    if distances.ndim != 1:
        raise ValidationError("top_k_indices expects a 1-d distance vector.")
    n = int(distances.shape[0])
    if n == 0:
        raise ValidationError("Case memory is empty; cannot retrieve neighbors.")
    kk = min(int(k), n)
    if kk < 1:
        raise ValidationError("k must be >= 1.")
    # argpartition then sort the shortlist for stable ordering.
    part = np.argpartition(distances, kk - 1)[:kk]
    order = part[np.argsort(distances[part], kind="stable")]
    return order


def distance_weights(
    distances: Sequence[float] | np.ndarray, *, eps: float = 1e-8
) -> np.ndarray:
    """Turn distances into influence, so nearer cases count for more.

    Weight is ``1 / (distance + eps)``. The reasoning is that a case at distance
    0.1 is a much better analogy than one at 1.0 and should not get an equal
    vote, which is exactly what unweighted majority voting gives it.

    Parameters
    ----------
    distances:
        Neighbour distances.
    eps:
        Floor added before inverting. Its real job is bounding the weight of an
        exact match: without it, a distance of zero would produce infinity and
        that single case would decide the prediction outright.

    Returns
    -------
    numpy.ndarray
        Weights, same shape as the input. Larger means more influence.

    Notes
    -----
    **Weights are unnormalised.** Callers that need proportions must divide by
    the sum.

    **The falloff is sharper than it looks.** Inverse distance is quite
    aggressive: a neighbour twice as far counts half as much, so with one very
    close case the rest barely participate. That is usually what you want and
    occasionally is not — ``'majority'`` gives every neighbour an equal say.

    Examples
    --------
    A neighbour twice as far carries half the weight:

    >>> distance_weights([0.5, 1.0, 2.0]).round(4).tolist()
    [2.0, 1.0, 0.5]

    An exact match is bounded rather than infinite:

    >>> float(distance_weights([0.0], eps=1e-8)[0])
    100000000.0
    """
    d = np.asarray(distances, dtype=float)
    return 1.0 / (d + float(eps))


def encode_categoricals(
    values: Sequence[Any] | np.ndarray,
    vocabulary: Sequence[Any],
) -> np.ndarray:
    """Convert category labels to the integer codes distance arithmetic needs.

    Codes are positions in the train-fitted vocabulary, so the same label always
    maps to the same code and query encoding matches memory encoding.

    Parameters
    ----------
    values:
        The category labels to encode.
    vocabulary:
        Categories observed during fitting, in order. Position determines code.

    Returns
    -------
    numpy.ndarray
        Integer codes. Unseen values map to ``-1``.

    Notes
    -----
    **Codes are labels, not quantities.** The mixed metric only ever asks
    whether two codes are equal; nothing subtracts them. An ordinal column
    encoded this way loses its ordering, and a numeric encoding would be needed
    to keep it.

    **``-1`` means unseen, and it is not a category.** Under the mixed metric it
    differs from every real code — including, deliberately, from another unseen
    value, since two labels being absent from training is no evidence they are
    alike.

    **Comparison is by string.** An integer ``1`` and the string ``"1"`` encode
    to the same code, which is usually what mixed-dtype data wants.

    Examples
    --------
    >>> vocab = ["north", "south", "east"]
    >>> encode_categoricals(["south", "north", "west"], vocab).tolist()
    [1, 0, -1]

    Integers and their string forms collapse to one code:

    >>> encode_categoricals([1, "1"], [1, 2]).tolist()
    [0, 0]
    """
    vocab = {str(v): i for i, v in enumerate(vocabulary)}
    out = np.empty(len(values), dtype=int)
    for i, v in enumerate(values):
        out[i] = vocab.get(str(v), -1)
    return out
