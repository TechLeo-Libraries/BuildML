"""The settings that decide how a case-based reasoner remembers and reuses.

Case-based reasoning answers a new question by finding the most similar past
cases and adapting what happened to them. There is no model in the usual sense —
nothing is compressed into coefficients or split points. The training data *is*
the model, which is what makes every prediction explainable by pointing at the
specific past cases behind it, and what makes the notion of similarity carry all
the weight.

Four decisions are recorded here, and they matter roughly in this order:

*Which columns count.* Every included feature contributes to distance, so an
irrelevant column is not merely useless — it actively pulls neighbours apart.

*How similarity is measured.* ``metric`` chooses between Euclidean, Manhattan,
cosine, and a Gower-style mixed distance for data with categorical columns.

*How neighbours become an answer.* ``reuse`` chooses between voting, distance
weighting, averaging, and fitting a small local model.

*How many neighbours.* ``k`` trades variance for bias: one neighbour is noisy,
fifty averages away the local structure the method exists to exploit.

Scaling deserves its own note because it is the most common way this goes
quietly wrong. Distance is dominated by whichever feature has the largest
numeric spread, so an unscaled salary column makes an age column irrelevant.
``standardize`` is on by default and is fitted on training rows alone.

See Also
--------
buildml.cbr.fit.fit_cbr : Building the case base.
buildml.cbr.predict.predict_cbr : Predicting, with the neighbours attached.
buildml.cbr.cases : Case memory and the distance functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CbrTask = Literal["classification", "regression"]

# Retrieval backend for case memory (honest defaults when extras installed).
CbrBackend = Literal["sklearn", "industry", "embedding", "torch"]

# Distance / similarity metrics over the case-memory feature space.
# - euclidean / manhattan / cosine: numeric features only (after optional train scaling)
# - mixed: Gower-style mix of range-normalized numeric |Δ| and categorical mismatch
CbrMetric = Literal["euclidean", "manhattan", "cosine", "mixed"]

# How neighbor solutions become a prediction.
CbrReuseMode = Literal[
    "majority",  # classification: unweighted majority vote
    "distance_weighted",  # classification vote / regression average with 1/(d+ε)
    "local_mean",  # regression: unweighted mean of neighbor solutions
    "local_ridge",  # regression: tiny Ridge on the k neighbors' features→solution
]

# Optional post-reuse adaptation (lite).
CbrAdaptMode = Literal["none", "offset"]


@dataclass(slots=True)
class CbrConfig:
    """Everything that determines how the case base is built and queried.

    One object carrying the whole configuration, recorded on results so a
    prediction can always be traced back to the settings that produced it.

    Attributes
    ----------
    task:
        ``'classification'`` or ``'regression'``. Decides which reuse modes are
        legal and how neighbour solutions are combined.
    backend:
        How neighbours are found. ``'sklearn'`` is exact brute-force search and
        always available; ``'industry'`` uses an approximate index for large
        case bases; ``'embedding'`` and ``'torch'`` learn a representation
        first. Exact search is the honest default — approximate search trades
        recall for speed, and at small scale there is nothing to buy.
    metric:
        ``'euclidean'`` for continuous features on a comparable scale,
        ``'manhattan'`` when you want outlying dimensions to count less,
        ``'cosine'`` when direction matters and magnitude does not, ``'mixed'``
        when categorical columns are present.
    reuse:
        How neighbour solutions become one answer. ``'majority'`` and
        ``'distance_weighted'`` for classification; ``'local_mean'``,
        ``'distance_weighted'``, and ``'local_ridge'`` for regression.
    adapt:
        Post-reuse correction. ``'none'`` returns the combined neighbour
        solution; ``'offset'`` applies a small local adjustment.
    k:
        Neighbours consulted per query. Small values track local structure and
        are noisy; large values are stable and blur it.
    columns:
        Feature columns, or ``None`` to infer. Every one included affects
        distance.
    categorical_columns:
        Columns treated as categorical, or ``None`` to infer from dtype.
    text_columns:
        Columns embedded as text rather than treated as categories.
    text_model_name:
        The sentence-transformer used for ``text_columns``.
    standardize:
        Centre and scale numeric features on training rows before measuring
        distance. Leave this on unless the raw scales are already comparable.
    distance_eps:
        Floor added before inverting a distance, so an exact match yields a
        large finite weight rather than dividing by zero.
    random_state:
        Seed for the components that sample, keeping runs reproducible.
    prefer_reduce_components:
        Use dimensionality-reduced components when a reduce plan is present.
        Distance degrades in high dimensions — everything becomes roughly
        equidistant — so reducing first often improves neighbours.
    disclosures:
        Statements about how the configuration was resolved, including any
        fallback from a requested backend.

    Notes
    -----
    **Unscaled features make one column decide everything.** A feature ranging
    over hundreds of thousands dominates one ranging over tens, whatever their
    relative importance. This is why ``standardize`` defaults to true.

    **``k`` interacts with class balance.** With a rare class and a large ``k``,
    majority voting can never predict it — the rare class is outnumbered in
    every neighbourhood. Distance weighting helps; matching ``k`` to the rarity
    helps more.

    See Also
    --------
    buildml.cbr.catalog.cbr_capability_matrix : Which backends are installed.
    """

    task: CbrTask = "classification"
    backend: CbrBackend = "sklearn"
    metric: CbrMetric = "euclidean"
    reuse: CbrReuseMode = "distance_weighted"
    adapt: CbrAdaptMode = "none"
    k: int = 5
    columns: tuple[str, ...] | None = None
    categorical_columns: tuple[str, ...] | None = None
    text_columns: tuple[str, ...] | None = None
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    standardize: bool = True
    distance_eps: float = 1e-8
    random_state: int | None = 0
    prefer_reduce_components: bool = True
    disclosures: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-safe mapping.

        Tuples become lists and nothing is dropped, so the mapping is a complete
        record of the settings — enough to reconstruct the configuration or to
        compare two runs field by field.

        Returns
        -------
        dict
            Every field, with sequences as lists and ``None`` preserved where a
            field was left to be inferred.
        """
        return {
            "task": self.task,
            "backend": self.backend,
            "metric": self.metric,
            "reuse": self.reuse,
            "adapt": self.adapt,
            "k": self.k,
            "columns": None if self.columns is None else list(self.columns),
            "categorical_columns": (
                None
                if self.categorical_columns is None
                else list(self.categorical_columns)
            ),
            "text_columns": (
                None if self.text_columns is None else list(self.text_columns)
            ),
            "text_model_name": self.text_model_name,
            "standardize": self.standardize,
            "distance_eps": self.distance_eps,
            "random_state": self.random_state,
            "prefer_reduce_components": self.prefer_reduce_components,
            "disclosures": list(self.disclosures),
        }
