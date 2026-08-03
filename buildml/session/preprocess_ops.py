"""Session-global preprocess and resample orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Mapping, cast

if TYPE_CHECKING:
    from buildml.session.session import Session

import pandas as pd

from buildml.session._imports import (
    ApplyPlansResult,
    CustomTransformSpec,
    Dataset,
    ValidationError,
    apply_outlier_plan,
    drop_columns_transform,
    extract_date_features,
    fit_binning,
    fit_custom_transform,
    fit_encoder,
    fit_feature_selector,
    fit_outlier_plan,
    fit_reducer,
    fit_scaler,
    fit_simple_imputer,
    fit_text_features,
    list_registered_transforms,
    list_resample_strategies,
    register_custom_transform,
    resample_train,
    run_apply_preprocess_plans,
    transform_binning,
    transform_custom,
    transform_encoder,
    transform_feature_selector,
    transform_reducer,
    transform_scaler,
    transform_simple_imputer,
    transform_text_features,
)


def drop_columns(session, columns: list[str] | tuple[str, ...]) -> "Session":
    """Remove columns you do not want the model to see.

    Use this for columns that would leak the answer (a field populated only
    after the outcome is known), free-text notes you are not vectorising,
    or duplicated identifiers. Marking a column ``'ignore'`` with
    :meth:`set_roles` keeps it in the table for later reference; dropping
    it removes it entirely and reclaims the memory.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    columns:
        Names of the columns to remove.

    Returns
    -------
    Session
        ``self``, so this call chains into the next step.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A named column is not present in the dataset.

    Notes
    -----
    Split membership is preserved (row order unchanged). Roles for dropped
    columns are removed.

    Dropping columns does not disturb an existing split, because splits are
    defined over rows. You can therefore drop before or after splitting
    with the same result.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame({"a": [1, 2], "scratch": [9, 9], "y": [0, 1]})
    >>> session = Session.ingest(frame)
    >>> _ = session.drop_columns(["scratch"])
    >>> session.dataset.columns
    ['a', 'y']

    See Also
    --------
    Session.set_roles : Exclude a column without deleting it.
    Session.select_features : Let a scoring rule choose what to keep.
    """
    session._dataset = drop_columns_transform(session.dataset, columns)
    session._record("drop_columns", {"columns": list(columns)})
    return cast("Session", session)
def impute(
    session,
    *,
    columns: list[str] | None = None,
    strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = "median",
    fill_value: Any | None = None,
) -> "Session":
    """Fill in missing values, using only what the training rows reveal.

    Most estimators cannot accept a missing value, so gaps have to be
    filled with a stand-in before fitting. The stand-in is computed from
    the training rows and then applied everywhere: that ordering is the
    whole point. If you filled from all rows, the median would encode a
    little of the test set into every training row, and your score would
    drift upward for no real reason.

    Which stand-in to use depends on the column. The median resists
    outliers, so it is the default and the safe choice for skewed
    quantities like income. The mean suits roughly symmetric measurements.
    The most frequent value is the sensible fallback for categoricals. A
    constant is right when the gap itself is meaningful: "no prior claim"
    is information, not an accident, and filling it with the median would
    erase that.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    columns:
        Which columns to fill. ``None`` selects numeric columns with the
        ``feature`` role and deliberately leaves ``ignore``, ``id``,
        ``target``, ``group``, ``time``, and ``weight`` alone, so
        identifiers and labels are never quietly altered. Name columns
        explicitly to override that protection.
    strategy:
        How to compute the stand-in: ``'median'`` (the default, robust to
        extreme values), ``'mean'``, ``'most_frequent'``, or ``'constant'``
        for a fixed value you supply.
    fill_value:
        The value used when ``strategy='constant'``. Ignored by the other
        strategies.

    Returns
    -------
    Session
        ``self``, so this call chains into the next transform.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split exists yet, a named column is absent, or
        ``strategy='constant'`` was chosen without a ``fill_value``.

    Notes
    -----
    **Leakage:** Requires an existing split. Statistics are learned from
    the train partition only, then applied to all rows.

    Filling a gap invents data. When a column is mostly missing, or when
    missingness is itself predictive, consider adding an indicator column
    or dropping the column instead. :attr:`last_preprocess` reports how
    many values were filled per column so you can judge.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame(
    ...     {"income": [30.0, np.nan, 50.0, 70.0], "y": [0, 1, 0, 1]}
    ... )
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5)
    >>> _ = session.impute(strategy="median")
    >>> session.dataset.frame["income"].isna().sum()
    np.int64(0)

    Treat an absent value as a fact rather than a gap:

    >>> _ = session.impute(
    ...     columns=["prior_claims"], strategy="constant", fill_value=0
    ... )  # doctest: +SKIP

    See Also
    --------
    Session.impute_plan : The fitted statistics, for reuse at score time.
    Session.encode : Run after imputing when categoricals have gaps.
    """
    session.assert_can_fit("train")
    plan = fit_simple_imputer(
        session.dataset,
        session._split_plan,
        columns=columns,
        strategy=strategy,
        fill_value=fill_value,
    )
    session._dataset = transform_simple_imputer(session.dataset, plan)
    session._impute_plan = plan
    session._record("impute", plan.to_dict())
    return cast("Session", session)
def encode(
    session,
    *,
    columns: list[str] | None = None,
    method: Literal['onehot', 'ordinal', 'infrequent', 'target'] = "onehot",
    min_frequency: float | int = 0.05,
    n_folds: int = 5,
    random_state: int = 0,
    smoothing: float = 10.0,
) -> "Session":
    """Turn category labels into numbers a model can work with.

    Estimators do arithmetic, and ``"Ireland"`` is not a number. Encoding
    is how a category becomes something computable: but the choice of
    encoding changes what the model is able to learn, so it is worth
    understanding rather than accepting the default blindly.

    ``'onehot'`` gives each level its own 0/1 column. It makes no claim
    about order or distance between levels, which is the honest
    representation for genuinely unordered categories, and it is the
    default. Its cost is width: a column with a thousand levels becomes a
    thousand columns.

    ``'ordinal'`` maps levels to ``0, 1, 2, …``. Compact, but it asserts
    that level 2 sits between level 1 and level 3. That is right for
    ``small < medium < large`` and wrong for country names: a linear model
    will happily conclude that Ireland is halfway between Iceland and
    Italy. Tree models are largely immune, which is why ordinal encoding is
    often fine with them and dangerous without them.

    ``'infrequent'`` pools every level that is rare in training into a
    single ``other`` bucket before one-hot encoding. This is the practical
    answer to high-cardinality columns: rare levels carry too few examples
    to learn from and generate columns that are almost entirely zero.

    ``'target'`` replaces each level with the average target for that level
   : extremely compact and often the strongest encoder, but the one that
    leaks most eagerly, since the target is being folded into a feature.
    BuildML defends against that on two fronts: training rows receive
    out-of-fold averages (a row never contributes to the mean it is given),
    and rare levels are pulled toward the overall average by ``smoothing``.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    columns:
        Which columns to encode. ``None`` selects categorical columns with
        the ``feature`` role, skipping ``ignore``, ``id``, ``target``,
        ``group``, ``time``, and ``weight``. Name columns explicitly to
        override.
    method:
        One of ``'onehot'``, ``'ordinal'``, ``'infrequent'``, or
        ``'target'``, as described above.
    min_frequency:
        For ``'infrequent'``, the line between "keep" and "pool". A float
        is a share of training rows (``0.05`` pools any level appearing in
        under 5% of them); an integer is a raw count. Raise it to compress
        harder, lower it to keep more distinct levels.
    n_folds:
        For ``'target'``, how many folds generate the out-of-fold averages.
        More folds mean each average is computed from more data and the
        encoding is less noisy, at proportionally more work.
    random_state:
        For ``'target'``, the seed for fold assignment, so the encoding is
        reproducible run to run.
    smoothing:
        For ``'target'``, how strongly rare levels are pulled toward the
        overall target mean. A level seen twice should not be trusted as
        much as one seen two thousand times; raising this trusts the global
        average more and the level-specific average less.

    Returns
    -------
    Session
        ``self``, so this call chains into the next transform.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split exists, a named column is absent, or ``'target'`` encoding
        was requested without a target column assigned.

    Notes
    -----
    **Leakage:** Requires a split. Vocabularies and target means are learned
    on train only. Target encoding writes out-of-fold values on train and
    full-train means on holdouts.

    Levels that appear only in test are unseen at fit time and encode to
    the all-zero row (one-hot) or the global prior (target). This is
    correct behaviour, not a bug: the model has no evidence about a
    category it never saw.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame(
    ...     {"city": ["dublin", "cork", "dublin", "cork"], "y": [0, 1, 0, 1]}
    ... )
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.inject_split(train_indices=[0, 1], test_indices=[2, 3])
    >>> _ = session.encode(method="onehot")
    >>> sorted(c for c in session.dataset.columns if c.startswith("city"))
    ['city_cork', 'city_dublin']

    Compress a wide identifier-like column instead of exploding it:

    >>> _ = session.encode(
    ...     columns=["merchant"], method="infrequent", min_frequency=0.01
    ... )  # doctest: +SKIP

    See Also
    --------
    Session.encode_plan : The learned vocabulary, for reuse at score time.
    Session.text_features : For free text rather than discrete labels.
    """
    session.assert_can_fit("train")
    plan = fit_encoder(
        session.dataset,
        session._split_plan,
        columns=columns,
        method=method,
        min_frequency=min_frequency,
        n_folds=n_folds,
        random_state=random_state,
        smoothing=smoothing,
    )
    session._dataset, result = transform_encoder(
        session.dataset, plan, split_plan=session._split_plan
    )
    session._encode_plan = plan
    session._last_preprocess = result
    session._record(
        "encode", plan.to_dict(), warnings=result.warnings, result_summary=result.to_dict()
    )
    return cast("Session", session)
def handle_outliers(
    session,
    *,
    columns: list[str] | None = None,
    method: Literal['iqr', 'zscore'] = "iqr",
    action: Literal['detect', 'cap', 'drop'] = "cap",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> "Session":
    """Find extreme numeric values, and decide what to do about them.

    A handful of very large values can dominate a model. Linear
    regressions chase them, scalers stretch to accommodate them, and
    distance-based methods let them distort every neighbourhood. This
    method locates them and applies the treatment you choose.

    Two detectors are available. ``'iqr'`` uses Tukey fences: the middle
    half of the training data defines a range, and anything more than
    ``iqr_multiplier`` times that range beyond it is flagged. It makes no
    assumption about the distribution's shape, which is why it is the
    default. ``'zscore'`` flags values more than ``zscore_threshold``
    standard deviations from the mean; that is cheaper to reason about but
    assumes roughly normal data, and it is self-defeating on heavy tails
    because the outliers themselves inflate the standard deviation.

    The treatment matters more than the detector. ``'detect'`` changes
    nothing and simply reports: always start here. ``'cap'`` pulls flagged
    values back to the fence, keeping the row and its other columns while
    removing the extreme's leverage. ``'drop'`` deletes the row entirely
    and rebuilds split membership around the loss.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    columns:
        Which numeric columns to screen. ``None`` selects numeric
        ``feature``-role columns, leaving identifiers, targets, and weights
        untouched.
    method:
        ``'iqr'`` for Tukey fences or ``'zscore'`` for standard-deviation
        distance.
    action:
        ``'detect'`` to report only, ``'cap'`` to clip to the fences, or
        ``'drop'`` to remove flagged rows.
    iqr_multiplier:
        How far beyond the middle-half range counts as extreme, for
        ``'iqr'``. The conventional ``1.5`` marks the usual boxplot
        whiskers; ``3.0`` flags only the genuinely far-out values.
    zscore_threshold:
        How many standard deviations from the mean counts as extreme, for
        ``'zscore'``.

    Returns
    -------
    Session
        ``self``, so this call chains into the next transform.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split exists, a named column is absent or non-numeric, or
        ``'drop'`` would empty a partition.

    Notes
    -----
    **Leakage:** Fence statistics are learned on train only, then applied
    with the frozen bounds. Heuristic screens are not proof of error.

    That last sentence is the important one. An outlier detector finds
    values that are unusual, not values that are wrong. In fraud, churn,
    and equipment failure the extreme rows are frequently the signal.
    Dropping them can quietly delete the very thing you set out to predict
   : run ``'detect'`` first and look at what was flagged before removing
    anything.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame(
    ...     {"amount": [10.0, 12.0, 11.0, 9000.0], "y": [0, 1, 0, 1]}
    ... )
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5)
    >>> _ = session.handle_outliers(action="detect")
    >>> session.outlier_plan.action
    'detect'

    Then, if the extremes really are recording errors, clip them:

    >>> _ = session.handle_outliers(method="iqr", action="cap")  # doctest: +SKIP

    See Also
    --------
    Session.fit_anomaly : When the extremes are what you want to find.
    Session.scale : Run after capping, so the scaler is not stretched.
    """
    session.assert_can_fit("train")
    assert session._split_plan is not None
    plan = fit_outlier_plan(
        session.dataset,
        session._split_plan,
        columns=columns,
        method=method,
        action=action,
        iqr_multiplier=iqr_multiplier,
        zscore_threshold=zscore_threshold,
    )
    dataset, split_plan, plan, result = apply_outlier_plan(
        session.dataset, session._split_plan, plan
    )
    session._dataset = dataset
    session._split_plan = split_plan
    session._outlier_plan = plan
    session._last_preprocess = result
    session._record(
        "handle_outliers", plan.to_dict(), warnings=result.warnings, result_summary=result.to_dict()
    )
    return cast("Session", session)
def bin(
    session,
    *,
    columns: list[str] | None = None,
    strategy: Literal['quantile', 'uniform'] = "quantile",
    n_bins: int = 5,
    encode_as: Literal['ordinal', 'onehot'] = "ordinal",
) -> "Session":
    """Group a continuous column into bands, trading detail for shape.

    Binning turns ``age = 34`` into ``age is in the 30–40 band``. You lose
    resolution, and in exchange you gain two things: the model can express
    a relationship that is not a straight line without you specifying its
    form, and the result is far easier to explain to someone who has to act
    on it. Risk bands, price tiers, and age brackets are how most people
    already think about these quantities.

    ``'quantile'`` places the edges so each band holds roughly the same
    number of training rows. Bands end up narrow where the data is dense
    and wide where it is sparse, so every band has enough examples to
    support an estimate. ``'uniform'`` makes every band the same width
    instead, which preserves the real spacing of the values but can leave
    some bands nearly empty when the distribution is skewed.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    columns:
        Which numeric columns to band. ``None`` selects numeric
        ``feature``-role columns.
    strategy:
        ``'quantile'`` for equal-population bands or ``'uniform'`` for
        equal-width bands.
    n_bins:
        How many bands to create. Fewer bands generalise more strongly and
        explain more cleanly; more bands retain more of the original
        signal. Past a point extra bands simply reintroduce the noise you
        were binning away.
    encode_as:
        ``'ordinal'`` writes the band number, which keeps one column and
        preserves the natural ordering of the bands. ``'onehot'`` writes an
        indicator per band, letting a linear model give each band its own
        independent effect at the cost of extra columns.

    Returns
    -------
    Session
        ``self``, so this call chains into the next transform.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split exists, a named column is absent or non-numeric, or
        ``n_bins`` exceeds the number of distinct training values
        available.

    Notes
    -----
    **Leakage:** Edges are learned on train only. End bins use open
    ``±inf`` edges so score-time extremes remain defined.

    That open-ended detail prevents a common production failure. If the
    outermost edges were closed at the training minimum and maximum, a new
    row beyond that range would fall into no band at all and produce a
    missing value at score time. Unbounded end bins mean every future value
    lands somewhere.

    Gradient-boosted trees already discover their own thresholds, so
    binning before them usually costs accuracy and gains nothing. Reach for
    it with linear models, or when the banded output is itself the
    deliverable.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame(
    ...     {"age": [21, 34, 47, 63], "y": [0, 1, 0, 1]}
    ... )
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5)
    >>> _ = session.bin(columns=["age"], n_bins=2, strategy="quantile")
    >>> session.binning_plan.n_bins
    2

    See Also
    --------
    Session.encode : Encode the resulting bands as independent indicators.
    Session.tune_threshold : Choose a cut-off on predictions, not inputs.
    """
    session.assert_can_fit("train")
    plan = fit_binning(
        session.dataset,
        session._split_plan,
        columns=columns,
        strategy=strategy,
        n_bins=n_bins,
        encode_as=encode_as,
    )
    session._dataset, result = transform_binning(session.dataset, plan)
    session._binning_plan = plan
    session._last_preprocess = result
    session._record(
        "bin", plan.to_dict(), warnings=result.warnings, result_summary=result.to_dict()
    )
    return cast("Session", session)
def select_features(
    session,
    *,
    strategy: Literal['variance', 'univariate', 'model'] = "variance",
    columns: list[str] | None = None,
    threshold: float = 0.0,
    k: int = 10,
    score_func: Literal['f_classif', 'f_regression', 'mutual_info'] = "f_classif",
    estimator: Any | None = None,
) -> "Session":
    """Keep the columns that carry signal and drop the rest.

    Fewer features usually means a model that trains faster, generalises
    better, and can be explained to someone. The difficulty is deciding
    which columns to lose. Three strategies are offered, in increasing
    order of how much they know about your problem.

    ``'variance'`` drops columns that barely change. A field that is the
    same value in 99% of rows cannot distinguish those rows, whatever the
    target is. This strategy never looks at the target, so it is cheap,
    safe, and a reasonable first pass: but it cannot tell a constant-ish
    column that matters from one that does not.

    ``'univariate'`` scores each column against the target on its own and
    keeps the best ``k``. It sees relevance, but only one column at a time:
    two features that are useless alone and powerful together will both be
    discarded, and ten copies of the same strong signal will all be kept.

    ``'model'`` fits an estimator and keeps the features it actually
    relied on. This is the only option that accounts for interactions and
    redundancy, because the model weighs features against each other. It
    costs a fit, and the selection inherits that model's biases :
    tree-based importances, for instance, favour high-cardinality columns.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    strategy:
        ``'variance'``, ``'univariate'``, or ``'model'``, as described
        above.
    columns:
        Restrict selection to these columns. ``None`` considers all
        ``feature``-role columns. Columns you exclude are kept
        unconditionally.
    threshold:
        For ``'variance'``, the minimum variance a column must have to
        survive. ``0.0`` removes only perfectly constant columns.
    k:
        For ``'univariate'``, how many top-scoring features to keep.
    score_func:
        For ``'univariate'``, how relevance is measured:
        ``'f_classif'`` for classification targets, ``'f_regression'`` for
        continuous ones, or ``'mutual_info'``, which detects non-linear
        relationships the F-tests miss but takes longer and is noisier on
        small samples.
    estimator:
        For ``'model'``, the fitted-on-train estimator whose importances
        drive selection. ``None`` uses a sensible default for the task.

    Returns
    -------
    Session
        ``self``, so this call chains into the next step.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split exists, features are still non-numeric or contain missing
        values, or ``k`` exceeds the number of available features.

    Notes
    -----
    **Leakage:** Selection fits on train only. Encode categoricals and
    impute before calling when features are non-numeric or contain nulls.

    Selection is itself a fitted decision. Choosing features on the whole
    dataset and then cross-validating is a classic way to produce scores
    that cannot be reproduced: the columns were already chosen with
    knowledge of the held-out rows. Selecting on train alone, as this does,
    avoids that.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame(
    ...     {
    ...         "useful": [1.0, 5.0, 2.0, 8.0],
    ...         "constant": [7.0, 7.0, 7.0, 7.0],
    ...         "y": [0, 1, 0, 1],
    ...     }
    ... )
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5)
    >>> _ = session.select_features(strategy="variance", threshold=0.0)
    >>> "constant" in session.feature_select_plan.dropped_features_
    True

    See Also
    --------
    Session.feature_importance : Explain a fitted model's reliance.
    Session.reduce_dimensions : Compress features instead of discarding.
    """
    session.assert_can_fit("train")
    plan = fit_feature_selector(
        session.dataset,
        session._split_plan,
        strategy=strategy,
        columns=columns,
        threshold=threshold,
        k=k,
        score_func=score_func,
        estimator=estimator,
    )
    session._dataset, result = transform_feature_selector(session.dataset, plan)
    session._feature_select_plan = plan
    session._last_preprocess = result
    session._record(
        "select_features", plan.to_dict(), warnings=result.warnings, result_summary=result.to_dict()
    )
    return cast("Session", session)
def scale(
    session, *, columns: list[str] | None = None, method: Literal['standard', 'minmax'] = "standard"
) -> "Session":
    """Put numeric columns on a comparable footing.

    Income runs to tens of thousands; a satisfaction rating runs from one
    to five. Any method that adds features together or measures distance
    between rows will let income drown out the rating purely because its
    numbers are bigger. Scaling removes that accident of units so the model
    weighs columns on evidence rather than magnitude.

    ``'standard'`` subtracts each column's training mean and divides by its
    training standard deviation, so the column ends up centred at zero with
    unit spread. Values are unbounded, which is the right behaviour when
    extremes are real, and it is the default.

    ``'minmax'`` squeezes each column into the ``[0, 1]`` range using the
    training minimum and maximum. Useful when a bounded input is required :
    some neural network layers, some visualisations: but fragile: one
    extreme training value compresses everything else into a narrow band,
    and a larger value at score time lands outside ``[0, 1]`` entirely.

    Scaling is essential for linear and logistic regression with
    regularisation, support vector machines, k-nearest neighbours,
    k-means, and PCA. Decision trees and their ensembles split one feature
    at a time and are completely indifferent to it.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    columns:
        Which columns to scale. ``None`` selects numeric ``feature``-role
        columns and skips ``ignore``, ``id``, ``target``, ``group``,
        ``time``, and ``weight``: so monetary amounts you are predicting
        and identifiers you need to read back stay in their original units.
        Name columns explicitly to override.
    method:
        ``'standard'`` for zero mean and unit variance, or ``'minmax'`` for
        a ``[0, 1]`` range.

    Returns
    -------
    Session
        ``self``, so this call chains into the fit.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split exists, a named column is absent or non-numeric, or a
        column still contains missing values: impute first.

    Notes
    -----
    **Leakage:** Requires a split. Scaler is fit on train only.

    Scale last, after imputing, encoding, and any outlier treatment. Each
    of those changes the distribution, and the scaler should learn from the
    distribution the model will actually receive.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame(
    ...     {"income": [30000.0, 52000.0, 41000.0, 78000.0],
    ...      "rating": [4.0, 2.0, 5.0, 3.0],
    ...      "y": [0, 1, 0, 1]}
    ... )
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5)
    >>> _ = session.scale(method="standard")
    >>> session.scale_plan.method
    'standard'

    See Also
    --------
    Session.impute : Run before scaling; scalers reject missing values.
    Session.reduce_dimensions : Scale first, or PCA follows the units.
    """
    session.assert_can_fit("train")
    plan = fit_scaler(session.dataset, session._split_plan, columns=columns, method=method)
    session._dataset = transform_scaler(session.dataset, plan)
    session._scale_plan = plan
    session._record("scale", plan.to_dict())
    return cast("Session", session)
def text_features(
    session,
    *,
    columns: list[str] | None = None,
    method: Literal['count', 'tfidf', 'hashing'] = "tfidf",
    max_features: int | None = 128,
    ngram_range: tuple[int, int] = (1, 1),
    drop_input_columns: bool = True,
) -> "Session":
    """Turn free-text columns into numeric features.

    A product review or a support ticket is a string, and a model needs
    numbers. These vectorisers convert each document into a row of counts
    or weights over words, replacing one text column with many numeric
    ones.

    ``'tfidf'`` counts each word, then discounts words that appear in many
    documents. "The" appears everywhere and distinguishes nothing;
    "refund" appears in a few documents and says a great deal. Weighting by
    that inverse document frequency is why TF-IDF is the sensible default.

    ``'count'`` keeps the raw counts with no discounting. Simpler to
    interpret, and the natural input to Naive Bayes, which expects counts.

    ``'hashing'`` skips the vocabulary altogether and hashes each word into
    a fixed number of slots. Nothing needs to be stored, so it handles
    streaming text and vocabularies too large to hold: at the price of
    collisions (two unrelated words can share a slot) and features you
    cannot map back to words.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    columns:
        Which text columns to vectorise. ``None`` selects string-valued
        ``feature``-role columns.
    method:
        ``'tfidf'``, ``'count'``, or ``'hashing'``, as described above.
    max_features:
        How wide the output is: the vocabulary size kept for ``'tfidf'``
        and ``'count'`` (the most frequent terms in training win), or the
        number of hash slots for ``'hashing'``. Larger retains more
        distinctions and costs more columns; too small for hashing means
        frequent collisions.
    ngram_range:
        The inclusive span of word-group sizes to include. ``(1, 1)`` uses
        single words only. ``(1, 2)`` adds adjacent pairs, which is how the
        vectoriser can tell "not good" from "good": worth the extra
        columns for sentiment-like problems.
    drop_input_columns:
        When True (the default), remove the original text column once its
        numeric features exist, since most estimators cannot consume the
        raw string anyway. Set False to keep the text for inspection.

    Returns
    -------
    Session
        ``self``, so this call chains into the fit.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split exists, or a named column is absent or not text-like.

    Notes
    -----
    **Leakage:** Requires a split. Vocabularies and IDF weights are learned
    from train documents only. Missing text becomes empty strings.

    Words that appear only in test documents are outside the fitted
    vocabulary and are ignored. That is correct: the model has no
    evidence about a word it never saw during training.

    These are bag-of-words representations: they record which words occur,
    not the order they occur in. When word order and meaning matter, reach
    for :meth:`fit_ssl_pretext` with a text backbone or the RAG methods
    instead.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame(
    ...     {
    ...         "review": ["great value", "poor value", "great service", "poor service"],
    ...         "y": [1, 0, 1, 0],
    ...     }
    ... )
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5)
    >>> _ = session.text_features(method="tfidf", max_features=8)
    >>> "review" in session.dataset.columns
    False

    Capture short phrases rather than isolated words:

    >>> _ = session.text_features(ngram_range=(1, 2))  # doctest: +SKIP

    See Also
    --------
    Session.encode : For discrete labels rather than free text.
    Session.rag_embed_and_index : For semantic search over documents.
    """
    session.assert_can_fit("train")
    plan = fit_text_features(
        session.dataset,
        session._split_plan,
        columns=columns,
        method=method,
        max_features=max_features,
        ngram_range=ngram_range,
        drop_input_columns=drop_input_columns,
    )
    session._dataset, result = transform_text_features(session.dataset, plan)
    session._text_plan = plan
    session._last_preprocess = result
    session._record(
        "text_features", plan.to_dict(), warnings=result.warnings, result_summary=result.to_dict()
    )
    return cast("Session", session)
def reduce_dimensions(
    session,
    *,
    columns: list[str] | None = None,
    method: Literal["pca", "umap", "tsne"] = "pca",
    n_components: int | float | None = None,
    drop_input_columns: bool = True,
    prefix: str = "pc",
    random_state: int | None = 0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: str | float = "auto",
) -> "Session":
    """Compress many numeric columns into a few informative ones.

    Where :meth:`select_features` discards columns, this one blends them.
    Each output column is built from all the inputs, so information spread
    thinly across fifty correlated measurements can survive in five
    columns. The cost is interpretability: ``pc_1`` is a mixture, not a
    measurement, and no business user will recognise it.

    ``'pca'`` finds the directions along which the training data varies
    most and projects onto them. It is linear, fast, deterministic, and
    genuinely reusable: a new row can be projected with the same fitted
    rotation, which is what makes it safe in a scoring pipeline.

    ``'umap'`` learns a non-linear embedding that tries to preserve local
    neighbourhood structure. It captures curved structure PCA cannot, and
    it can project new rows. Requires ``pip install
    'buildml[unsupervised]'``.

    ``'tsne'`` is transductive: it embeds the rows it was given and has no
    natural way to place a new one. BuildML transfers holdout rows by
    nearest neighbour and records that compromise in the plan's
    ``disclosures``. Treat t-SNE as a tool for looking at your training
    data, not as a step in a pipeline that will score fresh rows.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    columns:
        Which numeric columns to compress. ``None`` selects numeric
        ``feature``-role columns.
    method:
        ``'pca'``, ``'umap'``, or ``'tsne'``, as described above.
    n_components:
        How many output columns to produce. An integer sets the count
        directly. For PCA a float in ``(0, 1]`` instead names a target :
        ``0.95`` keeps however many components are needed to retain 95% of
        the training variance, which is usually the more meaningful way to
        ask. ``None`` uses the method's default.
    drop_input_columns:
        When True (the default), replace the source columns with the new
        ones. Keeping both is rarely useful, since the outputs are built
        from the inputs and the two are heavily redundant.
    prefix:
        Naming stem for the output columns, giving ``pc_1``, ``pc_2``, and
        so on.
    random_state:
        Seed for the methods with a stochastic component (UMAP and t-SNE),
        so repeated runs agree.
    umap_n_neighbors:
        For UMAP, how much of the neighbourhood each point considers. Small
        values preserve fine local detail; large values favour the global
        shape.
    umap_min_dist:
        For UMAP, how tightly points may be packed together in the output.
        Lower values produce tighter, more separated clumps.
    tsne_perplexity:
        For t-SNE, roughly how many neighbours each point is balanced
        against. It must be well below the number of rows, and the
        resulting picture changes noticeably with it: try several before
        drawing conclusions.
    tsne_learning_rate:
        For t-SNE, the optimiser step size. ``'auto'`` scales it to the
        sample size and is almost always the right choice.

    Returns
    -------
    Session
        ``self``, so this call chains into the fit.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No split exists, columns are non-numeric or contain missing values,
        or ``n_components`` exceeds the number of available columns.
    ~buildml.core.errors.MissingExtraError
        ``method='umap'`` without ``buildml[unsupervised]`` installed.

    Notes
    -----
    **Leakage:** Requires a split. The transform is learned on train only.
    Explained variance / embedding quality is unsupervised: not predictive utility.
    Scale numeric inputs first when magnitudes differ.

    That middle sentence is the trap. PCA maximises variance, and variance
    is not relevance: a component that explains 60% of the spread in your
    features can be irrelevant to the target, while the component
    explaining 2% carries the signal. Retaining 95% of variance does not
    retain 95% of predictive power.

    Scaling first is not optional advice. PCA works on covariance, so a
    column measured in thousands dominates one measured in units, and the
    leading components end up describing your choice of units rather than
    your data.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame(
    ...     {
    ...         "a": [1.0, 2.0, 3.0, 4.0],
    ...         "b": [2.0, 4.1, 5.9, 8.0],
    ...         "y": [0, 1, 0, 1],
    ...     }
    ... )
    >>> session = Session.ingest(frame).set_roles({"y": "target"})
    >>> _ = session.split(test_size=0.5)
    >>> _ = session.scale()
    >>> _ = session.reduce_dimensions(method="pca", n_components=1)
    >>> session.reduce_plan.n_components
    1

    Keep as many components as it takes to retain most of the variance:

    >>> _ = session.reduce_dimensions(method="pca", n_components=0.95)  # doctest: +SKIP

    See Also
    --------
    Session.select_features : Keep original columns instead of blending.
    Session.fit_clusters : Grouping, which often follows reduction.
    """
    session.assert_can_fit("train")
    plan = fit_reducer(
        session.dataset,
        session._split_plan,
        columns=columns,
        method=method,
        n_components=n_components,
        drop_input_columns=drop_input_columns,
        prefix=prefix,
        random_state=random_state,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        tsne_perplexity=tsne_perplexity,
        tsne_learning_rate=tsne_learning_rate,
    )
    session._dataset, result = transform_reducer(session.dataset, plan)
    session._reduce_plan = plan
    session._last_preprocess = result
    session._record(
        "reduce_dimensions",
        plan.to_dict(),
        warnings=result.warnings,
        result_summary=result.to_dict(),
    )
    return cast("Session", session)
def register_transform(
    session_cls,
    name: str,
    *,
    fit: Any,
    transform: Any,
    description: str = "",
    output_columns: Any | None = None,
    drop_input_columns: bool = False,
    serializable: bool = True,
    overwrite: bool = False,
) -> CustomTransformSpec:
    """Teach BuildML a preprocessing step of your own.

    The built-in transforms cover the common cases, but domain work
    routinely needs something specific: a currency conversion using rates
    learned from the training period, a geospatial encoding, a
    normalisation your field defines its own way.

    Registering it here rather than transforming the DataFrame by hand buys
    you the same guarantees the built-ins have. Your ``fit`` callable is
    shown training rows only, so the leakage rule holds. The fitted state is
    stored as a plan, so score-time replay reproduces it. And the step is
    recorded in the session history, so it appears in the walkthrough and
    the model card instead of being an invisible edit.

    Registration is on the class, not an instance: a transform registered
    once is available to every session in the process.

    Parameters
    ----------
    session_cls:
        The :class:`~buildml.session.session.Session` class (classmethod
        receiver) that owns the process-wide transform registry.
    name:
        The identifier you will pass to :meth:`apply_custom_transform`.
    fit:
        A callable receiving the training rows for the selected columns and
        returning whatever state the transform needs: a mapping, a fitted
        object, a tuple of statistics. Only training rows are ever passed
        in, which is what makes the step leakage-safe by construction.
    transform:
        A callable receiving that fitted state along with the rows to
        transform, and returning the transformed columns. Applied to every
        partition, and later to new data at score time.
    description:
        A short account of what the transform does. It appears in
        :meth:`list_transforms` and in the model card, so write it for
        whoever inherits the model.
    output_columns:
        The names of the columns produced. ``None`` keeps the input names,
        which is right for an in-place transformation and wrong for one
        that expands or renames.
    drop_input_columns:
        Remove the source columns after transforming. Set this when the
        outputs replace the inputs rather than supplementing them.
    serializable:
        Whether the fitted state can be pickled into a saved bundle. Set
        False for state holding an open connection or an unpicklable
        object; the transform then works in-process but cannot travel in a
        pipeline bundle.
    overwrite:
        Allow replacing an existing registration under the same name.
        Without it, re-registering raises: which catches the case of two
        modules quietly claiming the same name.

    Returns
    -------
    ~buildml.preprocess.custom.CustomTransformSpec
        The registered specification, as it will appear in
        :meth:`list_transforms`.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The name is already registered and ``overwrite`` is False, or
        ``fit`` or ``transform`` is not callable.

    Notes
    -----
    **Leakage:** The ``fit`` callable receives only train rows for the
    selected columns.

    Score-time replay needs the same name registered in the same process.
    A saved pipeline bundle stores the fitted state, not your Python code,
    so the scoring process must import whatever module performs the
    registration.

    Examples
    --------
    >>> from buildml import Session
    >>> Session.register_transform(
    ...     "log1p",
    ...     fit=lambda frame, **kwargs: {},
    ...     transform=lambda state, frame, **kwargs: frame.apply(
    ...         lambda col: (col + 1).map(float).map(__import__("math").log)
    ...     ),
    ...     description="Natural log of (x + 1), for right-skewed positives.",
    ...     overwrite=True,
    ... )  # doctest: +ELLIPSIS
    CustomTransformSpec(...)

    See Also
    --------
    Session.apply_custom_transform : Run a registered transform.
    Session.list_transforms : See what is currently registered.
    """
    return register_custom_transform(
        name,
        fit=fit,
        transform=transform,
        description=description,
        output_columns=output_columns,
        drop_input_columns=drop_input_columns,
        serializable=serializable,
        overwrite=overwrite,
    )


def list_transforms(session_cls) -> tuple[CustomTransformSpec, ...]:
    """List the custom transforms currently registered.

    Registration is process-wide, so this shows everything available to
    :meth:`apply_custom_transform`: including transforms registered by
    modules you imported rather than wrote.

    Parameters
    ----------
    session_cls:
        The :class:`~buildml.session.session.Session` class (classmethod
        receiver) that owns the process-wide transform registry.

    Returns
    -------
    tuple of ~buildml.preprocess.custom.CustomTransformSpec
        Every registered specification, ordered by name, each carrying its
        description and whether its fitted state can be serialised into a
        pipeline bundle.

    Examples
    --------
    >>> from buildml import Session
    >>> [spec.name for spec in Session.list_transforms()]  # doctest: +SKIP
    ['log1p']

    See Also
    --------
    Session.register_transform : Add one.
    """
    return list_registered_transforms()


def apply_custom_transform(
    session, name: str, *, columns: list[str], params: Mapping[str, Any] | None = None
) -> "Session":
    """Run a transform you registered, with the same leakage guarantees.

    Fits the named transform on the training rows and applies the result to
    every row, exactly as the built-in transforms behave. The fitted state
    is captured on :attr:`custom_plan` so score-time replay reproduces it,
    and the step is written into the session history so it shows up in the
    walkthrough and the model card.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    name:
        The name given to :meth:`register_transform`.
    columns:
        Which columns to pass to the transform. These are handed to your
        ``fit`` callable as training rows, and to ``transform`` for every
        partition.
    params:
        Extra keyword arguments forwarded to the registered ``fit``
        callable, letting one registration serve several configurations.

    Returns
    -------
    Session
        ``self``, so this call chains into the next step.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No transform is registered under that name, no split exists, or a
        named column is absent.

    Notes
    -----
    **Leakage:** Requires a split. Fit sees train rows only. Score-time
    replay requires the same name to remain registered in-process.

    Examples
    --------
    >>> _ = session.apply_custom_transform("log1p", columns=["amount"])  # doctest: +SKIP

    See Also
    --------
    Session.register_transform : Define the transform first.
    Session.custom_plan : The fitted state, for score-time replay.
    """
    session.assert_can_fit("train")
    plan = fit_custom_transform(
        session.dataset, session._split_plan, name=name, columns=columns, params=params
    )
    session._dataset, result = transform_custom(session.dataset, plan)
    session._custom_plan = plan
    session._last_preprocess = result
    session._record(
        "apply_custom_transform",
        plan.to_dict(),
        warnings=result.warnings,
        result_summary=result.to_dict(),
    )
    return cast("Session", session)
def extract_dates(
    session,
    columns: list[str] | tuple[str, ...] | None = None,
    *,
    include_time: bool = False,
    drop_original: bool = False,
) -> "Session":
    """Break timestamps apart into the calendar parts a model can use.

    A raw timestamp is nearly useless as a feature. As a number it counts
    seconds since 1970, which increases forever and tells a model nothing
    about the patterns that actually drive behaviour: those live in the
    parts. Retail spikes in December, support tickets arrive on weekdays,
    traffic peaks at rush hour. Splitting one datetime column into year,
    month, day, day-of-week, and optionally hour and minute makes each of
    those learnable.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    columns:
        Which datetime columns to expand. ``None`` finds every datetime
        column automatically.
    include_time:
        Also produce hour, minute, and second. Leave off for daily or
        coarser data, where the clock parts would be constant noise.
    drop_original:
        Remove the source timestamp after expanding. Keep it if a later
        step still needs to order rows: :meth:`time_split` reads the
        ``time``-role column, and dropping it out from under that will
        break the split.

    Returns
    -------
    Session
        ``self``, so this call chains into the next step.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A named column is absent or is not datetime-typed.

    Notes
    -----
    The expansion is row-wise and deterministic: nothing is learned from
    the data, so unlike the fitted transforms it carries no leakage risk
    and can be run before splitting.

    Calendar parts are numbers with a wrap-around: December (12) and
    January (1) are adjacent in reality but maximally distant as integers.
    Tree models cope, since they split on ranges. For linear models,
    consider one-hot encoding the month via :meth:`encode`, or supplying
    your own sine/cosine pair through :meth:`register_transform`.

    Examples
    --------
    >>> import pandas as pd
    >>> from buildml import Session
    >>> frame = pd.DataFrame(
    ...     {"ordered_at": pd.to_datetime(["2024-03-01", "2024-12-25"]), "y": [0, 1]}
    ... )
    >>> session = Session.ingest(frame)
    >>> _ = session.extract_dates(["ordered_at"])
    >>> "ordered_at_month" in session.dataset.columns
    True

    See Also
    --------
    Session.time_split : Split chronologically before expanding.
    Session.fit_forecast : When time is the axis, not just a feature.
    """
    session._dataset, plan = extract_date_features(
        session.dataset, columns=columns, include_time=include_time, drop_original=drop_original
    )
    session._date_plan = plan
    session._record("extract_dates", plan.to_dict())
    return cast("Session", session)
def apply_preprocess_plans(
    session,
    data: Dataset | pd.DataFrame | None = None,
    plans: dict[str, Any] | None = None,
    *,
    inplace: bool = True,
    use_session_plans: bool = True,
) -> ApplyPlansResult:
    """Replay fitted preprocessing on new rows, in the original order.

    Training-time preprocessing learns things: the median used to fill
    gaps, the category vocabulary, the scaler's mean and spread. New data
    must be transformed with *those* learned values, not with values
    recomputed from itself. This method replays the stored plans to do
    exactly that.

    Order matters as much as the values. Encoding before imputing, or
    scaling before encoding, produces a different matrix from the one the
    model was trained on. The sequence is therefore fixed: date expansion,
    imputation, outlier fences, encoding, binning, scaling, and finally
    feature selection.

    Nothing is fitted here. If a plan is missing, that step is skipped and
    recorded in the result rather than being quietly re-learned from the
    new data.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    data:
        The rows to transform, as a Dataset or a DataFrame. ``None`` uses
        this session's own dataset: which is what you want after
        :meth:`load_pipeline` has restored plans onto a session holding
        fresh data.
    plans:
        An explicit plan mapping, such as the ``plans.joblib`` payload from
        a checkpoint or pipeline bundle. ``None`` uses the plans attached
        to the session.
    inplace:
        When True and you are transforming the session's own dataset,
        replace it with the transformed version. Split membership is
        rebuilt if an outlier plan with ``action='drop'`` removed rows.
    use_session_plans:
        Fall back to session-attached plans for any step not covered by an
        explicit ``plans`` mapping, letting you override one step while
        keeping the rest.

    Returns
    -------
    ~buildml.preprocess.apply.ApplyPlansResult
        The transformed dataset, which steps were applied, which were
        skipped and why, and any warnings. Read the skipped list: a step
        silently absent means the model is about to receive features
        shaped differently from its training data.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A column a plan needs is missing from the incoming data, or a plan
        object is not one this method knows how to apply.

    Notes
    -----
    **Order:** dates → impute → outliers → encode → binning → scale →
    feature_select. Resample plans are lineage-only and are never
    reapplied at score time.

    **Leakage:** Plans must already be train-fitted; this method does not
    fit. Missing columns raise :class:`~buildml.core.errors.ValidationError`.

    Resampling is excluded on purpose. Rebalancing classes is a training
    trick to stop a model ignoring a rare class; applying it at score time
    would mean inventing or discarding real rows you were asked to predict.

    Examples
    --------
    >>> from buildml import Session
    >>> scorer = Session.ingest(new_rows)  # doctest: +SKIP
    >>> _ = scorer.load_pipeline("artifacts/churn_v3")  # doctest: +SKIP
    >>> applied = scorer.apply_preprocess_plans()  # doctest: +SKIP
    >>> applied.skipped  # doctest: +SKIP
    []

    See Also
    --------
    Session.predict_from_pipeline : Does this and the predict together.
    Session.load_pipeline : Restore the plans this replays.
    """
    if session._dataset is None and data is None:
        raise ValidationError("No dataset attached. Ingest data or pass data=...")
    resolved_plans = dict(plans or {})
    if use_session_plans:
        for key, value in session._plan_objects().items():
            resolved_plans.setdefault(key, value)
    target = session.dataset if data is None else data
    result = run_apply_preprocess_plans(
        target, resolved_plans, split_plan=session._split_plan
    )
    mutating = inplace and (
        data is None or (isinstance(data, Dataset) and data is session._dataset)
    )
    if mutating:
        session._dataset = result.dataset
        if result.split_plan is not None:
            session._split_plan = result.split_plan
    session._record(
        "apply_preprocess_plans",
        {"inplace": mutating, "applied": list(result.applied), "skipped": list(result.skipped)},
        warnings=result.warnings,
        result_summary=result.to_dict(),
    )
    return result


def resample(
    session,
    *,
    sampler: Literal['smote', 'random_oversample', 'random_undersample', 'adasyn', 'borderline_smote'] = "smote",
    random_state: int = 42,
    sampling_strategy: str | float | dict[str, float] = "auto",
) -> "Session":
    """Rebalance the training classes so the rare one is not ignored.

    When 2% of rows are fraud, a model can reach 98% accuracy by predicting
    "not fraud" every time. It has learned nothing, and the metric
    congratulates it. Resampling changes the training distribution so the
    rare class carries enough weight for the model to take it seriously.

    The oversampling methods add minority rows. ``'random_oversample'``
    duplicates existing ones, which is safe but gives the model repeated
    copies to overfit. ``'smote'`` instead synthesises new rows by
    interpolating between nearby minority examples, producing variety
    rather than duplicates: usually the better default.
    ``'borderline_smote'`` concentrates that synthesis near the decision
    boundary, where the difficult cases are. ``'adasyn'`` puts more
    synthetic rows around minority examples the model currently gets wrong.

    ``'random_undersample'`` goes the other way and discards majority rows.
    Fast, and it throws away real data: reasonable only when the majority
    class is enormous and largely redundant.

    Only the training partition is altered. Validation and test keep the
    real class balance, because those are meant to reflect the world you
    will deploy into.

    Requires ``pip install 'buildml[imbalanced]'``.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.
    sampler:
        Which method to apply, from those described above.
    random_state:
        Seed for the sampling and synthesis, making the result
        reproducible.
    sampling_strategy:
        How far to rebalance. ``'auto'`` levels the classes fully. A float
        sets the target minority-to-majority ratio, so ``0.5`` brings the
        minority to half the majority rather than all the way: often a
        better trade, since full balancing can push a model into
        over-predicting the rare class. A dict names target counts per
        class.

    Returns
    -------
    Session
        ``self``, so this call chains into the fit.

    Raises
    ------
    ~buildml.core.errors.MissingExtraError
        ``buildml[imbalanced]`` is not installed.
    ~buildml.core.errors.ValidationError
        No split exists, no target is assigned, the features are not yet
        numeric, or a minority class has too few rows for the chosen
        synthesis method.

    Notes
    -----
    SMOTE interpolates between neighbours, so it needs numeric features:
    encode and impute first. It also assumes the space between two
    minority rows is itself plausible, which is false when features are
    categorical or constrained: a synthetic point can be an impossible
    record.

    Resampling is not the only answer to imbalance, and often not the best.
    Class weights in the estimator, or moving the decision threshold with
    :meth:`tune_threshold`, address the same problem without inventing
    rows. Try those first.

    Examples
    --------
    >>> _ = session.resample(sampler="smote", sampling_strategy=0.5)  # doctest: +SKIP

    See Also
    --------
    Session.resample_strategies : Guidance on choosing among these.
    Session.tune_threshold : Handle imbalance at decision time instead.
    """
    dataset, plan, resample_plan = resample_train(
        session.dataset,
        session._split_plan,
        sampler=sampler,
        random_state=random_state,
        sampling_strategy=sampling_strategy,
    )
    session._dataset = dataset
    session._split_plan = plan
    session._resample_plan = resample_plan
    session._record("resample", resample_plan.to_dict())
    return cast("Session", session)
def resample_strategies(session) -> list[dict[str, Any]]:
    """List the available resampling methods and when each one fits.

    A reference you can read at runtime rather than looking up: each entry
    names a strategy, describes what it does to the data, and says when it
    is the appropriate choice.

    Parameters
    ----------
    session:
        Active Session instance this operation mutates or reads.

    Returns
    -------
    list of dict
        One entry per strategy accepted by :meth:`resample`, with its name,
        description, and guidance on when to use it.

    Examples
    --------
    >>> [s["name"] for s in session.resample_strategies()]  # doctest: +SKIP
    ['smote', 'random_oversample', 'random_undersample', ...]

    See Also
    --------
    Session.resample : Apply one of these strategies.
    """
    return list_resample_strategies()
