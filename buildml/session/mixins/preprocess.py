"""Session mixin: preprocess domain public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from buildml.session import preprocess_ops
from buildml.session.mixins._shared import *  # noqa: F403


class PreprocessSessionMixin:
    """Public Session methods for the preprocess domain.

    Preferred namespaced API: ``session.preprocess.*`` (classical/core dual: flat methods remain first-class without warnings).
    """
    # mypy: session private attrs (owned by Session.__init__)
    if TYPE_CHECKING:
        _binning_plan: Any
        _custom_plan: Any
        _date_plan: Any
        _encode_plan: Any
        _feature_select_plan: Any
        _impute_plan: Any
        _last_preprocess: Any
        _outlier_plan: Any
        _reduce_plan: Any
        _resample_plan: Any
        _scale_plan: Any
        _text_plan: Any

    def drop_columns(self, columns: list[str] | tuple[str, ...]) -> Session:
        """Remove columns you do not want the model to see.

        Session facade over :func:`buildml.session.preprocess_ops.drop_columns`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the next step.

        See Also
        --------
        :func:`buildml.session.preprocess_ops.drop_columns`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", preprocess_ops.drop_columns(self, columns=columns))

    def impute(
        self,
        *,
        columns: list[str] | None = None,
        strategy: Literal["mean", "median", "most_frequent", "constant"] = "median",
        fill_value: Any | None = None,
    ) -> Session:
        """Fill in missing values, using only what the training rows reveal.

        Session facade over :func:`buildml.session.preprocess_ops.impute`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the next transform.

        See Also
        --------
        :func:`buildml.session.preprocess_ops.impute`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", preprocess_ops.impute(
            self, columns=columns, strategy=strategy, fill_value=fill_value
        ))

    @property
    def impute_plan(self) -> SimpleImputePlan | None:
        """The fill values learned by the last :meth:`impute` call.

        Session-held result for ``impute_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("SimpleImputePlan | None", self._impute_plan)

    def encode(
        self,
        *,
        columns: list[str] | None = None,
        method: Literal["onehot", "ordinal", "infrequent", "target"] = "onehot",
        min_frequency: float | int = 0.05,
        n_folds: int = 5,
        random_state: int = 0,
        smoothing: float = 10.0,
    ) -> Session:
        """Turn category labels into numbers a model can work with.

        Session facade over :func:`buildml.session.preprocess_ops.encode`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the next transform.

        See Also
        --------
        :func:`buildml.session.preprocess_ops.encode`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", preprocess_ops.encode(
            self,
            columns=columns,
            method=method,
            min_frequency=min_frequency,
            n_folds=n_folds,
            random_state=random_state,
            smoothing=smoothing,
        ))

    @property
    def encode_plan(self) -> EncodePlan | None:
        """The category vocabulary learned by the last :meth:`encode` call.

        Session-held result for ``encode_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("EncodePlan | None", self._encode_plan)

    def handle_outliers(
        self,
        *,
        columns: list[str] | None = None,
        method: Literal["iqr", "zscore"] = "iqr",
        action: Literal["detect", "cap", "drop"] = "cap",
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
    ) -> Session:
        """Find extreme numeric values, and decide what to do about them.

        Session facade over :func:`buildml.session.preprocess_ops.handle_outliers`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the next transform.

        See Also
        --------
        :func:`buildml.session.preprocess_ops.handle_outliers`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", preprocess_ops.handle_outliers(
            self,
            columns=columns,
            method=method,
            action=action,
            iqr_multiplier=iqr_multiplier,
            zscore_threshold=zscore_threshold,
        ))

    @property
    def outlier_plan(self) -> OutlierPlan | None:
        """The fences and counts from the last :meth:`handle_outliers` call.

        Session-held result for ``outlier_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("OutlierPlan | None", self._outlier_plan)

    def bin(
        self,
        *,
        columns: list[str] | None = None,
        strategy: Literal["quantile", "uniform"] = "quantile",
        n_bins: int = 5,
        encode_as: Literal["ordinal", "onehot"] = "ordinal",
    ) -> Session:
        """Group a continuous column into bands, trading detail for shape.

        Session facade over :func:`buildml.session.preprocess_ops.bin`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the next transform.

        See Also
        --------
        :func:`buildml.session.preprocess_ops.bin`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", preprocess_ops.bin(
            self, columns=columns, strategy=strategy, n_bins=n_bins, encode_as=encode_as
        ))

    @property
    def binning_plan(self) -> BinningPlan | None:
        """The band edges learned by the last :meth:`bin` call.

        Session-held result for ``binning_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("BinningPlan | None", self._binning_plan)

    def select_features(
        self,
        *,
        strategy: Literal["variance", "univariate", "model"] = "variance",
        columns: list[str] | None = None,
        threshold: float = 0.0,
        k: int = 10,
        score_func: Literal["f_classif", "f_regression", "mutual_info"] = "f_classif",
        estimator: Any | None = None,
    ) -> Session:
        """Keep the columns that carry signal and drop the rest.

        Session facade over :func:`buildml.session.preprocess_ops.select_features`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the next step.

        See Also
        --------
        :func:`buildml.session.preprocess_ops.select_features`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", preprocess_ops.select_features(
            self,
            strategy=strategy,
            columns=columns,
            threshold=threshold,
            k=k,
            score_func=score_func,
            estimator=estimator,
        ))

    @property
    def feature_select_plan(self) -> FeatureSelectPlan | None:
        """What the last :meth:`select_features` call kept, and why.

        Session-held result for ``feature_select_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("FeatureSelectPlan | None", self._feature_select_plan)

    @property
    def last_preprocess(self) -> PreprocessResult | None:
        """The narrated outcome of the most recent preprocessing step.

        Session-held result for ``last_preprocess``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("PreprocessResult | None", self._last_preprocess)

    def scale(
        self,
        *,
        columns: list[str] | None = None,
        method: Literal["standard", "minmax"] = "standard",
    ) -> Session:
        """Put numeric columns on a comparable footing.

        Session facade over :func:`buildml.session.preprocess_ops.scale`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the fit.

        See Also
        --------
        :func:`buildml.session.preprocess_ops.scale`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", preprocess_ops.scale(self, columns=columns, method=method))

    @property
    def scale_plan(self) -> ScalePlan | None:
        """The fitted scaler from the last :meth:`scale` call.

        Session-held result for ``scale_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ScalePlan | None", self._scale_plan)

    def text_features(
        self,
        *,
        columns: list[str] | None = None,
        method: Literal["count", "tfidf", "hashing"] = "tfidf",
        max_features: int | None = 128,
        ngram_range: tuple[int, int] = (1, 1),
        drop_input_columns: bool = True,
    ) -> Session:
        """Turn free-text columns into numeric features.

        Session facade over :func:`buildml.session.preprocess_ops.text_features`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the fit.

        See Also
        --------
        :func:`buildml.session.preprocess_ops.text_features`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", preprocess_ops.text_features(
            self,
            columns=columns,
            method=method,
            max_features=max_features,
            ngram_range=ngram_range,
            drop_input_columns=drop_input_columns,
        ))

    @property
    def text_plan(self) -> TextFeaturePlan | None:
        """The fitted vectorisers from the last :meth:`text_features` call.

        Session-held result for ``text_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("TextFeaturePlan | None", self._text_plan)

    def reduce_dimensions(
        self,
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
    ) -> Session:
        """Compress many numeric columns into a few informative ones.

        Session facade over :func:`buildml.session.preprocess_ops.reduce_dimensions`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the fit.

        See Also
        --------
        :func:`buildml.session.preprocess_ops.reduce_dimensions`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", preprocess_ops.reduce_dimensions(
            self,
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
        ))

    @property
    def reduce_plan(self) -> ReducePlan | None:
        """The fitted projection from the last :meth:`reduce_dimensions` call.

        Session-held result for ``reduce_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ReducePlan | None", self._reduce_plan)

    @classmethod
    def register_transform(
        cls,
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

        Session facade over :func:`buildml.session.preprocess_ops.register_transform`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.preprocess.custom.CustomTransformSpec
            The registered specification, as it will appear in

        See Also
        --------
        :func:`buildml.session.preprocess_ops.register_transform`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("CustomTransformSpec", preprocess_ops.register_transform(
            cls,
            name=name,
            fit=fit,
            transform=transform,
            description=description,
            output_columns=output_columns,
            drop_input_columns=drop_input_columns,
            serializable=serializable,
            overwrite=overwrite,
        ))

    @classmethod
    def list_transforms(cls) -> tuple[CustomTransformSpec, ...]:
        """List the custom transforms currently registered.

        Session facade over :func:`buildml.session.preprocess_ops.list_transforms`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        tuple of ~buildml.preprocess.custom.CustomTransformSpec
            Every registered specification, ordered by name, each carrying its

        See Also
        --------
        :func:`buildml.session.preprocess_ops.list_transforms`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("tuple[CustomTransformSpec, ...]", preprocess_ops.list_transforms(cls))

    def apply_custom_transform(
        self,
        name: str,
        *,
        columns: list[str],
        params: Mapping[str, Any] | None = None,
    ) -> Session:
        """Run a transform you registered, with the same leakage guarantees.

        Session facade over :func:`buildml.session.preprocess_ops.apply_custom_transform`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the next step.

        See Also
        --------
        :func:`buildml.session.preprocess_ops.apply_custom_transform`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", preprocess_ops.apply_custom_transform(
            self, name=name, columns=columns, params=params
        ))

    @property
    def custom_plan(self) -> CustomTransformPlan | None:
        """The fitted state from the last :meth:`apply_custom_transform` call.

        Session-held result for ``custom_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("CustomTransformPlan | None", self._custom_plan)

    def extract_dates(
        self,
        columns: list[str] | tuple[str, ...] | None = None,
        *,
        include_time: bool = False,
        drop_original: bool = False,
    ) -> Session:
        """Break timestamps apart into the calendar parts a model can use.

        Session facade over :func:`buildml.session.preprocess_ops.extract_dates`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the next step.

        See Also
        --------
        :func:`buildml.session.preprocess_ops.extract_dates`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", preprocess_ops.extract_dates(
            self, columns=columns, include_time=include_time, drop_original=drop_original
        ))

    @property
    def date_plan(self) -> DateFeaturePlan | None:
        """Which calendar parts the last :meth:`extract_dates` call produced.

        Session-held result for ``date_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("DateFeaturePlan | None", self._date_plan)

    def apply_preprocess_plans(
        self,
        data: Dataset | pd.DataFrame | None = None,
        plans: dict[str, Any] | None = None,
        *,
        inplace: bool = True,
        use_session_plans: bool = True,
    ) -> ApplyPlansResult:
        """Replay fitted preprocessing on new rows, in the original order.

        Session facade over :func:`buildml.session.preprocess_ops.apply_preprocess_plans`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        ~buildml.preprocess.apply.ApplyPlansResult
            The transformed dataset, which steps were applied, which were

        See Also
        --------
        :func:`buildml.session.preprocess_ops.apply_preprocess_plans`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("ApplyPlansResult", preprocess_ops.apply_preprocess_plans(
            self, data=data, plans=plans, inplace=inplace, use_session_plans=use_session_plans
        ))

    def resample(
        self,
        *,
        sampler: Literal[
            "smote",
            "random_oversample",
            "random_undersample",
            "adasyn",
            "borderline_smote",
        ] = "smote",
        random_state: int = 42,
        sampling_strategy: str | float | dict[str, float] = "auto",
    ) -> Session:
        """Rebalance the training classes so the rare one is not ignored.

        Session facade over :func:`buildml.session.preprocess_ops.resample`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        Session
            ``self``, so this call chains into the fit.

        See Also
        --------
        :func:`buildml.session.preprocess_ops.resample`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("Session", preprocess_ops.resample(
            self, sampler=sampler, random_state=random_state, sampling_strategy=sampling_strategy
        ))

    def resample_strategies(self) -> list[dict[str, Any]]:
        """List the available resampling methods and when each one fits.

        Session facade over :func:`buildml.session.preprocess_ops.resample_strategies`. Canonical Parameters, Raises, Notes, and Examples live on that ops function: keep this method as a thin delegate.

        Returns
        -------
        list of dict
            One entry per strategy accepted by :meth:`resample`, with its name,

        See Also
        --------
        :func:`buildml.session.preprocess_ops.resample_strategies`
            Canonical documentation for parameters, raises, and examples.
        """
        return cast("list[dict[str, Any]]", preprocess_ops.resample_strategies(self))

    @property
    def resample_plan(self) -> ResamplePlan | None:
        """What the last :meth:`resample` call did to the training rows.

        Session-held result for ``resample_plan``.
        ``None`` until the matching Session fit/score/load call populates it.
        """
        return cast("ResamplePlan | None", self._resample_plan)
