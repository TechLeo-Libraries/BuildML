"""What an ensemble fit produces: the reusable plan and the record of it.

Two objects that overlap on purpose. The plan carries the fitted estimator and
is what you save, reload, and predict with. The result carries the same facts
without the estimator, and is what goes into history, reports, and metadata —
places where a multi-megabyte pickled model has no business being.

Both carry ``disclosures``, sentences describing what was fitted on what. They
are generated rather than written by hand, so they cannot drift from the code
that produced the model, and they are the honest answer to "did the meta-learner
see the test set" — which is the first question anyone should ask of a stack.

See Also
--------
buildml.ensemble.fit : What produces these.
buildml.ensemble.checkpoint : Persisting them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class EnsemblePlan:
    """The fitted ensemble, plus everything needed to explain it.

    The complete record of one ensemble fit: what was combined, how, on how much
    data, and with what caveats. The fitted estimator is scikit-learn
    compatible, so ``evaluate``, ``predict``, and ``save_pipeline`` treat the
    ensemble as an ordinary model.

    Several fields are strategy-specific and ``None`` outside their strategy —
    ``voting`` for voting, ``cv`` for stacking, ``holdout_fraction`` and
    ``blend_method`` for blending. One dataclass rather than three keeps
    everything downstream from branching on type.

    Attributes
    ----------
    strategy:
        ``'voting'``, ``'stacking'``, or ``'blending'``.
    task:
        ``'classification'`` or ``'regression'``.
    estimator_names:
        The base models' names, in order.
    feature_columns:
        Columns the ensemble expects, in order. The score-time contract.
    target_column:
        What was predicted.
    n_train_rows:
        Rows in the train partition.
    estimator_:
        The fitted ensemble. Excluded from ``repr`` because printing a fitted
        forest is not something anyone wants.
    final_estimator_name:
        The meta-learner's class name. ``None`` for voting, which has none.
    voting:
        ``'hard'`` or ``'soft'``. Voting only.
    cv:
        Folds used for out-of-fold meta-features. Stacking only.
    passthrough:
        Whether the meta-learner also saw the original features.
    holdout_fraction:
        Share of train reserved for the meta-learner. Blending only.
    blend_method:
        What the bases contributed. **Note this is what was actually used**,
        which may differ from what was requested if a base had no
        ``predict_proba``.
    refit_bases_on_full_train:
        Whether the bases were refit on all of train after the meta fit.
    disclosures:
        Generated sentences describing what was fitted on what. Safe to print
        into a report.
    warnings:
        Conditions worth attention — a blend holdout too small to be stable, for
        instance. Empty when nothing was flagged.
    config:
        The requested configuration, as given. Useful for reproducing a fit, and
        for seeing where the resolved values diverged from the request.

    Notes
    -----
    **The estimator also lives on the Session's ``FitResult``**, which is why
    the classical path keeps working unchanged after an ensemble fit.

    **A Session checkpoint does not carry this.** Use
    :func:`~buildml.ensemble.checkpoint.save_ensemble_bundle`, or the plan is
    gone on restore.

    See Also
    --------
    EnsembleFitResult : The same facts without the estimator.
    """

    strategy: Literal["voting", "stacking", "blending"]
    task: Literal["classification", "regression"]
    estimator_names: tuple[str, ...]
    feature_columns: tuple[str, ...]
    target_column: str
    n_train_rows: int
    estimator_: Any = field(repr=False)
    final_estimator_name: str | None = None
    voting: str | None = None
    cv: int | None = None
    passthrough: bool = False
    holdout_fraction: float | None = None
    blend_method: str | None = None
    refit_bases_on_full_train: bool = True
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten to JSON-safe values, replacing the estimator with its name.

        The estimator cannot be serialised as JSON, so ``estimator`` becomes its
        class name — enough to recognise what was built, not enough to run it.
        Loading the model back is what the joblib half of a bundle is for.

        Tuples become lists so the result round-trips through JSON unchanged; a
        tuple would come back as a list anyway, and having the two disagree
        makes comparing metadata files needlessly annoying.

        Returns
        -------
        dict
            JSON-serialisable metadata. ``estimator`` is a class name string,
            and every sequence is a list.

        Notes
        -----
        **This is a description, not a model.** Nothing here can reconstruct the
        ensemble.

        See Also
        --------
        buildml.ensemble.checkpoint.save_ensemble_bundle : Where this is written.
        """
        return {
            "strategy": self.strategy,
            "task": self.task,
            "estimator_names": list(self.estimator_names),
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "estimator": type(self.estimator_).__name__,
            "final_estimator_name": self.final_estimator_name,
            "voting": self.voting,
            "cv": self.cv,
            "passthrough": self.passthrough,
            "holdout_fraction": self.holdout_fraction,
            "blend_method": self.blend_method,
            "refit_bases_on_full_train": self.refit_bases_on_full_train,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "config": dict(self.config),
        }


@dataclass(slots=True)
class EnsembleFitResult:
    """The record of an ensemble fit, without the model attached.

    Deliberately estimator-free. This goes into Session history, report
    metadata, and log lines — places where holding a reference to a fitted
    model would pin megabytes in memory and make the record unserialisable.

    It answers the questions you ask *about* a fit rather than the ones you ask
    *of* a model. What was combined, on how much data, with what caveats.

    Attributes
    ----------
    strategy:
        ``'voting'``, ``'stacking'``, or ``'blending'``.
    task:
        ``'classification'`` or ``'regression'``.
    estimator_names:
        The base models' names, in order.
    n_train_rows:
        Rows the ensemble was fitted on.
    feature_columns:
        Columns it expects, in order.
    target_column:
        What was predicted.
    final_estimator_name:
        The meta-learner's class name, or ``None`` for voting.
    voting:
        ``'hard'`` or ``'soft'``. Voting only.
    cv:
        Out-of-fold folds. Stacking only.
    holdout_fraction:
        Share of train given to the meta-learner. Blending only.
    blend_method:
        What was actually blended, after any fallback.
    disclosures:
        Generated sentences about what was fitted on what.
    warnings:
        Conditions worth attention. **Read these** — an empty tuple is the
        signal that nothing was flagged, not that nothing was checked.

    See Also
    --------
    EnsemblePlan : The same facts with the fitted estimator.
    """

    strategy: Literal["voting", "stacking", "blending"]
    task: Literal["classification", "regression"]
    estimator_names: tuple[str, ...]
    n_train_rows: int
    feature_columns: tuple[str, ...]
    target_column: str
    final_estimator_name: str | None = None
    voting: str | None = None
    cv: int | None = None
    holdout_fraction: float | None = None
    blend_method: str | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Flatten to JSON-safe values for history and metadata.

        Every field is already serialisable — there is no estimator here — so
        this is a straight conversion, with tuples widened to lists so the
        result round-trips through JSON unchanged.

        Returns
        -------
        dict
            JSON-serialisable metadata, sequences as lists.

        See Also
        --------
        buildml.ensemble.explain_hooks.fit_result_summary : The smaller subset
            history keeps.
        """
        return {
            "strategy": self.strategy,
            "task": self.task,
            "estimator_names": list(self.estimator_names),
            "n_train_rows": self.n_train_rows,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "final_estimator_name": self.final_estimator_name,
            "voting": self.voting,
            "cv": self.cv,
            "holdout_fraction": self.holdout_fraction,
            "blend_method": self.blend_method,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }

    def show(self) -> None:
        """Print a one-glance summary with the first few disclosures.

        For the notebook, after a fit. One header line with the strategy, task,
        bases, and row count, then up to eight disclosures.

        The cap is there because a blend with several bases can generate more
        notes than anyone reads in a scroll-back, and the important ones come
        first. Iterate ``disclosures`` for all of them.

        Returns
        -------
        None
            Prints to stdout.

        Notes
        -----
        **``warnings`` is not printed here.** Check it separately; a blend
        holdout too small to be stable is the kind of thing that should not
        scroll past in a digest.
        """
        print(
            f"Ensemble · {self.strategy} · {self.task} · "
            f"bases={list(self.estimator_names)} · n_train={self.n_train_rows}"
        )
        for tip in self.disclosures[:8]:
            print(f"  - {tip}")
