"""What each stage of the case-based reasoning workflow hands back.

One result type per operation: fit, predict, evaluate, retrieve, retain: plus
:class:`CbrPlan`, which is the fitted artefact itself rather than a report of
what happened.

Two fields recur on all of them and are worth reading rather than skipping.
``disclosures`` state how a result was produced: which backend actually ran,
whether a requested one was unavailable, what was inferred rather than
specified. ``warnings`` flag things that are likely to mislead: a case base too
small for the requested ``k``, neighbours far enough away that the prediction is
extrapolation.

The prediction and retrieval results also carry
:class:`~buildml.cbr.cases.CaseTrace` objects, which is the distinctive thing
here. Most methods can tell you what they predicted; these can show you the
specific past cases the prediction came from.

See Also
--------
buildml.cbr.cases.CaseTrace : The per-query explanation.
buildml.cbr.types.CbrConfig : The settings recorded on these results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from buildml.cbr.cases import CaseBase, CaseTrace


@dataclass(slots=True)
class CbrPlan:
    """The fitted reasoner: case memory plus everything needed to query it.

    The equivalent of a trained model, except that the training data is inside
    it. Carries the case base, the resolved configuration, the label encoding,
    and the column contract a query must satisfy. Predicting, evaluating,
    retrieving, and retaining all take one of these.

    This is tabular case-memory reasoning for supervised problems. It is not
    document retrieval for generation, not a vector database, and not a
    cognitive CBR framework; for the first of those see :mod:`buildml.rag`.

    Attributes
    ----------
    task:
        ``'classification'`` or ``'regression'``.
    backend:
        The retrieval backend that actually ran, which may differ from the one
        requested if an optional dependency was absent.
    metric, reuse, adapt, k:
        The resolved distance, combination, adaptation, and neighbour count.
    columns, categorical_columns, text_columns:
        The feature contract. A query frame must supply these.
    text_model_name:
        The sentence-transformer behind text features.
    target_column:
        What was predicted.
    n_train_rows:
        Rows the fit saw. Compare against ``case_base.n_cases`` to see how much
        memory has grown through retention.
    case_base:
        The cases and their fitted distance machinery.
    classes_:
        Class labels in encoded order, or ``None`` for regression.
    label_encoder_:
        The encoder that maps labels to integers and back.
    distance_eps:
        Floor used when inverting distances into weights.
    standardize:
        Whether numeric features were centred and scaled on train rows.
    disclosures, warnings:
        How the plan was built, and what about it may mislead.
    used_reduce_components:
        Whether reduced components were searched instead of raw features.
        Neighbours in a reduced space are neighbours under that projection,
        which is not quite the same claim.
    config:
        The originally requested configuration, kept so a fallback is visible as
        a difference from what was asked for.

    Notes
    -----
    **The plan contains your training data.** Every case is retained in full,
    which is what makes explanation possible and also means a saved bundle
    carries whatever the features contained. Treat it with the same care as the
    source data.

    **Memory scales with rows, not with model complexity.** There is no fitting
    cost to speak of and no compression; a large training set produces a large
    plan and slower queries.

    See Also
    --------
    buildml.cbr.fit.fit_cbr : Producing one.
    buildml.cbr.checkpoint.save_cbr_bundle : Persisting one.
    """

    task: str
    backend: str
    metric: str
    reuse: str
    adapt: str
    k: int
    columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    text_columns: tuple[str, ...]
    text_model_name: str
    target_column: str
    n_train_rows: int
    case_base: CaseBase
    classes_: tuple[Any, ...] | None
    label_encoder_: Any = field(repr=False, default=None)
    distance_eps: float = 1e-8
    standardize: bool = True
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    used_reduce_components: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Summarise the plan as a JSON-safe mapping.

        A description rather than a serialisation: the configuration, the
        column contract, and case counts, with only a preview of the memory
        itself.

        Returns
        -------
        dict
            The configuration and counts, with ``case_base`` holding the
            preview from :meth:`~buildml.cbr.cases.CaseBase.to_dict`.

        Notes
        -----
        **This cannot be loaded back into a working plan.** The matrices,
        fitted statistics, and label encoder are all absent. Use
        :func:`~buildml.cbr.checkpoint.save_cbr_bundle` to persist.
        """
        return {
            "kind": "cbr",
            "task": self.task,
            "backend": self.backend,
            "metric": self.metric,
            "reuse": self.reuse,
            "adapt": self.adapt,
            "k": self.k,
            "columns": list(self.columns),
            "categorical_columns": list(self.categorical_columns),
            "text_columns": list(self.text_columns),
            "text_model_name": self.text_model_name,
            "target_column": self.target_column,
            "n_train_rows": self.n_train_rows,
            "n_cases": self.case_base.n_cases,
            "n_retained": self.case_base.n_retained,
            "classes": None if self.classes_ is None else list(self.classes_),
            "distance_eps": self.distance_eps,
            "standardize": self.standardize,
            "case_base": self.case_base.to_dict(),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
            "used_reduce_components": self.used_reduce_components,
            "config": dict(self.config),
        }


@dataclass(slots=True)
class CbrFitResult:
    """What the fit produced, and what it does not yet tell you.

    The report accompanying a new :class:`CbrPlan`: the resolved settings, the
    column contract, how many cases were stored, and an optional training score.

    Attributes
    ----------
    task, backend, metric, reuse, k:
        The settings that were actually used. Compare ``backend`` against what
        you asked for: a missing optional dependency falls back rather than
        failing, and the fallback is recorded here and in ``disclosures``.
    n_train_rows, n_cases:
        Rows seen and cases stored. These differ when rows were dropped for
        missing targets.
    columns, categorical_columns:
        The features that will define distance.
    target_column:
        What is being predicted.
    classes:
        Class labels for classification, ``None`` for regression.
    train_score:
        Accuracy or R² on the training rows, when computed.
    disclosures, warnings:
        How the case base was built, and what may mislead about it.

    Notes
    -----
    **The training score is close to meaningless here, and unusually so.** Every
    training row is in the case base, so each is its own nearest neighbour at
    distance zero. With distance weighting, that self-match dominates and the
    score approaches perfect regardless of whether the method works. Evaluate on
    a holdout partition.

    See Also
    --------
    buildml.cbr.evaluate.evaluate_cbr : The score that means something.
    """

    task: str
    backend: str
    metric: str
    reuse: str
    k: int
    n_train_rows: int
    n_cases: int
    columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    target_column: str
    classes: tuple[Any, ...] | None = None
    train_score: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the fit report as a JSON-safe mapping.

        Suitable for history records and run logs, where the settings that
        actually ran are what you will want months later.

        Returns
        -------
        dict
            Every field, with tuples as lists.
        """
        return {
            "task": self.task,
            "backend": self.backend,
            "metric": self.metric,
            "reuse": self.reuse,
            "k": self.k,
            "n_train_rows": self.n_train_rows,
            "n_cases": self.n_cases,
            "columns": list(self.columns),
            "categorical_columns": list(self.categorical_columns),
            "target_column": self.target_column,
            "classes": None if self.classes is None else list(self.classes),
            "train_score": self.train_score,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CbrEvalResult:
    """How well the reasoner did on rows that are not in its memory.

    The score that counts. Because every training row is its own nearest
    neighbour, only a partition held out of the case base gives an honest
    reading.

    Attributes
    ----------
    partition:
        Which partition was scored. Anything but a holdout is not a measurement.
    task:
        ``'classification'`` or ``'regression'``, determining which metrics
        appear.
    n_rows:
        Rows scored.
    metrics:
        Accuracy for classification; RMSE, MAE, and R² for regression.
    mean_neighbor_distance:
        Average distance to the retrieved neighbours across all queries. Read
        this alongside the metrics: it is the diagnostic that says whether the
        score describes interpolation or extrapolation.
    disclosures, warnings:
        How the evaluation ran, and what may mislead about it.

    Notes
    -----
    **A good score with large neighbour distances is fragile.** It means the
    reasoner is generalising from cases that are not very similar, and it will
    degrade sharply on inputs further still from memory. There is no absolute
    threshold; compare against the distances seen during fitting.

    See Also
    --------
    buildml.cbr.evaluate.evaluate_cbr : Producing this.
    """

    partition: str
    task: str
    n_rows: int
    metrics: dict[str, float]
    mean_neighbor_distance: float | None = None
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluation as a JSON-safe mapping.

        Keeps ``mean_neighbor_distance`` alongside the metrics, so the
        diagnostic that qualifies the score travels with it.

        Returns
        -------
        dict
            Every field, with the metrics as a plain dict.
        """
        return {
            "partition": self.partition,
            "task": self.task,
            "n_rows": self.n_rows,
            "metrics": dict(self.metrics),
            "mean_neighbor_distance": self.mean_neighbor_distance,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CbrPredictResult:
    """The predictions, and the past cases each one came from.

    What makes this different from any other predict result is ``traces``: for
    each row, the neighbours that were consulted, how far away they were, how
    much each counted, and what each of them resolved to. The prediction is not
    accompanied by an explanation; the prediction *is* the explanation, worked
    through.

    Attributes
    ----------
    partition:
        Which rows were predicted.
    task:
        ``'classification'`` or ``'regression'``.
    n_rows:
        Rows predicted.
    predictions:
        The predicted values, aligned with the input rows and decoded back to
        original labels for classification.
    traces:
        One :class:`~buildml.cbr.cases.CaseTrace` per row when tracing is on,
        empty otherwise.
    disclosures, warnings:
        How prediction ran, and what may mislead about it.

    Notes
    -----
    **A prediction is always produced, however unlike memory the query is.**
    Neighbours are the nearest available, not the nearest *close* ones. Check
    the trace distances before acting on a prediction for an unusual input.

    **Traces are large.** Each carries its neighbours' identifiers, distances,
    weights, and solutions, so a full prediction run over many rows holds
    considerably more than the predictions alone.

    See Also
    --------
    buildml.cbr.cases.CaseTrace : What a trace contains.
    """

    partition: str
    task: str
    n_rows: int
    predictions: tuple[Any, ...]
    traces: tuple[CaseTrace, ...] = ()
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise the prediction run without the predictions themselves.

        Counts rather than contents, because predictions and traces can be very
        large and this is what goes into history. Read ``predictions`` and
        ``traces`` off the object directly when you want the values.

        Returns
        -------
        dict
            ``partition``, ``task``, ``n_rows``, ``n_predictions``,
            ``n_traces``, and the disclosure fields.
        """
        return {
            "partition": self.partition,
            "task": self.task,
            "n_rows": self.n_rows,
            "n_predictions": len(self.predictions),
            "n_traces": len(self.traces),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CbrRetrieveResult:
    """The neighbours themselves, with no prediction made from them.

    Retrieval stopped halfway on purpose. Before trusting predictions it is
    worth looking at what the reasoner considers similar, and this is the
    surface for that: query a few rows, read the neighbours, and judge whether
    the notion of similarity matches your own. A feature set that produces
    unconvincing neighbours will produce unconvincing predictions, and this is
    the cheaper place to find out.

    Attributes
    ----------
    partition:
        Which rows were used as queries.
    k:
        Neighbours retrieved per query.
    metric:
        The distance function used.
    n_queries:
        How many queries were run.
    traces:
        One per query, carrying the neighbours and their distances. The
        ``prediction`` field is unset: nothing was combined.
    backend:
        Which retrieval backend ran.
    disclosures, warnings:
        How retrieval ran, and what may mislead about it.

    Notes
    -----
    **Compare distances across queries to find the weak spots.** A query whose
    nearest neighbour is far away is one the reasoner has no real basis for, and
    the pattern in which queries those are usually says something about coverage.

    See Also
    --------
    buildml.cbr.retrieve.retrieve_cases : Producing this.
    buildml.cbr.predict.predict_cbr : The full path, neighbours included.
    """

    partition: str
    k: int
    metric: str
    n_queries: int
    traces: tuple[CaseTrace, ...]
    backend: str = "sklearn"
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Summarise the retrieval run, counting traces rather than listing them.

        The traces are the point of retrieval and also the bulk of it, so the
        dictionary form records that they exist and leaves them on the object.

        Returns
        -------
        dict
            ``partition``, ``k``, ``metric``, ``backend``, ``n_queries``,
            ``n_traces``, and the disclosure fields. Read ``traces`` off the
            object for the neighbours themselves.
        """
        return {
            "partition": self.partition,
            "k": self.k,
            "metric": self.metric,
            "backend": self.backend,
            "n_queries": self.n_queries,
            "n_traces": len(self.traces),
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class CbrRetainResult:
    """What happened when new cases were offered to memory.

    Retention is the part of the CBR cycle that lets a reasoner improve without
    retraining: solve a case, confirm the outcome, keep it. The counts here are
    worth watching, because the same mechanism is how a case base quietly stops
    resembling the data it was validated against.

    Attributes
    ----------
    n_added:
        Cases admitted.
    n_cases_after:
        Total memory size afterwards.
    n_skipped:
        Cases refused: duplicates, missing targets, or feature mismatches. A
        high count relative to ``n_added`` usually means the incoming data does
        not match the plan's column contract.
    disclosures, warnings:
        How retention ran, and what may mislead about it.

    Notes
    -----
    **Retained cases change what the holdout score describes.** The evaluation
    measured the memory as it was; once it has grown, that number is a
    historical fact rather than a current one. Re-evaluate periodically.

    **Distance transforms are not refitted on retained cases.** The
    standardisation and vocabularies stay as they were fitted on train, which
    keeps evaluation honest but means a substantial distribution shift in
    retained data is scaled by increasingly stale statistics.

    See Also
    --------
    buildml.cbr.retain.retain_cbr : Producing this.
    """

    n_added: int
    n_cases_after: int
    n_skipped: int
    disclosures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the retention outcome as a JSON-safe mapping.

        Worth recording in history: the growth of a case base over time is the
        context for any later change in its behaviour.

        Returns
        -------
        dict
            Every field, with tuples as lists.
        """
        return {
            "n_added": self.n_added,
            "n_cases_after": self.n_cases_after,
            "n_skipped": self.n_skipped,
            "disclosures": list(self.disclosures),
            "warnings": list(self.warnings),
        }
