"""Score several estimators on identical terms and rank them.

Choosing an algorithm is usually an empirical question. Gradient boosting often
wins on tabular data and sometimes loses to a linear model; a random forest is
frequently close enough to a boosted tree that the simpler operational story
wins. Reasoning about which *should* work is a poor substitute for fitting a few
and looking.

The one thing that makes such a comparison meaningful is that every candidate
gets exactly the same treatment: the same split, the same features, the same
evaluation partition. That is what this module guarantees.

It is a screen, not a verdict. Every estimator runs at its default settings, and
defaults suit some algorithms far better than others: a boosted tree at defaults
is usually near its potential, while an SVM at defaults can be nowhere near its.
Use this to narrow the field to two or three, then tune those properly.

See Also
--------
buildml.model.selection : Tuning the shortlist.
buildml.model.supervised : The fit and evaluate underneath.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.model.supervised import EvaluateResult, FitResult, evaluate_estimator, fit_estimator


@dataclass(slots=True)
class ModelComparison:
    """Every candidate's scores, ranked, with the fitted models kept.

    The fitted estimators and their full evaluation cards are retained, not just
    the ranking table. That matters because the table shows which model scored
    highest and not why, and "why" is often where the decision actually lies :
    two models can post the same F1 while one misses a different class entirely.

    Attributes
    ----------
    task:
        ``'classification'`` or ``'regression'``, inferred from the estimators.
    ranking_metric:
        What the rows were sorted by.
    rows:
        One record per model with its metrics and row count, best first.
    fits:
        The fitted models by name, ready to predict with.
    evaluations:
        Each model's full evaluation card, including confusion matrices or
        residual summaries.
    recommendations:
        Notes on the comparison, including the gap between the top two.

    Notes
    -----
    **A small gap between the top two is not a ranking.** A single partition
    gives one number per model with no spread, so a difference of a few
    thousandths is well within resampling noise. Cross-validate the leaders
    before choosing between them.

    **Every model was fitted at its defaults.** The ranking reflects how well
    the defaults suit each algorithm as much as the algorithms themselves.

    See Also
    --------
    compare_estimators : Producing this comparison.
    """

    task: Literal["classification", "regression"]
    ranking_metric: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    fits: dict[str, FitResult] = field(default_factory=dict)
    evaluations: dict[str, EvaluateResult] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Lay the comparison out as a table, one row per model.

        The shape to eyeball, sort by a different metric, or drop into a report.

        Returns
        -------
        pandas.DataFrame
            A ``model`` column, one column per metric, and ``n_rows``. Ordered
            best-first by the ranking metric.

        Notes
        -----
        **Read across as well as down.** A model that leads on the ranking
        metric and trails badly on another is telling you the two metrics
        disagree about what matters.
        """
        return pd.DataFrame(self.rows)

    def to_dict(self) -> dict[str, Any]:
        """Convert the comparison to plain data for history and reports.

        The fitted models and their full evaluation cards are omitted, since
        neither serialises usefully; read them off the object instead.

        Returns
        -------
        dict
            ``task``, ``ranking_metric``, the ranked ``rows``, and
            ``recommendations``.
        """
        return {
            "task": self.task,
            "ranking_metric": self.ranking_metric,
            "rows": list(self.rows),
            "recommendations": list(self.recommendations),
        }

    def show(self) -> None:
        """Print the ranking table and the notes beneath it.

        For reading a comparison at a prompt.

        Notes
        -----
        **Check the top-two gap in the printed notes.** It is the quickest way
        to tell whether the ranking is a result or a coin toss.
        """
        print(self.to_frame().to_string(index=False))
        for tip in self.recommendations:
            print(f"- {tip}")


def compare_estimators(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    estimators: dict[str, Any],
    *,
    task: Literal["classification", "regression", "auto"] = "auto",
    partition: Literal["train", "validation", "test"] = "test",
    ranking_metric: str | None = None,
) -> ModelComparison:
    """Fit every candidate on the same data and rank what they score.

    Each estimator is trained on the train partition and scored on the same
    evaluation partition with the same features. Identical treatment is what
    makes the comparison mean anything: a model that scored well on a different
    split, or with different columns, cannot be ranked against the others at all.

    Parameters
    ----------
    dataset:
        The data, with roles set.
    split_plan:
        The split. Every model trains on train.
    estimators:
        Display names mapped to unfitted estimators. The names appear in the
        ranking, so make them recognisable.
    task:
        ``'classification'``, ``'regression'``, or ``'auto'``. Inferred once and
        applied to all, so mixing classifiers and regressors will not work.
    partition:
        Which partition to score on. Defaults to test.
    ranking_metric:
        What to sort by. Defaults to F1-weighted or R². Loss-like metrics sort
        ascending.

    Returns
    -------
    ModelComparison
        The ranking table, each fitted model, each full evaluation card, and
        notes including the top-two gap.

    Raises
    ------
    ValueError
        If ``estimators`` is empty.
    ValidationError
        If the split is missing, or if any estimator cannot be fitted on the
        resolved features: for example one that rejects a weight column.

    Notes
    -----
    **Scoring on test repeatedly wears it out.** Comparing five models on the
    test partition means five looks at data meant to be seen once. Compare on
    validation, and keep test for confirming the winner.

    **Defaults flatter some algorithms more than others.** Treat this as a
    screen: shortlist two or three, then tune them before deciding.

    **A single partition gives no spread**, so the ranking has no error bars.
    Use :func:`~buildml.model.selection.cv_score` on the leaders when the gap
    between them is small.

    Examples
    --------
    Screen three families on validation::

        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        comparison = compare_estimators(
            dataset, split_plan,
            {
                "logreg": LogisticRegression(max_iter=1000),
                "forest": RandomForestClassifier(random_state=0),
                "boosting": HistGradientBoostingClassifier(random_state=0),
            },
            partition="validation",
        )
        comparison.show()

    See Also
    --------
    buildml.model.selection.cv_score : Adding a spread to the comparison.
    buildml.model.supervised.evaluate_estimator : One model in more depth.
    """
    if not estimators:
        raise ValueError("estimators mapping must not be empty")

    fits: dict[str, FitResult] = {}
    evaluations: dict[str, EvaluateResult] = {}
    rows: list[dict[str, Any]] = []
    resolved_task: Literal["classification", "regression"] | None = None

    for name, estimator in estimators.items():
        fit = fit_estimator(dataset, split_plan, estimator, task=task)
        ev = evaluate_estimator(dataset, split_plan, fit, partition=partition)
        fits[name] = fit
        evaluations[name] = ev
        resolved_task = fit.task
        row = {"model": name, **ev.metrics, "n_rows": ev.n_rows}
        rows.append(row)

    assert resolved_task is not None
    metric = ranking_metric or ("r2" if resolved_task == "regression" else "f1_weighted")
    higher_is_better = metric not in {"mae", "mse", "rmse", "log_loss", "median_ae", "mape"}
    rows.sort(key=lambda item: item.get(metric, float("-inf")), reverse=higher_is_better)

    tips = [
        f"Ranked by '{metric}' on partition='{partition}'.",
        "Refit the chosen winner on the full training recipe before deployment.",
    ]
    if len(rows) >= 2 and metric in rows[0] and metric in rows[1]:
        gap = abs(float(rows[0][metric]) - float(rows[1][metric]))
        tips.append(f"Top-2 gap on {metric}: {gap:.6f}")

    return ModelComparison(
        task=resolved_task,
        ranking_metric=metric,
        rows=rows,
        fits=fits,
        evaluations=evaluations,
        recommendations=tips,
    )
