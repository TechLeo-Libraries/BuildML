"""The vocabulary of ensemble configuration: strategies, modes, and settings.

The three ``Literal`` aliases here are how invalid combinations get caught by a
type checker before anyone runs anything. ``VotingMethod`` cannot be
``'medium'``; ``BlendMethod`` cannot be ``'decision_function'``. Cheaper than a
runtime error, and visible in an editor's completions.

:class:`EnsembleConfig` records what was *requested*, which is not always what
happened. Soft voting is downgraded to hard for regression; probability blending
falls back to label blending when a base cannot supply probabilities. Keeping
the request separate from the outcome is what makes those adjustments visible
rather than mysterious.

See Also
--------
buildml.ensemble.results : What the fit actually produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EnsembleStrategy = Literal["voting", "stacking", "blending"]
VotingMethod = Literal["hard", "soft"]
BlendMethod = Literal["predict", "predict_proba"]


@dataclass(slots=True)
class EnsembleConfig:
    """Every ensemble setting in one record, as requested.

    A snapshot of the configuration, stored on the plan so a fit can be
    reproduced or explained later. Fields for all three strategies live
    together, and the irrelevant ones simply keep their defaults: ``cv`` sits
    at 5 in a voting config and means nothing there.

    The distinction that matters: this is the *request*. Where the resolved
    behaviour differed, the plan records the resolved value and this records
    what was asked for. Comparing the two is how you notice that the
    probability blending you configured actually ran on labels.

    Attributes
    ----------
    strategy:
        Which ensemble was built.
    estimator_names:
        The base models' names, in order.
    task:
        As requested, including ``'auto'``. The plan holds what ``'auto'``
        resolved to.
    voting:
        ``'hard'`` or ``'soft'``, as requested. Forced to ``'hard'`` for
        regression, and the plan shows that.
    weights:
        Relative influence per base, or ``None`` for equal.
    cv:
        Out-of-fold folds for stacking.
    passthrough:
        Whether the meta-learner also gets the original features.
    stack_method:
        What the bases contribute to a stack.
    final_estimator_name:
        The requested meta-learner's class name, or ``None`` for the default.
    holdout_fraction:
        Share of train reserved for a blend's meta-learner.
    blend_method:
        As requested. May have been downgraded; the plan holds the truth.
    random_state:
        Seed for a blend's inner split.
    refit_bases_on_full_train:
        Whether bases were refit on all of train after the meta fit.
    extras:
        Room for strategy options added later without changing this class.

    Notes
    -----
    **Irrelevant fields are not meaningful.** Read ``strategy`` first, then only
    the fields that belong to it.

    See Also
    --------
    buildml.ensemble.results.EnsemblePlan : What actually happened.
    """

    strategy: EnsembleStrategy
    estimator_names: tuple[str, ...]
    task: Literal["classification", "regression", "auto"] = "auto"
    # Voting
    voting: VotingMethod = "hard"
    weights: tuple[float, ...] | None = None
    # Stacking
    cv: int = 5
    passthrough: bool = False
    stack_method: str = "auto"
    final_estimator_name: str | None = None
    # Blending (holdout inside train)
    holdout_fraction: float = 0.2
    blend_method: BlendMethod = "predict_proba"
    random_state: int | None = 0
    refit_bases_on_full_train: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten to JSON-safe values for the plan and the bundle metadata.

        Every field is serialisable already; tuples widen to lists so the result
        round-trips through JSON unchanged, and ``extras`` is copied so a caller
        mutating the returned dict cannot reach back into the config.

        Returns
        -------
        dict
            JSON-serialisable settings, including the ones irrelevant to the
            strategy: completeness is worth more here than tidiness, since a
            reader comparing two configs wants the same keys in both.

        See Also
        --------
        buildml.ensemble.results.EnsemblePlan.to_dict : Where this is embedded.
        """
        return {
            "strategy": self.strategy,
            "estimator_names": list(self.estimator_names),
            "task": self.task,
            "voting": self.voting,
            "weights": None if self.weights is None else list(self.weights),
            "cv": self.cv,
            "passthrough": self.passthrough,
            "stack_method": self.stack_method,
            "final_estimator_name": self.final_estimator_name,
            "holdout_fraction": self.holdout_fraction,
            "blend_method": self.blend_method,
            "random_state": self.random_state,
            "refit_bases_on_full_train": self.refit_bases_on_full_train,
            "extras": dict(self.extras),
        }
