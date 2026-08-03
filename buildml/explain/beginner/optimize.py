# ruff: noqa: E501
"""Beginner layers for decision policies and constrained allocation."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

OPTIMIZE_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "decision-operating-point",
        plain=(
            "A prediction is not a decision. The operating point is the rule that turns scores into "
            "actions, chosen so that the total cost of your mistakes is as low as possible. BuildML lets "
            "you fit that rule on validation and save it alongside the model."
        ),
        analogy=(
            "The model tells you how likely rain is. The operating point is your household rule about when "
            "to actually carry an umbrella: and that depends on how much you hate getting wet versus "
            "carrying things."
        ),
        steps=(
            "Quantify your two costs: what a false positive costs, and what a false negative costs.",
            "Sweep the threshold across the validation partition, computing expected cost at each point.",
            "Pick the cutoff that minimizes expected cost, or that satisfies your capacity constraint.",
            "Freeze it as a decision plan so it travels with the model.",
            "Confirm the frozen policy once on test.",
        ),
        use=(
            "Whenever the model output drives an automated or semi-automated action.",
            "Whenever the two error types have genuinely different consequences, which is almost always.",
        ),
        avoid=(
            "Do not fit the operating point on the same partition you use to report performance.",
            "Do not keep a fixed cutoff after the base rate shifts; the same threshold implies a different alert volume.",
        ),
        myths=(
            (
                "Optimizing the metric optimizes the decision.",
                "Maximizing F1 assumes a particular implicit cost ratio. If your real costs are 1 to 20, F1 is optimizing the wrong thing.",
            ),
            (
                "The threshold is a modelling detail.",
                "It is usually the single highest-leverage decision in the whole project, and it belongs to the business, not the model.",
            ),
        ),
        example=(
            "session.fit_decision_policy(",
            "    method='threshold', partition='validation',",
            "    fp_cost=1.0, fn_cost=25.0,",
            ")",
            "decisions = session.apply_decisions(partition='test')",
            "session.evaluate_decisions(partition='test')",
        ),
        check=(
            "What is the money value of one false negative in your problem?",
            "Which partition chose your threshold, and which one reports its performance?",
        ),
        tools=("fit_decision_policy", "apply_decisions", "evaluate_decisions", "tune_threshold"),
        terms=("threshold", "expected value", "cost matrix", "precision", "recall"),
        difficulty=CORE,
    ),
    _layer(
        "decision-cost-matrix",
        plain=(
            "With more than two classes, a single threshold no longer works. A cost matrix says what each "
            "possible mistake costs: predicting B when the truth is A, predicting C when the truth is A, "
            "and so on: and the decision rule picks whichever action has the lowest expected cost."
        ),
        analogy=(
            "A triage desk. Sending a heart-attack patient home is catastrophic; admitting someone with "
            "indigestion is merely wasteful. The two errors are not remotely equivalent, and the rule has "
            "to reflect that."
        ),
        steps=(
            "Build a square table: rows are true classes, columns are the actions you could take.",
            "Fill each cell with the cost of taking that action when that class is true; the diagonal is usually zero.",
            "Get predicted probabilities for every class from your model.",
            "For each candidate action, compute the probability-weighted average cost.",
            "Choose the action with the lowest expected cost: which is often not the most likely class.",
        ),
        use=(
            "Multiclass problems where the consequences of different confusions differ substantially.",
            "Any setting where 'refer to a human' is one of the available actions with its own cost.",
        ),
        avoid=(
            "Do not use it with badly calibrated probabilities; the whole calculation multiplies them by costs, so distorted probabilities give distorted decisions.",
            "Do not invent cost numbers to make the maths work: the matrix should come from the business, and a wrong matrix is worse than none.",
        ),
        myths=(
            (
                "Picking the most likely class is the rational choice.",
                "It is optimal only when all mistakes cost the same. Under asymmetric costs, the rational action can have quite low probability.",
            ),
            (
                "The matrix has to be square with actions equal to classes.",
                "Actions can differ from classes: 'escalate to review' is an action with no corresponding true class, and it is often the most valuable column.",
            ),
        ),
        example=(
            "costs = [[0, 5, 50],",
            "         [2, 0, 20],",
            "         [80, 30, 0]]   # rows = truth, cols = action",
            "session.fit_decision_policy(method='cost_matrix', cost_matrix=costs)",
            "session.apply_decisions(partition='test')",
        ),
        check=(
            "Are your model's probabilities calibrated enough to multiply by money?",
            "Does your matrix include a 'do nothing' or 'refer to human' action?",
        ),
        tools=("fit_decision_policy", "apply_decisions", "calibration", "evaluate_decisions"),
        terms=("cost matrix", "expected value", "calibration", "predict_proba"),
        difficulty=ADVANCED,
    ),
    _layer(
        "decision-allocation",
        plain=(
            "Sometimes the constraint is not a threshold but a budget: you can only call 500 customers, "
            "only stock 30 items, only spend a fixed amount. Allocation picks the best set under that "
            "limit, which is a different problem from scoring rows independently."
        ),
        analogy=(
            "Packing a suitcase with a weight limit. You do not take everything valuable; you take the "
            "combination that fits and is worth the most."
        ),
        steps=(
            "Decide what you are allocating: a count (top-K), a budget with per-item costs (knapsack), or divisible shares (linear programming).",
            "Provide the value score per row: usually a model prediction, possibly an expected value.",
            "Provide the cost or weight per row when items are not equally expensive.",
            "State the constraint: how many, how much money, how much capacity.",
            "The solver returns the selected set, and you evaluate the realized value against alternatives.",
        ),
        use=(
            "Marketing campaigns, inventory buys, inspection scheduling, credit limits: anywhere capacity is finite.",
            "When per-item costs vary, which is exactly where simple top-K stops being optimal.",
        ),
        avoid=(
            "Do not use top-K when items have very different costs; a cheap moderately-good item can beat an expensive slightly-better one.",
            "Do not optimize against raw model scores when what you need is expected value: multiply by the payoff first.",
        ),
        myths=(
            (
                "Ranking by score and taking the top N is optimal.",
                "Only when every item costs the same. With varying costs, that is the classic mistake the knapsack formulation exists to fix.",
            ),
            (
                "The optimizer will find value the model missed.",
                "It allocates the value the model predicted. Garbage predictions produce a beautifully optimal allocation of garbage.",
            ),
        ),
        example=(
            "session.fit_decision_policy(",
            "    method='knapsack', value_column='expected_profit',",
            "    cost_column='contact_cost', budget=10_000.0,",
            ")",
            "selected = session.apply_decisions(partition='test')",
        ),
        check=(
            "Do your items have meaningfully different costs?",
            "Is your value column an expected value, or just a raw probability?",
        ),
        tools=("fit_decision_policy", "apply_decisions", "evaluate_decisions", "predict"),
        terms=("optimization", "expected value", "cost matrix", "threshold"),
        difficulty=ADVANCED,
    ),
    _layer(
        "decision-bundle-boundary",
        plain=(
            "A decision plan: the threshold, the cost matrix, or the allocation rule: saves as its own "
            "bundle. It is deliberately separate from the model, because the same model often serves "
            "several teams with different cost structures."
        ),
        analogy=(
            "The weather forecast is shared; each household's umbrella rule is their own. Bundling the rule "
            "into the forecast would force everyone to make the same choice."
        ),
        steps=(
            "Fit a decision policy so a plan exists.",
            "Call `save_decision_bundle(path)` to persist the rule and its parameters.",
            "Reload with `load_decision_bundle(path)` wherever the rule is applied.",
            "Apply it to fresh model scores with `apply_decisions`.",
            "Keep the model bundle and the checkpoint separately.",
        ),
        use=(
            "When one model feeds several teams with different cost structures.",
            "When the operating point needs its own review and approval cycle, separate from the model's.",
        ),
        avoid=(
            "Do not hard-code the threshold into application code; it becomes invisible and nobody re-reviews it.",
            "Do not apply a decision plan to scores from a different model without re-validating it.",
        ),
        myths=(
            (
                "The threshold belongs inside the model.",
                "Baking it in prevents different consumers from choosing different operating points on the same predictions.",
            ),
            (
                "A decision plan is trivial enough not to need versioning.",
                "It encodes a business cost judgement. When someone asks why alert volume tripled, the versioned plan is the answer.",
            ),
        ),
        example=(
            "session.save_decision_bundle('artifacts/collections-policy')",
            "app = Session.ingest(scores_frame).load_decision_bundle('artifacts/collections-policy')",
            "app.apply_decisions()",
        ),
        check=(
            "Who owns the cost numbers in your policy, and when were they last reviewed?",
            "Would two teams using this model want different operating points?",
        ),
        tools=("save_decision_bundle", "load_decision_bundle", "apply_decisions", "checkpoint_save"),
        terms=("bundle", "checkpoint", "threshold", "cost matrix"),
        difficulty=CORE,
    ),
)

__all__ = ["OPTIMIZE_BEGINNER"]
