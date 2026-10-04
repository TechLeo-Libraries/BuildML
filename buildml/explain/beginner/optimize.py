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
        myths=(('Optimizing the metric optimizes the decision.', 'F1 combines precision and recall; it does not directly minimize a supplied error-cost matrix. If operational costs matter, evaluate decisions using those costs.'), ('The threshold is a modelling detail.', 'The threshold affects error costs and workload. Choose it with the people responsible for those consequences.')),
        example=(
            'import pandas as pd',
            'from sklearn.datasets import make_classification',
            'from sklearn.linear_model import LogisticRegression',
            'from buildml import Session',
            '',
            'x, y = make_classification(n_samples=240, n_features=6, n_informative=4, random_state=0)',
            "frame = pd.DataFrame(x, columns=[f'x{i}' for i in range(6)])",
            "frame['target'] = y",
            "session = Session.ingest(frame).set_roles({**{f'x{i}': 'feature' for i in range(6)}, 'target': 'target'})",
            'session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)',
            "session.fit(LogisticRegression(max_iter=500), task='classification')",
            "session.decision.fit(method='threshold', partition='validation', fp_cost=1.0, fn_cost=25.0, backend='native')",
            "decisions = session.decision.apply(partition='test')",
            "print(session.decision.evaluate(partition='test').metrics)",
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
        use=('Multiclass problems where the consequences of different confusions differ substantially.', 'Multiclass classification where each available action corresponds to one predicted class. Additional review actions need a separate decision design.'),
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
                "Some decision systems support extra actions, but BuildML's cost_matrix method requires a square matrix with one action per class.",
            ),
        ),
        example=(
            'import pandas as pd',
            'from sklearn.datasets import make_classification',
            'from sklearn.linear_model import LogisticRegression',
            'from buildml import Session',
            '',
            'x, y = make_classification(n_samples=240, n_features=6, n_informative=4, random_state=0)',
            "frame = pd.DataFrame(x, columns=[f'x{i}' for i in range(6)])",
            "frame['target'] = y",
            "session = Session.ingest(frame).set_roles({**{f'x{i}': 'feature' for i in range(6)}, 'target': 'target'})",
            'session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)',
            "session.fit(LogisticRegression(max_iter=500), task='classification')",
            '# Rows are true classes; columns are predicted classes, in class_labels order.',
            "session.decision.fit(method='cost_matrix', partition='validation',",
            "                     cost_matrix=[[0, 1], [25, 0]], class_labels=['0', '1'], backend='native')",
            "print(session.decision.apply(partition='test').n_rows)",
        ),
        check=("Are your model's probabilities calibrated enough to multiply by money?", 'Does each matrix row and column use the same class order as the model probabilities?'),
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
        steps=('Decide what you are allocating: a count (top-K), a budget with per-item costs (knapsack), or divisible shares (linear programming).', 'Provide the value score per row: usually a model prediction, possibly an expected value.', 'Provide the cost or weight per row when items are not equally expensive.', 'State the constraint: how many, how much money, how much capacity.', 'Read the selected set and solver disclosures, then compare realized value with alternatives. Native knapsack can use a greedy approximation, so optimality is not guaranteed for every configuration.'),
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
                "It allocates resources using the supplied values. Inaccurate value estimates can lead to poor decisions even when the optimization is solved correctly.",
            ),
        ),
        example=(
            'import numpy as np',
            'import pandas as pd',
            'from buildml import Session',
            '',
            'rng = np.random.default_rng(0)',
            "frame = pd.DataFrame({'expected_profit': rng.uniform(5, 40, 100), 'contact_cost': rng.uniform(1, 8, 100)})",
            "session = Session.ingest(frame).set_roles({'expected_profit': 'feature', 'contact_cost': 'feature'})",
            'session.split(test_size=0.2, validation_size=0.2, random_state=0)',
            "session.decision.fit(method='knapsack', partition='validation', backend='native',",
            "                     score_source='column', value_column='expected_profit',",
            "                     cost_column='contact_cost', budget=30.0)",
            "selected = session.decision.apply(partition='test')",
            'print(selected)',
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
            "Call `session.decision.save_bundle(path)` to persist the rule and its parameters.",
            "Reload with `session.decision.load_bundle(path)` wherever the rule is applied.",
            "Apply it to fresh model scores with `session.decision.apply`.",
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
            'import pandas as pd',
            'from sklearn.datasets import make_classification',
            'from sklearn.linear_model import LogisticRegression',
            'from buildml import Session',
            '',
            'x, y = make_classification(n_samples=240, n_features=6, n_informative=4, random_state=0)',
            "frame = pd.DataFrame(x, columns=[f'x{i}' for i in range(6)])",
            "frame['target'] = y",
            "session = Session.ingest(frame).set_roles({**{f'x{i}': 'feature' for i in range(6)}, 'target': 'target'})",
            'session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)',
            "session.fit(LogisticRegression(max_iter=500), task='classification')",
            "session.decision.fit(method='threshold', partition='validation', fp_cost=1.0, fn_cost=25.0, backend='native')",
            'from tempfile import TemporaryDirectory',
            '',
            '# This plan uses model probabilities, so retain the fitted model when loading it.',
            'with TemporaryDirectory() as directory:',
            "    session.decision.save_bundle(directory + '/policy')",
            "    session.decision.load_bundle(directory + '/policy', trusted=True)",
            "    print(session.decision.apply(partition='test'))",
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
