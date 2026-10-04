# ruff: noqa: E501
"""Beginner layers for symbolic and neuro-symbolic modeling."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

SYMBOLIC_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "symbolic-rules",
        plain=(
            "Symbolic modeling here means plain if-then rules over your columns: 'if age is over 65 and "
            "claims in the last year is over 3, then flag for review'. They are compiled into an ordered "
            "list, evaluated in sequence, and every prediction comes with a trace of which rule decided it."
        ),
        analogy=(
            "A company policy document. Anyone can read it, argue with a specific clause, and see exactly "
            "which clause applied to their case."
        ),
        steps=(
            "Write rules as conditions over column values, each with an outcome.",
            "Order matters: the list is evaluated top to bottom and the first match wins.",
            "Provide a default outcome for rows no rule covers.",
            "Predict: you get the outcome plus the identity of the rule that produced it.",
            "Evaluate on held-out rows exactly as you would any other model.",
        ),
        use=(
            "When decisions must be explainable to a regulator, an auditor, or an affected person.",
            "When domain experts already have codified rules and you want to measure them properly.",
        ),
        avoid=('Complex interactions can make rule lists difficult to maintain. Compare with an appropriate model and review whether the rules remain understandable.', 'Do not use them when the pattern shifts frequently: every change means a human editing the list.'),
        myths=(('Rules are obsolete now that we have machine learning.', 'They can be useful when decisions must follow explicit policy or when users need to inspect individual conditions. Measure their coverage and accuracy.'), ('This is symbolic artificial intelligence in the classical research sense.', 'It is column predicates compiled into an ordered list with traces. There is no theorem prover and no general reasoner.')),
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
            "rules = [{'rule_id': 'high_x0', 'if': [{'column': 'x0', 'op': '>', 'value': 1.0}], 'then': 1}]",
            "session.symbolic.fit(source='declared', task='classification', rules=rules, default_consequent=0)",
            "print(session.symbolic.evaluate(partition='test').metrics)",
        ),
        check=(
            "What fraction of your rows fall through to the default?",
            "Would a domain expert recognize and endorse every clause?",
        ),
        tools=("fit_symbolic", "predict_symbolic", "evaluate_symbolic"),
        terms=("symbolic AI", "decision tree", "feature", "model"),
        difficulty=CORE,
    ),
    _layer(
        "symbolic-induction",
        plain=(
            'There are two ways to get a rule list. You can write it yourself from domain knowledge, but rules can still use unavailable features or be tuned after inspecting holdout results. Or you can induce it from data with a decision tree or a decision-list algorithm, in which case it is a fitted model bound by the usual train-only discipline.'
        ),
        analogy=(
            "A policy written by the committee versus one reverse-engineered from what the committee "
            "actually decided last year. Both produce a rule book; only one is an empirical claim."
        ),
        steps=(
            "Declared rules go in as-is. Their performance is measured, not learned.",
            "Induced rules are fitted on training rows only.",
            "A decision tree is converted into a rule list by reading each root-to-leaf path as one clause.",
            "A decision-list algorithm learns clauses greedily, each covering the rows the earlier ones missed.",
            "Either way, evaluate on held-out rows.",
        ),
        use=(
            "Declared rules when the policy is the ground truth and you want to audit its consequences.",
            "Induced rules when you want an explainable model and are prepared to review each learned clause.",
        ),
        avoid=('Do not induce rules on the full dataset and then evaluate on part of it; induction is fitting.', 'Do not accept an induced rule list without reading it: induction can produce spurious or impractical conditions. Check both held-out performance and domain suitability.'),
        myths=(('Induced rules are as trustworthy as expert rules because they came from data.', "They came from *this* data, including its quirks. A rule saying 'if customer_id > 90000 then churn' is a real pattern and complete nonsense."), ('Declared rules cannot overfit.', 'Manually written rules can overfit when revised to match inspected examples or holdout outcomes. Evaluate the final rules on independent data.')),
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
            "session.symbolic.fit(source='decision_tree', task='classification', max_depth=3, random_state=0)",
            'for rule in session.symbolic.plan.knowledge_base.rules:',
            '    print(rule.to_dict())',
        ),
        check=(
            "Have you read every induced rule and sanity-checked it?",
            "Does any rule condition mention a column that would not exist at prediction time?",
        ),
        tools=("fit_symbolic", "predict_symbolic", "evaluate_symbolic", "feature_importance"),
        terms=("symbolic AI", "decision tree", "leakage", "overfitting"),
        difficulty=CORE,
    ),
    _layer(
        "symbolic-traces",
        plain=(
            "Every symbolic prediction comes with its receipt: which rules matched the row, and which one "
            "actually produced the answer. That is the whole reason to use rules: you can point at the "
            "clause and defend or change it."
        ),
        analogy=(
            "An itemized bill instead of a total. When someone disputes it, you can point at the exact line."
        ),
        steps=(
            "Predict as usual with `session.symbolic.predict`.",
            "The result carries the fired rule identifiers per row.",
            "It also identifies the deciding rule: the first match in the ordered list.",
            "Rows that matched nothing are marked as falling through to the default.",
            "Aggregate across rows to see which rules carry the workload and which never fire.",
        ),
        use=(
            "Whenever a decision needs to be explained to someone affected by it.",
            "For maintenance: a rule that never fires is dead weight, and one that fires on everything is probably too broad.",
        ),
        avoid=(
            "Do not treat the trace as a causal explanation; it explains the decision procedure, not the world.",
            "Do not ignore fall-through rows: a high default rate means your rule set does not actually cover your data.",
        ),
        myths=(
            (
                "A trace explains why the prediction is correct.",
                "It explains why the system produced that output. Whether the rule itself is right is a separate question.",
            ),
            (
                "All matched rules contributed to the answer.",
                "In an ordered list only the first match decides. The others matched and were skipped, which is worth knowing when clauses conflict.",
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
            "session.symbolic.fit(source='decision_tree', task='classification', max_depth=3, random_state=0)",
            "result = session.symbolic.predict(partition='test', return_traces=True)",
            'print(result.predictions[:5])',
            'print(result.traces[0].fired_rule_ids)',
            "print(session.symbolic.evaluate(partition='test').metrics)",
        ),
        check=(
            "Which of your rules never fires on real data?",
            "How often do two rules match the same row with different outcomes?",
        ),
        tools=("predict_symbolic", "fit_symbolic", "evaluate_symbolic"),
        terms=("symbolic AI", "feature importance", "model"),
        difficulty=CORE,
    ),
    _layer(
        "neuro-symbolic-hybrid",
        plain=(
            "Neuro-symbolic modeling combines a statistical model with explicit rules in one place. The "
            "rules can override the model in specific situations, feed in as extra features, or repair "
            "predictions that violate a hard constraint."
        ),
        analogy=(
            "An experienced assessor who uses judgement most of the time but always defers to the hard "
            "legal limits. Neither pure intuition nor pure rulebook."
        ),
        steps=(
            "Overlay mode: the model predicts, and a matching rule replaces the answer where it applies.",
            "Rules-as-features mode: each rule becomes a yes/no column the model can learn to use.",
            "Constraint-repair mode: the model predicts freely and violations are corrected afterwards.",
            "Pick the mode that matches whether the rules are hints or hard requirements.",
            "Evaluate the combination end to end: a well-intended override can make things worse.",
        ),
        use=(
            "When some cases have non-negotiable answers regardless of what the data suggests.",
            "When expert knowledge covers situations your training data barely contains.",
        ),
        avoid=(
            "Do not use overlay mode with soft, heuristic rules; you are overriding a fitted model with a guess.",
            "Do not layer many interacting rules on top of a model without measuring: the combination becomes as hard to reason about as either part alone.",
        ),
        myths=(('Adding expert rules can only help.', 'An incorrect override can replace a correct model prediction. Compare the model and combined system on the same rows, including the subset affected by rules.'), ('Rules-as-features and overlay achieve the same thing.', 'As features, the model *learns* how much to trust each rule. As an overlay, the rule wins unconditionally. Very different behaviours.')),
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
            "rules = [{'rule_id': 'high_x0', 'if': [{'column': 'x0', 'op': '>', 'value': 1.0}], 'then': 1}]",
            "rules[0].update(hardness='hard', kind='constraint', priority=100)",
            "session.symbolic.fit_neuro(mode='constraint_overlay', backend='sklearn', rules=rules,",
            "                           rule_source='declared', base_estimator='logistic_regression', task='classification')",
            "print(session.symbolic.evaluate_neuro(partition='test').metrics)",
        ),
        check=(
            "On what fraction of rows does a rule override the model?",
            "Is the combined score better than the model alone on validation?",
        ),
        tools=("fit_neuro_symbolic", "predict_neuro_symbolic", "evaluate_neuro_symbolic", "fit_symbolic"),
        terms=("neuro-symbolic", "symbolic AI", "model", "feature"),
        difficulty=ADVANCED,
    ),
    _layer(
        "symbolic-bundle-boundary",
        plain=(
            "Rule lists and neuro-symbolic plans save as symbolic bundles: the rules themselves, their "
            "order, the default, and any attached model. Session checkpoints do not embed them."
        ),
        analogy=(
            "The policy document is version-controlled separately from the case files it was applied to. "
            "Both need history; they are not the same history."
        ),
        steps=(
            "Fit a symbolic or neuro-symbolic plan.",
            "Call `session.symbolic.save_bundle(path)`: rule text, order, and default all travel together.",
            "Reload with `session.symbolic.load_bundle(path)`.",
            "Predict with traces intact.",
            "Keep checkpoints separately for the data.",
        ),
        use=(
            "When the rule set is governed and every version needs to be recoverable for audit.",
            "When rules and a model must be deployed as one unit so the override behaviour is preserved.",
        ),
        avoid=(
            "Do not maintain the rules in a spreadsheet and the bundle separately; they will diverge and the bundle is what actually runs.",
            "Do not reorder rules at load time: order is semantics, not presentation.",
        ),
        myths=(
            (
                "Rules are just configuration, not model state.",
                "Their order determines the output, and induced rules were fitted from data. Both make them model state.",
            ),
            (
                "A checkpoint would capture the rules since they are small.",
                "Size is not the criterion. Checkpoints hold data workflow state; domain plans have their own bundles so load-time contracts can be enforced.",
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
            "session.symbolic.fit(source='decision_tree', task='classification', max_depth=3, random_state=0)",
            'from tempfile import TemporaryDirectory',
            '',
            'with TemporaryDirectory() as directory:',
            "    session.symbolic.save_bundle(directory + '/rules')",
            "    audit = Session.ingest(frame.drop(columns='target')).symbolic.load_bundle(directory + '/rules', trusted=True)",
            "    print(audit.symbolic.predict(partition='all').predictions[:5])",
        ),
        check=(
            "Can you reproduce the exact rule set that ran last quarter?",
            "Does your bundle include the attached model for neuro-symbolic modes?",
        ),
        tools=("save_symbolic_bundle", "load_symbolic_bundle", "predict_symbolic", "checkpoint_save"),
        terms=("bundle", "checkpoint", "symbolic AI", "neuro-symbolic"),
        difficulty=CORE,
    ),
)

__all__ = ["SYMBOLIC_BEGINNER"]
