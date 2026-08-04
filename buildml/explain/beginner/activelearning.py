# ruff: noqa: E501
"""Beginner layers for active learning."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

ACTIVELEARNING_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "activelearning-train-pool",
        plain=(
            "Active learning is about spending a limited labelling budget well. The pool it picks from is "
            "the unlabelled rows inside your training partition: rows whose target is blank. Validation "
            "and test rows are never candidates, because labelling them would consume the very data you "
            "need for an honest score."
        ),
        analogy=(
            "A student choosing which practice questions to ask the tutor about. They pick from the "
            "practice book, not from the sealed exam paper."
        ),
        steps=(
            "Put labelled and unlabelled rows in one table with blank targets for the unlabelled ones.",
            "Split as usual: the unlabelled pool lives inside the train partition.",
            "Fit an active learner on the labelled training rows.",
            "Ask for query suggestions; BuildML returns row indices from the training pool only.",
            "Label those rows externally, feed the labels back, and refit.",
        ),
        use=(
            "When labelling costs real money or expert time and you cannot label everything.",
            "When you have a large unlabelled backlog and need to decide what to send to annotators first.",
        ),
        avoid=(
            "Do not use it when labelling is cheap and fast; just label a random sample and move on.",
            "Do not let the pool include validation or test rows, even accidentally: you would be labelling your own exam.",
        ),
        myths=(
            (
                "Active learning can query any row in the dataset.",
                "It queries the training pool. Rows outside it are either already labelled or reserved for evaluation.",
            ),
            (
                "The labelled subset produced by active learning is a representative sample.",
                "It is deliberately biased toward difficult rows. That is the point, and it means you cannot use it to estimate class prevalence.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, random_state=0)",
            "session.active_learning.fit(estimator=LogisticRegression(max_iter=1000))",
            "indices = session.active_learning.suggest_query(n=20, strategy='margin')",
            "session.active_learning.label_rows({int(i): labels[i] for i in indices})",
        ),
        check=(
            "How many unlabelled rows are in your training partition?",
            "Could any suggested index point outside the training pool?",
        ),
        tools=("fit_active_learner", "suggest_query", "label_rows", "split"),
        terms=("active learning", "pseudo-label", "train", "semi-supervised"),
        difficulty=CORE,
    ),
    _layer(
        "activelearning-human-labels",
        plain=(
            "BuildML suggests which rows to label. It never makes up the labels. There is no built-in "
            "oracle, no auto-labelling, no silent guess: a human (or a test harness that stands in for "
            "one) provides the answers and hands them back."
        ),
        analogy=(
            "A research assistant marks the passages worth checking and brings them to you. They do not "
            "write your conclusions for you."
        ),
        steps=(
            "Call `session.active_learning.suggest_query` to get the row indices worth labelling next.",
            "Export those rows to whatever your annotation process is: a spreadsheet, a labelling tool, an expert review.",
            "Collect the real labels.",
            "Feed them back with `session.active_learning.label_rows`.",
            "Refit and repeat until the budget runs out or the score plateaus.",
        ),
        use=(
            "Whenever a genuine human labelling loop exists and you want to direct it.",
            "In tests, where a harness supplies known labels to simulate the loop deterministically.",
        ),
        avoid=(
            "Do not substitute model predictions for human labels and call it active learning: that is self-training, and it has quite different risks.",
            "Do not run the loop without recording who labelled what; label quality is part of your provenance.",
        ),
        myths=(
            (
                "Active learning automates labelling.",
                "It automates *prioritization*. The labelling itself is still a human cost, which is exactly the cost you are trying to spend wisely.",
            ),
            (
                "Any labeller will do since the model just needs a signal.",
                "Query strategies deliberately select ambiguous rows. Those are the hardest cases, so they need your *best* labellers, not your fastest.",
            ),
        ),
        example=(
            "indices = session.active_learning.suggest_query(n=25, strategy='least_confidence')",
            "batch = session.head(indices=indices)      # export for annotation",
            "# ... humans label the batch ...",
            "session.active_learning.label_rows(reviewed_labels)",
            "session.active_learning.fit(estimator=LogisticRegression(max_iter=1000))",
        ),
        check=(
            "Who is doing the labelling, and are the queried rows within their expertise?",
            "How are you recording label provenance and disagreement?",
        ),
        tools=("suggest_query", "label_rows", "fit_active_learner", "evaluate_active_learning"),
        terms=("active learning", "pseudo-label", "provenance", "target"),
        difficulty=CORE,
    ),
    _layer(
        "activelearning-uncertainty",
        plain=(
            "A query strategy is the rule for choosing which rows to label next. The simplest ones pick the "
            "rows the model is least sure about. Others pick rows that would most change the model, or rows "
            "that best cover the unexplored parts of the feature space."
        ),
        analogy=(
            "Revising for an exam. Uncertainty sampling means studying the topics you feel shakiest on. "
            "Coverage sampling means making sure you touch every chapter at least once. Both are reasonable; "
            "they fail differently."
        ),
        steps=(
            "Least-confidence picks rows whose top predicted probability is lowest.",
            "Margin picks rows where the top two classes are closest: usually the best default.",
            "Entropy picks rows whose whole probability distribution is flattest, which matters with many classes.",
            "Committee strategies train several models and pick rows they disagree about most.",
            "Coverage strategies such as CoreSet pick rows far from anything already labelled, guarding against blind spots.",
        ),
        use=(
            "Margin or entropy as a sensible default for most classification problems.",
            "Committee or coverage strategies when uncertainty sampling keeps picking near-duplicate rows.",
        ),
        avoid=(
            "Do not use uncertainty sampling with a badly calibrated model; its confidence numbers are the input to the whole strategy.",
            "Do not query one row at a time on large pools: batch queries and accept some redundancy, or you will refit forever.",
        ),
        myths=(
            (
                "The most uncertain rows are always the most valuable.",
                "Uncertainty concentrates on the decision boundary and can repeatedly select noise or mislabelled outliers. Coverage strategies exist because of this failure.",
            ),
            (
                "A more sophisticated strategy always beats random selection.",
                "Random sampling is a genuinely strong baseline. Always measure your strategy against it before assuming the complexity pays.",
            ),
        ),
        example=(
            "session.active_learning.suggest_query(n=20, strategy='margin')          # closest top-two",
            "session.active_learning.suggest_query(n=20, strategy='entropy')         # many classes",
            "session.active_learning.suggest_query(n=20, strategy='coreset')         # coverage",
            "session.active_learning.evaluate(partition='validation')",
        ),
        check=(
            "Does your strategy beat random selection at the same budget?",
            "Are the queried rows near-duplicates of each other?",
        ),
        tools=("suggest_query", "fit_active_learner", "evaluate_active_learning", "calibration"),
        terms=("active learning", "calibration", "predict_proba", "embedding"),
        difficulty=ADVANCED,
    ),
    _layer(
        "activelearning-bundle-boundary",
        plain=(
            "The active-learning state: the current model, which rows are still in the pool, and the "
            "history of what was queried: saves as its own bundle. That history matters: an active-learning "
            "run is a sequence, and resuming it needs the sequence."
        ),
        analogy=(
            "A research log listing which sources you have already checked. Losing it does not lose your "
            "findings, but it does mean re-checking things you already did."
        ),
        steps=(
            "Run at least one query-and-label round so a plan exists.",
            "Call `session.active_learning.save_bundle(path)` to store the model, the pool indices, and the query history.",
            "Reload with `session.active_learning.load_bundle(path)` to resume the loop.",
            "Continue querying from where you left off, without re-suggesting rows you already labelled.",
            "Keep checkpoints separately for the data state itself.",
        ),
        use=(
            "Whenever a labelling loop spans days or weeks and passes between people.",
            "When you need an audit trail of which rows were selected in which round and why.",
        ),
        avoid=(
            "Do not restart from scratch each session; you lose the pool bookkeeping and will re-query labelled rows.",
            "Do not assume a checkpoint preserves the query history: it does not embed the active-learning plan.",
        ),
        myths=(
            (
                "The pool is just 'rows with blank targets', so it can be recomputed.",
                "It can, but the query history cannot. Losing it loses your record of how the labelled set was built, which is part of why the model looks the way it does.",
            ),
            (
                "Only the model matters for resuming.",
                "Without the pool state you will suggest rows that are already labelled, wasting the budget you were trying to protect.",
            ),
        ),
        example=(
            "session.active_learning.save_bundle('artifacts/al-round-3')",
            "resumed = Session.ingest(frame).active_learning.load_bundle('artifacts/al-round-3')",
            "resumed.active_learning.suggest_query(n=20, strategy='margin')   # continues the sequence",
        ),
        check=(
            "Does your saved bundle know which rows have already been labelled?",
            "Could two people resume the same loop independently and collide?",
        ),
        tools=("save_active_learning_bundle", "load_active_learning_bundle", "suggest_query", "checkpoint_save"),
        terms=("bundle", "checkpoint", "active learning", "history"),
        difficulty=CORE,
    ),
)

__all__ = ["ACTIVELEARNING_BEGINNER"]
