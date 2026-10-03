# ruff: noqa: E501
"""Beginner layers for active learning."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

ACTIVELEARNING_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "activelearning-train-pool",
        plain=(
            "Active learning is about spending a limited labelling budget well. The pool it picks from is the unlabelled rows inside your training partition: rows whose target is blank. Validation and test rows are never candidates, because labelling them would consume the very data you need for held-out evaluation."
        ),
        analogy=(
            "A student choosing which practice questions to ask the tutor about. They pick from the "
            "practice book, not from the sealed exam paper."
        ),
        steps=(
            "Keep a labelled evaluation set separate from the training pool; unlabelled training rows have missing targets.",
            "Assign train, validation, and test indices explicitly when labels are incomplete. The example masks training labels after creating a stratified split.",
            "Fit an active learner on the labelled training rows.",
            "Ask for query suggestions; BuildML returns row indices from the training pool only.",
            "Label those rows externally, feed the labels back, and refit.",
        ),
        use=(
            "When labelling costs real money or expert time and you cannot label everything.",
            "When you have a large unlabelled backlog and need to decide what to send to annotators first.",
        ),
        avoid=(
            "Compare active selection with random sampling at the same annotation budget, particularly when labelling is inexpensive.",
            "Keep queried rows out of validation and test partitions to preserve independent evaluation.",
        ),
        myths=(
            (
                "Active learning can query any row in the dataset.",
                "It queries the training pool. Rows outside it are either already labelled or reserved for evaluation.",
            ),
            (
                "The labelled subset produced by active learning is a representative sample.",
                "Selection depends on the query strategy, so queried rows may not represent the population. Do not estimate class prevalence from them without accounting for that selection.",
            ),
        ),
        example=(
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from buildml import Session",
            "",
            "rng = np.random.default_rng(42)",
            'Path("artifacts").mkdir(exist_ok=True)',
            'frame = pd.DataFrame(rng.normal(size=(160, 2)), columns=["x1", "x2"])',
            'frame["label"] = (frame.x1 + frame.x2 > 0).astype(float)',
            'truth = frame["label"].copy()',
            'session = Session.ingest(frame).set_roles({"x1": "feature", "x2": "feature", "label": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=42)",
            "split = session.split_plan",
            "masked = frame.copy()",
            'masked.loc[list(split.train_indices)[::2], "label"] = np.nan',
            "session = Session.ingest(masked).set_roles(dict(session.dataset.roles))",
            "session.inject_split(train_indices=split.train_indices, validation_indices=split.validation_indices, test_indices=split.test_indices)",
            'session.active_learning.fit(backend="sklearn", base_estimator="logistic_regression", label_budget=40)',
            'indices = session.active_learning.suggest_query(batch_size=10, strategy="margin").indices',
            "# Example-only labels simulate an external annotation process.",
            "session.active_learning.label_rows(indices=indices, labels=[int(truth.loc[i]) for i in indices])",
            'print(session.active_learning.evaluate(partition="validation").metrics)',
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
            "BuildML suggests which rows to label. Your annotation process supplies the labels, and `session.active_learning.label_rows` records them. The example simulates annotation with labels saved before masking."
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
                "Query strategies deliberately select ambiguous rows. Review annotation guidelines and disagreements carefully, especially for ambiguous queries.",
            ),
        ),
        example=(
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from buildml import Session",
            "",
            "rng = np.random.default_rng(42)",
            'Path("artifacts").mkdir(exist_ok=True)',
            'frame = pd.DataFrame(rng.normal(size=(160, 2)), columns=["x1", "x2"])',
            'frame["label"] = (frame.x1 + frame.x2 > 0).astype(float)',
            'truth = frame["label"].copy()',
            'session = Session.ingest(frame).set_roles({"x1": "feature", "x2": "feature", "label": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=42)",
            "split = session.split_plan",
            "masked = frame.copy()",
            'masked.loc[list(split.train_indices)[::2], "label"] = np.nan',
            "session = Session.ingest(masked).set_roles(dict(session.dataset.roles))",
            "session.inject_split(train_indices=split.train_indices, validation_indices=split.validation_indices, test_indices=split.test_indices)",
            'session.active_learning.fit(backend="sklearn", base_estimator="logistic_regression", label_budget=40)',
            'indices = session.active_learning.suggest_query(batch_size=10, strategy="least_confidence").indices',
            "batch = session.to_pandas().iloc[list(indices)]",
            "# For this runnable simulation, use the labels saved before masking.",
            "# In an actual labeling workflow, replace these with reviewed annotations.",
            "reviewed_labels = [int(truth.iloc[i]) for i in indices]",
            "result = session.active_learning.label_rows(indices=indices, labels=reviewed_labels)",
            "print(result.n_newly_labeled, result.budget_remaining)",
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
            "Margin picks rows where the top two predicted class probabilities are closest.",
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
            "Use batch queries to reduce repeated fitting costs on large pools, and inspect batches for redundant rows.",
        ),
        myths=(
            (
                "The most uncertain rows are always the most valuable.",
                "Uncertainty concentrates on the decision boundary and can repeatedly select noise or mislabelled outliers. Coverage strategies exist because of this failure.",
            ),
            (
                "A more sophisticated strategy always beats random selection.",
                "Compare against random sampling at the same labelling budget before choosing a more complex strategy.",
            ),
        ),
        example=(
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from buildml import Session",
            "",
            "rng = np.random.default_rng(42)",
            'Path("artifacts").mkdir(exist_ok=True)',
            'frame = pd.DataFrame(rng.normal(size=(160, 2)), columns=["x1", "x2"])',
            'frame["label"] = (frame.x1 + frame.x2 > 0).astype(float)',
            'truth = frame["label"].copy()',
            'session = Session.ingest(frame).set_roles({"x1": "feature", "x2": "feature", "label": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=42)",
            "split = session.split_plan",
            "masked = frame.copy()",
            'masked.loc[list(split.train_indices)[::2], "label"] = np.nan',
            "session = Session.ingest(masked).set_roles(dict(session.dataset.roles))",
            "session.inject_split(train_indices=split.train_indices, validation_indices=split.validation_indices, test_indices=split.test_indices)",
            'session.active_learning.fit(backend="sklearn", base_estimator="logistic_regression", label_budget=40)',
            'session.active_learning.fit(backend="sklearn", base_estimator="logistic_regression", strategy="committee", label_budget=40)',
            'for strategy in ("margin", "entropy", "committee"):',
            "    query = session.active_learning.suggest_query(batch_size=10, strategy=strategy)",
            "    print(strategy, query.indices)",
            'print(session.active_learning.evaluate(partition="validation").metrics)',
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
            "Fit an active learner to create a plan, and save it after any completed labelling rounds.",
            "Call `session.active_learning.save_bundle(path)` to store the model, the pool indices, and the query history.",
            "Restore the labelled table and the same split, then load a trusted bundle with `session.active_learning.load_bundle(path, trusted=True)`.",
            "Continue querying from where you left off, without re-suggesting rows you already labelled.",
            "Keep checkpoints separately for the data state itself.",
        ),
        use=(
            "Whenever a labelling loop spans days or weeks and passes between people.",
            "When you need an audit trail of which rows were selected in which round and why.",
        ),
        avoid=(
            "Keep the updated labels, split, and active-learning bundle together when resuming; rebuilding from an earlier unlabelled table can repeat annotation work.",
            "Do not assume a checkpoint preserves the query history: it does not embed the active-learning plan.",
        ),
        myths=(
            (
                "The pool is just 'rows with blank targets', so it can be recomputed.",
                "It can, but the query history cannot. Losing it loses your record of how the labelled set was built, which is part of why the model looks the way it does.",
            ),
            (
                "Only the model matters for resuming.",
                "The updated labels determine pool membership, while the bundle preserves query history and budget state. Restore both to continue the same run.",
            ),
        ),
        example=(
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "from buildml import Session",
            "",
            "rng = np.random.default_rng(42)",
            'Path("artifacts").mkdir(exist_ok=True)',
            'frame = pd.DataFrame(rng.normal(size=(160, 2)), columns=["x1", "x2"])',
            'frame["label"] = (frame.x1 + frame.x2 > 0).astype(float)',
            'truth = frame["label"].copy()',
            'session = Session.ingest(frame).set_roles({"x1": "feature", "x2": "feature", "label": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=42)",
            "split = session.split_plan",
            "masked = frame.copy()",
            'masked.loc[list(split.train_indices)[::2], "label"] = np.nan',
            "session = Session.ingest(masked).set_roles(dict(session.dataset.roles))",
            "session.inject_split(train_indices=split.train_indices, validation_indices=split.validation_indices, test_indices=split.test_indices)",
            'session.active_learning.fit(backend="sklearn", base_estimator="logistic_regression", label_budget=40)',
            "indices = session.active_learning.suggest_query(batch_size=5).indices",
            "session.active_learning.label_rows(indices=indices, labels=[int(truth.loc[i]) for i in indices])",
            'session.active_learning.save_bundle("artifacts/active-learning")',
            "resumed = Session.ingest(session.to_pandas()).set_roles(dict(session.dataset.roles))",
            "resumed.inject_split(train_indices=split.train_indices, validation_indices=split.validation_indices, test_indices=split.test_indices)",
            'resumed.active_learning.load_bundle("artifacts/active-learning", trusted=True)',
            "print(resumed.active_learning.suggest_query(batch_size=5).indices)",
        ),
        check=(
            "Does your saved bundle know which rows have already been labelled?",
            "Could two people resume the same loop independently and collide?",
        ),
        tools=(
            "save_active_learning_bundle",
            "load_active_learning_bundle",
            "suggest_query",
            "checkpoint_save",
        ),
        terms=("bundle", "checkpoint", "active learning", "history"),
        difficulty=CORE,
    ),
)

__all__ = ["ACTIVELEARNING_BEGINNER"]
