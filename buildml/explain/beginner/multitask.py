# ruff: noqa: E501
"""Beginner layers for multi-task / multi-output modeling."""

from __future__ import annotations

from buildml.explain.beginner._builder import CORE, BeginnerLayer, _index, _layer

MULTITASK_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "multitask-multi-output",
        plain=(
            "Multi-task modeling predicts several targets at once from the same features. Instead of training three separate models for three questions, you train one that answers all three: with sharing determined by the selected method. Independent multi-output wrappers fit a separate estimator per target."
        ),
        analogy=(
            "One doctor who examines you once and reports on your heart, lungs, and blood pressure. Three "
            "separate appointments would repeat the same examination three times and never connect the findings."
        ),
        steps=(
            "Give two or more columns the target role.",
            "Choose an approach: wrap an ordinary estimator to run once per target, use a gradient-boosting model with native multi-target support, or use a neural network with a shared trunk and one head per task.",
            "Fit once on the training rows.",
            "Predict: you get one prediction per target per row.",
            "Evaluate per target, because the model can be excellent at one task and poor at another.",
        ),
        use=(
            "When your targets are genuinely related, so learning one helps the others.",
            "When maintaining several separate models is operationally painful and the accuracy cost of sharing is small.",
        ),
        avoid=(
            "Shared models can suffer negative transfer when targets conflict. Independent multi-output wrappers do not share fitted parameters.",
            "If one target has priority, compare its performance against a dedicated model and choose metrics that reflect that priority.",
        ),
        myths=(
            (
                "Multi-task learning always beats separate models.",
                "It helps when tasks share structure. When they do not, sharing can reduce performance, a phenomenon called negative transfer.",
            ),
            (
                "This is a deep-learning-only technique.",
                "Scikit-learn MultiOutput wrappers fit independent estimators behind one API. Other backends can share parameters; compare their per-target results.",
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
            'frame["renewed"] = 2 * frame.x1 + frame.x2',
            'frame["upsold"] = frame.x1 - 2 * frame.x2',
            'session = Session.ingest(frame).set_roles({"x1": "feature", "x2": "feature", "renewed": "target", "upsold": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, random_state=42)",
            'session.multitask.fit(backend="sklearn", method="multi_output", task="regression", base_estimator="ridge")',
            'print(session.multitask.evaluate(partition="validation").per_task_metrics)',
        ),
        check=(
            "Do your targets share the same drivers, or are they unrelated?",
            "Does each target do at least as well as it would in its own model?",
        ),
        tools=("fit_multitask", "predict_multitask", "evaluate_multitask", "set_roles"),
        terms=("multi-task", "target", "neural network", "gradient boosting"),
        difficulty=CORE,
    ),
    _layer(
        "multitask-chain",
        plain=(
            "A chain models dependence between targets on purpose. It predicts the first target, then feeds "
            "that prediction in as an extra feature when predicting the second, and so on down the chain. "
            "That makes the order of the chain a real modelling decision."
        ),
        analogy=(
            "A diagnosis sequence. Knowing whether someone has a fever changes what you look for next. The "
            "order you ask the questions in matters."
        ),
        steps=(
            "Decide the order: put targets you can predict most reliably first.",
            "The first estimator uses only the original features.",
            "Each subsequent estimator gets the original features plus the earlier predictions.",
            "Fit the whole chain on training rows.",
            "Evaluate per target, and try a different order to see how sensitive the result is.",
        ),
        use=(
            "When one target genuinely helps predict another: a diagnosis that implies a treatment, a stage that implies an outcome.",
            "For multi-label problems where labels co-occur in structured ways.",
        ),
        avoid=(
            "Do not chain independent targets; you add error propagation for no benefit.",
            "Do not use a chain when the first target is hard to predict: its errors contaminate everything downstream.",
        ),
        myths=(
            (
                "Chain order does not matter much.",
                "It matters a lot. A weak first link injects noise into every later estimator, and different orders can produce noticeably different results.",
            ),
            (
                "The chain uses the true earlier labels at prediction time.",
                "At training time it may; at prediction time only the *predicted* earlier values exist. That gap is where chains lose accuracy.",
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
            'frame["renewed"] = (frame.x1 > 0).astype(int)',
            'frame["upsold"] = (frame.x2 > 0).astype(int)',
            'session = Session.ingest(frame).set_roles({"x1": "feature", "x2": "feature", "renewed": "target", "upsold": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, random_state=42)",
            'session.multitask.fit(backend="sklearn", method="classifier_chain", task="classification", order=["renewed", "upsold"])',
            'print(session.multitask.evaluate(partition="validation").per_task_metrics)',
        ),
        check=(
            "Which of your targets is most reliably predictable? Is it first?",
            "How much does performance change if you reverse the order?",
        ),
        tools=("fit_multitask", "predict_multitask", "evaluate_multitask"),
        terms=("multi-task", "target", "estimator"),
        difficulty=CORE,
    ),
    _layer(
        "multitask-target-roles",
        plain=(
            "BuildML's classical `fit` requires exactly one target: that constraint keeps the ordinary path unambiguous. The multi-task path requires at least two. Assign the target roles and call the corresponding API explicitly."
        ),
        analogy=(
            "A form that accepts one answer per question versus one designed for multiple selections. "
            "Handing the wrong form to the wrong process produces a clear error rather than a quiet mess."
        ),
        steps=(
            "Count your target-role columns.",
            "Exactly one means classical `fit`, `evaluate`, `predict`, and everything built on them.",
            "Two or more means `session.multitask.fit` and its evaluate/predict pair.",
            "Switching between the two means changing roles with `set_roles`.",
            "If a classical operation errors about the target, check your role assignment first.",
        ),
        use=(
            "Whenever you are unsure why a classical operation is refusing to run.",
            "When moving a project from single-target to multi-target modelling.",
        ),
        avoid=(
            "Do not assign several targets and then expect `fit` to pick one; the ambiguity is refused deliberately.",
            "Do not mark a column as target just because it is an outcome you find interesting: an extra target changes which APIs are available.",
        ),
        myths=(
            (
                "Extra targets are ignored by single-target operations.",
                "They are not ignored; the operation refuses, because guessing which target you meant would be worse than an error.",
            ),
            (
                "Multi-task is only for advanced projects.",
                "Predicting two related business outcomes from one customer table is an entirely ordinary situation.",
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
            'frame["renewed"] = (frame.x1 > 0).astype(int)',
            'frame["upsold"] = (frame.x2 > 0).astype(int)',
            'session = Session.ingest(frame).set_roles({"x1": "feature", "x2": "feature", "renewed": "target", "upsold": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, random_state=42)",
            "from sklearn.linear_model import LogisticRegression",
            'session.set_roles({"upsold": "ignore"})',
            'session.fit(LogisticRegression(max_iter=500), task="classification")',
            'print(session.evaluate(partition="validation").metrics)',
            'session.set_roles({"upsold": "target"})',
            'session.multitask.fit(backend="sklearn", method="multi_output", task="classification")',
            'print(session.multitask.evaluate(partition="validation").per_task_metrics)',
        ),
        check=(
            "How many columns currently hold the target role?",
            "Which API family does that number put you in?",
        ),
        tools=("set_roles", "fit", "fit_multitask", "metadata"),
        terms=("role", "target", "multi-task"),
        difficulty=CORE,
    ),
    _layer(
        "multitask-bundle-boundary",
        plain=(
            "The fitted multi-task plan: every per-task estimator or the shared network, plus the task "
            "order and the feature contract: saves as its own bundle, separate from Session checkpoints."
        ),
        analogy=(
            "A folder holding all three specialists' notes and the order they were consulted in, filed "
            "separately from the patient's admission record."
        ),
        steps=(
            "Fit a multi-task model so a plan exists.",
            "Call `session.multitask.save_bundle(path)`.",
            "Reload with `session.multitask.load_bundle(path)`.",
            "Predict: you get every target back in the recorded order.",
            "Checkpoint separately if you also need the dataset state.",
        ),
        use=(
            "When several downstream systems consume different targets from the same model.",
            "When the chain order must be reproduced exactly at scoring time.",
        ),
        avoid=(
            "Do not rely on remembering the target order; the bundle records it because getting it wrong silently mislabels outputs.",
            "Do not expect a checkpoint to restore the multi-task plan.",
        ),
        myths=(
            (
                "Target order is cosmetic.",
                "For chains it is structural, and for every method it determines which output column means which target.",
            ),
            (
                "Multiple targets need multiple bundles.",
                "One plan covers all of them, which is precisely the operational benefit of multi-task modelling.",
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
            'frame["renewed"] = (frame.x1 > 0).astype(int)',
            'frame["upsold"] = (frame.x2 > 0).astype(int)',
            'session = Session.ingest(frame).set_roles({"x1": "feature", "x2": "feature", "renewed": "target", "upsold": "target"})',
            "session.split(test_size=0.2, validation_size=0.2, random_state=42)",
            'session.multitask.fit(backend="sklearn", method="multi_output", task="classification")',
            'session.multitask.save_bundle("artifacts/multiple-outcomes")',
            "new_frame = frame.iloc[:4].copy()",
            'job = Session.ingest(new_frame).multitask.load_bundle("artifacts/multiple-outcomes", trusted=True)',
            'print(job.multitask.predict(partition="all"))',
        ),
        check=(
            "Does your consuming system rely on target order or on target name?",
            "Which artifact restores the model, and which restores the data?",
        ),
        tools=(
            "save_multitask_bundle",
            "load_multitask_bundle",
            "predict_multitask",
            "checkpoint_save",
        ),
        terms=("bundle", "checkpoint", "multi-task", "plan"),
        difficulty=CORE,
    ),
)

__all__ = ["MULTITASK_BEGINNER"]
