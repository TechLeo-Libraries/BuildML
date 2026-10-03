# ruff: noqa: E501
"""Beginner layers for semi-supervised learning."""

from __future__ import annotations

from buildml.explain.beginner._builder import CORE, BeginnerLayer, _index, _layer

SEMISUPERVISED_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "semisupervised-label-missingness",
        plain=(
            "Semi-supervised learning is for the very common situation where you have a few thousand labelled rows and a few million unlabelled ones. In BuildML you express 'unlabelled' the natural way: leave the target blank. The same target role applies to labelled and unlabelled rows."
        ),
        analogy=(
            "A stack of exam papers where only some have been marked. The unmarked ones are still papers: "
            "you do not put them in a different building, you just note that they have no grade yet."
        ),
        steps=('Put your labelled and unlabelled rows in the same table.', 'Leave the target blank (NaN) for the unlabelled ones.', 'Assign roles as usual; the target column is still the target.', "BuildML maps blanks to scikit-learn's internal -1 convention behind the scenes.", 'Reserve labelled validation and test rows, and keep the unlabelled pool in training. Use explicit split indices when missing labels prevent stratification.'),
        use=(
            "When labelling is expensive but raw data is plentiful: medical imaging, moderation, industrial inspection.",
            "When you already have a big unlabelled backlog sitting in the same system as your labelled sample.",
        ),
        avoid=(
            "Do not use it when a blank target means something other than 'not yet labelled': for example, 'this event has not happened yet'.",
            "Do not mix truly-missing targets and deliberately-unlabelled targets in the same column without a way to tell them apart.",
        ),
        myths=(
            (
                "Unlabelled rows need a separate dataset or a special role.",
                "They are ordinary rows with a blank target. Keeping them in one table is what lets the algorithm use their feature values.",
            ),
            (
                "The -1 convention is something you have to encode yourself.",
                "BuildML handles the translation. You work in blanks; the adapter deals with scikit-learn's convention.",
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
            'import numpy as np',
            '',
            '# Simulate scarce labels on training rows; validation and test stay labeled.',
            'split = session.split_plan',
            'rng = np.random.default_rng(0)',
            'hidden = rng.choice(list(split.train_indices), size=len(split.train_indices) // 2, replace=False)',
            'masked = frame.copy()',
            "masked.loc[hidden, 'target'] = np.nan",
            'session = Session.ingest(masked).set_roles(session.dataset.roles)',
            'session.inject_split(train_indices=split.train_indices, test_indices=split.test_indices,',
            '                     validation_indices=split.validation_indices)',
            "session.scale(method='standard')",
            "session.semisupervised.fit(method='label_propagation')",
            "print(session.semisupervised.evaluate(partition='test').metrics)",
        ),
        check=(
            "What fraction of your training rows have a blank target?",
            "Does a blank in your target column always mean 'not labelled yet'?",
        ),
        tools=("fit_semisupervised", "set_roles", "split", "predict_semisupervised"),
        terms=("semi-supervised", "target", "missing value", "pseudo-label"),
        difficulty=CORE,
    ),
    _layer(
        "semisupervised-train-only-fit",
        plain=(
            "Semi-supervised methods spread label information from labelled rows to similar unlabelled ones. "
            "All of that happens inside the training partition. The plan is then frozen and evaluated on "
            "labelled holdout rows: you never invent labels for the rows you are scoring against."
        ),
        analogy=(
            "A study group where a few members know the answers and explain them to the others. That is "
            "fine. Handing the group the exam paper to work through together is not."
        ),
        steps=(
            "Split so labelled holdout rows exist for evaluation.",
            "Fit on training rows, labelled and unlabelled together.",
            "The method propagates labels through feature-space similarity, or self-trains by iteratively adding its own confident predictions.",
            "Freeze the resulting plan.",
            "Evaluate on labelled validation or test rows the fit never touched.",
        ),
        use=(
            "When your labelled sample alone is too small to train a stable model.",
            "When you have reason to believe similar feature values imply similar labels: which is the core assumption.",
        ),
        avoid=(
            "Do not use it when your unlabelled pool comes from a different population than your labelled sample; propagation will spread the wrong labels confidently.",
            "Do not let unlabelled rows from validation or test into the fit, even though they carry no labels: their feature values still shape the propagation.",
        ),
        myths=(('Unlabelled data is free information.', 'It helps only when the similarity assumption holds. If the assumption is unsuitable, unlabelled data can reduce accuracy. Compare against fitting only the labelled training rows.'), ("Self-training's own predictions are as good as real labels.", 'They are its current beliefs. Feeding them back can reinforce an early mistake into a confident systematic error.')),
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
            'import numpy as np',
            '',
            '# Simulate scarce labels on training rows; validation and test stay labeled.',
            'split = session.split_plan',
            'rng = np.random.default_rng(0)',
            'hidden = rng.choice(list(split.train_indices), size=len(split.train_indices) // 2, replace=False)',
            'masked = frame.copy()',
            "masked.loc[hidden, 'target'] = np.nan",
            'session = Session.ingest(masked).set_roles(session.dataset.roles)',
            'session.inject_split(train_indices=split.train_indices, test_indices=split.test_indices,',
            '                     validation_indices=split.validation_indices)',
            "session.scale(method='standard')",
            "session.semisupervised.fit(method='self_training', threshold=0.8, max_iter=10)",
            "print(session.semisupervised.evaluate(partition='validation').metrics)",
        ),
        check=(
            "Does the semi-supervised model beat a plain model trained on the labelled rows only?",
            "Are your labelled and unlabelled rows drawn from the same population?",
        ),
        tools=("fit_semisupervised", "evaluate_semisupervised", "predict_semisupervised"),
        terms=("semi-supervised", "pseudo-label", "leakage", "train"),
        difficulty=CORE,
    ),
    _layer(
        "semisupervised-vs-novelty",
        plain=(
            "Two BuildML paths both work with partly-unlabelled data and they solve different problems. "
            "Semi-supervised learning is classification with scarce labels: you want a category. Novelty "
            "detection is anomaly detection fitted on known-clean rows: you want a strangeness score."
        ),
        analogy=(
            "Sorting mail into departments when only some envelopes are pre-sorted, versus spotting the one "
            "envelope that does not belong in the building at all. Different jobs, different tools."
        ),
        steps=(
            "Ask what your output needs to be: a class label, or an unusualness score?",
            "If you need class labels and have some, use `session.semisupervised.fit`.",
            "If you need to flag the unfamiliar and can identify clean rows, use `session.anomaly.fit(mode='novelty')`.",
            "Note that they take different inputs: one needs a partly-labelled target, the other needs a certified-clean subset.",
            "Do not chain them casually: pseudo-labels from one are not clean training rows for the other.",
        ),
        use=(
            "Semi-supervised when the categories exist and you are short on examples of them.",
            "Novelty when the interesting cases are by definition ones you have never seen.",
        ),
        avoid=(
            "Do not use novelty detection as a substitute for a classifier when you have hundreds of labelled positives.",
            "Do not use semi-supervised learning to find unknown categories; it propagates the labels you already have.",
        ),
        myths=(
            (
                "Both are 'learning without labels', so either will do.",
                "One assigns known categories, the other measures distance from normal. Their outputs are not interchangeable and neither are their assumptions.",
            ),
            (
                "Novelty detection will find the rare class for me.",
                "It finds unusual rows. Your rare class may be perfectly ordinary-looking, and plenty of ordinary rows are unusual.",
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
            'import numpy as np',
            '',
            '# Simulate scarce labels on training rows; validation and test stay labeled.',
            'split = session.split_plan',
            'rng = np.random.default_rng(0)',
            'hidden = rng.choice(list(split.train_indices), size=len(split.train_indices) // 2, replace=False)',
            'masked = frame.copy()',
            "masked.loc[hidden, 'target'] = np.nan",
            'session = Session.ingest(masked).set_roles(session.dataset.roles)',
            'session.inject_split(train_indices=split.train_indices, test_indices=split.test_indices,',
            '                     validation_indices=split.validation_indices)',
            "session.scale(method='standard')",
            "session.semisupervised.fit(method='label_propagation')",
            "print(session.semisupervised.evaluate(partition='test').metrics)",
            '# A separate task: estimate unusualness without using category labels.',
            "session.anomaly.fit(method='one_class_svm', mode='novelty')",
            "print(session.anomaly.score(partition='test'))",
        ),
        check=(
            "Does your downstream process consume a class name or a score?",
            "Do you know in advance what the interesting cases look like?",
        ),
        tools=("fit_semisupervised", "fit_anomaly", "evaluate_semisupervised", "evaluate_anomaly"),
        terms=("semi-supervised", "anomaly detection", "pseudo-label", "supervised"),
        difficulty=CORE,
    ),
    _layer(
        "semisupervised-bundle-boundary",
        plain=(
            "The fitted semi-supervised plan saves as its own bundle, holding the estimator, the feature "
            "contract, and the propagation settings. Session checkpoints do not contain it."
        ),
        analogy=(
            "The finished sorting rules are a separate document from the pile of mail you sorted. Keeping "
            "the pile does not preserve the rules."
        ),
        steps=(
            "Fit a semi-supervised model so a plan exists.",
            "Call `session.semisupervised.save_bundle(path)`.",
            "Reload with `session.semisupervised.load_bundle(path)` in a new Session.",
            "Confirm the feature columns match, then predict.",
            "Checkpoint separately if you also need the partly-labelled frame back.",
        ),
        use=(
            "When the model is going into a job that labels new rows on a schedule.",
            "When the propagation settings need to be reproduced exactly for a later comparison.",
        ),
        avoid=('The bundle is a fitted-model artifact, not a dataset export. Its estimator may retain training features or inferred labels, depending on the method.', 'Do not swap in a different feature set at load time and assume the propagation still means the same thing.'),
        myths=(('The bundle stores the pseudo-labels it generated.', 'It stores the fitted estimator, which can retain inferred labels as internal state. Use the prediction API to obtain labels in the documented result format.'), ('One checkpoint covers all my artifacts.', 'Checkpoints hold data workflow state. Each domain plan has its own bundle so load-time contracts can be enforced properly.')),
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
            'import numpy as np',
            '',
            '# Simulate scarce labels on training rows; validation and test stay labeled.',
            'split = session.split_plan',
            'rng = np.random.default_rng(0)',
            'hidden = rng.choice(list(split.train_indices), size=len(split.train_indices) // 2, replace=False)',
            'masked = frame.copy()',
            "masked.loc[hidden, 'target'] = np.nan",
            'session = Session.ingest(masked).set_roles(session.dataset.roles)',
            'session.inject_split(train_indices=split.train_indices, test_indices=split.test_indices,',
            '                     validation_indices=split.validation_indices)',
            "session.scale(method='standard')",
            "session.semisupervised.fit(method='label_propagation')",
            'from tempfile import TemporaryDirectory',
            '',
            'with TemporaryDirectory() as directory:',
            "    session.semisupervised.save_bundle(directory + '/semisup')",
            '    # The bundle expects the scaled features used at fit time.',
            "    new_rows = session.to_pandas().drop(columns='target')",
            "    job = Session.ingest(new_rows).semisupervised.load_bundle(directory + '/semisup', trusted=True)",
            "    print(job.semisupervised.predict(partition='all').predictions[:5])",
        ),
        check=(
            "Does your reload path apply the same preprocessing the plan was fitted under?",
            "Which artifact holds the data, and which holds the model?",
        ),
        tools=("save_semisupervised_bundle", "load_semisupervised_bundle", "predict_semisupervised", "checkpoint_save"),
        terms=("bundle", "checkpoint", "plan", "semi-supervised"),
        difficulty=CORE,
    ),
    _layer(
        "semisupervised-ssl-pipeline",
        plain=(
            'Self-supervised pretraining and semi-supervised learning fit together neatly. First learn a compact representation from training features without using labels, then run label propagation in that representation instead of the raw feature space.'
        ),
        analogy=(
            "Before sorting a library by topic, you first learn what makes books similar at all. Sorting is "
            "far easier once you have a good sense of similarity."
        ),
        steps=('Call `session.ssl.fit_pretext` to learn an encoder from training features: labels are ignored entirely here.', 'Call `session.ssl.transform` to turn rows into embeddings.', 'Run `session.semisupervised.fit` on the embeddings and check whether their geometry improves held-out classification.', 'Freeze both stages and evaluate on labelled holdout rows.', 'Compare against propagation on raw features; the extra stage has to earn its place.'),
        use=(
            "When your raw features are high-dimensional, noisy, or on wildly different scales, so raw-space similarity is unreliable.",
            "When you have a large unlabelled pool that pretraining can exploit.",
        ),
        avoid=(
            "Do not add the pretraining stage on a small, clean, low-dimensional table; it adds complexity for little gain.",
            "Do not pretrain on rows from validation or test: the encoder is fitted information like any other.",
        ),
        myths=(
            (
                "Pretraining always improves downstream results.",
                "It helps when the raw geometry is poor and the unlabelled pool is large. On tidy tabular data it frequently changes nothing.",
            ),
            (
                "Ignoring labels during pretraining means it cannot leak.",
                "The encoder still learns from data. Fitting it on all partitions leaks structure into your evaluation.",
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
            'import numpy as np',
            '',
            '# Simulate scarce labels on training rows; validation and test stay labeled.',
            'split = session.split_plan',
            'rng = np.random.default_rng(0)',
            'hidden = rng.choice(list(split.train_indices), size=len(split.train_indices) // 2, replace=False)',
            'masked = frame.copy()',
            "masked.loc[hidden, 'target'] = np.nan",
            'session = Session.ingest(masked).set_roles(session.dataset.roles)',
            'session.inject_split(train_indices=split.train_indices, test_indices=split.test_indices,',
            '                     validation_indices=split.validation_indices)',
            "session.scale(method='standard')",
            "session.ssl.fit_pretext(method='masked_tabular', epochs=5, random_state=0)",
            "session.ssl.transform(attach=True, partition='all')",
            "session.semisupervised.fit(method='label_propagation', columns=list(session.ssl.plan.representation_columns),",
            '                          prefer_reduce_components=False)',
            "print(session.semisupervised.evaluate(partition='test').metrics)",
        ),
        check=(
            "Does the two-stage pipeline beat single-stage propagation on validation?",
            "Which partitions did the encoder see during pretraining?",
        ),
        tools=("fit_ssl_pretext", "transform_ssl", "fit_semisupervised", "evaluate_semisupervised"),
        terms=("self-supervised", "semi-supervised", "embedding", "pseudo-label"),
        difficulty=CORE,
    ),
)

__all__ = ["SEMISUPERVISED_BEGINNER"]
