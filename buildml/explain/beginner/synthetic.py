# ruff: noqa: E501
"""Beginner layers for synthetic tabular data."""

from __future__ import annotations

from buildml.explain.beginner._builder import CORE, FOUNDATION, BeginnerLayer, _index, _layer

SYNTHETIC_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "synthetic-train-only-generator",
        plain=(
            'A synthesizer uses training rows to construct a reusable sampling model. Bootstrap sampling resamples observed rows; other methods model distributions or interpolate. Generated rows can duplicate or closely resemble training data.'
        ),
        analogy=(
            'A sampling recipe describes how to produce more rows with selected properties of a reference table. Some recipes resample existing rows, while others generate new combinations.'
        ),
        steps=(
            "Split your data first, so training and holdout are already separate.",
            "`session.synthetic.fit` learns the column schema and the generator parameters from training rows only.",
            "Bootstrap resamples existing rows; Gaussian copula models the joint distribution; SMOTE interpolates between neighbours.",
            "`session.synthetic.sample(n=...)` draws as many new rows as you want.",
            "`session.synthetic.evaluate` checks the result against real holdout rows.",
        ),
        use=(
            "To augment a small training set when collecting more real data is not possible.",
            "To share a dataset shaped like the real one for testing, demos, or development.",
        ),
        avoid=(
            "Do not fit the generator before splitting; synthetic rows would then carry holdout structure into training.",
            "Do not fit a copula on a tiny training set: there is not enough there to estimate a joint distribution.",
        ),
        myths=(
            (
                "Synthetic data can only help, since it is not real.",
                "Generated rows reflect the generator's assumptions and errors. Measure whether adding them improves performance on real held-out data.",
            ),
            (
                "Fitting the generator on everything gives a better generator.",
                "Using test rows to fit the generator compromises the independence of subsequent test scores. Fit the generator on training rows only.",
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
            "session.synthetic.fit(method='gaussian_copula', random_state=0)",
            'sample = session.synthetic.sample(n=200, random_state=1)',
            'print(sample.frame.shape)',
            "print(session.synthetic.evaluate(mode='tstr', partition='test').metrics)",
        ),
        check=(
            "Did you split before fitting the synthesizer?",
            "Does a model trained on the synthetic rows do anything useful on real ones?",
        ),
        tools=("fit_synthesizer", "sample_synthetic", "evaluate_synthetic", "split"),
        terms=("synthetic data", "distribution", "leakage", "holdout"),
        difficulty=CORE,
    ),
    _layer(
        "synthetic-vs-resample",
        plain=(
            "These two look alike and answer different questions. `resample` fixes class imbalance by "
            "changing which training rows exist: it is a preprocessing step. `session.synthetic.fit` builds a "
            "reusable generator you can save, share, and sample from repeatedly."
        ),
        analogy=(
            "Adjusting the guest list so the room is balanced, versus hiring a company that can produce "
            "convincing extras on demand. Both change who is in the room; only one is a reusable service."
        ),
        steps=('Ask what you are trying to fix.', 'To adjust class proportions in the current training set, use `resample` and evaluate the resulting classifier on unchanged holdout data.', 'Want new rows on demand, saved as an artifact, possibly for sharing? Use `session.synthetic.fit`.', '`resample` changes training membership and records workflow history; it does not create a reusable generator bundle.', 'The synthetic path returns a frame by default and can save a bundle.'),
        use=(
            "`resample` for the specific, common problem of class imbalance before fitting.",
            "`session.synthetic.fit` when generation itself is the deliverable.",
        ),
        avoid=(
            "Do not use `resample` as a general synthetic-data product; it has no bundle and no evaluation surface.",
            "Do not use the synthetic path purely to balance classes; `resample` is simpler and purpose-built.",
        ),
        myths=(('SMOTE is SMOTE, so the two are interchangeable.', 'The algorithm overlaps; the product surface does not. One rebalances training membership; the other produces a saveable generator plan with disclosures and evaluation.'), ('Resample saves a generator I can reuse later.', 'It does not. It changes the current training rows and records the operation in session history. If you need reuse, you need a synthetic bundle.')),
        example=(
            '# Install first: pip install "buildml[imbalanced]"',
            'import pandas as pd',
            'from sklearn.datasets import make_classification',
            'from sklearn.linear_model import LogisticRegression',
            'from buildml import Session',
            '',
            'x, y = make_classification(n_samples=240, n_features=6, n_informative=4, weights=[0.8, 0.2], random_state=0)',
            "frame = pd.DataFrame(x, columns=[f'x{i}' for i in range(6)])",
            "frame['target'] = y",
            "session = Session.ingest(frame).set_roles({**{f'x{i}': 'feature' for i in range(6)}, 'target': 'target'})",
            'session.split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)',
            '# In-place class balancing for a training run:',
            'original_split = session.split_plan',
            "session.resample(sampler='smote')",
            '# Use the original split in a separate Session to fit a reusable generator.',
            'generator = Session.ingest(frame).set_roles(session.dataset.roles)',
            'generator.inject_split(train_indices=original_split.train_indices,',
            '                       test_indices=original_split.test_indices,',
            '                       validation_indices=original_split.validation_indices)',
            "generator.synthetic.fit(method='smote', random_state=0)",
            'print(generator.synthetic.sample(n=50, random_state=1).frame.shape)',
        ),
        check=(
            "Do you need the generated rows once, or repeatedly?",
            "Will anything outside this session need to produce rows like these?",
        ),
        tools=("resample", "fit_synthesizer", "sample_synthetic", "save_synthetic_bundle"),
        terms=("synthetic data", "class imbalance", "SMOTE", "bundle"),
        difficulty=CORE,
    ),
    _layer(
        "synthetic-fidelity-vs-tstr",
        plain=(
            "There are two very different ways to ask whether synthetic data is any good. Fidelity asks "
            "'do the numbers look statistically similar?'. TSTR: train on synthetic, test on real: asks "
            "'can a model learn from the fake data and still work on real data?'."
        ),
        analogy=(
            "A flight simulator can look photographically perfect and teach you nothing, or look crude and "
            "train excellent pilots. Appearance and usefulness are separate measurements."
        ),
        steps=(
            "Fidelity mode compares each column's distribution and the correlations between columns.",
            "It reports gaps: how far the synthetic distribution sits from the real one.",
            "TSTR mode trains a model entirely on synthetic rows.",
            "It then scores that model on real held-out rows.",
            "Compare against training on real data: that baseline is what tells you how much you lost.",
        ),
        use=(
            "Fidelity when the synthetic data will be looked at or analysed directly.",
            "TSTR when the synthetic data will be used to train models. This is the measurement that matters most.",
        ),
        avoid=(
            "Do not tune generator settings repeatedly against test TSTR; you will overfit the test set through the generator.",
            "Do not report fidelity as evidence of privacy. It measures similarity, and high similarity is arguably the opposite of private.",
        ),
        myths=(('High fidelity means the synthetic data is useful.', 'Marginal distributions can match while predictive relationships differ. Fidelity summaries inspect selected statistics; TSTR evaluates utility for a chosen downstream task.'), ('Good TSTR means the synthetic data is safe to release.', 'Utility and privacy are unrelated axes. A generator that reproduces training rows may retain useful predictive information while exposing those rows. TSTR does not assess that privacy risk.')),
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
            "session.synthetic.fit(method='gaussian_copula', random_state=0)",
            "print(session.synthetic.evaluate(mode='fidelity', partition='test', eval_backend='builtin').metrics)",
            "print(session.synthetic.evaluate(mode='tstr', partition='test').metrics)",
        ),
        check=(
            "How much worse is TSTR than training on real data?",
            "How many times have you adjusted the generator after seeing a test score?",
        ),
        tools=("evaluate_synthetic", "fit_synthesizer", "sample_synthetic"),
        terms=("synthetic data", "TSTR", "distribution", "holdout"),
        difficulty=CORE,
    ),
    _layer(
        "synthetic-merge-provenance",
        plain=(
            "By default, sampling hands you a separate frame and changes nothing. If you ask BuildML to "
            "merge synthetic rows into your training set, it adds a marker column recording which rows were "
            "generated: and it never touches validation or test."
        ),
        analogy=(
            "Stamping every reproduction in the archive. It can sit on the same shelf as the originals "
            "precisely because nobody can mistake it for one."
        ),
        steps=('`session.synthetic.sample(n=...)` returns a frame; `merge_mode` defaults to none.', "With `merge_mode='extend_train'`, the rows are appended to training only.", 'A provenance column (`_synthetic` by default) marks the generated rows.', 'That column gets the `ignore` role, so standard role-based feature selection excludes it.', 'Existing fit results are cleared, because the training set they were fitted on no longer exists.'),
        use=(
            "When you want to train on real plus synthetic rows and still be able to separate them afterwards.",
            "When an audit will ask which rows in this training set were real.",
        ),
        avoid=('Do not merge into validation or test: BuildML will not do it, and neither should you by hand.', 'Choose a new provenance column name. BuildML rejects a name already present in the dataset.'),
        myths=(
            (
                "The provenance column is just documentation.",
                "It lets you filter, weight, or exclude synthetic rows in any later step. Without it, that information is gone forever.",
            ),
            (
                "Merging is the normal way to use synthetic data.",
                "The default is deliberately not to merge. Getting a separate frame keeps you in control of what enters your training set and when.",
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
            "session.synthetic.fit(method='gaussian_copula', random_state=0)",
            "session.synthetic.sample(n=100, merge_mode='extend_train', provenance_column='_synthetic', random_state=1)",
            "session.fit(LogisticRegression(max_iter=500), task='classification')",
            "print(session.evaluate(partition='test').metrics)",
        ),
        check=(
            "What fraction of your training rows are now synthetic?",
            "Does your holdout still contain only real rows?",
        ),
        tools=("sample_synthetic", "fit_synthesizer", "set_roles", "fit"),
        terms=("synthetic data", "provenance", "role", "holdout"),
        difficulty=CORE,
    ),
    _layer(
        "synthetic-privacy-limits",
        plain=(
            "Synthetic does not mean anonymous. BuildML's synthesizers are built for utility, not privacy. Bootstrap sampling can reproduce training rows exactly, and copulas and SMOTE can memorize structure that identifies individuals."
        ),
        analogy=(
            "Changing everyone's name in a report does not anonymize it when the report still says 'the "
            "only left-handed pilot in the Reykjavik office'."
        ),
        steps=('Understand what your method does: bootstrap resamples real rows, so outputs can be exact duplicates.', 'Copulas and SMOTE build from real values and can still reproduce rare combinations.', 'None of these provide a formal privacy guarantee: no calibrated noise, no privacy accounting.', 'Read the disclosures attached to fitting, sampling, and the bundle.', 'Before sharing anything outside your organization, run an actual privacy review.'),
        use=(
            "Synthetic data for augmentation, testing, and internal development.",
            "A dedicated differential-privacy tool when you need a real privacy guarantee.",
        ),
        avoid=(
            "Do not release synthetic data publicly on the assumption that generation equals anonymization.",
            "Do not describe these outputs as differentially private in any document, model card, or contract.",
        ),
        myths=(
            (
                "Generated rows cannot correspond to real people.",
                "A bootstrap sample *is* a real row. Even a copula can output a combination held by exactly one person in your data.",
            ),
            (
                "Adding noise makes it private.",
                "Differential privacy requires noise calibrated to sensitivity plus a privacy budget accounted across every query. Ad-hoc noise provides no guarantee.",
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
            "result = session.synthetic.fit(method='bootstrap', smooth_sigma=0.0, random_state=0)",
            'for note in result.disclosures:',
            '    print(note)',
            'sample = session.synthetic.sample(n=10, random_state=1)',
            'print(sample.frame.shape)',
            '# With zero smoothing, bootstrap copies training rows; it does not anonymize them.',
        ),
        check=(
            "Would any generated row be recognizable to someone who knows the underlying population?",
            "Who is going to receive this data, and has a privacy review approved it?",
        ),
        tools=("fit_synthesizer", "sample_synthetic", "evaluate_synthetic"),
        terms=("synthetic data", "privacy", "differential privacy", "disclosure"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "synthetic-bundle-boundary",
        plain=(
            "The fitted generator saves as a synthetic bundle. A Session checkpoint stores your data, "
            "roles, splits, and history: it does not contain the generator."
        ),
        analogy=(
            "The mould and the batch of castings are separate items. Storing the castings does not give "
            "you the ability to make more."
        ),
        steps=(
            "Fit a synthesizer.",
            "Call `session.synthetic.save_bundle(path)`: the generator state and a metadata file are written.",
            "Reload with `session.synthetic.load_bundle(path)`.",
            "Sample new rows from the restored generator.",
            "Keep checkpoints separate for the workflow itself.",
        ),
        use=(
            "When another team or service needs to generate rows from your fitted model.",
            "When you must reproduce exactly the generator that produced a past dataset.",
        ),
        avoid=(
            "Do not assume `checkpoint_save` includes the synthesizer.",
            "Do not ship a partial bundle directory; the metadata and the generator state are both required.",
        ),
        myths=(
            (
                "The bundle contains synthetic data.",
                "It contains the generator. Data is what you produce by sampling from it, which is precisely why the bundle is the more useful artifact.",
            ),
            (
                "Sharing the bundle is safer than sharing the samples.",
                "It can be less safe. The bundle encodes the training distribution and can generate unlimited rows, including near-duplicates of real ones.",
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
            "session.synthetic.fit(method='gaussian_copula', random_state=0)",
            'from tempfile import TemporaryDirectory',
            '',
            'with TemporaryDirectory() as directory:',
            "    session.synthetic.save_bundle(directory + '/generator')",
            '    other = Session.ingest(frame).set_roles(session.dataset.roles)',
            '    other.inject_split(train_indices=session.split_plan.train_indices,',
            '                       test_indices=session.split_plan.test_indices,',
            '                       validation_indices=session.split_plan.validation_indices)',
            "    other.synthetic.load_bundle(directory + '/generator', trusted=True)",
            '    print(other.synthetic.sample(n=50, random_state=1).frame.shape)',
        ),
        check=(
            "Does the bundle directory contain both the metadata and the generator state?",
            "Who has access to this bundle, and does the privacy review cover them?",
        ),
        tools=("save_synthetic_bundle", "load_synthetic_bundle", "sample_synthetic", "checkpoint_save"),
        terms=("bundle", "checkpoint", "synthetic data", "privacy"),
        difficulty=CORE,
    ),
)

__all__ = ["SYNTHETIC_BEGINNER"]
