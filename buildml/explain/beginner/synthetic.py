# ruff: noqa: E501
"""Beginner layers for synthetic tabular data."""

from __future__ import annotations

from buildml.explain.beginner._builder import CORE, FOUNDATION, BeginnerLayer, _index, _layer

SYNTHETIC_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "synthetic-train-only-generator",
        plain=(
            "A synthesizer learns what your table looks like: the range of each column, how columns move "
            "together: and can then produce brand-new rows that resemble the real ones without being "
            "copies. It learns from training rows only."
        ),
        analogy=(
            "A forger who has studied a thousand genuine signatures and can produce a convincing new one. "
            "Convincing is the goal; it is still not anyone's actual signature."
        ),
        steps=(
            "Split your data first, so training and holdout are already separate.",
            "`fit_synthesizer` learns the column schema and the generator parameters from training rows only.",
            "Bootstrap resamples existing rows; Gaussian copula models the joint distribution; SMOTE interpolates between neighbours.",
            "`sample_synthetic(n=...)` draws as many new rows as you want.",
            "`evaluate_synthetic` checks the result against real holdout rows.",
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
                "It is generated from a model of your training data. If that model is wrong, you are training on confident fiction.",
            ),
            (
                "Fitting the generator on everything gives a better generator.",
                "It gives one that has seen your test set. Every downstream number then flatters you, and nothing you report is trustworthy.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, random_state=0)",
            "session.fit_synthesizer(method='gaussian_copula', random_state=0)",
            "extra = session.sample_synthetic(n=500)",
            "session.evaluate_synthetic(mode='tstr', partition='test')",
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
            "changing which training rows exist: it is a preprocessing step. `fit_synthesizer` builds a "
            "reusable generator you can save, share, and sample from repeatedly."
        ),
        analogy=(
            "Adjusting the guest list so the room is balanced, versus hiring a company that can produce "
            "convincing extras on demand. Both change who is in the room; only one is a reusable service."
        ),
        steps=(
            "Ask what you are trying to fix.",
            "Too few rows of the rare class, and you just want a fair classifier? Use `resample`.",
            "Want new rows on demand, saved as an artifact, possibly for sharing? Use `fit_synthesizer`.",
            "`resample` mutates training membership directly and persists nothing.",
            "The synthetic path returns a frame by default and can save a bundle.",
        ),
        use=(
            "`resample` for the specific, common problem of class imbalance before fitting.",
            "`fit_synthesizer` when generation itself is the deliverable.",
        ),
        avoid=(
            "Do not use `resample` as a general synthetic-data product; it has no bundle and no evaluation surface.",
            "Do not use the synthetic path purely to balance classes; `resample` is simpler and purpose-built.",
        ),
        myths=(
            (
                "SMOTE is SMOTE, so the two are interchangeable.",
                "The algorithm overlaps; the product surface does not. One rebalances training membership; the other produces a saveable generator plan with disclosures and evaluation.",
            ),
            (
                "Resample saves a generator I can reuse later.",
                "It does not. It changes the current training rows and that is all. If you need reuse, you need a synthetic bundle.",
            ),
        ),
        example=(
            "session.resample(sampler='smote')            # imbalance fix",
            "session.fit_synthesizer(method='smote')      # reusable generator",
            "session.save_synthetic_bundle('artifacts/gen')",
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
        myths=(
            (
                "High fidelity means the synthetic data is useful.",
                "A generator can match every marginal distribution perfectly and destroy the relationships a model needs. TSTR catches that; fidelity does not.",
            ),
            (
                "Good TSTR means the synthetic data is safe to release.",
                "Utility and privacy are unrelated axes. A generator that memorized your training rows would score wonderfully on TSTR.",
            ),
        ),
        example=(
            "session.evaluate_synthetic(mode='fidelity', partition='test')",
            "session.evaluate_synthetic(mode='tstr', partition='test')",
            "# compare TSTR against the real-data baseline before drawing conclusions",
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
        steps=(
            "`sample_synthetic(n=...)` returns a frame; `merge_mode` defaults to none.",
            "With `merge_mode='extend_train'`, the rows are appended to training only.",
            "A provenance column (`_synthetic` by default) marks the generated rows.",
            "That column gets the `ignore` role, so no model can accidentally learn from the marker itself.",
            "Existing fit results are cleared, because the training set they were fitted on no longer exists.",
        ),
        use=(
            "When you want to train on real plus synthetic rows and still be able to separate them afterwards.",
            "When an audit will ask which rows in this training set were real.",
        ),
        avoid=(
            "Do not merge into validation or test: BuildML will not do it, and neither should you by hand.",
            "Do not reuse an existing column name for provenance; you will silently overwrite real data.",
        ),
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
            "session.sample_synthetic(",
            "    n=200, merge_mode='extend_train', provenance_column='_synthetic',",
            ")",
            "session.fit()  # refit on the extended training set",
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
            "Synthetic does not mean anonymous. BuildML's synthesizers are built for utility, not privacy. "
            "Bootstrap sampling in particular can reproduce training rows almost exactly, and copulas and "
            "SMOTE can memorize structure that identifies individuals."
        ),
        analogy=(
            "Changing everyone's name in a report does not anonymize it when the report still says 'the "
            "only left-handed pilot in the Reykjavik office'."
        ),
        steps=(
            "Understand what your method does: bootstrap resamples real rows, so outputs can be near-duplicates.",
            "Copulas and SMOTE build from real values and can still reproduce rare combinations.",
            "None of these provide a formal privacy guarantee: no calibrated noise, no privacy accounting.",
            "Read the disclosures attached to fitting, sampling, and the bundle.",
            "Before sharing anything outside your organization, run an actual privacy review.",
        ),
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
            "plan = session.fit_synthesizer(method='bootstrap')",
            "for note in plan.disclosures: print(note)",
            "# keep real identifiers out of anything you share",
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
            "Call `save_synthetic_bundle(path)`: the generator state and a metadata file are written.",
            "Reload with `load_synthetic_bundle(path)`.",
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
            "session.fit_synthesizer(method='gaussian_copula')",
            "session.save_synthetic_bundle('artifacts/customer-gen')",
            "other = Session().load_synthetic_bundle('artifacts/customer-gen')",
            "other.sample_synthetic(n=1000)",
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
