# ruff: noqa: E501
"""Synthetic-data concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

SYNTHETIC_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="synthetic-train-only-generator",
            title="Train-only tabular synthesizers",
            summary=(
                "session.synthetic.fit estimates schema and generator parameters on "
                "Session train only, then session.synthetic.sample draws new rows."
            ),
            definition=(
                "A synthesizer is a fitted generative model of the train "
                "table (bootstrap, Gaussian copula, or SMOTE wrap). Holdouts "
                "must not contribute to its parameters."
            ),
            intuition=(
                "If the generator sees test rows, synthetic data carries "
                "holdout structure into later training: classic leakage."
            ),
            formal_idea=(
                "θ̂ = argmax_θ L(train); x̃ ~ p_θ̂; never θ̂(test)."
            ),
            why_it_matters=(
                "Synthetic augmentation that peeks at test inflates reported utility.",
            ),
            how_buildml_uses=(
                "session.synthetic.fit(...); session.synthetic.sample(...); "
                "session.synthetic.evaluate(partition='test').",
            ),
            interpretation_rules=(
                "Always train-only fit; prefer test for session.synthetic.evaluate.",
            ),
            assumptions=("Split present; non-empty train.",),
            failure_modes=("Fitting on validation/test; tiny train for copula.",),
            anti_patterns=("Calling bootstrap samples 'anonymous data'.",),
            worked_example_pattern=(
                "split → session.synthetic.fit(method='gaussian_copula') → "
                "session.synthetic.sample(n=500) → session.synthetic.evaluate(mode='tstr').",
            ),
            related_concepts=(
                "synthetic-vs-resample",
                "synthetic-fidelity-vs-tstr",
                "synthetic-privacy-limits",
                "leakage-boundary",
            ),
        ),
        _note(
            key="synthetic-vs-resample",
            title="Synthetic generators vs Session.resample",
            summary=(
                "resample rebalances train class counts (imblearn) as preprocess; "
                "session.synthetic.fit fits a reusable generator plan."
            ),
            definition=(
                "resample mutates train membership for imbalance; the synthetic "
                "API returns samples (or optionally merges) from a persisted "
                "SynthesizerPlan without implying class-balance as the goal."
            ),
            intuition=(
                "Need more minority rows for a classifier → resample. "
                "Need a general tabular generator / augmentation product → "
                "session.synthetic.fit."
            ),
            formal_idea=("Different objectives: P(y) rebalance vs p(x,y) model."),
            why_it_matters=(
                "Collapsing both into one API hides leakage and product boundaries.",
            ),
            how_buildml_uses=(
                "Session.resample(sampler='smote'); "
                "session.synthetic.fit(method='smote') for reusable sampling.",
            ),
            interpretation_rules=(
                "method='smote' in synthetic still requires buildml[imbalanced].",
            ),
            assumptions=("Target role for resample / SMOTE synthesizer.",),
            failure_modes=("Using resample as a general synthetic-data product.",),
            anti_patterns=("Assuming resample persists a generator bundle.",),
            worked_example_pattern=(
                "For imbalance: resample → fit. For generator: session.synthetic.fit → "
                "session.synthetic.sample → session.synthetic.save_bundle.",
            ),
            related_concepts=(
                "synthetic-train-only-generator",
                "synthetic-merge-provenance",
                "synthetic-bundle-boundary",
            ),
        ),
        _note(
            key="synthetic-fidelity-vs-tstr",
            title="Fidelity metrics vs TSTR utility",
            summary=(
                "session.synthetic.evaluate(mode='fidelity') scores distributional "
                "match; mode='tstr' trains on synthetic and tests on real."
            ),
            definition=(
                "Fidelity: KS / TV / correlation gaps. TSTR: Train on Synthetic, "
                "Test on Real: a downstream utility proxy."
            ),
            intuition=(
                "Good fidelity does not guarantee useful models; good TSTR does "
                "not prove privacy."
            ),
            formal_idea=(
                "fidelity ≈ d(P_real, P_syn); TSTR ≈ score(f̂_syn; D_real)."
            ),
            why_it_matters=(
                "Users must know which claim an eval supports.",
            ),
            how_buildml_uses=(
                "session.synthetic.evaluate(mode='fidelity'|'tstr', partition='test').",
            ),
            interpretation_rules=(
                "Prefer holdout partition for TSTR; disclose TRTR baseline when available.",
            ),
            assumptions=("Overlapping columns; TSTR needs a target.",),
            failure_modes=("Tuning generator knobs against test TSTR repeatedly.",),
            anti_patterns=("Reporting fidelity as a privacy certificate.",),
            worked_example_pattern=(
                "session.synthetic.fit → session.synthetic.evaluate(mode='fidelity') and "
                "session.synthetic.evaluate(mode='tstr').",
            ),
            related_concepts=(
                "synthetic-train-only-generator",
                "synthetic-privacy-limits",
            ),
        ),
        _note(
            key="synthetic-merge-provenance",
            title="Explicit merge with provenance",
            summary=(
                "session.synthetic.sample defaults to returning a Frame; "
                "merge_mode='extend_train' appends with an ignore-role marker."
            ),
            definition=(
                "Provenance column (default _synthetic) marks generated rows; "
                "role=ignore prevents silent feature poisoning. Holdouts stay intact."
            ),
            intuition=(
                "If synthetics land in the table without a flag, later fits "
                "treat them as real: often undesirable."
            ),
            formal_idea=("train' = train ∪ {(x̃, flag=1)}; val, test unchanged."),
            why_it_matters=("Silent merges corrupt audit trails and metrics.",),
            how_buildml_uses=(
                "session.synthetic.sample(merge_mode='extend_train', provenance_column='_synthetic').",
            ),
            interpretation_rules=(
                "Default merge_mode='none'; extend_train clears classical FitResult.",
            ),
            assumptions=("Split present for index rebuild.",),
            failure_modes=("Reusing an existing provenance column name.",),
            anti_patterns=("Concatenating synthetics into test.",),
            worked_example_pattern=(
                "session.synthetic.sample(n=200, merge_mode='extend_train') → refit on new train.",
            ),
            related_concepts=(
                "synthetic-vs-resample",
                "synthetic-train-only-generator",
            ),
        ),
        _note(
            key="synthetic-privacy-limits",
            title="Privacy limits of synthetic data",
            summary=(
                "BuildML synthesizers are not a differential-privacy product; "
                "bootstrap can near-duplicate train rows."
            ),
            definition=(
                "Utility synthesizers approximate p(x) or p(x,y). They do not "
                "provide (ε,δ)-DP guarantees unless a dedicated DP mechanism "
                "is implemented end-to-end."
            ),
            intuition=(
                "Resampled rows are almost the originals; copulas and SMOTE "
                "can still memorize structure."
            ),
            formal_idea=("Utility ≠ anonymity; DP needs calibrated noise + accounting."),
            why_it_matters=(
                "Shipping 'synthetic' data as a privacy control without DP is unsafe.",
            ),
            how_buildml_uses=(
                "Disclosures on fit/sample/evaluate and in synthetic bundles.",
            ),
            interpretation_rules=(
                "Do not claim anonymization; run a privacy review before sharing.",
            ),
            assumptions=("Honest disclosure in model cards / data sharing.",),
            failure_modes=("Treating bootstrap samples as public-safe releases.",),
            anti_patterns=("Marketing synthetic as DP without implementing DP.",),
            worked_example_pattern=(
                "Read session.synthetic.fit disclosures; keep real PII out of shared samples.",
            ),
            related_concepts=(
                "synthetic-fidelity-vs-tstr",
                "synthetic-bundle-boundary",
            ),
        ),
        _note(
            key="synthetic-bundle-boundary",
            title="Synthetic bundle vs Session checkpoint",
            summary=(
                "buildml.synthetic_bundle.v1 stores SynthesizerPlan; Session "
                "checkpoints do not embed it."
            ),
            definition=(
                "Bundles ship generator state (joblib) + meta.json. Checkpoints "
                "ship data/roles/splits/history and optional classical fits."
            ),
            intuition=(
                "Reload workflow via checkpoint_load; reload generator via "
                "session.synthetic.load_bundle."
            ),
            formal_idea=("Orthogonal artifacts; compose explicitly."),
            why_it_matters=("Avoid assuming one artifact contains the other.",),
            how_buildml_uses=(
                "session.synthetic.save_bundle / session.synthetic.load_bundle.",
            ),
            interpretation_rules=(
                "Confirm format buildml.synthetic_bundle.v1 in meta.json.",
            ),
            assumptions=("Writable path; compatible BuildML version.",),
            failure_modes=("Incomplete bundle missing synthetic_plan.joblib.",),
            anti_patterns=("Assuming checkpoint_save includes the synthesizer.",),
            worked_example_pattern=(
                "session.synthetic.fit → session.synthetic.save_bundle → session.synthetic.load_bundle → sample.",
            ),
            related_concepts=(
                "synthetic-train-only-generator",
                "synthetic-privacy-limits",
            ),
        ),
    )
}
