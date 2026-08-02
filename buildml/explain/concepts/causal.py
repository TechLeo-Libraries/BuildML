# ruff: noqa: E501
"""Causal ML concept notes (assumptions-required identification)."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

CAUSAL_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="causal-assumptions",
            title="CausalAssumptions are mandatory for estimation",
            summary="Causal APIs refuse to run without an explicit treatment, outcome, confounders, estimand, and unconfoundedness/positivity acknowledgements.",
            definition=(
                "CausalAssumptions is a caller-declared identification contract: "
                "treatment, outcome, confounders (or an explicit empty-confounder "
                "waiver), estimand (ATE), identification strategy (backdoor), "
                "and boolean acknowledgements that unconfoundedness and "
                "positivity are assumed."
            ),
            intuition=(
                "You cannot get causality from a correlation heatmap. You must "
                "write down what you are assuming before BuildML will estimate "
                "an effect."
            ),
            formal_idea=(
                "Backdoor adjustment: if W blocks all backdoor paths from T to Y "
                "and positivity holds, ATE = E[E[Y|T=1,W] − E[Y|T=0,W]]."
            ),
            why_it_matters=(
                "Without declared assumptions, 'causal' numbers are marketing.",
                "EDA associations must never be silently upgraded to effects.",
            ),
            how_buildml_uses=(
                "Session.declare_causal_assumptions → fit_causal / estimate_causal.",
                "Incomplete objects raise ValidationError.",
            ),
            interpretation_rules=(
                "Read disclosures: acknowledgements are necessary, not sufficient.",
                "Empty confounders require allow_empty_confounders=True (strong).",
            ),
            assumptions=(
                "Caller-supplied domain knowledge about confounding and overlap.",
            ),
            failure_modes=(
                "Unmeasured confounding; positivity violations; wrong estimand.",
            ),
            anti_patterns=(
                "Treating EDA correlation / MI / importance as identification.",
                "Omitting acknowledgements and expecting defaults to invent causality.",
            ),
            worked_example_pattern=(
                "declare_causal_assumptions(treatment='t', outcome='y', "
                "confounders=['x1','x2'], acknowledge_unconfoundedness=True, "
                "acknowledge_positivity=True) → fit_causal(method='aipw').",
            ),
            related_concepts=(
                "causal-ate-backdoor",
                "causal-aipw",
                "causal-eda-boundary",
                "causal-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="causal-ate-backdoor",
            title="Average treatment effect under backdoor adjustment",
            summary="This surface identifies ATE under a declared backdoor adjustment set — not IV, not front-door, not causal discovery.",
            definition=(
                "ATE = E[Y(1) − Y(0)]. Under backdoor, conditioning on declared "
                "confounders identifies the ATE when unconfoundedness and "
                "positivity hold."
            ),
            intuition=(
                "Compare outcomes as if everyone were treated vs untreated, "
                "after adjusting for the covariates you claim block confounding."
            ),
            formal_idea=(
                "Identification via backdoor functional; estimation via "
                "T-learner, IPW, or AIPW."
            ),
            why_it_matters=("ATE is the shipped estimand; other estimands are out of scope."),
            how_buildml_uses=(
                "assumptions.estimand='ATE'; assumptions.identification='backdoor'.",
            ),
            interpretation_rules=(
                "CIs quantify sampling uncertainty under the estimator — not "
                "uncertainty about whether assumptions are true.",
            ),
            assumptions=("No unmeasured confounding given W; positivity; SUTVA-style consistency."),
            failure_modes=("Hidden confounders; poor overlap; interference."),
            anti_patterns=("Claiming discovery of the causal graph from Session data."),
            worked_example_pattern="fit_causal(method='aipw') → evaluate_causal('validation').",
            related_concepts=("causal-assumptions", "causal-aipw", "causal-t-learner"),
        ),
        _note(
            key="causal-t-learner",
            title="T-learner outcome regression ATE",
            summary="Fit separate outcome models in treated and control arms; ATE is the mean predicted difference.",
            definition=(
                "T-learner fits μ1(w)=E[Y|T=1,W] and μ0(w)=E[Y|T=0,W] on train "
                "arms, then estimates ATE as the mean of μ1(w)−μ0(w)."
            ),
            intuition=(
                "Two outcome regressions — one per treatment arm — then average "
                "how much the predictions differ."
            ),
            formal_idea="ATÊ = n⁻¹ Σᵢ (μ̂1(Wᵢ) − μ̂0(Wᵢ)).",
            why_it_matters=("Complete, honest outcome-regression path without heavy deps."),
            how_buildml_uses=("fit_causal(method='t_learner')."),
            interpretation_rules=("Sensitive to outcome-model misspecification."),
            assumptions=("Correct outcome regressions + CausalAssumptions."),
            failure_modes=("Extrapolation when arms have different W support."),
            anti_patterns=("Skipping propensity diagnostics entirely when overlap is thin."),
            worked_example_pattern="fit_causal(method='t_learner', bootstrap_samples=100).",
            related_concepts=("causal-aipw", "causal-assumptions"),
        ),
        _note(
            key="causal-ipw",
            title="Inverse propensity weighting ATE",
            summary="Weight outcomes by inverse propensity of observed treatment; clip propensities for stability.",
            definition=(
                "IPW estimates ATE with scores T Y / e(W) − (1−T) Y / (1−e(W)), "
                "where e(W)=P(T=1|W) is fitted on train and clipped."
            ),
            intuition=(
                "Up-weight rare treatment/confounder combinations so the "
                "weighted population looks exchangeable."
            ),
            formal_idea="Horvitz–Thompson-style IPW functional with clipped ê(W).",
            why_it_matters=("Makes overlap failures visible via extreme weights / clips."),
            how_buildml_uses=("fit_causal(method='ipw'); evaluate reports propensity AUC/Brier."),
            interpretation_rules=("Clip hits are positivity warnings, not proof of ATE."),
            assumptions=("Correct propensity + CausalAssumptions."),
            failure_modes=("Near-violations of positivity; propensity misspecification."),
            anti_patterns=("Ignoring propensity_min/max disclosures."),
            worked_example_pattern="fit_causal(method='ipw') → refute_causal('placebo_treatment').",
            related_concepts=("causal-aipw", "causal-assumptions"),
        ),
        _note(
            key="causal-aipw",
            title="Augmented IPW (doubly robust) ATE",
            summary="Combine outcome regression and IPW so the ATE remains consistent if either nuisance is correct.",
            definition=(
                "AIPW uses the doubly robust score "
                "μ1−μ0 + T(Y−μ1)/e − (1−T)(Y−μ0)/(1−e)."
            ),
            intuition=(
                "Start from the T-learner difference, then add an IPW correction "
                "for outcome-model mistakes."
            ),
            formal_idea="Efficient influence-function / AIPW estimator for ATE.",
            why_it_matters=("Default method: more robust than pure regression or pure IPW."),
            how_buildml_uses=("fit_causal(method='aipw') is the Session default."),
            interpretation_rules=(
                "Double robustness is about consistency under correct nuisances — "
                "not immunity to false assumptions.",
            ),
            assumptions=("At least one of {outcome models, propensity} correct + CausalAssumptions."),
            failure_modes=("Both nuisances wrong; severe overlap failure."),
            anti_patterns=("Calling AIPW 'assumption-free'."),
            worked_example_pattern=(
                "declare_causal_assumptions(...) → fit_causal('aipw') → "
                "evaluate_causal → save_causal_bundle."
            ),
            related_concepts=("causal-t-learner", "causal-ipw", "causal-assumptions"),
        ),
        _note(
            key="causal-eda-boundary",
            title="EDA stays associational; causal is a separate path",
            summary="Teaching Studio / EDA / importance diagnostics refuse causal claims; only the assumption-declared causal path estimates effects.",
            definition=(
                "BuildML keeps association surfaces (EDA, MI, permutation "
                "importance, unsupervised labels) linguistically and API-wise "
                "separate from CausalAssumptions-backed estimation."
            ),
            intuition=(
                "Correlation screens help you ask questions; they do not answer "
                "counterfactual ones."
            ),
            formal_idea="Association ≠ identification.",
            why_it_matters=("Prevents silent upgrading of exploratory plots into policy effects."),
            how_buildml_uses=(
                "EDA text remains non-causal; causal methods require declare_causal_assumptions.",
            ),
            interpretation_rules=("Never paste an EDA finding into a causal claim without new assumptions."),
            assumptions=("N/A — this note is a product boundary."),
            failure_modes=("Stakeholder slides that equate MI with ATE."),
            anti_patterns=("Asking the AI operator to 'infer causality from eda()'."),
            worked_example_pattern=(
                "Use eda() for associations; use declare_causal_assumptions + "
                "fit_causal for effects."
            ),
            related_concepts=("causal-assumptions", "feature-importance", "mutual-information"),
        ),
        _note(
            key="causal-bundle-boundary",
            title="Causal bundles vs Session checkpoints",
            summary="buildml.causal_bundle.v1 stores assumptions + nuisances + estimates; checkpoints do not embed CausalPlan.",
            definition=(
                "Causal bundles persist CausalPlan (assumptions, mu0/mu1/"
                "propensity, train ATE/CI). Session checkpoints store data/"
                "roles/splits/history without the causal learner."
            ),
            intuition=("Reload the table with a checkpoint; reload the effect model with a bundle."),
            formal_idea="Distinct artifact kinds with an explicit compatibility boundary string.",
            why_it_matters=("Mixing artifact kinds causes silent missing-learner failures."),
            how_buildml_uses=("save_causal_bundle / load_causal_bundle."),
            interpretation_rules=("meta.json disclosures restate assumption requirements."),
            assumptions=("Bundle format buildml.causal_bundle.v1."),
            failure_modes=("Expecting checkpoint_load to restore CausalPlan."),
            anti_patterns=("Treating causal bundles as interchangeable with probabilistic bundles."),
            worked_example_pattern="save_causal_bundle('artifacts/causal_bundle').",
            related_concepts=("causal-assumptions", "leakage-boundary"),
        ),
    )
}
