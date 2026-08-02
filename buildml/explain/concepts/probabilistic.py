# ruff: noqa: E501
"""Bayesian / probabilistic ML concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

PROBABILISTIC_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="probabilistic-uncertainty",
            title="Session Bayesian / probabilistic uncertainty path",
            summary="Fit sklearn BayesianRidge / GP / GaussianNB with predictive std/proba and optional train-only split conformal intervals.",
            definition=(
                "Probabilistic ML in BuildML fits uncertainty-aware sklearn "
                "estimators on the Session train partition, exposes predictive "
                "standard deviations or class probabilities, and optionally "
                "calibrates distribution-free intervals via split conformal "
                "carved from train only."
            ),
            intuition=(
                "Instead of a single point guess, the model also says how wide "
                "its uncertainty band is — and conformal wraps that band so "
                "coverage claims do not rely only on Gaussian assumptions."
            ),
            formal_idea=(
                "BayesianRidge / GP posterior predictive N(μ(x), σ²(x)); "
                "split conformal: q̂ = Rank⌈(n+1)(1−α)⌉ of |y−ŷ| on a train "
                "calibration carve; interval [ŷ−q̂, ŷ+q̂]."
            ),
            why_it_matters=(
                "Point metrics alone hide overconfidence.",
                "Using validation/test to calibrate intervals is leakage.",
            ),
            how_buildml_uses=(
                "Session.fit_probabilistic → predict_interval / evaluate_probabilistic.",
                "Classical Session.calibration() remains for classical fit(...) classifiers.",
            ),
            interpretation_rules=(
                "Read NLL, interval_coverage / set_coverage, mean width, disclosures.",
                "Disclosures state this is not PyMC/Stan MCMC.",
            ),
            assumptions=(
                "Numeric non-null features; one target; split present; "
                "enough train rows for an optional conformal carve.",
            ),
            failure_modes=(
                "Too few train rows for conformal; GP cost on large n; "
                "claiming MCMC posterior samples from sklearn estimators.",
            ),
            anti_patterns=(
                "Calibrating conformal on Session test.",
                "Advertising a probabilistic-programming platform.",
            ),
            worked_example_pattern=(
                "fit_probabilistic(estimator='bayesian_ridge', conformal=True) → "
                "predict_interval('test') → evaluate_probabilistic('validation').",
            ),
            related_concepts=(
                "probabilistic-bayesian-ridge",
                "probabilistic-split-conformal",
                "probabilistic-bundle-boundary",
                "leakage-boundary",
                "diagnostic-uncertainty",
            ),
        ),
        _note(
            key="probabilistic-bayesian-ridge",
            title="BayesianRidge predictive mean and std",
            summary="sklearn BayesianRidge returns posterior predictive mean and std under a Gaussian linear model.",
            definition=(
                "estimator='bayesian_ridge' fits sklearn.linear_model.BayesianRidge "
                "and uses predict(..., return_std=True) for posterior predictive "
                "uncertainty under the model's Gaussian assumptions."
            ),
            intuition=(
                "A linear model that also estimates how noisy its own coefficients "
                "are, then turns that into a per-row uncertainty band."
            ),
            formal_idea=(
                "Approximate Bayesian linear regression with Gamma priors on "
                "precision; predictive variance from the posterior."
            ),
            why_it_matters=(
                "A complete, honest regression uncertainty path without heavy deps.",
            ),
            how_buildml_uses=(
                "fit_probabilistic(estimator='bayesian_ridge'); "
                "interval_method may combine posterior_std with split_conformal.",
            ),
            interpretation_rules=(
                "Gaussian NLL and posterior-std intervals assume approximate normality; "
                "prefer conformal coverage for distribution-free claims.",
            ),
            assumptions=("Linear-ish relationships; numeric target.",),
            failure_modes=("Strong nonlinearity; heteroscedasticity extremes.",),
            anti_patterns=("Calling BayesianRidge a full hierarchical Bayes model.",),
            worked_example_pattern=(
                "fit_probabilistic('bayesian_ridge', alpha=0.1, conformal=True).",
            ),
            related_concepts=(
                "probabilistic-uncertainty",
                "probabilistic-gaussian-process",
                "probabilistic-split-conformal",
            ),
        ),
        _note(
            key="probabilistic-gaussian-process",
            title="Gaussian process regressor / classifier",
            summary="GP regressor exposes return_std; GPC exposes predict_proba; both are sklearn GPs, not deep GPs.",
            definition=(
                "gaussian_process_regressor / gaussian_process_classifier wrap "
                "sklearn.gaussian_process with an RBF(+White) kernel and "
                "n_restarts_optimizer defaulting to 0 for deterministic runs."
            ),
            intuition=(
                "Nearby points share similar predictions; uncertainty grows away "
                "from observed train rows."
            ),
            formal_idea=(
                "GP prior over functions; posterior predictive at test x*."
            ),
            why_it_matters=("Non-linear uncertainty when BayesianRidge is too rigid.",),
            how_buildml_uses=(
                "fit_probabilistic(estimator='gaussian_process_regressor'|..._classifier).",
            ),
            interpretation_rules=(
                "GPs scale poorly with large n; disclosures note the cost.",
            ),
            assumptions=("Moderate train size; scaled numeric features help.",),
            failure_modes=("Large n; poorly scaled features; overclaiming deep GPs.",),
            anti_patterns=("Using GPs as a silent default on huge tables.",),
            worked_example_pattern=(
                "fit_probabilistic('gaussian_process_regressor', conformal=True).",
            ),
            related_concepts=(
                "probabilistic-uncertainty",
                "probabilistic-bayesian-ridge",
            ),
        ),
        _note(
            key="probabilistic-split-conformal",
            title="Train-only split conformal intervals / sets",
            summary="MAPIE-style absolute-residual (regression) or 1−p(y) (classification) conformal carve from train only.",
            definition=(
                "When conformal=True, BuildML randomly (or stratified) carves a "
                "calibration subset from Session train, fits on the remainder, "
                "and stores a finite-sample conformal quantile. Holdout never "
                "enters calibration."
            ),
            intuition=(
                "Keep a sealed envelope of train examples to measure how wrong "
                "the model is, then widen intervals enough to cover that error "
                "rate — without peeking at the test set."
            ),
            formal_idea=(
                "Split conformal: q̂ = s_{(k)}, k=⌈(n+1)(1−α)⌉ on calibration scores."
            ),
            why_it_matters=(
                "Distribution-free finite-sample coverage (exchangeability) "
                "without MAPIE as a hard dependency.",
            ),
            how_buildml_uses=(
                "fit_probabilistic(conformal=True) → predict_interval / evaluate coverage.",
            ),
            interpretation_rules=(
                "Coverage is marginal under exchangeability; conditional coverage "
                "is not guaranteed.",
            ),
            assumptions=(
                "Exchangeable scores; enough train rows; classification needs "
                "≥2 train rows per class for the stratified carve.",
            ),
            failure_modes=(
                "Tiny train; distribution shift vs holdout; alpha mismatch after load.",
            ),
            anti_patterns=(
                "Using Session test as the conformal calibration set.",
            ),
            worked_example_pattern=(
                "fit_probabilistic(conformal=True, alpha=0.1) → "
                "evaluate_probabilistic(); read interval_coverage.",
            ),
            related_concepts=(
                "probabilistic-uncertainty",
                "leakage-boundary",
            ),
        ),
        _note(
            key="probabilistic-bundle-boundary",
            title="Probabilistic bundle boundary",
            summary="buildml.probabilistic_bundle.v1 stores ProbabilisticPlan; Session checkpoints do not embed it.",
            definition=(
                "A probabilistic bundle persists the estimator, conformal quantile, "
                "train carve indices, and class vocabulary. Session checkpoints "
                "persist data/roles/splits/history — not ProbabilisticPlan."
            ),
            intuition=(
                "Saving the lab notebook is not the same as saving the uncertainty "
                "model and its conformal ruler."
            ),
            formal_idea=(
                "Artifacts are complementary: checkpoint_load ↛ probabilistic model; "
                "load_probabilistic_bundle ↛ dataset rows."
            ),
            why_it_matters=("Mixing artifacts causes silent missing-learner failures.",),
            how_buildml_uses=("save_probabilistic_bundle / load_probabilistic_bundle.",),
            interpretation_rules=("Read meta.json format buildml.probabilistic_bundle.v1.",),
            assumptions=("Feature/target columns still match at load time.",),
            failure_modes=("Expecting checkpoint_load to restore ProbabilisticPlan.",),
            anti_patterns=(
                "Treating federated or online bundles as probabilistic plans.",
            ),
            worked_example_pattern=(
                "session.save_probabilistic_bundle(path); other.load_probabilistic_bundle(path).",
            ),
            related_concepts=("probabilistic-uncertainty", "federated-bundle-boundary"),
        ),
        _note(
            key="probabilistic-mapie",
            title="MAPIE conformal backend",
            summary="Optional MAPIE split/CV+/jackknife+ conformal via buildml[probabilistic-industry].",
            definition=(
                "backend='mapie' wraps MAPIE MapieRegressor / MapieClassifier "
                "for distribution-free intervals (regression) or prediction sets "
                "(classification). Split uses a train-only carve; CV+ and "
                "jackknife+ use Session train with internal resampling."
            ),
            intuition=(
                "Industry conformal tooling when you want CV+ or jackknife+ "
                "beyond the in-tree absolute-residual split recipe."
            ),
            formal_idea=(
                "Conformal prediction with MAPIE method='base' or 'plus'."
            ),
            why_it_matters=(
                "Honest industry defaults when MAPIE is installed.",
            ),
            how_buildml_uses=(
                "fit_probabilistic(backend='mapie', estimator='cv_plus', task='regression').",
            ),
            interpretation_rules=(
                "Read interval_coverage / set_coverage on holdout; disclosures note train-only fit.",
            ),
            assumptions=("Exchangeable scores; Session split present.",),
            failure_modes=("Missing extra; too-small train for MAPIE cv.",),
            anti_patterns=("Calibrating MAPIE on Session test.",),
            worked_example_pattern=(
                "fit_probabilistic(backend='mapie', estimator='split', task='regression').",
            ),
            related_concepts=("probabilistic-split-conformal", "probabilistic-uncertainty"),
        ),
        _note(
            key="probabilistic-ngboost",
            title="NGBoost predictive distributions",
            summary="Natural gradient boosting with Normal/Bernoulli predictive distributions.",
            definition=(
                "backend='ngboost' fits NGBRegressor / NGBClassifier and exposes "
                "pred_dist for NLL and CRPS (regression) plus optional in-tree "
                "conformal overlay carved from train."
            ),
            intuition=(
                "Gradient boosting that outputs a full predictive distribution, "
                "not just a point estimate."
            ),
            formal_idea=(
                "Natural gradient boosting toward a parametric predictive family."
            ),
            why_it_matters=(
                "Strong tabular uncertainty without MCMC.",
            ),
            how_buildml_uses=(
                "fit_probabilistic(backend='ngboost', estimator='ngboost_regressor').",
            ),
            interpretation_rules=("Read nll and crps in evaluate_probabilistic.",),
            assumptions=("Numeric features; enough train rows for boosting.",),
            failure_modes=("Missing ngboost extra; binary-only Bernoulli classifier path.",),
            anti_patterns=("Calling NGBoost a Bayesian MCMC posterior.",),
            worked_example_pattern=(
                "fit_probabilistic(backend='ngboost', estimator='ngboost_regressor', conformal=True).",
            ),
            related_concepts=("probabilistic-uncertainty", "diagnostic-uncertainty"),
        ),
    )
}
