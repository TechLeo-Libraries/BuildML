"""Stage 05 · Interpretation - what a fitted model may be said to show."""

from __future__ import annotations

from buildml.dashboard.academy_curriculum._factory import L, with_starter
from buildml.dashboard.academy_curriculum._helpers import (
    first_feature,
    fmt_n,
    is_classification,
    list_names,
    target_name,
)
from buildml.dashboard.academy_curriculum._types import LessonSpec


def lessons() -> list[LessonSpec]:
    return [
        L(
            slug="feature-importance-methods",
            stage=5,
            order=10,
            concept_key="feature-importance",
            tags=("importance", "SHAP"),
            plain=(
                "Importance methods estimate which features the model relied on. "
                "They explain the model, not the world - and different methods disagree.",
            ),
            technical=(
                "session.feature_importance(...) and session.explain_shap(...) attribute model behaviour. "
                "Permutation importance needs a partition; SHAP needs the optional shap extra.",
            ),
            why=("Stakeholders will ask 'why'; answer with model attributions + caveats."),
            formula=None,
            calculation=lambda ctx: (
                f"After fit, expect attributions over ~{fmt_n(ctx.get('eligible'))} eligible features "
                f"(e.g. leaders from MI screen: {list_names(ctx.get('mi') or [])})."
            ),
            session_evidence=lambda ctx: (
                f"MI screen leaders (univariate, pre-model): {list_names(ctx.get('mi') or [])}."
            ),
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
                "",
                "session = session.impute().encode()",
                "est = RandomForestClassifier(random_state=0) if "
                f"{is_classification(ctx)} else RandomForestRegressor(random_state=0)",
                "session = session.fit(est)",
                "imp = session.feature_importance(partition=\"test\", n_repeats=8)",
                "print(imp)",
                "# Optional: session.explain_shap(partition=\"test\")  # requires buildml[shap]",
                'session.learn("feature-importance", level="beginner")',
            ),
            what_to_change=("Choose partition; do not treat importance as causality."),
            pitfalls=(
                "Reading importance as causal effect size.",
                "Comparing importances across highly collinear features without care.",
            ),
            decide="Report importance with method name, partition, and collinearity caveats.",
            read_steps=(
                "Compare permutation importance vs model native importance.",
                "Cross-check with MI leaders - disagreement is informative.",
            ),
        ),
        L(
            slug="effect-shapes",
            stage=5,
            order=20,
            concept_key="feature-importance",
            tags=("ALE", "PDP", "shape"),
            plain=(
                "Effect shapes ask how predictions change as a feature moves - partial dependence / "
                "ALE-style views. They are still model stories, not causal levers.",
            ),
            technical=(
                "Use model plots / SHAP dependence after fit. Watch interactions: average shapes can lie.",
            ),
            why=("Direction and non-linearity matter for trust and debugging."),
            formula=None,
            calculation=lambda ctx: (
                f"Candidate features for shape plots: {list_names(ctx.get('mi') or []) or first_feature(ctx)}."
            ),
            session_evidence=lambda ctx: f"MI leaders: {list_names(ctx.get('mi') or [])}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor",
                "",
                "session = session.impute().encode()",
                "est = GradientBoostingClassifier(random_state=0) if "
                f"{is_classification(ctx)} else GradientBoostingRegressor(random_state=0)",
                "session = session.fit(est)",
                "board = session.eval_plots(partition=\"test\", include_importance=True)",
                "print(board)",
                'session.learn("feature-importance", level="intermediate")',
            ),
            what_to_change=("Plot shapes for top features stakeholders will challenge."),
            pitfalls=("Reading average effect shapes under strong interactions as universal laws."),
            decide="Pair global importance with at least one local/shape view for top features.",
            read_steps=("Identify non-monotone shapes.", "Check whether domain experts find them plausible."),
        ),
        L(
            slug="learning-curves-and-capacity",
            stage=5,
            order=30,
            concept_key="training-curves",
            tags=("learning curve", "capacity"),
            plain=(
                "Learning curves show whether more data would help or whether the model family is already saturated.",
            ),
            technical=("session.learning_curve(estimator) varies training size under CV."),
            why=("Collecting more data is expensive - curves tell you if it pays."),
            formula=None,
            calculation=lambda ctx: (
                f"n={fmt_n(ctx.get('rows'))}; if curves still rising at full n, more data may help."
            ),
            session_evidence=lambda ctx: f"rows={fmt_n(ctx.get('rows'))}; eligible={fmt_n(ctx.get('eligible'))}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import LogisticRegression, Ridge",
                "",
                "session = session.impute().encode().scale()",
                "est = LogisticRegression(max_iter=200) if "
                f"{is_classification(ctx)} else Ridge()",
                "curve = session.learning_curve(est, cv=5)",
                "print(curve)",
                'session.learn("training-curves", level="beginner")',
            ),
            what_to_change=("Compare a weak and a strong model family's curves."),
            pitfalls=("Reading a single end-point score as capacity diagnosis."),
            decide="Use learning curves before large data-collection spends.",
            read_steps=("Check gap between train and validation curves.", "See if validation still slopes up."),
        ),
        L(
            slug="causal-caution",
            stage=5,
            order=40,
            concept_key="causal-assumptions",
            tags=("causal",),
            plain=(
                "Predictive success is not a license to say 'if we change X, Y will change'. "
                "That is a causal claim with assumptions EDA cannot finish for you.",
            ),
            technical=(
                "BuildML causal tools exist (fit_causal) but require identification assumptions. "
                "Read causal-eda-boundary before treating associations as levers.",
            ),
            why=("Acting on correlations as causes is how ML creates policy mistakes."),
            formula=None,
            calculation=lambda ctx: (
                f"Strong associations on this sheet (MI): {list_names(ctx.get('mi') or [])}. "
                "None of these are identified causal effects by default."
            ),
            session_evidence=lambda ctx: (
                "Causal readiness is human-gated; association screens are not identification."
            ),
            example_code=lambda ctx: with_starter(
                ctx,
                'session.learn("causal-assumptions", level="beginner")',
                'session.learn("causal-eda-boundary", level="intermediate")',
                "# Only after a causal diagram + assumptions:",
                "# session.fit_causal(...)",
            ),
            what_to_change=("Separate prediction KPIs from intervention KPIs in the project brief."),
            pitfalls=("Shipping feature importances as 'drivers' to executives without causal language."),
            decide="Label every stakeholder-facing claim as predictive or causal - never blur them.",
            read_steps=("List actions you might take.", "Ask which require causal identification."),
        ),
        L(
            slug="handoff-and-monitoring",
            stage=5,
            order=50,
            concept_key="operation-history",
            tags=("monitoring", "handoff"),
            plain=(
                "Shipping is not the end. Handoff means the recipe, thresholds, monitoring, and "
                "rollback plan travel with the model.",
            ),
            technical=(
                "save_pipeline / checkpoints capture fitted plans. Monitor drift, slices, and calibration "
                "against the training reference.",
            ),
            why=("Unmonitored models decay silently."),
            formula=None,
            calculation=lambda ctx: (
                f"Reference profile: n={fmt_n(ctx.get('rows'))}, features~{fmt_n(ctx.get('eligible'))}, "
                f"drift flags now={list_names(ctx.get('drifted') or []) or 'none'}."
            ),
            session_evidence=lambda ctx: (
                f"Engine={(ctx.get('ds') or {}).get('engine')}; record this in the handoff card."
            ),
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import LogisticRegression, Ridge",
                "",
                "session = session.impute().encode().scale()",
                "est = LogisticRegression(max_iter=200) if "
                f"{is_classification(ctx)} else Ridge()",
                "session = session.fit(est)",
                "path = session.save_pipeline(\"artifacts/model_pipeline\")  # <-- change path",
                "print(\"saved\", path)",
                'session.learn("operation-history", level="beginner")',
                "# Monitoring: re-run drift/slice checks on live windows vs this reference.",
            ),
            what_to_change=(
                "Set artifact paths; define alert thresholds for drift and slice gaps.",
                "Include threshold policy and primary metric in the model card.",
            ),
            pitfalls=("Handing over a pickle without the preprocessing plan.", "No owner for alerts."),
            decide="Do not ship without a monitoring owner, reference snapshot, and rollback path.",
            read_steps=(
                "Save pipeline + model card.",
                "List monitors (drift, slices, calibration).",
                "Name on-call owner.",
            ),
        ),
    ]
