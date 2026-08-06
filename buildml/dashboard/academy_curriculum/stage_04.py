"""Stage 04 · Evaluation - what a score is worth."""

from __future__ import annotations

from buildml.dashboard.academy_curriculum._factory import L, rows_blurb, with_starter
from buildml.dashboard.academy_curriculum._helpers import (
    fmt_n,
    fmt_pct,
    is_classification,
    is_regression,
    list_names,
    target_name,
)
from buildml.dashboard.academy_curriculum._types import LessonSpec


def lessons() -> list[LessonSpec]:
    return [*_core(), *_additions()]


def _core() -> list[LessonSpec]:
    return [
        L(
            slug="class-imbalance",
            stage=4,
            order=10,
            concept_key="class-imbalance",
            tags=("imbalance",),
            plain=(
                "When one class dominates, accuracy becomes a trap: predicting the majority always "
                "looks 'good' while missing the rare events you care about.",
            ),
            technical=(
                "Prefer stratified splits, appropriate metrics, and - only if needed - "
                "session.resample(...) inside training. Never resample the test set.",
            ),
            why=("Majority baselines beat careless models on accuracy."),
            formula="prevalence = n_positive / n; majority_accuracy = max(prevalence, 1-prevalence)",
            calculation=lambda ctx: _imb_calc(ctx),
            session_evidence=lambda ctx: _imb_calc(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "print(session.resample_strategies())  # what BuildML can do",
                "session = session.resample(sampler=\"random_oversample\")  # <-- or smote / random_undersample",
                "# Fit only after resample on train; evaluate on untouched test",
                'session.learn("class-imbalance", level="beginner")',
                stratify=True,
            ),
            what_to_change=(
                "Pick metric first (ROC-AUC / PR-AUC / recall@cost).",
                "Try class weights before heavy resampling.",
            ),
            pitfalls=(
                "Upsampling before split (leaks copies into test).",
                "Optimising accuracy under 1% prevalence.",
            ),
            decide="State prevalence and the metric that matches the costly error before modeling.",
            read_steps=("Compute class rates.", "Compare to majority baseline.", "Choose metric/cost."),
        ),
        L(
            slug="target-distribution",
            stage=4,
            order=20,
            concept_key="class-imbalance",
            tags=("target", "distribution"),
            plain=("Know the shape of y: class mix for classification, centre/spread/tails for regression."),
            technical=("Target screens in EDA summarise y; modeling choices should cite them."),
            why=("Metrics and transforms depend on y's shape."),
            formula=None,
            calculation=lambda ctx: _target_dist(ctx),
            session_evidence=lambda ctx: _target_dist(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "report = session.eda(include_plots=False, show=False)",
                "print(report.to_dict().get(\"target\"))",
            ),
            what_to_change=("For regression, consider log-target only with domain justification."),
            pitfalls=("Transforming y without inverting for reporting."),
            decide="Document y's distribution and any transform applied to it.",
            read_steps=("Read target summary.", "Check missing labels separately."),
        ),
        L(
            slug="metric-selection",
            stage=4,
            order=30,
            concept_key="model-selection",
            tags=("metrics",),
            plain=(
                "The metric is the definition of better. If it does not match the decision cost, "
                "model selection optimises the wrong thing.",
            ),
            technical=(
                "Classification: ROC-AUC, PR-AUC, log-loss, cost-weighted recall/precision. "
                "Regression: RMSE, MAE, MAPE (carefully). Pick one primary.",
            ),
            why=("Leaderboards without a decision metric produce pretty but useless winners."),
            formula=None,
            calculation=lambda ctx: (
                "Primary metric suggestion for this session: "
                + (
                    "PR-AUC or cost-sensitive recall if minority class matters; else ROC-AUC / log-loss."
                    if is_classification(ctx)
                    else "MAE if tails are noise; RMSE if large errors are especially costly."
                    if is_regression(ctx)
                    else "define a supervised target first."
                )
            ),
            session_evidence=lambda ctx: f"task={ctx.get('task')}; target={target_name(ctx)}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import LogisticRegression, Ridge",
                "",
                "session = session.impute().encode().scale()",
                "est = LogisticRegression(max_iter=200) if "
                f"{is_classification(ctx)} else Ridge()",
                "session = session.fit(est)",
                "card = session.evaluate(partition=\"test\")",
                "print(card.metrics)",
                'session.learn("model-selection", level="beginner")',
            ),
            what_to_change=("Set the primary metric to match business cost; ignore vanity metrics."),
            pitfalls=("Optimising accuracy under imbalance.", "Reporting many metrics without a primary."),
            decide="Freeze one primary metric before comparing models.",
            read_steps=("Map false positive/negative costs.", "Align metric to that map."),
        ),
        L(
            slug="thresholds-and-costs",
            stage=4,
            order=40,
            concept_key="thresholds",
            tags=("threshold", "cost"),
            plain=(
                "A probability is not a decision. The threshold turns scores into actions and must "
                "reflect costs, capacity, and policy.",
            ),
            technical=("session.tune_threshold(...) searches operating points on a validation partition."),
            why=("0.5 is rarely the right cut."),
            formula="choose τ maximising utility(TP,FP,FN,TN; costs) on validation",
            calculation=lambda ctx: (
                "Classification thresholds apply only when task=classification; "
                f"here task={ctx.get('task')}."
                if is_classification(ctx)
                else "Regression uses decision rules on continuous predictions - define them explicitly."
            ),
            session_evidence=lambda ctx: f"task={ctx.get('task')}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import LogisticRegression",
                "",
                "session = session.impute().encode().scale()",
                "session = session.fit(LogisticRegression(max_iter=200))",
                "threshold = session.tune_threshold(partition=\"validation\")  # needs val split",
                "print(threshold)",
                'session.learn("thresholds", level="beginner")',
                stratify=True,
            ),
            what_to_change=("Set costs / capacity constraints; tune on validation, freeze for test."),
            pitfalls=("Tuning threshold on the test set.", "Ignoring capacity (how many you can act on)."),
            decide="Publish the threshold policy with the model, not just the probabilities.",
            read_steps=("Plot precision-recall vs threshold.", "Incorporate action capacity."),
        ),
        L(
            slug="baselines",
            stage=4,
            order=50,
            concept_key="baselines",
            tags=("baseline",),
            plain=(
                "A baseline is the simplest reasonable predictor: majority class, mean target, "
                "or a tiny linear model. If you cannot beat it, you do not have a win.",
            ),
            technical=("Compare every candidate to an explicit baseline under the same split and metric."),
            why=("Complex models often lose to sane defaults once leakage is removed."),
            formula=None,
            calculation=lambda ctx: (
                _imb_calc(ctx)
                if is_classification(ctx)
                else f"Regression baseline: predict train mean/median of {target_name(ctx)}."
            ),
            session_evidence=lambda ctx: f"task={ctx.get('task')}; n={fmt_n(ctx.get('rows'))}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.dummy import DummyClassifier, DummyRegressor",
                "from sklearn.linear_model import LogisticRegression, Ridge",
                "",
                "session = session.impute().encode().scale()",
                "dummy = DummyClassifier(strategy=\"most_frequent\") if "
                f"{is_classification(ctx)} else DummyRegressor(strategy=\"mean\")",
                "session.compare_models(",
                "    {",
                "        \"baseline\": dummy,",
                "        \"linear\": LogisticRegression(max_iter=200) if "
                f"{is_classification(ctx)} else Ridge(),",
                "    }",
                ")",
                'session.learn("baselines", level="beginner")',
            ),
            what_to_change=("Keep the baseline in every comparison table."),
            pitfalls=("Reporting model scores without the majority/mean baseline beside them."),
            decide="Do not ship a model that does not beat a documented baseline on the primary metric.",
            read_steps=("Compute baseline score first.", "Demand meaningful lift."),
        ),
        L(
            slug="calibration",
            stage=4,
            order=60,
            concept_key="probability-calibration",
            tags=("calibration",),
            plain=(
                "A score of 0.8 should mean 'about 80% chance' if you treat it as a probability. "
                "Many classifiers are not born that way.",
            ),
            technical=("session.calibration(...) diagnoses/adjusts probability quality on a holdout."),
            why=("Bad calibration breaks threshold policies and capacity planning."),
            formula="perfect calibration: P(Y=1 | s(x)=p) ~ p",
            calculation=lambda ctx: (
                "Relevant for classification probability outputs; "
                f"task={ctx.get('task')}."
            ),
            session_evidence=lambda ctx: f"task={ctx.get('task')}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import LogisticRegression",
                "",
                "session = session.impute().encode().scale()",
                "session = session.fit(LogisticRegression(max_iter=200))",
                "cal = session.calibration(partition=\"validation\")",
                "print(cal)",
                'session.learn("probability-calibration", level="beginner")',
                stratify=True,
            ),
            what_to_change=("Calibrate on validation; re-check after major retrains."),
            pitfalls=("Calibrating on test.", "Trusting raw margins from boosted trees as probabilities."),
            decide="If decisions use probabilities, require a calibration check in the release gate.",
            read_steps=("Read reliability diagrams / ECE if available.", "Retune thresholds after calibration."),
        ),
    ]


def _additions() -> list[LessonSpec]:
    return [
        L(
            slug="confusion-matrix",
            stage=4,
            order=70,
            concept_key="thresholds",
            tags=("confusion",),
            plain=("The confusion matrix counts TP/FP/FN/TN at a chosen threshold - the anatomy of errors."),
            technical=("session.evaluate(...) exposes confusion diagnostics for classifiers."),
            why=("Single scalar metrics hide which mistakes you make."),
            formula="precision=TP/(TP+FP); recall=TP/(TP+FN)",
            calculation=lambda ctx: (
                f"Use after fit on task={ctx.get('task')}; needs a thresholded classifier."
            ),
            session_evidence=lambda ctx: f"task={ctx.get('task')}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import LogisticRegression",
                "session = session.impute().encode().scale().fit(LogisticRegression(max_iter=200))",
                "card = session.evaluate(partition=\"test\")",
                "print(card.metrics)",
                "print(getattr(card, \"diagnostics\", None))",
                stratify=True,
            ),
            what_to_change=("Inspect errors at the operating threshold you will ship."),
            pitfalls=("Reading a confusion matrix at 0.5 when you will deploy another cut."),
            decide="Publish the confusion matrix at the production threshold.",
            read_steps=("Compute precision/recall from the matrix.", "Sample FN/FP rows."),
        ),
        L(
            slug="ranking-curves",
            stage=4,
            order=80,
            concept_key="model-selection",
            tags=("ROC", "PR"),
            plain=("ROC and PR curves show ranking quality across thresholds; PR is often kinder to reality under imbalance."),
            technical=("Prefer PR when positives are rare; ROC can look optimistic."),
            why=("Threshold-free ranking metrics still must match the decision."),
            formula=None,
            calculation=lambda ctx: (
                f"Imbalance context: {_imb_calc(ctx)}" if is_classification(ctx) else "Classification-only topic."
            ),
            session_evidence=lambda ctx: _imb_calc(ctx) if is_classification(ctx) else "n/a",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import LogisticRegression",
                "session = session.impute().encode().scale().fit(LogisticRegression(max_iter=200))",
                "board = session.eval_plots(partition=\"test\", include_learning_curve=False)",
                "print(board)",
                stratify=True,
            ),
            what_to_change=("Choose ROC vs PR from prevalence and costs."),
            pitfalls=("Boasting ROC-AUC at 0.5% prevalence without PR."),
            decide="Report the ranking curve that matches prevalence and decision style.",
            read_steps=("Compare PR-AUC vs ROC-AUC.", "Mark the operating point."),
        ),
        L(
            slug="multiclass-and-averaging",
            stage=4,
            order=90,
            concept_key="model-selection",
            tags=("multiclass",),
            plain=("With more than two classes, 'average' metrics hide per-class failures unless you specify micro/macro/weighted."),
            technical=("Declare averaging and inspect per-class recalls before shipping."),
            why=("Macro-averages punish ignoring rare classes; micro can hide them."),
            formula=None,
            calculation=lambda ctx: _multiclass_calc(ctx),
            session_evidence=lambda ctx: _multiclass_calc(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import LogisticRegression",
                "session = session.impute().encode().scale().fit(LogisticRegression(max_iter=400))",
                "print(session.evaluate(partition=\"test\").metrics)",
                stratify=True,
            ),
            what_to_change=("Pick micro/macro/weighted deliberately; monitor worst class."),
            pitfalls=("Reporting only overall accuracy for multiclass."),
            decide="Publish per-class metrics alongside the average you optimise.",
            read_steps=("Count classes.", "Identify the worst class recall."),
        ),
        L(
            slug="residual-diagnostics",
            stage=4,
            order=100,
            concept_key="overfitting",
            tags=("residuals",),
            plain=("Residuals show where the model systematically fails - not just how wrong on average."),
            technical=("For regression, plot residual vs predicted and vs key features; check heteroscedasticity."),
            why=("Good RMSE can hide large structured mistakes in a segment."),
            formula="residual = y - ŷ",
            calculation=lambda ctx: (
                f"Focus for regression task; here task={ctx.get('task')}."
            ),
            session_evidence=lambda ctx: f"task={ctx.get('task')}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import Ridge",
                "session = session.impute().scale().fit(Ridge())",
                "card = session.evaluate(partition=\"test\")",
                "print(card.metrics)",
                "board = session.eval_plots(partition=\"test\")",
                stratify=False,
            ),
            what_to_change=("Investigate structured residual patterns before adding complexity."),
            pitfalls=("Only reading RMSE.", "Fixing residuals by peeking at test rows' identities."),
            decide="Require residual plots for regression releases.",
            read_steps=("Plot residual vs ŷ.", "Slice residuals by key segments."),
        ),
        L(
            slug="uncertainty-intervals",
            stage=4,
            order=110,
            concept_key="probabilistic-uncertainty",
            tags=("intervals",),
            plain=("A point prediction without uncertainty invites overconfident decisions."),
            technical=(
                "Use probabilistic models / calibration / quantile approaches when stakes require intervals. "
                "session.learn('probabilistic-uncertainty') introduces the vocabulary."
            ),
            why=("Capacity planning needs ranges, not bravado."),
            formula=None,
            calculation=lambda ctx: rows_blurb(ctx) + f"; task={ctx.get('task')}.",
            session_evidence=lambda ctx: f"task={ctx.get('task')}.",
            example_code=lambda ctx: with_starter(
                ctx,
                'session.learn("probabilistic-uncertainty", level="beginner")',
                "# Prefer models that expose predictive distributions when decisions need intervals.",
            ),
            what_to_change=("Choose interval method matching stake level."),
            pitfalls=("Showing +/-sigma from training residuals as if it were predictive uncertainty."),
            decide="If actions need confidence, ship an interval method - not only a point forecast.",
            read_steps=("Define the decision's tolerance for uncertainty.", "Validate coverage on holdout."),
        ),
        L(
            slug="slice-evaluation",
            stage=4,
            order=120,
            concept_key="evaluation-partitions",
            tags=("slices", "fairness"),
            plain=("Overall metrics can look fine while one segment fails. Slice evaluation is mandatory for trust."),
            technical=("session.error_slices(...) and fairness evaluations quantify segment gaps."),
            why=("Hidden segment failure is how models harm people and businesses."),
            formula=None,
            calculation=lambda ctx: (
                f"Candidate slice columns: {list_names(ctx.get('categorical') or [])}."
            ),
            session_evidence=lambda ctx: f"categoricals={list_names(ctx.get('categorical') or [])}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import LogisticRegression, Ridge",
                "session = session.impute().encode().scale()",
                "est = LogisticRegression(max_iter=200) if "
                f"{is_classification(ctx)} else Ridge()",
                "session = session.fit(est)",
                f"slices = session.error_slices(by=\"{ (ctx.get('categorical') or ['<segment_col>'])[0] }\", partition=\"test\")",
                "print(slices)",
            ),
            what_to_change=("Pre-declare critical slices (region, channel, cohort)."),
            pitfalls=("Only reporting the global mean metric."),
            decide="Set minimum acceptable performance on critical slices before release.",
            read_steps=("List critical segments.", "Compare metric gaps vs overall."),
        ),
    ]


def _imb_calc(ctx: dict) -> str:
    target = ctx.get("target")
    if not is_classification(ctx) or not isinstance(target, dict):
        return "Class imbalance applies to classification; this session is not a classification target."
    classes = target.get("classes") or []
    if not classes:
        return f"Classification target '{target_name(ctx)}' without class counts in-report."
    total = sum(int(c.get("count") or 0) for c in classes) or 1
    parts = [f"{c.get('label')}: {fmt_pct(int(c.get('count') or 0)/total)}" for c in classes]
    majority = max(int(c.get("count") or 0) for c in classes) / total
    return (
        "Class mix - "
        + "; ".join(parts)
        + f". Majority baseline accuracy ~ {fmt_pct(majority)}."
    )


def _target_dist(ctx: dict) -> str:
    if not ctx.get("has_target"):
        return "No target declared."
    if is_classification(ctx):
        return _imb_calc(ctx)
    stats = (ctx.get("target") or {}).get("stats") or {}
    return (
        f"Regression target '{target_name(ctx)}'"
        + (f", median={stats.get('median')}" if stats.get("median") is not None else "")
        + "."
    )


def _multiclass_calc(ctx: dict) -> str:
    target = ctx.get("target")
    if not is_classification(ctx) or not isinstance(target, dict):
        return "Not a classification task in this session."
    classes = target.get("classes") or []
    n = len(classes)
    if n <= 2:
        return f"Binary (or ≤2) class setup with {n} labelled classes in summary - multiclass averaging not central."
    return f"{n} classes observed: {list_names(classes)}. Choose micro/macro/weighted deliberately."
