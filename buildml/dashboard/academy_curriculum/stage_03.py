"""Stage 03 · Validation - what the evidence is allowed to certify."""

from __future__ import annotations

from buildml.dashboard.academy_curriculum._factory import L, rows_blurb, with_starter
from buildml.dashboard.academy_curriculum._helpers import (
    code_block,
    first_feature,
    first_numeric,
    fmt_n,
    fmt_pct,
    is_classification,
    list_names,
    target_name,
)
from buildml.dashboard.academy_curriculum._types import LessonSpec


def lessons() -> list[LessonSpec]:
    return [*_core(), *_additions()]


def _core() -> list[LessonSpec]:
    return [
        L(
            slug="data-splitting",
            stage=3,
            order=10,
            concept_key="data-splitting",
            tags=("split",),
            plain=(
                "A split assigns each row a job: train a model, guide choices, or assess honestly. "
                "A test set is valuable only because it stayed untouched.",
            ),
            technical=(
                "session.split(...) draws random partitions; inject_split adopts external indices. "
                "Full-frame EDA describes observed rows - it is not train-fitted transform evidence.",
            ),
            why=("Without a frozen test partition, metrics become marketing."),
            formula="n_test ~ floor(n x test_size) for fractional test_size",
            calculation=lambda ctx: (
                f"{fmt_n(ctx.get('rows'))} of {fmt_n(ctx.get('rowsTotal'))} rows examined"
                + (" (sampled). " if ctx.get("sampled") else ". ")
                + f"A 20% holdout would be ~{fmt_n(int(round(0.2 * (ctx.get('rows') or 0))))} rows. "
                "Every number on this sheet is full-frame observation unless noted."
            ),
            session_evidence=lambda ctx: rows_blurb(ctx)
            + ("; sampled profile" if ctx.get("sampled") else "; full extract profile"),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "session = (",
                "    Session.ingest(pd.read_csv(\"your_data.csv\"))  # <-- change",
                "    .set_roles({",
                f'        "{target_name(ctx)}": "target",',
                f'        "{first_feature(ctx)}": "feature",',
                "    })",
                ")",
                'session.explain("split", moment="before")',
                "session = session.split(",
                "    test_size=0.2,  # <-- change fraction or absolute count",
                "    validation_size=0.2,  # optional third partition",
                f"    stratify={str(is_classification(ctx))},  # True for classification usually",
                "    random_state=0,",
                ")",
                "session.assert_can_fit(\"train\")",
            ),
            what_to_change=(
                "Adjust test/validation sizes; set stratify for classification.",
                "Use inject_split for time/group splits computed outside BuildML.",
            ),
            pitfalls=lambda ctx: [
                "Reusing the test set to choose models - it becomes validation.",
                "Random splitting when rows share groups or time order.",
                (
                    f"Splitting only {fmt_n(ctx.get('rows'))} rows: prefer CV when holdouts are tiny."
                    if int(ctx.get("rows") or 0) < 300
                    else "Reading full-frame EDA as evidence about a fitted pipeline."
                ),
            ],
            decide="Freeze membership before fitting any reusable transformer.",
            read_steps=(
                "Choose split strategy from dependence structure (IID / group / time).",
                "Never peek at test to pick preprocessing.",
            ),
        ),
        L(
            slug="stratification",
            stage=3,
            order=20,
            concept_key="data-splitting",
            tags=("stratify",),
            plain=(
                "Stratification keeps a distribution - usually the label - similar across split members. "
                "For rare classes it prevents empty test positives by bad luck.",
            ),
            technical=("session.split(stratify=True) stratifies on the target when classification."),
            why=("Unstratified tiny positives make metrics lottery tickets."),
            formula=None,
            calculation=lambda ctx: _strat_calc(ctx),
            session_evidence=lambda ctx: _strat_calc(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "# stratify=True is already set for classification in the starter above when applicable",
                "print(session.split_plan)",
                'session.learn("data-splitting", level="beginner")',
                stratify=True,
            ),
            what_to_change=("Stratify on target; only add secondary strata if levels are thick enough."),
            pitfalls=("Stratifying on a rare category until the split becomes infeasible."),
            decide="For classification, default to stratified splits unless dependence structure forbids it.",
            read_steps=("Compare class rates train vs test after splitting.", "Watch rare levels."),
        ),
        L(
            slug="cross-validation",
            stage=3,
            order=30,
            concept_key="cross-validation",
            tags=("CV",),
            plain=(
                "Cross-validation rotates which fold is held out so you get several honest scores "
                "instead of one lucky split.",
            ),
            technical=(
                "session.cv_score(...) and nested_cv_score(...) evaluate estimators with fold discipline. "
                "Preprocessing must be inside the fold (BuildML train-fitted transforms respect the active split).",
            ),
            why=("Single holdouts are noisy; nested CV separates model selection from evaluation."),
            formula="CV score ~ mean_i metric(model_fit(train_(-i)), test_(i))",
            calculation=lambda ctx: (
                f"With n={fmt_n(ctx.get('rows'))}, 5-fold test folds are ~"
                f"{fmt_n(max(int((ctx.get('rows') or 0) / 5), 1))} rows each."
            ),
            session_evidence=lambda ctx: rows_blurb(ctx) + f"; task={ctx.get('task')}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.linear_model import LogisticRegression, Ridge",
                "",
                "session = session.impute(strategy=\"median\").encode(method=\"onehot\").scale()",
                "estimator = LogisticRegression(max_iter=200) if "
                f"{is_classification(ctx)} else Ridge()",
                "cv = session.cv_score(estimator, cv=5)  # <-- tune cv",
                "print(cv)",
                "# For tuning + honest outer score:",
                "# nested = session.nested_cv_score(estimator, param_grid={...})",
            ),
            what_to_change=("Set cv folds; use nested_cv_score when selecting models/hyperparameters."),
            pitfalls=("Preprocessing outside folds.", "Using test-set CV as if it were nested."),
            decide="Use CV (nested when tuning) instead of a single fragile holdout on small n.",
            read_steps=("Match fold strategy to grouping/time.", "Keep transforms train-fold-only."),
        ),
        L(
            slug="dataset-drift",
            stage=3,
            order=40,
            concept_key="dataset-drift",
            tags=("drift",),
            plain=(
                "Drift means the world the model sees later is not the world it trained on - "
                "features shift, labels shift, or both.",
            ),
            technical=(
                "EDA drift screens compare slices/time windows. Monitoring must continue after deploy.",
            ),
            why=("Silent drift makes yesterday's metrics lie about tomorrow."),
            formula=None,
            calculation=lambda ctx: (
                f"Drift-flagged columns in this report: {list_names(ctx.get('drifted') or []) or 'none flagged'}."
            ),
            session_evidence=lambda ctx: (
                f"Drift flags: {list_names(ctx.get('drifted') or []) or 'none'}; "
                "absence of flags is not a warranty about future traffic."
            ),
            example_code=lambda ctx: with_starter(
                ctx,
                "report = session.eda(include_plots=False, show=False)",
                "print(report.to_dict().get(\"drift\"))",
                'session.learn("dataset-drift", level="beginner")',
                "# After deploy: compare live feature distributions to the training snapshot.",
            ),
            what_to_change=("Define reference vs analysis windows; set monitoring alerts on key features."),
            pitfalls=("Assuming IID forever because training CV looked fine."),
            decide="Name the reference distribution and the monitoring plan before shipping.",
            read_steps=("Inspect drift section of the EDA report.", "Prioritise features with high MI/importance."),
        ),
        L(
            slug="leakage",
            stage=3,
            order=50,
            concept_key="leakage-boundary",
            tags=("leakage",),
            plain=(
                "Leakage is when information from the future - or from evaluation rows - sneaks into training. "
                "Scores look amazing; production does not.",
            ),
            technical=(
                "Train-only learning: imputers, encoders, scalers, selectors, and estimators fit on train "
                "(or inner folds) and apply frozen plans elsewhere.",
            ),
            why=("Leakage is the most common way to fool yourself with tabular ML."),
            formula="θ = L(train); score = m(f_θ(eval)) - never L(train ∪ eval)",
            calculation=lambda ctx: (
                "Leakage candidates often include post-outcome fields, target encodings fitted globally, "
                f"and id-like columns ({list_names(ctx.get('idLike') or [])})."
            ),
            session_evidence=lambda ctx: (
                f"Id-like={list_names(ctx.get('idLike') or [])}; "
                "treat full-frame fills/encodes as contaminated until proven train-folded."
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "session = Session.ingest(pd.read_csv(\"your_data.csv\"))",
                "session = session.set_roles({",
                f'    "{target_name(ctx)}": "target",',
                f'    "{first_feature(ctx)}": "feature",',
                *(
                    [f'    "{n}": "id",' for n in (ctx.get("idLike") or [])[:2]]
                    or ['    # mark identifiers / post-outcome fields as id or ignore']
                ),
                "})",
                "# Split FIRST, then fit transforms - order matters",
                "session = (",
                "    session.split(test_size=0.2, stratify=True, random_state=0)",
                "    .impute(strategy=\"median\")",
                "    .encode(method=\"onehot\")",
                "    .scale(method=\"standard\")",
                ")",
                'session.learn("leakage-boundary", level="beginner")',
            ),
            what_to_change=("Remove post-outcome features; keep transform fit order after split."),
            pitfalls=(
                "Cleaning the whole CSV then splitting last.",
                "Refitting encoders after seeing test errors.",
            ),
            decide="Draw the leakage boundary: which fields exist at score time, and which rows may train transformers.",
            read_steps=(
                "List columns unavailable at prediction time -> ignore.",
                "Confirm every fitted plan is train-only.",
            ),
        ),
        L(
            slug="temporal-structure",
            stage=3,
            order=60,
            concept_key="data-splitting",
            tags=("time", "split"),
            plain=("When rows are ordered in time, random splits let the future train the past."),
            technical=(
                "Prefer time-ordered holdouts via inject_split with indices sorted by timestamp. "
                "Rolling/expanding CV is the usual honest design.",
            ),
            why=("Random CV on time series invents impossible foresight."),
            formula=None,
            calculation=lambda ctx: (
                f"Time column: {(ctx.get('timeCol') or {}).get('name') if ctx.get('timeCol') else 'not detected - if your problem is temporal, declare it'}."
            ),
            session_evidence=lambda ctx: f"Temporal axis detected: {bool(ctx.get('timeCol'))}.",
            example_code=lambda ctx: code_block(
                "import numpy as np",
                "import pandas as pd",
                "from buildml import Session",
                "",
                "frame = pd.read_csv(\"your_data.csv\")",
                f"t = \"{(ctx.get('timeCol') or {}).get('name') or '<timestamp>'}\"",
                "frame = frame.sort_values(t)",
                "session = Session.ingest(frame).set_roles({",
                f'    t: "time", "{target_name(ctx)}": "target", "{first_feature(ctx)}": "feature",',
                "})",
                "n = len(frame)",
                "cut = int(n * 0.8)  # <-- change",
                "idx = np.arange(n)",
                "session = session.inject_split(",
                "    train_indices=idx[:cut].tolist(),",
                "    test_indices=idx[cut:].tolist(),",
                ")",
            ),
            what_to_change=("Set the time column and cutpoints; consider gap between train and test."),
            pitfalls=("Shuffling before a time split.", "Using future lags as features."),
            decide="If time matters, forbid random splits; use ordered holdouts.",
            read_steps=("Plot outcome over time.", "Ensure features are as-of safe."),
        ),
        L(
            slug="group-structure",
            stage=3,
            order=70,
            concept_key="data-splitting",
            tags=("groups",),
            plain=(
                "When many rows share an entity (customer, hospital, device), random splits leak "
                "entity-specific quirks into both train and test.",
            ),
            technical=(
                "Assign a group role and split by group (compute indices outside, inject_split). "
                "Grouped CV is the matching evaluation design.",
            ),
            why=("Entity leakage inflates scores and fails on new entities."),
            formula=None,
            calculation=lambda ctx: (
                f"Candidate group keys among id-like columns: {list_names(ctx.get('idLike') or [])}."
            ),
            session_evidence=lambda ctx: (
                f"Id-like columns (review as group keys): {list_names(ctx.get('idLike') or [])}."
            ),
            example_code=lambda ctx: code_block(
                "import numpy as np",
                "import pandas as pd",
                "from buildml import Session",
                "",
                "frame = pd.read_csv(\"your_data.csv\")",
                f"group = \"{(ctx.get('idLike') or ['<entity_id>'])[0]}\"  # <-- group key",
                "session = Session.ingest(frame).set_roles({",
                f'    group: "group", "{target_name(ctx)}": "target", "{first_feature(ctx)}": "feature",',
                "})",
                "groups = frame[group].unique()",
                "rng = np.random.default_rng(0)",
                "rng.shuffle(groups)",
                "cut = int(0.8 * len(groups))",
                "train_g, test_g = set(groups[:cut]), set(groups[cut:])",
                "train_idx = frame.index[frame[group].isin(train_g)].tolist()",
                "test_idx = frame.index[frame[group].isin(test_g)].tolist()",
                "session = session.inject_split(train_indices=train_idx, test_indices=test_idx)",
            ),
            what_to_change=("Set the group key; ensure no entity appears in both train and test."),
            pitfalls=("Stratifying labels while still leaking groups across folds."),
            decide="If rows share entities, split and validate by entity, not by row.",
            read_steps=("Estimate rows per entity.", "Forbid entity overlap across partitions."),
        ),
        L(
            slug="diagnostic-uncertainty",
            stage=3,
            order=80,
            concept_key="diagnostic-uncertainty",
            tags=("uncertainty",),
            plain=(
                "Diagnostics are uncertain: sampling error, analyzer limits, and missing context "
                "mean a green cell is not a certificate.",
            ),
            technical=(
                "BuildML findings are evidence with severity, not approvals. "
                "Human gates exist for questions data cannot answer.",
            ),
            why=("Over-trusting screens creates false confidence."),
            formula=None,
            calculation=lambda ctx: (
                f"Profile covers {rows_blurb(ctx)}; sampled={bool(ctx.get('sampled'))}. "
                "Treat every automatic flag as a prompt for judgement."
            ),
            session_evidence=lambda ctx: (
                f"Engine={(ctx.get('ds') or {}).get('engine')}; sampled={bool(ctx.get('sampled'))}."
            ),
            example_code=lambda ctx: with_starter(
                ctx,
                "report = session.eda(include_plots=False, show=False)",
                "for f in report.findings[:12]:",
                "    print(f.severity, f.title)",
                'session.learn("diagnostic-uncertainty", level="beginner")',
            ),
            what_to_change=("Record which findings you accept/reject and why."),
            pitfalls=("Equating 'no finding' with 'no risk'."),
            decide="Separate answerable numeric questions from human policy questions.",
            read_steps=("Read finding severity and assumptions.", "Escalate human gates explicitly."),
        ),
        L(
            slug="outlier-screens",
            stage=3,
            order=90,
            concept_key="outlier-handling",
            tags=("outliers",),
            plain=(
                "Outlier screens flag extreme values. They do not know whether those values are errors, "
                "rare truth, or the very events you care about.",
            ),
            technical=(
                "session.handle_outliers(method='iqr'|'zscore', action='detect'|'cap'|'drop') "
                "is train-aware after split. Multivariate anomaly screens may also appear in EDA.",
            ),
            why=("Blind dropping can erase the minority class you want to detect."),
            formula="IQR rule: flag x < Q1-1.5·IQR or x > Q3+1.5·IQR",
            calculation=lambda ctx: _outlier_calc(ctx),
            session_evidence=lambda ctx: _outlier_calc(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "session = session.handle_outliers(",
                f"    columns=[\"{first_numeric(ctx)}\"],  # <-- change",
                "    method=\"iqr\",",
                "    action=\"detect\",  # start with detect before cap/drop",
                "    iqr_multiplier=1.5,",
                ")",
                'session.learn("outlier-handling", level="beginner")',
            ),
            what_to_change=("Choose detect vs cap vs drop per column with domain owners."),
            pitfalls=("Dropping outliers that are the positive class.", "Fitting bounds on full data."),
            decide="Classify each flagged extreme as error / rare valid / target-relevant before acting.",
            read_steps=("Read per-column IQR rates.", "Inspect multivariate anomaly counts if present."),
        ),
    ]


def _additions() -> list[LessonSpec]:
    return [
        L(
            slug="pipeline-order",
            stage=3,
            order=100,
            concept_key="encoding-imputation-scaling",
            tags=("pipeline",),
            plain=("Order matters: split -> impute -> encode -> scale -> select -> model."),
            technical=(
                "BuildML session chaining encourages train-fitted order. "
                "Encoding before imputing categoricals, or scaling before imputing, creates avoidable bugs.",
            ),
            why=("Wrong order leaks or invents impossible states."),
            formula=None,
            calculation=lambda ctx: (
                f"This frame: missing_cells={fmt_n(ctx.get('missingCells'))}, "
                f"categoricals={fmt_n(len(ctx.get('categorical') or []))}, "
                f"numerics={fmt_n(len(ctx.get('numeric') or []))}."
            ),
            session_evidence=lambda ctx: rows_blurb(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "# Canonical classical order (adjust to your columns)",
                "session = (",
                "    session",
                "    .impute(strategy=\"median\")",
                "    .encode(method=\"onehot\")",
                "    .scale(method=\"standard\")",
                ")",
                'session.learn("encoding-imputation-scaling", level="beginner")',
            ),
            what_to_change=("Insert select_features / handle_outliers where your recipe needs them - still after split."),
            pitfalls=("Scaling before impute.", "Selecting features before encoding when screens need numeric X."),
            decide="Write the pipeline order once and refuse ad-hoc notebook rearrangements after seeing test scores.",
            read_steps=("List steps.", "Check each is train-fitted."),
        ),
        L(
            slug="nested-validation",
            stage=3,
            order=110,
            concept_key="cross-validation",
            tags=("nested CV",),
            plain=("When you tune, you need an outer score that never saw the tuning choices."),
            technical=("session.nested_cv_score separates inner selection from outer evaluation."),
            why=("Tuning on the same CV you report optimistically biases the metric."),
            formula=None,
            calculation=lambda ctx: (
                f"n={fmt_n(ctx.get('rows'))}: nested CV is expensive but honest when searching broadly."
            ),
            session_evidence=lambda ctx: rows_blurb(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
                "",
                "est = RandomForestClassifier(random_state=0) if "
                f"{is_classification(ctx)} else RandomForestRegressor(random_state=0)",
                "result = session.nested_cv_score(",
                "    est,",
                "    param_grid={\"n_estimators\": [50, 100], \"max_depth\": [3, 6]},  # <-- change",
                ")",
                "print(result)",
            ),
            what_to_change=("Set param grids narrowly; escalate compute only when needed."),
            pitfalls=("Reporting inner-CV best scores as final performance."),
            decide="If you tune, report outer nested scores (or a frozen final test once).",
            read_steps=("Separate selection metric from reporting metric.", "Keep test untouched."),
        ),
        L(
            slug="sample-size-and-power",
            stage=3,
            order=120,
            concept_key="diagnostic-uncertainty",
            tags=("sample size",),
            plain=("Small n makes every metric noisy. Fancy models do not create information."),
            technical=("With rare positives, effective sample size is the minority count, not n."),
            why=("Underpowered comparisons crown lucky winners."),
            formula=None,
            calculation=lambda ctx: _power_calc(ctx),
            session_evidence=lambda ctx: _power_calc(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                'session.learn("diagnostic-uncertainty", level="intermediate")',
                "# Prefer simpler models + nested CV when minority counts are tiny.",
            ),
            what_to_change=("Compute minority class counts; set minimums before complex search."),
            pitfalls=("Claiming 0.01 AUC gains on 40 positives."),
            decide="State the effective sample size that justifies your model complexity.",
            read_steps=("Count positives / unique groups.", "Match model capacity to n."),
        ),
        L(
            slug="multiple-comparisons",
            stage=3,
            order=130,
            concept_key="diagnostic-uncertainty",
            tags=("multiple testing",),
            plain=("Look at enough slices and something will look significant by chance."),
            technical=("Pre-register primary metrics; treat other slices as exploratory."),
            why=("P-hacking via dashboard clicking invents false readiness."),
            formula=None,
            calculation=lambda ctx: (
                f"With {fmt_n(ctx.get('colCount'))} columns and many screens, expect spurious flags."
            ),
            session_evidence=lambda ctx: rows_blurb(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "# Pick ONE primary metric before fitting competitors",
                "primary_metric = \"roc_auc\" if "
                f"{is_classification(ctx)} else \"rmse\"  # <-- change",
                "print(\"primary_metric=\", primary_metric)",
            ),
            what_to_change=("Pre-declare primary metric and primary segment."),
            pitfalls=("Mining all slices then reporting the best as confirmatory."),
            decide="Separate confirmatory vs exploratory analyses in writing.",
            read_steps=("List planned comparisons.", "Discount unplanned wins."),
        ),
        L(
            slug="reproducibility",
            stage=3,
            order=140,
            concept_key="reproducibility",
            tags=("seeds",),
            plain=("If you cannot rerun it, you cannot trust it. Seeds, package versions, and data pins matter."),
            technical=(
                "Pass random_state through split/encode/search; checkpoint sessions; "
                "record BuildML/engine versions from the sheet kicker."
            ),
            why=("Irreproducible wins are not wins."),
            formula=None,
            calculation=lambda ctx: (
                f"Engine={(ctx.get('ds') or {}).get('engine')}; "
                f"mode={(ctx.get('ds') or {}).get('version')} - record these in your run card."
            ),
            session_evidence=lambda ctx: f"ds={ctx.get('ds')}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "SEED = 0  # <-- change deliberately, not silently",
                "session = session.split(test_size=0.2, stratify=True, random_state=SEED)",
                "# Prefer checkpoint / save_pipeline when handing off",
                'session.learn("reproducibility", level="beginner")',
            ),
            what_to_change=("Fix seeds; pin data snapshot ids; log package versions."),
            pitfalls=("Resetting seeds until the metric looks good."),
            decide="Ship a run card: data pin, seed, code version, primary metric.",
            read_steps=("Confirm random_state on stochastic steps.", "Save pipelines for serve parity."),
        ),
        L(
            slug="shift-taxonomy",
            stage=3,
            order=150,
            concept_key="dataset-drift",
            tags=("covariate shift", "label shift"),
            plain=(
                "Not all drift is the same: covariate shift, label shift, and concept drift "
                "need different responses.",
            ),
            technical=(
                "Covariate: P(X) changes. Label: P(Y) changes. Concept: P(Y|X) changes. "
                "Monitoring should say which you suspect.",
            ),
            why=("Retraining on more data does not fix concept drift if the world rewrote the rules."),
            formula=None,
            calculation=lambda ctx: (
                f"Drift-flagged: {list_names(ctx.get('drifted') or []) or 'none'}. "
                "Classify each as covariate / label / concept with domain context."
            ),
            session_evidence=lambda ctx: f"Drift columns: {list_names(ctx.get('drifted') or [])}.",
            example_code=lambda ctx: with_starter(
                ctx,
                'session.learn("dataset-drift", level="intermediate")',
                "# Document: what changed - X, y, or y|X - before choosing a remedy.",
            ),
            what_to_change=("Map monitoring signals to shift type and playbooks."),
            pitfalls=("Calling every change 'drift' without a type."),
            decide="For each alert, name the shift type and the response playbook.",
            read_steps=("Compare feature dists vs label rates vs calibration."),
        ),
    ]


def _strat_calc(ctx: dict) -> str:
    target = ctx.get("target")
    if not isinstance(target, dict) or not target.get("name"):
        return "No target declared - stratification target is undefined."
    if not is_classification(ctx):
        return (
            f"Regression target '{target_name(ctx)}' - stratification is usually off; "
            "consider binning y only with care."
        )
    classes = target.get("classes") or []
    if not classes:
        return f"Classification target '{target_name(ctx)}' without class counts in this report."
    total = sum(int(c.get("count") or 0) for c in classes) or 1
    rare = min(classes, key=lambda c: int(c.get("count") or 0))
    return (
        f"Class mix on analysed rows; rarest '{rare.get('label')}' at "
        f"{fmt_pct(int(rare.get('count') or 0) / total)} - stratify so holdouts can contain it."
    )


def _outlier_calc(ctx: dict) -> str:
    rates = [
        (n.get("name"), float(n.get("outlierRate") or 0))
        for n in (ctx.get("numeric") or [])
        if isinstance(n, dict) and float(n.get("outlierRate") or 0) > 0
    ]
    rates.sort(key=lambda item: item[1], reverse=True)
    anom = ctx.get("anomalies")
    parts = []
    if rates:
        parts.append(
            "IQR rates: "
            + ", ".join(f"{name} {fmt_pct(rate)}" for name, rate in rates[:4])
        )
    else:
        parts.append("No positive per-column IQR outlier rates in context.")
    if isinstance(anom, dict):
        parts.append(
            f"Multivariate flagged {fmt_n(anom.get('flagged'))} / {fmt_n(anom.get('scored'))} "
            f"(contamination~{fmt_pct(float(anom.get('contamination') or 0))} setting)."
        )
    return " ".join(parts)


def _power_calc(ctx: dict) -> str:
    target = ctx.get("target")
    if is_classification(ctx) and isinstance(target, dict) and target.get("classes"):
        counts = [int(c.get("count") or 0) for c in target["classes"]]
        return (
            f"n={fmt_n(ctx.get('rows'))}; class counts={counts}; "
            f"minority={fmt_n(min(counts) if counts else 0)} (effective sample for rare-event models)."
        )
    return f"n={fmt_n(ctx.get('rows'))}; eligible features={fmt_n(ctx.get('eligible'))}."
