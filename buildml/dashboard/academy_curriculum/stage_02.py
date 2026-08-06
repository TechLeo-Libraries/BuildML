"""Stage 02 · Relationships - columns vs target and each other."""

from __future__ import annotations

from buildml.dashboard.academy_curriculum._factory import L, rows_blurb, with_starter
from buildml.dashboard.academy_curriculum._helpers import (
    code_block,
    first_feature,
    first_numeric,
    fmt_compact,
    fmt_dec,
    fmt_n,
    fmt_pct,
    list_names,
    plural,
    target_name,
)
from buildml.dashboard.academy_curriculum._types import LessonSpec


def lessons() -> list[LessonSpec]:
    return [
        *_core(),
        *_additions(),
    ]


def _core() -> list[LessonSpec]:
    return [
        L(
            slug="univariate-distributions",
            stage=2,
            order=10,
            concept_key="normality-screens",
            tags=("distribution",),
            plain=(
                "Before relationships, each column has a shape: centre, spread, tails, gaps. "
                "Two columns with the same mean can behave totally differently in a model.",
            ),
            technical=(
                "Prefer quartiles, min/max, and sparsity over the mean alone. "
                "BuildML univariate profiles expose these screens in session.eda().",
            ),
            why=("Shape drives transforms, outlier policy, and model family choice."),
            formula="IQR = Q3 - Q1; compare median vs mean for skew cues",
            calculation=lambda ctx: _uni_calc(ctx),
            session_evidence=lambda ctx: _uni_calc(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "report = session.eda(include_plots=False, show=False)",
                "uni = report.to_dict().get(\"univariate\", {}).get(\"per_column\", {})",
                f"print(uni.get(\"{first_numeric(ctx)}\"))  # quartiles / skew for one column",
                'session.learn("normality-screens", level="beginner")',
            ),
            what_to_change=("Swap the column you inspect; compare several features before modeling."),
            pitfalls=(
                "Reading the mean of a skewed column as typical.",
                "Summarising after imputation and mistaking narrowed variance for truth.",
            ),
            decide="Read quartiles + tails for every numeric feature before choosing transforms.",
            read_steps=(
                "Compare mean vs median.",
                "Check min/max against domain limits.",
                "Note spikes of repeated values.",
            ),
        ),
        L(
            slug="skew-and-transforms",
            stage=2,
            order=20,
            concept_key="feature-scaling",
            tags=("skew", "log"),
            plain=(
                "Skew means the tail dominates. Under squared error a few extreme rows can steer the fit.",
            ),
            technical=(
                "log1p / power transforms improve symmetry for linear/distance models. "
                "Apply as a train-fitted custom transform or upstream; trees rarely need them.",
            ),
            why=("Tail leverage warps linear models and some scalers."),
            formula="skew ~ E[((x-mu)/sigma)^3]; |skew| > 1 is a common review flag",
            calculation=lambda ctx: (
                f"{fmt_n(len(ctx.get('skewed') or []))} numeric "
                f"{plural(len(ctx.get('skewed') or []), 'column')} with |skew|>1: "
                + (
                    ", ".join(
                        f"{s.get('name')} ({fmt_dec(float(s.get('skew') or 0), 2)})"
                        for s in (ctx.get("skewed") or [])[:4]
                    )
                    or "none"
                )
            ),
            session_evidence=lambda ctx: (
                f"Skew flags: {list_names(ctx.get('skewed') or []) or 'none above |1|'}."
            ),
            example_code=lambda ctx: code_block(
                "import numpy as np",
                "import pandas as pd",
                "from buildml import Session",
                "",
                "frame = pd.read_csv(\"your_data.csv\")  # <-- change",
                f"col = \"{_skew_col(ctx)}\"",
                "frame[f\"{col}_log1p\"] = np.log1p(frame[col].clip(lower=0))",
                "session = Session.ingest(frame)",
                "session = session.set_roles({",
                f'    "{target_name(ctx)}": "target",',
                "    f\"{col}_log1p\": \"feature\",",
                "    col: \"ignore\",  # keep raw for audit if you want",
                "})",
                "session = session.split(test_size=0.2, random_state=0)",
                "# Report errors in original units after modeling.",
            ),
            what_to_change=("Pick skewed columns; prefer monotone transforms; validate on train folds."),
            pitfalls=(
                "Log-transforming zeros/negatives without an offset.",
                "Reporting RMSE in log units as business units.",
                "Transforming for tree models that do not need it.",
            ),
            decide="Choose transform vs robust model family per skewed column; document units of errors.",
            read_steps=("List |skew|>1 columns.", "Check zeros/negatives before log1p."),
        ),
        L(
            slug="correlation",
            stage=2,
            order=30,
            concept_key="feature-selection",
            tags=("correlation", "pearson", "spearman"),
            search_terms=("correlation", "pearson", "spearman", "collinearity"),
            plain=(
                "Pearson measures linear co-movement; Spearman measures monotone association. "
                "A curved relationship can have near-zero Pearson.",
            ),
            technical=(
                "Feature-feature correlation is redundancy; feature-target correlation is a weak screen. "
                "Neither implies mechanism.",
            ),
            why=("Near-duplicate columns destabilise linear coefficients."),
            formula="r = cov(x,y) / (sigma_x sigma_y)",
            calculation=lambda ctx: _corr_calc(ctx),
            session_evidence=lambda ctx: _corr_calc(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "report = session.eda(include_plots=False, show=False)",
                "pairs = report.to_dict().get(\"bivariate\", {}).get(\"top_abs_pearson_pairs\", [])",
                "print(pairs[:5])",
                'session.learn("feature-selection", level="beginner")',
                "# Compare mentally with Spearman heatmaps on the Relationships board.",
            ),
            what_to_change=("Set review threshold (e.g. |r|≥0.8); choose which of a pair to keep."),
            pitfalls=(
                "Reading near-zero Pearson as 'no relationship'.",
                "Dropping pairs mechanically without measurement-quality judgement.",
            ),
            decide="For each |r|≥0.8 pair, keep the more reliable measurement or combine deliberately.",
            read_steps=("List strongest pairs.", "Ask if one column is a re-expression of the other."),
        ),
        L(
            slug="mutual-information",
            stage=2,
            order=40,
            concept_key="mutual-information",
            tags=("MI",),
            plain=(
                "Mutual information asks how much knowing a feature reduces uncertainty about the target. "
                "It does not assume a straight line - and it has no sign.",
            ),
            technical=(
                "MI is estimated (kNN / binning). Small rank gaps are often noise. "
                "Never select features by MI on the full frame.",
            ),
            why=("Catches non-linear dependence that correlation misses."),
            formula="I(X;Y) = H(Y) - H(Y|X) ≥ 0",
            calculation=lambda ctx: _mi_calc(ctx),
            session_evidence=lambda ctx: _mi_calc(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "report = session.eda(include_plots=False, show=False)",
                "mi = report.to_dict().get(\"bivariate\", {}).get(\"mutual_information_vs_target\", [])",
                "print(mi[:8])  # screening aid, not a final ranking",
                "",
                "# If you select features, do it train-only:",
                "session = session.select_features(",
                "    strategy=\"univariate\",",
                "    score_func=\"mutual_info\",",
                "    k=20,  # <-- tune",
                ")",
                'session.explain("select_features", moment="before")',
            ),
            what_to_change=("Tune k / method; always select inside training folds."),
            pitfalls=(
                "Reading MI rank as importance or causality.",
                "Selecting on the full frame (leaks test structure).",
            ),
            decide="Use MI as a screen; confirm with in-fold selection / model-based importance.",
            read_steps=lambda ctx: [
                f"Note leader vs runner-up gap ({_mi_gap(ctx)}).",
                "Remember univariate MI misses interactions.",
            ],
        ),
        L(
            slug="variance-inflation",
            stage=2,
            order=50,
            concept_key="variance-inflation",
            tags=("VIF",),
            plain=(
                "VIF asks how well the other numeric features predict this one. High VIF means "
                "coefficients become unstable even if predictions stay fine.",
            ),
            technical=(
                "VIF_j = 1 / (1 - R²_j). Thresholds like 5 or 10 are review flags, not laws. "
                "Remove one collinear member at a time and recompute.",
            ),
            why=("Interpretation collapses under collinearity; trees care less."),
            formula="VIF_j = 1 / (1 - R²_j) where R²_j is from regressing x_j on other features",
            calculation=lambda ctx: _vif_calc(ctx),
            session_evidence=lambda ctx: _vif_calc(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "report = session.eda(include_plots=False, show=False)",
                "vif = report.to_dict().get(\"multivariate\", {}).get(\"vif\", [])",
                "print(vif)",
                "",
                f"# Example: ignore one collinear column, then re-profile",
                f"drop = \"{_vif_drop(ctx)}\"  # <-- choose from VIF list",
                "session = session.set_roles({drop: \"ignore\"})",
                'session.learn("variance-inflation", level="intermediate")',
            ),
            what_to_change=("Pick threshold; drop/ignore one column at a time; recompute VIF."),
            pitfalls=(
                "Dropping every high-VIF column at once.",
                "Treating VIF as a prediction problem for tree models.",
            ),
            decide="If you need interpretable coefficients, resolve VIF>threshold pairs deliberately.",
            read_steps=(
                "Sort features by VIF.",
                "Remove one member of a collinear set, recompute.",
            ),
        ),
        L(
            slug="interaction-effects",
            stage=2,
            order=60,
            concept_key="feature-selection",
            tags=("interactions",),
            plain=(
                "Some signals only appear when two features act together. Univariate screens are blind to that.",
            ),
            technical=(
                "Trees find interactions; linear models need explicit products or segmented features. "
                "Create candidates carefully - width grows fast.",
            ),
            why=("Dropping low-MI features can delete the partner of an interaction."),
            formula="interaction example: x_int = x_a * x_b (standardise first for linear models)",
            calculation=lambda ctx: (
                f"Univariate MI scored {fmt_n(len(ctx.get('mi') or []))} features; "
                "no automatic pairwise interaction screen is claimed here."
                if ctx.get("mi")
                else "No MI screen available; interactions remain untested."
            ),
            session_evidence=lambda ctx: (
                f"Top MI features: {list_names(ctx.get('mi') or [])}. Combinations are still unchecked."
            ),
            example_code=lambda ctx: code_block(
                "import pandas as pd",
                "from buildml import Session",
                "",
                "frame = pd.read_csv(\"your_data.csv\")",
                f"a, b = \"{_mi_name(ctx, 0)}\", \"{_mi_name(ctx, 1)}\"  # <-- candidate pair",
                "frame[\"interaction_ab\"] = frame[a] * frame[b]",
                "session = Session.ingest(frame).set_roles({",
                f'    "{target_name(ctx)}": "target",',
                "    a: \"feature\", b: \"feature\", \"interaction_ab\": \"feature\",",
                "}).split(test_size=0.2, random_state=0)",
                "session = session.scale(method=\"standard\")  # often wise for linear models",
            ),
            what_to_change=("Choose candidate pairs from domain knowledge; validate in CV."),
            pitfalls=("Adding all pairwise products.", "Reading tree importances as main-effect evidence."),
            decide="List a short interaction candidate set; test in-fold, do not explode the matrix.",
            read_steps=("Keep low-MI features if domain says they interact.", "Prefer model classes that capture interactions when many are expected."),
        ),
        L(
            slug="dimensionality-reduction",
            stage=2,
            order=70,
            concept_key="principal-components",
            tags=("PCA",),
            plain=(
                "Reduction compresses many numeric columns into fewer components. You gain conditioning "
                "and lose direct interpretability of original measurements.",
            ),
            technical=(
                "session.reduce_dimensions(method='pca', n_components=...) is a train-fitted transform. "
                "Choosing n_components is a hyper-parameter - validate it.",
            ),
            why=("Wide, collinear matrices hurt linear models; PCA is one remedy among selection/regularisation."),
            formula="PCA: X ~ T P^T with orthogonal components; n_components may be variance fraction",
            calculation=lambda ctx: (
                f"{fmt_n(ctx.get('eligible'))} eligible features for {fmt_n(ctx.get('rows'))} rows "
                f"(~{fmt_n(int((ctx.get('rows') or 1) / max(int(ctx.get('eligible') or 1), 1)))} rows/feature). "
                f"High-VIF candidates: {fmt_n(len([v for v in (ctx.get('vif') or []) if float(v.get('vif') or 0) >= float(ctx.get('vifThreshold') or 5)]))}."
            ),
            session_evidence=lambda ctx: rows_blurb(ctx) + f"; eligible={fmt_n(ctx.get('eligible'))}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "session = session.impute(strategy=\"median\")",
                "session = session.scale(method=\"standard\")  # PCA expects comparable scales",
                "session = session.reduce_dimensions(method=\"pca\", n_components=0.95)  # <-- tune",
                'session.explain("reduce_dimensions", moment="before")',
            ),
            what_to_change=("Tune n_components / method; decide whether to drop input columns."),
            pitfalls=(
                "Fitting PCA on the full frame before split.",
                "Reporting feature importances on principal components as if they were original columns.",
            ),
            decide="Prefer selective dropping/regularisation when you need named features; use PCA when compression is the goal.",
            read_steps=("Check rows-per-feature.", "Scale before PCA.", "Validate component count."),
        ),
        L(
            slug="feature-scaling",
            stage=2,
            order=80,
            concept_key="feature-scaling",
            tags=("scale",),
            plain=(
                "Scaling puts numeric columns on comparable footing. Distance models and regularised linear "
                "models need it; trees largely do not.",
            ),
            technical=(
                "session.scale(method='standard'|'minmax') learns on train after split. "
                "Full-frame standardisation is leakage.",
            ),
            why=("Unscaled ranges let large-unit features dominate penalties and distances."),
            formula="z = (x - mu_train) / sigma_train",
            calculation=lambda ctx: _scale_calc(ctx),
            session_evidence=lambda ctx: _scale_calc(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "session = session.impute(strategy=\"median\")",
                "session = session.scale(method=\"standard\")  # <-- or minmax",
                "# Fitted on train automatically after split",
                'session.learn("feature-scaling", level="beginner")',
            ),
            what_to_change=("Choose standard vs minmax; exclude one-hots if you do not want them scaled."),
            pitfalls=("Scaling before split.", "Standardising heavily skewed columns expecting symmetry."),
            decide="Scale inside the training fold for any distance/regularised model path.",
            read_steps=("Compare numeric ranges.", "Decide model family first (trees vs linear/SVM/NN)."),
        ),
    ]


def _additions() -> list[LessonSpec]:
    return [
        L(
            slug="categorical-association",
            stage=2,
            order=90,
            concept_key="mutual-information",
            tags=("categorical", "association"),
            plain=("Categorical-target links need association measures (MI, chi-square), not Pearson on codes."),
            technical=("Integer codes are not quantities; association ≠ ordered correlation."),
            why=("Treating label codes as numeric invents fake linear structure."),
            formula=None,
            calculation=lambda ctx: (
                f"Categoricals: {list_names(ctx.get('categorical') or [])}. "
                f"MI rows available: {fmt_n(len(ctx.get('mi') or []))}."
            ),
            session_evidence=lambda ctx: f"Task={ctx.get('task')}; categoricals={list_names(ctx.get('categorical') or [])}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "report = session.eda(include_plots=False, show=False)",
                "print(report.to_dict().get(\"bivariate\", {}).get(\"mutual_information_vs_target\", [])[:10])",
                f"session = session.encode(method=\"onehot\", columns=[\"{first_feature(ctx)}\"])  # if categorical",
            ),
            what_to_change=("Encode categoricals properly before linear models."),
            pitfalls=("Pearson on arbitrary category codes."),
            decide="Score categorical association with MI / model tools, not fake numeric correlation.",
            read_steps=("List categoricals.", "Inspect MI vs target."),
        ),
        L(
            slug="non-linearity-and-binning",
            stage=2,
            order=100,
            concept_key="feature-binning",
            tags=("binning",),
            plain=("When effect shapes bend, binning or non-linear models beat forcing a straight line."),
            technical=("session.bin(...) is train-fitted. Edges chosen on test are leakage."),
            why=("Linear coefficients hide thresholds and U-shapes."),
            formula=None,
            calculation=lambda ctx: (
                f"Consider binning for skewed numerics: {list_names(ctx.get('skewed') or []) or first_numeric(ctx)}."
            ),
            session_evidence=lambda ctx: f"Skewed numerics: {list_names(ctx.get('skewed') or [])}.",
            example_code=lambda ctx: with_starter(
                ctx,
                f"session = session.bin(columns=[\"{first_numeric(ctx)}\"], n_bins=5)  # <-- tune",
                'session.learn("feature-binning", level="beginner")',
            ),
            what_to_change=("Tune n_bins / strategy; prefer model non-linearity when possible."),
            pitfalls=("Binning using target-aware edges on the full frame."),
            decide="Choose binning vs a non-linear model deliberately; fit edges on train only.",
            read_steps=("Plot feature vs target residual structure.", "Avoid ad-hoc test-driven cuts."),
        ),
        L(
            slug="confounding-and-subgroups",
            stage=2,
            order=110,
            concept_key="causal-assumptions",
            tags=("confounding",),
            plain=("A strong association can be a confounder story, not a lever you can push."),
            technical=("EDA association ≠ causal effect. Subgroup checks reveal Simpson-like reversals."),
            why=("Shipping a correlated proxy as if it were a cause harms decisions."),
            formula=None,
            calculation=lambda ctx: (
                f"Top association screens (MI): {list_names(ctx.get('mi') or [])}. "
                "Ask which could be confounders for your decision."
            ),
            session_evidence=lambda ctx: "Causal claims need assumptions beyond this readiness sheet.",
            example_code=lambda ctx: with_starter(
                ctx,
                'session.learn("causal-assumptions", level="beginner")',
                'session.learn("causal-eda-boundary", level="intermediate")',
                "# Use session.fit_causal(...) only when identification assumptions are explicit.",
            ),
            what_to_change=("Name confounders; separate prediction from intervention questions."),
            pitfalls=("Reading feature importance as causal effect."),
            decide="Label each key association predictive vs potentially causal; do not conflate them.",
            read_steps=("List confounders from domain map.", "Check associations within subgroups."),
        ),
        L(
            slug="derived-and-redundant-columns",
            stage=2,
            order=120,
            concept_key="feature-selection",
            tags=("redundant",),
            plain=("Derived columns that restate others add width without new information - and inflate VIF."),
            technical=("Use correlation pairs + VIF + domain lineage to spot re-expressions."),
            why=("Redundant features waste regularisation budget and confuse importance."),
            formula=None,
            calculation=lambda ctx: _corr_calc(ctx),
            session_evidence=lambda ctx: _corr_calc(ctx),
            example_code=lambda ctx: with_starter(
                ctx,
                "report = session.eda(include_plots=False, show=False)",
                "print(report.to_dict().get(\"bivariate\", {}).get(\"top_abs_pearson_pairs\", [])[:10])",
                "session = session.select_features(strategy=\"variance\", threshold=0.0)  # drop constants",
            ),
            what_to_change=("Ignore or drop redundant re-expressions; keep the best-measured version."),
            pitfalls=("Keeping both raw and fully determined derived totals."),
            decide="For each redundant pair, keep one column and document why.",
            read_steps=("Walk strongest correlation pairs.", "Check lineage for derived fields."),
        ),
        L(
            slug="time-feature-engineering",
            stage=2,
            order=130,
            concept_key="feature-schema",
            tags=("time", "calendar"),
            plain=("Calendar parts and lags are features only if they would be known at prediction time."),
            technical=("session.extract_dates builds calendar fields; lags need careful as-of joins."),
            why=("Future-looking time features are classic leakage."),
            formula=None,
            calculation=lambda ctx: (
                f"Time column: {(ctx.get('timeCol') or {}).get('name') if ctx.get('timeCol') else 'not detected'}."
            ),
            session_evidence=lambda ctx: (
                f"Temporal axis present: {bool(ctx.get('timeCol'))}."
            ),
            example_code=lambda ctx: code_block(
                "import pandas as pd",
                "from buildml import Session",
                "",
                "frame = pd.read_csv(\"your_data.csv\")",
                f"t = \"{(ctx.get('timeCol') or {}).get('name') or '<timestamp>'}\"",
                "frame[t] = pd.to_datetime(frame[t], utc=True, errors=\"coerce\")",
                "session = Session.ingest(frame).set_roles({",
                f'    t: "time", "{target_name(ctx)}": "target", "{first_feature(ctx)}": "feature",',
                "})",
                "session = session.extract_dates(columns=[t])",
            ),
            what_to_change=("Only extract parts known at score time; build lags with as-of discipline."),
            pitfalls=("Using label-time calendar fields as predictors."),
            decide="Freeze which calendar/lag features exist at prediction time.",
            read_steps=("Confirm timestamp role.", "Reject post-outcome time fields."),
        ),
        L(
            slug="sparsity-and-dimensionality",
            stage=2,
            order=140,
            concept_key="overfitting",
            tags=("sparsity", "p>>n"),
            plain=("When features approach or exceed rows, models memorise noise unless constrained."),
            technical=("Watch rows-per-feature, one-hot width, and regularisation / selection."),
            why=("Over-wide matrices overfit and make holdout scores fragile."),
            formula="rules of thumb fail - validate; track n_rows / n_features after encoding",
            calculation=lambda ctx: (
                f"rows/feature ~ {fmt_compact((ctx.get('rows') or 1) / max(ctx.get('eligible') or 1, 1))} "
                f"with eligible={fmt_n(ctx.get('eligible'))}."
            ),
            session_evidence=lambda ctx: rows_blurb(ctx) + f"; eligible={fmt_n(ctx.get('eligible'))}.",
            example_code=lambda ctx: with_starter(
                ctx,
                "session = session.encode(method=\"infrequent\", min_frequency=0.05)",
                "session = session.select_features(",
                "    strategy=\"univariate\",",
                "    score_func=\"mutual_info\",",
                f"    k=min(50, {max(int(ctx.get('eligible') or 10), 1)}),  # <-- tune",
                ")",
                'session.learn("overfitting", level="beginner")',
            ),
            what_to_change=("Cap one-hot width; select/regularise; prefer simpler models when n is small."),
            pitfalls=("One-hotting high-card columns on small n."),
            decide="Set a maximum feature budget after encoding and enforce it with selection/regularisation.",
            read_steps=("Compute rows-per-feature post-encode.", "Prefer nested CV when tuning aggressively."),
        ),
    ]


def _skew_col(ctx: dict) -> str:
    skewed = ctx.get("skewed") or []
    if skewed and isinstance(skewed[0], dict) and skewed[0].get("name"):
        return str(skewed[0]["name"])
    return first_numeric(ctx)


def _uni_calc(ctx: dict) -> str:
    nums = [n for n in (ctx.get("numeric") or []) if isinstance(n, dict)]
    if not nums:
        return "This frame has no numeric columns to summarise."
    with_range = [n for n in nums if n.get("min") is not None and n.get("max") is not None]
    if not with_range:
        return (
            f"{fmt_n(len(nums))} numeric {plural(len(nums), 'column')} present, "
            "but ranges/quartiles were not fully supplied - shapes are unexamined, not proven symmetric."
        )
    widest = max(with_range, key=lambda n: abs(float(n.get("max") or 0) - float(n.get("min") or 0)))
    return (
        f"{fmt_n(len(nums))} numeric columns summarised; widest span {widest.get('name')} "
        f"from {fmt_compact(float(widest.get('min') or 0))} to {fmt_compact(float(widest.get('max') or 0))}."
    )


def _corr_calc(ctx: dict) -> str:
    pairs = ctx.get("corrPairs") or []
    if not pairs:
        return "No pairwise correlations were recorded for this frame."
    top = pairs[0]
    strong = [p for p in pairs if abs(float(p.get("r") or 0)) >= 0.8]
    return (
        f"{fmt_n(len(pairs))} pairs scored; strongest {top.get('a')} x {top.get('b')} "
        f"at r={fmt_dec(float(top.get('r') or 0), 3)}"
        + (f"; {fmt_n(len(strong))} with |r|≥0.8" if strong else "")
        + "."
    )


def _mi_calc(ctx: dict) -> str:
    mi = ctx.get("mi") or []
    if not mi:
        return "No mutual-information estimates - need a target and eligible features."
    top, last = mi[0], mi[-1]
    second = mi[1] if len(mi) > 1 else None
    gap = ""
    if second:
        rel = abs(float(top["mi"]) - float(second["mi"])) / max(float(top["mi"]), 1e-12)
        if rel < 0.15:
            gap = " Leader gap is small - order is not settled."
    return (
        f"{top['name']} leads at {fmt_dec(float(top['mi']), 6)}"
        + (f"; {second['name']} at {fmt_dec(float(second['mi']), 6)}" if second else "")
        + f"; weakest scored {last['name']} at {fmt_dec(float(last['mi']), 6)}."
        + gap
    )


def _mi_gap(ctx: dict) -> str:
    mi = ctx.get("mi") or []
    if len(mi) < 2:
        return "n/a"
    rel = abs(float(mi[0]["mi"]) - float(mi[1]["mi"])) / max(float(mi[0]["mi"]), 1e-12)
    return fmt_pct(rel)


def _mi_name(ctx: dict, idx: int) -> str:
    mi = ctx.get("mi") or []
    if len(mi) > idx:
        return str(mi[idx]["name"])
    return first_feature(ctx) if idx == 0 else first_numeric(ctx)


def _vif_calc(ctx: dict) -> str:
    vif = ctx.get("vif") or []
    if not vif:
        return "No VIF estimates available for an eligible numeric feature set."
    thr = float(ctx.get("vifThreshold") or 5)
    over = [v for v in vif if float(v.get("vif") or 0) >= thr]
    if over:
        return (
            ", ".join(f"{v.get('name')} at {fmt_dec(float(v.get('vif') or 0), 3)}" for v in over[:4])
            + (f" and {len(over) - 4} more" if len(over) > 4 else "")
            + f" sit above {fmt_dec(thr, 1)} (complete cases ~ {fmt_n(ctx.get('completeRows'))})."
        )
    return (
        f"All below {fmt_dec(thr, 1)}; highest {vif[0].get('name')} at "
        f"{fmt_dec(float(vif[0].get('vif') or 0), 3)}."
    )


def _vif_drop(ctx: dict) -> str:
    thr = float(ctx.get("vifThreshold") or 5)
    over = [v for v in (ctx.get("vif") or []) if float(v.get("vif") or 0) >= thr]
    if len(over) > 1:
        return str(over[1].get("name"))
    if ctx.get("vif"):
        return str(ctx["vif"][min(1, len(ctx["vif"]) - 1)].get("name"))
    return first_numeric(ctx)


def _scale_calc(ctx: dict) -> str:
    nums = [n for n in (ctx.get("numeric") or []) if isinstance(n, dict)]
    if not nums:
        return "No numeric columns, so no scaling decision arises."
    spans = []
    for n in nums:
        try:
            span = abs(float(n.get("max")) - float(n.get("min")))
        except (TypeError, ValueError):
            continue
        if span > 0:
            spans.append(span)
    if not spans:
        return (
            f"{fmt_n(len(nums))} numeric columns present but ranges unknown - "
            "scaling still matters for distance/regularised models, not trees."
        )
    ratio = max(spans) / min(spans)
    return (
        f"Numeric ranges differ by ~{fmt_compact(ratio)}x across {fmt_n(len(nums))} columns. "
        "Material for distance/regularised models; irrelevant for pure trees."
    )
