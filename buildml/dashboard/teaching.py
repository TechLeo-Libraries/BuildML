# ruff: noqa: E501
"""Teaching Studio builders: definition, method, pitfalls, worked examples."""

from __future__ import annotations

from typing import Any

from buildml.dashboard.serialize import flagged_column_names


def build_teaching_studios(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Write the teaching panel for each board, using this dataset's numbers.

    The studio's second half. Each board shows an analysis; each teaching panel
    explains what that analysis is, how to read it, what commonly goes wrong,
    and: crucially: walks through the reader's own result rather than a
    textbook one.

    That last part is what distinguishes this from a glossary. "Variance
    inflation factor measures collinearity" is a definition anyone can look up.
    "Your ``total_spend`` has a VIF of 23, which means it is almost perfectly
    predicted by the other columns, and its coefficient in a linear model would
    be arbitrary" is the sentence that teaches.

    Parameters
    ----------
    report:
        The report as a dict, from
        :meth:`~buildml.eda.report.EDAReport.to_dict`. Missing sections yield
        panels that explain the concept without a worked example, rather than
        panels that are absent.

    Returns
    -------
    dict
        Keyed by domain: ``cockpit``, ``quality``, ``features``,
        ``relationships``, ``multivariate``, ``target``, ``outliers``,
        ``visuals``. Each value holds the definition, method, pitfalls, and
        worked example for that board.

    Notes
    -----
    **The keys line up with the domain registry.** Adding a board means adding a
    builder here, or its teaching panel will be missing.

    **This reads the report and nothing else.** No data access, so it works
    against a saved report just as well as a live one: which is what lets the
    offline export carry its teaching content with it.

    See Also
    --------
    buildml.dashboard.domains : The boards these correspond to.
    buildml.explain : The concept definitions referenced.
    """
    return {
        "cockpit": _studio_cockpit(report),
        "quality": _studio_quality(report),
        "features": _studio_features(report),
        "relationships": _studio_relationships(report),
        "multivariate": _studio_multivariate(report),
        "target": _studio_target(report),
        "outliers": _studio_outliers(report),
        "visuals": _studio_visuals(report),
    }


def _studio_cockpit(report: dict[str, Any]) -> dict[str, Any]:
    overview = report.get("overview") or {}
    findings = report.get("findings") or []
    high = sum(
        1 for item in findings if str(item.get("severity", "")).lower() in {"high", "critical"}
    )
    has_lazy = bool(overview.get("has_lazy_native"))
    has_native = bool(overview.get("has_native"))
    engine_notes = list(overview.get("engine_disclosures") or [])
    warm = overview.get("warm_start_status") or {}
    warm_enabled = bool(warm.get("enabled"))
    warm_notes = list(warm.get("disclosures") or [])
    scope = overview.get("preprocess_scope_status") or {}
    scope_enabled = bool(scope.get("enabled"))
    scope_notes = list(scope.get("disclosures") or [])
    fold_scope = scope.get("fold_local") or {}
    session_scope = scope.get("session_global") or {}
    torch_status = overview.get("torch_training_status") or {}
    torch_enabled = bool(torch_status.get("enabled"))
    torch_notes = list(torch_status.get("disclosures") or [])
    torch_early = torch_status.get("early_stop") or {}
    lazy_line = (
        "has_lazy_native=True: collect-on-promote applies; sklearn still needs RAM for the design matrix."
        if has_lazy
        else (
            "has_native=True: native project/filter/sample can run before Pandas materialize."
            if has_native
            else "has_native=False: Session tabular path is Pandas-backed for this report."
        )
    )
    warm_line = (
        "warm_start_studies=True appears in Session history/results: Optuna study trial "
        "history was shared across nested-CV outer folds; Session test/validation were "
        "not scored by that search."
        if warm_enabled
        else "No nested_cv_score warm_start_studies=True record is attached to this report."
    )
    scope_bits: list[str] = []
    if fold_scope.get("text") is not None:
        scope_bits.append(f"fold-local text={fold_scope.get('text')!r}")
    if fold_scope.get("reduce") is not None:
        scope_bits.append(f"fold-local reduce={fold_scope.get('reduce')!r}")
    if session_scope.get("apply_custom_transform"):
        scope_bits.append("Session-global custom transform")
    if session_scope.get("resample"):
        scope_bits.append("Session-global resample")
    if session_scope.get("text_features") and fold_scope.get("text") is None:
        scope_bits.append("Session-global text_features")
    if session_scope.get("reduce_dimensions") and fold_scope.get("reduce") is None:
        scope_bits.append("Session-global reduce_dimensions")
    scope_line = (
        "Preprocess scope recorded: " + "; ".join(scope_bits) + "."
        if scope_bits
        else "No fold-local text/PCA or Session-global custom/resample/text/PCA plans are attached."
    )
    torch_line = (
        f"Torch trainer present: device="
        f"{(torch_status.get('device') or {}).get('resolved')}; "
        f"n_epochs_ran={torch_status.get('n_epochs_ran')}; "
        f"early_stop_monitor={torch_early.get('monitor')}; "
        f"early_stop_partition={torch_early.get('partition')}; "
        f"triggered={torch_early.get('triggered')}."
        if torch_enabled
        else "No live Torch dl_train_result is attached to this report."
    )
    interpretation = [
        "Read sampling and row-scope disclosure before trusting tail behavior or rare categories.",
        lazy_line,
        "Treat high or critical findings as blockers for model-selection claims until reviewed.",
        "Recommendations cite finding keys; open those keys and verify the underlying table before acting.",
        "A quiet cockpit (few findings) means fewer automated flags, not that the data-generating process is clean.",
        "Full-dataset descriptive EDA after a split summarizes observed rows; it is not train-fitted transform evidence.",
    ]
    pitfalls = [
        "Full-dataset EDA after a split is descriptive; it is not train-fitted evidence for transforms.",
        "Association screens do not establish causality.",
        "Availability of an API call does not mean the call is appropriate for this dataset.",
        "Clearing findings by dropping columns without role review can hide leakage or discard valid signal.",
        "Treating the cockpit as a model scorecard confuses data triage with estimator performance.",
        "Reading has_lazy_native as out-of-core model training; LazyFrame collect still feeds in-memory estimators.",
    ]
    practice = [
        "Confirm analysis_rows versus n_rows and note any sampling disclosure.",
        "Confirm engine / has_native / has_lazy_native before assuming materialization cost.",
        "Open every high/critical finding and read its evidence keys.",
        "Verify target, id, and ignore roles before imputation or encoding.",
        "Check whether a train/test split exists before interpreting drift findings.",
        "Write down one next preprocessing or role decision tied to a cited finding key.",
    ]
    if warm_enabled:
        interpretation.insert(2, warm_line)
        pitfalls.append(
            "Reading warm_start_studies as permission to use Session test during search; "
            "only Optuna priors were shared across outer folds."
        )
        practice.append(
            "If warm_start_studies=True, record that outer-fold inner searches shared "
            "Optuna trial history and re-read outer mean±std (not inner best scores)."
        )
    if scope_enabled:
        interpretation.insert(3 if warm_enabled else 2, scope_line)
        if fold_scope.get("text") is not None or fold_scope.get("reduce") is not None:
            pitfalls.append(
                "Confusing Session.text_features / Session.reduce_dimensions with "
                "PreprocessRecipe fold-local text/PCA; fold-local fits refit per fold-train."
            )
        if session_scope.get("apply_custom_transform") or session_scope.get("resample"):
            pitfalls.append(
                "Expecting custom transforms or resample inside PreprocessRecipe; both stay "
                "Session-global and are not refit per CV fold."
            )
        practice.append(
            "When fold-local text/PCA or Session-global custom/resample appear, record which "
            "scope applied before interpreting CV scores."
        )
    if torch_enabled:
        insert_at = 2 + int(warm_enabled) + int(scope_enabled)
        interpretation.insert(insert_at, torch_line)
        pitfalls.append(
            "Reading validation early-stopping curves as test performance; "
            "evaluate_torch(partition='test') is a separate final estimate."
        )
        practice.append(
            "When a Torch trainer is attached, note device, early-stop partition, and "
            "whether stopping triggered before claiming holdout metrics."
        )
    return {
        "domain": "cockpit",
        "title": "Command cockpit",
        "definition": (
            "The cockpit is the session-level triage board for Exploratory Data Analysis "
            "(EDA). It aggregates readiness signals from quality, association, drift, "
            "outlier, and scope screens into severity-ranked findings and linked "
            "recommendations.\n\n"
            "What you see here is not a second analysis layer with new statistics. It is "
            "an ordered map of evidence already computed elsewhere in the report, so you "
            "can decide what to inspect next without treating every association or chart "
            "as equally urgent."
        ),
        "why": (
            "EDA produces many tables at once. Without a severity-ranked overview, it is "
            "easy to chase visual novelty:strong correlations, skewed plots, busy heatmaps:"
            "while missing blockers such as identifier leakage, extreme missingness, or "
            "train/test drift.\n\n"
            "The cockpit keeps partition, sampling, and engine/lazy-native limits visible. "
            "Those limits change how much trust you should place in tail behavior, rare "
            "categories, and any claim that a holdout score represents deployment risk."
        ),
        "how": (
            "BuildML runs analyzers on the analysis frame, then aggregates their outputs "
            "into evidence-linked findings and recommendations. Each finding cites keys "
            "into report sections (for example quality.completeness or eda.scope).\n\n"
            "Severity reflects likely workflow impact on later modeling steps:not chart "
            "emphasis or marketing priority. Recommendations are suggestions tied to those "
            "keys; they do not auto-apply transforms."
        ),
        "interpretation": interpretation,
        "thresholds": [
            "Severity high/critical → review before comparing estimators or freezing preprocessing.",
            "analysis_rows < n_rows → sampling or row budget is active; rare events may be under-represented.",
            "Eligible feature count of zero → roles, constants, or id-like flags removed all candidates; fix schema first.",
            "Finding volume alone is not a quality score; one critical leakage flag outweighs many low notes.",
            *(
                [
                    "warm_start_studies=True → shared Optuna priors across outer folds; "
                    "read outer mean±std, not inner best trial scores, as the selection estimate."
                ]
                if warm_enabled
                else []
            ),
            *(
                [
                    "PreprocessRecipe text/reduce inside CV → fold-train fit only; "
                    "custom transforms and resample remain Session-global."
                ]
                if scope_enabled
                else []
            ),
            *(
                [
                    "torch_training_status.enabled → early-stop monitor is validation-scoped; "
                    "test metrics remain a final estimate after stopping decisions freeze."
                ]
                if torch_enabled
                else []
            ),
        ],
        "assumptions": [
            "Analyzer outputs and finding keys in the current report are complete and consistent with this Session.",
            "Column roles (feature / target / id / ignore) reflect intended modeling use.",
            "The analysis frame (including any sample) is treated as the descriptive universe for cockpit triage.",
            "Severity labels encode workflow priority heuristics, not calibrated probabilities of harm.",
            "Engine metadata (has_native / has_lazy_native) describes the Session handle, not an out-of-core sklearn path.",
            *(
                [
                    "When warm_start_status.enabled is True, Session history recorded "
                    "nested_cv_score with warm_start_studies=True."
                ]
                if warm_enabled
                else []
            ),
            *(
                [
                    "When preprocess_scope_status.enabled is True, history or live Session "
                    "plans recorded fold-local text/PCA and/or Session-global custom/resample/text/PCA."
                ]
                if scope_enabled
                else []
            ),
            *(
                [
                    "When torch_training_status.enabled is True, Session.dl_train_result "
                    "carries epoch history, device, and early-stop bookkeeping."
                ]
                if torch_enabled
                else []
            ),
        ],
        "pitfalls": pitfalls,
        "worked_example": {
            "summary": (
                f"This session examined {overview.get('analysis_rows', 0):,} of "
                f"{overview.get('n_rows', 0):,} rows across "
                f"{overview.get('n_columns', 0):,} columns "
                f"(engine={overview.get('engine')}, has_lazy_native={has_lazy}"
                f"{', warm_start_studies=True' if warm_enabled else ''}"
                f"{', preprocess_scope' if scope_enabled else ''}"
                f"{', torch_trainer' if torch_enabled else ''})."
            ),
            "values": {
                "analysis_rows": overview.get("analysis_rows"),
                "dataset_rows": overview.get("n_rows"),
                "high_or_critical_findings": high,
                "eligible_features": len(overview.get("eligible_feature_columns") or []),
                "engine": overview.get("engine"),
                "mode": overview.get("mode"),
                "has_native": has_native,
                "has_lazy_native": has_lazy,
                "engine_disclosures": engine_notes,
                "warm_start_studies": warm_enabled,
                "warm_start_status": warm if warm_enabled else {"enabled": False},
                "warm_start_disclosures": warm_notes,
                "preprocess_scope": scope if scope_enabled else {"enabled": False},
                "fold_local_text": fold_scope.get("text"),
                "fold_local_reduce": fold_scope.get("reduce"),
                "preprocess_scope_disclosures": scope_notes,
                "torch_training": torch_status if torch_enabled else {"enabled": False},
                "torch_training_disclosures": torch_notes,
            },
            "reading": (
                f"{high} finding(s) are marked high or critical. Start with those before "
                "comparing estimators. "
                + (
                    torch_notes[0]
                    if torch_notes
                    else (
                        scope_notes[0]
                        if scope_notes
                        else (
                            warm_notes[0]
                            if warm_notes
                            else (engine_notes[0] if engine_notes else lazy_line)
                        )
                    )
                )
            ),
        },
        "modeling_impact": (
            "Unresolved identifier leakage, extreme missingness, or train/test drift can "
            "invalidate later holdout scores even when a model fits cleanly on the analysis "
            "frame. Resolve cockpit blockers before spending time on estimator bake-offs."
            + (
                " When warm_start_studies was enabled, treat outer-fold scores as the "
                "post-selection estimate and note that inner Optuna priors were shared."
                if warm_enabled
                else ""
            )
            + (
                " When fold-local text/PCA appears in CV history, treat vocabulary/IDF and "
                "PCA rotation as fold-train fits; custom transforms and resample stay "
                "Session-global."
                if scope_enabled
                else ""
            )
        ),
        "practice_checklist": practice,
        "mastery_notes": [
            "Severity is a triage heuristic: calibrate it against domain cost (false deletes vs leakage).",
            "Evidence keys are the contract between cockpit text and analyzer tables; always verify both.",
            "Descriptive EDA and train-fitted pipelines answer different questions; keep that boundary explicit in reports.",
            "When sampling is active, re-check rare-category and outlier claims on a larger draw or stratified sample.",
            "Lazy-native handles defer collects for prep ops; they do not remove the sklearn memory boundary.",
            *(
                [
                    "warm_start_studies shares Optuna trial history across outer folds; "
                    "it does not authorize scoring Session test during search."
                ]
                if warm_enabled
                else []
            ),
            *(
                [
                    "PreprocessRecipe text/reduce refit on fold-train; Session.apply_custom_transform "
                    "and Session.resample do not enter the fold-local recipe."
                ]
                if scope_enabled
                else []
            ),
        ],
        "next_action": {
            "label": "Inspect quality and roles before imputation",
            "api": 'session.explain("impute", moment="before")',
            "parameters": {"operation": "impute", "moment": "before"},
            "evidence_keys": ["eda.scope", "quality.completeness"],
        },
        "concepts": [
            "column-roles",
            "leakage-boundary",
            "data-splitting",
            "engine-choice",
            *(["cross-validation", "model-selection"] if warm_enabled else []),
            *(
                ["text-features", "principal-components", "custom-transforms", "cross-validation"]
                if scope_enabled
                else []
            ),
        ],
    }


def _studio_quality(report: dict[str, Any]) -> dict[str, Any]:
    quality = report.get("quality") or {}
    missing = int(quality.get("missing_cell_count") or 0)
    completeness = quality.get("completeness_score")
    constants = list(quality.get("constant_columns") or [])
    ids = list(quality.get("id_like_columns") or [])
    return {
        "domain": "quality",
        "title": "Data quality",
        "definition": (
            "Quality screens measure whether the observed table is structurally usable "
            "for modeling: completeness (missing cells), constant and near-constant "
            "columns, identifier-like uniqueness, duplicate rows, and simple text-pattern "
            "rates on sampled strings.\n\n"
            "These measures describe what is present in the current frame. They do not "
            "by themselves identify why values are missing, whether duplicates are "
            "dependent, or whether a near-unique column is a true surrogate key."
        ),
        "why": (
            "Models and transforms assume a usable feature schema. Constant columns add "
            "no observed variation. Identifier-like columns can create leakage-like "
            "memorization when treated as predictors. Missingness changes which rows "
            "contribute to later fits and can bias naive complete-case analyses.\n\n"
            "Catching schema problems here is cheaper than discovering them after a "
            "failed encode step or an inflated holdout score driven by an id column."
        ),
        "how": (
            "Completeness uses full-frame missing cell counts relative to the cell "
            "budget of the analyzed table. Identifier-like detection flags columns whose "
            "observed uniqueness is near the non-null row count (near-unique columns).\n\n"
            "Constant and near-constant columns are detected from observed value counts. "
            "Duplicate-row counts summarize exact row repeats. Pattern rates (for example "
            "email-like or numeric-string patterns) are estimated on text samples, not "
            "necessarily every string cell."
        ),
        "interpretation": [
            "Completeness is descriptive; it does not identify the missingness mechanism (MCAR/MAR/MNAR).",
            "Constant columns have no observed variation in this dataset and should not enter feature matrices by default.",
            "Identifier-like columns are poor default predictors even when stored as numeric types.",
            "Duplicate rows inflate apparent prevalence and can leak across train/test if not handled at split time.",
            "Text pattern rates are sample estimates; absences of a pattern do not prove the column is clean.",
        ],
        "thresholds": [
            "Low completeness (many missing cells) → require an explicit impute/drop/indicator policy before fitting.",
            "Constant column → exclude from eligible features unless the domain requires a sentinel flag.",
            "Near-unique / id-like column → review role; default to id or ignore, not feature.",
            "Duplicate_row_count > 0 → decide whether duplicates are measurement error, legitimate repeats, or split risk.",
            "Heuristic cutoffs for id-like uniqueness are review flags, not proof of primary-key status.",
        ],
        "assumptions": [
            "Missing cells are correctly represented as nulls in the analysis engine (not sentinel strings unless converted).",
            "Row identity for duplicate detection matches the intended observation unit.",
            "Text pattern sampling is representative enough for a first-pass validity screen.",
            "Role metadata, if set, overrides naive inclusion of id-like or constant columns in later steps.",
        ],
        "pitfalls": [
            "A filled value removes nulls but does not prove missingness is harmless.",
            "Heuristic id detection can misfire on fine-grained categories; review roles.",
            "Duplicate rows can inflate apparent prevalence without proving statistical dependence.",
            "Imputing before role review can bake identifiers or constants into a pipeline.",
            "Treating completeness_score as a single pass/fail grade hides column-level missingness.",
        ],
        "worked_example": {
            "summary": (
                f"{missing:,} cells are missing; observed completeness is {_pct(completeness)}."
            ),
            "values": {
                "missing_cell_count": missing,
                "completeness_score": completeness,
                "constant_columns": constants[:12],
                "id_like_columns": ids[:12],
                "duplicate_row_count": quality.get("duplicate_row_count"),
            },
            "reading": _quality_reading(constants, ids, missing),
        },
        "modeling_impact": (
            "Impute or drop only after role review. Keep identifiers out of feature "
            "matrices unless the domain explicitly requires them. Fit any imputer on "
            "training rows only so holdout missingness does not leak into fill values."
        ),
        "practice_checklist": [
            "List constant and id-like columns and assign roles (feature / id / ignore).",
            "Compute or note per-column missing rates for high-missing fields, not only the global score.",
            "Decide impute vs drop vs missing-indicator before calling session.impute.",
            "Check duplicate_row_count against how you will split (group-aware if needed).",
            "For text columns with pattern flags, spot-check raw values before encoding.",
        ],
        "mastery_notes": [
            "Missingness mechanism determines whether mean/median imputation is defensible; EDA completeness alone cannot settle that.",
            "Near-uniqueness is a leakage prior: even weak association with the target can produce brittle, non-generalizable fits.",
            "Exact duplicate detection misses fuzzy duplicates; domain keys may need group-level deduplication.",
            "Pattern rates on samples are screening tools; production validation needs contract tests on full feeds.",
        ],
        "next_action": {
            "label": "Review roles, then choose a train-fitted imputer",
            "api": 'session.set_roles({...}); session.impute(strategy="median")',
            "parameters": {"strategy": "median"},
            "evidence_keys": ["quality.completeness", "quality.constants", "quality.identifiers"],
        },
        "concepts": ["missing-data", "column-roles", "feature-schema"],
    }


def _studio_features(report: dict[str, Any]) -> dict[str, Any]:
    uni = report.get("univariate") or {}
    per_column = uni.get("per_column") or {}
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    if isinstance(per_column, dict) and per_column:
        for name, stats in per_column.items():
            if not isinstance(stats, dict):
                continue
            kind = str(stats.get("kind", "numeric"))
            if kind == "categorical":
                categorical[str(name)] = stats
            else:
                numeric[str(name)] = stats
    else:
        # Legacy shapes for older report dicts.
        numeric = uni.get("numeric") or uni.get("numeric_summary") or {}
        categorical = uni.get("categorical") or uni.get("categorical_summary") or {}

    skew_ranked = sorted(
        [
            (name, float(stats["skew"]))
            for name, stats in numeric.items()
            if isinstance(stats, dict) and stats.get("skew") is not None
        ],
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:5]
    non_normal = [
        name
        for name, stats in numeric.items()
        if isinstance(stats, dict) and stats.get("appears_non_normal") is True
    ]
    high_card = sorted(
        [
            (name, int(stats.get("nunique") or 0), float(stats.get("entropy_bits") or 0.0))
            for name, stats in categorical.items()
            if isinstance(stats, dict)
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    example_col, example_stats = _first_mapping_item(numeric)
    return {
        "domain": "features",
        "title": "Feature profiles",
        "definition": (
            "Univariate profiling summarizes each column's observed distribution on the "
            "analysis frame: location and scale for numeric columns, skewness and "
            "kurtosis, cardinality and entropy for categoricals, and optional normality "
            "screens.\n\n"
            "The goal is to characterize shape well enough to choose imputation, "
            "scaling, encoding, and outlier policy:not to certify that a column is "
            "Gaussian or that a transform is mandatory."
        ),
        "why": (
            "Scaling, encoding, and outlier policy depend on shape. A right-skewed "
            "income column, a near-binary flag, and a high-cardinality category need "
            "different default treatments.\n\n"
            "Ignoring univariate structure leads to brittle linear baselines, oversized "
            "one-hot matrices, and imputers that target the wrong center of mass."
        ),
        "how": (
            "Numeric and categorical summaries use the analysis frame, which may be "
            "sampled under row budgets. BuildML records per-column stats (mean, median, "
            "std, skew, IQR, nunique, entropy_bits, and related fields).\n\n"
            "Normality screens are hypothesis tests with sample-size limits. A rejected "
            "null is a review flag that the Gaussian model is a poor fit for the sample:"
            "not proof that a particular transform is required for every estimator."
        ),
        "interpretation": [
            "Skew and heavy tails motivate robust imputers, nonlinear models, or careful winsorization:not automatic deletion.",
            "High categorical cardinality widens one-hot encodings and can dominate memory and regularization.",
            "Normality p-values shrink with large n; inspect effect size, skew, and plots alongside the flag.",
            "Entropy on encoded labels measures label diversity, not semantic diversity of the underlying concept.",
            "Compare median vs mean when skew is large; the gap tells you which center an imputer would chase.",
        ],
        "thresholds": [
            "|skew| > 0.75 → review for robust imputation, log/rank transforms, or tree-based models (heuristic review flag).",
            "appears_non_normal = true → normality screen rejected; inspect plots before assuming Gaussian errors.",
            "High nunique relative to rows → encoding and rare-level policy required; consider target or frequency encoding with leakage controls.",
            "Rare_level_rate elevated → holdout may see unseen levels; plan an unknown-category policy.",
            "Heuristic cutoffs are review flags, not proof that a transform is correct.",
        ],
        "assumptions": [
            "Dtype-based numeric vs categorical typing matches the intended measurement scale.",
            "The analysis sample preserves enough mass in the body of each distribution for location/scale estimates.",
            "Normality tests assume i.i.d. draws from a single distribution; mixtures and dependence weaken that reading.",
            "Missing values are handled consistently in per-column counts (nulls excluded from numeric moments).",
        ],
        "pitfalls": [
            "Sampling can miss rare categories that appear in holdout partitions.",
            "A failed normality screen does not mandate a parametric model assumption.",
            "Entropy on encoded labels is not semantic diversity.",
            "Scaling before train/test split leaks holdout scale into the train transform.",
            "Treating every skewed column with the same log1p recipe ignores zeros, negatives, and domain constraints.",
        ],
        "worked_example": {
            "summary": (
                f"Profiled {len(numeric):,} numeric and {len(categorical):,} "
                "categorical columns in the analysis frame."
            ),
            "values": {
                "numeric_columns": len(numeric),
                "categorical_columns": len(categorical),
                "most_skewed": [{"column": name, "skew": skew} for name, skew in skew_ranked],
                "appears_non_normal": non_normal[:12],
                "highest_cardinality": [
                    {"column": name, "nunique": nunique, "entropy_bits": entropy}
                    for name, nunique, entropy in high_card
                ],
                "example_column": example_col,
                "example_stats": _compact_stats(example_stats),
            },
            "reading": _features_reading(
                example_col, example_stats, skew_ranked, non_normal, high_card
            ),
        },
        "modeling_impact": (
            "Tree models tolerate raw scale more than distance-based or strongly "
            "regularized linear models. Match preprocessing to the estimator family you "
            "will compare, and fit scalers/encoders on training rows only."
        ),
        "practice_checklist": [
            "Identify the top skewed numeric columns and decide imputer center (median vs mean).",
            "List high-cardinality categoricals and choose encoding with an unknown-level policy.",
            "For normality-flagged columns, open the matching univariate chart before transforming.",
            "Note which columns will be scaled only after session.split.",
            "Record any intentional raw-scale retention for tree-only workflows.",
        ],
        "mastery_notes": [
            "Skew thresholds are conventional review cues; decision cost depends on estimator sensitivity and outlier policy.",
            "For mixtures (e.g. zero-inflated amounts), univariate moments mislead:segment or use two-part models.",
            "Cardinality interacts with regularization: high-dim one-hots need stronger penalties or alternative encodings.",
            "Re-profile after cleaning; imputation and clipping change skew, IQR, and normality screens.",
        ],
        "next_action": {
            "label": "Scale numeric features after a train split",
            "api": 'session.scale(method="standard")',
            "parameters": {"method": "standard"},
            "evidence_keys": ["eda.scope"],
        },
        "concepts": ["normality-screens", "feature-scaling", "missing-data"],
    }


def _studio_relationships(report: dict[str, Any]) -> dict[str, Any]:
    bivariate = report.get("bivariate") or {}
    mi = bivariate.get("mutual_information_vs_target") or {}
    top_mi = _topk_numeric(mi, k=5)
    corr = bivariate.get("pearson") or bivariate.get("correlation_pearson") or {}
    spearman = bivariate.get("spearman") or {}
    cat_pairs = bivariate.get("categorical_pairs") or []
    top_cramers = sorted(
        [row for row in cat_pairs if isinstance(row, dict) and row.get("cramers_v") is not None],
        key=lambda row: float(row.get("cramers_v") or 0.0),
        reverse=True,
    )[:5]
    return {
        "domain": "relationships",
        "title": "Relationships",
        "definition": (
            "Relationship screens measure pairwise association among features and "
            "between eligible features and the target. Numeric pairs use Pearson and "
            "Spearman correlation; categorical pairs use Cramér's V; target association "
            "uses mutual information (MI) on eligible feature columns.\n\n"
            "Association quantifies statistical dependence in the observed sample. It "
            "does not establish causality, temporal ordering, or fitness for deployment."
        ),
        "why": (
            "Strong association can flag useful predictors, redundant features, or "
            "leakage-like proxies that are unavailable at prediction time. Weak "
            "univariate association does not prove a feature is useless under "
            "interactions or nonlinear models.\n\n"
            "Ranking by MI or correlation focuses inspection time. Deletion or retention "
            "remains a modeling judgment grounded in roles, timing, and domain constraints."
        ),
        "how": (
            "Mutual information uses scikit-learn estimators on the analysis frame for "
            "eligible feature columns versus the target only. Pearson and Spearman "
            "matrices use pairwise complete observations among numeric columns.\n\n"
            "Cramér's V is computed from contingency tables for categorical pairs. "
            "BuildML may also surface top absolute Pearson pairs for quick triage. "
            "All of these are descriptive association screens on the analysis sample."
        ),
        "interpretation": [
            "MI and correlation are associations, not causal effects.",
            "High MI with an identifier-like or post-outcome field is a leakage review flag.",
            "Correlated predictors can split importance in later permutation tests without either being useless.",
            "Spearman can reveal monotonic nonlinear links that Pearson understates; large gaps between them are informative.",
            "Cramér's V near 1 for two categoricals signals redundancy or a deterministic mapping worth role review.",
        ],
        "thresholds": [
            "MI ranks are relative within this dataset; there is no universal 'good MI' cutoff.",
            "|Pearson| or |Spearman| high (e.g. > 0.8) → redundancy / collinearity review flag, not proof of harm.",
            "Cramér's V elevated → categorical association review; check whether one column is a recode of the other.",
            "Top-MI feature that is id-like, post-label, or unavailable at score time → treat as leakage until proven otherwise.",
            "Heuristic cutoffs are review flags, not proof of predictive value or causal effect.",
        ],
        "assumptions": [
            "Eligible features and target roles are correctly assigned before MI is interpreted.",
            "Pairwise complete rows for a correlation cell are representative of the dependence of interest.",
            "MI estimators see enough support per class/level; tiny cells make ranks unstable.",
            "Categorical MI/Cramér's V treat observed level labels as the alphabet (encoding artifacts matter).",
        ],
        "pitfalls": [
            "Sampled rows can distort rare-category MI.",
            "Nonlinear dependence can be invisible to Pearson correlation.",
            "Target leakage cannot be ruled out by association strength alone.",
            "Dropping all correlated features can discard complementary signal.",
            "Comparing raw MI across differently typed features without care can mis-rank continuous vs discrete columns.",
        ],
        "worked_example": {
            "summary": (
                f"Computed mutual information for {len(mi):,} eligible features against the target."
            ),
            "values": {
                "mi_top": top_mi,
                "mi_feature_count": len(mi),
                "correlation_available": bool(corr),
                "spearman_available": bool(spearman),
                "cramers_v_top": top_cramers,
            },
            "reading": (
                (
                    "Highest observed MI values: "
                    + ", ".join(f"{name}={value:.4f}" for name, value in top_mi)
                    + ". Review whether any top feature is a proxy for the outcome."
                )
                if top_mi
                else "No MI values were available (missing target or eligible features)."
            ),
        },
        "modeling_impact": (
            "Use association ranks to prioritize inspection, not automatic feature "
            "deletion. Confirm leakage-safe timing and roles before fitting transforms "
            "or comparing estimators on the associated features."
        ),
        "practice_checklist": [
            "List top MI features and verify each is available at prediction time.",
            "Scan top absolute Pearson pairs for redundant numeric blocks.",
            "Check Cramér's V leaders for categorical recodes or hierarchy leakage.",
            "Compare Pearson vs Spearman for key pairs when nonlinearity is plausible.",
            "Do not drop a feature solely because univariate MI is low:note interaction hypotheses.",
        ],
        "mastery_notes": [
            "Associations do not establish causality; interventional or temporal design is required for causal claims.",
            "MI is invariant to invertible transforms in theory but estimator bias/variance still depend on binning and sample size.",
            "Redundancy among features is sometimes desirable for robustness; collinearity cost is estimator-specific.",
            "Leakage review is a timing and availability problem first; association strength is only a symptom.",
        ],
        "next_action": {
            "label": "Compare simple estimators after leakage review",
            "api": "session.workflow()",
            "parameters": {},
            "evidence_keys": [],
        },
        "concepts": ["mutual-information", "feature-importance", "leakage-boundary"],
    }


def _studio_multivariate(report: dict[str, Any]) -> dict[str, Any]:
    multi = report.get("multivariate") or {}
    vif_rows = multi.get("vif") or []
    pca = multi.get("pca") or {}
    overview = report.get("overview") or {}
    scope = overview.get("preprocess_scope_status") or {}
    fold_reduce = (scope.get("fold_local") or {}).get("reduce")
    session_reduce = bool((scope.get("session_global") or {}).get("reduce_dimensions"))
    top_vif = sorted(
        [row for row in vif_rows if isinstance(row, dict)],
        key=lambda row: float(row.get("vif") or 0.0),
        reverse=True,
    )[:5]
    pca_scope_line = None
    if fold_reduce is not None:
        pca_scope_line = (
            f"EDA PCA below is a descriptive multivariate screen. Separately, "
            f"PreprocessRecipe reduce={fold_reduce!r} was recorded for fold-local CV "
            "(fold-train fit; not this EDA analyzer)."
        )
    elif session_reduce:
        pca_scope_line = (
            "EDA PCA below is a descriptive multivariate screen. Separately, "
            "Session.reduce_dimensions fitted a Session-global PCA plan on train "
            "(not this EDA analyzer)."
        )
    interpretation = [
        "VIF above 5 is a collinearity review flag, not proof of multicollinearity harm for every estimator.",
        "Dropping one of a correlated pair is a modeling choice with information loss; document the trade.",
        "PCA components are linear mixes; interpret loadings before using them as features.",
        "Tree ensembles often tolerate collinearity better than unregularized linear models; match the remedy to the model class.",
        "A large first PCA component means shared scale/variance:not necessarily a single latent concept.",
    ]
    pitfalls = [
        "VIF requires adequate numeric complete cases.",
        "PCA on unscaled features is dominated by large-magnitude columns.",
        "Dimensionality reduction fitted on full data before splitting leaks holdout structure.",
        "Dropping all high-VIF features can remove the only measurement of a latent factor.",
        "Interpreting PCA axes as causal factors without loadings and domain review is unjustified.",
    ]
    if pca_scope_line:
        interpretation.insert(2, pca_scope_line)
        pitfalls.append(
            "Treating the EDA multivariate PCA table as evidence that fold-local or "
            "Session.reduce_dimensions already ran; those are separate fit scopes."
        )
    return {
        "domain": "multivariate",
        "title": "Multivariate structure",
        "definition": (
            "Multivariate screens inspect joint numeric structure: variance inflation "
            "factors (VIF) for collinearity, optional correlation clustering cues, and "
            "a principal component analysis (PCA) summary of explained variance.\n\n"
            "These tools describe linear shared structure among numeric eligible "
            "features. They do not by themselves create a leakage-safe feature set or "
            "prove that a column should be dropped."
        ),
        "why": (
            "Collinearity inflates coefficient variance in linear models and can make "
            "feature importance unstable across resamples. Knowing which columns move "
            "together changes how you interpret coefficients, regularization, and "
            "ablation studies.\n\n"
            "PCA summarizes shared variance for diagnosis or controlled reduction. Fitted "
            "on full data before splitting, it can leak holdout structure into components."
        ),
        "how": (
            "VIF is estimated from numeric eligible features after basic cleaning: each "
            "feature is regressed on the others and VIF = 1 / (1 − R²). BuildML caps the "
            "column budget for this screen so large wide tables remain tractable.\n\n"
            "PCA reports explained-variance ratios for the computed components on the "
            "numeric block used by the analyzer. Interpret loadings before promoting "
            "components to production features."
        ),
        "interpretation": interpretation,
        "thresholds": [
            "VIF > 5 → collinearity review flag (common heuristic); inspect partners before deleting.",
            "VIF ≫ 10 → stronger review pressure for linear/GLM coefficient interpretation.",
            "PCA: cumulative explained variance depends on scaling; compare ratios only after consistent preprocessing.",
            "Heuristic cutoffs are review flags, not proof that coefficients are unusable.",
            "If VIF rows are empty, numeric complete cases or column budget may have blocked the screen.",
        ],
        "assumptions": [
            "Numeric features are roughly linearly related for VIF/PCA to be meaningful diagnostics.",
            "Adequate complete cases exist across the numeric block used for VIF.",
            "Column scales are comparable or will be scaled before interpreting PCA loadings/variance shares.",
            "Eligible feature membership excludes ids and constants that would distort joint structure.",
        ],
        "pitfalls": pitfalls,
        "worked_example": {
            "summary": f"VIF computed for {len(vif_rows):,} numeric feature rows.",
            "values": {
                "vif_top": top_vif,
                "pca_explained_variance_ratio": pca.get("explained_variance_ratio"),
                "pca_n_components": pca.get("n_components"),
                "fold_local_reduce": fold_reduce,
                "session_reduce_dimensions": session_reduce,
            },
            "reading": (
                (
                    "Highest VIF rows: "
                    + ", ".join(
                        f"{row.get('column')}={float(row.get('vif')):.2f}"
                        for row in top_vif
                        if row.get("vif") is not None
                    )
                    + ". Review before interpreting linear coefficients."
                    + (f" {pca_scope_line}" if pca_scope_line else "")
                )
                if top_vif
                else (
                    pca_scope_line
                    if pca_scope_line
                    else "No VIF rows were available for a worked example."
                )
            ),
        },
        "modeling_impact": (
            "Prefer tree ensembles or regularization when collinearity is high and "
            "coefficients must remain stable. Fit any PCA or other reducer only on "
            "training rows, then transform validation/test with the frozen mapping."
        ),
        "practice_checklist": [
            "List columns with VIF > 5 and identify their correlated partners.",
            "Decide retain / drop / regularize / combine for each collinear block.",
            "If using PCA as features, plan fit-on-train only and record n_components.",
            "Re-check coefficient signs after addressing collinearity in linear baselines.",
            "Confirm id-like columns were excluded from the numeric VIF block.",
        ],
        "mastery_notes": [
            "VIF diagnoses linear dependence among predictors; it is silent about nonlinear redundancy.",
            "Condition indices and variance-decomposition proportions refine VIF when many columns share structure.",
            "Partial dependence and permutation importance remain fragile under strong collinearity:report blocks, not isolated ranks.",
            "PCA for compression and PCA for visualization have different success criteria; do not conflate them.",
        ],
        "next_action": {
            "label": "Fit preprocessing only after split",
            "api": "session.split(test_size=0.2, stratify=True)",
            "parameters": {"test_size": 0.2, "stratify": True},
            "evidence_keys": [],
        },
        "concepts": ["variance-inflation", "principal-components", "leakage-boundary"],
    }


def _studio_target(report: dict[str, Any]) -> dict[str, Any]:
    target = report.get("target") or {}
    drift = report.get("drift") or {}
    flagged = flagged_column_names(drift.get("flagged_columns"))
    return {
        "domain": "target",
        "title": "Target and validation drift",
        "definition": (
            "Target screens describe the label for the declared task: class balance and "
            "support for classification, or range and moments for regression. Drift "
            "screens compare feature distributions between train and test partitions "
            "when a Session split exists.\n\n"
            "Together they answer: what outcome are we predicting, how imbalanced is it, "
            "and do train/test inputs look exchangeable under the current split?"
        ),
        "why": (
            "Severe imbalance changes which metrics and thresholds matter. A 1% positive "
            "class makes accuracy misleading and forces explicit recall/precision or "
            "cost-sensitive choices.\n\n"
            "Train/test drift can mean the random split is unrealistic, duplicates "
            "crossed partitions, or the collection process shifted. Ignoring drift "
            "produces optimistic scores that fail on the next batch."
        ),
        "how": (
            "Classification balance uses observed class counts on the analysis frame. "
            "Regression target summaries use the same univariate tooling as features "
            "for the label column.\n\n"
            "Drift uses partition membership from the Session split plan and flags "
            "columns by configured distributional distances on numeric and categorical "
            "features. If no split exists, drift is unavailable and the report states why."
        ),
        "interpretation": [
            "Report the partition with every later score (train / validation / test).",
            "A drift flag is a review prompt; effect size, support, and domain timing still matter.",
            "Validation supports choices; test estimates performance after choices are fixed.",
            "Class imbalance is about decision thresholds and metrics, not only resampling.",
            "Drift without labels diagnoses input change, not whether the model got worse.",
        ],
        "thresholds": [
            "Minority class fraction very small → prefer precision/recall/PR-AUC or cost metrics over raw accuracy.",
            "Drift flagged_columns non-empty → inspect collection timing, duplicates, and split strategy before model ranking.",
            "drift.available = false → do not claim train/test stability; create a split first if that claim matters.",
            "Heuristic drift cutoffs are review flags, not proof that deployment will fail.",
            "Resampling changes training prevalence, not the prevalence of an untouched holdout.",
        ],
        "assumptions": [
            "Target column semantics match the declared task (classification vs regression).",
            "Split membership aligns with row order and the intended population unit.",
            "Drift distances are comparable only for columns with adequate support in both partitions.",
            "Label definitions are stable across partitions (no silent relabeling mid-collection).",
        ],
        "pitfalls": [
            "Resampling changes training prevalence, not holdout prevalence.",
            "Drift without labels diagnoses input change, not model quality change.",
            "Repeatedly tuning on test folds the test into training.",
            "Stratify on a proxy that leaks the label can hide true deployment shift.",
            "Comparing models on different undocumented partitions makes bake-offs meaningless.",
        ],
        "worked_example": {
            "summary": (
                f"Target task={target.get('task') or target.get('type') or 'unspecified'}; "
                f"drift available={drift.get('available')}; "
                f"flagged columns={len(flagged)}."
            ),
            "values": {
                "target": target,
                "drift_available": drift.get("available"),
                "flagged_columns": flagged[:20],
                "drift_method": drift.get("method") or drift.get("summary"),
            },
            "reading": (
                (
                    "Flagged drift columns include "
                    + ", ".join(flagged[:8])
                    + ". Inspect collection timing and split assumptions before model ranking."
                )
                if flagged
                else (
                    "No columns were flagged for train/test drift, or drift was unavailable "
                    "because no split exists."
                )
            ),
        },
        "modeling_impact": (
            "Choose metrics and decision thresholds on validation data. Confirm once on "
            "untouched test data. If drift is flagged, fix the split or collection story "
            "before trusting estimator rankings."
        ),
        "practice_checklist": [
            "State task type and class fractions or regression range in one sentence.",
            "If imbalanced, pick primary metrics before fitting candidates.",
            "Create or confirm a split; re-check drift.available.",
            "For each flagged drift column, decide benign shift vs split defect vs leakage.",
            "Freeze test until validation-driven choices are done.",
        ],
        "mastery_notes": [
            "Prevalence shift and covariate shift need different remedies; conflating them wastes effort.",
            "Thresholds chosen on validation should be re-evaluated when deployment prevalence differs.",
            "Group-aware or time-based splits often beat i.i.d. random splits for operational data.",
            "A clean drift screen under a random split does not prove temporal stability in production.",
        ],
        "next_action": {
            "label": "Inspect workflow readiness for modeling",
            "api": "session.walkthrough()",
            "parameters": {},
            "evidence_keys": [],
        },
        "concepts": ["class-imbalance", "dataset-drift", "evaluation-partitions"],
    }


def _studio_outliers(report: dict[str, Any]) -> dict[str, Any]:
    outliers = report.get("outliers") or {}
    per_column = outliers.get("per_column") or outliers.get("univariate") or {}
    multi = outliers.get("multivariate") or {}
    ranked: list[tuple[str, float, int, list[Any]]] = []
    if isinstance(per_column, dict):
        for name, stats in per_column.items():
            if not isinstance(stats, dict):
                continue
            rate = stats.get("iqr_outlier_rate")
            if rate is None:
                rate = stats.get("rate") or stats.get("outlier_rate")
            if rate is None:
                continue
            count = int(stats.get("iqr_outlier_count") or stats.get("count") or 0)
            bounds = list(stats.get("iqr_bounds") or [])
            ranked.append((str(name), float(rate), count, bounds))
    ranked.sort(key=lambda item: item[1], reverse=True)
    top = ranked[:5]
    anomaly_rate = multi.get("anomaly_rate") if isinstance(multi, dict) else None
    anomaly_count = multi.get("anomaly_count") if isinstance(multi, dict) else None
    return {
        "domain": "outliers",
        "title": "Outlier screens",
        "definition": (
            "Outlier screens mark unusual univariate values and multivariate extremes "
            "relative to the observed analysis frame. Univariate rules flag points "
            "outside fences or far from a Gaussian center; multivariate scoring looks "
            "for jointly rare numeric rows.\n\n"
            "A flag means 'review this point or column.' It does not mean the value is "
            "an error, nor that deletion improves generalization."
        ),
        "why": (
            "Extreme points can dominate means, correlations, and linear fits. They may "
            "be data-entry errors, rare-but-valid events, sensor spikes, or ordinary "
            "heavy tails.\n\n"
            "Policy choices:keep, cap, transform, or drop:change the estimand. Making "
            "that choice explicit protects later metrics from silent, irreversible edits."
        ),
        "how": (
            "Univariate flags use the 1.5×IQR rule (Tukey fences) and a |z| > 3 Gaussian "
            "screen on numeric columns in the analysis frame.\n\n"
            "When at least two numeric features and at least 30 complete rows exist, "
            "IsolationForest reports a multivariate anomaly rate and count. If those "
            "support conditions fail, the multivariate block is unavailable."
        ),
        "interpretation": [
            "A flagged point is a review candidate, not automatic deletion criteria.",
            "Winsorizing or dropping rows changes the estimand; record the choice in operation history.",
            "Outlier rate depends on the analysis sample and column budget.",
            "High IQR rates on heavy-tailed legitimate variables are expected under Gaussian heuristics.",
            "Multivariate anomalies can be jointly rare even when each margin looks ordinary.",
        ],
        "thresholds": [
            "IQR fence at 1.5×IQR beyond Q1/Q3 → univariate review flag (Tukey heuristic).",
            "|z| > 3 → Gaussian-tail review flag; sensitive to mean/std distortion from the outliers themselves.",
            "IsolationForest anomaly_rate → joint-rarity screen when ≥2 numeric features and ≥30 complete rows.",
            "Heuristic cutoffs are review flags, not proof that rows should be deleted.",
            "Compare outlier counts to domain base rates before treating a rate as 'too high.'",
        ],
        "assumptions": [
            "Numeric columns are on scales where IQR or z-score distances are meaningful.",
            "Complete-case rows for IsolationForest represent the joint distribution of interest.",
            "The analysis sample is large enough that fences are not dominated by a handful of points.",
            "Label information is not used to decide outlier drops on holdout rows (leakage risk).",
        ],
        "pitfalls": [
            "Heavy-tailed legitimate data will look outlier-heavy under Gaussian heuristics.",
            "Removing outliers after peeking at test labels is a leakage risk.",
            "Multivariate distance needs complete numeric cases.",
            "Capping on full data before split leaks holdout extremes into train limits.",
            "Deleting anomalies that define the minority class can erase the signal you care about.",
        ],
        "worked_example": {
            "summary": (
                f"IQR screens covered {len(ranked):,} numeric feature columns"
                + (
                    f"; multivariate anomaly rate={float(anomaly_rate):.1%} ({anomaly_count} rows)."
                    if anomaly_rate is not None
                    else "; multivariate screen unavailable for this frame."
                )
            ),
            "values": {
                "columns_screened": len(ranked),
                "top_iqr_rates": [
                    {
                        "column": name,
                        "iqr_outlier_rate": rate,
                        "iqr_outlier_count": count,
                        "iqr_bounds": bounds,
                    }
                    for name, rate, count, bounds in top
                ],
                "multivariate": {
                    "method": multi.get("method") if isinstance(multi, dict) else None,
                    "anomaly_rate": anomaly_rate,
                    "anomaly_count": anomaly_count,
                    "n_rows_scored": multi.get("n_rows_scored")
                    if isinstance(multi, dict)
                    else None,
                },
                "feature_columns_analyzed": (outliers.get("feature_columns_analyzed") or [])[:20],
            },
            "reading": (
                (
                    "Highest IQR outlier rates: "
                    + ", ".join(f"{name}={rate:.1%} (n={count})" for name, rate, count, _ in top)
                    + ". Inspect those columns before fitting sensitive linear baselines."
                )
                if top
                else "No IQR outlier rates were available for a worked example."
            ),
        },
        "modeling_impact": (
            "Robust loss, tree models, or explicit caps can reduce sensitivity to "
            "extremes. Document any row drops or winsorization in operation history so "
            "metrics remain comparable across experiments."
        ),
        "practice_checklist": [
            "Inspect top IQR-rate columns in raw units and decide error vs valid tail.",
            "Choose a policy: keep, winsorize, transform, or drop:and apply it only with train-fit discipline.",
            "If IsolationForest ran, sample flagged rows and check joint plausibility.",
            "Re-run univariate summaries after any capping or deletion.",
            "Avoid test-driven outlier deletion; freeze rules on train/validation only.",
        ],
        "mastery_notes": [
            "IQR fences assume a roughly unimodal bulk; mixtures need segment-wise screens.",
            "z-score flags are circular when mean/std are estimated on the contaminated sample:consider robust centers.",
            "IsolationForest contamination and random_state choices change rates; treat the rate as a screen, not a prevalence estimate.",
            "Outlier policy is part of the estimand: report it beside metrics or the comparison is invalid.",
        ],
        "next_action": {
            "label": "Re-run EDA after cleaning decisions",
            "api": "session.eda(include_plots=False, show=True)",
            "parameters": {"include_plots": False, "show": True},
            "evidence_keys": [],
        },
        "concepts": ["diagnostic-uncertainty", "missing-data"],
    }


def _studio_visuals(report: dict[str, Any]) -> dict[str, Any]:
    plan = report.get("adaptive_plan") or []
    return {
        "domain": "visuals",
        "title": "Visual evidence",
        "definition": (
            "Visual boards render adaptive Plotly charts chosen from data shape, roles, "
            "cardinality, and missingness:not a fixed chart checklist. Each figure is "
            "tied to analyzer tables already computed for the session.\n\n"
            "Charts are evidence surfaces for claims you make in text: they should make "
            "distribution shape, imbalance, association, and sampling limits inspectable."
        ),
        "why": (
            "Tables hide distribution shape and joint structure. A skew statistic without "
            "a histogram, or a correlation without a scatter, is easy to over-trust.\n\n"
            "Adaptive selection spends attention where the data can support a chart. Empty "
            "boards usually mean missing roles, unsupported dtypes, or insufficient rows:"
            "not a silent pass."
        ),
        "how": (
            "During EDA, BuildML builds an adaptive plan: a list of chart intents keyed "
            "by kind and evidence needs. The dashboard converts those intents plus "
            "analyzer tables into interactive Plotly figures with a shared palette.\n\n"
            "Exports can freeze stills (PNG/PDF) or package offline HTML. Stills lose "
            "hover and zoom; use interactive boards when the claim depends on a local "
            "region of the plot."
        ),
        "interpretation": [
            "Expand a figure before accepting a summary sentence that cites it.",
            "Hover values are observations from the analysis frame (possibly sampled).",
            "Empty boards mean the analyzer lacked support (roles, dtypes, or rows).",
            "Color and size encodings are visual aids, not significance tests.",
            "Match each cockpit finding to a chart kind in the adaptive plan when one exists.",
        ],
        "thresholds": [
            "plan_count = 0 → no chart intents; fix roles/dtypes/row support before expecting visuals.",
            "Large n with overplotting → density or aggregation views beat raw scatter for rate claims.",
            "Sampled analysis frame → treat sparse categories and tails in plots as provisional.",
            "Export stills are review artifacts; they are not interactive evidence for fine structure.",
            "Visual salience is not a p-value; do not promote a bright pattern to causation.",
        ],
        "assumptions": [
            "Adaptive plan kinds correspond to analyzer outputs present in this report.",
            "Plotly rendering environment can load the figure payloads in the dashboard or export path.",
            "Axis labels and roles reflect the current Session schema.",
            "Viewers understand that association plots do not establish causality.",
        ],
        "pitfalls": [
            "Overplotting can exaggerate density in large samples.",
            "Color encodings are not significance tests.",
            "Exported PNG/PDF snapshots lose interactivity.",
            "Cherry-picking zoom regions after seeing labels is a soft leakage / HARKing risk.",
            "Using charts fitted narratives without returning to numeric tables invites confirmation bias.",
        ],
        "worked_example": {
            "summary": f"Adaptive plan contains {len(plan):,} chart specification(s).",
            "values": {
                "plan_count": len(plan),
                "plan_kinds": sorted(
                    {str(item.get("kind")) for item in plan if isinstance(item, dict)}
                ),
            },
            "reading": (
                "Open the Visual evidence board and drill from a cockpit finding into "
                "the matching chart kind listed in the adaptive plan."
            ),
        },
        "modeling_impact": (
            "Use charts to justify preprocessing choices you will later freeze on train. "
            "If a plot drives a decision, record the decision and the evidence key so "
            "the modeling path stays reproducible."
        ),
        "practice_checklist": [
            "Open at least one chart per high/critical cockpit finding when a kind exists.",
            "For skewed columns, compare histogram/ECDF to the skew statistic.",
            "For top MI or correlation pairs, inspect the bivariate plot before dropping features.",
            "Note sampling disclosure on any plot used in a written claim.",
            "After export, verify that stills still support the same claim without hover.",
        ],
        "mastery_notes": [
            "Adaptive charts are a query plan over evidence, not a complete atlas of the data.",
            "Perception biases (scale truncation, alpha stacking) can manufacture drama:check axes and counts.",
            "Reproducible visual claims cite analyzer tables and plan kinds, not only screenshots.",
            "When plots and tables disagree, trust neither blindly: recompute the underlying slice.",
        ],
        "next_action": {
            "label": "Export PDF stills or offline Studio HTML after review",
            "api": (
                "Use 'PDF briefing' (tables + static Plotly PNG stills) or "
                "'Offline HTML' in the app header, or "
                "session.eda(export_html=..., html_format='studio')"
            ),
            "parameters": {},
            "evidence_keys": [],
        },
        "concepts": ["diagnostic-uncertainty", "reproducibility"],
    }


def _features_reading(
    example_col: str | None,
    example_stats: Any,
    skew_ranked: list[tuple[str, float]],
    non_normal: list[str],
    high_card: list[tuple[str, int, float]],
) -> str:
    parts: list[str] = []
    if example_col:
        parts.append(f"Example column `{example_col}` shows {_format_stats(example_stats)}.")
    if skew_ranked:
        name, skew = skew_ranked[0]
        parts.append(f"Most skewed numeric column: `{name}` (skew={skew:.3f}).")
    if non_normal:
        parts.append(
            "Normality screens flagged: "
            + ", ".join(non_normal[:6])
            + ("…" if len(non_normal) > 6 else "")
            + "."
        )
    if high_card:
        name, nunique, entropy = high_card[0]
        parts.append(
            f"Highest categorical cardinality: `{name}` "
            f"({nunique} levels, entropy={entropy:.3f} bits)."
        )
    if not parts:
        return "No univariate summary was available for a worked example."
    parts.append("Use these shapes when choosing imputation, scaling, and encoding.")
    return " ".join(parts)


def _compact_stats(stats: Any) -> dict[str, Any] | Any:
    if not isinstance(stats, dict):
        return stats
    keep = (
        "kind",
        "count",
        "mean",
        "std",
        "median",
        "skew",
        "kurtosis",
        "iqr",
        "min",
        "max",
        "nunique",
        "entropy_bits",
        "mode",
        "rare_level_rate",
        "normality_method",
        "normality_pvalue",
        "appears_non_normal",
    )
    return {key: stats[key] for key in keep if key in stats}


def _quality_reading(constants: list[str], ids: list[str], missing: int) -> str:
    parts = []
    if missing:
        parts.append(f"{missing:,} missing cells require an explicit impute/drop policy.")
    if constants:
        parts.append(
            "Constant columns observed: "
            + ", ".join(constants[:8])
            + ("…" if len(constants) > 8 else "")
            + "."
        )
    if ids:
        parts.append(
            "Identifier-like columns observed: "
            + ", ".join(ids[:8])
            + ("…" if len(ids) > 8 else "")
            + "."
        )
    return " ".join(parts) if parts else "No constant or identifier-like flags were raised."


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.3%}"
    except (TypeError, ValueError):
        return "not available"


def _first_mapping_item(mapping: Any) -> tuple[str | None, Any]:
    if isinstance(mapping, dict) and mapping:
        key = next(iter(mapping))
        return str(key), mapping[key]
    if isinstance(mapping, list) and mapping:
        first = mapping[0]
        if isinstance(first, dict) and "column" in first:
            return str(first["column"]), first
    return None, None


def _format_stats(stats: Any) -> str:
    if not isinstance(stats, dict):
        return str(stats)
    pieces = []
    for key in ("mean", "std", "median", "skew", "n_unique", "missing_rate"):
        if key in stats and stats[key] is not None:
            value = stats[key]
            pieces.append(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}")
    return ", ".join(pieces[:6]) if pieces else str({k: stats[k] for k in list(stats)[:6]})


def _topk_numeric(mapping: dict[str, Any], *, k: int) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for key, value in mapping.items():
        try:
            scored.append((str(key), float(value)))
        except (TypeError, ValueError):
            if isinstance(value, dict) and "score" in value:
                try:
                    scored.append((str(key), float(value["score"])))
                except (TypeError, ValueError):
                    continue
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
