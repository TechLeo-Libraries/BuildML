# ruff: noqa: E501
"""Adaptive teaching depth for Readiness Gates deep-dive sidebar.

Each gate row can carry beginner→advanced pedagogy, session-number
calculations when relevant, and copy-paste Session API examples that name
columns from *this* report. Human marks stay UI-only; copy never implies
persistence to session or disk.
"""

from __future__ import annotations

import math
from typing import Any

STATUS_MEANINGS: dict[str, str] = {
    "clear": (
        "Settled by the frame — this report’s own numbers already answer the "
        "question for the current extract. Re-check after the next ingest."
    ),
    "open": (
        "Open and measurable — something countable is unresolved. The evidence "
        "and “closes when” lines name the number or check that would settle it."
    ),
    "human": (
        "Needs a human judgment — the dataset cannot answer this alone. A person "
        "must decide and write the answer down outside BuildML. You may mark the "
        "gate for this browser tab only; that mark is never saved."
    ),
    "na": (
        "Not applicable — the question does not arise for this frame (for "
        "example no target, no time column, or no gappy columns). Kept visible "
        "so the next extract does not inherit silence."
    ),
    "session_mark": (
        "“Mark for this session” is a local reminder in this browser tab. "
        "Refreshing the App clears it. BuildML never writes gate judgments to "
        "the Session, history, disk, or any dataset copy."
    ),
}


def _fmt_n(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_pct(rate: float, digits: int = 1) -> str:
    return f"{rate * 100:.{digits}f}%"


def _names(items: list[Any], limit: int = 3) -> list[str]:
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("column") or item.get("feature") or "")
        else:
            name = str(item)
        if name and name not in out:
            out.append(name)
        if len(out) >= limit:
            break
    return out


def _quote_list(names: list[str]) -> str:
    if not names:
        return '["your_column"]'
    return "[" + ", ".join(f'"{n}"' for n in names) + "]"


def _target_name(ctx: dict[str, Any]) -> str | None:
    target = ctx.get("target")
    if isinstance(target, dict) and target.get("name"):
        return str(target["name"])
    return None


def _task(ctx: dict[str, Any]) -> str | None:
    target = ctx.get("target")
    if isinstance(target, dict):
        return str(target.get("task") or "") or None
    return None


def _feature_sample(ctx: dict[str, Any], limit: int = 3) -> list[str]:
    numeric = _names(ctx.get("numeric") or [], limit)
    if numeric:
        return numeric
    cats = [str(x) for x in (ctx.get("categorical") or [])[:limit]]
    if cats:
        return cats
    cols = _names(ctx.get("cols") or [], limit)
    return cols or ["feature_a"]


def _roles_snippet(ctx: dict[str, Any]) -> str:
    target = _target_name(ctx)
    feats = _feature_sample(ctx, 3)
    parts = [f'"{f}": "feature"' for f in feats]
    if target:
        parts.append(f'"{target}": "target"')
    id_like = _names(ctx.get("idLike") or [], 1)
    if id_like:
        parts.append(f'"{id_like[0]}": "id"')
    return "{\n    " + ",\n    ".join(parts) + ",\n}"


def _ingest_header(ctx: dict[str, Any]) -> str:
    rows = _fmt_n(ctx.get("rows") or 0)
    cols = ctx.get("colCount") or 0
    engine = (ctx.get("ds") or {}).get("engine") or "pandas"
    return (
        f"# Adaptive to this session: {_fmt_n(ctx.get('rowsTotal') or rows)} rows "
        f"loaded, {rows} analysed, {cols} columns, engine={engine}\n"
        "from buildml import Session\n"
        "# Assume `frame` is your DataFrame (or pass a path to Session.ingest).\n"
    )


def _calc(
    *,
    label: str,
    formula: str,
    inputs: dict[str, Any],
    result: str,
    reading: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "formula": formula,
        "inputs": inputs,
        "result": result,
        "reading": reading,
    }


def _example(
    *,
    summary: str,
    code: str,
    change_these: list[str],
    flexible: list[str],
    reading: str,
) -> dict[str, Any]:
    return {
        "summary": summary,
        "code": code.rstrip() + "\n",
        "change_these": change_these,
        "flexible": flexible,
        "reading": reading,
    }


def _levels(beginner: str, intermediate: str, advanced: str) -> dict[str, str]:
    return {
        "beginner": beginner,
        "intermediate": intermediate,
        "advanced": advanced,
    }


# Per-gate static pedagogy (what / why / advanced). Adaptive fields fill at build time.
_CURRICULUM: dict[str, dict[str, Any]] = {
    "00.1": {
        "beginner": "Write one plain sentence: who will act on the model’s output, and at what moment.",
        "why": "Without a decision owner and trigger, every later metric is floating — you cannot tell if a score is good enough to act.",
        "levels": _levels(
            "Name the person/role, the action, and the moment (e.g. “loan officer reviews score before offer”).",
            "Tie the sentence to an output type (rank, probability, class) and a review cadence.",
            "State the fallback when the model is unavailable and who can override it.",
        ),
        "next": [
            "Draft the decision sentence in your runbook (outside BuildML).",
            "Confirm the output form matches that decision (class vs probability vs rank).",
        ],
    },
    "00.2": {
        "beginner": "Say what one row is (a customer, an order, a day…) and prove uniqueness with a key.",
        "why": "Wrong grain silently duplicates entities and inflates performance after a random split.",
        "levels": _levels(
            "Write “one row = …” and name the candidate key column.",
            "Run a uniqueness check on that key before any split.",
            "Document composite keys and slowly-changing entities if the grain is temporal.",
        ),
        "next": [
            "Assign id role to the key with session.set_roles.",
            "Deduplicate before split if exact duplicate rows exist.",
        ],
    },
    "00.3": {
        "beginner": "Know which rows were filtered out before this extract arrived.",
        "why": "Silent filters change the population; holdout scores stop representing production.",
        "levels": _levels(
            "Ask for the extract query and date window.",
            "List inclusion/exclusion rules and null-drop policies.",
            "Compare extract counts to the operational source of truth.",
        ),
        "next": [
            "Record the extract SQL / job id next to the pinned dataset.",
            "Note whether sampling was applied in this EDA pass.",
        ],
    },
    "00.4": {
        "beginner": "Write the exact label rule: what counts as positive, from when, over how long.",
        "why": "Ambiguous labels leak future information or mix incompatible outcomes into one column.",
        "levels": _levels(
            "Name the target column and whether the task is classification or regression.",
            "Write anchor time, horizon, and censoring/exclusion rules.",
            "Version the label definition with the extract so retrains stay comparable.",
        ),
        "next": [
            "Declare the target with session.set_roles before modeling.",
            "Keep post-outcome columns out of the feature matrix.",
        ],
    },
    "00.5": {
        "beginner": "For each column, know its source system and whether the value exists at prediction time.",
        "why": "Training on values that arrive only after the decision is leakage; production will not have them.",
        "levels": _levels(
            "List source system per column family.",
            "Mark known-at-prediction-time vs post-outcome fields.",
            "Attach refresh cadence and late-arriving data policies.",
        ),
        "next": [
            "Build a known-at-time inventory before feature selection.",
            "Drop or ignore columns that only exist after the outcome.",
        ],
    },
    "00.6": {
        "beginner": "List which columns are personal or legally protected.",
        "why": "Protected attributes need evaluation and governance even when they are not model inputs.",
        "levels": _levels(
            "Tag columns as personal, protected, or neither.",
            "Decide which stay for fairness evaluation only.",
            "Document retention and access controls for sensitive fields.",
        ),
        "next": [
            "Set ignore roles for columns that must not train the model.",
            "Keep evaluation-only attributes available for slice checks later.",
        ],
    },
    "01.1": {
        "beginner": "Set each column’s type on purpose instead of trusting CSV guesses.",
        "why": "Wrong dtypes break imputers, encoders, and date parsing before any model runs.",
        "levels": _levels(
            "Confirm numeric vs categorical vs datetime for every column.",
            "Parse dates and categoricals at load with explicit dtypes.",
            "Assert dtypes in CI so schema drift fails loudly.",
        ),
        "next": [
            "Fix dtypes before session.impute / session.encode.",
            "Re-run EDA after schema corrections.",
        ],
    },
    "01.2": {
        "beginner": "Every gappy column needs a fill plan fitted only on training rows.",
        "why": "Filling with global medians before the split leaks test information into train.",
        "levels": _levels(
            "List columns with missing cells and their rates.",
            "Choose a strategy (median/mode/constant) per column family.",
            "Add missingness indicators when the gap itself may be informative.",
        ),
        "next": [
            "Call session.impute after session.split.",
            "Prefer PreprocessRecipe inside CV so each fold refits.",
        ],
    },
    "01.3": {
        "beginner": "Decide whether gaps are random or systematic (MCAR / MAR / MNAR).",
        "why": "Systematic missingness changes which imputer is honest and when deletion biases the population.",
        "levels": _levels(
            "Compare missing rates across groups.",
            "Relate missingness to other observed columns.",
            "Treat MNAR as a modeling/domain problem, not only a fill trick.",
        ),
        "next": [
            "Plot missingness vs candidate drivers before choosing a strategy.",
            "Document the mechanism judgment outside the App (not persisted here).",
        ],
    },
    "01.4": {
        "beginner": "Count exact duplicate rows and repeated keys, then resolve them.",
        "why": "Duplicates overweight entities and can appear in both train and test after a random split.",
        "levels": _levels(
            "Count exact duplicate rows on the analysis frame.",
            "Check uniqueness on the entity key.",
            "Deduplicate before split; define near-duplicate rules after text cleaning.",
        ),
        "next": [
            "Drop exact duplicates at the stated grain before splitting.",
            "Keep an audit count of removed rows.",
        ],
    },
    "01.5": {
        "beginner": "Columns stuck on one value waste capacity — drop, coarsen, or keep on purpose.",
        "why": "Constants and near-constants add noise and can dominate regularized models.",
        "levels": _levels(
            "List constant and near-constant columns from quality screens.",
            "Drop pure constants; decide for near-constants with domain input.",
            "Re-check after filtering — rarity can change.",
        ),
        "next": [
            "session.drop_columns([...]) for confirmed constants.",
            "Coarsen rare levels before one-hot encoding.",
        ],
    },
    "01.6": {
        "beginner": "High-cardinality categories need an encoding plan and an unseen-level policy.",
        "why": "Naive one-hot explodes width; unseen levels at predict time break pipelines.",
        "levels": _levels(
            "Flag categoricals above ~20 levels.",
            "Choose group-rare, target encoding, or attributes instead of raw one-hot.",
            "Fit encodings in-fold; define an explicit unknown bucket.",
        ),
        "next": [
            "Try session.encode(method='infrequent', min_frequency=...) before one-hot.",
            "For supervised target encoding, keep it fold-local via PreprocessRecipe.",
        ],
    },
    "01.7": {
        "beginner": "Declare allowed min/max per numeric column; turn sentinel codes into missing.",
        "why": "Sentinels like -999 look like real magnitudes and distort scales and trees’ splits.",
        "levels": _levels(
            "Scan for negatives and impossible ranges.",
            "Convert domain sentinels to null before imputation.",
            "Assert ranges on every new extract.",
        ),
        "next": [
            "Replace sentinels upstream, then re-run session.eda.",
            "Document unit systems (currency, minutes, meters).",
        ],
    },
    "01.8": {
        "beginner": "Clean text categories (trim, case-fold) before counting levels.",
        "why": "“Yes” vs “yes ” looks like two levels and fragments rare-category stats.",
        "levels": _levels(
            "Strip whitespace and unify case before nunique counts.",
            "Normalize punctuation variants that mean the same level.",
            "Apply the same transform at prediction time.",
        ),
        "next": [
            "Normalize strings before session.encode.",
            "Re-profile cardinality after cleaning.",
        ],
    },
    "01.9": {
        "beginner": "For every join that built this table, confirm row counts matched expectations.",
        "why": "Unexpected fan-out duplicates entities; unexpected drop deletes a subpopulation.",
        "levels": _levels(
            "Record expected cardinality (1:1, 1:N).",
            "Track match rates and unmatched keys.",
            "Fail the pipeline when match rate falls below a floor.",
        ),
        "next": [
            "Log pre/post join counts in the extract job.",
            "Investigate fan-out before modeling.",
        ],
    },
    "01.10": {
        "beginner": "Test contradictions between columns (end before start, parts not summing to total).",
        "why": "Cross-field errors survive univariate screens and poison relationships.",
        "levels": _levels(
            "List known logical constraints from the domain.",
            "Run assertions on every extract.",
            "Decide repair vs drop vs flag for each violation class.",
        ),
        "next": [
            "Add domain assertions next to ingest.",
            "Keep a violation rate KPI in monitoring.",
        ],
    },
    "01.11": {
        "beginner": "Parse dates with an explicit format and timezone; check that the span looks real.",
        "why": "Misparsed dates scramble temporal splits and window features.",
        "levels": _levels(
            "Identify datetime columns and their observed span.",
            "Parse with format + timezone; refuse silent coercion.",
            "Validate against business calendars and coverage gaps.",
        ),
        "next": [
            "Assign time role when rows are ordered by event time.",
            "Prefer chronological split helpers over random split.",
        ],
    },
    "01.12": {
        "beginner": "Know recording precision and whether values pile up at a cap.",
        "why": "Heaping and caps look like distribution shape but are measurement artifacts.",
        "levels": _levels(
            "Look for spikes at round numbers and policy caps.",
            "Separate true tails from censoring.",
            "Model censored outcomes explicitly when needed.",
        ),
        "next": [
            "Inspect histograms for edge masses before transforms.",
            "Document caps in the assumptions ledger.",
        ],
    },
    "02.1": {
        "beginner": "Read each numeric column as a distribution (quartiles + histogram), not only a mean.",
        "why": "Means hide multimodality, zeros, and heavy tails that change model choice.",
        "levels": _levels(
            "Check min/median/max and missingness per numeric column.",
            "Name the shape (skewed, zero-inflated, multimodal).",
            "Tie shape to transform and model-family decisions.",
        ),
        "next": [
            "Open the Features / Visuals boards for histograms.",
            "Decide transforms only after naming the shape.",
        ],
    },
    "02.2": {
        "beginner": "For each skewed column, decide to transform or not — and why.",
        "why": "Linear models and distance methods feel skew; trees often do not need the same transform.",
        "levels": _levels(
            "Flag |skew| > 1 as a review cue (not a law).",
            "Choose log/Yeo-Johnson only if the model family benefits.",
            "Fit transforms in-fold so test rows do not set parameters.",
        ),
        "next": [
            "Compare model families before committing to a transform.",
            "Keep the decision in your run notes (not persisted as a gate mark).",
        ],
    },
    "02.3": {
        "beginner": "Remove columns that are just re-expressions of other columns.",
        "why": "Near-duplicate features share credit, confuse importance, and inflate VIF.",
        "levels": _levels(
            "Screen absolute Pearson pairs near 1.0.",
            "Keep the measured member; drop the derived twin.",
            "Recompute associations after drops.",
        ),
        "next": [
            "session.drop_columns on confirmed derived twins.",
            "Re-check VIF and importance after pruning.",
        ],
    },
    "02.4": {
        "beginner": "Features should be independent enough that coefficients remain readable.",
        "why": "High VIF means coefficients trade credit and signs become unstable.",
        "levels": _levels(
            "Read VIF against the session threshold (often 5).",
            "Remove one member of a collinear set at a time.",
            "Prefer domain choice over automated deletion alone.",
        ),
        "next": [
            "Drop or combine collinear features, then re-run multivariate EDA.",
            "For trees, treat VIF as interpretation risk more than fit risk.",
        ],
    },
    "02.5": {
        "beginner": "Check for curves and reversals vs the target, not only straight-line correlation.",
        "why": "Pearson/MI summaries miss U-shapes and saturations that binning or nonlinear models catch.",
        "levels": _levels(
            "Plot target mean by feature decile for top candidates.",
            "Name monotone vs saturating vs reversing shapes.",
            "Choose binning / splines / trees accordingly.",
        ),
        "next": [
            "Use session plots / bivariate boards for top MI features.",
            "Avoid assuming linearity from a single r value.",
        ],
    },
    "02.6": {
        "beginner": "Re-check headline relationships inside subgroups — they can reverse.",
        "why": "Simpson-style pooling hides confounding and unfair slice failures.",
        "levels": _levels(
            "Pick at least one plausible stratifier.",
            "Recompute the relationship within levels.",
            "Escalate reversals to causal / product review.",
        ),
        "next": [
            "Slice key metrics by major categorical columns.",
            "Do not treat pooled association as policy truth.",
        ],
    },
    "02.7": {
        "beginner": "After encoding, you need enough rows per feature to learn rather than memorize.",
        "why": "Wide sparse matrices overfit; holdout scores look lucky then collapse.",
        "levels": _levels(
            "Compute rows / eligible features before encoding.",
            "Aim roughly ≥10 rows per feature after encoding as a starting cue.",
            "Reduce width via dropping redundancy and coarsening categories first.",
        ),
        "next": [
            "Drop id-like and constant columns before encoding.",
            "Prefer infrequent encoding over raw one-hot for high cardinality.",
        ],
    },
    "02.8": {
        "beginner": "Record whether features need scaling for the chosen model family.",
        "why": "Distance and regularized linear models need comparable scales; trees usually do not.",
        "levels": _levels(
            "Compare numeric ranges across features.",
            "Pick standard/minmax/robust — or document “no scaler”.",
            "Fit scalers in-fold after the split.",
        ),
        "next": [
            "session.scale(method='standard') after impute/encode when needed.",
            "Skip scaling for pure tree pipelines if that is the recorded decision.",
        ],
    },
    "03.1": {
        "beginner": "Split on the structure the data has (time/groups), not blindly at random.",
        "why": "Random splits train on the future or leak the same entity into both sides.",
        "levels": _levels(
            "Detect time or repeated-entity structure.",
            "Choose random, stratified, group, or chronological split accordingly.",
            "Touch the final test partition once.",
        ),
        "next": [
            "Use session.split(...) with the matching strategy.",
            "For time-ordered rows, prefer chronological helpers / time role.",
        ],
    },
    "03.2": {
        "beginner": "Keep class balance (or binned target shares) stable across splits.",
        "why": "Unstratified splits starve rare classes in tiny test folds and distort metrics.",
        "levels": _levels(
            "Measure class counts before splitting.",
            "Enable stratify=True for classification.",
            "Verify post-split shares match the plan.",
        ),
        "next": [
            "session.split(test_size=..., stratify=True, random_state=...).",
            "For regression, consider binning the target only for stratification.",
        ],
    },
    "03.3": {
        "beginner": "Fit imputers, encoders, and scalers after the split — never on the full frame first.",
        "why": "Full-frame fits peek at holdout rows and inflate reported scores.",
        "levels": _levels(
            "Call mutate steps only on the Session after split.",
            "Inside CV, use PreprocessRecipe so each fold refits.",
            "Treat descriptive EDA on all analysis rows as triage, not as fitted prep.",
        ),
        "next": [
            "Order: set_roles → split → impute/encode/scale → fit.",
            "Read guides/leakage-cv-recipes.md before nested search.",
        ],
    },
    "03.4": {
        "beginner": "Every feature must be knowable at prediction time — no post-outcome values.",
        "why": "Leakage invents accuracy that vanishes the moment you deploy.",
        "levels": _levels(
            "Review id-like and target-derived columns.",
            "Assign ignore/id roles to non-features.",
            "Verify known-at time per column family.",
        ),
        "next": [
            "Keep identifiers in id role, not feature.",
            "Drop columns created after the outcome is known.",
        ],
    },
    "03.5": {
        "beginner": "When time exists, splits and window features must look only backwards.",
        "why": "Forward-looking windows are leakage dressed as feature engineering.",
        "levels": _levels(
            "Identify the time column and span.",
            "Use forward-chaining splits with a horizon gap.",
            "Close every window before the prediction moment.",
        ),
        "next": [
            "Assign time role and use chronological split patterns.",
            "Refuse random split for forecasting-style problems.",
        ],
    },
    "03.6": {
        "beginner": "If the same entity repeats across rows, keep that entity on one side of the split.",
        "why": "Entity leakage makes test scores look like memorization of the customer, not generalization.",
        "levels": _levels(
            "Name the group/entity column.",
            "Use group-aware split / CV.",
            "Or record a justified independence assumption.",
        ),
        "next": [
            "session.split(..., groups=...) when a group role exists.",
            "Audit near-unique id columns that should not be features.",
        ],
    },
    "03.7": {
        "beginner": "Rows used to pick the model must differ from rows used to report the final score.",
        "why": "Tuning on the reported test set overfits the leaderboard.",
        "levels": _levels(
            "Prefer nested CV or a three-way split.",
            "Report the outer-fold / final holdout number only once.",
            "Keep Session test untouched during search.",
        ),
        "next": [
            "Use session.nested_cv_score or an explicit validation partition.",
            "Do not call evaluate on test while searching hyperparameters.",
        ],
    },
    "03.8": {
        "beginner": "Ask whether the sample can detect an effect small enough to matter.",
        "why": "Tiny samples make metric swings look like model wins.",
        "levels": _levels(
            "State the smallest actionable lift.",
            "Compare it to rough sampling uncertainty.",
            "Collect more data or simplify the claim if intervals are wider than the effect.",
        ),
        "next": [
            "Write the actionable effect size before model bake-offs.",
            "Prefer uncertainty intervals on headline metrics.",
        ],
    },
    "03.9": {
        "beginner": "Many screened statistics need a chance correction before you trust the winners.",
        "why": "Fishing across dozens of pairs manufactures spurious “insights”.",
        "levels": _levels(
            "Count how many associations were screened.",
            "Apply FDR/Bonferroni on reported p-values when used.",
            "Confirm survivors out of sample.",
        ),
        "next": [
            "Treat EDA association boards as hypotheses, not conclusions.",
            "Pre-register the primary metric for modeling.",
        ],
    },
    "03.10": {
        "beginner": "Someone else should be able to re-run and get the same numbers.",
        "why": "Unpinned seeds, library versions, or extracts make results non-comparable.",
        "levels": _levels(
            "Set random_state on split and stochastic steps.",
            "Pin package versions and the extract id.",
            "Sensitivity-check across a few seeds.",
        ),
        "next": [
            "Pass random_state into session.split and search APIs.",
            "Checkpoint the Session when results matter.",
        ],
    },
    "03.11": {
        "beginner": "Before blaming the world for drift, rule out split and pipeline artifacts.",
        "why": "False drift alarms waste retrain cycles; real drift needs a named response.",
        "levels": _levels(
            "Read which columns crossed thresholds.",
            "Check split leakage and preprocessing differences first.",
            "Name population vs ingestion vs concept shift.",
        ),
        "next": [
            "Inspect validation.drift findings on the cockpit.",
            "Compare train vs test distributions for flagged columns.",
        ],
    },
    "03.12": {
        "beginner": "Explain each outlier as error, rare truth, subgroup, or sentinel — not “delete all”.",
        "why": "Blind deletion removes rare but real cases; keeping sentinels poisons fits.",
        "levels": _levels(
            "Review univariate fence rates and multivariate flags.",
            "Classify flags with domain rules.",
            "Choose winsorize / drop / keep / separate model per class.",
        ),
        "next": [
            "Use session outlier tools only after classification of flags.",
            "Document contamination settings when anomaly detectors run.",
        ],
    },
    "04.1": {
        "beginner": "Write the scoring metric down before fitting the first model.",
        "why": "Picking a metric after seeing leaderboards invites self-serving choices.",
        "levels": _levels(
            "Match metric to task and decision costs.",
            "Name one headline metric plus diagnostics.",
            "Freeze population and threshold assumptions with the metric.",
        ),
        "next": [
            "Pass the metric explicitly into evaluate / search APIs.",
            "Keep a diagnostic suite (calibration, slices) beside the headline.",
        ],
    },
    "04.2": {
        "beginner": "Score the dumbest predictor and today’s process before celebrating a model.",
        "why": "Beating a weak baseline is required; beating the incumbent is what creates value.",
        "levels": _levels(
            "Majority class / median predictors are the floor.",
            "Score them on the same rows and metric.",
            "Compare against the production process when one exists.",
        ),
        "next": [
            "Use session.evaluate baselines for the task.",
            "Record the baseline number next to every model claim.",
        ],
    },
    "04.3": {
        "beginner": "With imbalance, avoid metrics a majority-class guess can win.",
        "why": "Accuracy looks high while the rare class is ignored.",
        "levels": _levels(
            "Report class shares.",
            "Prefer ranking metrics plus precision/recall at a stated threshold.",
            "Prefer class weights over reckless resampling unless justified.",
        ),
        "next": [
            "Set scoring to ROC-AUC / average precision / F1 as appropriate.",
            "Inspect confusion matrices at the operating threshold.",
        ],
    },
    "04.4": {
        "beginner": "Choose the cut-off from costs of false alarms vs misses, not from 0.5 by habit.",
        "why": "Default 0.5 assumes equal costs and a balanced base rate — often false.",
        "levels": _levels(
            "Estimate relative costs C_FP and C_FN.",
            "Tune threshold on validation; freeze for test.",
            "Revisit when base rates shift.",
        ),
        "next": [
            "Use session threshold tools after a probabilistic model exists.",
            "Document the frozen cut in the handoff.",
        ],
    },
    "04.5": {
        "beginner": "Numbers people act on need an uncertainty range, not only a point.",
        "why": "Point estimates hide whether a “win” is noise.",
        "levels": _levels(
            "Report intervals at the independence level of the rows.",
            "Round to the precision the interval supports.",
            "Avoid overclaiming tiny lifts inside wide intervals.",
        ),
        "next": [
            "Bootstrap or use CV fold spreads for headline metrics.",
            "Show intervals on slice metrics too.",
        ],
    },
    "04.6": {
        "beginner": "Report performance per segment, not only one overall figure.",
        "why": "Overall averages hide failing subpopulations.",
        "levels": _levels(
            "Predefine slices with enough support.",
            "Report metrics + intervals per slice.",
            "Name a response when a material gap appears.",
        ),
        "next": [
            "Use error-slice / evaluation partition tools after fit.",
            "Include fairness-sensitive segments when applicable.",
        ],
    },
    "04.7": {
        "beginner": "If scores are used as probabilities, check them against observed rates.",
        "why": "Uncalibrated probabilities mis-rank expected value and break thresholds.",
        "levels": _levels(
            "Decide whether ranking alone is enough.",
            "If probabilities matter, plot reliability on held-out data.",
            "Recalibrate with a method fitted only on validation rows.",
        ),
        "next": [
            "Use session calibration diagnostics after fit.",
            "Mark the product decision locally if ranking suffices (not saved).",
        ],
    },
    "05.1": {
        "beginner": "Name the importance method, run it on held-out rows, and show variability.",
        "why": "Train-set importance and single runs invent confident stories from noise.",
        "levels": _levels(
            "Prefer permutation importance on holdout.",
            "Repeat and show spread.",
            "Note redundant groups that share credit.",
        ),
        "next": [
            "session.permutation_importance (or equivalent) on held-out rows.",
            "Cross-check with correlated-pair lists from EDA.",
        ],
    },
    "05.2": {
        "beginner": "Draw effect curves only where data actually exists.",
        "why": "Extrapolated ICE/PDP outside support invents policy advice.",
        "levels": _levels(
            "Clip to empirical percentiles.",
            "Show density under the curve.",
            "Be careful with skewed features that thin the tails.",
        ),
        "next": [
            "Request partial-dependence / ICE within observed ranges.",
            "Annotate sparse regions as unsupported.",
        ],
    },
    "05.3": {
        "beginner": "Learn whether more data or better features is the lever before spending.",
        "why": "Hyper-parameter search cannot fix a variance-dominated regime.",
        "levels": _levels(
            "Draw a learning curve with fold error bars.",
            "Read high variance vs high bias patterns.",
            "Search hyper-parameters only after the curve justifies it.",
        ),
        "next": [
            "session.learning_curve (or nested diagnostics) before large HPO budgets.",
            "Invest in features/data when variance dominates.",
        ],
    },
    "05.4": {
        "beginner": "State findings as associations unless you have a causal design.",
        "why": "Observational correlations are not permission to intervene.",
        "levels": _levels(
            "Use association language in reports.",
            "List unmeasured confounders.",
            "Escalate to causal methods only with identification assumptions.",
        ),
        "next": [
            "Keep EDA MI/correlation claims descriptive.",
            "Use session.causal tools only when the design supports them.",
        ],
    },
    "05.5": {
        "beginner": "Ship a written handoff: assumptions, thresholds, owners, monitoring.",
        "why": "Models without owners and review dates quietly rot in production.",
        "levels": _levels(
            "Write assumptions and chosen thresholds.",
            "Name monitoring owners and review dates.",
            "Pin the extract and decision log.",
        ),
        "next": [
            "Export briefing PDF / offline HTML for reviewers.",
            "Attach owners in your external runbook (marks here stay local).",
        ],
    },
}


def _calculation_for(gate_id: str, ctx: dict[str, Any], status: str) -> dict[str, Any] | None:
    rows = max(1, int(ctx.get("rows") or 1))
    eligible = max(1, int(ctx.get("eligible") or 1))
    target = ctx.get("target") if isinstance(ctx.get("target"), dict) else None

    if gate_id == "01.2":
        missing = ctx.get("missing") or []
        cells = int(ctx.get("missingCells") or 0)
        if cells <= 0:
            return _calc(
                label="Missing cell rate",
                formula="missing_cells / (rows × columns)",
                inputs={"missing_cells": 0, "rows": rows, "columns": ctx.get("colCount")},
                result="0%",
                reading="No imputation decision is forced by gaps in this extract.",
            )
        worst = missing[0] if missing else {"name": "?", "missingRate": 0.0}
        return _calc(
            label="Worst-column missing rate",
            formula="null_count(column) / rows",
            inputs={
                "column": worst.get("name"),
                "missing_rate": float(worst.get("missingRate") or 0),
                "missing_cells": cells,
                "gappy_columns": len(missing),
            },
            result=_fmt_pct(float(worst.get("missingRate") or 0)),
            reading=(
                f"{worst.get('name')} is the densest gap at "
                f"{_fmt_pct(float(worst.get('missingRate') or 0))}; "
                "choose an in-fold strategy for each gappy column."
            ),
        )

    if gate_id == "01.6":
        high = ctx.get("highCard") or []
        if not high:
            return None
        added = sum(int(x.get("distinct") or 0) for x in high)
        return _calc(
            label="One-hot width estimate",
            formula="sum(n_unique) over high-cardinality categoricals",
            inputs={"columns": len(high), "estimated_dummy_columns": added},
            result=_fmt_n(added),
            reading="That is the approximate width added if each level becomes its own column.",
        )

    if gate_id == "02.2":
        skewed = ctx.get("skewed") or []
        if not skewed:
            return _calc(
                label="Skew screen",
                formula="flag when |skewness| > 1",
                inputs={"threshold": 1.0, "flagged": 0},
                result="0 columns",
                reading="No numeric column crossed the |skew| > 1 review cue in this profile.",
            )
        top = skewed[0]
        return _calc(
            label="Skew screen",
            formula="flag when |skewness| > 1",
            inputs={
                "threshold": 1.0,
                "flagged": len(skewed),
                "top_column": top.get("name"),
                "top_skew": float(top.get("skew") or 0),
            },
            result=f"{len(skewed)} columns; lead {top.get('name')} = {float(top.get('skew') or 0):.2f}",
            reading="Treat the threshold as a review cue; decide transforms against the model family.",
        )

    if gate_id == "02.4":
        vif = ctx.get("vif") or []
        threshold = float(ctx.get("vifThreshold") or 5)
        if not vif:
            return None
        over = [v for v in vif if float(v.get("vif") or 0) >= threshold]
        lead = vif[0]
        return _calc(
            label="Variance inflation",
            formula="VIF_j = 1 / (1 − R²_j)",
            inputs={
                "threshold": threshold,
                "features_scored": len(vif),
                "above_threshold": len(over),
                "highest_feature": lead.get("name"),
                "highest_vif": float(lead.get("vif") or 0),
            },
            result=(
                f"{len(over)} above {threshold:.1f}; "
                f"max {lead.get('name')} = {float(lead.get('vif') or 0):.2f}"
            ),
            reading="Remove one collinear member at a time and recompute before reading coefficients.",
        )

    if gate_id == "02.7":
        ratio = rows / eligible
        return _calc(
            label="Rows per eligible feature",
            formula="analysis_rows / max(1, eligible_features)",
            inputs={"rows": rows, "eligible_features": eligible, "ratio": round(ratio, 2)},
            result=f"≈ {round(ratio)} rows / feature (before encoding)",
            reading=(
                "Clear cue when ratio ≥ 10 before encoding; after one-hot the ratio usually worsens."
                if status == "clear"
                else "Below ~10 rows/feature before encoding — reduce width before memorizing noise."
            ),
        )

    if gate_id == "02.8":
        spans: list[float] = []
        for item in ctx.get("profiledNumeric") or []:
            try:
                span = abs(float(item["max"]) - float(item["min"]))
            except (TypeError, ValueError, KeyError):
                continue
            if span > 0:
                spans.append(span)
        if not spans:
            return None
        factor = max(spans) / max(min(spans), 1e-12)
        return _calc(
            label="Numeric range factor",
            formula="max(span) / min(span) across profiled numeric columns",
            inputs={
                "min_span": min(spans),
                "max_span": max(spans),
                "factor": round(factor, 2),
                "numeric_columns": len(ctx.get("numeric") or []),
            },
            result=f"≈ {factor:,.1f}×",
            reading="Large factors push linear/distance models toward scaling; trees may still skip it.",
        )

    if gate_id == "03.2" and target and target.get("classes"):
        small = min(int(k["count"]) for k in target["classes"])
        return _calc(
            label="Unstratified rare-class test count",
            formula="min_class_count × test_size",
            inputs={"min_class_count": small, "test_size": 0.2, "expected_in_test": round(small * 0.2)},
            result=f"≈ {_fmt_n(round(small * 0.2))} rare-class rows in a 20% test slice",
            reading="If that count is tiny, stratification is not optional.",
        )

    if gate_id == "03.8":
        se = 1.96 * math.sqrt(0.25 / rows) * 100
        return _calc(
            label="Rough proportion margin (Wald 95%)",
            formula="±1.96 × √(0.25 / n) × 100  (worst-case p=0.5)",
            inputs={"n": rows, "z": 1.96, "p_worst": 0.5},
            result=f"±{se:.1f} percentage points before splitting",
            reading="Compare this width to the smallest lift you would act on.",
        )

    if gate_id == "03.9":
        tests = len(ctx.get("corrPairs") or []) + len(ctx.get("mi") or [])
        return _calc(
            label="Screening multiplicity",
            formula="n_corr_pairs + n_mi_estimates",
            inputs={
                "corr_pairs": len(ctx.get("corrPairs") or []),
                "mi_estimates": len(ctx.get("mi") or []),
                "tests": tests,
                "columns": ctx.get("colCount"),
            },
            result=f"{tests} screening statistics",
            reading="More screens ⇒ stronger need for FDR control and out-of-sample confirmation.",
        )

    if gate_id == "04.2" and target:
        if target.get("task") == "regression" and target.get("stats"):
            med = target["stats"].get("median")
            return _calc(
                label="Median baseline (regression floor)",
                formula="predict ŷ = median(y_train) for every row",
                inputs={"target": target.get("name"), "median": med},
                result=str(med),
                reading="Any serious model must beat this constant predictor on the same metric.",
            )
        if target.get("classes"):
            maj = max(int(k["count"]) for k in target["classes"]) / rows
            return _calc(
                label="Majority-class baseline accuracy",
                formula="max(class_count) / n",
                inputs={"target": target.get("name"), "majority_share": maj},
                result=_fmt_pct(maj),
                reading="Accuracy at or below this number means you have not beaten a constant guess.",
            )

    if gate_id == "04.3" and target and target.get("classes"):
        shares = {
            str(k["label"]): int(k["count"]) / rows for k in target["classes"]
        }
        return _calc(
            label="Class balance",
            formula="class_count / n",
            inputs={"target": target.get("name"), "shares": shares},
            result=", ".join(f"{lab} {_fmt_pct(rate)}" for lab, rate in shares.items()),
            reading="When shares are uneven, accuracy alone is usually the wrong headline.",
        )

    if gate_id == "04.4" and target and target.get("classes") and len(target["classes"]) == 2:
        pos = int(target["classes"][1]["count"]) / rows
        return _calc(
            label="Bayes-ish cost threshold sketch",
            formula="τ ≈ C_FP / (C_FP + C_FN)   (calibrated scores, equal utilities aside)",
            inputs={"positive_rate": pos, "default_threshold": 0.5},
            result=f"base rate {_fmt_pct(pos)}; default cut 0.5",
            reading="Replace equal costs with your real C_FP/C_FN and tune on validation.",
        )

    if gate_id == "05.3":
        ratio = rows / eligible
        return _calc(
            label="Capacity regime cue",
            formula="rows / eligible_features",
            inputs={"rows": rows, "eligible_features": eligible, "ratio": round(ratio, 2)},
            result=f"≈ {round(ratio)} rows / feature",
            reading=(
                "Below ~20 often means variance dominates — draw a learning curve before big HPO."
                if ratio < 20
                else "Wider data regime — still draw a learning curve before expensive search."
            ),
        )

    return None


def _worked_example_for(
    gate_id: str,
    ctx: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    header = _ingest_header(ctx)
    roles = _roles_snippet(ctx)
    target = _target_name(ctx)
    task = _task(ctx)
    feats = _feature_sample(ctx, 3)
    missing_cols = _names(ctx.get("missing") or [], 3)
    constants = _names(ctx.get("constants") or [], 3)
    high = _names(ctx.get("highCard") or [], 3)
    skewed = _names(ctx.get("skewed") or [], 3)
    id_like = _names(ctx.get("idLike") or [], 2)
    time_col = (ctx.get("timeCol") or {}).get("name") if ctx.get("timeCol") else None
    vif_over = [
        str(v.get("name"))
        for v in (ctx.get("vif") or [])
        if float(v.get("vif") or 0) >= float(ctx.get("vifThreshold") or 5)
    ][:3]

    common_change = [
        "Replace frame with your DataFrame or path.",
        "Edit role mappings to match your column names.",
        "Adjust test_size / random_state to your protocol.",
    ]
    common_flex = [
        "Engine choice (pandas/polars/duckdb) if installed.",
        "Whether plots are included in session.eda.",
        "Metric names passed to evaluate/search later.",
    ]

    if gate_id in {"00.1", "00.3", "00.5", "00.6", "01.9", "01.10", "05.5"}:
        code = (
            f"{header}"
            "session = (\n"
            "    Session.ingest(frame)\n"
            f"    .set_roles({roles})\n"
            "    .split(test_size=0.25, stratify="
            f"{'True' if task == 'classification' else 'False'}, random_state=0)\n"
            ")\n"
            "report = session.eda(include_plots=False, show=False)\n"
            "print(report.findings[:8])\n"
            "# Write the human answer (decision sentence / lineage / sensitivity inventory)\n"
            "# in your runbook. Do not expect the Gates UI to persist that judgment.\n"
        )
        return _example(
            summary=(
                f"Profile this extract ({_fmt_n(ctx.get('rows'))} analysis rows) and keep the "
                "human judgment in an external note — gate marks in the App stay tab-local."
            ),
            code=code,
            change_these=common_change
            + ["Swap the printed findings slice for the keys you care about."],
            flexible=common_flex
            + ["Export PDF/offline HTML for reviewers; marks still do not persist."],
            reading="EDA surfaces evidence; governance sentences live outside BuildML on purpose.",
        )

    if gate_id == "00.2":
        key = id_like[0] if id_like else "entity_id"
        code = (
            f"{header}"
            "session = Session.ingest(frame).set_roles({\n"
            f'    "{key}": "id",\n'
            + "".join(f'    "{f}": "feature",\n' for f in feats)
            + (f'    "{target}": "target",\n' if target else "")
            + "})\n"
            "report = session.eda(include_plots=False, show=False)\n"
            "quality = report.to_dict().get('quality', {})\n"
            "print('id_like', quality.get('id_like_columns'))\n"
            "print('duplicate_rows', quality.get('duplicate_row_count'))\n"
            f"# Assert uniqueness on `{key}` in your DataFrame before splitting.\n"
            f"assert frame['{key}'].is_unique  # adjust if your grain is composite\n"
        )
        return _example(
            summary=f"Treat `{key}` as the grain key and prove uniqueness before modeling.",
            code=code,
            change_these=common_change + [f"Replace `{key}` if your real key differs."],
            flexible=common_flex + ["Composite keys may need a concatenated uniqueness check."],
            reading="Id role keeps the key out of the feature matrix while preserving lineage.",
        )

    if gate_id == "00.4":
        if target:
            code = (
                f"{header}"
                "session = (\n"
                "    Session.ingest(frame)\n"
                f"    .set_roles({roles})\n"
                f"    .split(test_size=0.25, stratify={task == 'classification'}, random_state=0)\n"
                ")\n"
                f"# Target `{target}` is declared as {task}. Document the label rule externally:\n"
                "# positive definition, anchor time, horizon, censoring.\n"
                "report = session.eda(include_plots=False, show=False)\n"
                "print(report.to_dict().get('target'))\n"
            )
            summary = f"`{target}` is the {task} target in this session — write its construction rule down."
        else:
            code = (
                f"{header}"
                "session = Session.ingest(frame).set_roles({\n"
                + "".join(f'    "{f}": "feature",\n' for f in feats)
                + '    "label_column": "target",  # <-- declare your label\n'
                + "})\n"
                "print(session.roles)\n"
            )
            summary = "No target is declared yet; supervised gates stay blocked until you name one."
        return _example(
            summary=summary,
            code=code,
            change_these=common_change + ["Set the real label column name and task assumptions."],
            flexible=common_flex,
            reading="Target definition is a product rule, not something EDA can invent.",
        )

    if gate_id in {"01.1", "01.8", "01.12"}:
        code = (
            f"{header}"
            "session = Session.ingest(frame).set_roles("
            + roles.replace("\n", "\n    ")
            + ")\n"
            "report = session.eda(include_plots=False, show=False)\n"
            "overview = report.to_dict()['overview']\n"
            "print(overview.get('dtypes'))\n"
            "# Fix dtypes / string normalization on `frame` upstream, then re-ingest.\n"
        )
        return _example(
            summary="Inspect loaded dtypes from this report, then correct the frame before prep.",
            code=code,
            change_these=common_change + ["Apply explicit pandas/polars dtype parses at load."],
            flexible=common_flex,
            reading="Gates read storage as loaded; assertions belong in your ingest code.",
        )

    if gate_id in {"01.2", "01.3"}:
        cols = missing_cols or feats[:1]
        code = (
            f"{header}"
            "session = (\n"
            "    Session.ingest(frame)\n"
            f"    .set_roles({roles})\n"
            f"    .split(test_size=0.25, stratify={task == 'classification'}, random_state=0)\n"
            ")\n"
            "report = session.eda(include_plots=False, show=False)\n"
            "print(report.to_dict()['quality'].get('missing_rate_by_column'))\n"
            "# Fit imputation AFTER the split (and prefer PreprocessRecipe inside CV):\n"
            "session.impute(strategy='median'"
            + (f", columns={_quote_list(cols)}" if cols else "")
            + ")\n"
        )
        return _example(
            summary=(
                f"Missingness in this extract"
                + (f" (e.g. {_quote_list(cols)})" if cols else "")
                + "; impute only on post-split Session state."
            ),
            code=code,
            change_these=common_change
            + ["Pick strategy per column family (median/mode/constant).", "Add indicator columns if gaps are informative."],
            flexible=["PreprocessRecipe for fold-local impute inside cv_score / nested_cv_score."]
            + common_flex,
            reading="Descriptive missing rates are triage; fitted imputers must not see test rows.",
        )

    if gate_id == "01.4":
        code = (
            f"{header}"
            "session = Session.ingest(frame).set_roles("
            + roles
            + ")\n"
            "d = session.eda(include_plots=False, show=False).to_dict()['quality']\n"
            "print(d.get('duplicate_row_count'), d.get('id_like_columns'))\n"
            "frame = frame.drop_duplicates()\n"
            "session = Session.ingest(frame).set_roles("
            + roles
            + ").split(test_size=0.25, random_state=0)\n"
        )
        return _example(
            summary="Count duplicates from EDA quality, resolve grain, then re-ingest and split.",
            code=code,
            change_these=common_change + ["Use subset=[key] in drop_duplicates for entity grain."],
            flexible=common_flex,
            reading="Deduplicate before split so entities cannot straddle partitions.",
        )

    if gate_id == "01.5":
        drop_list = constants or ["constant_column"]
        code = (
            f"{header}"
            "session = Session.ingest(frame).set_roles("
            + roles
            + ")\n"
            "q = session.eda(include_plots=False, show=False).to_dict()['quality']\n"
            "print('constants', q.get('constant_columns'))\n"
            "print('near_constant', q.get('quasi_constant_columns'))\n"
            f"session.drop_columns({_quote_list(drop_list)})\n"
        )
        return _example(
            summary="Drop confirmed constants from this profile (near-constants need a deliberate keep/coarsen choice).",
            code=code,
            change_these=common_change + [f"Edit drop list; current sample uses {_quote_list(drop_list)}."],
            flexible=common_flex + ["Keep a near-constant only with a written reason."],
            reading="Constants never help supervised learners; near-constants are a domain call.",
        )

    if gate_id == "01.6":
        sample = high or (ctx.get("categorical") or ["category_col"])[:1]
        code = (
            f"{header}"
            "session = (\n"
            "    Session.ingest(frame)\n"
            f"    .set_roles({roles})\n"
            "    .split(test_size=0.25, stratify="
            f"{task == 'classification'}, random_state=0)\n"
            ")\n"
            f"# High-cardinality candidates in this extract: {_quote_list(list(sample))}\n"
            "session.encode(method='infrequent', min_frequency=0.05)\n"
            "# Or fold-local target encoding via PreprocessRecipe inside CV for supervised cases.\n"
        )
        return _example(
            summary="Encode rare levels instead of exploding one-hot width on this frame’s categoricals.",
            code=code,
            change_these=common_change
            + ["Tune min_frequency to your level distribution.", "Set an unseen-level policy for production."],
            flexible=["method='onehot' only after coarsening", "method='target' only in-fold"]
            + common_flex,
            reading="Unseen levels at predict time need an explicit bucket — plan it now.",
        )

    if gate_id == "01.7":
        negs = _names([n for n in (ctx.get("numeric") or []) if int(n.get("negatives") or 0) > 0], 3)
        col = negs[0] if negs else (feats[0] if feats else "amount")
        code = (
            f"{header}"
            f"# Example: convert domain sentinels on `{col}` before ingest.\n"
            f"frame['{col}'] = frame['{col}'].replace({{ -999: None, -1: None }})\n"
            "session = Session.ingest(frame).set_roles("
            + roles
            + ")\n"
            "print(session.eda(include_plots=False, show=False).to_dict()['univariate']['per_column'].get("
            f"'{col}'))\n"
        )
        return _example(
            summary=f"Assert ranges and neutralize sentinels (sample column `{col}` from this profile).",
            code=code,
            change_these=common_change + ["Replace sentinel map with your real codes.", f"Update `{col}`."],
            flexible=common_flex,
            reading="Sentinels must become missing before imputation/scaling.",
        )

    if gate_id == "01.11":
        tname = time_col or "event_ts"
        code = (
            f"{header}"
            "import pandas as pd  # if not already imported\n"
            f"frame['{tname}'] = pd.to_datetime(frame['{tname}'], format='%Y-%m-%d', utc=True)\n"
            "session = Session.ingest(frame).set_roles({\n"
            f'    "{tname}": "time",\n'
            + "".join(f'    "{f}": "feature",\n' for f in feats)
            + (f'    "{target}": "target",\n' if target else "")
            + "})\n"
            "# Prefer chronological partitioning when rows are event-ordered.\n"
            "session.split(test_size=0.25, random_state=0)  # replace with time-aware split helpers as needed\n"
        )
        return _example(
            summary=(
                f"Parse `{tname}` explicitly"
                + (" (detected in this profile)." if time_col else " (no datetime typed yet — example scaffold).")
            ),
            code=code,
            change_these=common_change + ["Set format/timezone to your source.", f"Rename `{tname}`."],
            flexible=common_flex + ["Forecast paths refuse random splits — use time_split patterns."],
            reading="Typed dates are necessary but not sufficient; splits must respect order.",
        )

    if gate_id in {"02.1", "02.2"}:
        focus = skewed[:2] or feats[:2]
        code = (
            f"{header}"
            "session = Session.ingest(frame).set_roles("
            + roles
            + ").split(test_size=0.25, random_state=0)\n"
            "uni = session.eda(include_plots=True, show=False).to_dict()['univariate']['per_column']\n"
            f"for col in {_quote_list(list(focus))}:\n"
            "    s = uni.get(col) or {}\n"
            "    print(col, 'skew', s.get('skew'), 'min', s.get('min'), 'max', s.get('max'))\n"
            "# Transform only if the chosen model family needs it; fit in-fold.\n"
        )
        return _example(
            summary=f"Read distribution summaries for {_quote_list(list(focus))} from this session’s univariate profile.",
            code=code,
            change_these=common_change + ["Add/remove columns in the print loop."],
            flexible=common_flex + ["include_plots=True when exploring shapes visually."],
            reading="Skew flags are review cues; model family decides the transform.",
        )

    if gate_id in {"02.3", "02.4", "02.5"}:
        drop_candidates = vif_over or feats[:1]
        code = (
            f"{header}"
            "session = (\n"
            "    Session.ingest(frame)\n"
            f"    .set_roles({roles})\n"
            "    .split(test_size=0.25, random_state=0)\n"
            ")\n"
            "rep = session.eda(include_plots=False, show=False).to_dict()\n"
            "print(rep.get('bivariate', {}).get('top_abs_pearson_pairs', [])[:5])\n"
            "print(rep.get('multivariate', {}).get('vif', [])[:5])\n"
            f"# If a derived twin or collinear member is confirmed, drop it:\n"
            f"session.drop_columns({_quote_list(drop_candidates)})\n"
        )
        return _example(
            summary="Inspect pairwise correlations and VIF from this report, then drop confirmed redundancy.",
            code=code,
            change_these=common_change
            + [f"Edit drop_columns list (suggested from this profile: {_quote_list(drop_candidates)})."],
            flexible=common_flex,
            reading="Re-run EDA after drops; importance and VIF both move.",
        )

    if gate_id in {"02.6", "04.6"}:
        cats = [str(x) for x in (ctx.get("categorical") or [])[:2]] or ["segment"]
        code = (
            f"{header}"
            "session = Session.ingest(frame).set_roles("
            + roles
            + ").split(test_size=0.25, random_state=0)\n"
            "# After fit/evaluate, request slice metrics for predefined segments, e.g.\n"
            f"# slices = {_quote_list(cats)}\n"
            "result = session.evaluate(partition='test')  # once a model exists\n"
            "print(result)\n"
            "# Also recompute key EDA relationships inside each segment during triage.\n"
        )
        return _example(
            summary=f"Plan segment checks using categoricals present here: {_quote_list(cats)}.",
            code=code,
            change_these=common_change + ["Choose segments with enough support.", "Swap evaluate kwargs for your API metric."],
            flexible=common_flex,
            reading="Pooled metrics hide reversals; slices are part of readiness, not polish.",
        )

    if gate_id in {"02.7", "05.3"}:
        code = (
            f"{header}"
            "session = (\n"
            "    Session.ingest(frame)\n"
            f"    .set_roles({roles})\n"
            "    .split(test_size=0.25, random_state=0)\n"
            ")\n"
            "overview = session.eda(include_plots=False, show=False).to_dict()['overview']\n"
            "rows = overview.get('analysis_rows')\n"
            "eligible = len(overview.get('eligible_feature_columns') or [])\n"
            "print('rows_per_feature', rows / max(1, eligible))\n"
            "# Reduce width before heavy one-hot: drop ids/constants, encode infrequent levels.\n"
            "session.drop_columns("
            + _quote_list(id_like or constants or ["id_like_column"])
            + ")\n"
            "# learning curve after a model path exists:\n"
            "# session.learning_curve()  # or project equivalent diagnostic\n"
        )
        return _example(
            summary=(
                f"This extract has {_fmt_n(ctx.get('rows'))} rows over {ctx.get('eligible')} eligible "
                "features — compute the ratio before encoding expands width."
            ),
            code=code,
            change_these=common_change + ["Update drop_columns to your idle columns."],
            flexible=common_flex,
            reading="If the ratio is low, spend on features/data before large HPO budgets.",
        )

    if gate_id == "02.8":
        code = (
            f"{header}"
            "session = (\n"
            "    Session.ingest(frame)\n"
            f"    .set_roles({roles})\n"
            "    .split(test_size=0.25, random_state=0)\n"
            ")\n"
            "session.impute(strategy='median')\n"
            "session.encode(method='onehot')\n"
            "session.scale(method='standard')  # skip if you recorded that trees need no scaler\n"
        )
        return _example(
            summary="Record a scaling decision for this frame’s numeric ranges; fit scalers after the split.",
            code=code,
            change_these=common_change
            + ["method='robust' / 'minmax' as needed.", "Omit scale() when the written decision says trees-only."],
            flexible=["Fold-local scaling via PreprocessRecipe inside CV."] + common_flex,
            reading="Scaling is a model-family decision documented beside the pipeline.",
        )

    if gate_id in {"03.1", "03.2", "03.5", "03.6"}:
        stratify = task == "classification"
        group_hint = "# session.split(..., groups='entity_id')  # when a group role exists\n"
        time_hint = (
            f"# time column detected: {time_col} — prefer chronological split helpers\n"
            if time_col
            else "# No time column typed; verify row independence before trusting a random split.\n"
        )
        code = (
            f"{header}"
            "session = Session.ingest(frame).set_roles("
            + roles
            + ")\n"
            f"{time_hint}"
            f"session.split(test_size=0.25, stratify={stratify}, random_state=0)\n"
            f"{group_hint}"
            "print(session.split_plan)\n"
        )
        return _example(
            summary="Choose a split that matches structure present (or absent) in this extract.",
            code=code,
            change_these=common_change
            + ["Enable groups= or time-aware splits when those roles exist.", f"stratify={stratify} follows this target task."],
            flexible=common_flex,
            reading="Wrong split structure is leakage; fix it before prep and fit.",
        )

    if gate_id in {"03.3", "03.4", "03.7", "03.10"}:
        code = (
            f"{header}"
            "session = (\n"
            "    Session.ingest(frame)\n"
            f"    .set_roles({roles})\n"
            "    .split(test_size=0.25, "
            f"stratify={task == 'classification'}, random_state=0)\n"
            ")\n"
            "# Safe order: split first, then fit prep on Session train only.\n"
            "session.impute(strategy='median').encode(method='onehot').scale(method='standard')\n"
            "# For model selection without touching Session test, prefer nested CV:\n"
            "# session.nested_cv_score(estimator=..., scoring='...', recipe=...)\n"
            "report = session.eda(include_plots=False, show=False)\n"
            "print('id_like', report.to_dict()['quality'].get('id_like_columns'))\n"
        )
        return _example(
            summary="Keep prep below the split, keep test untouched during selection, pin random_state.",
            code=code,
            change_these=common_change
            + ["Select scoring to match the pre-written metric.", "Put leakage-prone columns in id/ignore roles."],
            flexible=["PreprocessRecipe for fold-local prep", "Optuna/grid search APIs"] + common_flex,
            reading="EDA on all analysis rows is triage; fitted steps must respect partitions.",
        )

    if gate_id in {"03.8", "03.9", "03.11", "03.12"}:
        code = (
            f"{header}"
            "session = Session.ingest(frame).set_roles("
            + roles
            + ").split(test_size=0.25, random_state=0)\n"
            "rep = session.eda(include_plots=False, show=False).to_dict()\n"
            "print('drift', rep.get('drift'))\n"
            "print('outliers', rep.get('outliers', {}).get('multivariate'))\n"
            "print('corr_pairs', len(rep.get('bivariate', {}).get('top_abs_pearson_pairs') or []))\n"
            f"print('n', rep.get('overview', {{}}).get('analysis_rows'))\n"
        )
        return _example(
            summary="Pull drift, outlier, and association counts from this report before acting on flags.",
            code=code,
            change_these=common_change + ["Classify outlier flags with domain rules before deletion."],
            flexible=common_flex,
            reading="Flags are hypotheses: rule out split/pipeline artifacts first.",
        )

    if gate_id.startswith("04."):
        metric = "roc_auc" if task == "classification" else "neg_root_mean_squared_error"
        code = (
            f"{header}"
            "session = (\n"
            "    Session.ingest(frame)\n"
            f"    .set_roles({roles})\n"
            f"    .split(test_size=0.25, stratify={task == 'classification'}, random_state=0)\n"
            ")\n"
            "session.impute(strategy='median').encode(method='onehot').scale(method='standard')\n"
            "session.fit(model='hist_gradient_boosting')  # or another catalog estimator\n"
            f"evaluation = session.evaluate(partition='test', scoring='{metric}')\n"
            "print(evaluation)\n"
            "# Add threshold/calibration/slice diagnostics when the product uses probabilities.\n"
        )
        return _example(
            summary=(
                f"Evaluate with a pre-written metric on this "
                f"{task or 'supervised'} target"
                + (f" `{target}`" if target else "")
                + "."
            ),
            code=code,
            change_these=common_change
            + [f"Replace scoring='{metric}' with your frozen headline metric.", "Swap model= for your baseline family."],
            flexible=["Baselines inside evaluate", "Threshold tools", "Calibration plots"] + common_flex,
            reading="Metric, baseline, and threshold decisions belong in writing before leaderboard shopping.",
        )

    if gate_id.startswith("05."):
        code = (
            f"{header}"
            "session = (\n"
            "    Session.ingest(frame)\n"
            f"    .set_roles({roles})\n"
            "    .split(test_size=0.25, random_state=0)\n"
            ")\n"
            "session.impute(strategy='median').encode(method='onehot').scale(method='standard')\n"
            "session.fit(model='hist_gradient_boosting')\n"
            "imp = session.permutation_importance(partition='test', n_repeats=5, random_state=0)\n"
            "print(imp)\n"
            "# learning_curve / partial dependence: use the Session diagnostics your install exposes.\n"
            "# Keep causal language off unless identification assumptions are written down.\n"
        )
        return _example(
            summary="Interpretation tools run on held-out rows from this Session — not on train memorization.",
            code=code,
            change_these=common_change
            + ["Adjust n_repeats / partition.", "Clip effect plots to empirical percentiles."],
            flexible=common_flex + ["Export briefing PDF for the monitoring handoff."],
            reading="Importance without redundancy notes and holdout repeats is storytelling.",
        )

    # Fallback — still adaptive scaffolding, never demo-dataset narrative.
    code = (
        f"{header}"
        "session = Session.ingest(frame).set_roles("
        + roles
        + ").split(test_size=0.25, random_state=0)\n"
        "report = session.eda(include_plots=False, show=False)\n"
        f"print('gate', '{gate_id}', 'status', '{status}')\n"
        "print(report.findings[:5])\n"
    )
    return _example(
        summary=f"Re-profile the active extract and inspect findings tied to gate {gate_id}.",
        code=code,
        change_these=common_change,
        flexible=common_flex,
        reading="Use this session’s columns and counts — do not copy demo-dataset story text.",
    )


def _how_derived(gate_id: str, status: str, evidence: str, ctx: dict[str, Any]) -> str:
    task = _task(ctx)
    target = _target_name(ctx)
    bits = [
        f"Status `{status}` was derived only from the live EDA report for this App session.",
        evidence,
    ]
    if target:
        bits.append(f"Supervised context: target `{target}` ({task}).")
    else:
        bits.append("No target is declared on this Session, so supervised-only gates may be open or N/A.")
    if ctx.get("sampled"):
        bits.append(
            f"Sampling/warnings apply: {_fmt_n(ctx.get('rows'))} of "
            f"{_fmt_n(ctx.get('rowsTotal'))} rows were examined."
        )
    skipped = []
    if not ctx.get("vif"):
        skipped.append("VIF")
    if not ctx.get("hasCorr"):
        skipped.append("pairwise correlation")
    if not ctx.get("anomalies"):
        skipped.append("multivariate anomaly scoring")
    if skipped:
        bits.append(
            "Analyzers with no usable output here: "
            + ", ".join(skipped)
            + " (gates that depend on them become N/A or stay open with that disclosed)."
        )
    bits.append(f"Gate id {gate_id} does not store your judgment; only the computed status above is server-side.")
    return " ".join(bits)


def build_gate_teaching(
    *,
    gate_id: str,
    concept: str,
    status: str,
    evidence: str,
    closes: str,
    ctx: dict[str, Any],
    findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build adaptive teaching payload for one readiness gate."""
    curr = _CURRICULUM.get(gate_id) or {
        "beginner": "Settle this readiness question using the evidence from this extract.",
        "why": "Unresolved gates leave modeling decisions under-specified.",
        "levels": _levels(
            "Read the question in plain language.",
            "Connect the evidence line to a concrete check.",
            "Automate the check in your pipeline where possible.",
        ),
        "next": ["Re-run EDA after the next extract.", "Record human judgments outside the App."],
    }
    calculation = _calculation_for(gate_id, ctx, status)
    worked = _worked_example_for(gate_id, ctx, status)
    cited = findings or []

    return {
        "beginner": curr["beginner"],
        "why_before_modeling": curr["why"],
        "how_derived": _how_derived(gate_id, status, evidence, ctx),
        "levels": dict(curr["levels"]),
        "closes_when": closes,
        "calculation": calculation,
        "worked_example": worked,
        "status_meanings": dict(STATUS_MEANINGS),
        "status_meaning": STATUS_MEANINGS.get(status, status),
        "session_mark_note": STATUS_MEANINGS["session_mark"],
        "next_checks": list(curr.get("next") or []),
        "findings_cited": cited,
        "concept": concept,
        "adaptivity": {
            "task": _task(ctx),
            "target": _target_name(ctx),
            "rows": ctx.get("rows"),
            "columns": ctx.get("colCount"),
            "has_datetime": bool(ctx.get("timeCol")),
            "has_vif": bool(ctx.get("vif")),
            "has_corr": bool(ctx.get("hasCorr")),
            "missing_cells": ctx.get("missingCells"),
            "engine": (ctx.get("ds") or {}).get("engine"),
        },
        "completeness": {
            "has_beginner": True,
            "has_why": True,
            "has_levels": True,
            "has_how_derived": True,
            "has_worked_example": bool(worked.get("code")),
            "has_change_guidance": bool(worked.get("change_these")),
            "has_calculation": calculation is not None,
            "has_next_checks": bool(curr.get("next")),
            "adaptive": True,
            "persistence_claimed": False,
        },
    }


def enrich_gate_row(
    row: dict[str, Any],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Attach teaching payload onto a gates payload row (pure, returns new dict)."""
    teaching = build_gate_teaching(
        gate_id=str(row["id"]),
        concept=str(row.get("concept") or ""),
        status=str(row.get("status") or ""),
        evidence=str(row.get("evidence") or ""),
        closes=str(row.get("closes") or ""),
        ctx=ctx,
        findings=list(row.get("findings") or []),
    )
    enriched = dict(row)
    enriched["teaching"] = teaching
    return enriched
