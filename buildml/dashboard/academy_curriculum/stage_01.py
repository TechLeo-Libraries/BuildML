"""Stage 01 · Data quality - what the frame is before anything is fitted."""

from __future__ import annotations

from buildml.dashboard.academy_curriculum._helpers import (
    code_block,
    first_categorical,
    first_feature,
    first_missing,
    first_numeric,
    fmt_compact,
    fmt_n,
    fmt_pct,
    list_names,
    plural,
    quote_list,
    target_name,
)
from buildml.dashboard.academy_curriculum._types import LessonSpec, lesson


def lessons() -> list[LessonSpec]:
    out: list[LessonSpec] = []
    out.extend(_core())
    out.extend(_additions())
    return out


def _core() -> list[LessonSpec]:
    return [
        lesson(
            slug="column-roles",
            stage=1,
            title="column-roles",
            order=10,
            concept_key="column-roles",
            tags=("roles", "schema"),
            search_terms=("role", "feature", "target", "id", "ignore"),
            plain=(
                "A role says how a column participates in modeling - feature, target, identifier, ignore - "
                "and is independent of its dtype. An integer id and an integer count look the same in storage "
                "and mean opposite things.",
                "Roles left undeclared do not stay neutral: whatever is unmarked often ends up in the matrix.",
            ),
            technical=(
                "BuildML stores an explicit role map on the session dataset. Downstream impute/encode/scale/fit "
                "select feature columns by role, not by dtype alone. Id/group/time/weight/ignore follow separate rules.",
            ),
            why=(
                "Wrong roles silently leak identifiers or drop true predictors.",
                "Group and time roles change which split designs are valid.",
            ),
            formula="eligible_features = columns with role 'feature' (excluding id/target/ignore)",
            calculation=lambda ctx: (
                f"Near-unique / id-like: {list_names(ctx.get('idLike') or [])}. "
                f"Constants: {list_names(ctx.get('constants') or [])}. "
                f"Of {fmt_n(ctx.get('colCount'))} columns, {fmt_n(ctx.get('eligible'))} are marked eligible features; "
                f"target declared: {bool(ctx.get('has_target'))}."
            ),
            session_evidence=lambda ctx: (
                (
                    f"{list_names(ctx.get('idLike') or [])} "
                    f"{'is' if len(ctx.get('idLike') or []) == 1 else 'are'} near-unique across "
                    f"{fmt_n(ctx.get('rows'))} rows. "
                    if ctx.get("idLike")
                    else "No identifier-like columns were flagged. "
                )
                + (
                    f"Constants: {list_names(ctx.get('constants') or [])}. "
                    if ctx.get("constants")
                    else ""
                )
                + f"Eligible features: {fmt_n(ctx.get('eligible'))}."
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "session = Session.ingest(pd.read_csv(\"your_data.csv\"))  # <-- change path",
                "session = session.set_roles({",
                *(
                    [f'    "{n}": "id",' for n in (ctx.get("idLike") or [])[:2]]
                    or ['    # "<id_column>": "id",']
                ),
                *(
                    [f'    "{n}": "ignore",' for n in (ctx.get("constants") or [])[:2]]
                    or []
                ),
                f'    "{target_name(ctx)}": "target",  # <-- change',
                f'    "{first_feature(ctx)}": "feature",  # <-- add every predictor',
                "})",
                "",
                "# Confirm the contract before splitting",
                "print(session.dataset.roles)  # role map on the dataset",
                'session.learn("column-roles", level="beginner")',
            ),
            what_to_change=(
                "Map every column: feature / target / id / group / time / weight / ignore.",
                "Change target name to your label; mark true identifiers as id.",
                "Revisit roles after EDA if a 'feature' looks like an id or leakage carrier.",
            ),
            pitfalls=(
                "Assuming near-uniqueness always means identifier - continuous measurements are near-unique too.",
                "Leaving constants as features (zero variance breaks some scalers).",
                "Declaring roles after the split was drawn on the unfiltered frame.",
            ),
            decide=lambda ctx: (
                f"Write the role of all {fmt_n(ctx.get('colCount'))} columns explicitly and treat "
                "unclassified columns as ignore until reviewed."
            ),
            read_steps=lambda ctx: [
                f"Compare distinct count to row count (~0.95x{fmt_n(ctx.get('rows'))} is near-unique).",
                "Ask for each column: exists at prediction time? If not -> ignore.",
                f"Sanity-check eligible feature count ({fmt_n(ctx.get('eligible'))}) against frame width.",
            ],
        ),
        lesson(
            slug="dtypes-and-storage",
            stage=1,
            title="dtypes-and-storage",
            order=20,
            concept_key="feature-schema",
            tags=("dtype", "schema"),
            search_terms=("dtype", "type", "category", "datetime", "storage"),
            plain=(
                "A dtype is a storage decision, not a semantic one. Booleans stored as strings and dates stored "
                "as text load without complaint and then silently change every aggregate.",
            ),
            technical=(
                "Storage chooses available operations: categorical levels, datetime ordering, NaN in floats. "
                "CSV inference is a guess - declare intent at load time when you can.",
            ),
            why=(
                "Wrong dtypes invent categories, destroy padding in codes, and break temporal checks.",
                "Memory and encoder behaviour depend on storage.",
            ),
            formula=None,
            calculation=lambda ctx: _dtype_calc(ctx),
            session_evidence=lambda ctx: _dtype_calc(ctx),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "# Prefer explicit dtypes at load when you know them:",
                "frame = pd.read_csv(",
                "    \"your_data.csv\",  # <-- change",
                "    dtype={",
                f'        "{first_categorical(ctx)}": "string",  # <-- adjust',
                "    },",
                "    parse_dates=[],  # <-- list datetime columns",
                ")",
                "session = Session.ingest(frame)",
                "report = session.eda(include_plots=False, show=False)",
                "print(report.to_dict()[\"overview\"].get(\"dtypes\"))",
                "",
                "# Categories / codes that must stay strings should not be cast to int.",
                'session.learn("feature-schema", level="beginner")',
            ),
            what_to_change=(
                "Set dtype= and parse_dates= for your columns at load.",
                "Keep zero-padded codes as strings.",
            ),
            pitfalls=(
                "Letting the loader infer types, then treating inference as documentation.",
                "Numeric-looking identifiers losing padding as int64.",
                "Dates left as strings (lexical sort ≠ time order).",
            ),
            decide="Declare every dtype explicitly at load time and re-read the schema afterwards.",
            read_steps=(
                "Compare schema to five raw sample rows.",
                "Ask whether arithmetic on each numeric column is meaningful.",
                "Flag object columns that look boolean/numeric/date.",
            ),
        ),
        lesson(
            slug="missing-data",
            stage=1,
            title="missing-data",
            order=30,
            concept_key="missing-data",
            tags=("missing", "impute"),
            search_terms=("missing", "nan", "impute", "completeness"),
            plain=(
                "A missing cell is an absence of record, not a zero. Imputation fills gaps with a rule learned "
                "from rows you are allowed to learn from - it does not recover what was never written.",
            ),
            technical=(
                "Imputation is a fitted transform: fit fill values on train, apply everywhere. "
                "BuildML session.impute(...) learns on the training partition after split.",
            ),
            why=(
                "Full-frame medians leak test information into training.",
                "Gap rates change which rows a complete-case analysis actually uses.",
            ),
            formula="completeness = 1 - missing_cells / (rows x columns)",
            calculation=lambda ctx: (
                f"missing_cells = {fmt_n(ctx.get('missingCells'))}, "
                f"cells ~ {fmt_n(int(ctx.get('rows') or 0) * max(int(ctx.get('colCount') or 0), 1))}, "
                f"completeness ~ {fmt_pct(ctx.get('completeness') or 0)}. "
                + (
                    f"Worst column: {first_missing(ctx)} at "
                    f"{fmt_pct((ctx.get('missing') or [{}])[0].get('missingRate') or 0)}."
                    if ctx.get("missing")
                    else "No column-level gaps recorded."
                )
            ),
            session_evidence=lambda ctx: (
                f"No missing cells across {fmt_n(ctx.get('colCount'))} columns - still plan a strategy for future loads."
                if int(ctx.get("missingCells") or 0) == 0
                else (
                    f"{fmt_n(ctx.get('missingCells'))} cells missing across "
                    f"{fmt_n(len(ctx.get('missing') or []))} {plural(len(ctx.get('missing') or []), 'column')}; "
                    f"completeness {fmt_pct(ctx.get('completeness') or 0)}. "
                    f"Complete rows ~ {fmt_n(ctx.get('completeRows'))} of {fmt_n(ctx.get('rows'))}."
                )
            ),
            example_code=lambda ctx: _missing_example(ctx),
            what_to_change=(
                "Choose strategy per column (median / most_frequent / constant).",
                "Always split before impute so fills fit on train only.",
                "Consider missing indicators when absence may be informative.",
            ),
            pitfalls=lambda ctx: [
                "Computing the fill value over the full frame - leakage.",
                (
                    f"Treating a {fmt_pct((ctx.get('missing') or [{}])[0].get('missingRate') or 0)} gap as random "
                    "without checking mechanism."
                    if ctx.get("missing")
                    else "Assuming a complete extract means a complete source."
                ),
                (
                    f"Silent complete-case drops: only {fmt_n(ctx.get('completeRows'))} of "
                    f"{fmt_n(ctx.get('rows'))} rows are complete."
                    if int(ctx.get("completeRows") or 0) < int(ctx.get("rows") or 0)
                    else "Dropping incomplete rows without counting them first."
                ),
            ],
            decide=lambda ctx: (
                "Nothing to fill here - record the strategy you would use if the next extract is incomplete."
                if int(ctx.get("missingCells") or 0) == 0
                else "Pick one strategy per gappy column, fit inside the training fold, add indicators where gaps may be informative."
            ),
            read_steps=lambda ctx: [
                f"Read per-column rates before the total (worst: {first_missing(ctx)}).",
                f"Count complete rows: {fmt_n(ctx.get('completeRows'))} / {fmt_n(ctx.get('rows'))}.",
                "Ask what each gap means in the source system (not answered / not applicable / not yet / lost).",
            ],
        ),
        lesson(
            slug="missingness-mechanisms",
            stage=1,
            title="missingness-mechanisms",
            order=40,
            concept_key="missing-data",
            tags=("MCAR", "MAR", "MNAR"),
            search_terms=("MCAR", "MAR", "MNAR", "mechanism"),
            plain=(
                "Three mechanisms behave differently: missing completely at random loses precision only; "
                "missing at random can be repaired using other columns; missing not at random encodes the "
                "thing you care about, and no fill recovers it.",
            ),
            technical=(
                "You cannot prove the mechanism from rates alone. Test whether a missingness indicator "
                "predicts the target or correlates with covariates; keep indicators when association appears.",
            ),
            why=(
                "Imputation assumptions fail under MNAR.",
                "Discarding missingness indicators throws away signal.",
            ),
            formula="MCAR / MAR / MNAR - classified from evidence + domain knowledge, not from a single p-value",
            calculation=lambda ctx: (
                f"{fmt_n(len(ctx.get('missing') or []))} gappy "
                f"{plural(len(ctx.get('missing') or []), 'column')}; "
                "no automatic mechanism inference ran - rates are observed, reasons are not."
                if ctx.get("missing")
                else "Nothing missing in this extract, so there is no mechanism to classify."
            ),
            session_evidence=lambda ctx: (
                f"Led by {first_missing(ctx)} at "
                f"{fmt_pct((ctx.get('missing') or [{}])[0].get('missingRate') or 0)} - classify with domain owners."
                if ctx.get("missing")
                else "Complete extract in-session; still document expected mechanisms for production loads."
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "frame = pd.read_csv(\"your_data.csv\")  # <-- change",
                f"col = \"{first_missing(ctx)}\"  # <-- gappy column",
                "frame[f\"{col}__was_missing\"] = frame[col].isna().astype(int)",
                "session = Session.ingest(frame).set_roles({",
                f'    "{target_name(ctx)}": "target",',
                f'    "{first_feature(ctx)}": "feature",',
                f'    f"{{col}}__was_missing": "feature",  # keep the fact of absence',
                "})",
                "session = session.split(test_size=0.2, stratify=True, random_state=0)",
                "session = session.impute(strategy=\"median\")  # train-fitted",
                'session.explain("impute", moment="before")',
            ),
            what_to_change=(
                "Swap in your gappy columns and decide which indicators to keep.",
                "Interview source owners before modeling MNAR-looking gaps.",
            ),
            pitfalls=(
                "Imputing first and asking about mechanism afterwards.",
                "Filling 'not recorded' with the mode (invents a positive answer).",
                "Discarding missing indicators as noise when absence is predictive.",
            ),
            decide="Classify each gappy column MCAR/MAR/MNAR on available evidence and keep indicators when not MCAR.",
            read_steps=(
                "Compare target rate for missing vs present rows.",
                "Compare other feature distributions between missing/present groups.",
                "Ask the source owner about the worst column before modeling it.",
            ),
        ),
        lesson(
            slug="duplicate-records",
            stage=1,
            title="duplicate-records",
            order=50,
            concept_key="diagnostic-uncertainty",
            tags=("duplicates", "grain"),
            search_terms=("duplicate", "dedupe", "grain"),
            plain=(
                "Duplicates multiply some entities in training and evaluation. Exact copies are easy; "
                "same key with different payloads means your grain is wrong.",
            ),
            technical=(
                "Compare rows, distinct full rows, and distinct keys. Join fan-out creates duplicates "
                "that look legitimate until counts explode.",
            ),
            why=(
                "Duplicated entities overweight folds and leak across splits if not grouped.",
            ),
            formula="exact_dupes = rows - distinct(full_row); key_dupes = rows - distinct(key)",
            calculation=lambda ctx: (
                f"rows={fmt_n(ctx.get('rows'))}, exact duplicate rows="
                f"{fmt_n((ctx.get('duplicates') or {}).get('rows') or 0)}. "
                f"Id-like keys to test: {list_names(ctx.get('idLike') or [])}."
            ),
            session_evidence=lambda ctx: (
                f"Exact duplicate rows reported: {fmt_n((ctx.get('duplicates') or {}).get('rows') or 0)} "
                f"of {fmt_n(ctx.get('rows'))}."
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "frame = pd.read_csv(\"your_data.csv\")  # <-- change",
                "print(\"exact dupes:\", int(frame.duplicated().sum()))",
                f"key = \"{(ctx.get('idLike') or ['<entity_id>'])[0]}\"  # <-- grain key",
                "print(\"key dupes:\", int(frame.duplicated(subset=[key]).sum()))",
                "",
                "# De-duplicate to the intended grain BEFORE split",
                "frame = frame.drop_duplicates(subset=[key], keep=\"last\")  # <-- policy",
                "session = Session.ingest(frame)",
                "session = session.set_roles({key: \"id\", "
                f"\"{target_name(ctx)}\": \"target\"}})",
            ),
            what_to_change=(
                "Choose the grain key and keep policy (first/last/aggregate).",
                "De-duplicate before split.",
            ),
            pitfalls=(
                "Dropping duplicates without stating the key.",
                "Ignoring join-induced row multiplication.",
            ),
            decide="State the grain, enforce uniqueness on the entity key, de-duplicate before splitting.",
            read_steps=(
                "Count exact duplicates, then key duplicates.",
                "Inspect two colliding rows side by side.",
                "Check whether duplicates cluster by source/time.",
            ),
        ),
        lesson(
            slug="constant-and-near-constant",
            stage=1,
            title="constant-and-near-constant",
            order=60,
            concept_key="column-roles",
            tags=("constant", "variance"),
            search_terms=("constant", "near-constant", "quasi", "variance"),
            plain=(
                "A constant column teaches nothing. A near-constant column teaches a handful of rare rows "
                "and often fails to appear in some CV folds.",
            ),
            technical=(
                "Share of the mode near 1.0 ⇒ little information. Prefer ignore/reframe over hoping "
                "the model 'figures it out'.",
            ),
            why=(
                "Zero-variance breaks scaling; ultra-rare flags overfit.",
            ),
            formula="mode_rate = count(mode) / n; near-constant if mode_rate >= 0.99",
            calculation=lambda ctx: (
                f"Constants: {list_names(ctx.get('constants') or [])}. "
                f"Near-constants: {list_names(ctx.get('nearConstant') or [])}."
            ),
            session_evidence=lambda ctx: (
                f"{fmt_n(len(ctx.get('constants') or []))} constant and "
                f"{fmt_n(len(ctx.get('nearConstant') or []))} near-constant columns flagged."
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "session = Session.ingest(pd.read_csv(\"your_data.csv\"))  # <-- change",
                "roles = {",
                f'    "{target_name(ctx)}": "target",',
                f'    "{first_feature(ctx)}": "feature",',
                "}",
                "for col in "
                + repr(list(ctx.get("constants") or [])[:5] or ["<constant_col>"])
                + ":",
                "    roles[col] = \"ignore\"",
                "session = session.set_roles(roles)",
                'session.learn("column-roles", level="beginner")',
            ),
            what_to_change=(
                "Ignore true constants; reframe rare flags into coarser indicators if needed.",
            ),
            pitfalls=(
                "Keeping constants 'just in case'.",
                "Interpreting a rare flag fitted on three rows.",
            ),
            decide="Drop or ignore constants; decide keep/reframe/ignore for each near-constant.",
            read_steps=(
                "List mode rates for low-cardinality columns.",
                "Check whether minority levels appear in every intended fold.",
            ),
        ),
        lesson(
            slug="high-cardinality",
            stage=1,
            title="high-cardinality",
            order=70,
            concept_key="categorical-encoding",
            tags=("cardinality", "encoding"),
            search_terms=("cardinality", "one-hot", "target encoding", "rare levels"),
            plain=(
                "High-cardinality categoricals explode one-hot width or create rare levels that never "
                "repeat at score time. Encoding strategy is part of validation, not a cosmetic choice.",
            ),
            technical=(
                "BuildML encode(method='onehot'|'ordinal'|'infrequent'|'target') learns vocabularies on train. "
                "Target encoding is fold-aware when configured; still leakage-sensitive if misused.",
            ),
            why=(
                "Huge vocabularies overfit and break production with unseen levels.",
            ),
            formula="onehot_width ~ sum(n_levels_c - 1) over one-hot columns",
            calculation=lambda ctx: (
                f"High-cardinality flags: "
                + (
                    "; ".join(
                        f"{h.get('name')} (n~{fmt_n(h.get('distinct'))})"
                        for h in (ctx.get("highCard") or [])[:4]
                    )
                    or "none"
                )
                + f". Rows={fmt_n(ctx.get('rows'))}."
            ),
            session_evidence=lambda ctx: (
                f"{fmt_n(len(ctx.get('highCard') or []))} high-cardinality "
                f"{plural(len(ctx.get('highCard') or []), 'column')}: "
                f"{list_names(ctx.get('highCard') or [])}."
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "session = (",
                "    Session.ingest(pd.read_csv(\"your_data.csv\"))  # <-- change",
                "    .set_roles({",
                f'        "{target_name(ctx)}": "target",',
                f'        "{_high_card_col(ctx)}": "feature",',
                "    })",
                "    .split(test_size=0.2, stratify=True, random_state=0)",
                ")",
                "",
                "# Rare-level bundling before one-hot, or target encode carefully:",
                "session = session.encode(method=\"infrequent\", min_frequency=0.05)  # <-- tune",
                "# session = session.encode(method=\"target\", n_folds=5)  # alternative",
                'session.explain("encode", moment="before")',
            ),
            what_to_change=(
                "Pick encode method per column family; tune min_frequency.",
                "Consider hashing / grouping for extreme cardinalities.",
            ),
            pitfalls=(
                "One-hotting a free-text-like id column.",
                "Fitting encoder vocabularies on the full frame.",
            ),
            decide="Choose an encoding that keeps unseen levels safe at score time; validate it in-fold.",
            read_steps=(
                "Compare n_unique to rows for each categorical.",
                "Estimate one-hot width before encoding.",
                "Plan handling for levels unseen at prediction time.",
            ),
        ),
        lesson(
            slug="categorical-encoding",
            stage=1,
            title="categorical-encoding",
            order=80,
            concept_key="categorical-encoding",
            tags=("encoding",),
            search_terms=("encode", "onehot", "ordinal", "category"),
            plain=(
                "Models need numbers. Encoding maps category labels to numbers without pretending the "
                "integers are quantities - unless you truly have an ordered scale.",
            ),
            technical=(
                "session.encode(method=...) is train-fitted. Ordinal encoding invents order; one-hot does not. "
                "Infrequent bundling controls sparsity.",
            ),
            why=(
                "Wrong encoding creates false order or unmanageable width.",
            ),
            formula=None,
            calculation=lambda ctx: (
                f"{fmt_n(len(ctx.get('categorical') or []))} categorical "
                f"{plural(len(ctx.get('categorical') or []), 'column')}: "
                f"{list_names(ctx.get('categorical') or [])}."
            ),
            session_evidence=lambda ctx: (
                f"Categoricals in frame: {list_names(ctx.get('categorical') or []) or 'none flagged'}."
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "session = (",
                "    Session.ingest(pd.read_csv(\"your_data.csv\"))",
                "    .set_roles({",
                f'        "{target_name(ctx)}": "target",',
                f'        "{first_categorical(ctx)}": "feature",',
                "    })",
                "    .split(test_size=0.2, stratify=True, random_state=0)",
                "    .impute(strategy=\"most_frequent\")",
                f'    .encode(method="onehot", columns=["{first_categorical(ctx)}"])  # <-- change',
                ")",
                'session.learn("categorical-encoding", level="beginner")',
            ),
            what_to_change=(
                "Select columns and method (onehot/ordinal/infrequent/target).",
                "Impute categoricals before encode when needed.",
            ),
            pitfalls=(
                "Ordinal-encoding nominal labels (invents 2>1>0).",
                "Encoding before split.",
            ),
            decide="Pick an encoding per column that matches whether order is real.",
            read_steps=(
                "Label each categorical nominal vs ordinal.",
                "Estimate matrix width after encoding.",
            ),
        ),
        lesson(
            slug="measurement-units-and-ranges",
            stage=1,
            title="measurement-units-and-ranges",
            order=90,
            concept_key="diagnostic-uncertainty",
            tags=("units", "ranges"),
            search_terms=("units", "range", "sentinel", "scale"),
            plain=(
                "Numbers without units are rumours. Sentinels (9999, -1) look like extremes; mixed units "
                "make distance models nonsense.",
            ),
            technical=(
                "Range checks catch impossible values; unit harmonisation belongs before scaling. "
                "EDA range flags are hypotheses for domain review.",
            ),
            why=(
                "Unit mix-ups dominate model error while looking like 'signal'.",
            ),
            formula=None,
            calculation=lambda ctx: _range_calc(ctx),
            session_evidence=lambda ctx: _range_calc(ctx),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "frame = pd.read_csv(\"your_data.csv\")  # <-- change",
                f"col = \"{first_numeric(ctx)}\"",
                "print(frame[col].describe(percentiles=[0.01, 0.5, 0.99]))",
                "# Harmonise units BEFORE session modeling, e.g. cents -> dollars",
                "# frame[col] = frame[col] / 100",
                "session = Session.ingest(frame)",
                "report = session.eda(include_plots=False, show=False)",
                "print(report.to_dict().get(\"univariate\", {}).get(\"per_column\", {}).get(col))",
            ),
            what_to_change=(
                "Document units per numeric column; convert before ingest.",
                "Replace sentinels with true missing values.",
            ),
            pitfalls=(
                "Scaling mixed units and calling the result standardised.",
                "Leaving sentinel codes as numeric extremes.",
            ),
            decide="Write units for every numeric feature and clear sentinels to NA before fitting.",
            read_steps=(
                "Read min/max against domain limits.",
                "Check 1% and 99% for sentinels.",
            ),
        ),
        lesson(
            slug="text-hygiene",
            stage=1,
            title="text-hygiene",
            order=100,
            concept_key="text-features",
            tags=("text",),
            search_terms=("text", "string", "tfidf", "hygiene"),
            plain=(
                "Text columns need cleaning and a vectoriser - and the vectoriser is a fitted transform "
                "that must learn vocabulary on train only.",
            ),
            technical=(
                "session.text_features(method='tfidf'|'count'|'hashing') builds numeric features. "
                "Hygiene (case, whitespace) must match at score time.",
            ),
            why=(
                "Train/serve text mismatch creates silent feature drift.",
            ),
            formula=None,
            calculation=lambda ctx: (
                "Text hygiene findings depend on analyzers; treat free-text object columns as candidates "
                f"among {fmt_n(ctx.get('colCount'))} columns. Categorical/string-like: "
                f"{list_names(ctx.get('categorical') or [])}."
            ),
            session_evidence=lambda ctx: (
                f"Review string-like columns for case variants and junk tokens before vectorising. "
                f"Categoricals listed: {list_names(ctx.get('categorical') or [])}."
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "frame = pd.read_csv(\"your_data.csv\")  # <-- change",
                "text_col = \"<text_column>\"  # <-- change",
                "frame[text_col] = frame[text_col].astype(\"string\").str.strip().str.lower()",
                "session = (",
                "    Session.ingest(frame)",
                "    .set_roles({",
                "        text_col: \"feature\",",
                f'        "{target_name(ctx)}": "target",',
                "    })",
                "    .split(test_size=0.2, random_state=0)",
                "    .text_features(columns=[text_col], method=\"tfidf\", max_features=128)",
                ")",
                'session.learn("text-features", level="beginner")',
            ),
            what_to_change=(
                "Set text column names and hygiene rules; keep them identical at serve time.",
                "Tune max_features / ngram_range.",
            ),
            pitfalls=(
                "Fitting TF-IDF on the full corpus including test rows.",
                "Cleaning train text and forgetting the same transform in production.",
            ),
            decide="Freeze a text hygiene + vectoriser plan that can run unchanged at score time.",
            read_steps=(
                "Sample raw strings for case/junk/PII.",
                "Decide categorical vs free text.",
            ),
        ),
    ]


def _additions() -> list[LessonSpec]:
    return [
        lesson(
            slug="join-integrity",
            stage=1,
            title="join-integrity",
            order=110,
            concept_key="feature-schema",
            tags=("joins",),
            search_terms=("join", "merge", "match rate", "cardinality"),
            plain=(
                "Joins invent rows and invent missings. A 1:1 claim that is actually 1:many silently "
                "duplicates entities and breaks split discipline.",
            ),
            technical=(
                "Validate join keys with match rate and cardinality checks before modeling. "
                "BuildML does not replace warehouse tests - do them in pandas/SQL, then ingest.",
            ),
            why=("Fan-out duplicates overweight entities; failed matches create MNAR-looking gaps."),
            formula="match_rate = matched_left_keys / left_keys",
            calculation=lambda ctx: (
                f"After joins, row count is {fmt_n(ctx.get('rows'))}. "
                f"Duplicate rows now: {fmt_n((ctx.get('duplicates') or {}).get('rows') or 0)}. "
                "If you expected 1:1, compare rows to distinct entity keys."
            ),
            session_evidence=lambda ctx: (
                f"Frame rows={fmt_n(ctx.get('rows'))}; id-like={list_names(ctx.get('idLike') or [])}. "
                "Re-run join tests on the pipelines that built this extract."
            ),
            example_code=lambda ctx: code_block(
                "import pandas as pd",
                "from buildml import Session",
                "",
                "left = pd.read_csv(\"entities.csv\")   # <-- change",
                "right = pd.read_csv(\"attrs.csv\")    # <-- change",
                "key = \"entity_id\"                   # <-- change",
                "before = len(left)",
                "merged = left.merge(right, on=key, how=\"left\", validate=\"m:1\")  # raises if not m:1",
                "print(\"rows before/after\", before, len(merged))",
                "print(\"match rate\", merged[key].notna().mean())",
                "session = Session.ingest(merged)",
            ),
            what_to_change=(
                "Set join keys, expected cardinality (1:1 / m:1), and minimum match rate.",
            ),
            pitfalls=(
                "Default many-to-many merges without validate=.",
                "Imputing join missings without noticing the join failed.",
            ),
            decide="Assert join cardinality and match-rate thresholds in the data job that builds the frame.",
            read_steps=(
                "Count rows before and after each merge.",
                "Compute match rate and inspect unmatched keys.",
            ),
        ),
        lesson(
            slug="cross-field-consistency",
            stage=1,
            title="cross-field-consistency",
            order=120,
            concept_key="diagnostic-uncertainty",
            tags=("consistency",),
            search_terms=("consistency", "rules", "constraints"),
            plain=(
                "Some facts only make sense together: end_date ≥ start_date, child age < parent age, "
                "state matches ZIP. Single-column screens miss these.",
            ),
            technical=(
                "Cross-field assertions are domain rules. Encode them as checks before fit; "
                "failures are data bugs or rare-but-real edge cases - decide which.",
            ),
            why=("Inconsistent rows poison both training and metrics."),
            formula=None,
            calculation=lambda ctx: (
                f"With {fmt_n(ctx.get('colCount'))} columns, list rule candidates involving "
                f"{list_names((ctx.get('features') or [])[:4])}."
            ),
            session_evidence=lambda ctx: (
                "No automatic cross-field solver ran; use domain rules on this frame's columns."
            ),
            example_code=lambda ctx: code_block(
                "import pandas as pd",
                "from buildml import Session",
                "",
                "frame = pd.read_csv(\"your_data.csv\")  # <-- change",
                "# Example rule - replace with your constraints:",
                "# bad = frame[\"end_date\"] < frame[\"start_date\"]",
                "# print(bad.sum())",
                "session = Session.ingest(frame)",
                "session.eda(include_plots=False, show=False)",
            ),
            what_to_change=("Encode your real multi-column rules; quarantine or fix violators."),
            pitfalls=("Fixing inconsistencies using test labels."),
            decide="Write the top five cross-field rules and a fail policy (drop / repair / quarantine).",
            read_steps=("Brainstorm rules from domain docs.", "Count violators before modeling."),
        ),
        lesson(
            slug="datetime-parsing",
            stage=1,
            title="datetime-parsing",
            order=130,
            concept_key="feature-schema",
            tags=("datetime",),
            search_terms=("datetime", "timestamp", "timezone", "parse"),
            plain=(
                "Dates stored as text do not order, difference, or split temporally. Parsing - with timezone "
                "policy - is a prerequisite for any time-aware validation.",
            ),
            technical=(
                "Parse to datetime64 at load; use session.extract_dates(...) for calendar parts. "
                "Temporal splits need a true time axis, not lexical strings.",
            ),
            why=("Lexical 'dates' make leakage checks and rolling features wrong."),
            formula=None,
            calculation=lambda ctx: (
                f"Datetime columns flagged: "
                f"{list_names([ctx['timeCol']] if ctx.get('timeCol') else []) or 'none in this profile'}."
            ),
            session_evidence=lambda ctx: (
                f"Time column: {(ctx.get('timeCol') or {}).get('name') if ctx.get('timeCol') else 'not detected'}."
            ),
            example_code=lambda ctx: code_block(
                "import pandas as pd",
                "from buildml import Session",
                "",
                "frame = pd.read_csv(\"your_data.csv\")  # <-- change",
                f"time_col = \"{(ctx.get('timeCol') or {}).get('name') or '<timestamp>'}\"  # <-- change",
                "frame[time_col] = pd.to_datetime(frame[time_col], utc=True, errors=\"coerce\")",
                "session = Session.ingest(frame)",
                "session = session.set_roles({",
                "    time_col: \"time\",",
                f'    "{target_name(ctx)}": "target",',
                "})",
                "session = session.extract_dates(columns=[time_col])  # calendar parts",
            ),
            what_to_change=("Set timestamp column and timezone policy; assign role 'time'."),
            pitfalls=("Parsing with dayfirst ambiguity; mixing naive and aware timestamps."),
            decide="Parse timestamps once, store timezone policy, assign the time role before splitting.",
            read_steps=("Check nulls after parsing.", "Plot min/max time coverage."),
        ),
        lesson(
            slug="precision-and-heaping",
            stage=1,
            title="precision-and-heaping",
            order=140,
            concept_key="diagnostic-uncertainty",
            tags=("heaping", "precision"),
            search_terms=("heaping", "rounding", "precision", "spikes"),
            plain=(
                "Heaping is when values pile on round numbers (ages at 30/40, incomes at 50k). "
                "It is measurement behaviour, not a natural density spike.",
            ),
            technical=(
                "Spikes at round levels bias histograms and some continuous models. "
                "Decide whether to keep as-is, jitter carefully, or bin deliberately.",
            ),
            why=("Models can learn rounding artifacts as if they were causal thresholds."),
            formula=None,
            calculation=lambda ctx: (
                f"Inspect numeric columns for round-number spikes, e.g. {first_numeric(ctx)}. "
                f"Skewed columns may also show heaping: {list_names(ctx.get('skewed') or [])}."
            ),
            session_evidence=lambda ctx: (
                f"Numeric columns available for heaping review: {list_names(ctx.get('numeric') or [])}."
            ),
            example_code=lambda ctx: code_block(
                "import pandas as pd",
                "from buildml import Session",
                "",
                "frame = pd.read_csv(\"your_data.csv\")",
                f"col = \"{first_numeric(ctx)}\"",
                "vc = frame[col].value_counts().head(15)",
                "print(vc)  # look for round heaps",
                "session = Session.ingest(frame)",
                "# If you bin deliberately:",
                "session = session.split(test_size=0.2, random_state=0)",
                f"session = session.bin(columns=[\"{first_numeric(ctx)}\"], n_bins=5)  # <-- tune",
            ),
            what_to_change=("Choose keep / deliberate binning; never silent jitter without documenting."),
            pitfalls=("Mistaking heaping for multimodal truth.", "Binning using test-driven edges."),
            decide="Document measurement precision and whether heaping is artifact or policy.",
            read_steps=("Value-count the top spikes.", "Ask how the field is collected."),
        ),
        lesson(
            slug="nested-and-multivalued",
            stage=1,
            title="nested-and-multivalued",
            order=150,
            concept_key="feature-schema",
            tags=("nested", "lists"),
            search_terms=("json", "list", "multivalued", "nested"),
            plain=(
                "Lists, JSON blobs, and multi-valued fields are not atomic features. You must explode, "
                "aggregate, or embed them before classical tabular models see them.",
            ),
            technical=(
                "Choose a representation (multi-hot, counts, embeddings) and fit any vocabulary on train. "
                "Nested structures often hide leakage-rich identifiers.",
            ),
            why=("Feeding raw JSON strings teaches token noise, not structure."),
            formula=None,
            calculation=lambda ctx: (
                f"Scan object-like columns among {fmt_n(ctx.get('colCount'))} fields for list/JSON payloads "
                "before treating them as categoricals."
            ),
            session_evidence=lambda ctx: (
                "Flatten/aggregate nested fields upstream, then ingest a rectangular frame into BuildML."
            ),
            example_code=lambda ctx: code_block(
                "import pandas as pd",
                "from buildml import Session",
                "",
                "frame = pd.read_csv(\"your_data.csv\")  # <-- already flattened preferred",
                "# Example: multi-label tags -> count / multi-hot outside Session, then:",
                "session = Session.ingest(frame)",
                "session = session.set_roles({",
                f'    "{target_name(ctx)}": "target",',
                f'    "{first_feature(ctx)}": "feature",',
                "})",
                'session.learn("feature-schema", level="intermediate")',
            ),
            what_to_change=("Flatten nested fields with a documented aggregation grain."),
            pitfalls=("Exploding lists without re-aggregating to the modeling grain."),
            decide="Pick one rectangular representation for each nested field and freeze it.",
            read_steps=("Identify nested columns.", "Define aggregation to entity grain."),
        ),
    ]


def _high_card_col(ctx: dict) -> str:
    high = ctx.get("highCard") or []
    if high and isinstance(high[0], dict) and high[0].get("name"):
        return str(high[0]["name"])
    return first_categorical(ctx)


def _dtype_calc(ctx: dict) -> str:
    counts: dict[str, int] = {}
    for col in ctx.get("cols") or []:
        if not isinstance(col, dict):
            continue
        dtype = str(col.get("dtype") or "unknown")
        counts[dtype] = counts.get(dtype, 0) + 1
    summary = ", ".join(f"{n} {k}" for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    mem = ctx.get("memoryMB")
    mem_bit = f" Approx memory ~{fmt_compact(float(mem))} MB." if mem is not None else ""
    return f"{fmt_n(ctx.get('colCount'))} columns: {summary or 'dtype summary unavailable'}.{mem_bit}"


def _range_calc(ctx: dict) -> str:
    nums = [n for n in (ctx.get("numeric") or []) if isinstance(n, dict) and n.get("min") is not None]
    if not nums:
        return f"{fmt_n(len(ctx.get('numeric') or []))} numeric columns present; ranges not fully profiled."
    widest = max(nums, key=lambda n: abs(float(n.get("max") or 0) - float(n.get("min") or 0)))
    return (
        f"{fmt_n(len(nums))} numeric columns with ranges; widest span "
        f"{widest.get('name')}: {fmt_compact(float(widest.get('min') or 0))} -> "
        f"{fmt_compact(float(widest.get('max') or 0))}."
    )


def _missing_example(ctx: dict) -> str:
    missing = ctx.get("missing") or []
    nums = [m["name"] for m in missing if m.get("name")][:3]
    # dtype not always on missing entries; use numeric names intersection
    numeric_names = {n.get("name") for n in (ctx.get("numeric") or []) if isinstance(n, dict)}
    cat_names = set(ctx.get("categorical") or [])
    num_cols = [n for n in nums if n in numeric_names] or [
        m["name"] for m in missing if m.get("name") in numeric_names
    ][:3]
    cat_cols = [m["name"] for m in missing if m.get("name") in cat_names][:2]
    if not num_cols and not cat_cols and not missing:
        return code_block(
            "from buildml import Session",
            "import pandas as pd",
            "",
            "session = Session.ingest(pd.read_csv(\"your_data.csv\"))",
            "session = session.set_roles({",
            f'    "{target_name(ctx)}": "target",',
            f'    "{first_feature(ctx)}": "feature",',
            "})",
            "session = session.split(test_size=0.2, stratify=True, random_state=0)",
            'session.explain("impute", moment="before")',
            "# nothing to fill in this frame - keep the plan for the next extract",
        )
    lines = [
        "from buildml import Session",
        "import pandas as pd",
        "",
        "session = (",
        "    Session.ingest(pd.read_csv(\"your_data.csv\"))  # <-- change",
        "    .set_roles({",
        f'        "{target_name(ctx)}": "target",',
        f'        "{first_feature(ctx)}": "feature",',
        "    })",
        "    .split(test_size=0.2, stratify=True, random_state=0)  # split BEFORE impute",
        ")",
        "",
    ]
    if num_cols or missing:
        cols = num_cols or [m["name"] for m in missing[:2]]
        lines.append("session = session.impute(")
        lines.append("    strategy=\"median\",  # <-- or mean / constant")
        lines.append("    columns=[")
        lines.append(quote_list(cols))
        lines.append("    ],")
        lines.append(")")
    if cat_cols:
        lines.append("session = session.impute(")
        lines.append("    strategy=\"most_frequent\",")
        lines.append("    columns=[")
        lines.append(quote_list(cat_cols))
        lines.append("    ],")
        lines.append(")")
    lines.append("# fills are learned on train rows only")
    return code_block(*lines)
