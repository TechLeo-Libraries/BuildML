"""Stage 00 · Framing - what must be true before statistics mean anything."""

from __future__ import annotations

from buildml.dashboard.academy_curriculum._helpers import (
    code_block,
    first_feature,
    fmt_n,
    is_classification,
    list_names,
    target_name,
)
from buildml.dashboard.academy_curriculum._types import LessonSpec, lesson


def lessons() -> list[LessonSpec]:
    return [
        lesson(
            slug="problem-framing",
            stage=0,
            title="problem-framing",
            order=10,
            concept_key="diagnostic-uncertainty",
            tags=("framing", "task"),
            search_terms=("problem", "task", "prediction", "decision"),
            plain=(
                "Before you open a notebook, write the decision the model is supposed to support. "
                "A prediction without a decision is a report, not a product.",
                "Plain version: say who will use the score, when they see it, and what they will do differently if the score is high versus low.",
            ),
            technical=(
                "Problem framing fixes the prediction target, the decision horizon, the action space, "
                "and the cost of errors. EDA numbers only become evidence after those choices are explicit.",
                "BuildML treats framing as a teaching and readiness concern: diagnostics can describe the "
                "frame, but they cannot invent the business decision.",
            ),
            why=(
                "The same CSV supports many incompatible tasks; framing picks one.",
                "Metrics, splits, and leakage rules all depend on when the outcome is known.",
                "Without framing, 'good accuracy' is meaningless.",
            ),
            formula=None,
            calculation=lambda ctx: (
                f"This session examined {fmt_n(ctx.get('rows'))} rows x {fmt_n(ctx.get('colCount'))} columns "
                f"with task hint '{ctx.get('task') or 'unspecified'}'. Framing asks: what decision changes "
                f"if we predict {target_name(ctx)} vs a different label?"
            ),
            session_evidence=lambda ctx: (
                (
                    f"Target column is '{target_name(ctx)}' ({ctx.get('task')}). "
                    if ctx.get("has_target")
                    else "No target role is declared yet - framing is unfinished. "
                )
                + f"{fmt_n(ctx.get('eligible'))} eligible features are in play. "
                + (
                    "Classification framing also needs class costs and the positive-class definition."
                    if is_classification(ctx)
                    else "Regression framing needs the unit of the target and the error that matters operationally."
                    if ctx.get("has_target")
                    else "Declare a target (or explicitly choose unsupervised work) before trusting readiness gates."
                )
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "# 1) Load YOUR table (change path / columns).",
                "frame = pd.read_csv(\"your_data.csv\")  # <-- change",
                "session = Session.ingest(frame)",
                "",
                "# 2) Name the decision in comments, then encode it as roles.",
                f"# Decision: predict {target_name(ctx)} to support <action> before <horizon>.",
                "session = session.set_roles({",
                f'    "{target_name(ctx)}": "target",  # <-- change if your label differs',
                f'    "{first_feature(ctx)}": "feature",',
                "    # \"customer_id\": \"id\",",
                "})",
                "",
                "# 3) Ask BuildML to teach the framing vocabulary for this session.",
                'brief = session.learn("diagnostic-uncertainty", level="beginner")',
                "print(brief.concept.summary if brief.concept else brief.suggested)",
            ),
            what_to_change=(
                "Replace the CSV path and column names with your dataset.",
                "Rewrite the decision comment: user, action, horizon, and cost of mistakes.",
                "If unsupervised, omit target and use session.learn('cluster-validity-not-truth') instead.",
            ),
            pitfalls=(
                "Starting EDA before writing the decision the score will change.",
                "Treating 'predict churn' as framing when the real question is 'whom to call this week'.",
                "Changing the target definition after model selection without restarting validation.",
            ),
            decide=lambda ctx: (
                f"Write one sentence: 'We predict {target_name(ctx)} so that <role> can <action> before <time>.' "
                "If you cannot finish that sentence, stop modeling."
            ),
            read_steps=(
                "Write the decision, horizon, and action space in plain language.",
                "Name the label column and when it becomes known relative to prediction time.",
                "List which error is worse (false positive vs false negative, or over- vs under-prediction).",
            ),
        ),
        lesson(
            slug="unit-of-analysis",
            stage=0,
            title="unit-of-analysis",
            order=20,
            concept_key="column-roles",
            tags=("grain", "rows"),
            search_terms=("grain", "unit", "row", "entity", "duplicate"),
            plain=(
                "The unit of analysis is what one row means: one customer, one order, one day, one sensor ping. "
                "If you get the grain wrong, every rate and model score is about the wrong thing.",
            ),
            technical=(
                "Grain is the entity identity that should be unique (or intentionally repeated) in the modeling table. "
                "Duplicates, fan-out joins, and event logs often change grain silently.",
            ),
            why=(
                "Metrics computed at the wrong grain answer the wrong question.",
                "Leakage and group splits depend on which rows share an entity.",
            ),
            formula="rows == distinct(entity_key)  ⇒  claimed grain holds (for unique entities)",
            calculation=lambda ctx: (
                f"Observed rows = {fmt_n(ctx.get('rows'))}. Exact duplicate rows = "
                f"{fmt_n((ctx.get('duplicates') or {}).get('rows') or 0)}. "
                f"Identifier-like columns flagged: {list_names(ctx.get('idLike') or [])}. "
                "If your grain is 'one row per id', distinct(id) must equal rows after de-duplication."
            ),
            session_evidence=lambda ctx: (
                f"{fmt_n(ctx.get('rows'))} rows in the analysis frame; "
                f"{fmt_n((ctx.get('duplicates') or {}).get('rows') or 0)} exact duplicate rows; "
                f"id-like columns: {list_names(ctx.get('idLike') or [])}."
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "session = Session.ingest(pd.read_csv(\"your_data.csv\"))  # <-- change",
                "",
                "# Inspect whether near-unique columns are identities or measurements.",
                "report = session.eda(include_plots=False, show=False)",
                "quality = report.to_dict().get(\"quality\", {})",
                "print(\"id_like:\", quality.get(\"id_like_columns\"))",
                "print(\"duplicate_rows:\", quality.get(\"duplicate_row_count\"))",
                "",
                "# Declare identity columns so they never enter the feature matrix.",
                "session = session.set_roles({",
                *(
                    [f'    "{name}": "id",' for name in (ctx.get("idLike") or [])[:2]]
                    or ['    "<entity_id>": "id",  # <-- change']
                ),
                f'    "{target_name(ctx)}": "target",  # if supervised',
                "})",
            ),
            what_to_change=(
                "Set the entity key that defines your grain.",
                "If the table is event-level but you need customer-level predictions, aggregate first.",
                "Assign true identifiers the 'id' role (not 'feature').",
            ),
            pitfalls=(
                "Assuming CSV row order is a meaningful unit.",
                "Joining tables without checking row-count fan-out.",
                "Dropping duplicates without stating which key defines sameness.",
            ),
            decide=(
                "State the grain in one sentence and enforce it with a uniqueness check on the entity key "
                "before drawing any split."
            ),
            read_steps=(
                "Compare row count to distinct entity keys.",
                "Inspect colliding keys: same key, different payloads means finer grain than claimed.",
                "Check whether duplicates cluster by source or time (ingestion incident vs natural repeats).",
            ),
        ),
        lesson(
            slug="population-and-sampling-frame",
            stage=0,
            title="population-and-sampling-frame",
            order=30,
            concept_key="diagnostic-uncertainty",
            tags=("sampling", "population"),
            search_terms=("population", "sample", "bias", "coverage"),
            plain=(
                "Your table is almost never 'the world'. It is a sampling frame: the rows someone could observe. "
                "Models learn that frame, not the population you wish you had.",
            ),
            technical=(
                "Coverage bias, selection into the extract, and time windows define the estimand. "
                "EDA can describe the sample; it cannot certify external validity.",
            ),
            why=(
                "A model that fits yesterday's applicants may fail on today's traffic.",
                "Missing segments look like 'clean data' when they were filtered upstream.",
            ),
            formula=None,
            calculation=lambda ctx: (
                f"Analysis used {fmt_n(ctx.get('rows'))} of {fmt_n(ctx.get('rowsTotal'))} rows"
                + (
                    " (sampled for profiling - tails may differ on the full extract)."
                    if ctx.get("sampled")
                    else " (full extract in this session)."
                )
                + " Ask: who never enters this table?"
            ),
            session_evidence=lambda ctx: (
                f"Rows analysed: {fmt_n(ctx.get('rows'))}; rows reported total: {fmt_n(ctx.get('rowsTotal'))}; "
                f"sampled flag: {bool(ctx.get('sampled'))}."
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "session = Session.ingest(pd.read_csv(\"your_data.csv\"))  # <-- change",
                "report = session.eda(include_plots=False, show=False)",
                "overview = report.to_dict()[\"overview\"]",
                "print(overview.get(\"n_rows\"), overview.get(\"analysis_rows\"))",
                "",
                "# Document the frame you actually have:",
                "# population = <who should be scored>",
                "# sampling_frame = <who appears in this extract>",
                "# exclusion_rules = <filters applied upstream>",
                'session.learn("diagnostic-uncertainty", level="beginner")',
            ),
            what_to_change=(
                "Write the intended population and the extract's inclusion rules.",
                "If you sample for EDA, keep full-data checks for rare segments.",
            ),
            pitfalls=(
                "Treating a convenience extract as a random sample.",
                "Training on historical approvals only, then scoring all applicants.",
            ),
            decide="Record who is missing from the extract and whether the model will be asked to score them anyway.",
            read_steps=(
                "Compare analysis_rows to n_rows and note sampling.",
                "List upstream filters (SQL WHERE, product eligibility, consent).",
                "Name segments that matter operationally and check they appear.",
            ),
        ),
        lesson(
            slug="target-definition",
            stage=0,
            title="target-definition",
            order=40,
            concept_key="column-roles",
            tags=("target", "label"),
            search_terms=("label", "target", "outcome", "y"),
            plain=(
                "The target is the thing you want to predict - and the definition of that thing is a modeling choice. "
                "'Churned' might mean cancelled within 30 days, or inactive for 90 days; those are different problems.",
            ),
            technical=(
                "Target definition includes the outcome event, observation window, positive-class rule (classification), "
                "and units (regression). Label timing relative to features sets the leakage boundary.",
            ),
            why=(
                "A fuzzy label makes every metric uninterpretable.",
                "Relabeling after seeing model scores contaminates evaluation.",
            ),
            formula=None,
            calculation=lambda ctx: _target_calc(ctx),
            session_evidence=lambda ctx: _target_session(ctx),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "frame = pd.read_csv(\"your_data.csv\")  # <-- change",
                "# Define the label explicitly BEFORE ingest when possible:",
                "# frame[\"y\"] = (frame[\"cancelled_at\"] - frame[\"asof_date\"]).dt.days.between(1, 30)",
                "session = (",
                "    Session.ingest(frame)",
                "    .set_roles({",
                f'        "{target_name(ctx)}": "target",  # <-- your label column',
                f'        "{first_feature(ctx)}": "feature",',
                "    })",
                ")",
                "report = session.eda(include_plots=False, show=False)",
                "print(report.to_dict().get(\"target\"))",
            ),
            what_to_change=(
                "Change the label rule and column name to match your outcome definition.",
                "For classification, document the positive class and horizon.",
                "For regression, document units and any clipping applied to y.",
            ),
            pitfalls=(
                "Using a post-outcome field as a feature while defining y.",
                "Redefining positives after seeing which threshold looks good.",
            ),
            decide=lambda ctx: (
                f"Freeze the definition of '{target_name(ctx)}' (event + horizon + units) before feature work."
            ),
            read_steps=(
                "Write the label rule in one line of English and one line of code.",
                "Check label missingness separately from feature missingness.",
                "Confirm label timing: known only after prediction time?",
            ),
        ),
        lesson(
            slug="provenance-and-lineage",
            stage=0,
            title="provenance-and-lineage",
            order=50,
            concept_key="feature-schema",
            tags=("lineage", "schema"),
            search_terms=("provenance", "lineage", "source", "schema"),
            plain=(
                "Provenance is where each column came from and when it was written. Without it, you cannot tell "
                "a trustworthy feature from a leaky join artifact.",
            ),
            technical=(
                "Lineage covers source systems, join keys, as-of timing, and transform history. "
                "BuildML session history and checkpoints help, but source documentation still starts with you.",
            ),
            why=(
                "Silent schema drift breaks production quietly.",
                "Audit and reproducibility require knowing how the matrix was built.",
            ),
            formula=None,
            calculation=lambda ctx: (
                f"Schema snapshot: {fmt_n(ctx.get('colCount'))} columns; engine={((ctx.get('ds') or {}).get('engine'))}. "
                "Lineage is not a statistic - list source -> join -> as-of for each feature family."
            ),
            session_evidence=lambda ctx: (
                f"{fmt_n(ctx.get('colCount'))} columns profiled; dtypes present for "
                f"{sum(1 for c in (ctx.get('cols') or []) if (c.get('dtype') if isinstance(c, dict) else None))} columns."
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "session = Session.ingest(pd.read_csv(\"your_data.csv\"))  # <-- change",
                "report = session.eda(include_plots=False, show=False)",
                "overview = report.to_dict()[\"overview\"]",
                "print(\"columns:\", overview.get(\"columns\"))",
                "print(\"dtypes:\", overview.get(\"dtypes\"))",
                "",
                "# Teach + document schema expectations",
                'session.learn("feature-schema", level="beginner")',
                "# Keep a side table: column -> source_system -> asof_rule -> owner",
            ),
            what_to_change=(
                "Fill the lineage side table for your warehouse / feature store.",
                "Record as-of rules for any joined attributes.",
            ),
            pitfalls=(
                "Treating 'the CSV we got' as documentation.",
                "Joining future attributes because the batch job ran late.",
            ),
            decide="Attach an owner and as-of rule to every feature family before the first fit.",
            read_steps=(
                "List columns that came from joins vs raw events.",
                "Mark any column whose timestamp could be after the prediction moment.",
                "Confirm dtype intent matches storage (codes vs quantities).",
            ),
        ),
        lesson(
            slug="sensitive-attributes",
            stage=0,
            title="sensitive-attributes",
            order=60,
            concept_key="feature-schema",
            tags=("fairness", "sensitive"),
            search_terms=("sensitive", "fairness", "protected", "bias"),
            plain=(
                "Sensitive attributes are fields that identify protected groups or proxies for them. "
                "You must decide whether they are used for features, for monitoring only, or excluded entirely.",
            ),
            technical=(
                "Fairness evaluation needs group columns even when the model must not train on them. "
                "BuildML fairness tools expect explicit group handling; EDA should surface candidates early.",
            ),
            why=(
                "Hidden proxies recreate discrimination without naming the attribute.",
                "You cannot monitor slice performance without retaining group labels somewhere.",
            ),
            formula=None,
            calculation=lambda ctx: (
                f"Review {fmt_n(ctx.get('colCount'))} columns for direct sensitive fields and strong proxies "
                f"(geography, language, segment). High-cardinality fields: {list_names(ctx.get('highCard') or [])}."
            ),
            session_evidence=lambda ctx: (
                "This sheet does not auto-label protected attributes - that is a policy decision. "
                f"Candidate review set includes categoricals: {list_names(ctx.get('categorical') or [])}."
            ),
            example_code=lambda ctx: code_block(
                "from buildml import Session",
                "import pandas as pd",
                "",
                "session = Session.ingest(pd.read_csv(\"your_data.csv\"))  # <-- change",
                "session = session.set_roles({",
                "    # Keep group columns out of training features when policy requires it:",
                '    # "gender": "ignore",   # <-- change to your policy',
                '    # "region": "ignore",',
                f'    "{target_name(ctx)}": "target",',
                f'    "{first_feature(ctx)}": "feature",',
                "})",
                "",
                "# Later: evaluate slices with error_slices after fit (group column retained outside X).",
                'session.learn("feature-schema", level="intermediate")',
            ),
            what_to_change=(
                "Apply your organisation's sensitive-attribute policy.",
                "Decide monitor-only vs exclude vs legally required inclusion.",
            ),
            pitfalls=(
                "Dropping the group column and then being unable to audit disparities.",
                "Leaving ZIP/postcode as a feature while claiming the model is group-blind.",
            ),
            decide="Classify each sensitive or proxy column as train / monitor-only / exclude, in writing.",
            read_steps=(
                "List direct sensitive fields and likely proxies.",
                "Check whether any id-like or high-card field encodes group membership.",
                "Plan how slice evaluation will access group labels after fit.",
            ),
        ),
    ]


def _target_calc(ctx: dict) -> str:
    target = ctx.get("target")
    if not isinstance(target, dict) or not target.get("name"):
        return "No target is declared, so the label definition is still open."
    if is_classification(ctx):
        classes = target.get("classes") or []
        if not classes:
            return f"Classification target '{target['name']}' is set; class counts were not summarised in this report."
        total = sum(int(c.get("count") or 0) for c in classes)
        parts = [
            f"{c.get('label')}: {fmt_n(c.get('count'))} "
            f"({(100 * int(c.get('count') or 0) / total):.1f}%)"
            for c in classes
            if total
        ]
        return (
            f"Label '{target['name']}' class mix on analysed rows - "
            + "; ".join(parts)
            + ". Confirm this matches the intended positive definition."
        )
    stats = target.get("stats") or {}
    return (
        f"Regression target '{target['name']}'"
        + (f" with reported median {stats.get('median')}" if stats.get("median") is not None else "")
        + ". Confirm units and any clipping."
    )


def _target_session(ctx: dict) -> str:
    if not ctx.get("has_target"):
        return "No target column in this session - declare one or choose an unsupervised path."
    return (
        f"Target '{target_name(ctx)}' · task={ctx.get('task')}. "
        "Evidence below reflects this label definition as loaded, not an external ground-truth audit."
    )
