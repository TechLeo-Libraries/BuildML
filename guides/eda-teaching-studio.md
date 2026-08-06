# EDA and Teaching Studio

> **Install:**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Optional: `pip install "buildml[viz]"`, `"buildml[eda]"`,
> `"buildml[dashboard]"` for plots / the Industry EDA App.
> See [installation](../docs/installation.rst).

Explore **before** you mutate. `session.eda()` returns structured findings and
read-only recommendations. Teaching surfaces (`explain`, `learn`, `workflow`,
`walkthrough`, `dry_run`) expose the operation catalog and the concept notes
behind it: they do not certify that your split or model suits the domain.

If the vocabulary itself is new, start with `session.learn()`: everything below
reads at a `beginner`, `intermediate`, or `advanced` level, and beginner assumes
no prior machine-learning knowledge.

Related: [classical end-to-end](classical-end-to-end.md),
[usage](../docs/usage.rst), [glossary](glossary.md).

---

## Why a Teaching Studio exists

ML libraries usually document methods in isolation. BuildML ties every public
Session operation to a versioned catalog (kept in sync by CI). That lets you:

1. Ask “what does `impute` assume?” **before** calling it.
2. Ask “what *is* imputation?” without leaving the session (`learn`).
3. See which ops are `done` / `available` / `blocked` / `skipped`.
4. Preview a chain without appending history (`dry_run`).
5. Export an offline audit HTML for review.

The live dashboard (`eda_app`) is an optional FastAPI **Industry EDA App** with
document-sheet IA: Command cockpit (numbered spine 01-08), Readiness Gates,
Concept Academy, and secondary domain boards. Tokens and analytic coverage are
shared with **BUILDML STATIC EDA** (`html_format="research"`). It is not a
replacement for domain judgment.

**Concept Academy** is a staged ML-engineering learning hub (00 Framing through
05 Interpretation, plus 06 Domain depth). It teaches every BuildML
`CONCEPT_NOTES` entry (~204) as a first-class lesson (beginner through advanced
prose, a calculation walkthrough bound to this session's numbers or an honest
N/A, a copyable BuildML `Session` example, pitfalls, and "what to change for
your data" callouts), plus readiness-path curriculum slugs that are not
themselves catalog keys. Cited vs reference chips follow findings on the live
report, not a hardcoded demo story. Open `#/academy` in the app, or call
`build_academy_payload(report.to_dict())` from `buildml.dashboard.academy`.

---

## Use case: findings before preparation

```python
import pandas as pd

from buildml import Session

frame = pd.DataFrame(
    {
        "age": [21, None, 35, None, 29, 33, 52, 47],
        "income": [40, 55, 60, 80, 50, 70, 90, 65],
        "approved": [0, 1, 0, 1, 0, 1, 1, 0],
    }
)

session = (
    Session.ingest(frame)
    .set_roles({"age": "feature", "income": "feature", "approved": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
)

report = session.eda(include_plots=False)
for finding in report.findings[:10]:
    print(finding.severity, finding.title)
for rec in getattr(report, "recommendations", [])[:5] or []:
    print("rec:", rec)
```

Recommendations **name** Session operations; they do not execute them.

---

## Use case: offline HTML (studio vs research)

```python
# Offline Industry App snapshot (dashboard SPA assets embedded when available)
session.eda(export_html="artifacts/eda_studio.html", html_format="studio")

# BUILDML STATIC EDA (Industry readiness sheet; needs buildml[viz] for plots)
session.eda(
    include_plots=True,
    export_html="artifacts/eda_research.html",
    html_format="research",
    export_figures="artifacts/eda-figures",
)
```

`html_format="research"` is the Static product: KPI strip, findings register,
assumptions, ledger, recommended Session calls, figures, methods, and degraded
rows. It omits Gates, Academy, and human gate-status UX. HTML artifacts embed
required styles/assets so they open offline.

Local preview from a synthetic dirty frame:

```bash
python scripts/generate_static_eda_preview.py
# writes .buildml-artifacts/static_eda_cockpit.html
```

---

## Use case: live Industry EDA App

```python
# pip install "buildml[dashboard]"  # after GitHub 2.x
handle = session.eda_app(port=8765, open_browser=True)
# alias: session.open_eda_dashboard(port=8765)
print(handle.url)
# ... explore Cockpit, Readiness Gates, Concept Academy, domain boards ...
handle.stop()
```

Or from a dirty synthetic extract:

```bash
python scripts/launch_synthetic_eda_studio.py
```

Surfaces in the App (document sheets, not a sidebar studio):

| Board | Role |
| --- | --- |
| Command cockpit | KPI strip and numbered spine 01-08: findings register, assumptions, ledger, recommended sequence, domain briefs, figures, methods/limitations, skipped/degraded |
| Readiness gates | Second-pass tally, sticky filters, stage-grouped gate cards; click a gate for the learning sidebar (beginner through advanced, calculations, copy-paste Session examples). Session marks are UI-only |
| Concept academy | Sticky search/stage tools, contents board, two-column concept entries (~204 catalog lessons plus readiness-path slugs) |
| Domain boards | Quality, features, relationships, multivariate, target, outliers, visuals |

**Offline HTML** is the primary export in the app header (same SPA surface,
including Gates and Academy). CSV and PDF routes remain on the App API for
automation; they are not header actions. BUILDML STATIC EDA exposes **Offline
HTML only** in its header (no CSV or PDF briefing buttons); the button
re-downloads the already-offline research HTML snapshot.

### Dataset adaptability (shared contract)

Narrative must bind to the **live report**, not a demo/churn template. Shared
helpers live in:

| Layer | Module | Use |
| --- | --- | --- |
| Python | `buildml.dashboard.adapt` | `build_adapt_context(report)`, `session_sentence`, `what_to_change`, `list_names`, `target_phrase` |
| Frontend | `static/js/learn_ui.js` | `callout`, `codeBlock`, `calcBlock`, `whatToChange`, `sectionScaffold`, `wireLearnUi` |

Academy / Gates agents should:

1. Import adaptive facts from `adapt.py` (or read `meta.adapt` / `sheet.adapt` from the API) instead of hardcoding column names like `target_churn`. Prefer `build_gate_context` inputs already flattened in `adapt.build_adapt_context`.
2. Import presentation from `learn_ui.js` (ESM) **or** use `window.BuildMLLearnUI` (Academy view pattern). Script order in `templates/index.html`: `learn_ui.js` → `gates_view.js` → `academy_view.js` → `app.js`.
3. Leave curriculum bodies in `academy.py` / `gates.py` / `academy_curriculum/`; only bind session lines, evidence, and worked examples to live `adapt` fields / report numbers.
4. Offline HTML inlines the same module graph via blob URLs in `offline.py`. Keep import rewrite placeholders in sync when adding views.

Cockpit already exposes `sheet.adapt`, `sheet.session_sentence`, `sheet.what_to_change`, and spine meta counts for scannability.

**Gate marks are UI-only.** Toggling “Mark for this session” on a gate stays in
the open browser tab and is discarded on refresh. BuildML does not write gate
judgments to the Session, history, disk, or any saved dataset copy (privacy and
complexity: a durable mark would imply remembering *why* a decision was made).
Session marks also do not persist inside an Offline HTML file beyond the open
tab.

If the port is busy, pass another port.

### Adaptability proofs and gauntlet

Tier A proof [`proofs/eda-industry-adaptability/`](../proofs/eda-industry-adaptability/)
runs Static research HTML plus Dashboard App payloads across 12 frames (sklearn
real-world tables and synthetic stress cases). Regenerate from the repo root:

```bash
python proofs/eda-industry-adaptability/script.py
# or the convenience smoke script (writes under .buildml-artifacts/gauntlet/):
python scripts/eda_adaptability_gauntlet.py
```

Proof results land under `proofs/eda-industry-adaptability/results/` (gitignored).
Gauntlet artifacts under `.buildml-artifacts/gauntlet/` are also ignored. Exit
code 0 only when every case passes.

---

## Teaching surfaces: explain / learn / workflow / walkthrough

```python
before = session.explain("feature_importance", moment="before")
print(before.prerequisites, before.risks)

for step in session.workflow():
    if step.status == "blocked":
        print(step.operation, step.reasons or step.blockers)

preview = session.dry_run(["impute", "scale", "fit"])
summary = session.summarize_history()
print(summary.unresolved_risks)

walkthrough = session.walkthrough(export_html="artifacts/workflow.html")
```

- `available` means API prerequisites pass: **not** “you should run this.”
- `explain(..., moment="after")` joins catalog text to the latest recorded call.
- `dry_run` does not append history.

### Reading levels

Every explanation is written at three levels; `beginner` is the default and
assumes no prior machine-learning vocabulary.

```python
primer = session.explain("feature_importance").beginner
print(primer.plain_summary)          # what this is, in ordinary words
print(primer.analogy)                # the intuition
primer.steps                         # what happens, in order
primer.prerequisites_in_plain_words  # what must be true first, and how to get there
primer.key_parameters                # each knob: meaning, effect, typical choice
primer.common_pitfalls               # how this goes wrong
primer.glossary                      # the jargon this answer used, defined
primer.mini_example                  # a runnable sketch

session.explain("feature_importance", level="advanced")  # no scaffolding
```

The level changes how much is rendered, never what is true: assumptions,
leakage risks, and failure modes are present at every level. `advanced` drops
the analogy and the in-line glossary and widens the parameter and pitfall lists.

### `learn`: the concept behind the call

`explain` answers "what will this do *here, now*". `learn` answers "what is this,
and what should I understand first". It accepts a concept key, an operation name,
or the word you tripped over, and forgives spacing and hyphenation.

```python
session.learn()                       # foundation concepts, in reading order
brief = session.learn("leakage")      # a term resolves to the concept teaching it

brief.concept.plain_summary           # the idea from scratch
brief.concept.misconceptions          # what people wrongly believe, and the correction
brief.concept.check_yourself          # questions to test whether it landed
[note.key for note in brief.read_first]  # prerequisites, if any
[note.key for note in brief.read_next]   # where to go once it lands
brief.related_operations              # the BuildML calls that apply it

session.learn("split")                # an operation name returns its primer
session.learn("cross-validation", level="intermediate")
```

Concept notes, the glossary, and operation primers are the same objects the
walkthrough, the Industry App, and the AI operator's `explain_operation` /
`learn_concept` tools read from, so no surface teaches something another
contradicts. All of it is static teaching material: it describes ideas and
BuildML's contract, and inspects none of your data.

---

## Evaluation and diagnostic HTML

```python
session.impute(strategy="median").scale(method="standard")
from sklearn.linear_model import LogisticRegression

session.fit(LogisticRegression(max_iter=500), task="classification")
session.evaluate(
    partition="validation",
    include_plots=True,
    export_html="artifacts/evaluation.html",
)
# Adaptive plot boards (buildml[viz]):
# session.eval_plots(partition="validation", export_html="artifacts/plots.html")
```

See [diagnostics & search](classical-diagnostics-search.md).

---

## Failure modes

| Issue | Guidance |
| --- | --- |
| `MissingExtraError: dashboard` | Install `buildml[dashboard]` |
| `MissingExtraError: viz` | Install `buildml[viz]` for plots |
| Acting on recommendations blindly | Still call Session methods yourself; verify domain fit |
| Confusing AI advisor with EDA | AI is optional (`buildml[ai]`); EDA/App work offline |

---

## Related

- [AI operator safety](ai-operator-safety.md)
- [Classical end-to-end](classical-end-to-end.md)
- [Artifacts](artifacts-checkpoints-bundles.md)
