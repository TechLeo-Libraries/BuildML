# EDA and Teaching Studio

```bash
pip install buildml
# plots / profiling: pip install "buildml[viz]" "buildml[eda]"
# local app: pip install "buildml[dashboard]"
```

Use `session.eda()` to inspect your data before choosing preprocessing or
modeling steps. The report contains findings and recommended operations;
it leaves the dataset unchanged. Review each recommendation against your
analysis goal before applying it.

The teaching methods explain operations and their prerequisites. Start
with `session.learn()` for introductory concepts, or use `session.explain()`
for a particular operation. Explanations use the `beginner` reading level
by default.

Related: [classical end-to-end](classical-end-to-end.md),
[usage](../docs/usage.rst), [glossary](glossary.md).

---

## Inspect data and plan your workflow

Use `explain("impute")` to review the assumptions behind filling missing
values, and `learn("imputation")` to read the underlying concept. The workflow
view shows operations marked `done`, `available`, `blocked`, or `skipped`.
Use `dry_run` to preview a sequence without changing the Session or its
history, and export HTML to share the report.

The live dashboard (`session.eda_app`) is optional (`buildml[dashboard]`).
It runs a local FastAPI server and displays findings, workflow checks,
concept explanations, and analysis by topic. The static report
(`html_format="research"`) presents the same underlying analysis in a
printable layout.

Selections on readiness cards remain in the open browser tab. Refreshing clears them. BuildML
does not write those marks to the Session, history, disk, or a saved
dataset copy.

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
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
)

report = session.eda(partition="train", include_plots=False)
for finding in report.findings[:10]:
    print(finding.severity, finding.title)
for rec in getattr(report, "recommendations", [])[:5] or []:
    print("rec:", rec)
```

Recommendations name Session operations. They do not execute them.
Use training rows for exploratory decisions about features or models. The
separate drift section still compares full train and test distributions;
the report discloses both scopes. The default ``partition="all"`` includes
held-out rows when a split exists.

---

## Use case: offline HTML (studio vs research)

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
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
)

# Requires: pip install "buildml[viz]"
# Offline dashboard snapshot (SPA assets embedded when available)
session.eda(export_html="artifacts/eda_studio.html", html_format="studio")

# Static sheet (needs buildml[viz] for plots)
session.eda(
    include_plots=True,
    export_html="artifacts/eda_research.html",
    html_format="research",
    export_figures="artifacts/eda-figures",
)
```

`html_format="research"` produces a static report with summary metrics,
findings, assumptions, recommended Session calls, figures, methods, and
unavailable analyses. Interactive readiness cards and concept lessons are
available in the studio layout. Exported HTML includes the styles needed
for offline viewing.

From a BuildML source checkout, generate a preview with synthetic data:

```bash
python scripts/generate_static_eda_preview.py
# writes .buildml-artifacts/static_eda_cockpit.html
```

---

## Use case: live local dashboard

Open the printed URL in your browser. The server stays available until you
press Enter in the terminal.

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
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
)

# Requires: pip install "buildml[dashboard]"
handle = session.eda_app(port=8765, open_browser=False)
print(handle.url)
try:
    input("Press Enter to stop the dashboard. ")
finally:
    handle.stop()
```

From a BuildML source checkout, launch a dashboard with synthetic data:

```bash
python scripts/launch_synthetic_eda_studio.py
```

| Board | Role |
| --- | --- |
| Command cockpit | Summary metrics, findings, assumptions, recommended operations, figures, methods, and unavailable analyses |
| Readiness gates | Workflow checks grouped by stage, with explanations; selections are stored only in the browser tab |
| Concept academy | Searchable lessons with report values where available; unavailable values are marked N/A |
| Domain boards | Quality, features, relationships, multivariate, target, outliers, visuals |

The app header exports an offline HTML report with the same layout,
including the readiness checklist and concept lessons. CSV and PDF exports
are available through the app API for automation. The static report header
also provides an HTML export.

If the port is busy, pass another port.

The dashboard descriptions use the findings in your report. Reference
labels distinguish sources cited by a finding from additional reading.

---

## Teaching methods: explain, learn, workflow, and walkthrough

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
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
)

before = session.explain("feature_importance", moment="before")
print(before.prerequisite_chain, before.risks)

for step in session.workflow():
    if step.status == "blocked":
        print(step.operation, step.reasons or step.blockers)

preview = session.dry_run(["impute", "scale", "fit"])
summary = session.summarize_history()
print(summary.unresolved_risks)

walkthrough = session.walkthrough(export_html="artifacts/workflow.html")
```

`available` means the operation's API prerequisites are satisfied. Choose
operations according to your data and analysis goal. `explain(..., moment="after")` joins catalog text to the latest
recorded call. `dry_run` does not append history.

### Reading levels

Every explanation is written at three levels. `beginner` is the default
and assumes no prior machine-learning vocabulary.

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
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
)

primer = session.explain("feature_importance").beginner
print(primer.plain_summary)          # what this is, in ordinary words
print(primer.analogy)                # the intuition
primer.steps                         # what happens, in order
primer.prerequisites_in_plain_words  # what must be true first, and how to get there
primer.key_parameters                # parameter meanings, effects, and typical values
primer.common_pitfalls               # how this goes wrong
primer.glossary                      # the jargon this answer used, defined
primer.mini_example                  # an example of the operation

session.explain("feature_importance", level="advanced")  # a more detailed technical explanation
```

The reading level changes the explanation's detail: assumptions,
leakage risks, and failure modes are present at every level. `advanced`
drops the analogy and the in-line glossary and widens the parameter and
pitfall lists.

### `learn`: the concept behind the call

`explain` describes an operation in the current Session. `learn` provides
background concepts and recommended reading order. It accepts a concept key, an
operation name, or a related term. Spacing and hyphenation are normalized.

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
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
)

session.learn()                       # foundation concepts, in reading order
brief = session.learn("leakage")      # a term resolves to the concept teaching it

brief.concept.plain_summary           # the idea from scratch
brief.concept.misconceptions          # what people wrongly believe, and the correction
brief.concept.check_yourself          # questions to check understanding
[note.key for note in brief.read_first]  # prerequisites, if any
[note.key for note in brief.read_next]   # suggested follow-up concepts
brief.related_operations              # the BuildML calls that apply it

session.learn("split")                # an operation name returns its primer
session.learn("cross-validation", level="intermediate")
```

Concept notes, the glossary, and operation primers are the same objects
the walkthrough, the local dashboard, and the AI operator's
`explain_operation` / `learn_concept` tools read from. All of it is
static teaching material: it describes ideas and BuildML's contract, and
does not calculate new dataset statistics.

---

## Evaluation and diagnostic HTML

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
    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
)

# Requires: pip install "buildml[viz]"
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
| Acting on recommendations blindly | Review the evidence and choose appropriate operations before applying them |
| Confusing AI advisor with EDA | AI is optional (`buildml[ai]`); EDA/App work offline |

---

## Related

- [AI operator safety](ai-operator-safety.md)
- [Classical end-to-end](classical-end-to-end.md)
- [Artifacts](artifacts-checkpoints-bundles.md)
