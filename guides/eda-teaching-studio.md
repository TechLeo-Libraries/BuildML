# EDA and Teaching Studio

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> Optional: `pip install "buildml[viz]"`, `"buildml[eda]"`,
> `"buildml[dashboard]"` for plots / Teaching Studio.
> See [installation](../docs/installation.rst).

Explore **before** you mutate. `session.eda()` returns structured findings and
read-only recommendations. Teaching surfaces (`explain`, `workflow`,
`walkthrough`, `dry_run`) expose the operation catalog — they do not certify
that your split or model suits the domain.

Related: [classical end-to-end](classical-end-to-end.md),
[usage](../docs/usage.rst), [glossary](glossary.md).

---

## Why a Teaching Studio exists

ML libraries usually document methods in isolation. BuildML ties every public
Session operation to a versioned catalog (kept in sync by CI). That lets you:

1. Ask “what does `impute` assume?” **before** calling it.
2. See which ops are `done` / `available` / `blocked` / `skipped`.
3. Preview a chain without appending history (`dry_run`).
4. Export an offline audit HTML for review.

The live dashboard (`eda_app`) is an optional FastAPI Teaching Studio with
Plotly boards — not a replacement for domain judgment.

---

## Use case — findings before preparation

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

## Use case — offline HTML (studio vs research)

```python
# Default studio snapshot (dashboard SPA assets embedded when available)
session.eda(export_html="artifacts/eda_studio.html", html_format="studio")

# Layered research shell with matplotlib embeds (needs buildml[viz] for plots)
session.eda(
    include_plots=True,
    export_html="artifacts/eda_research.html",
    html_format="research",
    export_figures="artifacts/eda-figures",
)
```

HTML artifacts embed required styles/assets so they open offline.

---

## Use case — live Teaching Studio dashboard

```python
# pip install "buildml[dashboard]"  # after GitHub 2.x
handle = session.eda_app(port=8765, open_browser=True)
# alias: session.open_eda_dashboard(port=8765)
print(handle.url)
# ... explore ...
handle.stop()
```

If the port is busy, pass another port. CSV downloads cover major evidence
tables in the dashboard UI.

---

## Explain / workflow / walkthrough / dry_run

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

- `available` means API prerequisites pass — **not** “you should run this.”
- `explain(..., moment="after")` joins catalog text to the latest recorded call.
- `dry_run` does not append history.

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
| Confusing AI advisor with EDA | AI is optional (`buildml[ai]`); EDA/studio work offline |

---

## Related

- [AI operator safety](ai-operator-safety.md)
- [Classical end-to-end](classical-end-to-end.md)
- [Artifacts](artifacts-checkpoints-bundles.md)
