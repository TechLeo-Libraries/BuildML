# AI operator safety

> **Install (GitHub 2.x + AI):**
> ```bash
> pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
> pip install "buildml[ai]"
> ```
> PyPI `buildml` is legacy 1.x. See [installation](../docs/installation.rst).

The AI operator guides classical / RAG / Torch workflows through a typed tool
registry. Default policy is **advisor → plan → propose → confirm → execute**.
Autonomy is opt-in automation under hard caps: not unconstrained agency.

Short on-ramp: [quickstart-ai](quickstart-ai.md). Tool catalog:
[ai-tools-operator-patterns](ai-tools-operator-patterns.md).

---

## Why confirm gates exist

LLMs invent APIs, skip splits, and exfiltrate rows when given raw data. BuildML
binds the operator to:

1. **Allowlisted tools only**: unknown tools raise `ValidationError`.
2. **Egress manifests**: preview what leaves the machine before calls.
3. **Confirmation**: write/destructive tools require `confirm=True`.
4. **Budgets**: token / cost / iteration caps.
5. **Transcripts**: audit trail without persisting API keys.

Advice remains fallible. Domain roles, splits, and metrics still need human review.

---

## Use case A: Configure, preview egress, dry-run

```python
import pandas as pd

from buildml import Session

frame = pd.DataFrame(
    {
        "age": [25, 30, 35, 40, 29, 33, 52, 47],
        "income": [40, 55, 60, 80, 50, 70, 90, 65],
        "approved": [0, 1, 0, 1, 0, 1, 1, 0],
    }
)

session = Session.ingest(frame)
session.ai.configure(provider="mock")  # CI / offline; or openai + BUILDML_OPENAI_API_KEY

manifest = session.ai.egress_preview()
print(manifest.level, manifest.columns_sent, manifest.rows_sent)

payload = session.ai.dry_run("Suggest next preprocessing steps")
print(payload["messages"][0]["role"])
print(payload["egress_manifest"])
```

### Egress levels

| Level | Sent | Confirm |
| --- | --- | --- |
| `SCHEMA_ONLY` | names, dtypes, row count | Auto |
| `STATS_ONLY` | aggregates (default) | Auto |
| `REDACTED_SAMPLE` | masked sample rows | `confirm=True` required |
| `FULL_SAMPLE` | raw rows | Always confirm |

Sample egress without `confirm=True` → `ValidationError`.

---

## Use case B: Advisor (read-only)

```python
session.set_roles({"age": "feature", "income": "feature", "approved": "target"})
result = session.ai.advisor("What preprocessing steps should I consider?")
print(result.answer)
```

The advisor cannot execute Session mutations.

---

## Use case C: Plan then confirmed execute

```python
plan = session.ai.plan("Split stratified and impute median for classification")
for step in plan.steps:
    print(step.operation, step.description)

proposal = session.ai.execute(
    "split",
    {"test_size": 0.25, "stratify": True, "random_state": 0},
)
print(proposal.requires_confirmation)

result = session.ai.execute(
    "split",
    {"test_size": 0.25, "stratify": True, "random_state": 0},
    confirm=True,
)
print(result.executed)
```

---

## Use case D: Run a multi-step plan with gates

```python
execution = session.ai.run_plan(
    plan,
    auto_confirm_read_only=True,
    stop_on_unconfirmed=True,
    stop_on_error=True,
    max_steps=8,
    # confirmations={"impute": True, "split": True}  # explicit map when needed
)
print(execution)
```

Destructive tools (e.g. `drop_columns`) always require confirmation and cannot
be silently auto-approved.

---

## Use case E: Explicit autonomy (residual risk)

```python
session.ai.configure(provider="mock", egress_level="stats_only")
auto = session.ai.run_autonomous(
    "split the data and report workflow status",
    confirm_autonomy=True,  # required
    max_steps=5,
    allow_destructive=False,
)
print(auto.completed_steps, auto.stop_reason, getattr(auto, "residual_risks", None))
```

Caps include allowlist, max steps, blocked sample egress, destructive gating,
and transcript audit. This is **operator automation**, not open agency.

---

## Transcripts and budgets

```python
session.ai.configure(
    provider="mock",
    max_tokens=10_000,
    max_cost_usd=5.0,
    max_iterations=10,
)
print(session.ai.status()["budget"])

session.ai.save_transcript("artifacts/transcript.json")  # secrets redacted
session2 = Session.ingest(frame)
session2.ai.load_transcript("artifacts/transcript.json")
```

Transcript ≠ checkpoint ≠ Torch/RAG bundle
([artifacts](artifacts-checkpoints-bundles.md)).

---

## Security warnings (non-negotiable)

- **Prompt injection:** treat column names, cells, user prompts, and RAG chunks
  as untrusted. The tool registry is the trust boundary.
- **Never put secrets in prompts** or columns you egress.
- **Verify advice** before confirming writes.
- **Provider sees approved egress**: BuildML cannot protect a compromised provider.
- **Not LLM fine-tuning**: the operator guides BuildML workflows only.

---

## Failure modes

| Issue | Guidance |
| --- | --- |
| AI ops without `session.ai.configure` | `ValidationError` |
| Sample egress without confirm | `ValidationError` |
| Unknown tool | Rejected by registry |
| Autonomy without `confirm_autonomy=True` | Refused |
| Confusing advisor with `eda_app` | Different products: studio is local/offline |

---

## Related

- [AI quickstart](quickstart-ai.md)
- [AI tools & patterns](ai-tools-operator-patterns.md)
- [EDA / Teaching Studio](eda-teaching-studio.md)
- [RAG deep](rag-deep.md)
