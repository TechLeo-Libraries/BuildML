# AI operator quickstart

> **Install first (GitHub):** PyPI `buildml` is still legacy 1.x and does **not**
> install Session 2.x. Install 2.x from GitHub (or an editable checkout), then
> the AI extra. See [installation](../docs/installation.rst).

Optional LLM-assisted workflow guidance on the same `Session` as classical ML,
Torch, and RAG: roles, splits, history, and explain. Install the AI extra;
core `import buildml` never requires it.

**Go deeper:** [AI operator safety](ai-operator-safety.md) ·
[AI tools & operator patterns](ai-tools-operator-patterns.md).

```bash
# After a GitHub / editable 2.x install:
pip install "buildml[ai]"
# alias: pip install "buildml[llm]"
# or: pip install "buildml[ai] @ git+https://github.com/TechLeo-Libraries/BuildML.git"
```

Classical `Session.fit`, Torch `*_torch`, and RAG `rag_*` stay unchanged. AI
methods use the `ai_*` prefix and store results in `session.ai.result` /
`session.ai.transcript`.

This alpha defaults to **advisor → plan → confirmed execute**. Optional
`session.ai.run_autonomous` is explicit operator automation under hard caps (allowlist,
max steps, blocked sample egress, transcript audit): not unconstrained agency.

## Bring your own API key

BuildML does not ship, proxy, or embed API keys. You supply your own provider
credentials via environment variable or session config:

```bash
export BUILDML_OPENAI_API_KEY="sk-your-key-here"
```

Or in code:

```python
from buildml import Session

session = Session()
session.ai.configure(api_key="sk-your-key-here")
```

Keys are never logged, never persisted in transcripts or checkpoints, and
never echoed in error messages.

## Privacy defaults: STATS_ONLY

Before any data is sent to an external provider, BuildML shows exactly what
will leave your machine. The default egress level is `STATS_ONLY`: aggregates
and column names, never raw row values.

```python
import pandas as pd

from buildml import Session

frame = pd.DataFrame({
    "age": [25, 30, 35, 40, 29, 33, 52, 47],
    "income": [40, 55, 60, 80, 50, 70, 90, 65],
    "approved": [0, 1, 0, 1, 0, 1, 1, 0],
})

session = Session.ingest(frame)
session.ai.configure(provider="openai")

# Preview what will be sent before any API call
manifest = session.ai.egress_preview()
print(manifest.level)           # EgressLevel.STATS_ONLY (default)
print(manifest.columns_sent)    # column names only
print(manifest.rows_sent)       # 0 (no raw rows at STATS_ONLY)
```

Egress levels:

| Level | What is sent | Confirm policy |
|---|---|---|
| `SCHEMA_ONLY` | Column names, dtypes, row count | Auto |
| `STATS_ONLY` | Aggregates, percentiles, cardinality (default) | Auto |
| `REDACTED_SAMPLE` | N rows with PII columns masked | Confirm required |
| `FULL_SAMPLE` | Raw rows (explicit opt-in only) | Always confirm |

Escalate egress only when needed:

```python
result = session.ai.advisor(
    "What patterns do you see in the data?",
    level="redacted_sample",
    confirm=True,  # required for sample egress
)
```

## Dry run: inspect the full prompt payload

`session.ai.dry_run` returns the complete prompt payload without calling the provider.
Inspect exactly what would be sent:

```python
payload = session.ai.dry_run("Suggest next preprocessing steps")
print(payload["messages"])       # system + user messages
print(payload["tools"])          # available tool schemas
print(payload["egress_manifest"])  # egress details
# provider.calls is empty: no API request made
```

## Advisor: read-only Q&A

`session.ai.advisor` answers questions about your data and workflow without modifying
Session state. It uses the explain catalog and current Session context:

```python
session.set_roles({"age": "feature", "income": "feature", "approved": "target"})

result = session.ai.advisor("What preprocessing steps should I consider?")
print(result.answer)
print(result.egress_manifest)  # confirms what was sent
```

The advisor cannot execute operations. It returns suggestions, not actions.

## Plan: structured next steps

`session.ai.plan` proposes a sequence of operations based on your goal and current
Session state:

```python
plan = session.ai.plan("Build a classification model with proper preprocessing")
print(plan.goal)
for step in plan.steps:
    print(f"  {step.operation}: {step.description}")
    print(f"    Rationale: {step.rationale}")
```

Plans are proposals. Nothing executes until you explicitly confirm.

## Execute: propose → confirm → execute

`session.ai.execute` follows a two-phase pattern:

1. **Propose:** Returns what the tool will do, marked as requiring confirmation
2. **Execute:** Only runs when you pass `confirm=True`

```python
# Phase 1: proposal (no state change)
proposal = session.ai.execute(
    "set_roles",
    {"mapping": {"age": "feature", "income": "feature", "approved": "target"}},
)
print(proposal.requires_confirmation)  # True
print(proposal.tool_name)              # "set_roles"
print(proposal.arguments)              # the mapping

# Phase 2: confirmed execution (state changes)
result = session.ai.execute(
    "set_roles",
    {"mapping": {"age": "feature", "income": "feature", "approved": "target"}},
    confirm=True,
)
print(result.executed)  # True
print(session.dataset.roles)  # roles now assigned
```

Read-only tools (describe, explain, workflow_status) do not require
confirmation. Write tools (set_roles, impute, split, fit) always require
`confirm=True`. Destructive tools (drop_columns) always require confirmation
and cannot be auto-approved.

## Tool allowlist

The operator works through a typed tool registry. Tools not in the registry
are rejected:

```python
from buildml.ai import build_default_registry

registry = build_default_registry()
print(list(registry.keys()))  # available tools
```

Available tools include:
- **Read-only:** `describe_dataset`, `explain_operation`, `learn_concept`,
  `workflow_status`, `eda_summary`, `head`, `session.ai.status`, `evaluate`,
  `walkthrough`
- **Write (confirm required):** `set_roles`, `split`, `impute`, `encode`,
  `scale`, `fit`, `checkpoint_save`
- **Destructive (always confirm):** `drop_columns`

The operator cannot bypass this allowlist. Unknown tools raise `ValidationError`.

## Transcripts

Transcripts record the conversation, tool calls, egress manifests, and
confirmations. API keys and raw data (unless `FULL_SAMPLE` opt-in) are never
persisted:

```python
session.ai.advisor("Describe the data")
session.ai.execute("set_roles", {"mapping": {...}}, confirm=True)

# Save transcript (secrets redacted)
session.ai.save_transcript("artifacts/transcript.json")

# Load in another session
session2 = Session.ingest(frame)
session2.ai.load_transcript("artifacts/transcript.json")
```

Transcripts are separate from Session checkpoints and DL/RAG bundles.

## Budget limits

Configure token and cost budgets to prevent runaway usage:

```python
session.ai.configure(
    provider="openai",
    max_tokens=10000,
    max_cost_usd=5.0,
    max_iterations=10,  # default
)

status = session.ai.status()
print(status["budget"])  # tokens_used, cost_used_usd, limits
```

## MockProvider for offline / CI

Tests and offline workflows use `MockProvider`:

```python
session.ai.configure(provider="mock")
result = session.ai.advisor("Test question")
# Works offline; returns canned responses
```

CI runs with `MockProvider` only: real API keys are never required for tests.

## Explicit autonomy (opt-in)

Default AI stays propose→confirm→execute. For allowlisted automation with hard
caps (max steps, tool allowlist, blocked sample egress, transcript audit):

```python
session.ai.configure(provider="mock", egress_level="stats_only")
result = session.ai.run_autonomous(
    "split the data and report workflow status",
    confirm_autonomy=True,  # required
    max_steps=5,
)
print(result.completed_steps, result.stop_reason, result.residual_risks)
```

This is operator automation inside an allowlist: not unconstrained agency.

## Explain catalog

AI operations are documented in the explain catalog:

```python
before = session.explain("ai_advisor", moment="before")
print(before.operation, before.prerequisites)
print(before.leakage_risks)  # egress privacy warnings
```

## Security warnings

**Prompt injection:** The operator treats all data (column names, cell values,
user prompts, RAG chunks) as untrusted. Injection patterns are detected and
flagged. The tool registry is the trust boundary: the operator cannot execute
tools not in the allowlist.

**Never put secrets in prompts.** API keys, passwords, and sensitive values
should never appear in user prompts, column data, or any text sent to the
provider. The provider sees whatever egress payload you approve.

**Advice must be verified.** The operator provides suggestions based on
evidence, but it can be wrong. Always verify recommendations before acting on
them. The operator is not a substitute for domain expertise.

## Artifacts

| Artifact | Schema | Contains | Does not contain |
|---|---|---|---|
| Session checkpoint | existing formats | data, roles, splits, history | AI transcript, API keys |
| Torch trainer bundle | `buildml.torch_bundle.v1` | weights, optimizer, config | AI transcript |
| RAG bundle | `buildml.rag_bundle.v1` | embeddings, index, chunk config | AI transcript |
| AI transcript | `buildml.ai.transcript.v1` | conversation, tool calls, egress manifests | API keys, raw data (default) |

## Known limits (honest)

- **Bring-your-own API key.** BuildML never ships, proxies, or embeds keys.
- **Default egress is STATS_ONLY.** Raw rows require explicit opt-in and
  confirmation.
- **Propose → confirm → execute by default.** `session.ai.run_autonomous` is opt-in
  allowlisted automation with residual risk: review transcripts.
- **Tool registry is the trust boundary.** The operator cannot execute
  arbitrary code or tools not in the registry.
- **Transcript ≠ checkpoint.** AI conversation history is stored separately
  from Session data and model artifacts.
- **Not a replacement for `eda_app()`.** The operator supplements, not
  replaces, the explain catalog and structured results.
- **Not fine-tuning LLMs.** The operator guides BuildML workflows; it does not
  train or fine-tune language models.
- **Advice must be verified.** Evidence-bound recommendations are not infallible.
- **Provider sees approved egress.** BuildML cannot protect against a
  compromised provider.

See [glossary](glossary.md).
