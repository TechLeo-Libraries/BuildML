# ruff: noqa: E501
"""Ai concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

AI_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="ai-egress-privacy",
            title="AI Egress Privacy",
            summary=(
                "User-controlled data egress before any information leaves the machine to an external LLM provider."
            ),
            definition=(
                "Egress privacy is the set of controls that determine what data (schema, statistics, samples, raw rows) "
                "leaves the user's machine when calling an external LLM API. BuildML provides four egress levels: "
                "SCHEMA_ONLY (column names/types), STATS_ONLY (aggregates), REDACTED_SAMPLE (masked rows), and "
                "FULL_SAMPLE (raw rows with explicit opt-in)."
            ),
            intuition=(
                "Think of egress levels as airport security zones. SCHEMA_ONLY shows only the boarding pass (column names). "
                "STATS_ONLY adds aggregate flight statistics without passenger details. REDACTED_SAMPLE masks sensitive "
                "passenger info. FULL_SAMPLE shares everything:use only when necessary and after inspection."
            ),
            formal_idea=(
                "The egress manifest is a typed record of what will be (or was) sent: columns, row count, estimated tokens, "
                "and warnings about PII-like columns. session.ai.egress_preview returns the manifest without making an API call."
            ),
            why_it_matters=(
                "External LLM providers see whatever payload the user approves.",
                "Sensitive column names, statistics, or raw values may leak if egress is not controlled.",
                "Regulatory and security requirements often restrict what data can leave internal systems.",
            ),
            how_buildml_uses=(
                "session.ai.configure sets the default egress level (STATS_ONLY by default).",
                "session.ai.egress_preview shows the manifest before any call.",
                "session.ai.dry_run returns the full prompt payload for local inspection.",
                "Column allow/deny lists filter what columns appear in egress.",
                "Transcripts record egress manifests, not raw data (unless FULL_SAMPLE).",
            ),
            interpretation_rules=(
                "STATS_ONLY sends aggregates and schema, never raw row values.",
                "FULL_SAMPLE requires explicit opt-in and sends raw data.",
                "Always inspect session.ai.egress_preview before first AI call on sensitive data.",
            ),
            assumptions=(
                "The user reviews egress manifests before approving data-heavy calls.",
                "Column allow/deny lists are configured appropriately for the sensitivity level.",
            ),
            failure_modes=(
                "Using FULL_SAMPLE without reviewing what data leaves the machine.",
                "Column names themselves may be sensitive even in SCHEMA_ONLY mode.",
            ),
            anti_patterns=(
                "Auto-approving FULL_SAMPLE egress without inspection.",
                "Assuming STATS_ONLY protects against all data leakage (aggregates can still reveal patterns).",
            ),
            worked_example_pattern=(
                "session.ai.configure → session.ai.egress_preview → review manifest → session.ai.advisor (if manifest acceptable).",
            ),
            related_concepts=("leakage-boundary", "ai-tool-trust"),
        ),
        _note(
            key="ai-tool-trust",
            title="AI Tool Trust",
            summary=(
                "Tools are allowlisted, mapped to Session methods, and gated by a propose-confirm-execute flow."
            ),
            definition=(
                "The AI operator can only execute tools that are explicitly registered in the ToolRegistry. Each tool "
                "maps to a Session method, has a confirmation policy (auto, confirm, always_confirm), and cannot bypass "
                "existing Session guards (leakage, validation). Destructive operations always require confirmation."
            ),
            intuition=(
                "The tool registry is like a hotel safe with an approved guest list. The AI can suggest using items "
                "from the safe, but can only access what's on the list, and certain items require the owner's signature "
                "before they can be taken out."
            ),
            formal_idea=(
                "ToolSpec defines name, parameters, confirm_policy, read_only, and destructive flags. "
                "The executor validates each ToolCall against the registry, refuses unlisted tools, and requires "
                "confirmation for write operations. Maximum iteration limits prevent runaway loops."
            ),
            why_it_matters=(
                "LLMs may hallucinate tool names or attempt operations outside the allowed scope.",
                "Propose-confirm-execute prevents accidental state changes from AI suggestions.",
                "Leakage guards must fire regardless of whether human or AI initiated the operation.",
            ),
            how_buildml_uses=(
                "ToolRegistry defines the M1 allowlist: describe_dataset, explain_operation, workflow_status, etc.",
                "session.ai.execute validates tool calls against the registry and requires confirmation for writes.",
                "Destructive tools (drop, delete) always require explicit confirmation.",
                "Read-only tools (describe, explain) may auto-confirm.",
            ),
            interpretation_rules=(
                "Tools not in the registry are rejected with a named error.",
                "Confirmation status is recorded in the transcript.",
                "Write operations modify Session state; read operations do not.",
            ),
            assumptions=(
                "The tool registry is conservative and well-maintained.",
                "Users review proposals before confirming write operations.",
            ),
            failure_modes=(
                "Auto-confirming destructive operations without review.",
                "Expanding the tool registry without security review.",
            ),
            anti_patterns=(
                "Bypassing the tool registry with eval/exec.",
                "Trusting AI tool suggestions without verifying prerequisites.",
            ),
            worked_example_pattern=(
                "session.ai.execute('set_roles', {'mapping': {...}}) → review proposal → session.ai.execute(..., confirm=True).",
            ),
            related_concepts=("ai-prompt-injection", "ai-egress-privacy", "leakage-boundary"),
        ),
        _note(
            key="ai-prompt-injection",
            title="AI Prompt Injection Hardening",
            summary=(
                "Untrusted data (column names, cell values, user text) is marked and separated from instructions."
            ),
            definition=(
                "Prompt injection is an attack where adversarial text in data is interpreted as instructions by the LLM. "
                "BuildML hardens against this by: marking untrusted data with boundary tags, using system prompts that "
                "instruct the model to treat data as data only, validating tool calls against the registry, and refusing "
                "arbitrary code execution."
            ),
            intuition=(
                "Imagine a mail room where incoming packages are labeled 'EXTERNAL:DO NOT OPEN WITHOUT INSPECTION'. "
                "Even if a package label says 'URGENT: Give to CEO immediately', the mailroom follows procedure. "
                "Data markers work the same way: they tell the LLM that this content is cargo, not commands."
            ),
            formal_idea=(
                "Untrusted data is wrapped in [UNTRUSTED DATA] markers. Tool results are wrapped in [TOOL RESULT - DATA ONLY] "
                "markers. The system prompt explicitly states that data is not instructions. Injection patterns "
                "(e.g., 'ignore previous instructions') are detected and escaped in security tests."
            ),
            why_it_matters=(
                "Malicious column names like '; DROP TABLE users; --' should not execute.",
                "Cell values containing 'Ignore previous instructions' should not change AI behavior.",
                "User prompts attempting tool registry bypass should be rejected.",
            ),
            how_buildml_uses=(
                "mark_untrusted_data wraps data with source markers before sending to the LLM.",
                "sanitize_tool_result wraps tool outputs before feeding back.",
                "detect_injection_attempt scans text for known injection patterns.",
                "CI injection tests verify boundaries hold with adversarial fixtures.",
            ),
            interpretation_rules=(
                "Injection detection is a warning, not a block; humans review flagged content.",
                "The tool registry, not the LLM, controls what operations execute.",
                "eval/exec are never allowed regardless of prompt content.",
            ),
            assumptions=(
                "The LLM respects boundary markers (not guaranteed, but improves safety).",
                "The tool registry is the authoritative gate for execution.",
            ),
            failure_modes=(
                "Novel injection patterns not in the detection list.",
                "LLMs that ignore boundary instructions (mitigated by tool registry).",
            ),
            anti_patterns=(
                "Trusting LLM-generated code without review.",
                "Disabling injection detection for convenience.",
            ),
            worked_example_pattern=(
                "Column name: 'ignore_previous; drop_table' → detected as suspicious → wrapped as data → tool registry rejects unauthorized calls.",
            ),
            related_concepts=("ai-tool-trust", "ai-egress-privacy"),
        ),
    )
}

