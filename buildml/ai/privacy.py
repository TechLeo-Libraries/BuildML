"""Decide what data leaves the machine, and record exactly what did.

Sending data to a hosted language model is a disclosure to a third party. It
cannot be undone, and retention policies are outside your control. This module
is the boundary where that decision gets made explicitly rather than by
accident.

Two things happen here for every call. A **payload** is built at the configured
:class:`~buildml.ai.types.EgressLevel` — schema, statistics, a redacted sample,
or raw rows. And an :class:`EgressManifest` is produced alongside it, naming
every column sent, every column withheld, every rename applied, and how many
rows went. The manifest travels with the result and into the transcript, so
"what did we send them" always has an answer.

Redaction here is defence in depth, not a guarantee. :func:`detect_pii_columns`
matches column *names* against common patterns — it catches ``customer_email``
and misses ``field_7`` holding the same addresses. Treat it as a reminder to
configure ``deny_columns``, never as a substitute.

Notes
-----
**The lowest level that works is the right level.** For planning and advice, a
model reasoning from schema and statistics is usually as useful as one reading
rows, at a fraction of the exposure.

See Also
--------
buildml.ai.types.EgressLevel : The four levels.
buildml.ai.security : Redaction of secrets in free text.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from buildml.ai.types import EgressLevel


@dataclass(slots=True)
class EgressConfig:
    """The rules governing what may be sent to a provider.

    Attributes
    ----------
    level:
        How much detail is permitted. Defaults to
        :attr:`~buildml.ai.types.EgressLevel.STATS_ONLY` — aggregates but no
        rows.
    allow_columns:
        An explicit allowlist. When set, **only** these columns are considered,
        and everything else is denied. The safer of the two lists: a column
        added to your data later is excluded by default rather than included.
    deny_columns:
        A denylist, used when ``allow_columns`` is ``None``. Everything not
        named is sent, so a newly added sensitive column slips through until
        you notice.
    rename_columns:
        Original to replacement names. Useful when the name discloses more than
        the values.
    strip_headers:
        Replace every column name with ``col_0``, ``col_1``, and so on. Maximum
        name privacy, at the cost of the model no longer understanding what
        anything means. Overrides ``rename_columns``.
    sample_rows:
        How many rows to include at the sample levels. Ignored at
        ``SCHEMA_ONLY`` and ``STATS_ONLY``.
    redact_patterns:
        Regular expressions replaced with ``[REDACTED]`` inside string values.
        For content that is sensitive regardless of which column holds it.

    Notes
    -----
    **Prefer ``allow_columns`` over ``deny_columns``.** A denylist is correct
    only for the schema you wrote it against; an allowlist stays correct as the
    data changes.

    **The level caps everything else.** No column configuration causes rows to
    be sent at ``SCHEMA_ONLY``.

    Examples
    --------
    Allow three columns, at statistics only::

        config = EgressConfig(
            level=EgressLevel.STATS_ONLY,
            allow_columns=("age", "region", "outcome"),
        )

    See Also
    --------
    build_egress_payload : Applying this.
    EgressManifest : The record of what it produced.
    """

    level: EgressLevel = EgressLevel.STATS_ONLY
    allow_columns: tuple[str, ...] | None = None
    deny_columns: tuple[str, ...] = ()
    rename_columns: dict[str, str] = field(default_factory=dict)
    strip_headers: bool = False
    sample_rows: int = 5
    redact_patterns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the egress rules as JSON-safe values.

        Records the policy that was in force, which is the other half of an
        audit trail — the manifest says what was sent, this says what was
        allowed.

        Returns
        -------
        dict
            Level as a string, the allow and deny lists, renames, header
            stripping, sample size, and redaction patterns.
        """
        return {
            "level": self.level.value,
            "allow_columns": list(self.allow_columns) if self.allow_columns else None,
            "deny_columns": list(self.deny_columns),
            "rename_columns": dict(self.rename_columns),
            "strip_headers": self.strip_headers,
            "sample_rows": self.sample_rows,
            "redact_patterns": list(self.redact_patterns),
        }


@dataclass(frozen=True, slots=True)
class EgressManifest:
    """The record of what actually left the machine on one call.

    Immutable, because a record that can be edited after the fact is not a
    record. Produced alongside every payload and carried into the result and
    the transcript.

    Attributes
    ----------
    level:
        Which level was applied.
    columns_sent:
        The names as the provider saw them — after renaming or header
        stripping, so this reflects the wire, not your schema.
    columns_denied:
        What was withheld. **Worth reading as carefully as ``columns_sent``**;
        a sensitive column absent from this list was not withheld.
    columns_renamed:
        Original to replacement, so the sent names can be mapped back.
    rows_sent:
        How many rows went. Zero at ``SCHEMA_ONLY`` and ``STATS_ONLY``.
    estimated_tokens:
        A rough size estimate, useful for cost and context-limit checks.
        Approximate — see :func:`build_egress_payload`.
    warnings:
        Columns whose names matched a personal-data pattern, and what was done
        about them.

    Notes
    -----
    **The manifest describes the payload, not the whole request.** Your prompt
    text is sent too, and is not accounted for here. Free text is where secrets
    most often escape; see :mod:`buildml.ai.security`.

    See Also
    --------
    EgressConfig : The rules that produced this.
    buildml.ai.results.TranscriptEntry : Where it is preserved.
    """

    level: EgressLevel
    columns_sent: tuple[str, ...]
    columns_denied: tuple[str, ...]
    columns_renamed: dict[str, str]
    rows_sent: int
    estimated_tokens: int | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the disclosure record as JSON-safe values.

        The form written into transcripts, and the thing to look at when
        someone asks what a provider received.

        Returns
        -------
        dict
            Level as a string, columns sent, columns denied, renames, row
            count, estimated tokens, and warnings.
        """
        return {
            "level": self.level.value,
            "columns_sent": list(self.columns_sent),
            "columns_denied": list(self.columns_denied),
            "columns_renamed": dict(self.columns_renamed),
            "rows_sent": self.rows_sent,
            "estimated_tokens": self.estimated_tokens,
            "warnings": list(self.warnings),
        }


_PII_PATTERNS = (
    re.compile(r"(?i)(email|e[-_]?mail)", re.IGNORECASE),
    re.compile(r"(?i)(phone|mobile|cell)", re.IGNORECASE),
    re.compile(r"(?i)(ssn|social.*security)", re.IGNORECASE),
    re.compile(r"(?i)(address|street|zip|postal)", re.IGNORECASE),
    re.compile(r"(?i)(credit.*card|card.*number)", re.IGNORECASE),
    re.compile(r"(?i)(password|passwd|pwd)", re.IGNORECASE),
    re.compile(r"(?i)(birth.*date|dob|date.*birth)", re.IGNORECASE),
    re.compile(r"(?i)(first.*name|last.*name|full.*name)", re.IGNORECASE),
)


def detect_pii_columns(columns: list[str]) -> list[str]:
    """Flag column names that look like they hold personal data.

    Matches names against patterns for email, phone, national identifiers,
    addresses, card numbers, passwords, dates of birth, and personal names. A
    prompt to configure the egress rules, not a filter.

    Parameters
    ----------
    columns:
        Column names to check.

    Returns
    -------
    list of str
        Names that matched, in input order.

    Notes
    -----
    **Names only. Contents are never examined.** A column called ``notes``
    holding email addresses does not match; a column called ``email_opt_in``
    holding booleans does. Both outcomes are wrong in the way name-matching is
    always wrong.

    **A clean result is not clearance.** It means no name tripped a pattern,
    which is a statement about your naming conventions rather than about your
    data.

    Examples
    --------
    >>> detect_pii_columns(["user_email", "signup_date", "phone_number"])
    ['user_email', 'phone_number']

    See Also
    --------
    EgressConfig : Where you act on the finding.
    """
    suspicious = []
    for col in columns:
        for pattern in _PII_PATTERNS:
            if pattern.search(col):
                suspicious.append(col)
                break
    return suspicious


def filter_columns(
    columns: list[str],
    *,
    allow: tuple[str, ...] | None = None,
    deny: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """Split column names into what may be sent and what may not.

    When an allowlist is given it wins outright and the denylist is ignored:
    anything not named is denied. Otherwise the denylist applies and everything
    else passes.

    Parameters
    ----------
    columns:
        The names to partition.
    allow:
        The allowlist. ``None`` means no allowlist, not an empty one — an empty
        tuple denies everything.
    deny:
        The denylist, used only when ``allow`` is ``None``.

    Returns
    -------
    tuple of (list of str, list of str)
        Allowed names and denied names, both in input order.

    Notes
    -----
    **An allowlist is closed by default**, which is why it is the safer choice.
    A column added to your data next month is denied without anyone
    remembering to deny it.

    Names in ``allow`` or ``deny`` that do not exist in ``columns`` are simply
    ignored, so a config can safely name columns that appear in some datasets
    and not others.

    Examples
    --------
    >>> filter_columns(["a", "b", "c"], deny=("b",))
    (['a', 'c'], ['b'])
    >>> filter_columns(["a", "b", "c"], allow=("a",))
    (['a'], ['b', 'c'])

    See Also
    --------
    EgressConfig : Where the lists come from.
    """
    if allow is not None:
        allowed = [c for c in columns if c in allow]
        denied = [c for c in columns if c not in allow]
    else:
        allowed = [c for c in columns if c not in deny]
        denied = [c for c in columns if c in deny]
    return allowed, denied


def rename_columns(
    columns: list[str],
    mapping: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Rename columns for sending, and report which renames took effect.

    Some column names disclose more than their values do. Renaming lets the
    model keep reasoning about structure while the original name stays home.

    Parameters
    ----------
    columns:
        Names to rename.
    mapping:
        Original to replacement. Names absent from the mapping pass through
        unchanged.

    Returns
    -------
    tuple of (list of str, dict)
        The outgoing names in order, and only the renames that actually
        applied — so the manifest records what happened rather than what was
        configured.

    Notes
    -----
    **Renaming hides the label, not the content.** A column renamed to
    ``feature_3`` still contains whatever it contained. Use ``deny_columns``
    when the values are the problem.

    No uniqueness check is performed. Mapping two columns to the same name
    produces duplicates, and the resulting payload will be confusing rather
    than wrong.

    Examples
    --------
    >>> rename_columns(["patient_id", "age"], {"patient_id": "id"})
    (['id', 'age'], {'patient_id': 'id'})

    See Also
    --------
    scrub_headers : Discarding names entirely.
    """
    renamed = []
    applied: dict[str, str] = {}
    for col in columns:
        if col in mapping:
            renamed.append(mapping[col])
            applied[col] = mapping[col]
        else:
            renamed.append(col)
    return renamed, applied


def scrub_headers(columns: list[str]) -> list[str]:
    """Replace every column name with a positional placeholder.

    The strongest form of name privacy: ``col_0``, ``col_1``, and so on, in
    order. Nothing about your schema is disclosed.

    Parameters
    ----------
    columns:
        The names to discard. Only the count matters.

    Returns
    -------
    list of str
        Placeholder names, one per input column.

    Notes
    -----
    **This costs the model most of its usefulness.** Advice about which columns
    to encode, which look like identifiers, or which the target plausibly
    depends on all rests on the names. With placeholders the model can reason
    about shape and dtype and very little else. Reach for ``rename_columns``
    first, and for this only when no name can be sent.

    Examples
    --------
    >>> scrub_headers(["patient_name", "diagnosis"])
    ['col_0', 'col_1']

    See Also
    --------
    rename_columns : Substituting names rather than discarding them.
    """
    return [f"col_{i}" for i in range(len(columns))]


def redact_value(value: Any, patterns: tuple[str, ...] = ()) -> Any:
    """Replace matching substrings in a string value with ``[REDACTED]``.

    For content that is sensitive wherever it appears — a key, an identifier
    format, a name — rather than sensitive because of which column it is in.

    Parameters
    ----------
    value:
        The value. Non-strings are returned untouched, since the patterns are
        text patterns.
    patterns:
        Regular expressions, applied in order.

    Returns
    -------
    Any
        The value with matches replaced, or the original when it is not a
        string or nothing matched.

    Notes
    -----
    **Only strings are examined.** A number, a timestamp, or a nested object is
    passed through as-is, so redaction here does not cover a sensitive value
    stored in a non-text column.

    Patterns compose: each is applied to the output of the last, so an earlier
    replacement can prevent a later match.

    Examples
    --------
    >>> redact_value("contact bob@example.com", (r"[\\w.]+@[\\w.]+",))
    'contact [REDACTED]'
    >>> redact_value(42, (r"\\d+",))
    42

    See Also
    --------
    buildml.ai.security : Redacting secrets in prompts and tool output.
    """
    if not isinstance(value, str):
        return value
    result = value
    for pattern in patterns:
        result = re.sub(pattern, "[REDACTED]", result)
    return result


def build_schema_payload(
    df: pd.DataFrame,
    config: EgressConfig,
) -> tuple[dict[str, Any], EgressManifest]:
    """Describe the data's structure without disclosing any of its content.

    Sends column names, dtypes, and the row count. No values, no aggregates.
    The most private level that still lets a model give useful advice — it can
    reason about types, spot likely identifiers, and suggest a preprocessing
    order from names alone.

    Parameters
    ----------
    df:
        The data. Read for its schema only.
    config:
        Column filtering and renaming rules. The level field is ignored; the
        caller has already dispatched here.

    Returns
    -------
    tuple of (dict, EgressManifest)
        The payload — columns, dtypes, row count — and the record of what it
        contains.

    Notes
    -----
    **The row count is a value, in a small way.** It is an aggregate over the
    whole table and rarely identifying, but it is not literally nothing.

    **Column names can disclose plenty.** A column named
    ``failed_drug_test_date`` tells a story before any value is sent. That is
    what ``rename_columns`` and ``strip_headers`` are for, and why this
    function still runs personal-data name detection and warns.

    See Also
    --------
    build_stats_payload : Adding aggregates.
    build_egress_payload : The dispatcher.
    """
    columns = list(df.columns)
    allowed, denied = filter_columns(
        columns, allow=config.allow_columns, deny=config.deny_columns
    )

    if config.strip_headers:
        sent_columns = scrub_headers(allowed)
        renamed_map: dict[str, str] = {}
    else:
        sent_columns, renamed_map = rename_columns(allowed, config.rename_columns)

    pii_warnings = detect_pii_columns(sent_columns)
    warnings = tuple(
        f"Column '{c}' matches PII pattern; consider deny_columns or rename_columns."
        for c in pii_warnings
    )

    schema_info = {
        "columns": sent_columns,
        "dtypes": {
            sent_columns[i]: str(df[allowed[i]].dtype)
            for i in range(len(allowed))
        },
        "row_count": len(df),
    }

    manifest = EgressManifest(
        level=EgressLevel.SCHEMA_ONLY,
        columns_sent=tuple(sent_columns),
        columns_denied=tuple(denied),
        columns_renamed=renamed_map,
        rows_sent=0,
        estimated_tokens=_estimate_tokens(str(schema_info)),
        warnings=warnings,
    )

    return schema_info, manifest


def build_stats_payload(
    df: pd.DataFrame,
    config: EgressConfig,
) -> tuple[dict[str, Any], EgressManifest]:
    """Describe the data's shape and distributions, but send no rows.

    Everything :func:`build_schema_payload` sends, plus per-column null counts,
    distinct counts, and — for numeric columns — mean, standard deviation,
    minimum, and maximum. Enough for a model to notice skew, missingness,
    constant columns, and likely identifiers.

    The default level, because it is where the advice gets substantially better
    while individual records stay behind.

    Parameters
    ----------
    df:
        The data. Aggregated, never sampled.
    config:
        Column filtering and renaming rules.

    Returns
    -------
    tuple of (dict, EgressManifest)
        The payload — per-column statistics keyed by sent name — and the
        manifest, with ``rows_sent`` at zero.

    Notes
    -----
    **An aggregate over few rows can identify someone.** A maximum is one
    person's value. A mean over three rows is nearly their values. Statistics
    protect individuals in large tables and much less so in small ones.

    **Minimum and maximum are literal values from your data**, unlike a mean.
    For a column of salaries or ages, that may be more than you intended to
    disclose.

    Only numeric columns get distribution statistics. Categorical columns
    contribute dtype, null count, and cardinality — never the categories
    themselves, since those are values.

    See Also
    --------
    build_schema_payload : Structure alone.
    build_redacted_sample_payload : Adding masked rows.
    """
    columns = list(df.columns)
    allowed, denied = filter_columns(
        columns, allow=config.allow_columns, deny=config.deny_columns
    )

    if config.strip_headers:
        sent_columns = scrub_headers(allowed)
        renamed_map: dict[str, str] = {}
    else:
        sent_columns, renamed_map = rename_columns(allowed, config.rename_columns)

    pii_warnings = detect_pii_columns(sent_columns)
    warnings = tuple(
        f"Column '{c}' matches PII pattern; consider deny_columns or rename_columns."
        for c in pii_warnings
    )

    stats: dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(sent_columns),
        "columns": {},
    }

    for i, orig_col in enumerate(allowed):
        col_name = sent_columns[i]
        series = df[orig_col]
        col_stats: dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            col_stats["mean"] = float(desc.get("mean", 0))
            col_stats["std"] = float(desc.get("std", 0))
            col_stats["min"] = float(desc.get("min", 0))
            col_stats["max"] = float(desc.get("max", 0))
        stats["columns"][col_name] = col_stats

    manifest = EgressManifest(
        level=EgressLevel.STATS_ONLY,
        columns_sent=tuple(sent_columns),
        columns_denied=tuple(denied),
        columns_renamed=renamed_map,
        rows_sent=0,
        estimated_tokens=_estimate_tokens(str(stats)),
        warnings=warnings,
    )

    return stats, manifest


def build_redacted_sample_payload(
    df: pd.DataFrame,
    config: EgressConfig,
) -> tuple[dict[str, Any], EgressManifest]:
    """Send real rows, with suspected personal data hashed out.

    Takes the first ``sample_rows`` rows of the allowed columns. Columns whose
    *names* match a personal-data pattern have their values replaced with a
    short hash; configured redaction patterns are then applied to what remains
    in string columns.

    Real rows are what let a model see the things statistics hide — mixed
    formats, embedded units, encoding damage, a categorical stored as text.

    Parameters
    ----------
    df:
        The data.
    config:
        Column rules, sample size, and redaction patterns.

    Returns
    -------
    tuple of (dict, EgressManifest)
        The payload — row count, sample size, columns, and the sample records
        — and the manifest with the true ``rows_sent``.

    Notes
    -----
    **Masking is driven by column names, so it misses what is misnamed.** A
    column called ``notes`` containing addresses is sent verbatim. Name the
    columns you know are sensitive in ``deny_columns``; do not rely on
    detection.

    **The sample is the first rows, not a random draw.** Sorted or grouped data
    therefore sends an unrepresentative slice — often all of one category. That
    keeps the payload reproducible, which is worth more here than
    representativeness, but it is worth knowing.

    **Hashing is one-way but not anonymising.** The same value hashes to the
    same token every time, so repeats stay linkable across the sample, and a
    small known value space can be reversed by trying it.

    See Also
    --------
    build_stats_payload : No rows at all.
    build_full_sample_payload : No masking.
    """
    columns = list(df.columns)
    allowed, denied = filter_columns(
        columns, allow=config.allow_columns, deny=config.deny_columns
    )

    if config.strip_headers:
        sent_columns = scrub_headers(allowed)
        renamed_map: dict[str, str] = {}
    else:
        sent_columns, renamed_map = rename_columns(allowed, config.rename_columns)

    pii_columns = detect_pii_columns(allowed)
    sample_size = min(config.sample_rows, len(df))
    sample_df = df[allowed].head(sample_size).copy()

    for pii_col in pii_columns:
        if pii_col in sample_df.columns:
            sample_df[pii_col] = sample_df[pii_col].apply(_hash_value)

    for pat in config.redact_patterns:
        for col in sample_df.columns:
            if sample_df[col].dtype == object:
                sample_df[col] = sample_df[col].apply(
                    lambda x, p=pat: redact_value(x, (p,))
                )

    sample_df.columns = pd.Index(sent_columns)
    sample_records = sample_df.to_dict(orient="records")

    pii_warnings = tuple(
        f"Column '{c}' masked (PII pattern detected)." for c in pii_columns
    )

    payload = {
        "row_count": len(df),
        "sample_rows": sample_size,
        "columns": sent_columns,
        "sample": sample_records,
    }

    manifest = EgressManifest(
        level=EgressLevel.REDACTED_SAMPLE,
        columns_sent=tuple(sent_columns),
        columns_denied=tuple(denied),
        columns_renamed=renamed_map,
        rows_sent=sample_size,
        estimated_tokens=_estimate_tokens(str(payload)),
        warnings=pii_warnings,
    )

    return payload, manifest


def build_full_sample_payload(
    df: pd.DataFrame,
    config: EgressConfig,
) -> tuple[dict[str, Any], EgressManifest]:
    """Send real rows exactly as they are, with nothing masked.

    Column filtering and renaming still apply, but every value in an allowed
    column goes as written. Personal-data name detection still runs and still
    warns — and the warning is all it does here.

    Parameters
    ----------
    df:
        The data.
    config:
        Column rules and sample size. ``redact_patterns`` is not applied at
        this level.

    Returns
    -------
    tuple of (dict, EgressManifest)
        The payload and its manifest, whose warnings name any column that
        looked personal and was sent regardless.

    Notes
    -----
    **This level exists for data you would be comfortable publishing.** Test
    fixtures, public datasets, synthetic data. For anything else, the level
    below sends the same structural information with the identifying parts
    removed.

    **Redaction patterns are deliberately not applied.** A level named
    ``FULL_SAMPLE`` that quietly filtered some values would be lying about what
    it does. Use ``REDACTED_SAMPLE`` when you want filtering.

    As with the redacted level, the sample is the first rows rather than a
    random draw.

    See Also
    --------
    build_redacted_sample_payload : Rows with masking.
    """
    columns = list(df.columns)
    allowed, denied = filter_columns(
        columns, allow=config.allow_columns, deny=config.deny_columns
    )

    if config.strip_headers:
        sent_columns = scrub_headers(allowed)
        renamed_map: dict[str, str] = {}
    else:
        sent_columns, renamed_map = rename_columns(allowed, config.rename_columns)

    sample_size = min(config.sample_rows, len(df))
    sample_df = df[allowed].head(sample_size).copy()
    sample_df.columns = pd.Index(sent_columns)
    sample_records = sample_df.to_dict(orient="records")

    pii_columns = detect_pii_columns(sent_columns)
    warnings = tuple(
        f"Column '{c}' matches PII pattern; FULL_SAMPLE sends raw values."
        for c in pii_columns
    )

    payload = {
        "row_count": len(df),
        "sample_rows": sample_size,
        "columns": sent_columns,
        "sample": sample_records,
    }

    manifest = EgressManifest(
        level=EgressLevel.FULL_SAMPLE,
        columns_sent=tuple(sent_columns),
        columns_denied=tuple(denied),
        columns_renamed=renamed_map,
        rows_sent=sample_size,
        estimated_tokens=_estimate_tokens(str(payload)),
        warnings=warnings,
    )

    return payload, manifest


def build_egress_payload(
    df: pd.DataFrame | None,
    config: EgressConfig,
) -> tuple[dict[str, Any] | None, EgressManifest]:
    """Build what gets sent, and the record of what that was.

    The entry point for the whole module. Dispatches on the configured level to
    the matching builder, and handles the no-data case without special-casing
    at every call site.

    Parameters
    ----------
    df:
        The data, or ``None`` when nothing is loaded.
    config:
        The egress rules, including which level to apply.

    Returns
    -------
    tuple of (dict or None, EgressManifest)
        The payload and its manifest. The payload is ``None`` when there is no
        data, and the manifest is still returned — an empty one, with a warning
        — so callers never have to branch on whether a record exists.

    Notes
    -----
    **The manifest is always produced, and should always be recorded.** It is
    what makes a later question about disclosure answerable.

    **An unrecognised level falls back to statistics**, the conservative
    choice: a configuration mistake results in less being sent, never more.

    **``estimated_tokens`` is a rough approximation**, derived from word count.
    Real tokenisers split on subwords and punctuation, so treat it as an order
    of magnitude for budgeting, not a figure to plan a context limit around.

    Examples
    --------
    Build a schema-only payload::

        config = EgressConfig(level=EgressLevel.SCHEMA_ONLY)
        payload, manifest = build_egress_payload(frame, config)
        manifest.columns_denied

    See Also
    --------
    EgressConfig : The rules.
    EgressManifest : The record.
    """
    if df is None or len(df) == 0:
        return None, EgressManifest(
            level=config.level,
            columns_sent=(),
            columns_denied=(),
            columns_renamed={},
            rows_sent=0,
            estimated_tokens=0,
            warnings=("No dataset attached; schema-only context unavailable.",),
        )

    if config.level == EgressLevel.SCHEMA_ONLY:
        return build_schema_payload(df, config)
    elif config.level == EgressLevel.STATS_ONLY:
        return build_stats_payload(df, config)
    elif config.level == EgressLevel.REDACTED_SAMPLE:
        return build_redacted_sample_payload(df, config)
    elif config.level == EgressLevel.FULL_SAMPLE:
        return build_full_sample_payload(df, config)
    else:
        return build_stats_payload(df, config)


def _hash_value(value: Any) -> str:
    """Hash a value for redaction."""
    if pd.isna(value):
        return "[NULL]"
    return f"[HASH:{hashlib.sha256(str(value).encode()).hexdigest()[:8]}]"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (words / 0.75)."""
    words = len(text.split())
    return int(words / 0.75)
