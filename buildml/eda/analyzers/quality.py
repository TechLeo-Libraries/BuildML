"""Find the problems that break models before any modelling starts.

Not statistics: defects. A constant column, a duplicated row, an identifier
that will be treated as a feature, a numeric field stored as text with a few
``"N/A"`` values in it. None of these are interesting distributions; all of them
change what happens downstream, and most are invisible in a ``describe()``.

The checks are deliberately blunt and cheap. Each is a heuristic with a
threshold, and the thresholds are conventions rather than discoveries: 95% for
quasi-constant, 98% distinct for identifier-like, 5% to 95% numeric-looking for
mixed types. They will occasionally flag something legitimate. That is the right
error to make here: a false positive costs a glance, a missed identifier column
costs a model that scores perfectly in testing and fails completely in
production.

See Also
--------
buildml.eda.findings.build_findings : Turning these flags into ranked advice.
buildml.eda.analyzers.univariate : Distributions rather than defects.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL = re.compile(r"^https?://", re.I)
_PHONE = re.compile(r"^\+?[\d\-\s\(\)]{7,}$")


def analyze_quality(full: pd.DataFrame, sample: pd.DataFrame) -> dict[str, Any]:
    """Scan for the structural defects that make a column useless or dangerous.

    Six families of problem, each with a rule of thumb behind it.

    *Missingness*, per column and per row. The row view matters and is often
    skipped: ten columns each 5% missing might be 50% of rows affected if the
    gaps are in different places, or 5% if they coincide. Which one it is
    determines whether dropping incomplete rows is viable.

    *Constant and quasi-constant columns.* A column with one value carries no
    information. One where 95% of rows share a value carries almost none, and
    will be split on by a tree anyway.

    *Duplicate rows.* Usually a join gone wrong. They inflate the apparent
    sample size and, if they land on both sides of a split, leak between train
    and test.

    *High-cardinality and identifier-like columns.* A column with a distinct
    value for nearly every row is a key, not a feature. Left in, it lets a model
    memorise the training set: the classic cause of perfect validation scores
    and useless predictions.

    *Mixed types.* A text column where between 5% and 95% of values look numeric
    is one where something has gone wrong: a numeric field with ``"N/A"``
    sentinels, or two sources concatenated.

    *String patterns.* The share of values that look like emails, URLs, or phone
    numbers, which usually means personally identifying data that should not be
    a feature at all.

    Parameters
    ----------
    full:
        The complete frame. All the checks run against this, since a sample
        cannot count duplicates or find rare missing values reliably.
    sample:
        The subsample used elsewhere in the EDA pass. Only its size is recorded
        here, so the report can state what the association analyses were
        computed on.

    Returns
    -------
    dict
        Missingness by column and by row (with quantiles), duplicate count and
        rate, the flagged column lists (``constant_columns``,
        ``quasi_constant_columns``, ``high_cardinality_columns``,
        ``id_like_columns``, ``mixed_type_suspect_columns``),
        ``string_pattern_hints``, ``sample_used_for_associations``, and
        ``completeness_score``: the share of cells that are present, where 1.0
        is a frame with no gaps.

    Notes
    -----
    **These are heuristics, and they will occasionally be wrong.** A legitimate
    high-cardinality feature: a postcode, a product SKU: trips the
    identifier-like check. Look at what was flagged and decide; do not drop
    columns on the strength of a threshold.

    **An identifier-like column is the flag to take most seriously.** It is the
    most common cause of a model that looks excellent and does nothing.

    **Pattern detection samples.** At most 40 text columns, at most 5,000 values
    each. A rare email address in column 41 will not be found.

    **This is a full pass over the frame, several times over.** Duplicate
    detection in particular is not cheap on a wide frame.

    See Also
    --------
    buildml.eda.findings.build_findings : What acts on these flags.
    """
    missing_by_column = {str(k): int(v) for k, v in full.isna().sum().items()}
    n = len(full) or 1
    missing_rate = {c: missing_by_column[c] / n for c in missing_by_column}

    constant_columns = [str(c) for c in full.columns if full[c].nunique(dropna=False) <= 1]
    quasi_constant = [
        str(c)
        for c in full.columns
        if str(c) not in constant_columns
        and full[c].value_counts(normalize=True, dropna=False).iloc[0] >= 0.95
    ]
    duplicate_row_count = int(full.duplicated().sum())
    high_cardinality = [
        str(c)
        for c in full.columns
        if not pd.api.types.is_numeric_dtype(full[c])
        and full[c].nunique(dropna=True) > min(1000, max(50, int(0.5 * len(full))))
    ]

    id_like = []
    for col in full.columns:
        nunq = full[col].nunique(dropna=True)
        if nunq >= 0.98 * full[col].notna().sum() and nunq > 20:
            id_like.append(str(col))

    mixed_type_suspects = []
    text_cols = full.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        s = full[col].dropna().astype(str).head(500)
        numeric_like = s.str.match(r"^-?\d+(\.\d+)?$").mean() if len(s) else 0
        if 0.05 < numeric_like < 0.95:
            mixed_type_suspects.append(str(col))

    pattern_hits: dict[str, dict[str, float]] = {}
    for col in list(text_cols.astype(str))[:40]:
        s = full[col].dropna().astype(str)
        if s.empty:
            continue
        sample_s = s if len(s) <= 5000 else s.sample(5000, random_state=0)
        pattern_hits[col] = {
            "email_rate": float(sample_s.str.match(_EMAIL).mean()),
            "url_rate": float(sample_s.str.match(_URL).mean()),
            "phone_rate": float(sample_s.str.match(_PHONE).mean()),
            "blank_like_rate": float(sample_s.str.strip().eq("").mean()),
        }

    row_missingness = full.isna().sum(axis=1)
    return {
        "missing_cell_count": int(full.isna().sum().sum()),
        "missing_by_column": missing_by_column,
        "missing_rate_by_column": missing_rate,
        "rows_with_any_missing": int((row_missingness > 0).sum()),
        "rows_with_any_missing_rate": float((row_missingness > 0).mean()),
        "missingness_by_row_quantiles": {
            "q50": float(row_missingness.quantile(0.5)),
            "q90": float(row_missingness.quantile(0.9)),
            "q99": float(row_missingness.quantile(0.99)),
            "max": int(row_missingness.max()),
        },
        "duplicate_row_count": duplicate_row_count,
        "duplicate_row_rate": float(duplicate_row_count / len(full)) if len(full) else 0.0,
        "constant_columns": constant_columns,
        "quasi_constant_columns": quasi_constant,
        "high_cardinality_columns": high_cardinality,
        "id_like_columns": id_like,
        "mixed_type_suspect_columns": mixed_type_suspects,
        "string_pattern_hints": pattern_hits,
        "sample_used_for_associations": int(len(sample)),
        "completeness_score": (
            float(1.0 - (full.isna().sum().sum() / full.size)) if full.size else 1.0
        ),
    }
