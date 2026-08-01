"""Data-quality analyzer: completeness, uniqueness, validity, patterns."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL = re.compile(r"^https?://", re.I)
_PHONE = re.compile(r"^\+?[\d\-\s\(\)]{7,}$")


def analyze_quality(full: pd.DataFrame, sample: pd.DataFrame) -> dict[str, Any]:
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
