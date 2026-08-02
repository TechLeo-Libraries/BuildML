"""Industry active-learning query scoring (native CoreSet / QBC)."""

from __future__ import annotations

from buildml.activelearning.adapters.industry_native import score_industry_native_pool

score_industry_pool = score_industry_native_pool

__all__ = ["score_industry_pool"]
