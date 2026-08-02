"""LTR catalog and honest capability matrix."""

from __future__ import annotations

from typing import Any, Literal

from buildml.dl.extras import torch_spec_available
from buildml.ranking.extras import (
    catboost_available,
    lightgbm_available,
    ranking_industry_available,
    xgboost_available,
)

RankingBackendName = Literal["sklearn", "industry", "torch"]
SklearnRankerMethod = Literal["pointwise", "pairwise"]
IndustryRankerMethod = Literal["lambdarank_lgbm", "rank_ndcg_xgb", "yetirank_catboost"]
TorchRankerMethod = Literal["listwise_lite"]
RankerMethodName = SklearnRankerMethod | IndustryRankerMethod | TorchRankerMethod


def ranking_capability_matrix() -> dict[str, Any]:
    """Honest capability matrix for tabular LTR backends and methods."""
    return {
        "backends": {
            "sklearn": {
                "available": True,
                "extra": None,
                "methods": ["pointwise", "pairwise"],
                "modality": "tabular",
                "notes": (
                    "Core sklearn pointwise relevance regression (Ridge/HGB) and "
                    "pairwise RankSVM-lite; always available fallback."
                ),
            },
            "industry": {
                "available": ranking_industry_available(),
                "extra": "ranking-industry",
                "methods": [
                    m
                    for m in ("lambdarank_lgbm", "rank_ndcg_xgb", "yetirank_catboost")
                    if (
                        (m == "lambdarank_lgbm" and lightgbm_available())
                        or (m == "rank_ndcg_xgb" and xgboost_available())
                        or (m == "yetirank_catboost" and catboost_available())
                    )
                ],
                "modality": "tabular",
                "notes": (
                    "Listwise GBDT rankers (LightGBM LambdaRank, XGBoost rank:ndcg, "
                    "CatBoost YetiRank) when installed. Default backend when available."
                ),
            },
            "torch": {
                "available": torch_spec_available(),
                "extra": "torch",
                "methods": ["listwise_lite"],
                "modality": "tabular",
                "notes": (
                    "ListNet-style listwise-lite MLP trained per query group "
                    "(buildml[torch])."
                ),
            },
        },
        "ltr_vs_rag_vs_recommenders": {
            "ltr": (
                "Tabular query–item feature rows with relevance labels; "
                "fit_ranker / evaluate_ranker on judgment tables."
            ),
            "rag": (
                "Chunk embedding retrieve + optional generate; rag_evaluate uses "
                "chunk-level nDCG/MRR — different protocol from LTR."
            ),
            "recommenders": (
                "User–item interaction CF or content recommenders; "
                "evaluate_recommender on known-item ranking — not query–document LTR."
            ),
        },
        "split_discipline": {
            "preferred": "Session.group_split(group_column=query_column)",
            "rule": "Holdout query ids must not appear in train when possible.",
        },
        "evaluation": {
            "metrics": ["ndcg_at_k", "map_at_k", "mrr_at_k"],
            "macro": "Per-query macro average over holdout queries with ≥1 relevant item.",
        },
        "default_backend_when_installed": _default_backend_when_installed(),
        "default_method_when_installed": _default_method_when_installed(),
        "install_hints": {
            "ranking-industry": (
                "pip install 'buildml[ranking-industry]'  "
                "# LightGBM LambdaRank / XGBoost rank:ndcg / CatBoost YetiRank"
            ),
            "torch": (
                "pip install 'buildml[torch]'  "
                "# listwise-lite torch ranker"
            ),
        },
        "non_goals": [
            "Search-engine product (crawler, inverted index, serving stack)",
            "RAG chunk retrieve/generate (see buildml.rag)",
            "Recommender user–item CF (see buildml.recommenders)",
        ],
        "industry_extra_present": ranking_industry_available(),
        "lightgbm_present": lightgbm_available(),
        "xgboost_present": xgboost_available(),
        "catboost_present": catboost_available(),
        "torch_spec_present": torch_spec_available(),
    }


def _default_backend_when_installed() -> str:
    if ranking_industry_available():
        return "industry"
    if torch_spec_available():
        return "torch"
    return "sklearn"


def _default_method_when_installed() -> str:
    if lightgbm_available():
        return "lambdarank_lgbm"
    if xgboost_available():
        return "rank_ndcg_xgb"
    if catboost_available():
        return "yetirank_catboost"
    if torch_spec_available():
        return "listwise_lite"
    return "pointwise"


def list_ranking_methods(
    *,
    backend: RankingBackendName | None = None,
) -> list[str]:
    """List LTR methods for a backend (or all available when backend is None)."""
    matrix = ranking_capability_matrix()
    if backend is not None:
        entry = matrix["backends"].get(backend)
        if entry is None or not entry.get("available"):
            return []
        return list(entry.get("methods") or [])
    methods: list[str] = []
    for entry in matrix["backends"].values():
        if not entry.get("available"):
            continue
        for method in entry.get("methods") or []:
            if method not in methods:
                methods.append(method)
    return methods


def backend_available(name: RankingBackendName) -> bool:
    entry = ranking_capability_matrix()["backends"].get(name)
    if entry is None:
        return False
    return bool(entry.get("available"))


def resolve_backend_method(
    *,
    backend: RankingBackendName | None,
    method: str,
) -> tuple[RankingBackendName, str]:
    """Validate backend/method pairing and apply honest defaults."""
    from buildml.core.errors import MissingExtraError, ValidationError

    resolved_backend: RankingBackendName
    if backend is None:
        if method in {"pointwise", "pairwise"}:
            resolved_backend = "sklearn"
        elif method in {"lambdarank_lgbm", "rank_ndcg_xgb", "yetirank_catboost"}:
            resolved_backend = "industry"
        elif method == "listwise_lite":
            resolved_backend = "torch"
        else:
            resolved_backend = _default_backend_when_installed()  # type: ignore[assignment]
            allowed_default = list_ranking_methods(backend=resolved_backend)
            if allowed_default:
                method = allowed_default[0]
            else:
                method = "pointwise"
                resolved_backend = "sklearn"
    else:
        resolved_backend = backend

    allowed = list_ranking_methods(backend=resolved_backend)
    if method not in allowed:
        raise ValidationError(
            f"method='{method}' is not valid for backend='{resolved_backend}'. "
            f"Choose from {allowed}."
        )
    if not backend_available(resolved_backend):
        extra = ranking_capability_matrix()["backends"][resolved_backend].get("extra")
        raise MissingExtraError(str(extra or "ranking-industry"), f"backend='{resolved_backend}'")
    return resolved_backend, method
