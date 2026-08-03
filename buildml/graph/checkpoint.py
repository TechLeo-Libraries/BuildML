"""Graph-learning bundle persistence (distinct from Session checkpoints)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.errors import ValidationError
from buildml.graph.results import GraphEvalResult, GraphFitResult, GraphPlan

BUNDLE_FORMAT = "buildml.graph_bundle.v1"
CHECKPOINT_BOUNDARY = (
    "Graph bundles, causal bundles, Torch trainer bundles, RAG bundles, and "
    "Session checkpoints are complementary, not interchangeable. A graph "
    "bundle (buildml.graph_bundle.v1) stores a GraphPlan (GraphSpec + fitted "
    "classical estimator or pure-Torch GCN + label encoder). A Session "
    "checkpoint stores data, roles, splits, history, and optional classical "
    "preprocess plans; it does not embed the graph learner. Reload tabular "
    "workflow via checkpoint_load; reload the learner via load_graph_bundle. "
    "Honesty: node classification with NetworkX metrics + sklearn and/or a "
    "small pure-Torch GCN and/or PyG GCN/GraphSAGE/GAT — not Neo4j/KG."
)


def save_graph_bundle(
    path: str | Path,
    plan: GraphPlan,
    *,
    fit_result: GraphFitResult | None = None,
    eval_result: GraphEvalResult | None = None,
) -> Path:
    """Write a graph bundle directory (``buildml.graph_bundle.v1``).

    Persists the fitted :class:`~buildml.graph.results.GraphPlan` separately
    from Session checkpoints so tabular workflow and graph-learner state reload
    independently.

    Parameters
    ----------
    path:
        Destination directory for ``meta.json`` and ``graph_plan.joblib``.
    plan:
        Train-fitted graph plan to persist.
    fit_result, eval_result:
        Optional last operation reports for bundle metadata.

    Returns
    -------
    pathlib.Path
        The bundle directory that was written.

    Raises
    ------
    ValidationError
        When ``plan`` is ``None``.
    """
    if plan is None:
        raise ValidationError("No GraphPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {"plan": plan}
    joblib.dump(payload, destination / "graph_plan.joblib")
    meta: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_graph_bundle(path: str | Path) -> GraphPlan:
    """Load a graph bundle into a :class:`GraphPlan`.

    Validates bundle format and restores the fitted plan from
    ``graph_plan.joblib``.

    Parameters
    ----------
    path:
        Directory containing ``meta.json`` and ``graph_plan.joblib``.

    Returns
    -------
    GraphPlan
        Deserialised graph-learning plan.

    Raises
    ------
    ValidationError
        When files are missing, format is unsupported, or payload is invalid.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "graph_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete graph bundle at {root}. "
            f"Expected meta.json and graph_plan.joblib ({BUNDLE_FORMAT})."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValidationError(
            f"Unsupported graph bundle format {fmt!r}; expected {BUNDLE_FORMAT}."
        )
    loaded = joblib.load(plan_path)
    if isinstance(loaded, GraphPlan):
        return loaded
    if not isinstance(loaded, dict) or "plan" not in loaded:
        raise ValidationError(
            "graph_plan.joblib must contain a GraphPlan or a payload with key 'plan'."
        )
    plan = loaded["plan"]
    if not isinstance(plan, GraphPlan):
        raise ValidationError("Loaded plan object is not a GraphPlan")
    return plan
