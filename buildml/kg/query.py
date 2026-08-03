"""Symbolic neighborhood / path / typed queries over the train KG.

Not an LLM, not Neo4j / Cypher, not RAG. Pure adjacency traversal on the
train triple store materialized at fit time.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Literal

from buildml.core.errors import ValidationError
from buildml.kg.results import KgPlan, KgQueryResult
from buildml.kg.types import KgQueryMode

Direction = Literal["out", "in", "both"]


def query_kg(
    plan: KgPlan,
    *,
    mode: KgQueryMode = "neighbors",
    entity: Any | None = None,
    source: Any | None = None,
    target: Any | None = None,
    relation: Any | None = None,
    direction: Direction = "out",
    max_hops: int = 3,
) -> KgQueryResult:
    """Run a symbolic query against the train triple graph.

    Traverses fit-time adjacency only — not an LLM, Neo4j, or RAG retrieve path.

    Parameters
    ----------
    plan:
        Train-fitted plan with ``out_edges_`` and ``in_edges_`` adjacency.
    mode:
        ``neighbors``, ``path``, or ``typed`` query mode.
    entity:
        Anchor entity for neighbor or typed queries.
    source, target:
        Endpoints for ``path`` mode BFS.
    relation:
        Optional relation filter for neighbor/typed modes.
    direction:
        ``out``, ``in``, or ``both`` for neighbor traversal.
    max_hops:
        Maximum BFS depth for ``path`` mode.

    Returns
    -------
    KgQueryResult
        Neighbor lists, typed matches, or shortest path steps.

    Raises
    ------
    ValidationError
        When mode/direction is invalid, required entities are missing, or hops < 1.

    Notes
    -----
    Modes
    ^^^^^
    - ``neighbors``: 1-hop neighbors of ``entity`` (optional relation filter).
    - ``typed``: tails of ``(entity, relation, ?)`` (out) or heads of
      ``(?, relation, entity)`` (in).
    - ``path``: shortest relation-labeled path from ``source`` to ``target``
      (BFS, capped by ``max_hops``).
    """
    if mode not in {"neighbors", "path", "typed"}:
        raise ValidationError("mode must be 'neighbors', 'path', or 'typed'.")
    if direction not in {"out", "in", "both"}:
        raise ValidationError("direction must be 'out', 'in', or 'both'.")
    if int(max_hops) < 1:
        raise ValidationError("max_hops must be >= 1.")

    disclosures = [
        "Symbolic query over train adjacency only (fit-time triples).",
        "Not an LLM query, not Neo4j/Cypher, not RAG retrieve.",
        "Holdout triples are never present in the query graph.",
    ]
    warnings: list[str] = []

    if mode == "neighbors":
        if entity is None:
            raise ValidationError("mode='neighbors' requires entity=.")
        eid = plan.entity_index_.get(entity)
        if eid is None:
            warnings.append(f"Entity {entity!r} absent from train vocab.")
            return KgQueryResult(
                mode=mode,
                n_results=0,
                results=(),
                source=entity,
                relation=relation,
                disclosures=tuple(disclosures),
                warnings=tuple(warnings),
            )
        rel_filter = None
        if relation is not None:
            rel_filter = plan.relation_index_.get(relation)
            if rel_filter is None:
                warnings.append(f"Relation {relation!r} absent from train vocab.")
                return KgQueryResult(
                    mode=mode,
                    n_results=0,
                    results=(),
                    source=entity,
                    relation=relation,
                    disclosures=tuple(disclosures),
                    warnings=tuple(warnings),
                )
        seen: set[tuple[Any, Any, str]] = set()
        rows: list[tuple[Any, Any, str]] = []
        if direction in {"out", "both"}:
            for r_id, n_id in plan.out_edges_.get(eid, []):
                if rel_filter is not None and r_id != rel_filter:
                    continue
                row = (plan.entity_ids[n_id], plan.relation_ids[r_id], "out")
                if row not in seen:
                    seen.add(row)
                    rows.append(row)
        if direction in {"in", "both"}:
            for r_id, n_id in plan.in_edges_.get(eid, []):
                if rel_filter is not None and r_id != rel_filter:
                    continue
                row = (plan.entity_ids[n_id], plan.relation_ids[r_id], "in")
                if row not in seen:
                    seen.add(row)
                    rows.append(row)
        return KgQueryResult(
            mode=mode,
            n_results=len(rows),
            results=tuple(rows),
            source=entity,
            relation=relation,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    if mode == "typed":
        if entity is None or relation is None:
            raise ValidationError("mode='typed' requires entity= and relation=.")
        eid = plan.entity_index_.get(entity)
        rid = plan.relation_index_.get(relation)
        if eid is None or rid is None:
            warnings.append("entity or relation absent from train vocab.")
            return KgQueryResult(
                mode=mode,
                n_results=0,
                results=(),
                source=entity,
                relation=relation,
                disclosures=tuple(disclosures),
                warnings=tuple(warnings),
            )
        if direction == "in":
            neighbors = [
                plan.entity_ids[n]
                for r, n in plan.in_edges_.get(eid, [])
                if r == rid
            ]
        else:
            # out (default) and both → prefer out for typed (h, r, ?)
            neighbors = [
                plan.entity_ids[n]
                for r, n in plan.out_edges_.get(eid, [])
                if r == rid
            ]
            if direction == "both":
                neighbors.extend(
                    plan.entity_ids[n]
                    for r, n in plan.in_edges_.get(eid, [])
                    if r == rid
                )
        # Deduplicate preserving order
        uniq: list[Any] = []
        seen_e: set[Any] = set()
        for n in neighbors:
            if n not in seen_e:
                seen_e.add(n)
                uniq.append(n)
        return KgQueryResult(
            mode=mode,
            n_results=len(uniq),
            results=tuple(uniq),
            source=entity,
            relation=relation,
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    # path
    if source is None or target is None:
        raise ValidationError("mode='path' requires source= and target=.")
    s_id = plan.entity_index_.get(source)
    t_id = plan.entity_index_.get(target)
    if s_id is None or t_id is None:
        warnings.append("source or target absent from train vocab.")
        return KgQueryResult(
            mode=mode,
            n_results=0,
            results=(),
            source=source,
            target=target,
            max_hops=int(max_hops),
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )
    if s_id == t_id:
        return KgQueryResult(
            mode=mode,
            n_results=0,
            results=(),
            source=source,
            target=target,
            max_hops=int(max_hops),
            disclosures=tuple(disclosures) + ("source == target; empty path.",),
            warnings=tuple(warnings),
        )

    # BFS over directed out-edges: state = entity; store predecessor
    prev: dict[int, tuple[int, int] | None] = {s_id: None}
    queue: deque[int] = deque([s_id])
    depth = {s_id: 0}
    found = False
    while queue:
        cur = queue.popleft()
        if cur == t_id:
            found = True
            break
        if depth[cur] >= int(max_hops):
            continue
        for r_id, n_id in plan.out_edges_.get(cur, []):
            if n_id in prev:
                continue
            prev[n_id] = (cur, r_id)
            depth[n_id] = depth[cur] + 1
            queue.append(n_id)

    if not found or t_id not in prev:
        warnings.append(
            f"No path from {source!r} to {target!r} within max_hops={max_hops}."
        )
        return KgQueryResult(
            mode=mode,
            n_results=0,
            results=(),
            source=source,
            target=target,
            max_hops=int(max_hops),
            disclosures=tuple(disclosures),
            warnings=tuple(warnings),
        )

    # Reconstruct path as (from_entity, relation, to_entity) edges.
    edges_rev: list[tuple[Any, Any, Any]] = []
    cur = t_id
    while cur != s_id:
        parent_info = prev[cur]
        assert parent_info is not None
        parent, r_id = parent_info
        edges_rev.append(
            (
                plan.entity_ids[parent],
                plan.relation_ids[r_id],
                plan.entity_ids[cur],
            )
        )
        cur = parent
    edges = list(reversed(edges_rev))

    return KgQueryResult(
        mode=mode,
        n_results=len(edges),
        results=tuple(edges),
        source=source,
        target=target,
        max_hops=int(max_hops),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
