"""Fit knowledge-graph embedding models on Session train triples only."""

from __future__ import annotations

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.kg.features import (
    build_adjacency,
    build_triples,
    build_vocabularies,
    encode_triples,
    resolve_triple_columns,
    train_partition_frame,
    triple_set,
)
from buildml.kg.models import fit_embeddings
from buildml.kg.results import KgFitResult, KgPlan
from buildml.kg.types import KgConfig, KgMethod, KgNorm


def fit_kg(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    method: KgMethod = "transe",
    head_column: str | None = None,
    relation_column: str | None = None,
    tail_column: str | None = None,
    embedding_dim: int = 50,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    margin: float = 1.0,
    neg_ratio: int = 1,
    norm: KgNorm = "l1",
    random_state: int | None = 0,
) -> tuple[KgPlan, KgFitResult]:
    """Fit a leakage-safe KG embedding model on the Session **train** partition.

    Pipeline
    --------
    1. Resolve head / relation / tail columns.
    2. Materialize unique train triples only (never test).
    3. Build entity/relation vocabularies from train.
    4. Train TransE or DistMult with uniform negative sampling.
    5. Store train adjacency for symbolic neighborhood/path/typed queries.

    Honesty: Session KG learning/query — not a Neo4j / graph-DB product,
    not Graph ML node classification, not RAG.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    if method not in {"transe", "distmult"}:
        raise ValidationError(f"Unknown KG method: {method!r}")
    if int(embedding_dim) < 2:
        raise ValidationError("embedding_dim must be >= 2.")
    if int(epochs) < 1:
        raise ValidationError("epochs must be >= 1.")
    if int(batch_size) < 1:
        raise ValidationError("batch_size must be >= 1.")
    if float(learning_rate) <= 0:
        raise ValidationError("learning_rate must be > 0.")
    if int(neg_ratio) < 1:
        raise ValidationError("neg_ratio must be >= 1.")
    if norm not in {"l1", "l2"}:
        raise ValidationError("norm must be 'l1' or 'l2'.")

    head_col, rel_col, tail_col, disclosures = resolve_triple_columns(
        dataset,
        head_column=head_column,
        relation_column=relation_column,
        tail_column=tail_column,
    )
    warnings: list[str] = []

    train = train_partition_frame(dataset, split_plan)
    triples = build_triples(
        train,
        head_column=head_col,
        relation_column=rel_col,
        tail_column=tail_col,
    )
    n_dup = int(triples.attrs.get("n_duplicate_dropped", 0))
    if n_dup:
        disclosures.append(f"Dropped {n_dup} duplicate train triples.")

    entity_ids, relation_ids, entity_index, relation_index = build_vocabularies(
        triples,
        head_column=head_col,
        relation_column=rel_col,
        tail_column=tail_col,
    )
    if len(entity_ids) < 2:
        raise ValidationError(
            f"Need ≥2 train entities; got {len(entity_ids)}."
        )
    if len(relation_ids) < 1:
        raise ValidationError("Need ≥1 train relation.")

    heads_i, rels_i, tails_i = encode_triples(
        triples,
        head_column=head_col,
        relation_column=rel_col,
        tail_column=tail_col,
        entity_index=entity_index,
        relation_index=relation_index,
    )
    out_edges, in_edges = build_adjacency(heads_i, rels_i, tails_i)
    known = triple_set(heads_i, rels_i, tails_i)

    ent_emb, rel_emb, final_loss = fit_embeddings(
        method,
        heads_i,
        rels_i,
        tails_i,
        n_entities=len(entity_ids),
        n_relations=len(relation_ids),
        embedding_dim=int(embedding_dim),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        margin=float(margin),
        neg_ratio=int(neg_ratio),
        norm=norm,
        random_state=random_state,
    )

    disclosures.append(
        f"{method}: trained on {len(heads_i)} unique train triples; "
        f"entities={len(entity_ids)}, relations={len(relation_ids)}, "
        f"dim={embedding_dim}, epochs={epochs}."
    )
    disclosures.append(
        f"Negative sampling: uniform head/tail corruption, "
        f"neg_ratio={neg_ratio} per positive; never uses holdout triples."
    )
    disclosures.append(
        "Filtered ranking at eval uses train true-triple set; holdout triples "
        "never update embeddings."
    )
    disclosures.append(
        "Symbolic query_kg (neighbors/path/typed) uses the train adjacency "
        "only — not an LLM and not a graph database."
    )
    if method == "transe":
        disclosures.append(
            f"TransE scoring: -||h+r-t||_{norm}; entity embeddings L2-normalized."
        )
    else:
        disclosures.append("DistMult scoring: <h, r, t> = sum(h*r*t).")

    config = KgConfig(
        method=method,
        head_column=head_col,
        relation_column=rel_col,
        tail_column=tail_col,
        embedding_dim=int(embedding_dim),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        margin=float(margin),
        neg_ratio=int(neg_ratio),
        norm=norm,
        random_state=random_state,
    )

    plan = KgPlan(
        method=method,
        head_column=head_col,
        relation_column=rel_col,
        tail_column=tail_col,
        embedding_dim=int(embedding_dim),
        n_train_triples=int(len(heads_i)),
        n_entities=len(entity_ids),
        n_relations=len(relation_ids),
        entity_ids=entity_ids,
        relation_ids=relation_ids,
        entity_index_=entity_index,
        relation_index_=relation_index,
        train_heads_=heads_i,
        train_relations_=rels_i,
        train_tails_=tails_i,
        entity_embeddings_=ent_emb,
        relation_embeddings_=rel_emb,
        true_triple_set_=known,
        out_edges_=out_edges,
        in_edges_=in_edges,
        epochs_run=int(epochs),
        final_loss=final_loss,
        neg_ratio=int(neg_ratio),
        norm=norm,
        margin=float(margin),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
        config=config.to_dict(),
    )
    result = KgFitResult(
        method=method,
        n_train_triples=plan.n_train_triples,
        n_entities=plan.n_entities,
        n_relations=plan.n_relations,
        embedding_dim=plan.embedding_dim,
        head_column=head_col,
        relation_column=rel_col,
        tail_column=tail_col,
        epochs_run=plan.epochs_run,
        final_loss=final_loss,
        neg_ratio=int(neg_ratio),
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
    return plan, result
