"""PyKEEN adapter: RotatE, ComplEx, TransE, DistMult on train-only triples."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.kg.extras import require_pykeen
from buildml.kg.features import (
    build_adjacency,
    build_triples,
    build_vocabularies,
    encode_triples,
    partition_frame,
    resolve_triple_columns,
    train_partition_frame,
    triple_set,
)
from buildml.kg.results import KgFitResult, KgPlan
from buildml.kg.types import KgConfig, KgNorm

_PYKEEN_MODEL_MAP = {
    "transe": "TransE",
    "distmult": "DistMult",
    "rotate": "RotatE",
    "complex": "ComplEx",
}


def _triples_factory(
    triples_factory_cls: Any,
    *,
    mapped_triples: Any,
    entity_index: dict[Any, int],
    relation_index: dict[Any, int],
    n_entities: int,
    n_relations: int,
) -> Any:
    """Build a PyKEEN ``TriplesFactory`` across 1.10 and 1.11 constructors.

    PyKEEN 1.11 made ``entity_to_id`` and ``relation_to_id`` required. Older
    constructors accepted only mapped triples plus counts.
    """
    kwargs: dict[str, Any] = {
        "mapped_triples": mapped_triples,
        "num_entities": n_entities,
        "num_relations": n_relations,
    }
    params = inspect.signature(triples_factory_cls.__init__).parameters
    if "entity_to_id" in params:
        kwargs["entity_to_id"] = {str(key): int(val) for key, val in entity_index.items()}
        kwargs["relation_to_id"] = {
            str(key): int(val) for key, val in relation_index.items()
        }
    return triples_factory_cls(**kwargs)


def _holdout_triples_factory(
    triples_factory_cls: Any,
    *,
    dataset: Dataset,
    split_plan: SplitPlan,
    head_column: str,
    relation_column: str,
    tail_column: str,
    entity_index: dict[Any, int],
    relation_index: dict[Any, int],
    n_entities: int,
    n_relations: int,
    torch_module: Any,
) -> tuple[Any | None, str | None, int]:
    """Encode validation then test triples against the train vocabulary."""
    for partition in ("validation", "test"):
        try:
            frame = partition_frame(dataset, split_plan, partition)
        except ValidationError:
            continue
        triples = build_triples(
            frame,
            head_column=head_column,
            relation_column=relation_column,
            tail_column=tail_column,
        )
        try:
            heads_h, rels_h, tails_h = encode_triples(
                triples,
                head_column=head_column,
                relation_column=relation_column,
                tail_column=tail_column,
                entity_index=entity_index,
                relation_index=relation_index,
            )
        except ValidationError:
            continue
        if len(heads_h) < 1:
            continue
        mapped = np.stack([heads_h, rels_h, tails_h], axis=1).astype(np.int64)
        factory = _triples_factory(
            triples_factory_cls,
            mapped_triples=torch_module.as_tensor(mapped, dtype=torch_module.long),
            entity_index=entity_index,
            relation_index=relation_index,
            n_entities=n_entities,
            n_relations=n_relations,
        )
        return factory, partition, int(len(heads_h))
    return None, None, 0

_COMPLEX_METHODS = frozenset({"rotate", "complex"})


def _tensor_to_numpy(arr: Any) -> np.ndarray:
    """Convert torch / complex tensors to numpy (complex128 when needed)."""
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu()
    if hasattr(arr, "numpy"):
        out = arr.numpy()
    else:
        out = np.asarray(arr)
    if np.iscomplexobj(out):
        return out.astype(np.complex128)
    return out.astype(float)


def _extract_embeddings(model: Any, method: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return entity/relation matrices and embedding_kind for scoring."""
    entity_repr = model.entity_representations[0](indices=None)
    relation_repr = model.relation_representations[0](indices=None)
    ent = _tensor_to_numpy(entity_repr)
    rel = _tensor_to_numpy(relation_repr)

    if method == "rotate":
        if not np.iscomplexobj(ent):
            raise ValidationError("RotatE entity embeddings must be complex-valued.")
        if np.iscomplexobj(rel):
            rel = np.real(rel)
        return ent, rel.astype(float), "rotate"
    if method == "complex":
        if not np.iscomplexobj(ent) or not np.iscomplexobj(rel):
            raise ValidationError("ComplEx embeddings must be complex-valued.")
        return ent, rel, "complex"
    if np.iscomplexobj(ent):
        ent = np.real(ent)
    if np.iscomplexobj(rel):
        rel = np.real(rel)
    return ent.astype(float), rel.astype(float), "real"


def fit_pykeen(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    method: str = "rotate",
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
    """Fit a PyKEEN embedding model on Session train triples only.

    Materializes train triples into a PyKEEN factory, runs the pipeline with
    disclosed negative sampling, and exports embeddings back into a :class:`KgPlan`.

    Parameters
    ----------
    dataset:
        Session dataset with triple columns.
    split_plan:
        Split plan defining the train partition.
    method:
        ``transe``, ``distmult``, ``rotate``, or ``complex``.
    head_column, relation_column, tail_column:
        Explicit triple column names.
    embedding_dim:
        Latent dimension forwarded to PyKEEN.
    epochs, batch_size, learning_rate:
        PyKEEN training schedule controls.
    margin:
        Margin hyperparameter when supported by the model.
    neg_ratio:
        Negative sampling ratio disclosed on the fit result.
    norm:
        Translation norm label recorded on the plan (TransE-related).
    random_state:
        Seed forwarded to PyKEEN training.

    Returns
    -------
    tuple[KgPlan, KgFitResult]
        Fitted plan with exported numpy/complex embeddings and fit summary.

    Raises
    ------
    ValidationError
        When method is unknown, train partition is missing, or embedding export fails.
    MissingExtraError
        When pykeen or torch is not installed.
    """
    require_pykeen(feature="PyKEEN KG backend")
    import torch
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    assert_fit_partition(split_plan, "train")
    assert split_plan is not None

    method_key = str(method).lower().replace("-", "_")
    if method_key not in _PYKEEN_MODEL_MAP:
        raise ValidationError(
            f"Unknown PyKEEN KG method: {method!r}. "
            f"Supported: {sorted(_PYKEEN_MODEL_MAP)}."
        )
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
        raise ValidationError(f"Need ≥2 train entities; got {len(entity_ids)}.")
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

    mapped = np.stack([heads_i, rels_i, tails_i], axis=1).astype(np.int64)
    factory = _triples_factory(
        TriplesFactory,
        mapped_triples=torch.as_tensor(mapped, dtype=torch.long),
        entity_index=entity_index,
        relation_index=relation_index,
        n_entities=len(entity_ids),
        n_relations=len(relation_ids),
    )

    model_name = _PYKEEN_MODEL_MAP[method_key]
    seed = 0 if random_state is None else int(random_state)
    testing_factory, testing_part, n_testing = _holdout_triples_factory(
        TriplesFactory,
        dataset=dataset,
        split_plan=split_plan,
        head_column=head_col,
        relation_column=rel_col,
        tail_column=tail_col,
        entity_index=entity_index,
        relation_index=relation_index,
        n_entities=len(entity_ids),
        n_relations=len(relation_ids),
        torch_module=torch,
    )
    reused_train_as_testing = testing_factory is None
    if reused_train_as_testing:
        testing_factory = factory
        disclosures.append(
            "PyKEEN 1.11 pipeline requires a testing triples factory. No "
            "in-vocabulary holdout triples were available, so the train "
            "factory was passed for that slot only. BuildML evaluate_kg "
            "still scores Session holdout partitions separately."
        )
    else:
        disclosures.append(
            f"PyKEEN pipeline testing factory uses Session {testing_part} "
            f"({n_testing} in-vocabulary triples). Embeddings still fit on train only."
        )
    result = pipeline(
        training=factory,
        testing=testing_factory,
        model=model_name,
        model_kwargs={"embedding_dim": int(embedding_dim)},
        training_kwargs={
            "num_epochs": int(epochs),
            "batch_size": int(batch_size),
        },
        optimizer_kwargs={"lr": float(learning_rate)},
        random_seed=seed,
        device="cpu",
        use_testing_data=not reused_train_as_testing,
    )
    model = result.model
    ent_emb, rel_emb, embedding_kind = _extract_embeddings(model, method_key)

    final_loss = None
    losses = getattr(result, "losses", None)
    if losses is not None and len(losses) > 0:
        try:
            final_loss = float(losses[-1])
        except (TypeError, ValueError):
            final_loss = None

    disclosures.append(
        f"pykeen/{method_key}: trained on {len(heads_i)} unique train triples; "
        f"entities={len(entity_ids)}, relations={len(relation_ids)}, "
        f"dim={embedding_dim}, epochs={epochs}, model={model_name}."
    )
    disclosures.append(
        "Negative sampling: PyKEEN sLCWA/LCWA on train factory only; "
        f"neg_ratio={neg_ratio} recorded for parity with native disclosures "
        "(PyKEEN controls internal corruption counts)."
    )
    disclosures.append(
        "Filtered ranking at eval uses train true-triple set; holdout triples "
        "never update embeddings."
    )
    disclosures.append(
        "Symbolic query_kg (neighbors/path/typed) uses the train adjacency "
        "only: not an LLM and not a graph database."
    )
    if method_key == "transe":
        disclosures.append(
            f"PyKEEN TransE scoring: -||h+r-t||_{norm}; norm recorded for eval parity."
        )
    elif method_key == "distmult":
        disclosures.append("PyKEEN DistMult scoring: <h, r, t> = sum(h*r*t).")
    elif method_key == "rotate":
        disclosures.append(
            "PyKEEN RotatE scoring: -||h * exp(i r) - t||_2 in complex space."
        )
    elif method_key == "complex":
        disclosures.append(
            "PyKEEN ComplEx scoring: Re(<h, r, conj(t)>) in complex space."
        )

    config = KgConfig(
        backend="pykeen",
        method=method_key,  # type: ignore[arg-type]
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
        backend="pykeen",
        method=method_key,
        embedding_kind=embedding_kind,
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
    fit_result = KgFitResult(
        backend="pykeen",
        method=method_key,
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
    return plan, fit_result
