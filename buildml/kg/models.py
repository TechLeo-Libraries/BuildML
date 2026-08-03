"""Pure-numpy TransE and DistMult embedding trainers / scorers.

Dependency policy: core-only (numpy). No Neo4j, no torch required.
Negative sampling: uniform head/tail corruption of train triples only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from buildml.core.errors import ValidationError

NormName = Literal["l1", "l2"]


def _unit_norm_rows(mat: np.ndarray) -> None:
    """In-place L2 row normalization (TransE entity constraint)."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    mat /= norms


def score_transe(
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    *,
    norm: NormName = "l1",
) -> np.ndarray:
    """Score TransE triples with higher-is-better negative translation distance.

    Computes ``-||h + r - t||`` under the chosen norm for link prediction and
    evaluation ranking paths.

    Parameters
    ----------
    heads, relations, tails:
        Integer-encoded triple component arrays.
    entity_emb, relation_emb:
        Fitted embedding matrices.
    norm:
        ``l1`` or ``l2`` translation norm.

    Returns
    -------
    numpy.ndarray
        Score vector aligned with input triple rows.
    """
    h = entity_emb[heads]
    r = relation_emb[relations]
    t = entity_emb[tails]
    diff = h + r - t
    if norm == "l1":
        dist = np.sum(np.abs(diff), axis=-1)
    else:
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return -dist


def score_distmult(
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
) -> np.ndarray:
    """Score DistMult triples with higher-is-better trilinear product.

    Computes ``sum(h * r * t)`` elementwise across the embedding dimension.

    Parameters
    ----------
    heads, relations, tails:
        Integer-encoded triple component arrays.
    entity_emb, relation_emb:
        Fitted embedding matrices.

    Returns
    -------
    numpy.ndarray
        Score vector aligned with input triple rows.
    """
    h = entity_emb[heads]
    r = relation_emb[relations]
    t = entity_emb[tails]
    return np.sum(h * r * t, axis=-1)


def score_rotate(
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
) -> np.ndarray:
    """Score RotatE triples with higher-is-better complex rotation distance.

    Used when PyKEEN exports complex entity embeddings for RotatE scoring.

    Parameters
    ----------
    heads, relations, tails:
        Integer-encoded triple component arrays.
    entity_emb, relation_emb:
        Complex entity embeddings and real relation phase vectors.

    Returns
    -------
    numpy.ndarray
        Score vector aligned with input triple rows.
    """
    h = entity_emb[heads]
    r_phase = relation_emb[relations]
    t = entity_emb[tails]
    rotation = np.exp(1j * r_phase)
    diff = h * rotation - t
    dist = np.linalg.norm(diff, axis=-1)
    return -dist.astype(float)


def score_complex(
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
) -> np.ndarray:
    """Score ComplEx triples with higher-is-better complex trilinear product.

    Computes ``Re(<h, r, conj(t)>)`` for link prediction on complex embeddings.

    Parameters
    ----------
    heads, relations, tails:
        Integer-encoded triple component arrays.
    entity_emb, relation_emb:
        Complex entity and relation embedding matrices.

    Returns
    -------
    numpy.ndarray
        Score vector aligned with input triple rows.
    """
    h = entity_emb[heads]
    r = relation_emb[relations]
    t = entity_emb[tails]
    return np.real(np.sum(h * r * np.conj(t), axis=-1)).astype(float)


def score_triples_batch(
    method: str,
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    *,
    norm: NormName = "l1",
) -> np.ndarray:
    """Dispatch triple scoring to the method-specific scorer.

    Central routing used by batch scoring, link prediction, and evaluation paths
    to pick TransE, DistMult, RotatE, or ComplEx scorers.

    Parameters
    ----------
    method:
        ``transe``, ``distmult``, ``rotate``, or ``complex``.
    heads, relations, tails:
        Integer-encoded triple arrays.
    entity_emb, relation_emb:
        Fitted embedding matrices.
    norm:
        Translation norm for TransE only.

    Returns
    -------
    numpy.ndarray
        Score vector for the batch of triples.

    Raises
    ------
    ValidationError
        When ``method`` is not supported.
    """
    if method == "transe":
        return score_transe(
            heads, relations, tails, entity_emb, relation_emb, norm=norm
        )
    if method == "distmult":
        return score_distmult(heads, relations, tails, entity_emb, relation_emb)
    if method == "rotate":
        return score_rotate(heads, relations, tails, entity_emb, relation_emb)
    if method == "complex":
        return score_complex(heads, relations, tails, entity_emb, relation_emb)
    raise ValidationError(f"Unknown KG method: {method!r}")


def score_all_tails(
    method: str,
    head: int,
    relation: int,
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    *,
    norm: NormName = "l1",
) -> np.ndarray:
    """Score (h, r, ?) against every entity as a candidate tail.

    Used by tail link prediction and filtered evaluation ranking.

    Parameters
    ----------
    method:
        Embedding method key on the plan.
    head, relation:
        Query head and relation integer ids.
    entity_emb, relation_emb:
        Fitted embedding matrices.
    norm:
        Translation norm for TransE scoring.

    Returns
    -------
    numpy.ndarray
        Score vector with one entry per entity in the vocabulary.
    """
    n = entity_emb.shape[0]
    heads = np.full(n, head, dtype=np.int64)
    rels = np.full(n, relation, dtype=np.int64)
    tails = np.arange(n, dtype=np.int64)
    return score_triples_batch(
        method, heads, rels, tails, entity_emb, relation_emb, norm=norm
    )


def score_all_heads(
    method: str,
    relation: int,
    tail: int,
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    *,
    norm: NormName = "l1",
) -> np.ndarray:
    """Score (?, r, t) against every entity as a candidate head.

    Used by head link prediction and filtered evaluation ranking.

    Parameters
    ----------
    method:
        Embedding method key on the plan.
    relation, tail:
        Query relation and tail integer ids.
    entity_emb, relation_emb:
        Fitted embedding matrices.
    norm:
        Translation norm for TransE scoring.

    Returns
    -------
    numpy.ndarray
        Score vector with one entry per entity in the vocabulary.
    """
    n = entity_emb.shape[0]
    heads = np.arange(n, dtype=np.int64)
    rels = np.full(n, relation, dtype=np.int64)
    tails = np.full(n, tail, dtype=np.int64)
    return score_triples_batch(
        method, heads, rels, tails, entity_emb, relation_emb, norm=norm
    )


def score_all_relations(
    method: str,
    head: int,
    tail: int,
    entity_emb: np.ndarray,
    relation_emb: np.ndarray,
    *,
    norm: NormName = "l1",
) -> np.ndarray:
    """Score (h, ?, t) against every relation as a candidate link.

    Used by relation link prediction in :func:`buildml.kg.predict.predict_links`.

    Parameters
    ----------
    method:
        Embedding method key on the plan.
    head, tail:
        Query head and tail integer ids.
    entity_emb, relation_emb:
        Fitted embedding matrices.
    norm:
        Translation norm for TransE scoring.

    Returns
    -------
    numpy.ndarray
        Score vector with one entry per relation in the vocabulary.
    """
    n_rel = relation_emb.shape[0]
    heads = np.full(n_rel, head, dtype=np.int64)
    rels = np.arange(n_rel, dtype=np.int64)
    tails = np.full(n_rel, tail, dtype=np.int64)
    return score_triples_batch(
        method, heads, rels, tails, entity_emb, relation_emb, norm=norm
    )


def _corrupt_batch(
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    n_entities: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform head-or-tail corruption (equal probability)."""
    n = len(heads)
    corrupt_heads = heads.copy()
    corrupt_tails = tails.copy()
    flip_head = rng.random(n) < 0.5
    # Sample replacements; retry once if collision with original entity
    replacements = rng.integers(0, n_entities, size=n)
    same = np.where(
        flip_head,
        replacements == heads,
        replacements == tails,
    )
    if same.any():
        replacements[same] = (replacements[same] + 1) % n_entities
    corrupt_heads[flip_head] = replacements[flip_head]
    corrupt_tails[~flip_head] = replacements[~flip_head]
    return corrupt_heads, relations.copy(), corrupt_tails


def fit_transe(
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    *,
    n_entities: int,
    n_relations: int,
    embedding_dim: int = 50,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    margin: float = 1.0,
    neg_ratio: int = 1,
    norm: NormName = "l1",
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train TransE embeddings with margin ranking loss on train triples.

    Uses uniform head-or-tail negative corruption disclosed on the fit result.

    Parameters
    ----------
    heads, relations, tails:
        Encoded train triple arrays.
    n_entities, n_relations:
        Vocabulary sizes.
    embedding_dim, epochs, batch_size, learning_rate:
        Training hyperparameters.
    margin:
        Margin for the ranking loss.
    neg_ratio:
        Negative samples drawn per positive triple.
    norm:
        ``l1`` or ``l2`` translation norm.
    random_state:
        Seed for initialization and corruption sampling.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float]
        Entity embeddings, relation embeddings, and final epoch loss.

    Raises
    ------
    ValidationError
        When vocabulary sizes are too small for TransE training.
    """
    if n_entities < 2 or n_relations < 1:
        raise ValidationError("TransE needs ≥2 entities and ≥1 relation.")
    rng = np.random.default_rng(random_state)
    ent = rng.normal(0.0, 1.0 / embedding_dim, size=(n_entities, embedding_dim))
    rel = rng.normal(0.0, 1.0 / embedding_dim, size=(n_relations, embedding_dim))
    _unit_norm_rows(ent)
    _unit_norm_rows(rel)

    n = len(heads)
    order = np.arange(n)
    last_loss = 0.0
    for _ in range(int(epochs)):
        rng.shuffle(order)
        epoch_loss = 0.0
        n_steps = 0
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            bh = heads[idx]
            br = relations[idx]
            bt = tails[idx]
            batch_loss = 0.0
            for _neg in range(max(1, int(neg_ratio))):
                nh, nr, nt = _corrupt_batch(bh, br, bt, n_entities, rng)
                pos = score_transe(bh, br, bt, ent, rel, norm=norm)
                neg = score_transe(nh, nr, nt, ent, rel, norm=norm)
                # margin ranking: max(0, margin - pos + (-neg_dist) wait)
                # scores are -distance, so higher is better.
                # loss = max(0, margin - s_pos + s_neg) with s = -d
                # = max(0, margin - (-d_pos) + (-d_neg)) = max(0, margin + d_pos - d_neg)
                losses = np.maximum(0.0, margin - pos + neg)
                batch_loss += float(losses.mean())
                # Subgradient: only update triples with positive loss
                active = losses > 0
                if not active.any():
                    continue
                ah = bh[active]
                ar = br[active]
                at = bt[active]
                anh = nh[active]
                anr = nr[active]
                ant = nt[active]
                # Gradients of distance w.r.t embeddings (for L1/L2)
                # We descend on loss ≈ margin + d_pos - d_neg
                h_p = ent[ah]
                r_p = rel[ar]
                t_p = ent[at]
                diff_p = h_p + r_p - t_p
                h_n = ent[anh]
                r_n = rel[anr]
                t_n = ent[ant]
                diff_n = h_n + r_n - t_n
                if norm == "l1":
                    g_p = np.sign(diff_p)
                    g_n = np.sign(diff_n)
                else:
                    # d = ||x||_2; grad = x / ||x||
                    np_p = np.linalg.norm(diff_p, axis=1, keepdims=True)
                    np_n = np.linalg.norm(diff_n, axis=1, keepdims=True)
                    g_p = diff_p / np.maximum(np_p, 1e-12)
                    g_n = diff_n / np.maximum(np_n, 1e-12)
                lr = float(learning_rate)
                # pos: +g to h,r; -g to t  (increase distance contribution)
                # Actually we minimize margin + d_pos - d_neg, so:
                # ∂d_pos/∂h = g_p, ∂d_pos/∂r = g_p, ∂d_pos/∂t = -g_p
                # ∂(-d_neg)/∂h_n = -g_n, etc.
                for i in range(len(ah)):
                    ent[ah[i]] -= lr * g_p[i]
                    rel[ar[i]] -= lr * g_p[i]
                    ent[at[i]] += lr * g_p[i]
                    ent[anh[i]] += lr * g_n[i]
                    rel[anr[i]] += lr * g_n[i]
                    ent[ant[i]] -= lr * g_n[i]
            if batch_loss > 0:
                epoch_loss += batch_loss / max(1, int(neg_ratio))
                n_steps += 1
            _unit_norm_rows(ent)
        last_loss = epoch_loss / max(1, n_steps)
    return ent.astype(float), rel.astype(float), float(last_loss)


def fit_distmult(
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    *,
    n_entities: int,
    n_relations: int,
    embedding_dim: int = 50,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    margin: float = 1.0,
    neg_ratio: int = 1,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train DistMult embeddings with margin ranking loss on train triples.

    Uses uniform head-or-tail negative corruption on Session train triples only.

    Parameters
    ----------
    heads, relations, tails:
        Encoded train triple arrays.
    n_entities, n_relations:
        Vocabulary sizes.
    embedding_dim, epochs, batch_size, learning_rate:
        Training hyperparameters.
    margin:
        Margin for the ranking loss.
    neg_ratio:
        Negative samples drawn per positive triple.
    random_state:
        Seed for initialization and corruption sampling.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float]
        Entity embeddings, relation embeddings, and final epoch loss.

    Raises
    ------
    ValidationError
        When vocabulary sizes are too small for DistMult training.
    """
    if n_entities < 2 or n_relations < 1:
        raise ValidationError("DistMult needs ≥2 entities and ≥1 relation.")
    rng = np.random.default_rng(random_state)
    scale = 1.0 / np.sqrt(embedding_dim)
    ent = rng.normal(0.0, scale, size=(n_entities, embedding_dim))
    rel = rng.normal(0.0, scale, size=(n_relations, embedding_dim))

    n = len(heads)
    order = np.arange(n)
    last_loss = 0.0
    for _ in range(int(epochs)):
        rng.shuffle(order)
        epoch_loss = 0.0
        n_steps = 0
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            bh = heads[idx]
            br = relations[idx]
            bt = tails[idx]
            batch_loss = 0.0
            for _neg in range(max(1, int(neg_ratio))):
                nh, nr, nt = _corrupt_batch(bh, br, bt, n_entities, rng)
                pos = score_distmult(bh, br, bt, ent, rel)
                neg = score_distmult(nh, nr, nt, ent, rel)
                losses = np.maximum(0.0, margin - pos + neg)
                batch_loss += float(losses.mean())
                active = losses > 0
                if not active.any():
                    continue
                ah, ar, at = bh[active], br[active], bt[active]
                anh, anr, ant = nh[active], nr[active], nt[active]
                lr = float(learning_rate)
                # score = sum(h * r * t); ∂s/∂h = r*t, ∂s/∂r = h*t, ∂s/∂t = h*r
                # loss = max(0, margin - s_pos + s_neg)
                # ∇_pos = -∂s_pos, ∇_neg = +∂s_neg
                for i in range(len(ah)):
                    hp, rp, tp = ent[ah[i]], rel[ar[i]], ent[at[i]]
                    hn, rn, tn = ent[anh[i]], rel[anr[i]], ent[ant[i]]
                    # positive contribution: increase pos score
                    ent[ah[i]] += lr * (rp * tp)
                    rel[ar[i]] += lr * (hp * tp)
                    ent[at[i]] += lr * (hp * rp)
                    # negative contribution: decrease neg score
                    ent[anh[i]] -= lr * (rn * tn)
                    rel[anr[i]] -= lr * (hn * tn)
                    ent[ant[i]] -= lr * (hn * rn)
            if batch_loss > 0:
                epoch_loss += batch_loss / max(1, int(neg_ratio))
                n_steps += 1
            # Soft constraint: clip embeddings
            np.clip(ent, -2.0, 2.0, out=ent)
            np.clip(rel, -2.0, 2.0, out=rel)
        last_loss = epoch_loss / max(1, n_steps)
    return ent.astype(float), rel.astype(float), float(last_loss)


def fit_embeddings(
    method: str,
    heads: np.ndarray,
    relations: np.ndarray,
    tails: np.ndarray,
    *,
    n_entities: int,
    n_relations: int,
    embedding_dim: int = 50,
    epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 0.01,
    margin: float = 1.0,
    neg_ratio: int = 1,
    norm: NormName = "l1",
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Dispatch native KG embedding training to TransE or DistMult.

    Called by the native fit path after triple materialization on Session train
    triples only.

    Parameters
    ----------
    method:
        ``transe`` or ``distmult`` for the native numpy backend.
    heads, relations, tails:
        Encoded train triple arrays.
    n_entities, n_relations:
        Vocabulary sizes.
    embedding_dim, epochs, batch_size, learning_rate, margin, neg_ratio:
        Training hyperparameters forwarded to the method trainer.
    norm:
        Translation norm for TransE.
    random_state:
        Seed for initialization and negative sampling.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, float]
        Entity embeddings, relation embeddings, and final epoch loss.

    Raises
    ------
    ValidationError
        When ``method`` is not supported on the native backend.
    """
    if method == "transe":
        return fit_transe(
            heads,
            relations,
            tails,
            n_entities=n_entities,
            n_relations=n_relations,
            embedding_dim=embedding_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            margin=margin,
            neg_ratio=neg_ratio,
            norm=norm,
            random_state=random_state,
        )
    if method == "distmult":
        return fit_distmult(
            heads,
            relations,
            tails,
            n_entities=n_entities,
            n_relations=n_relations,
            embedding_dim=embedding_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            margin=margin,
            neg_ratio=neg_ratio,
            random_state=random_state,
        )
    raise ValidationError(f"Unknown KG method: {method!r}")
