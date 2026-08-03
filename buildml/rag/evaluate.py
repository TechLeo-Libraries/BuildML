"""Measure whether retrieval and generation are actually working.

Reading a few results and deciding they look reasonable is how most RAG systems
get tuned, and it is unreliable in a specific way: you look at the queries you
thought of, and the failures are on the ones you did not. Measuring against a
labelled set replaces that with a number you can compare between configurations.

The labels are called *qrels*: for each query, which documents or chunks should
have come back. Building fifty of them by hand is a couple of hours of work and
is almost always worth it, because without them every tuning decision is a
guess.

Four retrieval metrics, each answering a different question:

``recall@k``
    Of the documents that should have been found, what fraction were? The
    metric that matters when the answer might be spread across several sources.
``hit_rate@k``
    Did *anything* relevant come back? The floor: if this is low, nothing
    downstream can work.
``MRR``
    How high up was the first relevant result? Matters when the model reads the
    top passage most carefully, which it does.
``nDCG@k``
    Position-weighted credit across all relevant results, discounted by rank.
    The most complete single number when you want just one.

Generation evaluation is deliberately more modest. The heuristics here are
lexical: token overlap against a reference answer, citation coverage: and they
catch gross failure rather than judging quality. They are honest about that
rather than dressing up a weak signal as a score.

See Also
--------
buildml.rag.retrieve.retrieve : What is being measured.
buildml.rag.results.RagEvalResult : The metrics, with per-query detail.
buildml.rag.corpus : Holding documents out via ``role="eval_only"``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from buildml.core.errors import ValidationError
from buildml.rag.generate import EchoGroundedProvider, generate_grounded, score_faithfulness
from buildml.rag.index import RagIndex, build_index
from buildml.rag.results import ConfigCompareResult, CorpusHandle, RagEvalResult, RagGenerateEvalResult
from buildml.rag.retrieve import retrieve
from buildml.rag.types import EvalConfig, RelevanceMode, RetrieveConfig


def _normalize_qrels(
    qrels: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[str]],
    *,
    relevance_mode: RelevanceMode,
) -> list[tuple[str, set[str]]]:
    """Accept the several shapes people write qrels in, return one shape.

    Labels arrive as a dict of query to IDs, or as a list of row mappings using
    any of ``relevant_doc_ids``, ``relevant_docs``, or a single ``doc_id`` (and
    the ``chunk`` equivalents). Accepting all of them costs a little branching
    here and saves every caller from reformatting.

    Parameters
    ----------
    qrels:
        The gold labels, in any supported shape.
    relevance_mode:
        Whether IDs name documents or chunks. Determines which keys are read.

    Returns
    -------
    list of tuple
        ``(query, relevant_ids)`` pairs with IDs as strings.

    Raises
    ------
    ValidationError
        If a row is not a mapping, lacks a query, carries no recognised
        relevance key, has an empty relevance set, or the whole set is empty.

    Notes
    -----
    **Empty relevance sets are rejected rather than skipped.** A query with no
    relevant documents cannot be scored: recall would divide by zero: and
    silently dropping it would inflate the average over the rest.

    **IDs are stringified**, so integer document IDs in the labels match string
    IDs from the corpus.
    """
    pairs: list[tuple[str, set[str]]] = []
    if isinstance(qrels, Mapping):
        for query, docs in qrels.items():
            relevant = {str(d) for d in docs}
            if not relevant:
                raise ValidationError(f"qrels entry for {query!r} has no relevant ids.")
            pairs.append((str(query), relevant))
        return pairs
    for i, row in enumerate(qrels):
        if not isinstance(row, Mapping):
            raise ValidationError(f"qrels[{i}] must be a mapping.")
        query = row.get("query")
        if query is None:
            raise ValidationError(f"qrels[{i}] is missing 'query'.")
        relevant: set[str]
        if relevance_mode == "chunk":
            if "relevant_chunk_ids" in row:
                relevant = {str(d) for d in row["relevant_chunk_ids"]}
            elif "relevant_chunks" in row:
                relevant = {str(d) for d in row["relevant_chunks"]}
            elif "chunk_id" in row:
                relevant = {str(row["chunk_id"])}
            else:
                raise ValidationError(
                    f"qrels[{i}] needs relevant_chunk_ids, relevant_chunks, or "
                    "chunk_id when relevance_mode='chunk'."
                )
        elif "relevant_doc_ids" in row:
            relevant = {str(d) for d in row["relevant_doc_ids"]}
        elif "relevant_docs" in row:
            relevant = {str(d) for d in row["relevant_docs"]}
        elif "doc_id" in row:
            relevant = {str(row["doc_id"])}
        else:
            raise ValidationError(
                f"qrels[{i}] needs relevant_doc_ids, relevant_docs, or doc_id."
            )
        if not relevant:
            raise ValidationError(f"qrels[{i}] has an empty relevance set.")
        pairs.append((str(query), relevant))
    if not pairs:
        raise ValidationError("qrels is empty.")
    return pairs


def _dcg(relevances: Sequence[float]) -> float:
    """Sum relevance gains, discounted by how far down the list they appear.

    Discounted cumulative gain. Each position contributes
    ``(2**rel - 1) / log2(rank + 1)``, so a relevant result at rank 1 is worth
    considerably more than the same result at rank 10: which matches how people
    read ranked lists.

    Parameters
    ----------
    relevances:
        Relevance grades in rank order. Binary here: 1.0 or 0.0.

    Returns
    -------
    float
        The discounted sum. Unbounded, hence the normalisation in
        :func:`_ndcg_at_k`.
    """
    total = 0.0
    for i, rel in enumerate(relevances, start=1):
        if rel <= 0:
            continue
        total += (2.0**rel - 1.0) / math.log2(i + 1.0)
    return total


def _ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], *, k: int) -> float:
    """Score a ranking against the best ranking that was possible.

    Normalises :func:`_dcg` by the DCG of a perfect ordering, which puts the
    result in ``[0, 1]`` and makes queries comparable even when they have
    different numbers of relevant documents: a query with one relevant
    document cannot lose points for not filling all ``k`` slots.

    Parameters
    ----------
    ranked_ids:
        Retrieved IDs, best first. Truncated to ``k``.
    relevant:
        The IDs that should have been retrieved.
    k:
        Cutoff.

    Returns
    -------
    float
        1.0 for a perfect ranking, 0.0 when nothing relevant appears in the top
        ``k`` or nothing is relevant at all.

    Notes
    -----
    **Relevance is binary here.** Real nDCG supports graded relevance, which
    would need graded labels; the qrels format collects sets, not grades.
    """
    gains = [1.0 if item in relevant else 0.0 for item in ranked_ids[:k]]
    dcg = _dcg(gains)
    ideal = _dcg([1.0] * min(len(relevant), k))
    if ideal <= 0:
        return 0.0
    return dcg / ideal


def evaluate_retrieval(
    index: RagIndex,
    qrels: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[str]],
    *,
    k: int = 5,
    relevance_mode: RelevanceMode = "document",
    retrieve_config: RetrieveConfig | None = None,
    mode: str | None = None,
) -> RagEvalResult:
    """Score retrieval against labelled queries.

    Runs every query in the qrels through the index and computes recall@k,
    MRR, nDCG@k, and hit-rate@k, plus a per-query breakdown. The per-query rows
    are usually the more useful half: an average hides which queries fail, and
    the failures are what you fix.

    Parameters
    ----------
    index:
        The index to evaluate.
    qrels:
        Gold labels: query to relevant IDs, in any supported shape.
    k:
        Cutoff for every metric. Should match what you retrieve in production.
    relevance_mode:
        ``'document'`` scores by parent document, ``'chunk'`` by exact chunk.
    retrieve_config:
        Retrieval settings. Its ``k`` is overridden by the ``k`` argument.
    mode:
        Retrieval mode override, for comparing modes on one index.

    Returns
    -------
    RagEvalResult
        Averaged metrics, per-query detail, and the settings that produced them.

    Raises
    ------
    ValidationError
        If there is no index, ``k`` is not positive, ``relevance_mode`` is
        unrecognised, or the qrels are malformed or empty.

    Notes
    -----
    **Document mode deduplicates before scoring.** Three chunks from the same
    document count once and take the best rank, so a document that chunked
    finely does not crowd out the ranking. Chunk mode does not, and is the
    stricter test.

    **Which mode to use depends on the question.** Document mode reflects
    whether the right source was found, which is what usually matters. Chunk
    mode reflects whether the right passage was found, which matters when
    chunking is what you are tuning.

    **These measure ranking, not answers.** Perfect retrieval does not
    guarantee a good generated answer; see :func:`evaluate_generation`.

    **Cost is one retrieval per query**, so reranking makes evaluation slow in
    proportion to the label set.

    Examples
    --------
    Evaluate at 5, then read the failures::

        result = evaluate_retrieval(index, qrels, k=5)
        print(result.recall_at_k, result.mrr)
        for row in result.per_query:
            if not row["hit"]:
                print("missed:", row["query"])

    See Also
    --------
    compare_retrieval_configs : Sweeping several configurations at once.
    buildml.rag.results.RagEvalResult : The fields in full.
    """
    if index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    if k <= 0:
        raise ValidationError(f"k must be positive; got {k}")
    if relevance_mode not in {"document", "chunk"}:
        raise ValidationError(
            f"relevance_mode must be 'document' or 'chunk'; got {relevance_mode!r}"
        )
    pairs = _normalize_qrels(qrels, relevance_mode=relevance_mode)
    cfg = retrieve_config or RetrieveConfig(k=k)
    if mode is not None:
        cfg = RetrieveConfig(
            k=k,
            mode=mode,  # type: ignore[arg-type]
            fusion=cfg.fusion,
            rrf_k=cfg.rrf_k,
            dense_weight=cfg.dense_weight,
            bm25_k1=cfg.bm25_k1,
            bm25_b=cfg.bm25_b,
            filters=cfg.filters,
            rerank=cfg.rerank,
            rerank_model=cfg.rerank_model,
            rerank_candidates=cfg.rerank_candidates,
        )
    else:
        cfg = RetrieveConfig(
            k=k,
            mode=cfg.mode,
            fusion=cfg.fusion,
            rrf_k=cfg.rrf_k,
            dense_weight=cfg.dense_weight,
            bm25_k1=cfg.bm25_k1,
            bm25_b=cfg.bm25_b,
            filters=cfg.filters,
            rerank=cfg.rerank,
            rerank_model=cfg.rerank_model,
            rerank_candidates=cfg.rerank_candidates,
        )

    recalls: list[float] = []
    rr_values: list[float] = []
    ndcgs: list[float] = []
    hits_flags: list[float] = []
    per_query: list[dict[str, Any]] = []
    for query, relevant in pairs:
        result = retrieve(index, query, k=k, config=cfg)
        if relevance_mode == "chunk":
            ranked_ids = [h.chunk_id for h in result.hits]
        else:
            ranked_ids = [h.doc_id for h in result.hits]
        # Deduplicate doc ids for document-mode recall/MRR while keeping first rank.
        if relevance_mode == "document":
            seen: set[str] = set()
            unique_ranked: list[str] = []
            for doc_id in ranked_ids:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                unique_ranked.append(doc_id)
            ranked_for_metrics = unique_ranked
        else:
            ranked_for_metrics = ranked_ids

        hit_set = set(ranked_for_metrics) & relevant
        recall = len(hit_set) / len(relevant)
        recalls.append(recall)
        rr = 0.0
        for rank, item_id in enumerate(ranked_for_metrics, start=1):
            if item_id in relevant:
                rr = 1.0 / rank
                break
        rr_values.append(rr)
        ndcg = _ndcg_at_k(ranked_for_metrics, relevant, k=k)
        ndcgs.append(ndcg)
        hit_flag = 1.0 if hit_set else 0.0
        hits_flags.append(hit_flag)
        per_query.append(
            {
                "query": query,
                "relevant_ids": sorted(relevant),
                "retrieved_ids": ranked_for_metrics,
                "recall_at_k": recall,
                "rr": rr,
                "ndcg_at_k": ndcg,
                "hit": bool(hit_set),
                # Backward-compatible aliases used by M1 tests/readers.
                "relevant_doc_ids": sorted(relevant) if relevance_mode == "document" else [],
                "retrieved_doc_ids": ranked_for_metrics if relevance_mode == "document" else [],
            }
        )
    n = len(pairs)
    return RagEvalResult(
        n_queries=n,
        k=k,
        recall_at_k=float(sum(recalls) / n),
        mrr=float(sum(rr_values) / n),
        ndcg_at_k=float(sum(ndcgs) / n),
        hit_rate_at_k=float(sum(hits_flags) / n),
        per_query=tuple(per_query),
        relevance_mode=relevance_mode,
        retrieve_mode=cfg.mode,
        disclosures=(
            f"relevance_mode={relevance_mode}",
            f"retrieve_mode={cfg.mode}",
            f"k={k}",
            f"embedder_id={index.embed_config.embedder_id}",
            "recall@k / MRR / nDCG@k are ranking metrics, not classification accuracy.",
        ),
        warnings=(),
    )


def compare_retrieval_configs(
    corpus: CorpusHandle,
    configs: Sequence[Mapping[str, Any] | EvalConfig],
    qrels: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[str]],
    *,
    k: int = 5,
    relevance_mode: RelevanceMode = "document",
) -> ConfigCompareResult:
    """Build an index per configuration and score them all on the same labels.

    The way to answer "should I chunk at 500 or 1000?" and "is hybrid worth it
    here?" with evidence. Each configuration gets its own index built from the
    same corpus and is evaluated against the same qrels, so the only thing that
    varies is the configuration.

    Parameters
    ----------
    corpus:
        The documents. Re-chunked and re-embedded per configuration.
    configs:
        One entry per configuration. Each may be an
        :class:`~buildml.rag.types.EvalConfig`, or a mapping with any of
        ``name``, ``chunk_size``, ``chunk_overlap``, ``embedder``, ``retrieve``,
        ``relevance_mode``, and ``k``. Anything omitted takes the default.
    qrels:
        Gold labels, shared across every configuration.
    k:
        Default cutoff, overridable per row.
    relevance_mode:
        Default scoring granularity, overridable per row.

    Returns
    -------
    ConfigCompareResult
        One row per configuration with its settings and metrics side by side.

    Raises
    ------
    ValidationError
        If ``configs`` is empty, or any index build or evaluation fails.

    Notes
    -----
    **Every row rebuilds the index from scratch**, which is the point: chunk
    size changes the chunks, so the indexes genuinely differ: and it is why
    this is slow. Cost is roughly the number of configurations times the corpus
    size, and with a semantic embedder every row re-embeds everything.

    **Vary one thing at a time if you want to attribute the difference.** A row
    that changes chunk size and embedder together tells you the pair is better,
    not which half did the work.

    **Small differences are noise.** With fifty labelled queries, a recall gap
    of a point or two is within sampling error. Trust gaps that are large or
    that persist across ``k``.

    Examples
    --------
    Compare two chunk sizes::

        result = compare_retrieval_configs(
            corpus,
            [{"name": "small", "chunk_size": 300},
             {"name": "large", "chunk_size": 1000}],
            qrels,
            k=5,
        )
        for row in result.rows:
            print(row["name"], row["recall_at_k"], row["ndcg_at_k"])

    See Also
    --------
    evaluate_retrieval : Scoring one index.
    buildml.rag.results.ConfigCompareResult : The comparison table.
    """
    if not configs:
        raise ValidationError("compare_retrieval_configs requires at least one config.")
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(configs):
        if isinstance(item, EvalConfig):
            name = f"config-{i}"
            chunk_size = None
            chunk_overlap = None
            embedder = None
            retrieve_cfg = item.retrieve
            mode = item.relevance_mode
            eval_k = item.k
        else:
            name = str(item.get("name") or f"config-{i}")
            chunk_size = item.get("chunk_size")
            chunk_overlap = item.get("chunk_overlap")
            embedder = item.get("embedder")
            retrieve_raw = item.get("retrieve")
            if isinstance(retrieve_raw, RetrieveConfig):
                retrieve_cfg = retrieve_raw
            elif isinstance(retrieve_raw, Mapping):
                retrieve_cfg = RetrieveConfig.from_dict(dict(retrieve_raw))
            else:
                retrieve_cfg = RetrieveConfig(k=k)
            mode = item.get("relevance_mode") or relevance_mode
            eval_k = int(item.get("k") or k)
        index = build_index(
            corpus,
            chunk_size=None if chunk_size is None else int(chunk_size),
            chunk_overlap=None if chunk_overlap is None else int(chunk_overlap),
            embedder=embedder,
        )
        metrics = evaluate_retrieval(
            index,
            qrels,
            k=eval_k,
            relevance_mode=mode,  # type: ignore[arg-type]
            retrieve_config=retrieve_cfg,
        )
        rows.append(
            {
                "name": name,
                "n_chunks": index.to_index_result().n_chunks,
                "embedder_id": index.embed_config.embedder_id,
                "dim": index.embed_config.dim,
                "chunk_config": index.chunk_config.to_dict(),
                "retrieve_mode": metrics.retrieve_mode,
                "relevance_mode": metrics.relevance_mode,
                "k": metrics.k,
                "recall_at_k": metrics.recall_at_k,
                "mrr": metrics.mrr,
                "ndcg_at_k": metrics.ndcg_at_k,
                "hit_rate_at_k": metrics.hit_rate_at_k,
            }
        )
    return ConfigCompareResult(
        rows=tuple(rows),
        k=k,
        relevance_mode=relevance_mode,
        disclosures=(
            f"Compared {len(rows)} retrieval config(s).",
            f"relevance_mode={relevance_mode}",
            f"k={k}",
            "Each row rebuilds its own index; scores are not shared across rows.",
        ),
    )


def _token_overlap(a: str, b: str) -> float:
    """Measure what fraction of ``a``'s words also appear in ``b``.

    A crude proxy for "does this answer resemble the reference?". Asymmetric by
    design: dividing by ``a``'s tokens means a generated answer is penalised for
    words the reference does not contain, but not for reference words it omits.

    Parameters
    ----------
    a:
        The text being scored, usually the generated answer.
    b:
        The text compared against, usually the reference.

    Returns
    -------
    float
        Overlap fraction in ``[0, 1]``. Zero when either side is empty.

    Notes
    -----
    **This does not understand meaning.** A correct paraphrase using different
    words scores low; a wrong answer reusing the reference's vocabulary scores
    high. It is a smoke test, not a judge.
    """
    import re

    ta = {t for t in re.findall(r"[A-Za-z0-9_]+", a.lower()) if t}
    tb = {t for t in re.findall(r"[A-Za-z0-9_]+", b.lower()) if t}
    if not ta or not tb:
        return 0.0
    return float(len(ta & tb) / len(ta))


def evaluate_generation(
    index: RagIndex,
    examples: Sequence[Mapping[str, Any]],
    *,
    k: int = 5,
    retrieve_config: RetrieveConfig | None = None,
    provider: Any | None = None,
) -> RagGenerateEvalResult:
    """Score generated answers against reference answers, cheaply.

    Runs the full retrieve-and-generate path for each example and reports mean
    faithfulness, mean answer relevance, and citation coverage, with per-example
    detail.

    Be clear about what this is. Both signals are lexical: faithfulness checks
    citation markers and answer-to-context overlap, relevance checks
    answer-to-reference overlap. They detect a system that has stopped working :
    answers ignoring their context, answers unrelated to the reference: and
    they cannot rank two reasonable answers. Treat movement as a signal to go
    read the outputs, not as a quality score.

    Parameters
    ----------
    index:
        The index to retrieve from.
    examples:
        One mapping per example, each with ``query`` and ``reference_answer``
        (or ``answer``).
    k:
        Passages retrieved per query.
    retrieve_config:
        Retrieval settings; defaults resolve from the current install.
    provider:
        The chat provider. Defaults to :class:`EchoGroundedProvider`, which
        needs no network: useful for testing the plumbing, meaningless as a
        quality measurement.

    Returns
    -------
    RagGenerateEvalResult
        Averaged scores, per-example detail, and disclosures naming the
        heuristics used.

    Raises
    ------
    ValidationError
        If there is no index, ``examples`` is empty, or an example lacks a query
        or reference answer.

    Notes
    -----
    **The default provider makes the numbers meaningless.** Echo answers cite
    their sources and say nothing, which scores respectably on both heuristics.
    Pass a real provider to measure anything real.

    **A real provider means one API call per example**, with the cost and
    latency that implies.

    **Fix retrieval first.** These scores are bounded above by whether the right
    passages were retrieved; run :func:`evaluate_retrieval` before spending
    effort on prompts.

    Examples
    --------
    Score against a small labelled set::

        examples = [
            {"query": "refund window?", "reference_answer": "30 days"},
        ]
        result = evaluate_generation(index, examples, provider=provider)
        print(result.mean_faithfulness, result.mean_answer_relevance)

    See Also
    --------
    evaluate_retrieval : Measuring the stage this depends on.
    buildml.rag.generate.score_faithfulness : The per-answer heuristic.
    """
    if index is None:
        raise ValidationError("No RAG index. Call rag_embed_and_index(...) first.")
    if not examples:
        raise ValidationError("evaluate_generation requires at least one example.")
    from buildml.rag.defaults import default_retrieve_config

    cfg = retrieve_config or default_retrieve_config(k=k)
    resolved_provider = provider or EchoGroundedProvider()
    faith_scores: list[float] = []
    rel_scores: list[float] = []
    cite_cov: list[float] = []
    per_query: list[dict[str, Any]] = []
    for i, row in enumerate(examples):
        if not isinstance(row, Mapping):
            raise ValidationError(f"examples[{i}] must be a mapping.")
        query = row.get("query")
        reference = row.get("reference_answer") or row.get("answer")
        if not query or not reference:
            raise ValidationError(f"examples[{i}] needs query and reference_answer.")
        gen = generate_grounded(
            index,
            str(query),
            resolved_provider,
            k=k,
            retrieve_config=cfg,
        )
        faith = gen.faithfulness or score_faithfulness(
            gen.answer,
            gen.citations,
            context=gen.prompt_context,
        )
        relevance = _token_overlap(gen.answer, str(reference))
        faith_scores.append(faith.score)
        rel_scores.append(relevance)
        cite_cov.append(faith.citation_marker_coverage)
        per_query.append(
            {
                "query": query,
                "reference_answer": reference,
                "answer": gen.answer,
                "faithfulness": faith.to_dict(),
                "answer_relevance": relevance,
            }
        )
    n = len(examples)
    return RagGenerateEvalResult(
        n_queries=n,
        mean_faithfulness=float(sum(faith_scores) / n),
        mean_answer_relevance=float(sum(rel_scores) / n),
        citation_coverage=float(sum(cite_cov) / n),
        per_query=tuple(per_query),
        disclosures=(
            f"retrieve_mode={cfg.mode}",
            f"k={k}",
            "Faithfulness and answer relevance are cheap lexical heuristics, not NLI judges.",
        ),
        warnings=(),
    )
