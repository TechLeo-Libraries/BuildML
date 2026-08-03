# ruff: noqa: E501
"""Rag concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

RAG_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="rag-eval-contamination",
            title="RAG eval contamination",
            summary="Evaluation answers must stay out of the indexed corpus or retrieval metrics become circular.",
            definition=(
                "Eval contamination in retrieval is indexing documents that contain labeled answers or "
                "passages used only for evaluation queries, so the index can return the answer by identity "
                "rather than by generalization over held-out text."
            ),
            intuition=(
                "If the answer sheet is in the library, finding it is not evidence that search works on "
                "new questions. Mark eval-only documents and keep them out of the index build."
            ),
            formal_idea=(
                "Let C_index be the indexed corpus and Q_eval the evaluation query set with relevance "
                "labels over documents D_eval. Require C_index ∩ D_eval = ∅ for any claim that retrieval "
                "metrics estimate performance on unseen answers."
            ),
            why_it_matters=(
                "Contaminated indexes inflate recall@k and MRR without improving real retrieval.",
                "Teaching and model cards need an explicit corpus vs eval-query disclosure.",
            ),
            how_buildml_uses=(
                "Documents may carry role=index or role=eval_only.",
                "rag_embed_and_index raises LeakageError when any eval_only document is present.",
                "Catalog leakage fields warn against indexing labeled eval answers.",
            ),
            interpretation_rules=(
                "Read which corpus was indexed beside every recall@k / MRR number.",
                "If eval answers were indexed, treat metrics as invalid for generalization claims.",
            ),
            assumptions=(
                "Callers label eval-only material before indexing.",
                "Qrels refer to document ids that exist in the index corpus when measuring hits.",
            ),
            failure_modes=(
                "Silently concatenating a FAQ answer key into the index folder.",
                "Reusing the same docs for indexing and as the only relevant targets without disclosure.",
            ),
            anti_patterns=(
                "Indexing eval_only documents to 'make the demo look good'.",
            ),
            worked_example_pattern=(
                "rag_ingest_corpus(index docs) → rag_embed_and_index → rag_evaluate(qrels on held-out queries).",
            ),
            related_concepts=("leakage-boundary", "rag-chunk-index-boundary", "evaluation-partitions"),
        ),
        _note(
            key="rag-chunk-index-boundary",
            title="RAG chunk and index boundary",
            summary="Chunking and indexing are retrieval prep steps; they are not classical fit and not a Session checkpoint.",
            definition=(
                "The chunk/index boundary is the contract that documents are split into chunks, embedded, "
                "and stored in a vector index artifact separate from Session workflow checkpoints and "
                "Torch trainer bundles."
            ),
            intuition=(
                "Think of the index as a searchable card catalog built from the books you chose to shelve. "
                "Saving your lab notebook (Session checkpoint) does not shelve the books for you."
            ),
            formal_idea=(
                "A RAG bundle records chunk config, embedder id/dim, chunk metadata, and embeddings under "
                "schema buildml.rag_bundle.v1. Session checkpoints omit that payload; Torch bundles are "
                "orthogonal supervised-training artifacts."
            ),
            why_it_matters=(
                "Mixing artifact kinds causes failed loads and false resume expectations.",
                "Chunk size/overlap change the retrieval unit; ids must stay deterministic for audits.",
            ),
            how_buildml_uses=(
                "Session.rag_chunk / rag_embed_and_index build an in-memory RagIndex.",
                "save_rag_bundle / load_rag_bundle round-trip buildml.rag_bundle.v1.",
                "Wrong schema ids raise ValidationError with an explicit expected format.",
            ),
            interpretation_rules=(
                "Never imply checkpoint_load restored a vector index.",
                "State embedder id and dimension beside retrieve/eval results.",
            ),
            assumptions=(
                "Index corpus membership is fixed for a given bundle.",
                "Query embedding uses a compatible embedder after load.",
            ),
            failure_modes=(
                "Passing a Session checkpoint path to load_rag_bundle.",
                "Changing chunk config between index build and eval without rebuilding.",
            ),
            anti_patterns=(
                "Embedding the vector index inside a Session checkpoint.",
            ),
            worked_example_pattern=(
                "rag_ingest_corpus → rag_chunk → rag_embed_and_index → save_rag_bundle.",
            ),
            related_concepts=("rag-eval-contamination", "reproducibility", "leakage-boundary"),
        ),
        _note(
            key="rag-retrieval-metrics",
            title="RAG retrieval metrics",
            summary=(
                "recall@k, MRR, nDCG@k, and hit-rate@k measure ranking quality against gold labels; "
                "they are not classification accuracy."
            ),
            definition=(
                "Retrieval metrics score whether relevant documents or chunks appear in the top-k ranked "
                "hits for each evaluation query, using gold relevance labels (qrels)."
            ),
            intuition=(
                "Ask whether the right book showed up near the top of the search results:not whether a "
                "classifier predicted a class label."
            ),
            formal_idea=(
                "For query q with relevant set R_q, recall@k = |{ids in top-k} ∩ R_q| / |R_q|. "
                "MRR averages 1/rank of the first relevant hit. nDCG@k discounts later ranks; "
                "hit-rate@k is the fraction of queries with at least one relevant hit."
            ),
            why_it_matters=(
                "A single unlabeled 'accuracy' hides k, relevance mode, and corpus identity.",
                "Document-level vs chunk-level relevance change what a hit means.",
            ),
            how_buildml_uses=(
                "rag_evaluate supports relevance_mode=document (default) or chunk, plus retrieve mode overrides.",
                "evaluate_generation scores faithfulness + answer relevance with EchoGroundedProvider for CI.",
                "RagEvalResult exposes recall_at_k, mrr, ndcg_at_k, hit_rate_at_k, and disclosures.",
                "compare_retrieval_configs rebuilds indexes per config row for side-by-side metrics.",
            ),
            interpretation_rules=(
                "Always read metrics with k, relevance_mode, and retrieve_mode.",
                "Do not call recall@k 'accuracy'.",
            ),
            assumptions=(
                "Qrels ids match the claimed relevance_mode (doc_id or chunk_id).",
                "The same embedder/index pair is used for every eval query in the run.",
            ),
            failure_modes=(
                "Comparing recall@5 from one embedder to recall@20 from another without disclosure.",
                "Treating hashing-embedder demos as semantic retrieval quality claims.",
            ),
            anti_patterns=(
                "Reporting only top-1 hit rate as proof the RAG system is production-ready.",
            ),
            worked_example_pattern=(
                "rag_embed_and_index → rag_evaluate(qrels, k=5) → read recall_at_k, mrr, and ndcg_at_k.",
            ),
            related_concepts=("rag-eval-contamination", "rag-chunk-index-boundary", "evaluation-partitions"),
        ),
    )
}

