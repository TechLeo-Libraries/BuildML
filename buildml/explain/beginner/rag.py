# ruff: noqa: E501
"""Beginner layers for retrieval-augmented generation."""

from __future__ import annotations

from buildml.explain.beginner._builder import CORE, FOUNDATION, BeginnerLayer, _index, _layer

RAG_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "rag-eval-contamination",
        plain=(
            "RAG evaluation asks whether your search finds the right passages. If the passage containing "
            "the answer to your test question is in the index *because* you put the test set in there, you "
            "are grading a search engine on documents you planted. The score becomes circular."
        ),
        analogy=(
            "Hiding an Easter egg yourself and then timing how long it takes you to find it. You will be "
            "impressively fast, and the time means nothing."
        ),
        steps=(
            "Decide what your evaluation set is: questions plus the passages that genuinely should answer them.",
            "Build the index from your real corpus, not from a corpus assembled around the questions.",
            "Keep any answer text written specifically for evaluation out of the indexed documents.",
            "Run retrieval and score how often the genuinely relevant passages appear near the top.",
            "Record what is in the index when you report the number, because it is part of the claim.",
        ),
        use=(
            "Before trusting any retrieval metric, and before comparing two retrieval configurations.",
            "Whenever the corpus and the evaluation set were built by the same process or the same person.",
        ),
        avoid=('Questions written from the corpus are useful controlled tests. Also evaluate representative user queries, and record how each set was constructed.', 'Do not add documents to the index between evaluation runs without re-baselining; the comparison is no longer like for like.'),
        myths=(
            (
                "Retrieval cannot leak because there is no training.",
                "Leakage is about information availability, not about gradient updates. A planted answer is available in exactly the way real answers are not.",
            ),
            (
                "High recall@k means the system will work for users.",
                "It means the system finds passages your evaluation says are right. Real user queries are messier, shorter, and less well aligned with your corpus.",
            ),
        ),
        example=(
            'from buildml import Session',
            '',
            'documents = [',
            "    {'doc_id': 'refunds', 'text': 'Refunds are available within thirty days of purchase.'},",
            "    {'doc_id': 'delivery', 'text': 'Standard delivery takes three to five business days.'},",
            ']',
            'session = Session()',
            'session.rag.ingest_corpus(documents)',
            'session.rag.chunk(size=160, overlap=16)',
            "session.rag.embed_and_index(embedder='hashing')",
            '# Toy relevance judgments remain separate from the indexed documents.',
            "qrels = {'refunds within thirty days': ['refunds'], 'standard delivery business days': ['delivery']}",
            'report = session.rag.evaluate(qrels, k=1)',
            'print(report.recall_at_k, report.mrr, report.ndcg_at_k)',
        ),
        check=(
            "Where did the documents containing your gold answers come from?",
            "Would a real user phrase the question the way your evaluation does?",
        ),
        tools=("rag_evaluate", "rag_ingest_corpus", "rag_embed_and_index", "rag_retrieve"),
        terms=("RAG", "recall@k", "leakage", "chunk"),
        difficulty=CORE,
    ),
    _layer(
        "rag-chunk-index-boundary",
        plain=(
            "RAG has two preparation steps that people often confuse with training. Chunking cuts documents "
            "into passages small enough to search. Indexing turns those passages into vectors and stores "
            "them for fast lookup. Neither is fitting a model, and neither is a Session checkpoint."
        ),
        analogy=(
            "Chunking is cutting a book into paragraphs; indexing is writing the index at the back. Neither "
            "step teaches anyone the content: they just make it findable."
        ),
        steps=(
            "Ingest your corpus so BuildML knows what documents exist.",
            "Chunk: choose a passage size and an overlap so a sentence spanning a boundary is not lost.",
            "Embed and index: each chunk becomes a vector, stored in a searchable structure.",
            "Retrieve: a query becomes a vector, and the index returns the nearest chunks.",
            "Save a RAG bundle if you want the index to survive the session: a checkpoint will not carry it.",
        ),
        use=(
            "Whenever you want a language model to answer from your documents rather than from its training data.",
            "For search over an internal knowledge base, even without generation on top.",
        ),
        avoid=(
            "Do not chunk so small that a passage loses its context, or so large that a retrieved chunk is mostly irrelevant text.",
            "Do not treat the index as a model artifact you can hand to a classical pipeline; it belongs to the RAG surface.",
        ),
        myths=(('Building an index is training a model.', 'Index preparation does not train a supervised predictor. BuildML can use fixed token hashing or a pretrained embedding model; record which backend produced the vectors.'), ('Bigger chunks are safer because they contain more context.', 'Larger chunks can preserve context but also include irrelevant material. Compare chunk sizes using retrieval and answer-quality evaluations.')),
        example=(
            'from buildml import Session',
            '',
            'documents = [',
            "    {'doc_id': 'refunds', 'text': 'Refunds are available within thirty days of purchase.'},",
            "    {'doc_id': 'delivery', 'text': 'Standard delivery takes three to five business days.'},",
            ']',
            'session = Session()',
            'session.rag.ingest_corpus(documents)',
            'session.rag.chunk(size=160, overlap=16)',
            "session.rag.embed_and_index(embedder='hashing')",
            "hits = session.rag.retrieve('refunds within thirty days', k=1, mode='dense')",
            'print(hits.hits[0].doc_id)',
        ),
        check=(
            "Does one of your chunks, read alone, make sense to a human?",
            "Which artifact would you reload tomorrow to get your index back?",
        ),
        tools=("rag_chunk", "rag_embed_and_index", "rag_retrieve", "save_rag_bundle"),
        terms=("RAG", "chunk", "embedding", "vector index"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "rag-retrieval-metrics",
        plain=(
            'Retrieval metrics score a ranked list, not a yes/no answer. Recall@k measures the fraction of relevant passages retrieved in the first k results. MRR summarizes the rank of the first relevant result. nDCG@k rewards putting the most relevant material at the very top. None of them is accuracy.'
        ),
        analogy=(
            "Judging a librarian. Did they hand you a shelf containing the right book (recall@k)? Was it "
            "near the top of the pile (MRR)? Were the best books first (nDCG)?"
        ),
        steps=('Assemble questions paired with the passages that should answer them.', 'Choose k to match how many passages you actually feed the language model.', 'Compute recall@k to check whether the retrieved context contains the relevant passages needed for a supported answer.', 'Then look at MRR and nDCG@k to see whether ranking, not coverage, is the weak point.', 'Compare configurations at the same k, on the same questions, over the same corpus.'),
        use=('Whenever you change the embedding model, the chunk size, or the number of retrieved passages.', 'To distinguish missing or poorly ranked context from errors introduced during answer generation.'),
        avoid=(
            "Do not report these metrics as end-to-end answer quality; a perfect retrieval score says nothing about what the language model then wrote.",
            "Do not compare recall@5 against recall@20 as if they were the same measurement.",
        ),
        myths=(('Retrieval metrics measure whether the answer is correct.', 'They measure whether the right source was found. Generation quality is a separate evaluation with separate failure modes.'), ('A higher k is always better.', 'For a fixed ranked list, recall cannot decrease as k increases, but longer context may contain irrelevant passages. Evaluate the effect on generated answers separately.')),
        example=(
            'from buildml import Session',
            '',
            'documents = [',
            "    {'doc_id': 'refunds', 'text': 'Refunds are available within thirty days of purchase.'},",
            "    {'doc_id': 'delivery', 'text': 'Standard delivery takes three to five business days.'},",
            ']',
            'session = Session()',
            'session.rag.ingest_corpus(documents)',
            'session.rag.chunk(size=160, overlap=16)',
            "session.rag.embed_and_index(embedder='hashing')",
            '# Toy relevance judgments remain separate from the indexed documents.',
            "qrels = {'refunds within thirty days': ['refunds'], 'standard delivery business days': ['delivery']}",
            'report = session.rag.evaluate(qrels, k=1)',
            'print(report.recall_at_k, report.mrr, report.ndcg_at_k)',
        ),
        check=(
            "Is your bottleneck coverage (recall) or ordering (MRR)?",
            "How many passages does your generation step actually receive?",
        ),
        tools=("rag_evaluate", "rag_retrieve", "rag_generate", "rag_embed_and_index"),
        terms=("recall@k", "MRR", "nDCG", "RAG", "chunk"),
        difficulty=CORE,
    ),
)

__all__ = ["RAG_BEGINNER"]
