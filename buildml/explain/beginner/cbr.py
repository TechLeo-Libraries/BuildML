# ruff: noqa: E501
"""Beginner layers for case-based reasoning."""

from __future__ import annotations

from buildml.explain.beginner._builder import CORE, FOUNDATION, BeginnerLayer, _index, _layer

CBR_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "cbr-case-memory",
        plain=(
            "Case-based reasoning does not build a model in the usual sense. It remembers your training "
            "rows as solved cases, and when a new row arrives it looks up the most similar past cases and "
            "reuses their answers. The memory is built from training rows only."
        ),
        analogy=(
            "A claims adjuster with twenty years of files. Faced with a new claim, they do not consult a "
            "formula: they recall the three most similar claims they have handled and what happened."
        ),
        steps=(
            "Each training row becomes a case: its features plus the known outcome.",
            "The case memory is assembled from the training partition, never from validation or test.",
            "A new query row is compared with every case using a distance measure.",
            "The `k` closest cases are retrieved.",
            "Their outcomes are combined into a prediction, and the trace records which cases were used.",
        ),
        use=(
            "When you must be able to say 'we decided this because of these three specific past cases'.",
            "When the decision boundary is irregular and lumpy, which distance-based lookup handles naturally.",
        ),
        avoid=(
            "Do not use it with very many rows or very many columns; every prediction scans the memory, and distances become meaningless in high dimensions.",
            "Do not use it when your features are on wildly different scales and you have not standardized them.",
        ),
        myths=(
            (
                "Case-based reasoning is tabular retrieval-augmented generation.",
                "There is no language model and no text. It retrieves labelled rows and reuses their outcomes. Different module, different bundle, different failure modes.",
            ),
            (
                "The training score tells you how well it works.",
                "Every training row is its own nearest neighbour, so the training score is close to perfect by construction and tells you nothing.",
            ),
        ),
        example=(
            "session.cbr.fit(metric='euclidean', k=5)",
            "result = session.cbr.predict(partition='test', return_traces=True)",
            "print(result.traces[0].neighbor_case_ids, result.traces[0].distances)",
        ),
        check=(
            "Are your features scaled? Otherwise one large-valued column decides every distance.",
            "How large is your case memory, and how fast does a prediction need to be?",
        ),
        tools=("fit_cbr", "retrieve_cases", "predict_cbr", "evaluate_cbr"),
        terms=("case-based reasoning", "nearest neighbours", "distance metric", "leakage"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "cbr-retrieve-reuse",
        plain=(
            "The classic cycle has four steps: retrieve similar cases, reuse their answers, optionally "
            "adapt the result to the new situation, and optionally retain the new case once you learn its "
            "true outcome. BuildML implements all four with guardrails on the last one."
        ),
        analogy=(
            "Look up similar past jobs, quote roughly what they cost, adjust for this customer's specifics, "
            "then file the completed job so the next quote is better informed."
        ),
        steps=(
            "Retrieve: find the `k` nearest cases using euclidean, manhattan, cosine, or a mixed metric for numeric-plus-categorical data.",
            "Reuse for classification: majority vote, or a vote weighted so closer cases count more.",
            "Reuse for regression: distance-weighted mean, plain local mean, or a small ridge fit on the neighbours.",
            "Adapt: an optional offset that shifts the answer based on how the query differs from its neighbours.",
            "Retain: `session.cbr.retain` appends newly labelled cases, requires a source disclosure, and refuses any row from validation or test.",
        ),
        use=(
            "Distance-weighted reuse when neighbour quality varies a lot with distance.",
            "Local ridge for regression when the outcome trends smoothly across the neighbourhood.",
            "Retain when genuinely new labelled cases arrive from production.",
        ),
        avoid=(
            "Do not retain rows from your own holdout partitions: BuildML refuses, and the refusal is protecting your evaluation.",
            "Do not use a mixed metric without understanding how categorical mismatches are scored; unknown categories fall back to a sentinel.",
        ),
        myths=(
            (
                "Retaining more cases always improves the system.",
                "Retaining unrepresentative or mislabelled cases makes future retrievals worse. Growth also slows every prediction.",
            ),
            (
                "Retain is just an append, so disclosure is bureaucracy.",
                "The disclosure is what lets a future reader tell learned-from-training memory apart from operationally added memory. Without it the leakage question is unanswerable.",
            ),
        ),
        example=(
            "session.cbr.fit(reuse='distance_weighted', metric='mixed', k=7)",
            "session.cbr.evaluate(partition='validation')",
            "session.cbr.retain(",
            "    labeled_frame=new_resolved_cases,",
            "    source_disclosure='Q3 manually adjudicated claims',",
            ")",
        ),
        check=(
            "Where did every case in your memory come from?",
            "Does the reuse mode match your task: voting for labels, averaging for numbers?",
        ),
        tools=("fit_cbr", "predict_cbr", "retain_cbr", "evaluate_cbr"),
        terms=("case-based reasoning", "nearest neighbours", "distance metric", "leakage"),
        difficulty=CORE,
    ),
    _layer(
        "cbr-vs-rag",
        plain=(
            "Both case-based reasoning and retrieval-augmented generation start with 'find similar things', "
            "and there the resemblance ends. RAG retrieves text passages so a language model can write a "
            "grounded answer. Case-based reasoning retrieves labelled table rows and reuses their outcomes "
            "directly. No text, no language model."
        ),
        analogy=(
            "A librarian finding relevant passages for you to read, versus a colleague recalling how three "
            "similar cases were resolved. Both retrieve; the outputs are nothing alike."
        ),
        steps=(
            "Ask what your query is: a question in words, or a row of features?",
            "Ask what you need back: a written answer with citations, or a prediction?",
            "Text in, text out means RAG.",
            "Row in, label or number out means case-based reasoning.",
            "They live in different modules with different Session state, metrics, and bundles.",
        ),
        use=(
            "RAG for grounding answers in your documents.",
            "Case-based reasoning for explainable predictions grounded in past labelled examples.",
        ),
        avoid=(
            "Do not route case-based reasoning through the RAG functions; they are separate surfaces and will not accept each other's state.",
            "Do not install the RAG extras expecting to need them for case-based reasoning: it has no such dependency.",
        ),
        myths=(
            (
                "Case-based reasoning is retrieval-augmented generation for tables.",
                "The phrase sounds right and is wrong. There is no generation step at all, and the retrieval unit is a labelled row rather than a text chunk.",
            ),
            (
                "One index could serve both.",
                "The similarity notions differ completely: feature distance versus semantic text similarity: as do the leakage rules around what may enter the index.",
            ),
        ),
        example=(
            "session.cbr.fit(k=5)              # tabular case lookup",
            "session.rag.ingest_corpus(docs)   # document retrieval, separate surface",
        ),
        check=(
            "Is your input a row or a question?",
            "Do you need an explanation naming past cases, or a written answer with citations?",
        ),
        tools=("fit_cbr", "predict_cbr", "rag_ingest_corpus", "rag_retrieve"),
        terms=("case-based reasoning", "RAG", "retrieval", "embedding"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "cbr-bundle-boundary",
        plain=(
            "The case memory and its retrieval settings save as a case-based-reasoning bundle. A Session "
            "checkpoint stores your data and workflow and does not contain the case memory."
        ),
        analogy=(
            "The filing cabinet of past cases and this week's desk notes are separate things. Locking the "
            "desk does not back up the cabinet."
        ),
        steps=(
            "Fit a case base.",
            "Call `session.cbr.save_bundle(path)`: cases, metric, `k`, and reuse mode all travel together.",
            "Reload with `session.cbr.load_bundle(path)`.",
            "Predict and retrieve traces exactly as before.",
            "Use checkpoints separately for the data and workflow.",
        ),
        use=(
            "When predictions with case traces are served from a process that did not do the fitting.",
            "When an audit needs the exact memory that produced a past decision.",
        ),
        avoid=(
            "Do not expect `checkpoint_load` to restore the case memory.",
            "Do not swap in a RAG bundle; the formats differ and loading checks.",
        ),
        myths=(
            (
                "The bundle is just configuration.",
                "It contains the cases themselves. That is most of its size and all of its value.",
            ),
            (
                "Loading a bundle into a Session with different columns is fine.",
                "The feature columns must match, since distances are computed over them. Mismatch is an error rather than a silent wrong answer.",
            ),
        ),
        example=(
            "session.cbr.save_bundle('artifacts/claims-cases')",
            "svc = Session.ingest(incoming).cbr.load_bundle('artifacts/claims-cases')",
            "svc.cbr.predict(return_traces=True)",
        ),
        check=(
            "Has your case memory grown through retain since the bundle was saved?",
            "Do the serving rows carry exactly the feature columns the bundle expects?",
        ),
        tools=("save_cbr_bundle", "load_cbr_bundle", "predict_cbr", "checkpoint_save"),
        terms=("bundle", "checkpoint", "case-based reasoning"),
        difficulty=CORE,
    ),
)

__all__ = ["CBR_BEGINNER"]
