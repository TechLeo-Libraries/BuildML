"""Natural-language processing (document-level text modelling and analysis).

Phase coverage (internal tracker - depth-first; do not spray stubs)
------------------------------------------------------------------
Phase 1 (**complete**): unsupervised -> ensembles -> AutoML -> forecasting -> anomaly.

Phase 2:
  1-10. Semi-supervised -> ... -> symbolic / neuro-symbolic - prior items.
  11-14. Case-based reasoning -> imitation + RL -> TDA -> app systems.
  **This module:** NLP deepening (Session text path brought to the full bar:
  classify, interpret, topics, keyphrases, sentiment, entities, extractive
  summaries, language ID, corpus profiling, bundles).
  Next: CV deepening if still partial. Speech: ASR keep/improve; TTS out.

Explicit non-goals (no product surfaces): abstractive / generative summarization
and text generation, machine translation, multi-label and span-level (sequence
labelling) supervision, transformer fine-tuning (the Torch text path owns that),
document retrieval for generation (``buildml.rag`` owns that), coreference
resolution, full dependency-parse products.

Honesty (this package):
  - One text column on the Session dataset is the unit of work; documents are
    rows, and every vocabulary-bearing fit is train-only.
  - Single-label document classification over train-fitted bag-of-n-grams
    (default), frozen sentence-transformer vectors, or a frozen pooled
    transformer encoder; token attributions are exact for linear heads and are
    refused when the representation has no invertible vocabulary.
  - Unsupervised surfaces (topics, keyphrases, lexicon sentiment, rule entities,
    extractive summaries, language ID, corpus profiling) state what they do and
    do not claim gold-standard quality metrics they cannot compute.
  - Corpus profiling screens the split for exact and near-duplicate text
    contamination and reports it rather than silently deduplicating.
  - **Not** RAG. Sharing a text column does not make NLP a retrieval product.
  - Core stays light: numpy / pandas / scikit-learn plus shipped lexicons.

Lazy imports - core never grows heavy transformer / spaCy stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "Entity",
    "Keyphrase",
    "NlpBackend",
    "NlpConfig",
    "NlpCorpusProfile",
    "NlpEntityResult",
    "NlpEstimator",
    "NlpEvalResult",
    "NlpFitResult",
    "NlpInterpretResult",
    "NlpKeyphraseResult",
    "NlpLanguageResult",
    "NlpPredictResult",
    "NlpSentimentResult",
    "NlpSummaryResult",
    "NlpTask",
    "NlpTextPlan",
    "NlpTopicAssignResult",
    "NlpTopicPlan",
    "NlpTopicResult",
    "NlpVectorizeConfig",
    "TextNormalizeConfig",
    "TextNormalizePlan",
    "TokenAttribution",
    "Topic",
    "analyze_sentiment",
    "assign_topics",
    "build_normalize_plan",
    "detect_document_language",
    "detect_language",
    "evaluate_text_classifier",
    "extract_entities",
    "extract_keyphrases",
    "fit_text_classifier",
    "fit_topics",
    "interpret_text_prediction",
    "load_nlp_bundle",
    "nlp_capability_matrix",
    "nlp_status",
    "nlp_status_for_session",
    "normalize_document",
    "predict_text",
    "profile_text_corpus",
    "save_nlp_bundle",
    "score_document",
    "split_sentences",
    "summarize_text",
    "tokenize_document",
]

_TYPES = {
    "NlpTask",
    "NlpBackend",
    "NlpEstimator",
    "NlpConfig",
    "NlpVectorizeConfig",
    "TextNormalizeConfig",
}
_RESULTS = {
    "Entity",
    "Keyphrase",
    "NlpCorpusProfile",
    "NlpEntityResult",
    "NlpEvalResult",
    "NlpFitResult",
    "NlpInterpretResult",
    "NlpKeyphraseResult",
    "NlpLanguageResult",
    "NlpPredictResult",
    "NlpSentimentResult",
    "NlpSummaryResult",
    "NlpTextPlan",
    "NlpTopicAssignResult",
    "NlpTopicPlan",
    "NlpTopicResult",
    "TokenAttribution",
    "Topic",
}
_NORMALIZE = {
    "TextNormalizePlan",
    "build_normalize_plan",
    "normalize_document",
    "split_sentences",
    "tokenize_document",
}
_CHECKPOINT = {
    "BUNDLE_FORMAT",
    "CHECKPOINT_BOUNDARY",
    "save_nlp_bundle",
    "load_nlp_bundle",
}
_HOOKS = {"nlp_status", "nlp_status_for_session"}


def __getattr__(name: str) -> Any:
    if name in _TYPES:
        from buildml.nlp import types as types_mod

        return getattr(types_mod, name)
    if name in _RESULTS:
        from buildml.nlp import results as results_mod

        return getattr(results_mod, name)
    if name in _NORMALIZE:
        from buildml.nlp import normalize as normalize_mod

        return getattr(normalize_mod, name)
    if name in _CHECKPOINT:
        from buildml.nlp import checkpoint as checkpoint_mod

        return getattr(checkpoint_mod, name)
    if name in _HOOKS:
        from buildml.nlp import explain_hooks as hooks

        return getattr(hooks, name)
    if name == "nlp_capability_matrix":
        from buildml.nlp.catalog import nlp_capability_matrix

        return nlp_capability_matrix
    if name == "fit_text_classifier":
        from buildml.nlp.fit import fit_text_classifier

        return fit_text_classifier
    if name == "predict_text":
        from buildml.nlp.predict import predict_text

        return predict_text
    if name == "evaluate_text_classifier":
        from buildml.nlp.evaluate import evaluate_text_classifier

        return evaluate_text_classifier
    if name == "interpret_text_prediction":
        from buildml.nlp.interpret import interpret_text_prediction

        return interpret_text_prediction
    if name in {"fit_topics", "assign_topics"}:
        from buildml.nlp import topics as topics_mod

        return getattr(topics_mod, name)
    if name == "extract_keyphrases":
        from buildml.nlp.keyphrases import extract_keyphrases

        return extract_keyphrases
    if name in {"analyze_sentiment", "score_document"}:
        from buildml.nlp import sentiment as sentiment_mod

        return getattr(sentiment_mod, name)
    if name == "extract_entities":
        from buildml.nlp.entities import extract_entities

        return extract_entities
    if name == "summarize_text":
        from buildml.nlp.summarize import summarize_text

        return summarize_text
    if name in {"detect_language", "detect_document_language"}:
        from buildml.nlp import language as language_mod

        return getattr(language_mod, name)
    if name == "profile_text_corpus":
        from buildml.nlp.profile import profile_text_corpus

        return profile_text_corpus
    raise AttributeError(f"module 'buildml.nlp' has no attribute {name!r}")
