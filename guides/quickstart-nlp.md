# Quickstart: Natural language processing (NLP)

Session path for a **text column that lives on the dataset**: screen the corpus,
fit a single-label document classifier on train, read the holdout once, attribute
the decision to exact tokens, then layer unsupervised description: topics,
keyphrases, extractive summaries, entities, sentiment, language: and persist via
`buildml.nlp_bundle.v1`.

Honesty: **document-level** modelling. Not sequence labelling, not multi-label,
not generation, not translation, not transformer fine-tuning, and **not RAG**.
Core stays light (numpy / pandas / scikit-learn bag-of-n-grams, plus a native
tokenizer, stemmer, sentiment lexicon, and language detector); neural backends
activate only when `buildml[nlp]` is installed and are never the default.

**Proof:** [ticket-routing-nlp](../proofs/ticket-routing-nlp/) (+ Tier C
`sklearn.Pipeline(TfidfVectorizer + LogisticRegression)` twin).

**Go deeper:** [NLP deep](nlp-deep.md) ·
[Artifacts](artifacts-checkpoints-bundles.md) ·
[Preprocess depth](preprocess-depth.md) (for `text_features`) ·
[RAG quickstart](quickstart-rag.md) (for retrieval)

```python
import pandas as pd
from buildml import Session

frame = pd.DataFrame(
    {
        "body": [
            "Invoice INV-4482 charged the annual fee twice on the same card.",
            "The order was promised for the 3rd and arrived nine days late.",
            "Single sign-on stopped working for the whole workspace this morning.",
            # ... hundreds more tickets ...
        ],
        "queue": ["billing", "shipping", "account"],
    }
)

session = (
    Session.ingest(frame)
    .set_roles({"body": "feature", "queue": "target"})
    .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
)

# What can this install actually do, and what would more cost?
print(Session.nlp_capability_matrix()["default_backend_when_installed"])  # 'sklearn'

# 1. Screen the split before trusting any score.
profile = session.profile_text_corpus(near_duplicate_threshold=0.9)
print(profile.train_holdout_exact_overlap, profile.findings)

# 2. Fit on train only. Normalization, vocabulary, and document frequencies
#    are all learned from train; holdout is transform-and-score.
fit = session.fit_text_classifier(
    text_column="body",
    vectorizer="tfidf",
    estimator="logistic",
    ngram_range=(1, 2),
    min_df=2,
    class_weight="balanced",
)
print(fit.backend, fit.estimator, fit.vocabulary_size, fit.class_counts)

# 3. Choose on validation, then read test once.
print(session.evaluate_text_classifier(partition="validation").metrics)
test = session.evaluate_text_classifier(partition="test")
print(test.metrics, test.per_class, test.oov_rate)

predicted = session.predict_text(partition="test")

# 4. Exact token attribution: coefficient x feature value, an identity for a
#    linear head. Refused outright for hashing and dense backends.
interpret = session.interpret_text_prediction(partition="test", top_k=8, max_documents=5)
for item in interpret.document_attributions[0]:
    print(item.token, round(item.contribution, 4))

# 5. Unsupervised structure fitted on train, assigned to holdout.
topics = session.fit_topics(method="nmf", n_topics=4, min_df=3)
print([t.label for t in topics.topics], topics.mean_coherence)  # NPMI on train
print(session.assign_topics(partition="test").topic_share)

# 6. Description surfaces that claim no quality metric.
print(session.extract_keyphrases(partition="train", method="tfidf", top_n=10).corpus_keyphrases)
print(session.summarize_text(partition="test", method="textrank", n_sentences=2).summaries[0])
print(session.extract_entities(partition="test", backend="rules").label_counts)
print(session.analyze_sentiment(partition="test", backend="lexicon").negative_rate)
print(session.detect_language(partition="all").dominant_language)

# 7. The bundle carries the normalization plan, so a reload scores identically.
session.save_nlp_bundle("artifacts/nlp_bundle")
```

| In scope | Out of scope |
| --- | --- |
| Train-only vectorizer, vocabulary, document frequencies, head | Fitting any vectorizer on the full frame |
| Single-label document classification | Multi-label; span / token labelling (NER supervision) |
| Exact token attributions for linear heads | SHAP/LIME on non-linear heads |
| NMF / LDA topics + NPMI coherence on train | Claiming topics are validated categories |
| TF-IDF / RAKE / TextRank keyphrases | Gold keyphrase metrics |
| Extractive summaries (TextRank / LexRank / lead) | Abstractive / generated prose (see `buildml.ai`) |
| Rule entities; spaCy NER when installed | Coreference, dependency-parse products |
| Lexicon / supervised / transformer sentiment | Domain-tuned sentiment without your own labels |
| Native + langdetect language ID | Machine translation |
| Corpus profile that **reports** contamination | Silently deduplicating your split |
| Frozen encoders + linear head | Transformer fine-tuning (Torch text path) |
| `buildml.nlp_bundle.v1` | Session checkpoint embedding the plan |

Optional extras: `buildml[nlp]` (NLTK morphology, langdetect, sentence-transformer
embeddings, frozen transformer encoders), `buildml[nlp-industry]` (spaCy
statistical NER: then `python -m spacy download en_core_web_sm`). Both are
included in `buildml[production]`.

**NLP vs its neighbours.** `Session.text_features` writes numeric columns back
onto the dataset so tabular models can consume text; NLP keeps its representation
inside the NLP plan. `buildml.rag` ingests and retrieves documents to ground
generated answers. `Session.make_text_torch_loaders` / `fit_torch` fine-tune
neural sequence models on token ids. `buildml.ai` calls an external LLM provider
under an operator policy: NLP never touches the network. Sharing a text column
does not merge these surfaces.
