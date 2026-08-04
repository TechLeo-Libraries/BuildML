# Natural language processing deep guide

## Scope

BuildML's NLP path is a **Session-native text surface for one text column on the
dataset**: profile → fit → select → score → attribute, with unsupervised
description layered on the same split and a dedicated bundle.

| Surface | Role |
| --- | --- |
| `session.nlp.capability_matrix()` | Honest backend / task / extra / non-goal matrix (static) |
| `session.nlp.profile_corpus` | Corpus health + split-contamination screen |
| `session.nlp.detect_language` | Per-document language ID with confidence |
| `session.nlp.fit_classifier` | Single-label document classifier on Session **train** |
| `session.nlp.predict` | Score a partition with the train-fitted plan |
| `session.nlp.evaluate` | Holdout metrics + per-class report + confusion + OOV rate |
| `session.nlp.interpret` | Exact token attributions for linear heads |
| `session.nlp.fit_topics` | NMF / LDA on **train**, with NPMI coherence |
| `session.nlp.assign_topics` | Transform-and-assign a partition (never refits) |
| `session.nlp.extract_keyphrases` | TF-IDF / RAKE / TextRank phrase ranking |
| `session.nlp.analyze_sentiment` | Lexicon / supervised / transformer valence |
| `session.nlp.extract_entities` | Rule (regex + gazetteer) or spaCy mentions with offsets |
| `session.nlp.summarize` | Extractive TextRank / LexRank / lead summaries |
| `session.nlp.save_bundle` / `session.nlp.load_bundle` | `buildml.nlp_bundle.v1` |

Read-only accessors mirror the other domains: `session.nlp.text_plan`,
`session.nlp.topic_plan`, `session.nlp.fit_result`, `session.nlp.eval_result`, `session.nlp.predict_result`,
`session.nlp.interpret_result`, `session.nlp.topic_result`, `session.nlp.topic_assign_result`,
`session.nlp.keyphrase_result`, `session.nlp.sentiment_result`, `session.nlp.entity_result`,
`session.nlp.summary_result`, `session.nlp.language_result`, `session.nlp.profile_result`.

## Backends

| `backend` | Extra | Representation | Token attributions |
| --- | --- | --- | --- |
| `sklearn` (**default, always**) | core | Train-fitted bag-of-n-grams: `tfidf` / `count` / `hashing` × `word` / `char` / `char_wb` | Yes (except `hashing`) |
| `embedding` | `buildml[nlp]` | Frozen sentence-transformer document vectors | No |
| `transformer` | `buildml[nlp]` | Mean-pooled frozen Hugging Face encoder | No |

The default stays `sklearn` **even when the extras are installed**. It is
reproducible, needs no download, and is the only representation that can explain
its own decisions. Choose a dense backend deliberately, when word overlap
genuinely is not enough: and accept losing attribution when you do.

```python
matrix = session.nlp.capability_matrix()
print(matrix["default_backend_when_installed"])         # 'sklearn'
print(matrix["backends"]["embedding"]["available"])     # needs buildml[nlp]
print(matrix["non_goals"])                              # what this will never do

session.nlp.fit_classifier(backend="embedding", estimator="logistic")
```

Heads: `logistic`, `linear_svm`, `complement_nb`, `multinomial_nb`, `sgd` on
`sklearn`; only the signed-safe subset (`logistic`, `linear_svm`, `sgd`) on the
dense backends. **Naive Bayes on embeddings is refused**, not silently degraded:
it models features as counts and requires them non-negative, and encoder vectors
are signed. `vectorizer='hashing'` with a dense backend is refused for the same
class of reason: there is no hashing step to configure.

## Normalization is deterministic, vocabulary is learned

The split between the two is the whole leakage story for text.

| Stage | Learns from data? | When it runs |
| --- | --- | --- |
| Normalization steps (`strip_html`, `strip_urls`, `strip_emails`, `lowercase`, `strip_accents`, `strip_numbers`, `strip_punctuation`, `collapse_whitespace`, `collapse_repeats`) | No: stateless string rewriting | Any partition, any time |
| Tokenization, token-length filters, stemming, lemmatization | No: rule-based per document | Any partition, any time |
| Stopword list | No: supplied or built-in | Any partition |
| Vocabulary, document frequencies, IDF weights, `min_df` / `max_df` cuts | **Yes** | Train only |
| Topic decomposition (NMF / LDA components) | **Yes** | Train only |
| Classifier coefficients | **Yes** | Train only |

Because normalization learns nothing, it cannot leak, which is why the plan can
apply it to holdout rows freely. Everything in the "yes" column is frozen at
`session.nlp.fit_classifier` / `session.nlp.fit_topics` and reused verbatim afterwards.

Default steps are `strip_html`, `strip_urls`, `strip_emails`, `lowercase`,
`collapse_whitespace`: the ones that are almost always right. Override
explicitly:

```python
session.nlp.fit_classifier(
    normalize_steps=["strip_html", "strip_urls", "lowercase", "collapse_repeats"],
    stopword_language="en",
    min_token_length=2,
    stem=True,          # native conservative English suffix rules
    lemmatize=False,    # needs buildml[nlp] + downloaded WordNet
)
```

`collapse_repeats` folds three-or-more character runs to two (`sooooo` → `soo`),
which keeps emphatic spelling recognisable without exploding the vocabulary.
Built-in stopword lists cover seven languages; pass `stopwords=[...]` for
anything else rather than getting a silent empty list.

## Profile before you believe a score

`session.nlp.profile_corpus` is the step most text pipelines skip, and the reason their
holdout numbers are wrong.

```python
profile = session.nlp.profile_corpus(
    text_column="body",
    near_duplicate_threshold=0.9,
    detect_languages=True,
)
print(profile.n_documents, profile.vocabulary_size, profile.hapax_rate)
print(profile.train_holdout_exact_overlap)      # holdout rows copied from train
print(profile.train_holdout_near_duplicate)     # cosine >= threshold
print(profile.holdout_oov_token_rate)           # share of holdout tokens unseen in train
print(profile.findings)                         # plain-language disclosures
```

It **reports** contamination; it never silently drops rows. A high exact overlap
means your holdout accuracy is optimistic by roughly that share, and you should
know that before you quote the number, not after someone else finds it. The
near-duplicate threshold is recorded on the result so the claim stays auditable.

## Evaluation

`session.nlp.evaluate` returns accuracy, balanced accuracy, macro/weighted F1,
macro precision/recall, log loss, and ROC AUC (one-vs-rest for multi-class, when
the head exposes probabilities), plus a per-class report, the confusion matrix in
fitted-class order, and the holdout out-of-vocabulary token rate.

```python
print(session.nlp.evaluate(partition="validation").metrics)  # choose here
test = session.nlp.evaluate(partition="test")                # read once
print(test.per_class["billing"], test.confusion, test.oov_rate)
```

`log_loss` and `roc_auc` are omitted rather than faked for margin-only heads
(`linear_svm`, hinge-loss `sgd`). The OOV rate is the honest companion to the
score: a strong number with a 40% OOV rate is telling you the vocabulary did not
transfer.

## Token attribution is an identity, not an approximation

For a linear head on an invertible vocabulary, the contribution of a token is
exactly `coefficient × feature value`, and those contributions plus the intercept
reconstruct the decision function. There is nothing to approximate.

```python
interpret = session.nlp.interpret(
    partition="test", target_class="billing", top_k=10, max_documents=5
)
print(interpret.method)                 # 'linear-coefficient x feature-value'
for item in interpret.document_attributions[0]:
    print(item.token, item.weight, item.value, item.contribution)
print(interpret.global_top_tokens)      # per-class, from coefficients alone
```

Refused, with the reason, when it cannot be exact:

- `vectorizer='hashing'`: no invertible vocabulary, so tokens cannot be recovered
  from feature positions
- `backend='embedding'` / `'transformer'`: features are latent dimensions, not tokens
- heads without per-feature weights

Naive Bayes gets centred log-likelihoods (`method` says so) because raw class
log-probabilities are not comparable across classes. This is not a SHAP or LIME
substitute for non-linear models: it is the exact answer for the linear case,
and a refusal otherwise.

## Topics

```python
topics = session.nlp.fit_topics(
    method="nmf",          # 'nmf' on TF-IDF, 'lda' on counts
    n_topics=6,
    min_df=3,
    max_df=0.9,
    stopword_language="en",
)
for topic in topics.topics:
    print(topic.index, topic.label, topic.terms[:8], topic.train_mass, topic.coherence)
print(topics.mean_coherence)        # NPMI in [-1, 1], computed on train only
print(topics.reconstruction_error)  # NMF; topics.perplexity for LDA

assigned = session.nlp.assign_topics(partition="test")
print(assigned.dominant_topics[:10], assigned.topic_share)
```

NPMI coherence is computed on the **train** partition with the train vocabulary,
and clamped to its mathematical bounds. `session.nlp.assign_topics` is a pure transform: it
never refits the vectorizer or the decomposition, which is what makes holdout
topic shares comparable to train ones. Topic `label`s are generated from top
terms: they are a reading aid, not validated category names.

## Description surfaces (no quality metric is claimed)

```python
kp = session.nlp.extract_keyphrases(
    partition="train", method="tfidf",   # or 'rake', 'textrank'
    top_n=15, max_phrase_words=3, per_document=True, max_documents=25,
)
print([k.phrase for k in kp.corpus_keyphrases], kp.document_keyphrases[0])

s = session.nlp.summarize(
    partition="test", method="textrank",  # or 'lexrank', 'lead'
    n_sentences=3, max_documents=25,
)
print(s.summaries[0], s.mean_compression)

ents = session.nlp.extract_entities(
    partition="test", backend="rules",
    gazetteers={"QUEUE_TERM": ["invoice", "courier", "workspace"]},
)
print(ents.label_counts, ents.top_mentions)
print(ents.document_entities[0][0])          # text, label, start, end, source

sent = session.nlp.analyze_sentiment(partition="test", backend="lexicon", threshold=0.05)
print(sent.positive_rate, sent.negative_rate, sent.matched_term_rate)

lang = session.nlp.detect_language(partition="all")  # backend='native' or 'langdetect'
print(lang.dominant_language, lang.language_counts, lang.undetermined_rate)
```

Each of these is unsupervised and reports its own limits:

- **Keyphrases**: three genuinely different scorers. TF-IDF finds
  corpus-distinctive terms, RAKE finds phrases between stopword boundaries,
  TextRank finds phrases central to a co-occurrence graph. No gold metric.
- **Summaries**: sentences are **selected, never generated**. Abstractive
  summarization is an explicit non-goal; use `buildml.ai` with provider
  disclosure if you need prose.
- **Entities**: rules are precision-first on structured mentions (dates,
  amounts, percentages, emails, URLs, IPs, phones, times, reference codes,
  suffixed organisation names, titled person names, and any gazetteer terms you
  supply) with exact character offsets, and blind to everything else.
  spaCy generalises to unseen names in exchange for confident false positives,
  and needs `buildml[nlp-industry]` plus a downloaded model.
- **Sentiment**: the lexicon backend applies valence with negation and
  intensifier handling and is **domain-blind**: it will misread your jargon.
  `matched_term_rate` tells you how much of the corpus it actually had an
  opinion about, which is the number to check before quoting a rate.
  `backend='supervised'` reuses your fitted classifier; `'transformer'` needs
  `buildml[nlp]`.
- **Language**: native detection combines Unicode script probes with
  function-word scoring for seven Latin-script languages. Both backends degrade
  on very short strings, and a confident answer about a three-word document
  should be distrusted whichever backend gave it.

## Leakage discipline

- Require a `SplitPlan` before `session.nlp.fit_classifier` and `session.nlp.fit_topics`.
- Vocabulary, document frequencies, IDF, topic components, and the head: **train only**.
- Holdout partitions: predict / evaluate / assign / describe only.
- `session.nlp.assign_topics` and `session.nlp.predict` never refit any train-fitted state.
- `session.nlp.profile_corpus` screens the split and discloses contamination instead of
  repairing it behind your back.
- Bundles store the plan; Session checkpoints do **not**.

## Anti-patterns

- Fitting a vectorizer on the full frame before `split`: the classic text leak,
  because IDF and the `min_df` cut both see holdout documents.
- Quoting holdout accuracy without reading `train_holdout_exact_overlap`.
- Reading a strong score next to a high `oov_rate` as generalisation.
- Treating `fit.train_score` as holdout performance.
- Treating topic `label`s as validated categories.
- Quoting a lexicon sentiment rate without checking `matched_term_rate`.
- Expecting token attributions from `hashing` or a dense backend.
- Routing NLP through `session.rag.retrieve` / `session.rag.generate`, or calling document
  classification "RAG" because both touch text.
- Expecting `checkpoint_load` to restore the `NlpTextPlan`.

## NLP vs its neighbours

| | **NLP** | Neighbour |
| --- | --- | --- |
| `Session.text_features` | Representation stays inside the NLP plan | Writes numeric columns back onto the dataset for tabular models |
| `buildml.rag` | Supervised / unsupervised modelling of a text column | Ingest, chunk, index, retrieve to ground generated answers with citations |
| `buildml.dl` text path | Encoders stay **frozen** | `session.dl.make_text_loaders` / `session.dl.fit` fine-tune sequence models on token ids |
| `buildml.ai` | Never calls the network | External LLM provider under an operator policy |
| `buildml.cbr` (`backend='embedding'`) | Documents and their labels | Cases (features + solution) retrieved for reuse |

Sharing a text column, or a sentence-transformer, does not merge these surfaces.

## Bundle boundary

See `buildml.nlp.checkpoint.CHECKPOINT_BOUNDARY`. `buildml.nlp_bundle.v1` stores
the normalization plan, the train-fitted representation, the fitted head, and
optionally the topic plan. It does **not** store data, roles, splits, or history :
reload the workflow via `checkpoint_load` and the text model via
`session.nlp.load_bundle`. Because the normalization plan travels with the vectorizer, a
reloaded bundle reproduces the holdout score exactly; the proof and the
integration smoke both assert that.

## Proof

[ticket-routing-nlp](../proofs/ticket-routing-nlp/): Tier A route with a Tier C
hand-built `sklearn.Pipeline(TfidfVectorizer + LogisticRegression)` twin on the
same split indices. The twin matches the model; what it does not provide without
extra code is the contamination screen, the stored normalization plan, token
attribution, topic coherence, and the audit history.

## Benchmark

`benchmarks/nlp/representation_tradeoff.py`: one fixed corpus and split across
word/char n-grams, count vs TF-IDF vs hashing, and the dense backends when
`--include-optional` is passed. Records holdout accuracy, fit/score latency,
vocabulary size, and whether token attribution survives each choice.
