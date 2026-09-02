# Natural language on a table

```bash
pip install buildml
# frozen encoders, langdetect, NLTK morphology: pip install "buildml[nlp]"
# spaCy NER: pip install "buildml[nlp-industry]"
```

You have a text column on a Session table, a label, and a split. You
want a document classifier that learned its vocabulary from train
only, plus optional topics and descriptions on the same rows. That
is `session.nlp.*`. It is not RAG, not sequence labelling, not
generation, and not Torch fine-tuning (`session.dl.make_text_loaders`
owns that).

Default backend is sklearn bag-of-n-grams even when extras are
installed: `vectorizer="tfidf"`, `analyzer="word"`,
`ngram_range=(1, 2)`, `estimator="logistic"`. Name `text_column`.
If you omit it, the Session infers only when one string feature is
obviously the prose column; several similar candidates raise instead
of guessing. Fit and topics need a split. Evaluate defaults to
`partition="validation"` so you can choose there and read test once.

`backend=None` stays sklearn. `embedding` and `transformer` need
`buildml[nlp]` and you have to ask for them. Naive Bayes on dense
vectors is refused. Hashing with a dense backend is refused.
Token attribution is refused for hashing and for latent encoders.

You choose the representation and the head. The API refuses a missing
split, an ambiguous text column, and combinations that cannot work.

Short on-ramp: [NLP quickstart](quickstart-nlp.md). Proof:
[ticket-routing-nlp](../proofs/ticket-routing-nlp/).

## Profile, fit, choose, read test

```python
import pandas as pd

from buildml import Session

frame = pd.DataFrame(
    {
        "body": [
            "Invoice INV-4482 charged the annual fee twice on the same card.",
            "The order was promised for the 3rd and arrived nine days late.",
            "Single sign-on stopped working for the whole workspace this morning.",
            "Refund the duplicate charge on invoice INV-4482 please.",
            "Package showed delivered but nobody was home all week.",
            "Password reset mail never arrived for the admin account.",
            "Why was the VAT line added twice on the same invoice?",
            "Courier left the parcel with a neighbour we do not know.",
            "SSO tokens expire after five minutes and kick everyone out.",
            "Billing portal still shows last year's plan after we upgraded.",
            "Tracking number never updated after the first scan.",
            "Cannot invite a teammate because the workspace seat count is wrong.",
        ],
        "queue": [
            "billing",
            "shipping",
            "account",
            "billing",
            "shipping",
            "account",
            "billing",
            "shipping",
            "account",
            "billing",
            "shipping",
            "account",
        ],
    }
)

session = (
    Session.ingest(frame)
    .set_roles({"body": "feature", "queue": "target"})
    .split(test_size=0.25, validation_size=0.25, random_state=0, stratify=True)
)

profile = session.nlp.profile_corpus(
    text_column="body",
    near_duplicate_threshold=0.9,
)
print(profile.train_holdout_exact_overlap, profile.holdout_oov_token_rate)
print(profile.findings)

fit = session.nlp.fit_classifier(
    text_column="body",
    vectorizer="tfidf",
    estimator="logistic",
    ngram_range=(1, 2),
    min_df=1,
    class_weight="balanced",
)
print(fit.backend, fit.estimator, fit.vocabulary_size)

print(session.nlp.evaluate(partition="validation").metrics)
test = session.nlp.evaluate(partition="test")
print(test.metrics, test.oov_rate)

predicted = session.nlp.predict(partition="test")
interpret = session.nlp.interpret(
    partition="test", target_class="billing", top_k=8, max_documents=5
)
for item in interpret.document_attributions[0]:
    print(item.token, round(item.contribution, 4))
```

`profile_corpus` reports contamination. It never drops rows. Exact
overlap means holdout accuracy is optimistic by roughly that share.
Read it before you quote a number.

`evaluate` returns accuracy, balanced accuracy, macro/weighted F1,
macro precision/recall, plus per-class report, confusion in fitted
class order, and holdout OOV rate. `log_loss` and `roc_auc` are
omitted for margin-only heads (`linear_svm`, hinge-loss `sgd`) rather
than faked. A strong score next to a 40% OOV rate is telling you the
vocabulary did not transfer. `fit.train_score` is not holdout
performance.

## Backends

| `backend` | Extra | Representation | Token attributions |
| --- | --- | --- | --- |
| `sklearn` (default, always) | core | bag-of-n-grams: `tfidf` / `count` / `hashing` × `word` / `char` / `char_wb` | yes, except `hashing` |
| `embedding` | `buildml[nlp]` | frozen sentence-transformer document vectors | no |
| `transformer` | `buildml[nlp]` | mean-pooled frozen Hugging Face encoder | no |

Heads on sklearn: `logistic` (default), `linear_svm`,
`complement_nb`, `multinomial_nb`, `sgd`. Dense backends keep the
signed-safe subset (`logistic`, `linear_svm`, `sgd`). Naive Bayes
models counts and needs non-negative features; encoder vectors are
signed, so that pairing raises.

```python
session.nlp.fit_classifier(
    backend="embedding",
    estimator="logistic",
    text_column="body",
)
```

Dense backends download a model and lose attribution. Use them when
word overlap is genuinely not enough.

## Normalization vs vocabulary

Normalization is stateless string rewriting. It cannot leak, so the
plan applies it to holdout freely. Default steps:
`strip_html`, `strip_urls`, `strip_emails`, `lowercase`,
`collapse_whitespace`. Override when you need to:

```python
session.nlp.fit_classifier(
    text_column="body",
    normalize_steps=["strip_html", "strip_urls", "lowercase", "collapse_repeats"],
    stopword_language="en",
    min_token_length=2,
    stem=True,
    lemmatize=False,
)
```

`collapse_repeats` folds three-or-more character runs to two
(`sooooo` → `soo`). Stemming is conservative English suffix rules.
Lemmatize needs `buildml[nlp]` and downloaded WordNet. Built-in
stopword lists cover seven languages; pass `stopwords=[...]` for
anything else.

These **learn from train only** and freeze on the plan: vocabulary,
document frequencies, IDF, `min_df` / `max_df` cuts, topic
components, classifier coefficients. Tokenization, length filters,
stemming, and stopwords are rule-based per document.

## Token attribution

For a linear head on an invertible vocabulary, a token's contribution
is `coefficient × feature value`. Those contributions plus the
intercept reconstruct the decision function.

```python
interpret = session.nlp.interpret(
    partition="test", target_class="billing", top_k=10, max_documents=5
)
print(interpret.method)
print(interpret.global_top_tokens)
```

Refused, with the reason, when it cannot be exact:

- `vectorizer="hashing"`: no invertible vocabulary
- `backend="embedding"` / `"transformer"`: latent dimensions, not tokens
- heads without per-feature weights

Naive Bayes gets centred log-likelihoods (`method` says so). This is
not SHAP. It is the exact linear case, and a refusal otherwise.

## Topics

Default method is NMF (`n_topics=6`) on TF-IDF. LDA uses counts.
Coherence is NPMI on **train** only, clamped to [-1, 1].
`assign_topics` is a pure transform: it never refits.

```python
topics = session.nlp.fit_topics(
    method="nmf",
    n_topics=6,
    text_column="body",
    min_df=2,
    max_df=0.9,
    stopword_language="en",
)
for topic in topics.topics:
    print(topic.index, topic.label, topic.terms[:8], topic.coherence)
print(topics.mean_coherence)

assigned = session.nlp.assign_topics(partition="test")
print(assigned.dominant_topics[:10], assigned.topic_share)
```

Topic `label`s are generated from top terms. They are a reading aid,
not validated category names.

## Description surfaces

These claim no gold quality metric.

```python
kp = session.nlp.extract_keyphrases(
    partition="train",
    method="tfidf",  # or "rake", "textrank"
    top_n=15,
    max_phrase_words=3,
    per_document=True,
    max_documents=25,
)

s = session.nlp.summarize(
    partition="test",
    method="textrank",  # or "lexrank", "lead"
    n_sentences=3,
    max_documents=25,
)

ents = session.nlp.extract_entities(
    partition="test",
    backend="rules",
    gazetteers={"QUEUE_TERM": ["invoice", "courier", "workspace"]},
)

sent = session.nlp.analyze_sentiment(
    partition="test", backend="lexicon", threshold=0.05
)
print(sent.positive_rate, sent.matched_term_rate)

lang = session.nlp.detect_language(partition="all")
print(lang.dominant_language, lang.undetermined_rate)
```

Keyphrases: TF-IDF finds corpus-distinctive terms, RAKE finds phrases
between stopword boundaries, TextRank finds phrases central to a
co-occurrence graph.

Summaries **select sentences**. They never generate. Abstractive
summarization is out of this path.

Entities: rules are precision-first on structured mentions (dates,
amounts, emails, URLs, phones, gazetteer terms you supply) with
character offsets, and blind to everything else. spaCy needs
`buildml[nlp-industry]` plus a downloaded model
(`en_core_web_sm` by default).

Sentiment: lexicon valence with negation and intensifiers. It is
domain-blind. Check `matched_term_rate` before quoting a rate.
`backend="supervised"` reuses your fitted classifier.
`backend="transformer"` needs `buildml[nlp]`.

Language: native combines Unicode script probes with function-word
scoring for seven Latin-script languages. `langdetect` needs
`buildml[nlp]`. Both degrade on very short strings.

## Neighbours on the same text

`session.text_features` writes numeric columns back onto the table
for a tabular model. `session.nlp` keeps the representation inside
the NLP plan. `session.rag` indexes chunks to ground generated
answers. `session.dl.make_text_loaders` trains on token ids. Sharing
a column does not merge these surfaces.

## Bundle

`session.nlp.save_bundle` writes `buildml.nlp_bundle.v1`:
normalization, train-fitted representation, fitted head, optional
topic plan. It does not store data, roles, splits, or history.
Session checkpoints do not embed `NlpTextPlan`. Reload the workflow
via `checkpoint_load` and the text model via
`session.nlp.load_bundle(..., trusted=True)`.

## Benchmark

`benchmarks/nlp/representation_tradeoff.py` holds corpus and split
fixed across word/char n-grams, count vs TF-IDF vs hashing, and the
dense backends when `--include-optional` is passed. It records
holdout accuracy, fit/score latency, vocabulary size, and whether
token attribution survives each choice.
