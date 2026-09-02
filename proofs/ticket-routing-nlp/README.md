# ticket-routing-nlp

You have free-text support tickets and a queue label (billing, shipping,
account, hardware). You want a holdout routing number plus token-level
attributions so a misroute can be audited.

## Data

Synthetic / in-script license-clear tickets from
`proofs/_lib/datasets.py::load_support_tickets_synthetic`. Tickets are
composed from per-queue sentence pools with deliberate vocabulary overlap,
plus an `ambiguous_rate` share (18%) written entirely from queue-agnostic
sentences. Those rows carry a label no reader could recover from the text,
which puts a stated ceiling on achievable accuracy: around 0.86 for four
balanced queues, so this proof cannot report a suspiciously perfect score.

The sentence pools are finite, so repeated documents are expected.
`session.nlp.profile_corpus` reports them: at seed 11 it finds 58 exact and
109 near-duplicate holdout documents against train, and says the holdout
metrics are optimistic by that amount. That disclosure is the point of
running the profile first.

## Leakage

Stratified split before any text operation. Normalization plan, vocabulary,
document frequencies, and the head are fitted on train only; validation and
test are transform-and-score. Topic vectorizer and NMF are fitted on train
only, so `session.nlp.assign_topics` on holdout is a pure transform. Model
choice reads validation; test is evaluated once after the model is locked.
`session.nlp.profile_corpus` screens the split for exact and near-duplicate
text contamination and reports it rather than silently deduplicating.

## How to run

```bash
python proofs/ticket-routing-nlp/script.py
python proofs/ticket-routing-nlp/baseline_industry.py
```

## What you'll get

`results/results.json` with accuracy, balanced accuracy, macro/weighted F1,
log loss, per-class report, confusion matrix, holdout out-of-vocabulary
token rate, and NPMI topic coherence.

At seed 11: validation accuracy 0.826, test accuracy 0.872, macro F1 0.872,
ROC AUC 0.986, holdout OOV token rate 0.016, mean NPMI topic coherence
0.609. The reloaded bundle reproduces the test accuracy exactly
(`bundle_reproduces_holdout_score: true`).

`results/comparison.json` is a hand-built
`sklearn.Pipeline(TfidfVectorizer + LogisticRegression)` twin on the same
split indices. The twin matches the model; it does not provide the
contamination screen, stored normalization plan, token attribution, topic
coherence, or audit history without extra code.

## Limitations

Synthetic tickets; not a real customer-support corpus. The corpus contains
exact and near-duplicate documents across the split. They are reported, not
removed, so the headline accuracy should be read as optimistic by roughly
the disclosed overlap. Single-label document classification only: no span
labelling, no generation. Lexicon sentiment and rule entities are
unsupervised baselines with no gold metric attached. Topic labels are
generated from top terms and are not validated category names.

Related: [NLP quickstart](../../guides/quickstart-nlp.md).
