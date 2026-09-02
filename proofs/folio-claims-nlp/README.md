# folio-claims-nlp

This script composes `session.nlp`, `session.cbr`, and `session.symbolic` on
one synthetic P&C claims table. It is not a product BuildML ships.

The script profiles the note corpus and fits TF-IDF + logistic (plus NMF
topics), builds CBR case memory from structured claim features (train
only), and induces symbolic decision-tree guardrails on the same case
split.

## Data

Claim notes reuse synthetic ticket language. Not a real P&C extract.

## Leakage

NLP stratified split before TF-IDF / topics fit. CBR case memory built from
train cases only. Symbolic rules induced on the same train split as CBR.
Test text / CBR / symbolic eval after each stage locks.

## What fails if leakage is ignored

Fitting the text vectorizer on test notes invents desk accuracy. Putting
test claims into CBR memory makes escalate accuracy meaningless. Inducing
guardrail rules on the full book looks more "compliant" than they would in
production.

## How to run

```bash
python proofs/folio-claims-nlp/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`ticket-routing-nlp`, `case-memory-claims`, `warranty-cbr-memory`,
`policy-rules-neuro-symbolic`, `claim-severity-regression`.

## Limitations

Claim notes reuse synthetic ticket language. Not a real P&C extract.
