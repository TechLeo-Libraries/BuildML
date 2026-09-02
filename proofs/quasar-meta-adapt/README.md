# quasar-meta-adapt

This script composes `session.metalearning`, `session.ssl`, and classical
`session.fit` on one synthetic catalog-category table. It is not a product BuildML ships.

Categories are held out via group split. The script fits prototypical (or
warm-start) metalearning on train categories, a masked-tabular SSL pretext
plus optional probe head, and a classical logistic baseline on the same
split.

## Data

Synthetic categories.

## Leakage

`group_split` by `category_id` before meta / SSL / classical fit. Episodic
metalearning eval on held-out categories. SSL pretext + probe fit on train
only. Test used after each stage locks.

## What fails if leakage is ignored

Episodes that include test categories in the support set fake cold-start
skill. SSL pretext on the full table leaks holdout geometry into
embeddings. A classical baseline trained with test rows is not a fair
comparator.

## How to run

```bash
python proofs/quasar-meta-adapt/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`coldstart-meta-adapt`, `few-shot-domain-adapt`, `tabular-ssl-probe`,
`ssl-representation-probe`, `loan-approval-classical`.

## Limitations

Synthetic categories. Metalearning may fall back to warm_start.
