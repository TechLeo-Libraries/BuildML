# dynamo-click-lab

This script composes `session.online`, `session.metalearning`, and classical
`session.fit` on one synthetic clickstream table. It is not a product BuildML ships.

The script streams train-cursor `partial_fit` conversion updates, runs
prototypical / warm-start metalearning with `group_split` by category, and
fits a classical logistic conversion scorer on the same clickstream split.

## Data

Synthetic clickstream and categories. Not Kafka/Flink.

## Leakage

Online `partial_fit` consumes the train cursor only. Metalearning
`group_split` by `category_id`; episodic eval on held-out categories.
Classical scorer uses the same clickstream stratified split. Test evaluate
after locks.

## What fails if leakage is ignored

Streaming updates that include test rows make online metrics meaningless.
Episodes that mix train and test categories invent cold-start accuracy.
Fitting classical scores on the full clickstream invents holdout ROC.

## How to run

```bash
python proofs/dynamo-click-lab/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`clickstream-online`, `stream-fraud-online`, `coldstart-meta-adapt`,
`few-shot-domain-adapt`, `loan-approval-classical`.

## Limitations

Batch chunks, not Kafka/Flink. Synthetic categories only.
