# Dynamo Click Lab

**Tier B** cross-domain product proof — online stream conversion + metalearning
cold-start + classical supervised baseline for synthetic clickstream.

## Product narrative

Dynamo studies conversion under streaming updates and few-shot new categories:

1. Streams train-cursor `partial_fit` conversion updates
2. Runs prototypical / warm-start metalearning with `group_split` by category
3. Fits a classical logistic conversion scorer on the same clickstream split

## Status

`completed` — run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\dynamo-click-lab\script.py
```

## Leakage controls

- Online `partial_fit` consumes train cursor only
- Metalearning `group_split` by `category_id`; episodic eval on held-out categories
- Classical scorer uses the same clickstream stratified split
- Test evaluate after locks

## What fails if leakage is ignored

- Streaming updates that include test rows make online metrics meaningless
- Episodes that mix train and test categories invent cold-start accuracy
- Fitting classical scores on the full clickstream invents holdout ROC

## Upstream Tier A building blocks

`clickstream-online`, `stream-fraud-online`, `coldstart-meta-adapt`,
`few-shot-domain-adapt`, `loan-approval-classical`

## Limitations

Batch chunks, not Kafka/Flink. Synthetic categories only.
