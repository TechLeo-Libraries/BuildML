# Quasar Meta Adapt

**Tier B** cross-domain product proof — metalearning few-shot adaptation +
SSL pretext/probe + classical supervised baseline for cold-start categories.

## Product narrative

Quasar adapts repurchase models to new catalog categories. Categories are
held out via group split; SSL representations and a classical logistic
baseline provide complementary views. The platform:

1. Fits prototypical (or warm-start) metalearning on train categories
2. Runs masked-tabular SSL pretext + optional probe head
3. Trains a classical logistic baseline on the same honest split

## Status

`completed` — run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\quasar-meta-adapt\script.py
```

## Leakage controls

- `group_split` by `category_id` before meta / SSL / classical fit
- Episodic metalearning eval on held-out categories
- SSL pretext + probe fit on train only
- Test used after each stage locks

## What fails if leakage is ignored

- Episodes that include test categories in the support set fake cold-start skill
- SSL pretext on the full table leaks holdout geometry into embeddings
- Classical baseline trained with test rows is not a fair comparator

## Upstream Tier A building blocks

`coldstart-meta-adapt`, `few-shot-domain-adapt`, `tabular-ssl-probe`,
`ssl-representation-probe`, `loan-approval-classical`

## Limitations

Synthetic categories. Metalearning may fall back to warm_start.
