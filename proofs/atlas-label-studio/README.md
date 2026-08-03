# Atlas Label Studio

**Tier B** cross-domain product proof: SSL pretext + semi-supervised
propagation + active-learning budget loop with a simulated oracle.

## Product narrative

Atlas is a labeling studio for scarce supervised signal. Most train labels are
masked; holdouts keep full labels for evaluation only. The product:

1. Fits a masked-tabular SSL pretext (and optional head) on train features
2. Runs label propagation / semi-supervised learning on the scarce labeled pool
3. Runs a margin-sampling active-learning budget loop querying **train** only
4. Uses a simulated oracle (ground-truth for queried train indices): not a
   production workforce UI

## Status

`completed`: run `script.py`; see `results/summary.json` plus `ssl.json`,
`semisupervised.json`, `active_learning.json`.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\atlas-label-studio\script.py
```

## Leakage controls

- Stratified split before masking / pretext / AL
- Label masking applied to train indices only
- Holdouts retain full labels solely for evaluation
- AL queries drawn from the train unlabeled pool only

## What fails if leakage is ignored

- Masking validation/test then “recovering” labels via the graph overstates SSL gains
- Allowing AL to query the test pool turns the budget curve into a cheat sheet
- Fitting SSL pretext on the full table leaks holdout geometry into embeddings

## Upstream Tier A building blocks

`ssl-representation-probe`, `semi-label-efficiency`, `active-labeling-budget`

## Limitations

Simulated oracle; synthetic blobs; production label noise not modeled.
Missing extras are skipped with JSON disclosures.
