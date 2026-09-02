# mosaic-warranty-desk

This script composes `session.cbr`, `session.symbolic`, and classical
`session.fit` on one synthetic warranty table. It is not a product BuildML
ships.

The script builds CBR case memory from train claims only, induces symbolic
decision-tree guardrails on the same split, and fits a classical logistic
scorer for calibrated approve scores.

## Data

Synthetic warranty claims. Not a real OEM extract.

## Leakage

Stratified split before CBR / symbolic / classical. CBR case memory built
from train cases only. Symbolic rules induced on the same train split;
test after lock. Classical scorer uses `inject_split`: never refits on
test.

## What fails if leakage is ignored

Putting test claims into CBR memory makes accuracy meaningless. Inducing
guardrail rules on the full book looks more "fair" than they would in
production. Fitting classical scores on the full table invents holdout ROC.

## How to run

```bash
python proofs/mosaic-warranty-desk/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`warranty-cbr-memory`, `case-memory-claims`, `policy-rules-neuro-symbolic`,
`compliance-neuro-symbolic`, `loan-approval-classical`.

## Limitations

Synthetic warranty claims. CBR is not RAG.
