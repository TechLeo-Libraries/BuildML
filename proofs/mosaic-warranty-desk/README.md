# Mosaic Warranty Desk

**Tier B** cross-domain product proof: CBR case memory + symbolic guardrails +
classical scoring for synthetic warranty claim decisions.

## Product narrative

Mosaic adjudicates warranty claims by retrieving similar past cases, inducing
explainable deny rules, and scoring with a classical logistic baseline:

1. Builds CBR case memory from train claims only
2. Induces symbolic decision-tree guardrails on the same split
3. Fits a classical supervised scorer for calibrated approve scores

## Status

`completed`: run `script.py`; see `results/summary.json` and stage JSONs.

## How to run

```bash
.\.venv\Scripts\python.exe proofs\mosaic-warranty-desk\script.py
```

## Leakage controls

- Stratified split before CBR / symbolic / classical
- CBR case memory built from train cases only
- Symbolic rules induced on the same train split; test after lock
- Classical scorer uses `inject_split`: never refits on test

## What fails if leakage is ignored

- Putting test claims into CBR memory makes accuracy meaningless
- Inducing guardrail rules on the full book looks more “fair” than production
- Fitting classical scores on the full table invents holdout ROC

## Upstream Tier A building blocks

`warranty-cbr-memory`, `case-memory-claims`, `policy-rules-neuro-symbolic`,
`compliance-neuro-symbolic`, `loan-approval-classical`

## Limitations

Synthetic warranty claims: not a real OEM extract. CBR ≠ RAG.
