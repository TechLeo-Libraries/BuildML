# warranty-cbr-memory

You have warranty claim features and an approve/deny label. You want similar
historical cases via case-based reasoning: distinct from insurance claims
memory in `case-memory-claims`.

## Data

Synthetic / in-script license-clear warranty claim feature table generated
by `script.py`.

## Leakage

Stratified split before CBR fit. Case memory is built from train only. Test
retrieval / evaluation after lock.

## How to run

```bash
python proofs/warranty-cbr-memory/script.py
python proofs/warranty-cbr-memory/baseline_industry.py
```

## What you'll get

`results/results.json` with accuracy and retrieval metadata.
`results/comparison.json` is sklearn `KNeighborsClassifier` on the same
split.

## Limitations

CBR is not RAG; synthetic warranty claims only.

Related: [CBR quickstart](../../guides/quickstart-cbr.md),
[examples/cbr_knn_loop.py](../../examples/cbr_knn_loop.py).
