# forge-synth-lab

This script composes `session.synthetic`, classical `session.fit` (TSTR),
and `session.unsupervised` on one synthetic retail catalog table. It is not
a product BuildML ships.

A train-only synthesizer produces samples. A classical classifier trained
on synthetic rows is evaluated on real test (TSTR). Clustering explores
synthetic segment structure with external labels for eval only.

## Data

Synthetic retail catalog table.

## Leakage

Synthesizer fit on train only. Fidelity / `session.synthetic.evaluate` vs
real holdout. TSTR classifier trained on synthetic rows; metrics on real
test. Cluster fit on the synthetic sample's own split.

## What fails if leakage is ignored

Fitting the synthesizer on the full table makes fidelity look perfect. TSTR
that peeks at real test labels during synth training is not utility.
Clustering with test-conditioned features overstates segment purity.

## How to run

```bash
python proofs/forge-synth-lab/script.py
```

## What you'll get

`results/` summary and per-stage JSON.

## Upstream

`tabular-synth-utility`, `synthetic-privacy-utility`,
`sku-embedding-clusters`, `cluster-customer-segments`,
`loan-approval-classical`.

## Limitations

No differential privacy / anonymity claims. Utility is not privacy.
