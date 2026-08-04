# Forge Synth Lab

**Tier B** cross-domain product proof: tabular synthesis + classical TSTR
utility + unsupervised clustering on synthetic samples.

## Product narrative

Forge is a synthetic data lab for retail catalog tables. A train-only
synthesizer produces samples; classical TSTR checks utility against the real
holdout; clustering explores synthetic segment structure. The platform:

1. Fits a Gaussian-copula synthesizer on train features only
2. Trains a classical classifier on synthetic rows and evaluates on real test (TSTR)
3. Clusters synthetic numeric features with external labels for eval only

## Status

Run `script.py`. Outputs land under `results/` (summary and stage JSON)..

## How to run

```bash
python proofs\forge-synth-lab\script.py
```

## Leakage controls

- Synthesizer fit on train only
- Fidelity / `session.synthetic.evaluate` vs real holdout
- TSTR classifier trained on synthetic rows; metrics on real test
- Cluster fit on synthetic sample's own split

## What fails if leakage is ignored

- Fitting the synthesizer on the full table makes fidelity look perfect
- TSTR that peeks at real test labels during synth training is not utility
- Clustering with test-conditioned features overstates segment purity

## Upstream Tier A building blocks

`tabular-synth-utility`, `synthetic-privacy-utility`, `sku-embedding-clusters`,
`cluster-customer-segments`, `loan-approval-classical`

## Limitations

**NO differential privacy / anonymity claims.** Utility ≠ privacy.
