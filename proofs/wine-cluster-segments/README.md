# wine-cluster-segments

You have wine chemical profiles and want clusters you can check against known
cultivars. This is the unsupervised Session path on a public table, with
cultivar held out of fit.

## Data

**REAL_PUBLIC_DATASET** -- `sklearn.datasets.load_wine` (UCI Wine recognition,
redistributed with sklearn). Offline; no network. Cultivar is used only as an
external validation label (`role=ignore`).

## Leakage

Random train / validation / test before scale, PCA, or cluster fit. Scale,
PCA, and `session.unsupervised.fit` run on train only. Cultivar is never a
fit target. Test evaluation runs after the model is locked.

## How to run

```bash
python proofs/wine-cluster-segments/script.py
python proofs/wine-cluster-segments/baseline_industry.py
```

## What you'll get

`results/results.json` with internal cluster quality plus external ARI / NMI.
The script refuses ARI/NMI >= 1.0 and ARI >= 0.98. `results/comparison.json`
is sklearn `StandardScaler` + `PCA` + `KMeans` on the same `SplitPlan`.
Cultivar labels stay evaluation-only.

## Limitations

Small n; cultivar labels exist for research validation only.

Related: [Unsupervised quickstart](../../guides/quickstart-unsupervised.md).
