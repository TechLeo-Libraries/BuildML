# wine-cluster-segments

## Business purpose

Cluster wine chemical profiles and validate against known cultivars with
external ARI / NMI. Exercises unsupervised Session APIs on a **real public
dataset**.

## Data source

**REAL_PUBLIC_DATASET** — `sklearn.datasets.load_wine` (UCI Wine recognition,
redistributed with sklearn). Offline; no network. Cultivar is used only as an
external validation label (`role=ignore`).

## Leakage controls

- Random train / validation / test before scale / PCA / cluster fit
- Scale + PCA + `unsupervised.fit` on train only
- Cultivar never used as a fit target
- Test evaluation after the model is locked

## BuildML API steps

1. `ingest` → roles → `split` → `scale` → `reduce_dimensions(pca)`
2. `session.unsupervised.fit(method="kmeans", n_clusters=3)`
3. `session.unsupervised.evaluate` on validation/test with external cultivar
4. `session.unsupervised.save_bundle`

## Metrics

Internal cluster quality + external ARI / NMI. Refuses ARI/NMI ≥ 1.0 and
ARI ≥ 0.98 (anti near-perfect theater).

## How to run

```bash
python proofs/wine-cluster-segments/script.py
```

## Limitations

Small n; cultivar labels exist for research validation only.
