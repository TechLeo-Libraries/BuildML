# eda-industry-adaptability

You want evidence that Industry EDA surfaces (Static EDA HTML and the live
Dashboard / App sheet) adapt across many dataset shapes: real sklearn tables
and synthetic stress frames. The question is whether readiness sheet
completeness, report fit, and adapt guidance are bound to each table, not
copied from demo columns.

## Data

Twelve cases shared with `scripts/eda_adaptability_gauntlet.py`:

| Dataset | Kind | Task | Stress |
| --- | --- | --- | --- |
| iris | sklearn | classification | small, clean multiclass |
| wine | sklearn | classification | wider chemistry features |
| breast_cancer | sklearn | classification | higher-dim binary |
| diabetes | sklearn | regression | continuous target |
| california_housing_2.5k | sklearn | regression | larger n (sampled 2.5k) |
| titanic_like | synthetic-realworld | classification | missingness + categoricals |
| synthetic_dirty_cls | synthetic-buildml | classification | dirty churn-style frame |
| high_cardinality | synthetic-buildml | classification | near-id SKU + imbalance |
| wide_many_cols | synthetic-buildml | classification | wide p, column missingness |
| small_n_textish | synthetic-buildml | classification | tiny n, messy strings |
| tall_regression_spikes | synthetic-buildml | regression | tall n, heavy spikes |
| no_target_profile | synthetic-buildml | unsupervised | no target profile |

## Leakage

EDA is screening only: no model fit, no transform fit that poisons holdout.
When a target is declared, Session `split` runs for partition context;
Static and App still report full-dataset EDA diagnostics (disclosed as
exploration, not causal discovery).

## How to run

```bash
pip install -e ".[dashboard,dev]"
python proofs/eda-industry-adaptability/script.py
python scripts/eda_adaptability_gauntlet.py
```

The gauntlet is the same twelve cases with an alternate artifact root.
Requires `buildml[dashboard]` for App evidence.

## What you'll get

Pass/fail per dataset on Static markers (register, ledger, sequence,
assumptions, Offline HTML primary; no CSV/PDF briefing header) and App
payloads (kpis, register, ledger, assumptions, adapt binding, gates/academy).
Aggregate in `results/results.json` (`metrics.n_passed` / `n_datasets`).

Artifacts (gitignored): `results/cases/*_static.html`, `*_app.json`,
`summary.md` / `summary.json`.

There is no sklearn metric twin. Parity is Static HTML versus App sheet /
API payloads on the same report object (`session.eda`, then
`export_eda_html` on the research HTML path, plus DashboardState /
`/api/cockpit`, `/api/gates`, `/api/domains/academy`).

## Limitations

Screening evidence only; not deployment certification. California housing
fetch needs sklearn dataset download cache on first run. Requires
`buildml[dashboard]` for App evidence.

Related: [EDA / Teaching Studio](../../guides/eda-teaching-studio.md).
