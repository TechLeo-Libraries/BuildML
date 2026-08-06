# eda-industry-adaptability

## Business purpose

Prove that Industry EDA surfaces (BUILDML STATIC EDA research HTML and the
live Dashboard / App sheet) adapt across many dataset shapes: real sklearn
tables and synthetic stress frames. Operators need evidence that readiness
sheet completeness, report fit, and adapt guidance are not demo-column
templates.

## Data source

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

## Leakage controls

EDA is screening only: no model fit, no transform fit that poisons holdout.
When a target is declared, Session `split` runs for partition context; Static
and App still report full-dataset EDA diagnostics (disclosed as exploration,
not causal discovery).

## BuildML API steps

1. `Session.ingest` → `set_roles` → optional `split`
2. `session.eda(include_plots=False)` materializes the report
3. Static: `export_eda_html(..., html_format research path)` → Offline HTML primary
4. App: DashboardState + `/api/cockpit`, `/api/gates`, `/api/domains/academy`

## Metrics

Pass/fail per dataset on Static markers (register, ledger, sequence,
assumptions, Offline HTML primary; no CSV/PDF briefing header) and App
payloads (kpis, register, ledger, assumptions, adapt binding, gates/academy).
Aggregate in `results/results.json` (`metrics.n_passed` / `n_datasets`).

## How to run

```bash
pip install -e ".[dashboard,dev]"
python proofs/eda-industry-adaptability/script.py
# same cases, alternate artifact root:
python scripts/eda_adaptability_gauntlet.py
```

Artifacts (gitignored): `results/cases/*_static.html`, `*_app.json`,
`summary.md` / `summary.json`.

## Industry comparison (Tier C)

No sklearn metric twin. Parity is Static research HTML versus App sheet /
API payloads on the same report object (workflow surface parity).

## Limitations

Screening evidence only; not deployment certification. California housing
fetch needs sklearn dataset download cache on first run. Requires
`buildml[dashboard]` for App evidence.
