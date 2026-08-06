# BuildML Proof Suite

Deep proofs that BuildML Session contracts work end-to-end on the executed
configuration — not thin unit tests, and **not** a claim that every optional
industry wheel is healthy on every host. Each project uses honest splits,
train-only fitting, holdout evaluation, persisted artifacts where applicable,
and JSON metrics under `results/` (gitignored).

| Tier | Count | What it proves |
| --- | ---: | --- |
| **A** | **63/63** | One named product scenario per major Session domain (incl. ensembles + Torch + fairness + real public datasets + Industry EDA) |
| **B** | **36/36** | Cross-domain products composing multiple Session surfaces |
| **C** | **58/62** | Same-split industry twin + `comparison.json` per Tier A (when present / runnable; real-public cohort may be A-only) |

The suite grew by +30 Tier A, +30 Tier B, and +30 Tier C twins beyond the
baseline cohort (ensembles, Torch/DL, and previously uncomposed Tier B domains).

Optional deepeners are not failures: `gymnasium` via `buildml[rl]` enables
CartPole / FrozenLake paths; TDA prefers editable `pip install -e ".[tda]"`;
Torch / FLAML / PyKEEN / etc. deepen paths only when **runtime import probes**
succeed — `find_spec` alone is not enough. See capability-matrix
`*_import_honesty` fields and `scripts/probe_industry_extras.py`.

**Richer backends when installed:**

| Package | Proof path deepened |
| --- | --- |
| `implicit` | `movie-recs-collaborative`, `catalog-recs-implicit` → ALS |
| `sentence-transformers` | `support-kb-rag`, `policy-handbook-rag` → dense / auto embedder |
| `flaml` (+ GBDT extras) | `churn-automl-search`, Tier B `ledger-underwriting-studio` / `orbit-multitask-hub` / `keystone-underwrite-ml` |
| AutoGluon | preferred when FLAML absent and `autogluon.tabular` imports |
| `torch` | `torch-tabular-underwrite`, `torch-text-intent`, Tier B `nova-torch-bench` |
| `ripser` / `persim` | `credit-tda-shape`, `process-tda-shape`, Tier B `prism` / `kiln` / `volt` |

Library note: default `Session.scale` / `encode` / `impute` (and related
numeric/categorical preprocess) skip `ignore` / `id` / `target` / `group` /
`time` / `weight` roles: knapsack cost columns stay usable without
feature-scoped workarounds (Tier B `aegis` / `ledger` / decision-heavy products).

---

## How to prove BuildML works

1. Install Session 2.x (PyPI or this checkout):
   ```bash
   pip install buildml
   # or for proofs development:
   pip install -e ".[dev]"
   # domain extras as needed, e.g.:
   pip install -e ".[tda,rl,rag,torch]"
   ```
2. Run one project, the CI smoke subset (always re-runs; no skip-existing), or
   the full harness from the repo root:
   ```bash
   python proofs/loan-approval-classical/script.py
   # CI proofs-smoke installs .[dev,dashboard] so eda-industry-adaptability runs
   pip install -e ".[dev,dashboard]"
   python -m proofs._lib.run_all --smoke
   python -m proofs._lib.run_all --tier all
   ```
3. Read `proofs/<slug>/results/*.json`: look for `"status": "completed"`.
   Harness semantics:
   - Process exit 0 alone is **not** enough under `--smoke`.
   - `--smoke` re-runs the CI Tier A subset and treats `skipped_missing_extra` /
     `partial` as **`unexpected_skip`** (harness exit 1) unless you also
     pass `--allow-skip`.
   - `CI_SMOKE_TIER_A` includes `eda-industry-adaptability` (Static Offline HTML
     + App sheet across 12 datasets); CI installs `buildml[dashboard]` for that
     gate. Unit contracts also live in `tests/unit/test_eda_adaptability.py` and
     `tests/unit/test_final_quality_gates.py`.
   - Smoke validates **selected runnable Session contracts** and result JSON
     status — not performance certification, public-dataset leaderboard proof,
     or every optional industry wheel.
   - Several Tier A scripts refuse perfect holdout scores (metric ceilings) on
     intentionally noisy / overlapping synthetics and on **REAL_PUBLIC_DATASET**
     proofs (holdout scores must stay strictly below 1.0 where gated).
   - Real public-dataset proofs publish provenance under `results.json` → `data`
     (`name`, `source`, `license`/`provenance`, `n_rows`, `n_features`, `task`,
     `evidence_tier=REAL_PUBLIC_DATASET`).
   - Local non-smoke runs still allow soft-skips by default for optional extras.
   Tier C writes `results/comparison.json` on the **same split** as the BuildML path.
   A twin may be an in-tree fallback when its optional industry wheel is absent
   or fails a runtime import probe.

Shared helpers live in [`_lib/`](_lib/) (seed, results writer, leakage asserts,
synthetic loaders, extra probes, Tier C `write_comparison`, `run_all`).

### Optional extras (TDA and others)

Prefer an **editable** install from the repo so `pyproject.toml` extras resolve:

```bash
pip install -e ".[tda]"
# or a fuller set, e.g.
pip install -e ".[tda,automl,rag,rl,torch]"
```

`pip install "buildml[tda]"` against a **non-editable / older wheel** can fail
because that installed package metadata may not declare the `tda` extra even
when the current `pyproject.toml` does. If you already installed ripser/persim
directly, Tier A TDA proofs can still complete; the editable extra is
the supported path for fresh envs.

**Torch:** proof JSON may include a `torch` block (`skip_torch_paths=false`
when a working CPU/CUDA wheel imports). Windows / Python 3.13 often use a
CPU wheel from the PyTorch index.

---

## Interpreting Tier C comparisons (qualitative bar 5-B)

Tier C is a **same-split industry twin**, not a bake-off for bragging rights.

- Deltas are descriptive on **one synthetic (or public-demo) draw**.
- Prefer reading **workflow parity** and **leakage discipline** over tiny metric
  gaps: BuildML must fit/select on train/validation only and evaluate on held-out
  test; the twin must use the same indices.
- Competitive bar **5-B**: BuildML should be in the same qualitative band as a
  competent sklearn / classical baseline: not required to dominate every metric.
- `"status": "filled"` means the twin ran and wrote `comparison.json`; it does
  **not** mean production certification.

---

## Docs inventory (sync to R1–R6 reality)

| Area | Canonical guide(s) | Proof deep-link |
| --- | --- | --- |
| Root overview | [`README.md`](../README.md) | → `proofs/` |
| Guide index | [`guides/README.md`](../guides/README.md) | Mapping table + per-domain links |
| Industry EDA | `eda-teaching-studio.md`, `docs/usage.rst` | → `eda-industry-adaptability` (Static + App, 12 datasets) |
| Classical | `quickstart-classical.md`, `classical-end-to-end.md` | → `loan-approval-classical`, `mortgage-default-classical`, `claim-severity-regression`, `breast-cancer-classical`, `diabetes-progression-regression` |
| Ensembles | `quickstart-ensemble.md` (when present) | → `voting-ensemble-attrition`, `stacking-credit-risk`, `blending-payment-risk` · Tier B `citadel-ensemble-desk`, `keystone-underwrite-ml` |
| Torch / DL | `quickstart-dl.md` / Torch guides | → `torch-tabular-underwrite`, `torch-text-intent` · Tier B `nova-torch-bench` |
| AutoML | `quickstart-automl.md`, `automl-deep.md` | → `churn-automl-search` · Tier B `ledger`, `orbit`, `keystone` |
| Anomaly | `quickstart-anomaly.md` | → `network-intrusion-anomaly`, `payment-rail-anomaly`, `iot-sensor-anomaly` · Tier B `aegis`, `sentinel`, `rivulet`, `volt` |
| Forecasting / TS | `quickstart-forecasting.md`, `quickstart-timeseries-analysis.md` | → `store-sales-forecast`, `energy-load-forecast` · Tier B `harbor`, `ballast`, `terrace` |
| RAG | `quickstart-rag.md` | → `support-kb-rag`, `policy-handbook-rag` · Tier B `pulse`, `parchment`, `helix`, `zenith` |
| Unsupervised | `quickstart-unsupervised.md` | → `cluster-customer-segments`, `sku-embedding-clusters`, `wine-cluster-segments` · Tier B `canyon`, `forge`, `kiln` |
| Recommenders / LTR | `quickstart-recommenders.md`, `quickstart-ranking.md` | → `movie-recs-collaborative`, `catalog-recs-implicit`, `search-relevance-ltr`, `sponsored-ad-ltr` · Tier B `meridian`, `aurora`, `compass` |
| KG / TDA / Causal / Graph / Prob | matching quickstarts | → matching Tier A + expansion slugs · Tier B `helix`, `prism`, `lattice`, `apex`, `cornerstone`, `ballast`, `relay` |
| Semi / AL / SSL | matching quickstarts | → Tier A + expansion · Tier B `atlas`, `beacon`, `zenith` |
| Online / Decisions | matching quickstarts | → Tier A + expansion · Tier B `aegis`, `harbor`, `ledger`, `rivulet`, `campaign` products |
| Federated / Multi-task / Meta / Symbolic / CBR / Synthetic / IL+RL | matching quickstarts | → matching Tier A (+ Tier B where listed) |
| NLP (text) | `quickstart-nlp.md`, `nlp-deep.md` | → `ticket-routing-nlp`, `torch-text-intent` · Tier B `folio-claims-nlp`, `zenith-support-os` |
| Sphinx | `docs/index.rst`, `docs/features.rst`, `docs/guide-index.rst` | Mirror Markdown proof pointers |

---

## Tier A: Single-domain deep projects

### Baseline cohort (1–27)

| # | Project | Domain | Status | Notes |
| ---: | --- | --- | --- | --- |
| 1 | [loan-approval-classical](loan-approval-classical/) | Classical supervised | **completed** | Tier C `comparison.json` filled |
| 2 | [churn-automl-search](churn-automl-search/) | AutoML | **completed** | Tier C RandomizedSearchCV twin |
| 3 | [network-intrusion-anomaly](network-intrusion-anomaly/) | Anomaly | **completed** | Tier C IsolationForest twin |
| 4 | [store-sales-forecast](store-sales-forecast/) | Forecast + TS analysis | **completed** | Tier C SARIMAX / seasonal_naive |
| 5 | [support-kb-rag](support-kb-rag/) | RAG | **completed** | Tier C TF-IDF cosine twin |
| 6 | [movie-recs-collaborative](movie-recs-collaborative/) | Recommenders | **completed** | Tier C item-cosine twin |
| 7 | [search-relevance-ltr](search-relevance-ltr/) | Learning-to-rank | **completed** | Tier C Ridge pointwise twin |
| 8 | [kg-biomed-linkpred](kg-biomed-linkpred/) | Knowledge graphs | **completed** | Tier C co-occurrence PMI twin |
| 9 | [credit-tda-shape](credit-tda-shape/) | TDA | **completed** | ripser/persim; Tier C logistic twin |
| 10 | [semi-label-efficiency](semi-label-efficiency/) | Semi-supervised | **completed** | Tier C LabelPropagation twin |
| 11 | [active-labeling-budget](active-labeling-budget/) | Active learning | **completed** | Tier C margin-sampling twin |
| 12 | [stream-fraud-online](stream-fraud-online/) | Online / continual | **completed** | Tier C SGD partial_fit twin |
| 13 | [multi-target-underwriting](multi-target-underwriting/) | Multi-task | **completed** | Tier C MultiOutputClassifier twin |
| 14 | [few-shot-domain-adapt](few-shot-domain-adapt/) | Meta-learning | **completed** | Tier C NearestCentroid k-shot twin |
| 15 | [policy-rules-neuro-symbolic](policy-rules-neuro-symbolic/) | Symbolic | **completed** | Tier C DecisionTree twin |
| 16 | [case-memory-claims](case-memory-claims/) | CBR | **completed** | Tier C KNeighbors twin |
| 17 | [cost-sensitive-collections](cost-sensitive-collections/) | Optimize / decisions | **completed** | Tier C val cost-threshold twin |
| 18 | [synthetic-privacy-utility](synthetic-privacy-utility/) | Synthetic | **completed** | Tier C column-bootstrap twin |
| 19 | [cluster-customer-segments](cluster-customer-segments/) | Unsupervised | **completed** | Tier C KMeans+PCA twin |
| 20 | [ssl-representation-probe](ssl-representation-probe/) | Self-supervised | **completed** | Tier C PCA+probe twin |
| 21 | [causal-treatment-effect](causal-treatment-effect/) | Causal | **completed** | Tier C sklearn AIPW twin |
| 22 | [federated-hospital-sim](federated-hospital-sim/) | Federated | **completed** | Tier C pooled SGD twin |
| 23 | [graph-fraud-rings](graph-fraud-rings/) | Graph | **completed** | Tier C networkx+LR twin |
| 24 | [prob-interval-risk](prob-interval-risk/) | Probabilistic | **completed** | Tier C BayesianRidge+quantile |
| 25 | [imitation-cartpole-control](imitation-cartpole-control/) | IL + RL | **completed** | Tier C sklearn BC twin; gym REINFORCE when `gymnasium` installed |
| 26 | [ticket-routing-nlp](ticket-routing-nlp/) | NLP (text) | **completed** | Tier C `TfidfVectorizer`+`LogisticRegression` Pipeline twin |
| 27 | [tabular-q-frozenlake](tabular-q-frozenlake/) | Tabular RL | **completed** | FrozenLake Q-learning when `gymnasium` present |

### Expansion cohort (28–57)

| # | Project | Domain | Status | Notes |
| ---: | --- | --- | --- | --- |
| 28 | [mortgage-default-classical](mortgage-default-classical/) | Classical | **completed** | Mortgage default (distinct from consumer loan) |
| 29 | [claim-severity-regression](claim-severity-regression/) | Classical regression | **completed** | P&C severity |
| 30 | [voting-ensemble-attrition](voting-ensemble-attrition/) | Ensemble voting | **completed** | Soft voting LR+RF |
| 31 | [stacking-credit-risk](stacking-credit-risk/) | Ensemble stacking | **completed** | OOF stacking CV inside train |
| 32 | [blending-payment-risk](blending-payment-risk/) | Ensemble blending | **completed** | Train-holdout blend |
| 33 | [torch-tabular-underwrite](torch-tabular-underwrite/) | Torch tabular | **completed** | Honest skip if Torch unavailable |
| 34 | [torch-text-intent](torch-text-intent/) | Torch text | **completed** | Support-ticket intent; skip if no Torch |
| 35 | [payment-rail-anomaly](payment-rail-anomaly/) | Anomaly | **completed** | Payment authorization rails |
| 36 | [iot-sensor-anomaly](iot-sensor-anomaly/) | Anomaly | **completed** | Factory IoT sensors |
| 37 | [energy-load-forecast](energy-load-forecast/) | Forecast | **completed** | Hourly grid load |
| 38 | [weather-prob-intervals](weather-prob-intervals/) | Probabilistic | **completed** | Conformal / Bayesian intervals |
| 39 | [policy-handbook-rag](policy-handbook-rag/) | RAG | **completed** | Policy handbook corpus |
| 40 | [catalog-recs-implicit](catalog-recs-implicit/) | Recommenders | **completed** | E-commerce catalog |
| 41 | [sponsored-ad-ltr](sponsored-ad-ltr/) | LTR | **completed** | Sponsored ad judgments |
| 42 | [logistics-kg-linkpred](logistics-kg-linkpred/) | KG | **completed** | Logistics / supply triples |
| 43 | [process-tda-shape](process-tda-shape/) | TDA | **completed** | Manufacturing process clouds |
| 44 | [radiology-semi-labels](radiology-semi-labels/) | Semi-supervised | **completed** | Imaging-feature proxy |
| 45 | [defect-active-budget](defect-active-budget/) | Active learning | **completed** | Defect labeling budget |
| 46 | [clickstream-online](clickstream-online/) | Online | **completed** | Conversion stream |
| 47 | [sku-multitask-retail](sku-multitask-retail/) | Multi-task | **completed** | Buy + high-margin |
| 48 | [coldstart-meta-adapt](coldstart-meta-adapt/) | Meta-learning | **completed** | Cold-start domain adapt |
| 49 | [compliance-neuro-symbolic](compliance-neuro-symbolic/) | Symbolic | **completed** | Compliance rules |
| 50 | [warranty-cbr-memory](warranty-cbr-memory/) | CBR | **completed** | Warranty case memory |
| 51 | [campaign-budget-optimize](campaign-budget-optimize/) | Optimize / decisions | **completed** | Campaign knapsack / threshold |
| 52 | [tabular-synth-utility](tabular-synth-utility/) | Synthetic | **completed** | Utility / TSTR disclosure |
| 53 | [sku-embedding-clusters](sku-embedding-clusters/) | Unsupervised | **completed** | Product embedding clusters |
| 54 | [tabular-ssl-probe](tabular-ssl-probe/) | Self-supervised | **completed** | Masked tabular pretext |
| 55 | [uplift-marketing-causal](uplift-marketing-causal/) | Causal | **completed** | Marketing uplift AIPW |
| 56 | [edge-fleet-federated](edge-fleet-federated/) | Federated | **completed** | Edge device clients |
| 57 | [peer-lending-graph](peer-lending-graph/) | Graph | **completed** | P2P lending rings |

### Real public-dataset cohort (58–61)

| # | Project | Domain | Status | Notes |
| ---: | --- | --- | --- | --- |
| 58 | [breast-cancer-classical](breast-cancer-classical/) | Classical supervised | **completed** | **REAL_PUBLIC_DATASET** sklearn breast_cancer (offline) |
| 59 | [diabetes-progression-regression](diabetes-progression-regression/) | Classical regression | **completed** | **REAL_PUBLIC_DATASET** sklearn diabetes (offline) |
| 60 | [wine-cluster-segments](wine-cluster-segments/) | Unsupervised | **completed** | **REAL_PUBLIC_DATASET** sklearn wine + external ARI (offline) |
| 61 | [adult-fairness-observational](adult-fairness-observational/) | Fairness (observational) | **completed** | **REAL_PUBLIC_DATASET** OpenML Adult/credit-g; offline disclosed proxy fallback |

### Industry EDA cohort (62–63)

| # | Project | Domain | Status | Notes |
| ---: | --- | --- | --- | --- |
| 62 | [loan-fairness-observational](loan-fairness-observational/) | Fairness (observational) | **completed** | Synthetic credit; sensitive `region` |
| 63 | [eda-industry-adaptability](eda-industry-adaptability/) | Industry EDA | **completed** | 12 datasets; Static research HTML + App sheet/API; Offline HTML primary |

Each Tier A README includes: business purpose, data source, leakage controls,
BuildML API steps, metrics, limitations, and an **Industry comparison** section
(where a Tier C twin exists).

---

## Tier B: Named cross-domain products

### Baseline cohort

| Product | Domains combined | Status |
| --- | --- | --- |
| [Aegis Fraud Platform](aegis-fraud-platform/) | graph + anomaly + supervised + online + decision thresholds + optional rules | **completed** |
| [Harbor Demand Desk](harbor-demand-desk/) | TS analysis + forecast + optimize allocation + probabilistic intervals | **completed** |
| [Atlas Label Studio](atlas-label-studio/) | SSL + semi-supervised + active learning budget loop | **completed** |
| [Pulse Support Copilot](pulse-support-copilot/) | RAG + ranking + CBR case memory + symbolic guardrails | **completed** |
| [Ledger Underwriting Studio](ledger-underwriting-studio/) | classical + AutoML + causal assumptions + cost-sensitive decisions + calibration | **completed** |
| [Nexus Federated Clinical](nexus-federated-clinical/) | federated sim + probabilistic uncertainty + evaluation disclosures | **completed** |

### Expansion cohort (30)

| Product | Domains combined | Status |
| --- | --- | --- |
| [Meridian Recs Commerce](meridian-recs-commerce/) | recommenders + ranking + classical + decisions | **completed** |
| [Helix Knowledge Mesh](helix-knowledge-mesh/) | KG + RAG + symbolic | **completed** |
| [Prism Shape Monitor](prism-shape-monitor/) | TDA + anomaly + classical | **completed** |
| [Orbit Multi-Task Hub](orbit-multitask-hub/) | multitask + AutoML + decisions | **completed** |
| [Quasar Meta Adapt](quasar-meta-adapt/) | metalearning + SSL + classical | **completed** |
| [Forge Synth Lab](forge-synth-lab/) | synthetic + classical TSTR + clusters | **completed** |
| [Canyon Segment Studio](canyon-segment-studio/) | unsupervised + classical + decisions | **completed** |
| [Vector Control Deck](vector-control-deck/) | IL/RL + decisions + classical | **completed** |
| [Citadel Ensemble Desk](citadel-ensemble-desk/) | voting/stacking + anomaly + decisions | **completed** |
| [Nova Torch Bench](nova-torch-bench/) | torch + classical + probabilistic | **completed** |
| [Sentinel IoT Watch](sentinel-iot-watch/) | anomaly + online + forecast | **completed** |
| [Ballast Energy Desk](ballast-energy-desk/) | forecast + probabilistic + optimize | **completed** |
| [Parchment Policy Copilot](parchment-policy-copilot/) | RAG + ranking + CBR | **completed** |
| [Lattice Supply Graph](lattice-supply-graph/) | graph + KG + classical | **completed** |
| [Beacon Label Factory](beacon-label-factory/) | SSL + semi-supervised + active learning | **completed** |
| [Rivulet Stream Risk](rivulet-stream-risk/) | online + anomaly + decisions | **completed** |
| [Cornerstone Mortgage Suite](cornerstone-mortgage-suite/) | classical + causal + decisions | **completed** |
| [Apex Uplift Studio](apex-uplift-studio/) | causal + classical + decisions | **completed** |
| [Relay Edge Federated](relay-edge-federated/) | federated + probabilistic + classical | **completed** |
| [Mosaic Warranty Desk](mosaic-warranty-desk/) | CBR + symbolic + classical | **completed** |
| [Kiln Process TDA](kiln-process-tda/) | TDA + unsupervised + anomaly | **completed** |
| [Aurora Ad Ranker](aurora-ad-ranker/) | ranking + classical + decisions | **completed** |
| [Compass Catalog Recs](compass-catalog-recs/) | recommenders + graph + classical | **completed** |
| [Folio Claims NLP](folio-claims-nlp/) | NLP + CBR + symbolic | **completed** |
| [Dynamo Click Lab](dynamo-click-lab/) | online + metalearning + classical | **completed** |
| [Scaffold Compliance AI](scaffold-compliance-ai/) | symbolic + neuro-symbolic + decisions | **completed** |
| [Terrace Retail Mesh](terrace-retail-mesh/) | multitask + forecast + recommenders | **completed** |
| [Volt Sensor Fusion](volt-sensor-fusion/) | anomaly + TDA + classical | **completed** |
| [Keystone Underwrite ML](keystone-underwrite-ml/) | stacking + AutoML + causal | **completed** |
| [Zenith Support OS](zenith-support-os/) | RAG + NLP + active learning | **completed** |

---

## Tier C: Industry baseline twins

For each completed Tier A: `baseline_industry.py` **or** a comparison section
in `script.py` on the **same split**, writing `results/comparison.json`.

Most Tier A projects ship a Tier C twin (`baseline_industry.py`, or
embedded comparison for `loan-approval-classical`). Expansion twins follow the
same `write_comparison` envelope and leakage disclosures as the baseline cohort.
The **REAL_PUBLIC_DATASET** cohort (breast-cancer / diabetes / wine / adult-fairness)
is Tier A evidence-first; Tier C twins may be added later.

Re-run Tier C after the matching Tier A `script.py`:

```bash
.\.venv\Scripts\python.exe proofs\<slug>\script.py
.\.venv\Scripts\python.exe proofs\<slug>\baseline_industry.py
# or
.\.venv\Scripts\python.exe -m proofs._lib.run_all --tier C
```

---

## Cleanup / gitignore policy

- Generated `proofs/**/results/` and `proofs/**/artifacts/` are gitignored.
- Runners (`script.py`, `baseline_industry.py`), READMEs, and `_lib/` stay tracked.
- `benchmarks/**/results/` gitignored; benchmark runners kept.

---

## Optional backlog (not blockers)

1. AutoGluon on Py3.13 / Windows when upstream wheels resolve (FLAML covers
   industry AutoML today).
2. LightFM / learn2learn / giotto-tda / neuralforecast where markers still skip.
3. Release cut (PyPI 2.x): only when explicitly requested.
