# Artifacts: checkpoints vs bundles vs Torch/RAG/AI

> **Install (GitHub 2.x):**
> `pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"`
> See [installation](../docs/installation.rst).

BuildML separates **workflow resume** from **deployable scoring** from
**domain-specific trainer/index/transcript** artifacts. Mixing them causes
silent gaps (no weights in a checkpoint, no dataset in a pipeline).

---

## Conceptual matrix

| Artifact | Typical API | Contains | Does **not** contain |
| --- | --- | --- | --- |
| Session checkpoint | `checkpoint_save` / `checkpoint_load` | data, roles, splits, history, optional preprocess **plan objects**, integrity manifest | Fitted estimator weights, Torch trainer, RAG index, AI keys/transcript |
| Estimator model bundle | `save_model` / `load_model` | estimator + feature contract | Preprocess plans, dataset, splits |
| Pipeline bundle | `save_pipeline` / `load_pipeline` | plans + estimator + model card + schema contract | Dataset rows, full history, Torch/RAG |
| Score helper | `predict_from_pipeline` | one-shot inference | Does not mutate Session |
| Torch trainer bundle | `session.dl.save_bundle` / `session.dl.load_bundle` | weights, optimizer (+ scheduler), config, history, feature contract, optional multimodal_preprocess meta | Dataset, split indices; **load does not rebuild DataLoaders** (rebuild with `session.dl.make_multimodal_loaders(..., use_saved_preprocess=True)` or `preprocess=`) |
| RAG bundle | `session.rag.save_bundle` / `session.rag.load_bundle` | embeddings, index, chunk config | Tabular Session data, Torch weights |
| Unsupervised bundle | `session.unsupervised.save_bundle` / `session.unsupervised.load_bundle` | `ClusterPlan` (estimator + feature contract + assign disclosures) | Dataset, splits, Torch/RAG |
| Ensemble bundle | `session.ensemble.save_bundle` / `session.ensemble.load_bundle` | `EnsemblePlan` + FitResult contract (strategy disclosures) | Dataset, splits, preprocess plans |
| AutoML bundle | `session.automl.save_bundle` / `session.automl.load_bundle` | `AutoMLPlan` + FitResult contract (family/recipe disclosures) | Dataset, splits, Session-global plans |
| Forecast bundle | `session.forecast.save_bundle` / `session.forecast.load_bundle` | `ForecastPlan` (baseline or lag estimator + lag/exog contract) | Dataset, splits, Torch/RAG |
| Anomaly bundle | `session.anomaly.save_bundle` / `session.anomaly.load_bundle` | `AnomalyPlan` (estimator + feature/threshold/alert-rate disclosures) | Dataset, splits, Torch/RAG |
| Semi-supervised bundle | `session.semisupervised.save_bundle` / `session.semisupervised.load_bundle` | `SemiSupervisedPlan` (estimator + label missingness contract) | Dataset, splits, Torch/RAG |
| Self-supervised bundle | `session.ssl.save_bundle` / `session.ssl.load_bundle` | `SelfSupervisedPlan` (+ optional `SSLHeadPlan`) | Dataset, splits, Torch trainer weights |
| Active-learning bundle | `session.active_learning.save_bundle` / `session.active_learning.load_bundle` | `ActiveLearningPlan` (estimator + pool indices + query history + budget) | Dataset, splits, fake oracle |
| Online / continual bundle | `session.online.save_bundle` / `session.online.load_bundle` | `OnlinePlan` (incremental estimator + cursor + update history + classes) | Dataset, splits, streaming platform |
| Multi-task bundle | `session.multitask.save_bundle` / `session.multitask.load_bundle` | `MultiTaskPlan` (multi-output / chain estimator + target contract + encoders) | Dataset, splits, deep MTL platform |
| Meta-learning bundle | `session.metalearning.save_bundle` / `session.metalearning.load_bundle` | `MetaLearningPlan` (episodic protocol + task/feature contract + optional warm-start init) | Dataset, splits, foundation-model / MAML-at-scale |
| Federated bundle | `session.federated.save_bundle` / `session.federated.load_bundle` | `FederatedPlan` (global linear/SGD model + client contract + round history) | Dataset, splits, Flower/OpenFL network stack, cryptographic secure aggregation |
| Probabilistic bundle | `session.probabilistic.save_bundle` / `session.probabilistic.load_bundle` | `ProbabilisticPlan` (BayesianRidge/GP/NB + optional conformal quantile) | Dataset, splits, PyMC/Stan MCMC platform, Bayesian deep nets |
| Causal bundle | `session.causal.save_bundle` / `session.causal.load_bundle` | `CausalPlan` (assumptions + nuisance models + train ATE) | Dataset, splits, DoWhy/EconML platform, causal discovery |
| Graph bundle | `session.graph.save_bundle` / `session.graph.load_bundle` | `GraphPlan` (GraphSpec + classical/GCN estimator + label encoder) | Dataset, splits, PyG research suite, Neo4j/KG |
| Symbolic bundle | `session.symbolic.save_bundle` / `session.symbolic.load_bundle` | `SymbolicPlan` / `NeuroSymbolicPlan` (rule KB ± sklearn hybrid) | Dataset, splits, Prolog/Z3, AGI reasoner, fuzzy product |
| CBR bundle | `session.cbr.save_bundle` / `session.cbr.load_bundle` | `CbrPlan` (train case memory + metric/reuse config) | Dataset, splits, RAG corpus, vector DB product |
| Imitation bundle | `session.rl.save_imitation_bundle` / `session.rl.load_imitation_bundle` | `ImitationPlan` (behavioral cloning policy) | Dataset, splits, inverse RL / robotics stack |
| RL bundle | `session.rl.save_bundle` / `session.rl.load_bundle` | `RlPlan` (contextual bandit, tabular Q-table, Gymnasium REINFORCE-lite, or SB3) | Dataset, splits, MuJoCo / multi-agent platform |
| TDA bundle | `session.tda.save_bundle` / `session.tda.load_bundle` | `TdaPlan` (ripser/persim vectorizer + train NN + optional head) | Dataset, splits, Mapper research suite, every TDA paper |
| Recommender bundle | `session.recommender.save_bundle` / `session.recommender.load_bundle` | `RecommenderPlan` (train catalog + matrix + similarities/factors) | Dataset, splits, Netflix-scale platform, RAG corpus, EDA Findings |
| Ranker (LTR) bundle | `session.ranking.save_bundle` / `session.ranking.load_bundle` | `RankerPlan` (feature contract + pointwise/pairwise estimator) | Dataset, splits, search-engine product, RAG corpus, recommender catalog |
| KG bundle | `session.kg.save_bundle` / `session.kg.load_bundle` | `KgPlan` (train vocab + TransE/DistMult embeddings + adjacency) | Dataset, splits, Neo4j, Graph ML node-classify, RAG |
| Decision bundle | `session.decision.save_bundle` / `session.decision.load_bundle` | `DecisionPlan` (threshold / cost matrix / allocation rules) | Dataset, splits, general OR / MIP platform, Optuna HPO |
| NLP bundle | `session.nlp.save_bundle` / `session.nlp.load_bundle` | `NlpTextPlan` (normalization recipe + train-fitted representation + head) ± `NlpTopicPlan` | Dataset, splits, RAG corpus, Torch fine-tuning checkpoint, downloaded encoder weights |
| AI transcript | `session.ai.save_transcript` / `session.ai.load_transcript` | conversation, tool calls, egress manifests | API keys; raw rows unless FULL_SAMPLE opt-in |
| TorchServe pack | `session.dl.pack_torchserve` | directory recipe for operator-owned TorchServe | Running server |
| TensorRT plan | `session.dl.prepare_tensorrt` | `trtexec` plan files | Built `.engine` (operator builds) |
| K8s DDP YAML | `session.dl.emit_k8s_ddp` | Job template | Live multi-cluster orchestration |

Schemas to remember: `buildml.torch_bundle.v1`, `buildml.rag_bundle.v1`,
`buildml.unsupervised_bundle.v1`, `buildml.ensemble_bundle.v1`,
`buildml.automl_bundle.v1`, `buildml.forecast_bundle.v1`,
`buildml.anomaly_bundle.v1`, `buildml.semisupervised_bundle.v1`,
`buildml.selfsupervised_bundle.v1`, `buildml.activelearning_bundle.v1`,
`buildml.online_bundle.v1`, `buildml.multitask_bundle.v1`,
`buildml.metalearning_bundle.v1`,
`buildml.federated_bundle.v1`,
`buildml.probabilistic_bundle.v1`,
`buildml.causal_bundle.v1`,
`buildml.graph_bundle.v1`,
`buildml.symbolic_bundle.v1`,
`buildml.cbr_bundle.v1`,
`buildml.imitation_bundle.v1`,
`buildml.rl_bundle.v1`,
`buildml.tda_bundle.v1`,
`buildml.recommender_bundle.v1`,
`buildml.ranker_bundle.v1`,
`buildml.kg_bundle.v1`,
`buildml.decision_bundle.v1`,
`buildml.nlp_bundle.v1`,
`buildml.ai.transcript.v1`.

---

## Use case: checkpoint mid-loop, pipeline at the end

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

frame = pd.DataFrame(
    {
        "age": [21, None, 35, 40, 29, 33, 52, 47],
        "income": [40, 55, 60, 80, 50, 70, 90, 65],
        "approved": [0, 1, 0, 1, 0, 1, 1, 0],
    }
)

session = (
    Session.ingest(frame)
    .set_roles({"age": "feature", "income": "feature", "approved": "target"})
    .split(test_size=0.25, stratify=True, random_state=0)
    .impute(strategy="median")
    .scale(method="standard")
)

session.checkpoint_save(
    "artifacts/checkpoint",
    sidecar_layout="auto",
    sidecar_partition_rows=25_000,
    sidecar_compression="zstd",
)

restored = Session.checkpoint_load("artifacts/checkpoint")
print(restored.reattach_result.status)

restored.fit(LogisticRegression(max_iter=500), task="classification")
restored.save_pipeline("artifacts/pipeline", evaluate_partition="test")
print(restored.model_card.lineage.get("plans_present"))

# Estimator-only (no plans): prefer pipeline when prep must travel:
restored.save_model("artifacts/model_only")
```

`data_only=True` on load deliberately discards prior workflow semantics: use
when you want the frame without replaying history.

---

## Use case: predict_from_pipeline on new rows

```python
from buildml.pipeline import predict_from_pipeline

holdout = restored.partition("test")
scored = predict_from_pipeline(
    "artifacts/pipeline",
    holdout,
    return_proba=True,
)
print(scored)
```

Schema mismatches raise clearly. Resample plans do not synthesize inference rows.

---

## Use case: Torch / RAG / AI stay separate

```python
# Torch (buildml[torch])
# session.dl.save_bundle("artifacts/torch_bundle")
# restored.dl.load_bundle(path, module, map_location="cpu")
# restored.dl.make_loaders(...)  # required again: load does not rebuild loaders

# RAG (buildml[rag])
# session.rag.save_bundle("artifacts/rag_bundle")
# Session().rag.load_bundle("artifacts/rag_bundle")

# AI (buildml[ai])
# session.ai.save_transcript("artifacts/transcript.json")  # secrets redacted
```

Serving a pipeline or TorchScript artifact:
[serve-deploy](serve-deploy.md).

---

## Reattach statuses

Inspect `reattach_result` after checkpoint load. Typical outcomes include
resume-ready vs blocked (schema/integrity mismatch) vs fresh-ingest guidance.
Do not assume a checkpoint is a deployable model.

---

## Failure modes

| Mistake | Consequence |
| --- | --- |
| Expecting weights in a checkpoint | No estimator: call `save_pipeline` / `session.dl.save_bundle` |
| Expecting dataset in a pipeline | Scoring artifact only |
| Loading Torch bundle and evaluating without loaders | `ValidationError`: rebuild correct loader kind |
| Committing AI transcripts with FULL_SAMPLE | Privacy risk: prefer STATS_ONLY + redact |
| Treating TorchServe/TRT/K8s helpers as managed cloud | Recipes/templates only |

---

## Related

- [Classical end-to-end](classical-end-to-end.md)
- [Torch deep](torch-deep.md)
- [RAG deep](rag-deep.md)
- [Serve & deploy](serve-deploy.md)
- [AI safety](ai-operator-safety.md)
