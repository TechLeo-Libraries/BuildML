# ruff: noqa: E501
"""Federated learning (local FedAvg simulation) concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

FEDERATED_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="federated-simulation",
            title="Local federated learning simulation on Session clients",
            summary="Partition train rows by a client/group column; run local updates; average coefficients in-process — not a networked FL platform.",
            definition=(
                "Federated learning in BuildML simulates FedAvg-style rounds on "
                "Session data: each distinct client/group id is a 'client' whose "
                "local updates see only that client's train rows. The server "
                "aggregates coef_/intercept_ with sample-size weights in-process."
            ),
            intuition=(
                "Many small sites train privately on their own sheets, then send "
                "weight updates to a coordinator that averages them — here the "
                "sites are columns in one table and the coordinator is local."
            ),
            formal_idea=(
                "w_{t+1} = Σ_k (n_k / n) w_{t+1}^{(k)} where each client k starts "
                "from w_t and trains on D_k^{train} only."
            ),
            why_it_matters=(
                "Centralized pooling hides client heterogeneity and privacy boundaries.",
                "Training on holdout partitions is leakage.",
            ),
            how_buildml_uses=(
                "Session.fit_federated → evaluate_federated / predict_federated.",
                "Client column from role='group' or client_column=.",
            ),
            interpretation_rules=(
                "Read n_clients, round_history, global + per-client holdout metrics, disclosures.",
                "Disclosures state this is not Flower/OpenFL and not secure aggregation.",
            ),
            assumptions=(
                "Exactly one target; a client/group column; numeric non-null features; "
                "estimators expose coef_/intercept_.",
            ),
            failure_modes=(
                "Fewer than two eligible clients; clients with too few train rows.",
            ),
            anti_patterns=(
                "Claiming production FL networking or cryptographic privacy from this simulation.",
            ),
            worked_example_pattern=(
                "fit_federated(method='fedavg', n_rounds=5) → evaluate_federated('validation').",
            ),
            related_concepts=(
                "federated-fedavg",
                "federated-fedprox",
                "federated-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="federated-fedavg",
            title="FedAvg coefficient averaging (weighted-by-n)",
            summary="Clients train from the global linear/SGD weights; the server averages coef_/intercept_ weighted by local n.",
            definition=(
                "method='fedavg' clones global coef_/intercept_ to each selected "
                "client, runs local_epochs of partial_fit or .fit on that client's "
                "train rows, then aggregates with weights proportional to n_k."
            ),
            intuition=(
                "Everyone starts from the same chalkboard, practices on their own "
                "problems, then the class average becomes the new chalkboard."
            ),
            formal_idea=(
                "McMahan et al. FedAvg for models with additive parameters; here "
                "sklearn linear/SGD families."
            ),
            why_it_matters=(
                "A complete, honest FL teaching/research path without pretending a network stack.",
            ),
            how_buildml_uses=(
                "fit_federated(method='fedavg', estimator='sgd_classifier'|...).",
            ),
            interpretation_rules=(
                "Inspect round_history mean_client_train_metric and holdout accuracy/R².",
            ),
            assumptions=("Compatible coef_ shapes across clients (shared feature space).",),
            failure_modes=("Non-linear models without coef_; severe client drift.",),
            anti_patterns=("Calling this secure multi-party computation.",),
            worked_example_pattern=(
                "fit_federated(method='fedavg', estimator='logistic_regression', n_rounds=8).",
            ),
            related_concepts=("federated-simulation", "federated-fedprox"),
        ),
        _note(
            key="federated-fedprox",
            title="FedProx proximal pull toward the global model",
            summary="After each local epoch, pull client coefficients toward the round's global weights by mu.",
            definition=(
                "method='fedprox' requires mu > 0. After each local epoch, "
                "coef ← coef − mu·(coef − global). This is a practical proximal "
                "regularization toward the server model for heterogeneous clients."
            ),
            intuition=(
                "Practice on your own data, but don't wander too far from the "
                "shared starting point."
            ),
            formal_idea=(
                "Approx. Li et al. FedProx: local objective + (μ/2)||w − w_global||²."
            ),
            why_it_matters=("Stabilizes aggregation under client heterogeneity.",),
            how_buildml_uses=("fit_federated(method='fedprox', mu=0.1, ...).",),
            interpretation_rules=(
                "Compare fedprox vs fedavg holdout metrics; read mu in disclosures.",
            ),
            assumptions=("mu > 0; same linear/SGD coefficient path as FedAvg.",),
            failure_modes=("mu too large freezes local learning; mu=0 is refused.",),
            anti_patterns=("Claiming full FedProx paper suite beyond proximal pull.",),
            worked_example_pattern=(
                "fit_federated(method='fedprox', mu=0.05, n_rounds=5).",
            ),
            related_concepts=("federated-simulation", "federated-fedavg"),
        ),
        _note(
            key="federated-bundle-boundary",
            title="Federated bundle boundary",
            summary="buildml.federated_bundle.v1 stores FederatedPlan; Session checkpoints do not embed it.",
            definition=(
                "A federated bundle persists the global estimator, client column "
                "contract, round history, and class vocabulary. Session checkpoints "
                "persist data/roles/splits/history — not FederatedPlan."
            ),
            intuition=(
                "Saving the lab notebook is not the same as saving the averaged "
                "global model from the federation rounds."
            ),
            formal_idea=(
                "Artifacts are complementary: checkpoint_load ↛ federated model; "
                "load_federated_bundle ↛ dataset rows."
            ),
            why_it_matters=("Mixing artifacts causes silent missing-learner failures.",),
            how_buildml_uses=("save_federated_bundle / load_federated_bundle.",),
            interpretation_rules=("Read meta.json format buildml.federated_bundle.v1.",),
            assumptions=("Feature/client/target columns still match at load time.",),
            failure_modes=("Expecting checkpoint_load to restore FederatedPlan.",),
            anti_patterns=(
                "Treating online or meta-learning bundles as federated plans.",
            ),
            worked_example_pattern=(
                "session.save_federated_bundle(path); other.load_federated_bundle(path).",
            ),
            related_concepts=("federated-simulation", "metalearning-bundle-boundary"),
        ),
    )
}
