# ruff: noqa: E501
"""Beginner layers for federated learning simulation."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

FEDERATED_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "federated-simulation",
        plain=(
            "Federated learning trains one shared model across several parties without pooling their data. "
            "Each party trains locally and sends only the resulting model weights to a coordinator, which "
            "averages them. BuildML *simulates* this on one table: a column says which client each row "
            "belongs to, and everything runs in your own process."
        ),
        analogy=(
            "Several hospitals each analyse their own patient records and mail only their findings to a "
            "central statistician. No patient record ever leaves its hospital."
        ),
        steps=(
            "Mark the column identifying which client each row came from — a `group` role or `client_column`.",
            "Each round, every client starts from the current shared weights.",
            "Each client trains on its own training rows only.",
            "The coordinator averages the resulting weights, weighted by how many rows each client had.",
            "Repeat for `n_rounds`, then evaluate globally and per client on held-out rows.",
        ),
        use=(
            "To study whether a federated approach would work before building real infrastructure.",
            "To measure how much client heterogeneity hurts, and whether the averaged model serves every client fairly.",
        ),
        avoid=(
            "Do not present this as a deployed federated system; the data is all in one process and one table.",
            "Do not use it as a privacy mechanism — no cryptography or differential privacy is involved.",
        ),
        myths=(
            (
                "Running this means my data is private.",
                "It means the *simulation* respects the client boundary. Your rows are all sitting in one dataframe on one machine. The privacy properties of real federated learning come from the deployment, not the algorithm.",
            ),
            (
                "The federated model should match a model trained on pooled data.",
                "It usually will not, and the gap is the point of measuring. Averaging weights across heterogeneous clients loses something.",
            ),
        ),
        example=(
            "session.set_roles({'hospital_id': 'group', 'readmitted': 'target'})",
            "session.fit_federated(method='fedavg', n_rounds=5, random_state=0)",
            "report = session.evaluate_federated(partition='validation')",
            "print(report.global_metrics, report.per_client_metrics)",
        ),
        check=(
            "How many clients do you have, and do any of them have very few rows?",
            "Does the global model perform acceptably for your *worst* client, not just on average?",
        ),
        tools=("fit_federated", "evaluate_federated", "predict_federated", "set_roles"),
        terms=("federated learning", "client", "aggregation", "leakage"),
        difficulty=ADVANCED,
    ),
    _layer(
        "federated-flower-backend",
        plain=(
            "Flower is a real federated-learning framework. With `buildml[federated-industry]` installed, "
            "`backend='flower'` wraps each client partition as a Flower client and lets Flower do the "
            "aggregation. It still runs locally in your process unless you deploy a Flower runtime yourself."
        ),
        analogy=(
            "Doing the same classroom exercise, but using the official scoring software instead of adding "
            "the numbers by hand. Same exercise, more standard tooling."
        ),
        steps=(
            "Install the industry extra so `flwr` is available.",
            "Pass `backend='flower'` to `fit_federated`.",
            "Each client partition becomes a Flower NumPyClient.",
            "Local fitting still goes through the same scikit-learn linear or SGD path.",
            "Flower's own weighted-averaging strategy combines the client updates.",
        ),
        use=(
            "When you plan to deploy on Flower later and want your simulation to use the same aggregation code.",
            "When you want to cross-check BuildML's native averaging against a standard implementation.",
        ),
        avoid=(
            "Do not assume the Flower backend implies network transport, gRPC, or secure aggregation — it does not.",
            "Do not request it without the extra installed; you will get a clear error naming what to install.",
        ),
        myths=(
            (
                "Using Flower makes it a real federated deployment.",
                "It makes the aggregation step use Flower's library. Everything still executes in-process on one dataframe. The disclosures say so explicitly.",
            ),
            (
                "The Flower backend will give different, better results.",
                "It performs the same weighted average over the same local fits. Expect the same numbers, not better ones.",
            ),
        ),
        example=(
            "# pip install \"buildml[federated-industry]\"",
            "session.fit_federated(backend='flower', method='fedavg', n_rounds=5)",
            "print(session.federated_plan.backend, session.federated_plan.disclosures)",
        ),
        check=(
            "Do you actually need Flower, or is the native path enough for what you are measuring?",
            "Have you read the disclosures about what this simulation does and does not provide?",
        ),
        tools=("fit_federated", "evaluate_federated", "export_round_history"),
        terms=("federated learning", "backend", "extra", "aggregation"),
        difficulty=ADVANCED,
    ),
    _layer(
        "federated-fedavg",
        plain=(
            "FedAvg is the basic recipe. Every client starts a round from the same shared weights, trains "
            "on its own rows for a few passes, and sends the updated weights back. The coordinator averages "
            "them, giving more weight to clients with more rows."
        ),
        analogy=(
            "Everyone copies the same starting draft, edits it based on their own experience, and the "
            "coordinator merges the edits — with a bigger say for people who reviewed more material."
        ),
        steps=(
            "Copy the current global coefficients to each selected client.",
            "Each client runs `local_epochs` of training on its own training rows.",
            "Each returns its updated coefficients and its row count.",
            "The coordinator computes the row-count-weighted average.",
            "That becomes the global model for the next round.",
        ),
        use=(
            "As the default and baseline federated method.",
            "When clients have broadly similar data distributions.",
        ),
        avoid=(
            "Do not use it with models that have no coefficients to average — tree ensembles cannot be combined this way.",
            "Do not use plain FedAvg when clients differ sharply; local models drift apart and the average satisfies nobody.",
        ),
        myths=(
            (
                "Averaging weights is secure multi-party computation.",
                "It is arithmetic on plain numbers. Weight updates can leak information about training data, which is precisely why real deployments add cryptographic protections.",
            ),
            (
                "More rounds always converge to a better model.",
                "With heterogeneous clients, more rounds can oscillate. Watch the round history rather than assuming monotone improvement.",
            ),
        ),
        example=(
            "session.fit_federated(",
            "    method='fedavg', estimator='logistic_regression',",
            "    n_rounds=8, local_epochs=2, random_state=0,",
            ")",
            "for row in session.federated_plan.round_history: print(row)",
        ),
        check=(
            "Does your chosen estimator actually expose coefficients?",
            "Does the round history improve steadily or bounce around?",
        ),
        tools=("fit_federated", "evaluate_federated", "predict_federated"),
        terms=("federated learning", "aggregation", "client", "gradient descent"),
        difficulty=ADVANCED,
    ),
    _layer(
        "federated-fedprox",
        plain=(
            "FedProx is FedAvg with a leash. After each local training pass, each client's weights are "
            "pulled part of the way back toward the shared global weights. The strength of the pull is `mu`. "
            "It stops clients with unusual data from wandering too far."
        ),
        analogy=(
            "Letting everyone edit the draft, but with a rule that no edit may stray more than so far from "
            "the agreed version. Fewer brilliant rewrites, far fewer incompatible ones."
        ),
        steps=(
            "Run local training exactly as in FedAvg.",
            "After each local epoch, move the client's coefficients a fraction `mu` back toward the round's global weights.",
            "Aggregate as usual.",
            "Small `mu` means a gentle pull; large `mu` means clients barely move at all.",
            "`mu` must be greater than zero — at zero this is just FedAvg, and BuildML refuses the ambiguity.",
        ),
        use=(
            "When your clients have visibly different data and FedAvg is unstable across rounds.",
            "When a few large or unusual clients dominate the average.",
        ),
        avoid=(
            "Do not set `mu` high enough to freeze local learning; you will converge quickly to a model that learned nothing local.",
            "Do not use it before trying FedAvg — you need the baseline to know whether the pull helps.",
        ),
        myths=(
            (
                "FedProx is strictly better than FedAvg.",
                "It trades local adaptation for stability. When clients are similar, that trade costs you accuracy for no benefit.",
            ),
            (
                "This is the complete FedProx method from the paper.",
                "BuildML implements the practical proximal pull. The disclosures say so; do not cite it as the full published algorithm.",
            ),
        ),
        example=(
            "session.fit_federated(method='fedprox', mu=0.05, n_rounds=5, random_state=0)",
            "# compare against the fedavg run on the same holdout",
        ),
        check=(
            "How does holdout performance compare with your FedAvg baseline?",
            "Is `mu` small enough that clients still learn something local?",
        ),
        tools=("fit_federated", "evaluate_federated"),
        terms=("federated learning", "regularization", "client", "aggregation"),
        difficulty=ADVANCED,
    ),
    _layer(
        "federated-bundle-boundary",
        plain=(
            "The federated plan — the averaged global model, the client column contract, the round history, "
            "and the class vocabulary — saves as its own bundle. Session checkpoints do not include it."
        ),
        analogy=(
            "The agreed final draft is archived separately from each contributor's working notes."
        ),
        steps=(
            "Run federated training so a plan exists.",
            "Call `save_federated_bundle(path)`.",
            "Reload with `load_federated_bundle(path)`.",
            "Predict with the global model on new rows.",
            "Keep checkpoints separate for data and workflow state.",
        ),
        use=(
            "When the averaged global model is deployed somewhere other than the simulation environment.",
            "When the round history must be preserved as evidence of how the model was produced.",
        ),
        avoid=(
            "Do not expect `checkpoint_load` to restore the federated plan.",
            "Do not confuse this with online or meta-learning bundles, which look superficially similar and are not interchangeable.",
        ),
        myths=(
            (
                "The bundle contains each client's local model.",
                "It contains the aggregated global model plus the round history. Individual client models are intermediate state, not the deliverable.",
            ),
            (
                "Round history is just logging.",
                "It is the record of how the global model came to be, including per-round client metrics. For anything audited, that record is part of the artifact.",
            ),
        ),
        example=(
            "session.save_federated_bundle('artifacts/consortium-model')",
            "svc = Session.ingest(new_rows).load_federated_bundle('artifacts/consortium-model')",
            "svc.predict_federated()",
        ),
        check=(
            "Does the serving data carry the same feature and client columns?",
            "Have you kept the round history alongside the model for review?",
        ),
        tools=("save_federated_bundle", "load_federated_bundle", "predict_federated", "checkpoint_save"),
        terms=("bundle", "checkpoint", "federated learning"),
        difficulty=CORE,
    ),
)

__all__ = ["FEDERATED_BEGINNER"]
