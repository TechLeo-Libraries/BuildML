# ruff: noqa: E501
"""Unsupervised learning concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

UNSUPERVISED_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="unsupervised-train-fit-holdout-assign",
            title="Unsupervised train-fit / holdout-assign",
            summary="Fit cluster geometry on train only; assign holdout rows with a frozen plan: never refit on evaluation partitions.",
            definition=(
                "Train-fit / holdout-assign is the leakage-safe unsupervised contract: "
                "estimate cluster structure on the training partition, freeze the plan "
                "(centroids, core samples, or native predict), and label validation/test "
                "rows without updating that geometry."
            ),
            intuition=(
                "If you redraw cluster boundaries using the exam answers, the exam no longer "
                "measures generalization. Fit the map on train; stamp holdout points onto that map."
            ),
            formal_idea=(
                "Let f_train be a clusterer fit on X_train. For partition P ∈ {validation, test}, "
                "labels ŷ_P = assign(f_train, X_P) with assign ∈ {predict, nearest_centroid, "
                "nearest_core}. Internal metrics m(X_P, ŷ_P) use those frozen labels."
            ),
            why_it_matters=(
                "Refitting on all rows contaminates holdout geometry metrics.",
                "Teaching and model cards need an explicit fit-partition disclosure.",
            ),
            how_buildml_uses=(
                "Session.fit_clusters requires a SplitPlan and fits on train only.",
                "Session.assign_clusters / evaluate_clusters reuse the frozen ClusterPlan.",
                "Agglomerative and DBSCAN disclose approximate holdout assign strategies.",
            ),
            interpretation_rules=(
                "Read partition name beside every silhouette / CH / DB score.",
                "Train-partition metrics are optimistic for model selection.",
            ),
            assumptions=(
                "A disjoint SplitPlan exists before fit_clusters.",
                "Feature columns are numeric and imputed before distance-based methods.",
            ),
            failure_modes=(
                "Fitting clusters on the full frame before splitting.",
                "Treating holdout assign approximations as identical to native predict.",
            ),
            anti_patterns=(
                "Calling sklearn fit_predict on concatenated train+test for 'evaluation'.",
            ),
            worked_example_pattern=(
                "split → scale → (optional reduce_dimensions) → fit_clusters → "
                "evaluate_clusters(partition='validation').",
            ),
            related_concepts=("leakage-boundary", "evaluation-partitions", "principal-components"),
        ),
        _note(
            key="cluster-validity-not-truth",
            title="Cluster validity is not ground truth",
            summary="Silhouette and related scores measure geometric cohesion/separation: not a verified taxonomy or business truth.",
            definition=(
                "Internal cluster validity indices quantify how compact and separated clusters "
                "are under a distance/geometry assumption. They do not certify that clusters "
                "match an unobserved true partition or a decision-useful segmentation."
            ),
            intuition=(
                "A tidy packing of points can still be the wrong story for the domain. "
                "Validity helps compare geometries; it does not name clusters."
            ),
            formal_idea=(
                "Scores such as silhouette s(X, ŷ), Calinski–Harabasz, and Davies–Bouldin "
                "are functions of features and assigned labels only. External scores (ARI/NMI) "
                "need a reference labeling supplied by the caller and still do not imply causality."
            ),
            why_it_matters=(
                "Over-trusting silhouette leads to brittle operational segments.",
                "External agreement can look strong while missing the decision objective.",
            ),
            how_buildml_uses=(
                "evaluate_clusters reports internal metrics with explicit disclosures.",
                "external_label_column adds ARI/NMI only when the caller provides labels.",
                "Catalog and guides refuse to equate validity with ground truth.",
            ),
            interpretation_rules=(
                "Pair metrics with domain review and stability checks.",
                "Report noise_rate for DBSCAN beside validity scores.",
            ),
            assumptions=(
                "Distance geometry is meaningful after scaling/encoding choices.",
                "Reference labels, when used, are honestly scoped.",
            ),
            failure_modes=(
                "Publishing silhouette as 'accuracy'.",
                "Ignoring that different k can trade validity against usefulness.",
            ),
            anti_patterns=(
                "Selecting k solely to maximize train silhouette without holdout assign.",
            ),
            worked_example_pattern=(
                "fit_clusters → evaluate_clusters on validation → inspect cluster sizes and disclosures.",
            ),
            related_concepts=("unsupervised-train-fit-holdout-assign", "evaluation-partitions"),
        ),
        _note(
            key="pca-cluster-integration",
            title="PCA and clustering integration",
            summary="Dimensionality reduction stays on Session.reduce_dimensions; clustering optionally consumes those train-fitted components.",
            definition=(
                "PCA-cluster integration means fitting a ReducePlan on train (Session.reduce_dimensions), "
                "then optionally fitting clusters on the resulting component columns without refitting PCA "
                "inside the clusterer."
            ),
            intuition=(
                "First rotate the axes on train; then group points in that rotated space. "
                "Do not invent a second private PCA path that disagrees with preprocess."
            ),
            formal_idea=(
                "With ReducePlan R fit on X_train, features for clustering are R(X). "
                "ClusterPlan C is fit on R(X_train); assign uses C(R(X_P)) with R frozen."
            ),
            why_it_matters=(
                "Forking PCA implementations creates leakage and teaching drift.",
                "Explained variance remains unsupervised and is not cluster quality.",
            ),
            how_buildml_uses=(
                "fit_clusters(prefer_reduce_components=True) prefers ReducePlan component columns when present.",
                "used_reduce_components is recorded on ClusterFitResult / ClusterPlan.",
                "EDA PCA screens remain descriptive and separate from Session plans.",
            ),
            interpretation_rules=(
                "Interpret loadings before naming clusters as original features.",
                "Scale before PCA when magnitudes differ.",
            ),
            assumptions=(
                "Component columns from reduce_dimensions remain on the Session frame.",
                "Imputation happened before reduce and cluster.",
            ),
            failure_modes=(
                "Dropping component columns then expecting prefer_reduce_components to find them.",
                "Treating cumulative explained variance as a clustering objective.",
            ),
            anti_patterns=(
                "Re-fitting PCA on all rows inside a custom clustering script while using Session splits.",
            ),
            worked_example_pattern=(
                "scale → reduce_dimensions → fit_clusters → evaluate_clusters.",
            ),
            related_concepts=("unsupervised-train-fit-holdout-assign", "principal-components", "leakage-boundary"),
        ),
        _note(
            key="cluster-kmeans-vs-density",
            title="K-means / GMM vs density clustering (DBSCAN, HDBSCAN)",
            summary="Centroid methods assume spherical groups; density methods find arbitrary shapes and label noise.",
            definition=(
                "K-means and GMM fit prototypes (centroids or mixture components) and assign "
                "by distance/likelihood. DBSCAN/HDBSCAN/OPTICS grow clusters from local density "
                "and may mark points as noise (-1)."
            ),
            intuition=(
                "K-means draws H circles and asks everyone to join the nearest center. "
                "DBSCAN walks neighborhoods: sparse points stay unclustered instead of being "
                "forced into a group."
            ),
            formal_idea=(
                "K-means minimizes within-cluster sum of squares; GMM maximizes mixture "
                "likelihood with BIC/AIC for k. DBSCAN connects ε-neighbors with min_samples; "
                "HDBSCAN extracts stable density modes."
            ),
            why_it_matters=(
                "Wrong family yields nonsense segments and unstable assign on holdout.",
                "Catalog lists both: pick by geometry, not by default."
            ),
            how_buildml_uses=(
                "fit_clusters(method='kmeans'|'gmm'|'dbscan'|'hdbscan'|...).",
                "assign_strategy and noise_rate disclosed for density methods.",
                "evaluate_clusters reports internal validity, not ground-truth accuracy.",
            ),
            interpretation_rules=(
                "Read cluster_sizes and noise_rate for DBSCAN/HDBSCAN.",
                "Prefer scale before k-means when feature scales differ.",
            ),
            assumptions=(
                "Metric geometry is meaningful after preprocessing.",
                "k or density hyperparameters are chosen on train/disclosed holdout assign.",
            ),
            failure_modes=(
                "Using k-means on elongated or nested manifolds.",
                "Treating HDBSCAN noise as a failure instead of a feature.",
            ),
            anti_patterns=(
                "Picking k solely to maximize train silhouette without holdout assign.",
            ),
            worked_example_pattern=(
                "scale → fit_clusters(method='hdbscan') → assign_clusters('validation').",
            ),
            related_concepts=("unsupervised-cluster-validity", "unsupervised-train-fit-holdout-assign"),
        ),
        _note(
            key="unsupervised-bundle-boundary",
            title="Unsupervised bundle boundary",
            summary="Cluster plans persist as buildml.unsupervised_bundle.v2: complementary to Session checkpoints and Torch/RAG bundles.",
            definition=(
                "The unsupervised bundle boundary is the contract that a train-fitted ClusterPlan "
                "(estimator, feature columns, assign strategy, disclosures) is stored under "
                "buildml.unsupervised_bundle.v2 (v1 loadable), separate from Session workflow checkpoints and "
                "from Torch/RAG/classical pipeline artifacts."
            ),
            intuition=(
                "Saving your notebook (checkpoint) does not shelf the clustering map; "
                "saving the clustering map does not restore the dataset."
            ),
            formal_idea=(
                "Bundle layout: meta.json + cluster_plan.joblib. Session checkpoints may carry "
                "classical preprocess plans (including ReducePlan) but do not embed ClusterPlan."
            ),
            why_it_matters=(
                "Mixing artifact kinds causes failed loads and false resume expectations.",
                "Assign strategy disclosures must travel with agglomerative/DBSCAN plans.",
            ),
            how_buildml_uses=(
                "save_unsupervised_bundle / load_unsupervised_bundle on Session.",
                "CHECKPOINT_BOUNDARY string documents complementarity.",
            ),
            interpretation_rules=(
                "After load_unsupervised_bundle, rebuild or reload the tabular Session separately if needed.",
                "Confirm feature columns still exist before assign_clusters.",
            ),
            assumptions=(
                "joblib can serialize the sklearn estimator in the plan.",
                "Feature schema at assign time matches the plan columns.",
            ),
            failure_modes=(
                "Expecting checkpoint_load to restore fit_clusters state.",
                "Loading a Torch/RAG bundle as unsupervised.",
            ),
            anti_patterns=(
                "Hand-copying only centroids without disclosures/assign strategy.",
            ),
            worked_example_pattern=(
                "fit_clusters → save_unsupervised_bundle → Session().load_unsupervised_bundle → assign_clusters.",
            ),
            related_concepts=("unsupervised-train-fit-holdout-assign", "rag-chunk-index-boundary"),
        ),
    )
}
