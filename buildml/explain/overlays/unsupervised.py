# ruff: noqa: E501, F401
"""Unsupervised Session operation overlays (human teaching prose)."""

from __future__ import annotations

from buildml.explain.overlays._common import (
    DATASET,
    SPLIT,
    OperationKind,
    _operation,
    _p,
)
from buildml.explain.schemas import OperationSpec, Prerequisite

CLUSTER_PLAN = Prerequisite(
    "cluster-plan",
    "A train-fitted ClusterPlan is attached to the Session.",
    check_hint="Session.cluster_plan is not None.",
)

_OPERATIONS: tuple[OperationSpec, ...] = (
    _operation(
        "fit_clusters",
        OperationKind.MODEL,
        "Fit a clusterer on the train partition and store a ClusterPlan.",
        "Learn unsupervised structure without using evaluation-partition geometry.",
        "Unsupervised clustering fit step.",
        (
            "Require a SplitPlan and refuse fit without train.",
            "Resolve numeric columns; prefer ReducePlan component columns when present.",
            "Fit kmeans, agglomerative, dbscan, gmm, spectral, optics, mean_shift on train; "
            "hdbscan when buildml[unsupervised] installed; dec/idec when buildml[torch] installed.",
            "Record assign strategy disclosures for non-native predictors.",
        ),
        parameters=(
            _p("method", "kmeans | agglomerative | dbscan | gmm | hdbscan | spectral | optics | mean_shift | dec | idec", "Clustering algorithm.", "kmeans"),
            _p("n_clusters", "int | None", "Requested k for kmeans/agglomerative.", 8),
            _p("columns", "list[str] | None", "Optional explicit numeric columns."),
            _p("random_state", "int | None", "RNG seed for kmeans.", 0),
            _p("n_init", "int | str", "KMeans n_init.", "auto"),
            _p("max_iter", "int", "KMeans max iterations.", 300),
            _p("linkage", "str", "Agglomerative linkage.", "ward"),
            _p("eps", "float", "DBSCAN neighborhood radius.", 0.5),
            _p("min_samples", "int", "DBSCAN min samples.", 5),
            _p(
                "prefer_reduce_components",
                "bool",
                "Prefer Session.reduce_dimensions component columns when available.",
                True,
            ),
            _p("label_column", "str", "Column name used when attach=True later.", "cluster_id"),
            _p("auto_k", "bool", "Elbow (k-means) or BIC range (GMM) on train.", False),
        ),
        inputs=("Split Session with numeric features (typically scaled; optionally PCA components).",),
        outputs=("ClusterFitResult; ClusterPlan stored on the Session.",),
        prerequisites=(DATASET, SPLIT),
        ordering=(
            "After split and usually after impute/scale; optional reduce_dimensions before clustering.",
        ),
        alternatives=(
            "Use Session.reduce_dimensions alone when you only need PCA components.",
            "EDA multivariate/IsolationForest screens remain descriptive: not this fit API.",
        ),
        rationale=(
            "Use when the goal is structure discovery or segmentation with leakage-safe holdout assign.",
        ),
        assumptions=(
            "Features are numeric and non-null; distance methods assume sensible scaling.",
            "n_clusters is meaningful for kmeans/agglomerative; DBSCAN is density-driven.",
        ),
        failures=(
            "Missing split, nulls, non-numeric columns, n_clusters larger than n_train, empty DBSCAN cores.",
        ),
        leakage=(
            "Fitting on all rows before splitting contaminates holdout geometry metrics.",
            "Using target/id columns as features mixes supervised labels into unsupervised structure.",
        ),
        anti_patterns=(
            "Treating cluster labels as supervised targets without a separate labeled study.",
            "Confusing EDA IsolationForest anomaly screens with a fitted ClusterPlan.",
            "Refitting PCA inside a custom script instead of Session.reduce_dimensions.",
        ),
        state_changes=(
            "Stores cluster_plan and cluster_fit_result; clears prior assign/eval slots.",
        ),
        result_reading=(
            "Read method, n_clusters, cluster_sizes, assign_strategy, used_reduce_components, disclosures.",
        ),
        next_steps=("assign_clusters and/or evaluate_clusters; optionally save_unsupervised_bundle.",),
        concepts=(
            "unsupervised-train-fit-holdout-assign",
            "pca-cluster-integration",
            "cluster-validity-not-truth",
            "leakage-boundary",
        ),
    ),
    _operation(
        "assign_clusters",
        OperationKind.TRANSFORM,
        "Assign cluster labels with a train-fitted ClusterPlan (no refit).",
        "Label a partition or the full frame using frozen train geometry.",
        "Unsupervised assign step.",
        (
            "Require an attached ClusterPlan.",
            "Apply native predict (kmeans), nearest-centroid (agglomerative), or nearest-core (dbscan).",
            "Optionally attach label_column when partition='all'.",
        ),
        parameters=(
            _p("partition", "train | validation | test | all", "Rows to label.", "test"),
            _p(
                "attach",
                "bool",
                "Write label_column onto the Session frame (requires partition='all').",
                False,
            ),
        ),
        inputs=("Active ClusterPlan and matching feature columns.",),
        outputs=("ClusterAssignResult; optional mutated dataset when attach=True.",),
        prerequisites=(DATASET, CLUSTER_PLAN),
        ordering=("After fit_clusters or load_unsupervised_bundle.",),
        alternatives=("evaluate_clusters when you need metrics rather than raw labels.",),
        rationale=("Use to materialize segment ids for inspection or downstream joins.",),
        assumptions=("Plan feature columns still exist; assign strategy disclosures are accepted.",),
        failures=("No plan, missing columns, attach=True with partition≠all, existing label_column.",),
        leakage=("Refitting on the assign partition would break the train-fit contract.",),
        anti_patterns=(
            "Attaching labels before reviewing holdout metrics.",
            "Ignoring noise labels (-1) from DBSCAN.",
        ),
        state_changes=("Stores cluster_assign_result; may add a feature-role label column.",),
        result_reading=("Read n_rows, n_noise, assign_strategy, and disclosures.",),
        next_steps=("evaluate_clusters; or use attached labels cautiously downstream.",),
        concepts=(
            "unsupervised-train-fit-holdout-assign",
            "cluster-validity-not-truth",
            "leakage-boundary",
        ),
    ),
    _operation(
        "evaluate_clusters",
        OperationKind.DIAGNOSTIC,
        "Score train-fitted clusters on a partition with internal (and optional external) metrics.",
        "Quantify geometric validity without claiming ground-truth taxonomy.",
        "Unsupervised evaluation step.",
        (
            "Assign labels on the requested partition with the frozen plan.",
            "Compute silhouette / Calinski–Harabasz / Davies–Bouldin when feasible.",
            "Optionally compute ARI/NMI against an external_label_column with disclosure.",
        ),
        parameters=(
            _p(
                "partition",
                "train | validation | test | all",
                "Evaluation partition (validation falls back to test if absent).",
                "validation",
            ),
            _p(
                "external_label_column",
                "str | None",
                "Optional reference labels for ARI/NMI (not used in fit).",
            ),
            _p("sample_size", "int | None", "Optional silhouette subsample cap.", 2000),
            _p("random_state", "int | None", "Subsample seed.", 0),
            _p("compute_stability", "bool", "Bootstrap stability diagnostics on train.", False),
            _p("compute_elbow", "bool", "Elbow inertia curve on train (diagnostic refits).", False),
        ),
        inputs=("Active ClusterPlan and partition features.",),
        outputs=("ClusterEvalResult stored on the Session.",),
        prerequisites=(DATASET, CLUSTER_PLAN),
        ordering=("After fit_clusters; prefer validation/test over train for claims.",),
        alternatives=("assign_clusters for labels without metrics.",),
        rationale=("Use to compare k/methods under a leakage-safe assigner.",),
        assumptions=("Enough non-noise clusters/rows for internal metrics when reporting them.",),
        failures=("No plan, missing external column, null external labels, empty validation without fallback.",),
        leakage=(
            "Evaluating on train only for selection overstates separation.",
            "Supplying leaked external labels that were derived from the same fit.",
        ),
        anti_patterns=(
            "Publishing silhouette as accuracy.",
            "Equating ARI with causal recovery of a true segmentation.",
        ),
        state_changes=("Stores cluster_eval_result.",),
        result_reading=(
            "Read metrics, external_metrics, n_clusters_observed, disclosures, recommendations.",
        ),
        next_steps=("save_unsupervised_bundle; or revise features/k and refit on train.",),
        concepts=(
            "cluster-validity-not-truth",
            "unsupervised-train-fit-holdout-assign",
            "evaluation-partitions",
        ),
    ),
    _operation(
        "save_unsupervised_bundle",
        OperationKind.PERSIST,
        "Persist the active ClusterPlan as buildml.unsupervised_bundle.v2.",
        "Save a deployable/reloadable clustering map distinct from Session checkpoints.",
        "Unsupervised artifact save.",
        (
            "Require an attached ClusterPlan.",
            "Write meta.json and cluster_plan.joblib with compatibility disclosures.",
        ),
        parameters=(_p("path", "str | Path", "Destination directory.", required=True),),
        inputs=("Active ClusterPlan (optional fit/eval summaries in meta).",),
        outputs=("Bundle directory path.",),
        prerequisites=(CLUSTER_PLAN,),
        ordering=("After fit_clusters (and usually after evaluate_clusters).",),
        alternatives=("Keep working inside one Session without persistence.",),
        rationale=("Use when assign must resume without replaying the full tabular Session.",),
        assumptions=("joblib can serialize the sklearn estimator.",),
        failures=("No plan; incomplete destination permissions.",),
        leakage=("Bundles do not include the dataset; they are not a substitute for careful split hygiene.",),
        anti_patterns=(
            "Expecting checkpoint_load to restore ClusterPlan.",
            "Treating unsupervised bundles as interchangeable with Torch/RAG bundles.",
        ),
        state_changes=("Records save path in history; does not clear the plan.",),
        result_reading=("Confirm meta.json format == buildml.unsupervised_bundle.v2 (v1 loadable).",),
        next_steps=("load_unsupervised_bundle on a fresh Session when needed.",),
        concepts=("unsupervised-bundle-boundary", "unsupervised-train-fit-holdout-assign"),
    ),
    _operation(
        "load_unsupervised_bundle",
        OperationKind.PERSIST,
        "Load a buildml.unsupervised_bundle.v2 (or v1) ClusterPlan into the Session.",
        "Restore a frozen clustering map for assign/evaluate.",
        "Unsupervised artifact load.",
        (
            "Validate format string.",
            "Deserialize ClusterPlan and attach to Session slots.",
        ),
        parameters=(
            _p("path", "str | Path", "Bundle directory.", required=True),
            _p(
                "trusted",
                "bool",
                "Must be True to deserialize pickle/joblib/torch payloads (default False).",
                False,
            ),
        ),
        inputs=("Unsupervised bundle directory.",),
        outputs=("Session with cluster_plan attached.",),
        prerequisites=(),
        ordering=("Before assign_clusters / evaluate_clusters on a restored plan.",),
        alternatives=("Refit with fit_clusters when the feature space changed.",),
        rationale=("Use to resume assign without refitting geometry.",),
        assumptions=("Feature columns at assign time still match the plan.",),
        failures=("Missing files, wrong format, corrupt joblib payload.",),
        leakage=("Loading does not recreate SplitPlan; attach data/splits separately for holdout claims.",),
        anti_patterns=("Loading a RAG/Torch bundle path into load_unsupervised_bundle.",),
        state_changes=("Stores cluster_plan; clears prior fit/assign/eval result slots.",),
        result_reading=("Check method, columns, assign_strategy from plan.to_dict().",),
        next_steps=("assign_clusters or evaluate_clusters once features are available.",),
        concepts=("unsupervised-bundle-boundary", "unsupervised-train-fit-holdout-assign"),
    ),
)
