# ruff: noqa: E501
"""Beginner layers for clustering and dimensionality reduction."""

from __future__ import annotations

from buildml.explain.beginner._builder import CORE, FOUNDATION, BeginnerLayer, _index, _layer

UNSUPERVISED_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "unsupervised-train-fit-holdout-assign",
        plain=(
            "Clustering has no answer key, but it still has a train/holdout discipline. You work out where "
            "the groups are using training rows only, freeze that map, and then place held-out rows onto "
            "the frozen map without moving any boundaries. That way a quality score on held-out rows means "
            "something."
        ),
        analogy=(
            "You draw a map of a city from the streets you have walked. Later, a visitor tells you where "
            "they are and you locate them on your existing map. You do not redraw the city around them."
        ),
        steps=(
            "Split your rows first, exactly as you would for supervised learning.",
            "Scale the numeric columns, because clustering is built on distances and unscaled units decide the answer for you.",
            "Fit the clusterer on training rows: this learns the centres, the shapes, or the core points.",
            "Assign validation or test rows using the frozen plan: nearest centre, native predict, or nearest core point.",
            "Score cluster quality on the held-out assignment, and always print which partition the score came from.",
        ),
        use=(
            "Whenever you want to claim your segmentation generalizes rather than just describes the rows you had.",
            "Before shipping segments into an operational process, where new customers will arrive and need placing.",
        ),
        avoid=(
            "Do not use this ceremony for a one-off descriptive exploration where you never intend to place new rows.",
            "Do not treat the approximate holdout assignment for hierarchical clustering or DBSCAN as identical to a native predict: BuildML discloses the difference for a reason.",
        ),
        myths=(
            (
                "There is no target, so there is no leakage risk.",
                "Fitting cluster geometry on all rows and then scoring that geometry on 'held-out' rows is leakage. The score measures a map that already saw those points.",
            ),
            (
                "Clustering all your data at once is the natural thing to do.",
                "It is fine for description. It is not fine for any claim about how the segmentation behaves on rows it has not seen.",
            ),
        ),
        example=(
            "session.split(test_size=0.2, random_state=0)",
            "session.scale(strategy='standard')",
            "session.fit_clusters(method='kmeans', n_clusters=4, random_state=0)",
            "session.evaluate_clusters(partition='validation')",
        ),
        check=(
            "Which partition did your silhouette score come from?",
            "How would a brand-new customer get assigned to a segment tomorrow?",
        ),
        tools=("fit_clusters", "assign_clusters", "evaluate_clusters", "scale"),
        terms=("clustering", "centroid", "leakage", "split", "silhouette"),
        difficulty=FOUNDATION,
    ),
    _layer(
        "cluster-validity-not-truth",
        plain=(
            "Cluster quality scores such as silhouette measure geometry: are the groups tight, and are they "
            "far apart? A high score means the points pack neatly. It does not mean the groups are real, "
            "meaningful, or useful for your business."
        ),
        analogy=(
            "Sorting a wardrobe by colour produces beautifully tidy piles. Whether colour is the right way "
            "to organize your clothes is a completely different question, and no tidiness score can answer it."
        ),
        steps=(
            "Fit clusters and compute the internal scores: silhouette, Calinski-Harabasz, Davies-Bouldin.",
            "Read them as a way of comparing candidate settings, such as different numbers of clusters.",
            "Look at cluster sizes; a 'great' score built from one giant cluster and three tiny ones is a warning.",
            "Inspect the actual rows in each cluster and try to describe them in plain business language.",
            "If you have a reference labelling, add ARI or NMI: but treat that as agreement, not as proof of correctness.",
        ),
        use=(
            "To compare two clusterings of the same data under the same features and scaling.",
            "As a sanity check that your groups are not simply arbitrary slices of one blob.",
        ),
        avoid=(
            "Do not report silhouette to stakeholders as an accuracy figure. It is not on the same scale and does not mean the same thing.",
            "Do not choose the number of clusters on score alone when a slightly worse score gives you segments the business can actually act on.",
        ),
        myths=(
            (
                "A silhouette of 0.7 means the clusters are correct.",
                "It means they are geometrically well separated under your distance metric. Change the scaling and the same data gives a different number.",
            ),
            (
                "The best k is the one with the highest score.",
                "Score usually favours few, large, round clusters. Usefulness often lives at a different k entirely.",
            ),
        ),
        example=(
            "report = session.evaluate_clusters(partition='validation')",
            "print(report.silhouette, report.cluster_sizes)",
            "print(report.disclosures)   # what the score cannot tell you",
        ),
        check=(
            "Can you describe each of your clusters in one sentence a colleague would recognize?",
            "How much does your score change if you scale the features differently?",
        ),
        tools=("evaluate_clusters", "fit_clusters", "eda"),
        terms=("silhouette", "clustering", "metric", "unsupervised"),
        difficulty=CORE,
    ),
    _layer(
        "cluster-kmeans-vs-density",
        plain=(
            "K-means and GMM look for round, compact groups around centers. DBSCAN and HDBSCAN grow "
            "clusters from local density and can leave sparse points unlabeled as noise. Pick the family "
            "that matches your geometry, not whichever method is the default."
        ),
        analogy=(
            "K-means is assigning everyone to the nearest city on a map. DBSCAN is finding crowded "
            "neighborhoods and refusing to claim someone lives downtown when they are miles from anyone."
        ),
        steps=(
            "Scale numeric features when magnitudes differ.",
            "Try k-means/GMM when groups are roughly spherical and you can pick k.",
            "Try DBSCAN/HDBSCAN when shapes are irregular or you expect outliers/noise.",
            "Read cluster_sizes and noise_rate on density methods.",
            "Assign on validation: never pick k only from train silhouette alone.",
        ),
        use=(
            "Segmentation with clear blob-shaped groups (k-means/GMM).",
            "Fraud or anomaly-shaped sparse clusters in dense background (density methods).",
        ),
        avoid=(
            "Do not run k-means on elongated manifolds without reduction first.",
            "Do not treat DBSCAN noise labels as errors: they are often the point.",
        ),
        myths=(
            (
                "Higher k always means better clusters.",
                "It means smaller, tighter groups: not necessarily useful ones.",
            ),
            (
                "HDBSCAN always beats k-means.",
                "It wins on messy geometry; k-means is simpler and faster on round blobs.",
            ),
        ),
        example=(
            "session.fit_clusters(method='hdbscan', min_cluster_size=25)",
            "session.assign_clusters(partition='validation')",
            "session.evaluate_clusters(partition='validation')",
        ),
        check=(
            "Does your chosen method match how the points actually look in 2D/3D plots?",
            "For density clustering: what fraction of rows are noise?",
        ),
        tools=("fit_clusters", "assign_clusters", "evaluate_clusters", "reduce_dimensions"),
        terms=("clustering", "unsupervised", "metric"),
        difficulty=CORE,
    ),
    _layer(
        "pca-cluster-integration",
        plain=(
            "You can compress your numeric columns with PCA first and cluster in that compressed space. "
            "In BuildML those are two separate, explicit steps: `reduce_dimensions` fits the compression on "
            "training rows, and `fit_clusters` can then consume the resulting component columns."
        ),
        analogy=(
            "Flattening a 3D sculpture into a photograph, then grouping photographs. You take the photo "
            "once, from a fixed angle, and every later comparison uses that same angle."
        ),
        steps=(
            "Split, then scale: PCA follows variance and unscaled columns hijack it.",
            "Call `reduce_dimensions` to fit the components on training rows; the component columns join the frame.",
            "Call `fit_clusters(prefer_reduce_components=True)` so clustering runs on those components.",
            "BuildML records `used_reduce_components` on the plan so the choice is visible later.",
            "Assign and evaluate as usual: both PCA and the clusterer stay frozen.",
        ),
        use=(
            "When you have many correlated numeric columns and distances in the raw space are dominated by redundancy.",
            "When you want a two-component view you can actually plot alongside the clusters.",
        ),
        avoid=(
            "Do not reduce when you only have a handful of well-understood features; you lose interpretability for nothing.",
            "Do not fit a second, private PCA inside your own script while using Session splits: that is how the two paths silently disagree.",
        ),
        myths=(
            (
                "PCA improves clustering quality.",
                "It changes the distance geometry. Sometimes that helps, sometimes it discards exactly the direction that separated your groups.",
            ),
            (
                "Explained variance measures how good the clusters are.",
                "Explained variance is about the compression, computed before any clustering happens. The two numbers are unrelated.",
            ),
        ),
        example=(
            "session.scale(strategy='standard')",
            "session.reduce_dimensions(n_components=0.9)",
            "session.fit_clusters(method='kmeans', n_clusters=4, prefer_reduce_components=True)",
            "session.evaluate_clusters(partition='validation')",
        ),
        check=(
            "Are your clusters running on component columns or raw features: and does the plan say so?",
            "What do the loadings of your first two components actually represent?",
        ),
        tools=("reduce_dimensions", "fit_clusters", "scale", "evaluate_clusters"),
        terms=("PCA", "dimensionality reduction", "clustering", "plan"),
        difficulty=CORE,
    ),
    _layer(
        "unsupervised-bundle-boundary",
        plain=(
            "Your fitted clustering is saved as its own artifact: an unsupervised bundle: separate from a "
            "Session checkpoint. The checkpoint holds your data workflow; the bundle holds the cluster map. "
            "Loading one does not give you the other."
        ),
        analogy=(
            "Saving your notes and saving the map you drew are two different files. Reopening your notes "
            "does not put the map back on the wall."
        ),
        steps=(
            "Fit clusters so a plan exists on the Session.",
            "Call `save_unsupervised_bundle(path)`: it writes the estimator, the feature columns, the assign strategy, and the disclosures.",
            "Later, create a Session and call `load_unsupervised_bundle(path)`.",
            "Confirm the feature columns the plan expects still exist on your frame.",
            "Call `assign_clusters` to place new rows on the restored map.",
        ),
        use=(
            "When the segmentation itself is the deliverable and will be applied to future data.",
            "When you want the assign strategy and its disclosures to travel with the centroids, rather than being re-derived from memory.",
        ),
        avoid=(
            "Do not expect `checkpoint_load` to restore your clustering: checkpoints deliberately do not embed cluster plans.",
            "Do not hand-copy centroid coordinates into another system; you lose the feature contract and the assign disclosures.",
        ),
        myths=(
            (
                "A checkpoint saves everything about my session.",
                "It saves data, roles, split membership, and history. Fitted domain plans live in their own bundles so each artifact has one clear meaning.",
            ),
            (
                "Bundles are interchangeable.",
                "An unsupervised bundle, a Torch bundle, and a RAG bundle have different layouts and different contracts. Loading the wrong one fails by design.",
            ),
        ),
        example=(
            "session.fit_clusters(method='kmeans', n_clusters=4, random_state=0)",
            "session.save_unsupervised_bundle('artifacts/segments')",
            "later = Session.ingest(new_frame).load_unsupervised_bundle('artifacts/segments')",
            "labels = later.assign_clusters()",
        ),
        check=(
            "If you deleted your notebook today, which file would restore the segmentation?",
            "Do the columns in your new frame match the ones the plan was fitted on?",
        ),
        tools=("save_unsupervised_bundle", "load_unsupervised_bundle", "assign_clusters", "checkpoint_save"),
        terms=("bundle", "checkpoint", "plan", "schema"),
        difficulty=CORE,
    ),
)

__all__ = ["UNSUPERVISED_BEGINNER"]
