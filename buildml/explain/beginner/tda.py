# ruff: noqa: E501
"""Beginner layers for topological data analysis."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

TDA_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "tda-persistent-homology",
        plain=(
            "Topological data analysis measures the *shape* of your data. Around each row, BuildML gathers "
            "its nearest training neighbours into a small cloud of points and asks structural questions: "
            "does this cloud form one blob or several, does it contain a hole? Those answers become numbers "
            "you can feed to an ordinary model."
        ),
        analogy=(
            "Describing a crowd without naming anyone: are they in one group or three, is there a ring "
            "around something in the middle? Those facts survive stretching and rotation, which is exactly "
            "why they are useful."
        ),
        steps=(
            "For each row, find its `knn` nearest neighbours among the training rows only.",
            "Grow a ball around every point in that little cloud, slowly increasing the radius.",
            "Record when clusters merge and when loops appear and close — each event is a birth/death pair.",
            "The collection of pairs is a persistence diagram: features that persist over a wide radius range are real structure, brief ones are noise.",
            "Vectorize the diagram into fixed-length numbers so a normal classifier or regressor can use it.",
        ),
        use=(
            "When the geometric arrangement of similar rows carries signal that individual column values miss.",
            "For sensor, spectral, and shape-like data where cycles and connectivity mean something physical.",
        ),
        avoid=(
            "Do not reach for it before trying ordinary features; it is expensive and often adds little on plain tabular data.",
            "Do not use it with a tiny `knn` — a five-point cloud has no interesting shape to find.",
        ),
        myths=(
            (
                "Topological data analysis replaces ordinary features.",
                "It produces additional features. A model still has to learn from them, and the boring columns usually still do most of the work.",
            ),
            (
                "The neighbour search can use all the rows.",
                "The neighbour index is built from training rows only. If a test row could pull in other test rows, the holdout would be contaminated.",
            ),
        ),
        example=(
            "# pip install \"buildml[tda]\"",
            "session.fit_tda(knn=25, vectorization='persistence_image', head='random_forest')",
            "session.evaluate_tda(partition='test')",
        ),
        check=(
            "Is your `knn` large enough for a loop to even be visible?",
            "Have you standardized your features? Distances are meaningless when one column dominates the scale.",
        ),
        tools=("fit_tda", "transform_tda", "predict_tda", "evaluate_tda"),
        terms=("topological data analysis", "persistence diagram", "extra", "feature scaling"),
        difficulty=ADVANCED,
    ),
    _layer(
        "tda-vectorization",
        plain=(
            "A persistence diagram is a scatter of points, and models need fixed-length rows of numbers. "
            "Vectorization converts the diagram into that fixed vector — as a blurred image, a set of "
            "layered peaks, or a weighted average curve."
        ),
        analogy=(
            "Turning a signature into a fixed set of measurements so a computer can compare signatures. "
            "You lose the picture and gain comparability."
        ),
        steps=(
            "Persistence image: blur the diagram points onto a small grid and read out the pixels.",
            "Landscape: convert each point into a tent shape and record the top few layers on a grid.",
            "Silhouette: take one weighted average of all the tents.",
            "The grid ranges are fixed from the training diagrams and frozen.",
            "Every row then produces a vector of the same length, whatever its diagram looked like.",
        ),
        use=(
            "Persistence images when you want a general-purpose, well-tested default.",
            "Landscapes or silhouettes when you want fewer, smoother features and faster fitting.",
        ),
        avoid=(
            "Do not recompute the grid ranges on test diagrams; the frozen training ranges are what keep evaluation honest.",
            "Do not crank the resolution up — you get hundreds of nearly empty columns and a slower, worse model.",
        ),
        myths=(
            (
                "A bigger vector captures more shape.",
                "Past a modest resolution you are adding empty cells. Feature count grows fast and most of it is zeros.",
            ),
            (
                "An all-zero vector means something broke.",
                "It usually means the diagrams were empty, which is normal for small clouds. It does mean topology is contributing nothing for those rows.",
            ),
        ),
        example=(
            "session.fit_tda(vectorization='landscape', n_bins=20, n_layers=3)",
            "features = session.transform_tda(partition='train')",
            "print(features.shape)",
        ),
        check=(
            "How many topological features did you create, and how many rows do you have?",
            "What fraction of your vectors are all zeros?",
        ),
        tools=("fit_tda", "transform_tda", "evaluate_tda"),
        terms=("persistence diagram", "topological data analysis", "feature", "overfitting"),
        difficulty=ADVANCED,
    ),
    _layer(
        "tda-supervised-head",
        plain=(
            "Once shapes are numbers, you attach an ordinary model on top — logistic regression, random "
            "forest, gradient boosting. That model is the 'head'. You can also skip it and just take the "
            "features out for use elsewhere."
        ),
        analogy=(
            "The topology part is the measuring instrument; the head is the person reading the "
            "measurements and making the call. Either can be swapped independently."
        ),
        steps=(
            "Choose a head appropriate to your task — a classifier for labels, a regressor for numbers.",
            "It is fitted on the training rows' topological features and training labels only.",
            "`predict_tda` runs new rows through the frozen extraction pipeline and the head.",
            "`evaluate_tda` scores the whole thing on a held-out partition.",
            "Set `head='none'` when you only want the features and will model them yourself.",
        ),
        use=(
            "When you want an end-to-end topological model measured in one place.",
            "With `head='none'` when topological features are one input among many in a larger pipeline.",
        ),
        avoid=(
            "Do not call `evaluate_tda` with `head='none'` — there is no model to score, and BuildML raises rather than guessing.",
            "Do not judge the approach on the training score; topological features can memorize local neighbourhoods.",
        ),
        myths=(
            (
                "The head choice barely matters once you have topological features.",
                "It matters as much as it does anywhere else. Persistence-image features are high-dimensional and sparse, which suits some heads far better than others.",
            ),
            (
                "A good training score means the topology found real structure.",
                "The neighbour construction makes memorization easy. Only the holdout score tells you anything.",
            ),
        ),
        example=(
            "session.fit_tda(head='random_forest', random_state=0)",
            "report = session.evaluate_tda(partition='test')",
            "print(report.metrics)",
        ),
        check=(
            "Is your holdout score better than the same head on plain features?",
            "Do you actually need the head, or are you feeding these features into something else?",
        ),
        tools=("fit_tda", "predict_tda", "evaluate_tda", "transform_tda"),
        terms=("topological data analysis", "model", "holdout", "overfitting"),
        difficulty=CORE,
    ),
    _layer(
        "tda-bundle-boundary",
        plain=(
            "The fitted topology pipeline — the neighbour index, the diagram settings, the frozen "
            "vectorizer ranges, and the head — saves as its own bundle. A Session checkpoint does not "
            "contain it."
        ),
        analogy=(
            "The calibrated instrument travels in its own case. Your lab notebook is a separate item and "
            "does not include the instrument."
        ),
        steps=(
            "Fit a topological pipeline.",
            "Call `save_tda_bundle(path)`.",
            "Reload with `load_tda_bundle(path)` — the training neighbour index comes back with it.",
            "Transform or predict on new rows.",
            "Use checkpoints separately for data and workflow state.",
        ),
        use=(
            "When topological scoring runs outside the notebook where it was fitted.",
            "When you need to reproduce a result months later with the exact same frozen ranges.",
        ),
        avoid=(
            "Do not expect `checkpoint_load` to bring the topology pipeline back.",
            "Do not mix these with graph, RAG, or reinforcement-learning bundles — the formats are distinct and loading enforces it.",
        ),
        myths=(
            (
                "The bundle is small since it is just settings.",
                "It carries the training neighbour index, which is needed to build point clouds for new rows. That can be sizeable.",
            ),
            (
                "One bundle type can hold everything.",
                "Each domain has its own load-time contract. Separate bundles let BuildML fail loudly on mismatch rather than silently restoring nothing.",
            ),
        ),
        example=(
            "session.save_tda_bundle('artifacts/shape-model')",
            "service = Session.ingest(new_rows).load_tda_bundle('artifacts/shape-model')",
            "service.predict_tda()",
        ),
        check=(
            "Does your saved bundle include a head, or features only?",
            "Have you recorded which BuildML extras were installed when it was fitted?",
        ),
        tools=("save_tda_bundle", "load_tda_bundle", "fit_tda", "checkpoint_save"),
        terms=("bundle", "checkpoint", "topological data analysis"),
        difficulty=CORE,
    ),
    _layer(
        "tda-extra-boundary",
        plain=(
            "Topology needs specialist libraries that BuildML does not install by default. "
            "`buildml[tda]` gives you the native path; `buildml[tda-industry]` adds giotto-tda. Without "
            "them, `fit_tda` raises a clear error naming the extra you need."
        ),
        analogy=(
            "A power tool sold without the specialist bit. The manual tells you exactly which bit to buy "
            "rather than letting you jam the wrong one in."
        ),
        steps=(
            "`import buildml` never requires these libraries — the base install stays light.",
            "Install `buildml[tda]` for ripser and persim, the native persistence path.",
            "Install `buildml[tda-industry]` for giotto-tda and its extra vectorizers.",
            "Call `tda_capability_matrix()` to see what is actually available in your environment.",
            "If both are installed, giotto is the default backend; silhouette vectorization stays native-only.",
        ),
        use=(
            "Install the native extra when you want persistence images and landscapes.",
            "Install the industry extra when you want Betti curves or a Mapper summary.",
        ),
        avoid=(
            "Do not assume a topology feature exists because the method name appears in the documentation; check the capability matrix.",
            "Do not import ripser or giotto at your own module import time either — it defeats the point of the optional boundary.",
        ),
        myths=(
            (
                "An error about a missing extra means topology is broken.",
                "It means an optional dependency is not installed. The message names the exact extra to add.",
            ),
            (
                "The industry backend is strictly better.",
                "It has more vectorizers. The native path is lighter, installs more easily on some platforms, and is the only one with silhouettes.",
            ),
        ),
        example=(
            "pip install \"buildml[tda]\"           # native: ripser + persim",
            "pip install \"buildml[tda-industry]\"  # adds giotto-tda",
            "Session.tda_capability_matrix()",
        ),
        check=(
            "Which extras does your deployment environment actually have?",
            "Does your chosen vectorization exist on the backend you will run?",
        ),
        tools=("tda_capability_matrix", "fit_tda", "require_tda_stack"),
        terms=("extra", "topological data analysis", "backend"),
        difficulty=CORE,
    ),
    _layer(
        "tda-giotto-backend",
        plain=(
            "giotto-tda is a widely used topology library. With `buildml[tda-industry]` installed, "
            "`backend='giotto'` runs its persistence computation and vectorizers behind exactly the same "
            "BuildML calls, plus an optional Mapper summary of your training data."
        ),
        analogy=(
            "Swapping in a different brand of engine while the dashboard and pedals stay where they are."
        ),
        steps=(
            "Install the industry extra and pass `backend='giotto'`.",
            "Vietoris–Rips persistence is computed by giotto rather than ripser.",
            "Betti curves, persistence images, and landscapes come from giotto's vectorizers.",
            "The vectorizer ranges are still frozen on training data.",
            "Set `mapper=True` for a diagnostic Mapper summary — a shape overview, not a feature source.",
        ),
        use=(
            "When you specifically want Betti curves, which the native path does not provide.",
            "When you are cross-checking BuildML's topology against a familiar library.",
        ),
        avoid=(
            "Do not treat the Mapper output as model input; it is disclosure for your eyes only.",
            "Do not expect interactive Mapper visualization from the Session API — that lives in giotto's own tooling.",
        ),
        myths=(
            (
                "Mapper output feeds the supervised head.",
                "It does not. It is a training-set diagnostic, deliberately excluded from features to avoid confusing disclosure with signal.",
            ),
            (
                "Native and giotto give identical numbers.",
                "They implement the same mathematics with different details and defaults. Expect close, not identical, and never mix a bundle fitted on one with the other.",
            ),
        ),
        example=(
            "session.fit_tda(backend='giotto', vectorization='betti_curve', mapper=True)",
            "print(Session.tda_capability_matrix()['backends']['giotto'])",
        ),
        check=(
            "Which backend was your saved bundle fitted with?",
            "Are you reading Mapper output as a diagnostic or mistaking it for features?",
        ),
        tools=("fit_tda", "tda_capability_matrix", "evaluate_tda"),
        terms=("topological data analysis", "backend", "extra", "persistence diagram"),
        difficulty=ADVANCED,
    ),
)

__all__ = ["TDA_BEGINNER"]
