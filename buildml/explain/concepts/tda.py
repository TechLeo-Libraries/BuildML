# ruff: noqa: E501
"""Topological Data Analysis concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

TDA_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="tda-persistent-homology",
            title="Persistent homology on local point clouds",
            summary=(
                "session.tda.fit builds local Vietoris–Rips diagrams (ripser) from kNN "
                "train neighborhoods, then vectorizes them for sklearn."
            ),
            definition=(
                "Persistent homology tracks topological features (connected "
                "components, loops, …) across a filtration. BuildML forms a "
                "local point cloud per row from knn nearest **train** neighbors "
                "and computes Vietoris–Rips diagrams with ripser."
            ),
            intuition=(
                "Around each sample, look at a small cloud of similar train "
                "points and summarize the shape of that cloud as birth/death pairs."
            ),
            formal_idea=(
                "VR(ε) complex on N_k(x) ⊆ X_train; persistence modules H_*(VR); "
                "diagrams Dgm_i = {(b,d)}."
            ),
            why_it_matters=(
                "Train-only NN index + diagrams preserve holdout honesty.",
                "Gives geometry-aware features beyond linear summaries.",
            ),
            how_buildml_uses=(
                "session.tda.fit → session.tda.transform / session.tda.predict / session.tda.evaluate.",
            ),
            interpretation_rules=(
                "Empty H1 diagrams are common for tiny clouds: vectors may be sparse.",
                "Prefer holdout session.tda.evaluate over train_score.",
            ),
            assumptions=(
                "Numeric features; split present; buildml[tda] installed.",
            ),
            failure_modes=(
                "Too-small knn → trivial diagrams; scale mismatch without standardize.",
            ),
            anti_patterns=(
                "Refitting ripser ranges on the full frame including test.",
                "Calling this a full Mapper / TDA research platform.",
            ),
            worked_example_pattern=(
                "session.tda.fit(vectorization='persistence_image') → session.tda.evaluate().",
            ),
            related_concepts=(
                "tda-vectorization",
                "tda-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="tda-vectorization",
            title="Persistence images, landscapes, silhouettes",
            summary=(
                "Train-fitted vectorizers map diagrams to fixed-length features "
                "for sklearn heads."
            ),
            definition=(
                "Persistence images (persim) rasterize weighted diagrams; "
                "landscapes sample layered tent functions (in-tree); silhouettes "
                "are weighted averages of tents on a grid (in-tree)."
            ),
            intuition=(
                "Turn a scatter of birth/death points into a numeric vector "
                "the same way every time, using ranges learned on train."
            ),
            formal_idea=(
                "PI: KDE on (birth, persistence); PL: λ_k(t); silhouette: "
                "Σ w_i Λ_i(t) / Σ w_i."
            ),
            why_it_matters=(
                "Frozen train ranges prevent holdout leakage into the feature map.",
            ),
            how_buildml_uses=(
                "vectorization='persistence_image'|'landscape'|'silhouette' on session.tda.fit.",
            ),
            interpretation_rules=(
                "feature_dim = per-homology-dim size × len(homology_dims).",
            ),
            assumptions=("Finite train diagrams exist to set ranges.",),
            failure_modes=("All-empty diagrams → near-zero vectors; weak signal.",),
            anti_patterns=("Re-estimating birth_range on test diagrams.",),
            worked_example_pattern=(
                "session.tda.fit(vectorization='landscape', n_bins=20, n_layers=3).",
            ),
            related_concepts=("tda-persistent-homology", "tda-supervised-head"),
        ),
        _note(
            key="tda-supervised-head",
            title="Sklearn head on topological features",
            summary=(
                "Optional classify/regress head fitted on train TDA vectors; "
                "session.tda.evaluate scores holdout with the frozen pipeline."
            ),
            definition=(
                "After vectorization, a sklearn estimator (logistic / RF / ridge / "
                "HGB) is fit on train topological features and train labels."
            ),
            intuition=(
                "Use the shape features as inputs to a normal classifier or regressor."
            ),
            formal_idea="ŷ = h(φ_θ̂(x)) with θ̂, h fitted on train only.",
            why_it_matters=(
                "Separates topology extraction from supervised scoring.",
            ),
            how_buildml_uses=(
                "head=... on session.tda.fit; session.tda.predict / session.tda.evaluate; head='none' for features only.",
            ),
            interpretation_rules=(
                "classification → accuracy/macro_f1; regression → rmse/mae/r2.",
            ),
            assumptions=("Non-null train targets when head!='none'.",),
            failure_modes=("head='none' then session.tda.evaluate → ValidationError.",),
            anti_patterns=("Fitting the head on concatenated train+test TDA features.",),
            worked_example_pattern=(
                "session.tda.fit(head='random_forest') → session.tda.evaluate(partition='test').",
            ),
            related_concepts=("tda-vectorization", "tda-bundle-boundary"),
        ),
        _note(
            key="tda-bundle-boundary",
            title="TDA bundle vs Session checkpoint",
            summary=(
                "buildml.tda_bundle.v2 stores TdaPlan (v1 loadable); checkpoints "
                "do not embed the TDA transformer."
            ),
            definition=(
                "A TDA bundle directory holds meta.json + tda_plan.joblib under "
                "schema buildml.tda_bundle.v2 (v1 bundles remain loadable)."
            ),
            intuition="Save the PH pipeline separately from workflow resume state.",
            formal_idea="TdaPlan is not embedded in a Session checkpoint payload.",
            why_it_matters=("Avoid silent gaps when reloading workflows.",),
            how_buildml_uses=("session.tda.save_bundle / session.tda.load_bundle.",),
            interpretation_rules=(
                "Reload via session.tda.load_bundle after checkpoint_load.",
            ),
            assumptions=("Bundle format matches buildml.tda_bundle.v2 (or v1).",),
            failure_modes=("Mixing TDA bundles with RL / RAG / graph bundles.",),
            anti_patterns=("Expecting checkpoint_load to restore TdaPlan.",),
            worked_example_pattern=(
                "session.tda.save_bundle(path) → session.tda.load_bundle(path).",
            ),
            related_concepts=("tda-persistent-homology",),
        ),
        _note(
            key="tda-extra-boundary",
            title="buildml[tda] and buildml[tda-industry] optional stacks",
            summary=(
                "ripser + persim (native) and giotto-tda (industry) are optional; "
                "MissingExtraError if absent. import buildml never requires them."
            ),
            definition=(
                "buildml[tda] installs ripser (VR PH) and persim (persistence images). "
                "buildml[tda-industry] adds giotto-tda for BettiCurve, gtda vectorizers, "
                "and optional KeplerMapper train summaries. Silhouette vectorization "
                "is in-tree on the native path."
            ),
            intuition="Core stays light; install tda for native PH, tda-industry for giotto.",
            formal_idea="MissingExtraError('tda'|'tda-industry', feature) on ImportError.",
            why_it_matters=("Keeps the default install small.",),
            how_buildml_uses=(
                "pip install 'buildml[tda]' (native); "
                "pip install 'buildml[tda-industry]' (giotto); "
                "session.tda.capability_matrix(); require_tda_stack()."
            ,),
            interpretation_rules=(
                "Default backend when both installed: giotto (industry depth). "
                "Silhouette remains native-only.",
            ),
            assumptions=("Compatible wheels for the platform Python.",),
            failure_modes=(
                "Extra not installed → MissingExtraError on session.tda.fit.",
            ),
            anti_patterns=("Importing ripser/gtda at buildml package import time.",),
            worked_example_pattern=(
                "session.tda.fit(backend='giotto', vectorization='betti_curve')."
            ,),
            related_concepts=("tda-persistent-homology", "tda-giotto-backend"),
        ),
        _note(
            key="tda-giotto-backend",
            title="giotto-tda industry backend",
            summary=(
                "When buildml[tda-industry] is installed, backend='giotto' runs "
                "gtda VietorisRipsPersistence + BettiCurve / PersistenceImage / "
                "PersistenceLandscape; optional Mapper summary on train."
            ),
            definition=(
                "The giotto adapter wraps giotto-tda homology and diagram "
                "vectorizers behind the same Session fit/transform/evaluate API."
            ),
            intuition=(
                "Industry-standard sklearn-compatible TDA transformers on local "
                "kNN train point clouds."
            ),
            formal_idea="backend='giotto' → gtda PH + frozen train vectorizer.",
            why_it_matters=(
                "Betti curves and gtda vectorizers without abandoning the native path.",
            ),
            how_buildml_uses=(
                "session.tda.fit(backend='giotto', vectorization='betti_curve', mapper=True)."
            ,),
            interpretation_rules=(
                "Mapper output is diagnostic disclosure only: not supervised features.",
            ),
            assumptions=("buildml[tda-industry] installed.",),
            failure_modes=("giotto missing → MissingExtraError; Mapper may warn on tiny train.",),
            anti_patterns=("Expecting interactive Mapper visualization from Session API.",),
            worked_example_pattern=(
                "session.tda.capability_matrix()['backends']['giotto']."
            ,),
            related_concepts=("tda-extra-boundary", "tda-vectorization"),
        ),
    )
}
