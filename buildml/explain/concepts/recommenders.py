# ruff: noqa: E501
"""Recommendation systems concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

RECOMMENDER_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="recommender-collaborative-filtering",
            title="Collaborative filtering (user/item interactions)",
            summary=(
                "session.recommender.fit learns from train user–item interactions "
                "(neighborhood CF or matrix factorization) for top-K ranking."
            ),
            definition=(
                "Collaborative filtering predicts preferences from the "
                "interaction matrix alone: similar users or items, or latent "
                "factors (SVD/NMF), without requiring side features."
            ),
            intuition=(
                "People who liked the same things before will like similar "
                "things next; similar items co-occur for the same users."
            ),
            formal_idea=(
                "R ∈ R^{|U|×|I|} train-only; item_knn / user_knn via cosine; "
                "SVD/NMF: R ≈ UVᵀ."
            ),
            why_it_matters=(
                "Train-only matrix preserves holdout honesty.",
                "Ranking metrics need a clear candidate catalog.",
            ),
            how_buildml_uses=(
                "session.recommender.fit(method='item_knn'|'user_knn'|'svd'|'nmf').",
            ),
            interpretation_rules=(
                "Prefer holdout Precision@K / Recall@K / nDCG@K / MAP@K.",
                "Known-item protocol: candidates ⊆ train items.",
            ),
            assumptions=(
                "user_column + item_column present; split present; enough overlap.",
            ),
            failure_modes=(
                "Cold users/items; tiny catalogs; popularity collapse.",
            ),
            anti_patterns=(
                "Fitting on full-frame interactions including test.",
                "Calling this a Netflix-scale recsys platform.",
                "Confusing with RAG retrieve or EDA Recommendation Findings.",
            ),
            worked_example_pattern=(
                "session.recommender.fit(method='item_knn', user_column=..., item_column=...) "
                "→ session.recommender.evaluate(k=10).",
            ),
            related_concepts=(
                "recommender-ranking-metrics",
                "recommender-cold-start",
                "recommender-industry-backends",
                "recommender-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="recommender-content-based",
            title="Content-based item scoring",
            summary=(
                "method='content' builds rating-weighted user profiles from "
                "numeric item features observed in train."
            ),
            definition=(
                "Content-based recommenders score catalog items by similarity "
                "between a user profile (aggregate of consumed item features) "
                "and candidate item feature vectors."
            ),
            intuition=(
                "If you liked fast red cars, recommend other items with similar "
                "numeric descriptors: even without collaborative neighbors."
            ),
            formal_idea="u = Σ r_i x_i / Σ |r_i|; score(i) = cos(u, x_i).",
            why_it_matters=(
                "Helps when collaborative signal is thin but item features exist.",
            ),
            how_buildml_uses=(
                "session.recommender.fit(method='content', item_feature_columns=[...]).",
            ),
            interpretation_rules=(
                "Still restricted to the train item catalog (known-item).",
            ),
            assumptions=("Numeric item_feature_columns on train rows.",),
            failure_modes=("Missing/non-numeric features; empty user history.",),
            anti_patterns=("Fitting content scalers on holdout item rows.",),
            worked_example_pattern=(
                "session.recommender.fit(method='content', item_feature_columns=['f1','f2']).",
            ),
            related_concepts=(
                "recommender-collaborative-filtering",
                "recommender-cold-start",
            ),
        ),
        _note(
            key="recommender-ranking-metrics",
            title="Precision@K, Recall@K, nDCG@K, MAP@K",
            summary=(
                "session.recommender.evaluate scores top-K lists against holdout "
                "known-item positives for warm users."
            ),
            definition=(
                "Precision@K = |hit∩topK|/K; Recall@K = |hit∩topK|/|relevant|; "
                "nDCG@K discounts hits by log rank; MAP@K averages precision at "
                "each hit rank."
            ),
            intuition=(
                "Did the right items appear near the top of the list for each user?"
            ),
            formal_idea="Macro-average over warm holdout users with ≥1 known relevant item.",
            why_it_matters=(
                "Accuracy/RMSE on ratings is not the same as ranking quality.",
            ),
            how_buildml_uses=("session.recommender.evaluate(partition=..., k=...).",),
            interpretation_rules=(
                "Cold-start users are excluded from averages and counted separately.",
                "Holdout-only items are dropped from relevant sets (disclosed).",
            ),
            assumptions=("Frozen train plan; exclude train history from candidates.",),
            failure_modes=("No warm users → zeros with disclosure.",),
            anti_patterns=("Reporting train reconstruction error as ranking quality.",),
            worked_example_pattern=("session.recommender.evaluate(partition='test', k=10).",),
            related_concepts=(
                "recommender-collaborative-filtering",
                "recommender-cold-start",
            ),
        ),
        _note(
            key="recommender-cold-start",
            title="Cold-start users/items and known-item protocol",
            summary=(
                "Users/items absent from train are disclosed; candidates are "
                "always the train item catalog."
            ),
            definition=(
                "Cold-start users have no train history; cold items never appear "
                "in the train catalog. BuildML uses a known-item protocol: "
                "recommend only train items; cold users use popularity or skip."
            ),
            intuition=(
                "You cannot honestly score an item the model has never seen as "
                "a collaborative candidate without an external side model."
            ),
            formal_idea="Candidates ⊆ I_train; cold users → popularity or ∅.",
            why_it_matters=("Prevents silent leakage of holdout-only catalog ids.",),
            how_buildml_uses=("cold_start='popularity'|'skip' on session.recommender.fit.",),
            interpretation_rules=(
                "n_cold_start_users on eval/recommend is a first-class disclosure.",
            ),
            assumptions=("Train catalog is the deployment candidate set for CF.",),
            failure_modes=("Most users cold → ranking averages over few warm users.",),
            anti_patterns=("Adding test-only items into the similarity graph.",),
            worked_example_pattern=(
                "session.recommender.fit(cold_start='popularity') → recommend(partition='test').",
            ),
            related_concepts=(
                "recommender-ranking-metrics",
                "recommender-bundle-boundary",
            ),
        ),
        _note(
            key="recommender-industry-backends",
            title="Industry recommender backends (implicit, LightFM)",
            summary=(
                "With buildml[recommenders-industry], implicit ALS/BPR is the "
                "default for feedback='implicit'; LightFM supports hybrid side features."
            ),
            definition=(
                "Industry backends wrap mature libraries instead of reimplementing "
                "ALS/BPR: implicit for sparse implicit-feedback CF; LightFM for "
                "hybrid user/item features plus interactions."
            ),
            intuition=(
                "Implicit feedback (clicks, views) needs sparse matrix factorization "
                "engines; side features help cold catalog items via hybrid models."
            ),
            formal_idea=(
                "backend='implicit' → ALS/BPR on CSR user×item; "
                "backend='lightfm' → WARP on interactions + optional feature CSR."
            ),
            why_it_matters=(
                "Honest defaults when extras are installed; sklearn core remains "
                "available without industry dependencies.",
            ),
            how_buildml_uses=(
                "session.recommender.fit(feedback='implicit') defaults to als; "
                "session.recommender.fit(method='lightfm', item_feature_columns=[...]).",
            ),
            interpretation_rules=(
                "Inspect session.recommender.capability_matrix() for install state.",
                "Known-item protocol applies to all backends.",
            ),
            assumptions=(
                "recommenders-industry extra for implicit/LightFM paths.",
            ),
            failure_modes=(
                "MissingExtraError when industry backend requested without install.",
            ),
            anti_patterns=(
                "Reimplementing ALS from scratch in BuildML core.",
                "Using implicit backend with explicit-only rating semantics.",
            ),
            worked_example_pattern=(
                "pip install 'buildml[recommenders-industry]' → "
                "session.recommender.fit(feedback='implicit') → session.recommender.evaluate(k=10).",
            ),
            related_concepts=(
                "recommender-collaborative-filtering",
                "recommender-content-based",
                "recommender-ranking-metrics",
            ),
        ),
        _note(
            key="recommender-bundle-boundary",
            title="Recommender bundle vs Session checkpoint vs RAG / EDA",
            summary=(
                "buildml.recommender_bundle.v1 stores RecommenderPlan; checkpoints "
                "do not. Distinct from RAG bundles and EDA Recommendation Findings."
            ),
            definition=(
                "A recommender bundle directory holds meta.json + "
                "recommender_plan.joblib under schema buildml.recommender_bundle.v1."
            ),
            intuition="Save the CF model separately from workflow resume state.",
            formal_idea="RecommenderPlan is not embedded in a Session checkpoint payload.",
            why_it_matters=("Avoid silent gaps when reloading workflows.",),
            how_buildml_uses=("session.recommender.save_bundle / session.recommender.load_bundle.",),
            interpretation_rules=(
                "Reload via session.recommender.load_bundle after checkpoint_load.",
                "Do not confuse with rag_* or explain.schemas.Recommendation.",
            ),
            assumptions=("Bundle format matches buildml.recommender_bundle.v1.",),
            failure_modes=("Mixing recommender bundles with RAG / TDA bundles.",),
            anti_patterns=("Expecting checkpoint_load to restore RecommenderPlan.",),
            worked_example_pattern=(
                "session.recommender.save_bundle(path) → session.recommender.load_bundle(path).",
            ),
            related_concepts=("recommender-collaborative-filtering",),
        ),
    )
}
