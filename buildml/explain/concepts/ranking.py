# ruff: noqa: E501
"""Learning-to-rank (tabular search ranking) concept notes."""

from __future__ import annotations

from buildml.explain.concepts._builder import _note
from buildml.explain.schemas import ConceptNote

RANKING_NOTES: dict[str, ConceptNote] = {
    note.key: note
    for note in (
        _note(
            key="ltr-tabular-ranking",
            title="Tabular learning-to-rank (query–item judgments)",
            summary=(
                "fit_ranker learns from train rows of (query, item, features, "
                "relevance) to score and order items per query."
            ),
            definition=(
                "Learning-to-rank (LTR) treats each labeled judgment as a "
                "feature row tied to a query group. Models predict relevance "
                "scores used to order candidates within a query."
            ),
            intuition=(
                "Given query features and item/document features, learn which "
                "candidates should rank higher for that query — not which "
                "users like which catalog items (recommenders), and not "
                "embedding nearest-neighbor retrieve (RAG)."
            ),
            formal_idea=(
                "Rows (q, d, x, y); pointwise: ŷ = f(x); pairwise: prefer "
                "sign(y_i − y_j) via f(x_i) − f(x_j)."
            ),
            why_it_matters=(
                "Query-grouped splits keep holdout labels honest.",
                "Ranking metrics need per-query candidate lists.",
            ),
            how_buildml_uses=(
                "Session.fit_ranker(method='pointwise'|'pairwise', "
                "query_column=..., item_column=...).",
            ),
            interpretation_rules=(
                "Prefer holdout nDCG@K / MAP@K / MRR@K macro-averaged over queries.",
                "Prefer group_split on query_column.",
            ),
            assumptions=(
                "query_column + item_column + numeric features + relevance; split present.",
            ),
            failure_modes=(
                "Row-random splits leaking query labels; tiny candidate sets; all-zero labels.",
            ),
            anti_patterns=(
                "Fitting on full-frame judgments including test queries.",
                "Calling this a search-engine product or RAG retrieve.",
                "Confusing with fit_recommender user–item CF.",
            ),
            worked_example_pattern=(
                "group_split(group_column='query_id') → "
                "fit_ranker(method='pointwise', query_column=..., item_column=...) "
                "→ evaluate_ranker(k=10).",
            ),
            related_concepts=(
                "ltr-pointwise",
                "ltr-pairwise",
                "ltr-ranking-metrics",
                "ltr-bundle-boundary",
                "leakage-boundary",
            ),
        ),
        _note(
            key="ltr-pointwise",
            title="Pointwise relevance regression",
            summary=(
                "method='pointwise' fits Ridge or HistGradientBoostingRegressor "
                "to predict graded relevance from features."
            ),
            definition=(
                "Pointwise LTR ignores explicit pair/list structure at training "
                "time and treats ranking as supervised regression (or "
                "classification) on relevance labels."
            ),
            intuition=(
                "If the model predicts higher relevance for better matches, "
                "sorting by score yields a ranking."
            ),
            formal_idea=("min_f Σ_i ℓ(f(x_i), y_i) on train judgments."),
            why_it_matters=(
                "Simple, strong baseline; works with graded labels.",
                "Does not directly optimize pairwise preferences.",
            ),
            how_buildml_uses=(
                "fit_ranker(method='pointwise', pointwise_estimator='ridge'|'hgb').",
            ),
            interpretation_rules=(
                "Still evaluate with ranking metrics, not only RMSE.",
            ),
            assumptions=("Numeric features; finite relevance labels."),
            failure_modes=("Calibration mismatch; dominance by frequent queries."),
            anti_patterns=("Reporting only regression loss as ranking quality."),
            worked_example_pattern=(
                "fit_ranker(method='pointwise', pointwise_estimator='ridge') → rank()."
            ),
            related_concepts=("ltr-tabular-ranking", "ltr-pairwise", "ltr-ranking-metrics"),
        ),
        _note(
            key="ltr-pairwise",
            title="Pairwise RankSVM-lite",
            summary=(
                "method='pairwise' trains LinearSVC on within-query feature "
                "differences (RankSVM-style)."
            ),
            definition=(
                "Pairwise LTR samples item pairs inside each query and learns "
                "a scoring function whose differences prefer the higher-labeled "
                "item."
            ),
            intuition=(
                "Instead of predicting absolute grades, learn 'A should beat B "
                "for this query' from feature differences."
            ),
            formal_idea=(
                "For pairs (i, j) in query q with y_i > y_j: classify x_i − x_j "
                "as +1 via LinearSVC; score(x) = w·x."
            ),
            why_it_matters=(
                "Closer to ranking preferences than pointwise RMSE.",
                "Still sklearn-core — not LambdaMART / LightGBM.",
            ),
            how_buildml_uses=(
                "fit_ranker(method='pairwise', max_pairs_per_query=...).",
            ),
            interpretation_rules=(
                "Needs queries with ≥2 items and distinct grades.",
            ),
            assumptions=("Multiple candidates per train query."),
            failure_modes=("No distinct-grade pairs; extreme class imbalance in pairs."),
            anti_patterns=(
                "Calling this LambdaMART or a production search stack.",
            ),
            worked_example_pattern=(
                "fit_ranker(method='pairwise') → evaluate_ranker(k=5)."
            ),
            related_concepts=("ltr-tabular-ranking", "ltr-pointwise", "ltr-ranking-metrics"),
        ),
        _note(
            key="ltr-ranking-metrics",
            title="LTR ranking metrics (nDCG / MAP / MRR)",
            summary=(
                "evaluate_ranker reports graded nDCG@K, MAP@K, and MRR@K "
                "macro-averaged over holdout queries."
            ),
            definition=(
                "nDCG@K uses graded gains (2^rel − 1); MAP@K and MRR@K "
                "binarize relevance above a threshold and average over queries."
            ),
            intuition=(
                "Good rankings put high-relevance items near the top for each "
                "query; metrics summarize that across the holdout query set."
            ),
            formal_idea=(
                "Macro mean over queries of nDCG@K / AP@K / RR@K on score-sorted lists."
            ),
            why_it_matters=(
                "Regression loss alone does not prove ranking quality.",
                "Same metric names appear in RAG/recommenders with different protocols.",
            ),
            how_buildml_uses=("Session.evaluate_ranker(partition='test', k=...)."),
            interpretation_rules=(
                "Compare only under the same k, threshold, and split policy.",
                "Do not equate with RAG chunk nDCG or recommender known-item nDCG.",
            ),
            assumptions=("Holdout queries with ≥1 relevant item."),
            failure_modes=("All-zero labels; tiny k; leaked query overlap."),
            anti_patterns=("Mixing RAG evaluate metrics with LTR evaluate_ranker."),
            worked_example_pattern=(
                "evaluate_ranker(k=10) → inspect metrics['ndcg_at_k']."
            ),
            related_concepts=("ltr-tabular-ranking", "leakage-boundary"),
        ),
        _note(
            key="ltr-bundle-boundary",
            title="Ranker bundle vs Session checkpoint / RAG / recommenders",
            summary=(
                "save_ranker_bundle stores RankerPlan as buildml.ranker_bundle.v1; "
                "checkpoints and RAG/recommender bundles are separate."
            ),
            definition=(
                "A ranker bundle persists the train-fitted feature contract and "
                "estimator. It does not embed the dataset or replace a Session "
                "checkpoint, RAG index, or recommender plan."
            ),
            intuition=(
                "Reload workflow state with checkpoint_load; reload the ranker "
                "with load_ranker_bundle."
            ),
            formal_idea=("buildml.ranker_bundle.v1 = meta.json + ranker_plan.joblib."),
            why_it_matters=("Prevents silent mixing of artifact types."),
            how_buildml_uses=(
                "Session.save_ranker_bundle / load_ranker_bundle.",
            ),
            interpretation_rules=(
                "Bundles are complementary to checkpoints, not interchangeable.",
            ),
            assumptions=("A RankerPlan exists."),
            failure_modes=("Expecting dataset rows inside the bundle."),
            anti_patterns=(
                "Treating ranker bundles as RAG corpora or recommender catalogs.",
            ),
            worked_example_pattern=(
                "save_ranker_bundle(path) → load_ranker_bundle(path) → evaluate_ranker()."
            ),
            related_concepts=("ltr-tabular-ranking", "checkpoint-boundary"),
        ),
    )
}
