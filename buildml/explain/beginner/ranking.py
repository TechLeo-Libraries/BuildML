# ruff: noqa: E501
"""Beginner layers for learning-to-rank."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

RANKING_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "ltr-tabular-ranking",
        plain=(
            "Learning to rank is for problems where the output is an *ordered list*, not a single label. "
            "Your data has queries, candidate items for each query, features describing the pairing, and a "
            "relevance judgement. The model learns to put the relevant items on top."
        ),
        analogy=(
            "A search results page. It does not matter whether the model thinks result #7 is 62% relevant; "
            "what matters is that the genuinely useful pages sit above the useless ones."
        ),
        steps=(
            "Shape your data as one row per query-item pair.",
            "Add a query identifier column so the model knows which rows compete with each other.",
            "Add the relevance judgement: binary clicked/not, or graded 0 to 4.",
            "Fit a ranker; it learns a scoring function used to sort within each query.",
            "Evaluate with ranking metrics averaged over held-out queries, never over individual rows.",
        ),
        use=(
            "Search results, candidate shortlists, triage queues, next-best-action ordering.",
            "Whenever a human or a downstream process consumes a top-N list rather than a single answer.",
        ),
        avoid=(
            "Do not use it when there is no grouping: with a single global ordering and no queries, ordinary regression on the score is simpler.",
            "Do not use it when only the top-1 answer is ever consumed and you have clean labels; classification is more direct.",
        ),
        myths=(
            (
                "A regression model on relevance is equivalent.",
                "Regression tries to get every absolute score right. Ranking only needs the ordering within each query, which is an easier and better-targeted objective.",
            ),
            (
                "Rows can be split randomly like any other dataset.",
                "All rows for one query must stay together. Splitting a query across partitions leaks its relevance pattern.",
            ),
        ),
        example=(
            "session.ranking.fit(",
            "    method='pointwise', query_column='query_id',",
            "    relevance_column='relevance',",
            ")",
            "ordered = session.ranking.rank(query_ids=['q_17'])",
            "session.ranking.evaluate(partition='test', k=10)",
        ),
        check=(
            "Are all rows for a given query on the same side of your split?",
            "How many candidate items does a typical query have?",
        ),
        tools=("fit_ranker", "rank", "evaluate_ranker", "group_split"),
        terms=("learning to rank", "nDCG", "recommender", "group split"),
        difficulty=CORE,
    ),
    _layer(
        "ltr-pointwise",
        plain=(
            "The pointwise approach is the simplest way in: forget that this is a ranking problem, predict "
            "each item's relevance score independently, then sort by the prediction. It ignores the "
            "competition between items and often works surprisingly well anyway."
        ),
        analogy=(
            "Grading every essay on its own merits and then ordering by grade. You never compare two essays "
            "directly, yet you still end up with a ranking."
        ),
        steps=(
            "Treat the relevance judgement as an ordinary regression target.",
            "Fit a regressor: ridge for a linear baseline, gradient boosting for something stronger.",
            "Predict a score for every candidate item.",
            "Sort within each query by predicted score.",
            "Evaluate with ranking metrics, which is where you find out whether the ordering is any good.",
        ),
        use=(
            "As your first ranking baseline: it is fast, simple, and reuses tools you already know.",
            "When relevance judgements are graded and reasonably reliable.",
        ),
        avoid=(
            "Do not use it when queries have wildly different score scales; a model trying to match absolute values wastes capacity on calibration that ranking does not need.",
            "Do not use it when top-of-list accuracy is critical; pairwise and listwise methods focus their effort there.",
        ),
        myths=(
            (
                "Pointwise ranking is not real learning to rank.",
                "It is the standard baseline, and on many datasets it is within a few points of far more complex methods.",
            ),
            (
                "Lower regression error means better ranking.",
                "The two can diverge. A model with worse absolute error but better within-query ordering ranks better, which is all that matters.",
            ),
        ),
        example=(
            "session.ranking.fit(",
            "    method='pointwise', query_column='query_id',",
            "    relevance_column='relevance', estimator='hist_gbdt',",
            ")",
            "session.ranking.evaluate(partition='validation', k=10)",
        ),
        check=(
            "Does your relevance scale mean the same thing across queries?",
            "How does pointwise compare with pairwise on your validation nDCG?",
        ),
        tools=("fit_ranker", "rank", "evaluate_ranker"),
        terms=("learning to rank", "nDCG", "gradient boosting", "metric"),
        difficulty=CORE,
    ),
    _layer(
        "ltr-pairwise",
        plain=(
            "The pairwise approach learns from comparisons instead of absolute scores. For every pair of "
            "items within a query where one is more relevant, it learns to score that one higher. The "
            "training signal becomes 'A beats B', which is exactly the thing you care about."
        ),
        analogy=(
            "Ranking chess players. You never need an absolute skill number: you just need to know who "
            "beat whom, and a consistent ordering falls out."
        ),
        steps=(
            "Within each query, form pairs where the relevance judgements differ.",
            "Compute the feature difference between the two items in each pair.",
            "Train a classifier to predict which one wins from that difference: RankSVM-style.",
            "At prediction time, score each item individually with the learned weights and sort.",
            "Evaluate with ranking metrics on held-out queries.",
        ),
        use=(
            "When relevance labels are only meaningfully comparable within a query, not across queries.",
            "When you have implicit feedback where you know 'this was clicked and that was not' but have no grades.",
        ),
        avoid=(
            "Do not use it on queries with very many candidates without sampling pairs; the number of pairs grows with the square of the candidate count.",
            "Do not use it when your judgements are already well-calibrated grades that transfer across queries: pointwise exploits that and pairwise discards it.",
        ),
        myths=(
            (
                "Pairwise always beats pointwise.",
                "It targets the ordering more directly and it also throws away magnitude information. Which wins is an empirical question on your data.",
            ),
            (
                "Every pair should be used.",
                "Pairs from near-identical relevance levels add noise. Most practical implementations sample or restrict pairs.",
            ),
        ),
        example=(
            "session.ranking.fit(",
            "    method='pairwise', query_column='query_id',",
            "    relevance_column='relevance',",
            ")",
            "session.ranking.evaluate(partition='validation', k=10)",
        ),
        check=(
            "How many pairs does your largest query generate?",
            "Are your relevance grades comparable across different queries?",
        ),
        tools=("fit_ranker", "rank", "evaluate_ranker"),
        terms=("learning to rank", "nDCG", "implicit feedback", "metric"),
        difficulty=ADVANCED,
    ),
    _layer(
        "ltr-industry-rankers",
        plain=(
            "The gradient-boosting libraries all ship dedicated ranking objectives: LightGBM's LambdaRank, "
            "XGBoost's rank:ndcg, CatBoost's YetiRank. These optimize the ranking metric directly rather "
            "than approaching it through regression or classification, and they are usually the strongest "
            "option available."
        ),
        analogy=(
            "Training specifically for the event you will compete in, rather than doing general fitness and "
            "hoping it transfers."
        ),
        steps=(
            "Install `pip install buildml[ranking-industry]`.",
            "Pass `backend='industry'` and choose the library.",
            "Provide query groups: these libraries need to know which rows belong to the same query.",
            "The objective optimizes an nDCG-style target directly during boosting.",
            "Evaluate with the same held-out ranking metrics for a fair comparison.",
        ),
        use=(
            "When ranking quality genuinely matters and you have enough queries to train on.",
            "As the production choice once a simpler baseline has proven the problem is worth solving.",
        ),
        avoid=(
            "Do not start here; establish a pointwise baseline first so you know what the complexity buys.",
            "Do not use it with very few queries: these objectives need many groups to estimate the gradient well.",
        ),
        myths=(
            (
                "LambdaRank optimizes nDCG exactly.",
                "nDCG is not differentiable. LambdaRank uses a gradient weighting derived from it, which works very well and is not the same as exact optimization.",
            ),
            (
                "The industry ranker will fix bad features.",
                "A better objective on the same weak features gives you a slightly better ordering of noise. Features remain the dominant factor.",
            ),
        ),
        example=(
            "# pip install \"buildml[ranking-industry]\"",
            "session.ranking.fit(",
            "    backend='industry', method='lambdarank',",
            "    query_column='query_id', relevance_column='relevance',",
            ")",
            "session.ranking.evaluate(partition='test', k=10)",
        ),
        check=(
            "How many distinct queries are in your training partition?",
            "How much does the industry ranker beat your pointwise baseline?",
        ),
        tools=("fit_ranker", "rank", "evaluate_ranker"),
        terms=("learning to rank", "gradient boosting", "nDCG", "extra"),
        difficulty=ADVANCED,
    ),
    _layer(
        "ltr-ranking-metrics",
        plain=(
            "Ranking metrics score the ordering within each query and then average across queries. nDCG@K "
            "rewards putting highly relevant items near the top. MAP@K averages precision as you walk down "
            "the list. MRR@K only cares where the first relevant item landed."
        ),
        analogy=(
            "Three different complaints about a search page: 'the best result was buried' (nDCG), 'most of "
            "page one was junk' (MAP), and 'I had to scroll to find anything useful at all' (MRR)."
        ),
        steps=(
            "Choose K to match how many results your interface actually shows.",
            "For each held-out query, score the model's ordering against the true relevance grades.",
            "Average across queries: every query counts equally, regardless of how many candidates it had.",
            "Use nDCG when relevance is graded; MRR when only the first hit matters, as in question answering.",
            "Report K and the metric together; neither means anything alone.",
        ),
        use=(
            "For every ranking comparison. These are the metrics the objectives are trying to move.",
            "When diagnosing which part of the list is weak: a good MRR with poor nDCG means the top is fine and the rest is not.",
        ),
        avoid=(
            "Do not average over rows instead of queries; a query with 500 candidates would dominate one with 5.",
            "Do not compare nDCG@5 against nDCG@20 across experiments.",
        ),
        myths=(
            (
                "nDCG is comparable across datasets.",
                "It is normalized within each query, so it is comparable across models on the same data and not across different query sets.",
            ),
            (
                "A high MRR means the ranking is good.",
                "MRR only looks at the first relevant hit. A list that nails position one and then collapses scores well on MRR and poorly on nDCG.",
            ),
        ),
        example=(
            "report = session.ranking.evaluate(partition='test', k=10)",
            "print(report.ndcg_at_k, report.map_at_k, report.mrr_at_k)",
            "print(report.n_queries)",
        ),
        check=(
            "How many held-out queries produced your averages?",
            "Does K match what a user actually sees?",
        ),
        tools=("evaluate_ranker", "rank", "fit_ranker"),
        terms=("nDCG", "MRR", "precision", "learning to rank"),
        difficulty=CORE,
    ),
    _layer(
        "ltr-bundle-boundary",
        plain=(
            "The fitted ranker saves as its own bundle holding the scoring model, the feature contract, and "
            "the query configuration. It is distinct from Session checkpoints, from RAG bundles, and from "
            "recommender bundles: all of which also produce ordered lists."
        ),
        analogy=(
            "Three departments all produce shortlists. Their filing systems are separate because their "
            "inputs, their rules, and their reviewers are different."
        ),
        steps=(
            "Fit a ranker so a plan exists.",
            "Call `session.ranking.save_bundle(path)`.",
            "Reload with `session.ranking.load_bundle(path)` in your serving path.",
            "Call `session.ranking.rank` with fresh query-item rows carrying the same features.",
            "Keep checkpoints and other domain bundles separate.",
        ),
        use=(
            "When ranking is served online and must reproduce the validated ordering exactly.",
            "When the query column and feature list need to travel with the model.",
        ),
        avoid=(
            "Do not confuse it with a RAG bundle; RAG retrieves passages by embedding similarity, a ranker scores supplied candidates with supplied features.",
            "Do not serve a ranker against features computed differently from training.",
        ),
        myths=(
            (
                "A ranker and a recommender are the same thing.",
                "A ranker orders candidates you supply, using features you supply. A recommender generates the candidates itself from interaction history.",
            ),
            (
                "The bundle can rank any list.",
                "It needs the exact feature columns it was trained on. A different feature set silently produces a meaningless ordering.",
            ),
        ),
        example=(
            "session.ranking.save_bundle('artifacts/search-ranker')",
            "serving = Session.ingest(candidate_frame).ranking.load_bundle('artifacts/search-ranker')",
            "serving.rank(query_ids=['q_17'])",
        ),
        check=(
            "Does your serving path compute features identically to training?",
            "Which surface generates your candidates before the ranker sees them?",
        ),
        tools=("save_ranker_bundle", "load_ranker_bundle", "rank", "checkpoint_save"),
        terms=("bundle", "checkpoint", "learning to rank", "RAG", "recommender"),
        difficulty=CORE,
    ),
)

__all__ = ["RANKING_BEGINNER"]
