# ruff: noqa: E501
"""Beginner layers for recommender systems."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

RECOMMENDER_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "recommender-collaborative-filtering",
        plain=(
            "Collaborative filtering recommends by pattern-matching across people. It does not need to know "
            "anything about what the items *are*: only who interacted with what. If people who liked the "
            "things you liked also liked something else, that something else gets recommended to you."
        ),
        analogy=(
            "A friend saying 'people with your taste in books loved this one'. They have not read the book "
            "either; they are going purely on whose shelves it appears on."
        ),
        steps=(
            "Build a table of interactions: who, what, and optionally how much they liked it.",
            "Split so that some interactions are held out for evaluation.",
            "Fit on training interactions: either neighbourhood-style (find similar users or items) or matrix factorization (learn hidden traits).",
            "Ask for the top-K items for a user; candidates come from the training catalogue.",
            "Evaluate against the held-out interactions to see whether the real items surfaced.",
        ),
        use=(
            "When you have plenty of interaction history and item descriptions are unavailable or unhelpful.",
            "When you want the system to discover unexpected connections that no attribute would reveal.",
        ),
        avoid=(
            "Do not use it for brand-new users or brand-new items; with no interactions there is nothing to match on.",
            "Do not use it when interactions are extremely sparse: a handful of interactions per user gives the model almost nothing.",
        ),
        myths=(
            (
                "The recommender understands the items.",
                "It has never seen a title or a description. Everything it knows comes from co-occurrence patterns in the interaction table.",
            ),
            (
                "More recommendations means better coverage.",
                "Longer lists inflate recall and bury the good suggestions. The metric improves while the user experience gets worse.",
            ),
        ),
        example=(
            "session.fit_recommender(",
            "    method='matrix_factorization', user_column='user_id',",
            "    item_column='product_id', rating_column='rating', n_factors=32,",
            ")",
            "top = session.recommend(user_ids=['u_1042'], k=10)",
        ),
        check=(
            "How many interactions does your median user have?",
            "What happens in your product when a brand-new user arrives?",
        ),
        tools=("fit_recommender", "recommend", "evaluate_recommender"),
        terms=("recommender", "collaborative filtering", "matrix factorization", "implicit feedback"),
        difficulty=CORE,
    ),
    _layer(
        "recommender-content-based",
        plain=(
            "Content-based recommendation works from what items *are* rather than who liked them. It builds "
            "a profile of each user by averaging the features of items they rated well, then scores new "
            "items by how close they are to that profile."
        ),
        analogy=(
            "A shop assistant who notices you keep buying dark roast coffee and suggests another dark "
            "roast. They are reasoning about the product, not about other customers."
        ),
        steps=(
            "Assemble numeric features describing each item: price, category flags, attributes, embeddings.",
            "For each user, take the items they interacted with in training and average their features, weighted by rating.",
            "That average is the user profile.",
            "Score candidate items by similarity to the profile.",
            "Return the top-K most similar items the user has not already seen.",
        ),
        use=(
            "When new items appear constantly and have no interaction history yet: content features work from day one.",
            "When you can explain a recommendation by pointing at an attribute, which matters in regulated or trust-sensitive settings.",
        ),
        avoid=(
            "Do not use it when your item features are thin or uninformative; averaging noise gives you a noise profile.",
            "Do not expect serendipity: it recommends more of what the user already chose, which can feel narrow over time.",
        ),
        myths=(
            (
                "Content-based is simply worse than collaborative filtering.",
                "It handles new items far better and is much easier to explain. The two fail in opposite situations, which is why hybrids exist.",
            ),
            (
                "A user profile captures taste.",
                "It captures the average of what they engaged with. A user with two distinct interests gets an average profile that matches neither.",
            ),
        ),
        example=(
            "session.fit_recommender(",
            "    method='content', user_column='user_id', item_column='product_id',",
            "    rating_column='rating', item_feature_columns=['price', 'is_organic', 'category_id'],",
            ")",
            "session.recommend(user_ids=['u_1042'], k=10)",
        ),
        check=(
            "Do your item features actually distinguish items a user would treat differently?",
            "Would a user with two unrelated interests be served well by one averaged profile?",
        ),
        tools=("fit_recommender", "recommend", "evaluate_recommender"),
        terms=("recommender", "cold start", "embedding", "feature"),
        difficulty=CORE,
    ),
    _layer(
        "recommender-ranking-metrics",
        plain=(
            "Recommender quality is measured on the top-K list, not on individual predictions. Precision@K "
            "asks what fraction of your K suggestions were good. Recall@K asks what fraction of the good "
            "items you managed to surface. nDCG@K additionally rewards putting the best ones first."
        ),
        analogy=(
            "Judging a shortlist of ten candidates. How many were suitable (precision), how many of the "
            "suitable people made the list at all (recall), and were the strongest ones at the top (nDCG)?"
        ),
        steps=(
            "Hold out some interactions the model never trained on.",
            "For each user, generate the top-K recommendations from the training catalogue.",
            "Compare against that user's held-out items: those are the known positives.",
            "Compute precision@K, recall@K, nDCG@K, and MAP@K averaged across users.",
            "Fix K to the number your interface actually shows, and keep it fixed across comparisons.",
        ),
        use=(
            "For every recommender comparison: offline ranking metrics are the standard first filter.",
            "When tuning the number of factors, the similarity measure, or the candidate generation step.",
        ),
        avoid=(
            "Do not read offline metrics as predicted business impact; users cannot click items you never showed them, so held-out data is systematically incomplete.",
            "Do not compare precision@5 with precision@20 as if they measured the same thing.",
        ),
        myths=(
            (
                "An item the user did not interact with is a negative example.",
                "It usually means they never saw it. This missing-not-at-random problem is the central difficulty of offline recommender evaluation.",
            ),
            (
                "Better offline metrics mean better online results.",
                "The correlation is real but weak. Offline evaluation cannot measure novelty, diversity, or the effect of showing something new.",
            ),
        ),
        example=(
            "report = session.evaluate_recommender(partition='test', k=10)",
            "print(report.precision_at_k, report.recall_at_k, report.ndcg_at_k)",
            "print(report.n_warm_users, report.n_cold_users)",
        ),
        check=(
            "Does your K match what the interface actually displays?",
            "How many users had any held-out interactions at all?",
        ),
        tools=("evaluate_recommender", "recommend", "fit_recommender"),
        terms=("recommender", "precision", "recall", "nDCG", "implicit feedback"),
        difficulty=CORE,
    ),
    _layer(
        "recommender-cold-start",
        plain=(
            "Cold start is the problem of serving someone or something you have no history for. A brand-new "
            "user has no interactions to match on; a brand-new item has never been interacted with. BuildML "
            "discloses which users and items were cold rather than quietly scoring them badly."
        ),
        analogy=(
            "A regular walks in and the barista knows their order. A first-time visitor gets a guess based "
            "on what is popular: and the shop should be honest that it is a guess."
        ),
        steps=(
            "After fitting, BuildML records which users and items appeared in training.",
            "At recommendation time, any user or item outside that set is marked cold.",
            "Candidates always come from the training item catalogue: a never-seen item cannot be scored by collaborative filtering.",
            "Evaluation reports warm and cold counts separately so a good warm score cannot hide total cold failure.",
            "Handle cold cases explicitly: popularity fallback, content-based features, or an onboarding step.",
        ),
        use=(
            "Always. Every real recommender faces cold start continuously as users and inventory turn over.",
            "Especially in marketplaces and catalogues where new items appear daily.",
        ),
        avoid=(
            "Do not evaluate only on warm users and report the number as system performance.",
            "Do not fall back silently to popularity without recording that you did; the behaviour difference matters for diagnosis.",
        ),
        myths=(
            (
                "Cold start is a small edge case.",
                "In a growing product, a large share of daily traffic is new users and new items. It is often the majority of the hard cases.",
            ),
            (
                "More data solves cold start.",
                "More historical data does nothing for a user who signed up this morning. Only content features, context, or an onboarding flow help.",
            ),
        ),
        example=(
            "report = session.evaluate_recommender(partition='test', k=10)",
            "print(report.n_cold_users, report.n_cold_items)",
            "print(report.disclosures)   # candidate set is the train catalogue",
        ),
        check=(
            "What fraction of yesterday's requests came from users unseen in training?",
            "What does your system return for a completely new user?",
        ),
        tools=("evaluate_recommender", "recommend", "fit_recommender"),
        terms=("cold start", "recommender", "collaborative filtering", "disclosure"),
        difficulty=CORE,
    ),
    _layer(
        "recommender-industry-backends",
        plain=(
            "With the optional recommenders extra installed, BuildML can delegate to established libraries: "
            "`implicit` for large sparse implicit-feedback data, and LightFM for hybrid models that mix "
            "interactions with item and user features."
        ),
        analogy=(
            "Swapping a hand tool for a power tool once the job gets big enough. Same task, far more "
            "throughput, and a few more ways to injure yourself if you skip the manual."
        ),
        steps=(
            "Install `pip install buildml[recommenders-industry]`.",
            "For implicit feedback: clicks, views, plays: the `implicit` backend with ALS or BPR becomes the default.",
            "For hybrid models that need side features, use LightFM.",
            "BuildML still controls the split boundary and the candidate catalogue.",
            "Evaluate with the same top-K metrics so results stay comparable with the native backend.",
        ),
        use=(
            "When your interaction matrix is large and sparse, where these implementations are dramatically faster.",
            "When you have useful user or item side features and want them in the same model as the interactions.",
        ),
        avoid=(
            "Do not install the extra for a small dataset; the native path is adequate and has fewer moving parts.",
            "Do not use an implicit-feedback model on explicit star ratings without thinking: the loss functions assume different things about what a non-interaction means.",
        ),
        myths=(
            (
                "Implicit and explicit feedback are interchangeable inputs.",
                "Explicit ratings carry a stated negative ('I gave it one star'). Implicit data has no negatives at all, only absences, and the algorithms differ accordingly.",
            ),
            (
                "A hybrid model is always better than a pure one.",
                "Only when the side features carry signal. Weak features add parameters and noise to a model that was doing fine on interactions.",
            ),
        ),
        example=(
            "# pip install \"buildml[recommenders-industry]\"",
            "session.fit_recommender(",
            "    method='als', feedback='implicit',",
            "    user_column='user_id', item_column='product_id', n_factors=64,",
            ")",
            "session.evaluate_recommender(partition='test', k=10)",
        ),
        check=(
            "Is your feedback implicit (clicks) or explicit (ratings)?",
            "Do your side features add anything the interaction history does not already capture?",
        ),
        tools=("fit_recommender", "recommend", "evaluate_recommender"),
        terms=("implicit feedback", "matrix factorization", "recommender", "extra"),
        difficulty=ADVANCED,
    ),
    _layer(
        "recommender-bundle-boundary",
        plain=(
            "A recommender bundle stores the fitted plan: the learned factors or similarity structure, the "
            "user and item vocabularies, and the candidate catalogue. It is distinct from Session "
            "checkpoints, from RAG bundles, and from the EDA recommendations that share the word."
        ),
        analogy=(
            "Three different things called 'recommendations': what to watch tonight, what a search engine "
            "retrieves, and what your accountant advises. Same word, entirely different filing cabinets."
        ),
        steps=(
            "Fit a recommender so a plan exists.",
            "Call `save_recommender_bundle(path)`.",
            "Reload with `load_recommender_bundle(path)` in the serving job.",
            "Call `recommend` with user identifiers; the catalogue comes from the bundle.",
            "Keep checkpoints separate for the interaction data itself.",
        ),
        use=(
            "When recommendations are served from a job that never sees your notebook.",
            "When the item catalogue must be pinned to what the model was trained on.",
        ),
        avoid=(
            "Do not confuse this with EDA `Recommendation` findings, which are workflow advice and mutate nothing.",
            "Do not serve from a bundle whose catalogue is months stale without measuring what it is missing.",
        ),
        myths=(
            (
                "A recommender bundle is a kind of RAG index.",
                "Both retrieve, but one ranks catalogue items for a user and the other retrieves text passages for a query. Different contracts, different loaders.",
            ),
            (
                "The bundle contains the interaction history.",
                "It contains the fitted plan and vocabularies. The raw interactions stay in your data store.",
            ),
        ),
        example=(
            "session.save_recommender_bundle('artifacts/product-recs')",
            "serving = Session.ingest(users_frame).load_recommender_bundle('artifacts/product-recs')",
            "serving.recommend(user_ids=batch_ids, k=10)",
        ),
        check=(
            "How old is the catalogue inside your deployed bundle?",
            "Which of the three 'recommendation' concepts is your colleague asking about?",
        ),
        tools=("save_recommender_bundle", "load_recommender_bundle", "recommend", "checkpoint_save"),
        terms=("bundle", "checkpoint", "recommender", "RAG"),
        difficulty=CORE,
    ),
)

__all__ = ["RECOMMENDER_BEGINNER"]
