"""Frozen public consumers read native rows, including after lazy projections."""

from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session

BACKINGS = [
    ("pandas", "memory"),
    ("duckdb", "cold"),
    ("duckdb", "warm"),
    ("duckdb", "filtered"),
    ("polars", "cold"),
    ("polars", "warm"),
    ("polars", "filtered"),
]


def _split(session, n):
    session.inject_split(
        train_indices=list(range(n - 8)),
        validation_indices=list(range(n - 8, n - 4)),
        test_indices=list(range(n - 4, n)),
    )
    return session


@contextmanager
def _population(frame, backing, tmp_path, roles=None, split=False):
    engine, cache = backing
    if engine != "pandas":
        pytest.importorskip(engine)
    source = frame.reset_index(drop=True)
    path = tmp_path / "rows.parquet"
    source.to_parquet(path, index=False)
    owner = Session.ingest(
        source if engine == "pandas" else path,
        engine=engine,
        mode="memory" if engine == "pandas" else "lazy",
    )
    try:
        session, expected = owner, source
        if cache == "filtered":
            # Real derived native handles retain their owner's connection.
            owner.to_pandas()
            mask = [i % 4 != 1 for i in range(len(source))]
            columns = list(source.columns)[::-1]
            derived = owner.dataset.filter_rows(mask).project(columns)
            session = Session(dataset=derived)
            expected = source.loc[mask, columns].reset_index(drop=True)
        elif cache == "warm":
            owner.to_pandas()
        if cache in {"cold", "filtered"}:
            assert session.dataset.pandas_stale and session.dataset.frame.empty
        if roles:
            session.set_roles(roles)
        if split:
            _split(session, len(expected))
        native = session.dataset.native
        yield session, expected
        if engine != "pandas" and session.dataset.native is native:
            # Reads must not close or replace an owned/shared native relation.
            assert session.dataset.n_rows == len(expected)
            assert owner.dataset.n_rows == len(source)
    finally:
        owner.close_native()
        if session.dataset is not owner.dataset:
            session.close_native()


def _ranking_data():
    rng = np.random.default_rng(144)
    x = rng.normal(size=(32, 2))
    return pd.DataFrame(
        {
            "query": np.repeat(["q0", "q1", "q2", "q3"], 8),
            "item": [f"i{i}" for i in range(32)],
            "x": x[:, 0],
            "z": x[:, 1],
            "grade": (2 * x[:, 0] - x[:, 1] > 0).astype(float) * 3,
        }
    )


RANK_ROLES = {"query": "group", "item": "id", "x": "feature", "z": "feature", "grade": "target"}


def _fit_ranker(session, method, automatic=False):
    return session.ranking.fit(
        backend="sklearn",
        method=method,
        query_column="query",
        item_column="item",
        relevance_column="grade",
        feature_columns=None if automatic else ["x", "z"],
        alpha=0.7,
        C=0.4,
        random_state=8,
    )


@pytest.fixture(scope="module", params=["pointwise", "pairwise"])
def ranker(request, tmp_path_factory):
    frame = _ranking_data()
    source = _split(Session.ingest(frame).set_roles(RANK_ROLES), len(frame))
    _fit_ranker(source, request.param)
    bundle = source.ranking.save_bundle(tmp_path_factory.mktemp("native-ranker") / "bundle")
    return source, frame, bundle


def _rank_scores(source, rows):
    train = _ranking_data().iloc[:24]
    mean, scale = train[["x", "z"]].mean().to_numpy(), train[["x", "z"]].std(ddof=0).to_numpy()
    x = (rows[["x", "z"]].to_numpy() - mean) / scale
    if source.ranker_plan.method == "pointwise":
        design = (train[["x", "z"]].to_numpy() - mean) / scale
        y = train.grade.to_numpy()
        coef = np.linalg.solve(design.T @ design + 0.7 * np.eye(2), design.T @ (y - y.mean()))
        return x @ coef + y.mean()
    state = source.ranker_plan.estimator_
    return x @ state.coef_ + state.intercept_


@pytest.mark.parametrize("backing", BACKINGS)
@pytest.mark.parametrize("selection", ["all", "explicit_queries"])
def test_ranker_bundle_reads_whole_native_query_population(ranker, backing, selection, tmp_path):
    source, frame, bundle = ranker
    with _population(frame, backing, tmp_path) as (session, rows):
        session.ranking.load_bundle(bundle, trusted=True)
        before = joblib.hash(session.ranker_plan)
        queries = list(dict.fromkeys(rows["query"]))
        arguments = {"partition": "all"} if selection == "all" else {"query_ids": queries[::-1]}
        result = session.ranking.rank(**arguments, k=99)
        scores = _rank_scores(source, rows)
        assert list(result.query_ids) == (queries if selection == "all" else queries[::-1])
        for query, items, values in zip(
            result.query_ids, result.rankings, result.scores, strict=True
        ):
            indices = [i for i, value in enumerate(rows["query"]) if value == query]
            assert set(items) == set(rows["item"].iloc[indices])
            expected = dict(zip(rows["item"], scores, strict=True))
            np.testing.assert_allclose(values, [expected[item] for item in items], atol=1e-10)
            assert list(values) == sorted(values, reverse=True)
        assert joblib.hash(session.ranker_plan) == before
        pd.testing.assert_frame_equal(session.to_pandas(), rows, check_dtype=False)


@pytest.mark.parametrize("engine", ["pandas", "duckdb", "polars"])
@pytest.mark.parametrize("method", ["pointwise", "pairwise"])
@pytest.mark.parametrize("automatic", [False, True])
def test_ranker_can_fit_numeric_roles_directly_from_native_ingest(
    engine, method, automatic, tmp_path
):
    frame = _ranking_data()
    backing = (engine, "memory" if engine == "pandas" else "cold")
    with _population(frame, backing, tmp_path, RANK_ROLES, split=True) as (session, rows):
        _fit_ranker(session, method, automatic)
        plan = session.ranker_plan
        assert plan.feature_columns == ("x", "z")
        np.testing.assert_allclose(plan.feature_mean_, frame.iloc[:24][["x", "z"]].mean())
        np.testing.assert_allclose(plan.feature_scale_, frame.iloc[:24][["x", "z"]].std(ddof=0))
        result = session.ranking.rank(partition="all", k=99)
        assert sum(map(len, result.rankings)) == len(rows)


@pytest.fixture(scope="module", params=["item_knn", "svd"])
def recommender(request, tmp_path_factory):
    frame = pd.DataFrame(
        [(f"u{u}", f"i{i}", float((u + i) % 5)) for u in range(4) for i in range(6)],
        columns=["user", "item", "rating"],
    )
    source = _split(
        Session.ingest(frame).set_roles({"user": "id", "item": "id", "rating": "target"}),
        len(frame),
    )
    source.recommender.fit(
        backend="sklearn",
        method=request.param,
        user_column="user",
        item_column="item",
        rating_column="rating",
        n_factors=2,
        random_state=2,
    )
    bundle = source.recommender.save_bundle(
        tmp_path_factory.mktemp("native-recommender") / "bundle"
    )
    # Repeated, null and cold users must not remove valid native users or duplicate results.
    candidates = pd.DataFrame(
        {"user": ["cold", "u1", "u0", "u1", None, "u2"], "item": ["i0"] * 6, "rating": [1.0] * 6}
    )
    return source, candidates, bundle


@pytest.mark.parametrize("backing", BACKINGS)
def test_recommender_bundle_scores_unique_nonnull_native_users(recommender, backing, tmp_path):
    source, candidates, bundle = recommender
    with _population(candidates, backing, tmp_path) as (session, rows):
        session.recommender.load_bundle(bundle, trusted=True)
        before = joblib.hash(session.recommender_plan)
        users = list(dict.fromkeys(value for value in rows.user if pd.notna(value)))
        result = session.recommender.recommend(partition="all", k=99, exclude_train_items=False)
        assert list(result.user_ids) == users
        plan = session.recommender_plan
        matrix = np.zeros_like(plan.matrix_)
        observed = np.zeros_like(matrix, dtype=bool)
        for row in source.partition("train").itertuples():
            u, i = plan.user_index_[row.user], plan.item_index_[row.item]
            matrix[u, i], observed[u, i] = row.rating, True
        for user, items, values in zip(
            result.user_ids, result.recommendations, result.scores, strict=True
        ):
            if user not in plan.user_index_:
                expected = observed.sum(axis=0)
            elif plan.method == "svd":
                centered = np.where(observed, matrix - matrix[observed].mean(), 0.0)
                a, b, c = np.linalg.svd(centered, full_matrices=False)
                expected = (a[plan.user_index_[user], :2] * b[:2]) @ c[:2] + matrix[observed].mean()
            else:
                vectors = matrix.T
                norms = np.linalg.norm(vectors, axis=1)
                sim = vectors @ vectors.T / np.maximum(norms[:, None] * norms[None, :], 1e-300)
                u = plan.user_index_[user]
                expected = []
                for weights in sim[:, observed[u]]:
                    denominator = np.abs(weights).sum()
                    expected.append(
                        float(weights @ matrix[u, observed[u]] / denominator)
                        if denominator > 1e-12
                        else 0.0
                    )
            assert set(items) == set(plan.item_ids)
            np.testing.assert_allclose(
                values, [expected[plan.item_index_[item]] for item in items], atol=1e-10
            )
        assert joblib.hash(session.recommender_plan) == before
        pd.testing.assert_frame_equal(session.to_pandas(), rows, check_dtype=False)


@pytest.fixture(scope="module", params=[("transe", "l1"), ("transe", "l2"), ("distmult", "l1")])
def graph(request, tmp_path_factory):
    method, norm = request.param
    frame = pd.DataFrame(
        [
            (f"e{u}", f"r{r}", f"e{(u + i) % 5}")
            for u in range(5)
            for i in range(1, 4)
            for r in range(2)
        ],
        columns=["head", "relation", "tail"],
    )
    source = _split(Session.ingest(frame).set_roles(dict.fromkeys(frame.columns, "id")), len(frame))
    source.kg.fit(
        backend="native",
        method=method,
        norm=norm,
        head_column="head",
        relation_column="relation",
        tail_column="tail",
        embedding_dim=3,
        epochs=1,
        random_state=3,
    )
    bundle = source.kg.save_bundle(tmp_path_factory.mktemp("native-kg") / "bundle")
    return source, frame, bundle


@pytest.mark.parametrize("backing", BACKINGS)
def test_kg_bundle_scores_every_native_triple_with_frozen_equations(graph, backing, tmp_path):
    _, frame, bundle = graph
    candidates = pd.concat(
        [
            frame,
            pd.DataFrame([("outside", "r0", "e1"), ("e0", "outside", "e1")], columns=frame.columns),
        ],
        ignore_index=True,
    )
    with _population(candidates, backing, tmp_path) as (session, rows):
        session.kg.load_bundle(bundle, trusted=True)
        plan = session.kg_plan
        before = joblib.hash(plan)
        result = session.kg.score_triples(partition="all")
        assert result.n_triples == len(rows)
        assert result.heads == tuple(rows["head"])
        expected = []
        for h, r, t in rows[["head", "relation", "tail"]].itertuples(index=False, name=None):
            if (
                h not in plan.entity_index_
                or t not in plan.entity_index_
                or r not in plan.relation_index_
            ):
                expected.append(-np.inf)
                continue
            h, r, t = (
                plan.entity_embeddings_[plan.entity_index_[h]],
                plan.relation_embeddings_[plan.relation_index_[r]],
                plan.entity_embeddings_[plan.entity_index_[t]],
            )
            expected.append(
                np.sum(h * r * t)
                if plan.method == "distmult"
                else -np.linalg.norm(h + r - t, ord=1 if plan.norm == "l1" else 2)
            )
        np.testing.assert_allclose(result.scores, expected, atol=1e-12)
        assert joblib.hash(plan) == before
        pd.testing.assert_frame_equal(session.to_pandas(), rows, check_dtype=False)


@pytest.fixture(scope="module", params=["topk", "knapsack", "lp_allocate"])
def allocation(request, tmp_path_factory):
    frame = pd.DataFrame(
        {
            "id": [f"a{i}" for i in range(16)],
            "value": np.arange(16) * 0.73 + 1,
            "cost": np.arange(16) % 4 + 1,
        }
    )
    source = _split(Session.ingest(frame), len(frame))
    args = {"capacity": 3} if request.param == "topk" else {"budget": 5.5}
    source.decision.fit(
        method=request.param,
        backend="native",
        partition="validation",
        score_source="column",
        score_column="value",
        cost_column="cost",
        id_column="id",
        **args,
    )
    bundle = source.decision.save_bundle(tmp_path_factory.mktemp("native-allocation") / "bundle")
    return frame, bundle


@pytest.mark.parametrize("backing", BACKINGS)
def test_allocation_bundle_uses_all_native_candidates_and_original_ids(
    allocation, backing, tmp_path
):
    import itertools

    frame, bundle = allocation
    with _population(frame, backing, tmp_path, split=True) as (session, rows):
        session.decision.load_bundle(bundle, trusted=True)
        plan = session.decision_plan
        before = joblib.hash(plan)
        result = session.decision.apply(partition="all")
        values, costs = rows.value.to_numpy(), rows.cost.to_numpy()
        fractions = np.zeros(len(rows))
        if plan.method == "topk":
            fractions[np.argsort(-values)[:3]] = 1
        elif plan.method == "knapsack":
            choices = [
                (sum(values[i] for i, bit in enumerate(bits) if bit), bits)
                for bits in itertools.product([0, 1], repeat=len(rows))
                if sum(costs[i] for i, bit in enumerate(bits) if bit) <= 5.5
            ]
            _, bits = max(choices, key=lambda pair: pair[0])
            fractions = np.asarray(bits)
        else:
            remaining = 5.5
            for i in np.argsort(-values / costs):
                fractions[i] = max(0.0, min(1.0, remaining / costs[i]))
                remaining -= fractions[i] * costs[i]
        assert result.n_rows == len(rows)
        assert set(result.selected_ids) == set(rows.id.iloc[np.flatnonzero(fractions > 1e-9)])
        assert result.selected_value == pytest.approx(float(values @ fractions))
        assert result.selected_cost == pytest.approx(float(costs @ fractions))
        assert joblib.hash(plan) == before
        pd.testing.assert_frame_equal(session.to_pandas(), rows, check_dtype=False)


@pytest.fixture(scope="module", params=["bootstrap", "gaussian_copula"])
def synthesizer(request, tmp_path_factory):
    frame = pd.DataFrame(
        {"x": np.arange(24) * 0.3, "z": np.cos(np.arange(24)), "label": ["a", "b"] * 12}
    )
    source = _split(
        Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "label": "target"}),
        len(frame),
    )
    source.synthetic.fit(method=request.param, random_state=4)
    bundle = source.synthetic.save_bundle(tmp_path_factory.mktemp("native-synthetic") / "bundle")
    return frame, bundle


@pytest.mark.parametrize("backing", BACKINGS)
def test_synthetic_bundle_merge_retains_real_native_train_and_holdouts(
    synthesizer, backing, tmp_path
):
    frame, bundle = synthesizer
    roles = {"x": "feature", "z": "feature", "label": "target"}
    with _population(frame, backing, tmp_path, roles, split=True) as (session, rows):
        session.synthetic.load_bundle(bundle, trusted=True)
        before = joblib.hash(session.synthesizer_plan)
        old_split = session.split_plan
        result = session.synthetic.sample(n=3, merge_mode="extend_train", random_state=17)
        combined = session.to_pandas()
        n_train = len(rows) - 8
        # Compare values and dtypes as well as membership; holdout bytes remain real.
        for name in ("validation", "test"):
            expected = rows.iloc[list(old_split.indices_for(name))].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                session.partition(name).drop(columns=["_synthetic"]).reset_index(drop=True),
                expected,
                check_dtype=False,
            )
            assert not session.partition(name)["_synthetic"].any()
        pd.testing.assert_frame_equal(
            combined.iloc[:n_train].drop(columns=["_synthetic"]).reset_index(drop=True),
            rows.iloc[:n_train].reset_index(drop=True),
            check_dtype=False,
        )
        assert combined["_synthetic"].sum() == 3 and len(combined) == len(rows) + 3
        pd.testing.assert_frame_equal(
            combined.iloc[n_train : n_train + 3][list(result.frame.columns)].reset_index(drop=True),
            result.frame.reset_index(drop=True),
            check_dtype=False,
        )
        assert session.dataset.roles["_synthetic"].value == "ignore"
        assert joblib.hash(session.synthesizer_plan) == before


@pytest.fixture(scope="module", params=["landscape", "silhouette"])
def topology(request, tmp_path_factory):
    pytest.importorskip("ripser")
    pytest.importorskip("persim")
    frame = pd.DataFrame(
        {"x": np.sin(np.arange(24) * 0.7), "z": np.arange(24) * 0.13, "target": [0, 1] * 12}
    )
    source = _split(
        Session.ingest(frame).set_roles({"x": "feature", "z": "feature", "target": "target"}),
        len(frame),
    )
    source.tda.fit(
        backend="native",
        vectorization=request.param,
        columns=["x", "z"],
        homology_dims=[0],
        knn=4,
        maxdim=0,
        n_bins=6,
        n_layers=2,
        head="none",
        random_state=5,
    )
    bundle = source.tda.save_bundle(tmp_path_factory.mktemp("native-tda") / "bundle")
    return frame, bundle


@pytest.mark.parametrize("backing", BACKINGS)
def test_tda_bundle_transforms_whole_native_population_with_mst_oracle(topology, backing, tmp_path):
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import cdist

    frame, bundle = topology
    with _population(frame, backing, tmp_path) as (session, rows):
        session.tda.load_bundle(bundle, trusted=True)
        plan = session.tda_plan

        # NearestNeighbors can populate internal runtime caches on a query.
        # Freeze its fitted data and the learned numerical state, not caches.
        def learned_state():
            return (
                plan.columns,
                plan.mean_,
                plan.scale_,
                plan.train_x_,
                plan.vectorizer_state_,
                plan.nn_._fit_X,
            )

        before = joblib.hash(learned_state())
        result = session.tda.transform(partition="all")
        assert result.n_rows == len(rows)
        # Finite H0 deaths equal the Euclidean MST edge lengths of each frozen
        # train-neighbour cloud. No ripser/BuildML vectorizer in the oracle.
        train = frame.iloc[:16][["x", "z"]].to_numpy()
        mean, scale = train.mean(axis=0), train.std(axis=0)
        train = (train - mean) / scale
        state = plan.vectorizer_state_
        grid = np.linspace(*state["t_range"], state["n_bins"])
        expected = []
        for row in (rows[["x", "z"]].to_numpy() - mean) / scale:
            nearest = np.argsort(np.linalg.norm(train - row, axis=1))[:4]
            deaths = minimum_spanning_tree(cdist(train[nearest], train[nearest])).data
            tents = np.maximum(
                0.0, deaths[:, None] / 2 - np.abs(grid[None, :] - deaths[:, None] / 2)
            )
            if plan.vectorization == "landscape":
                expected.append(np.sort(tents, axis=0)[::-1][:2].ravel())
            else:
                expected.append(np.sum(tents * deaths[:, None], axis=0) / deaths.sum())
        np.testing.assert_allclose(result.features, expected, atol=2e-7)
        np.testing.assert_allclose(plan.mean_, mean)
        assert joblib.hash(learned_state()) == before
        pd.testing.assert_frame_equal(session.to_pandas(), rows, check_dtype=False)
