"""Finite recommender memory and saved rankings match raw train arithmetic."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError

MASK = np.random.default_rng(412).uniform(size=(6, 7)) > 0.35
MASK[:, 1] = True
MASK[1, 0] = True


def frame():
    rng = np.random.default_rng(915)
    side = rng.normal(size=(7, 3))
    rows = [
        dict(
            user=u,
            item=i,
            rating=float(rng.uniform(0.2, 4.4)),
            a=side[i, 0],
            b=side[i, 1],
            c=side[i, 2],
        )
        for u in range(6)
        for i in range(7)
    ]
    rows[7]["rating"] = 0.0
    rows[15]["rating"] = -1.1
    rows.append({**rows[1], "rating": 2.1})
    rows.extend([{**rows[2], "user": 99}, {**rows[2], "item": 99}])
    return pd.DataFrame(rows, index=1000 + 3 * np.arange(len(rows)))


def ready(raw=None):
    raw = frame() if raw is None else raw
    train = [
        i
        for i, row in enumerate(raw.itertuples())
        if row.user < 6 and row.item < 7 and MASK[row.user, row.item]
    ]
    other = [i for i in range(len(raw)) if i not in train]
    return (
        Session.ingest(raw)
        .set_roles(
            {
                "user": "id",
                "item": "id",
                "rating": "target",
                "a": "feature",
                "b": "feature",
                "c": "feature",
            }
        )
        .inject_split(
            train_indices=train,
            validation_indices=other[: len(other) // 2],
            test_indices=other[len(other) // 2 :],
        )
    )


def fit(s, method="content", **options):
    options.setdefault("cold_start", "popularity")
    return s.recommender.fit(
        backend="sklearn",
        method=method,
        user_column="user",
        item_column="item",
        rating_column="rating",
        item_feature_columns=["a", "b", "c"],
        random_state=9,
        **options,
    )


def raw_memory(train, feedback, users=None, items=None):
    groups = {}
    for row in train.itertuples():
        groups.setdefault((row.user, row.item), []).append(float(row.rating))
    users = sorted({u for u, _ in groups}) if users is None else list(users)
    items = sorted({i for _, i in groups}) if items is None else list(items)
    matrix, observed = (
        np.zeros((len(users), len(items))),
        np.zeros((len(users), len(items)), dtype=bool),
    )
    for (u, i), ratings in groups.items():
        matrix[users.index(u), items.index(i)] = (
            1.0 if feedback == "implicit" else sum(ratings) / len(ratings)
        )
        observed[users.index(u), items.index(i)] = True
    return matrix, observed, users, items


def cosine(x):
    norm = np.linalg.norm(x, axis=1)
    return (x @ x.T) / (norm[:, None] * norm[None, :])


def raw_scores(method, matrix, observed, features, neighbors, factors, feedback):
    if method == "svd":
        mu = matrix[observed].mean() if feedback == "explicit" else 0.0
        centered = np.where(observed, matrix - mu, 0.0)
        left, singular, right = np.linalg.svd(centered, full_matrices=False)
        assert singular[factors - 1] - singular[factors] > 1e-6
        return (left[:, :factors] * singular[:factors]) @ right[:factors] + mu
    if method == "content":
        result = []
        for row in matrix:
            used = row != 0
            weights = np.abs(row[used])
            profile = (features[used] * weights[:, None]).sum(axis=0) / weights.sum()
            result.append(
                features @ profile / (np.linalg.norm(features, axis=1) * np.linalg.norm(profile))
            )
        return np.asarray(result)
    similarity = cosine(matrix.T if method == "item_knn" else matrix)
    result = np.zeros_like(matrix)
    for user in range(len(matrix)):
        if method == "item_knn":
            rated = np.flatnonzero(observed[user])
            for item in range(matrix.shape[1]):
                selected = sorted(rated, key=lambda i: -abs(similarity[item, i]))[:neighbors]
                weights = similarity[item, selected]
                if np.abs(weights).sum() > 1e-12:
                    result[user, item] = (
                        np.dot(weights, matrix[user, selected]) / np.abs(weights).sum()
                    )
        else:
            selected = sorted(
                [i for i in range(len(matrix)) if i != user],
                key=lambda i: -abs(similarity[user, i]),
            )[:neighbors]
            weights = similarity[user, selected]
            for item in range(matrix.shape[1]):
                denominator = np.dot(np.abs(weights), observed[selected, item])
                if denominator > 1e-12:
                    result[user, item] = np.dot(weights, matrix[selected, item]) / denominator
    return result


@pytest.mark.parametrize("method", ["item_knn", "user_knn", "svd", "content"])
@pytest.mark.parametrize("recipe", ["explicit-one", "explicit-three", "implicit-all"])
def test_real_rankings_catalog_exclusion_and_bundles_match_raw_matrix_arithmetic(
    method, recipe, tmp_path
):
    s = ready()
    original = s.dataset.to_pandas().copy(deep=True)
    feedback = "implicit" if recipe == "implicit-all" else "explicit"
    neighbors = 99 if feedback == "implicit" else 1 if recipe == "explicit-one" else 3
    factors = 2 if feedback == "implicit" else neighbors
    fit(s, method, feedback=feedback, n_neighbors=np.int64(neighbors), n_factors=np.int64(factors))
    plan = s.recommender_plan
    train = s.partition("train")
    assert set(plan.user_ids) == set(train.user) and set(plan.item_ids) == set(train.item)
    matrix, observed, users, items = raw_memory(train, feedback, plan.user_ids, plan.item_ids)
    assert plan.user_index_ == {u: i for i, u in enumerate(users)}
    assert plan.item_index_ == {v: i for i, v in enumerate(items)}
    assert type(plan.n_neighbors) is type(plan.n_factors) is int
    np.testing.assert_allclose(plan.matrix_, matrix)
    np.testing.assert_array_equal(plan.observed_mask_, observed)
    train = s.partition("train")
    features = np.asarray(
        [train.loc[train.item == i, ["a", "b", "c"]].to_numpy().mean(axis=0) for i in items]
    )
    mean, scale = features.mean(axis=0), features.std(axis=0)
    features = (features - mean) / scale
    if method == "content":
        np.testing.assert_allclose(plan.item_feature_mean_, mean)
        np.testing.assert_allclose(plan.item_feature_scale_, scale)
        np.testing.assert_allclose(plan.item_features_, features)
    expected = raw_scores(method, matrix, observed, features, neighbors, factors, feedback)
    saved = s.recommender.save_bundle(tmp_path / "memory")
    restored = ready()
    restored.recommender.load_bundle(saved, trusted=True)
    for owner in [s, restored]:
        for exclude in [False, True]:
            result = owner.recommender.recommend(
                user_ids=np.asarray([0, 4, 99]),
                k=np.int64(99),
                exclude_train_items=np.bool_(exclude),
            )
            for u, ids, scores in zip(result.user_ids, result.recommendations, result.scores):
                candidates = [
                    i
                    for i in items
                    if u == 99 or not exclude or not observed[users.index(u), items.index(i)]
                ]
                assert set(ids) == set(candidates) and len(ids) == len(candidates)
                wanted = observed.sum(axis=0) if u == 99 else expected[users.index(u)]
                np.testing.assert_allclose(
                    scores, [wanted[items.index(i)] for i in ids], atol=2e-9, rtol=2e-9
                )
                assert all(a >= b - 1e-12 for a, b in zip(scores, scores[1:]))
            assert result.cold_start_users == (99,)
            json.dumps(result.to_dict(), allow_nan=False)
        evaluation = owner.recommender.evaluate(partition="test", k=99)
        holdout = owner.partition("test")
        relevant = {}
        for row in holdout.itertuples():
            if row.user in users and row.item in items:
                relevant.setdefault(row.user, set()).add(row.item)
        expected_precision = np.mean([len(v) / 99 for v in relevant.values()])
        assert evaluation.metrics["precision_at_k"] == pytest.approx(expected_precision)
        assert evaluation.metrics["recall_at_k"] == 1.0
        assert evaluation.n_users_scored == len(relevant) and evaluation.n_cold_start_users == 1
    pd.testing.assert_frame_equal(s.dataset.to_pandas(), original)


@pytest.fixture
def valid_bundle(tmp_path):
    s = ready()
    fit(s, n_neighbors=3, n_factors=2)
    return s.recommender.save_bundle(tmp_path / "valid")


def state(s):
    s.recommender.recommend(user_ids=[0], k=3, exclude_train_items=False)
    plan = s.recommender_plan
    return (
        plan,
        joblib.hash(plan),
        s.history,
        s._recommender_fit_result,
        s._recommender_eval_result,
        s._recommender_recommend_result,
        s.dataset.to_pandas().copy(deep=True),
    )


def unchanged(s, before):
    plan, fingerprint, history, fitted, evaluated, recommended, original = before
    assert s.recommender_plan is plan and joblib.hash(plan) == fingerprint
    assert s.history == history and s._recommender_fit_result is fitted
    assert (
        s._recommender_eval_result is evaluated and s._recommender_recommend_result is recommended
    )
    pd.testing.assert_frame_equal(s.dataset.to_pandas(), original)


@pytest.mark.parametrize("method", ["item_knn", "user_knn", "svd", "nmf", "content"])
@pytest.mark.parametrize(
    "bad", [np.inf, -np.inf, 2 + 3j], ids=["infinite", "negative-infinite", "complex"]
)
def test_invalid_explicit_ratings_refuse_before_replacing_loaded_memory(method, bad, valid_bundle):
    raw = frame()
    if isinstance(bad, complex):
        raw["rating"] = raw.rating.astype(complex)
    raw.loc[1003, "rating"] = bad
    s = ready(raw)
    s.recommender.load_bundle(valid_bundle, trusted=True)
    before = state(s)
    with pytest.raises(ValidationError, match="rating.*finite|rating.*real"):
        fit(s, method)
    unchanged(s, before)


@pytest.mark.parametrize(
    "bad",
    [np.inf, -np.inf, np.nan, 2 + 3j, "bad", pd.NA],
    ids=["infinite", "negative-infinite", "nan", "complex", "text", "nullable"],
)
def test_invalid_content_side_features_refuse_before_replacing_loaded_memory(bad, valid_bundle):
    raw = frame()
    if isinstance(bad, complex):
        raw["a"] = raw.a.astype(complex)
    elif isinstance(bad, str):
        raw["a"] = raw.a.astype(object)
    elif bad is pd.NA:
        raw["a"] = raw.a.astype("Float64")
    raw.loc[1003, "a"] = bad
    s = ready(raw)
    s.recommender.load_bundle(valid_bundle, trusted=True)
    before = state(s)
    with pytest.raises(ValidationError, match="feature.*numeric|feature.*finite|feature.*real"):
        fit(s)
    unchanged(s, before)


@pytest.mark.parametrize("name", ["n_neighbors", "n_factors"])
@pytest.mark.parametrize("bad", [True, np.bool_(False), 1.5, "3", np.nan, np.inf, 0, -2])
def test_training_dimensions_require_positive_integral_counts_and_preserve_prior_evidence(
    name, bad
):
    s = ready()
    method = "item_knn" if name == "n_neighbors" else "svd"
    fit(s, method)
    s.recommender.evaluate(partition="test")
    before = state(s)
    with pytest.raises(ValidationError, match=name):
        fit(s, method, **{name: bad})
    unchanged(s, before)


@pytest.mark.parametrize("missing", [None, "not numeric"])
def test_documented_numeric_coercion_null_drops_and_duplicate_means_remain_supported(missing):
    raw = frame()
    raw["rating"] = raw.rating.map(str).astype(object)
    raw.loc[1003, "rating"] = missing
    s = ready(raw)
    fit(s, "item_knn")
    train = s.partition("train")
    clean = train[pd.to_numeric(train.rating, errors="coerce").notna()].copy()
    clean["rating"] = pd.to_numeric(clean.rating)
    matrix, observed, _, _ = raw_memory(
        clean, "explicit", s.recommender_plan.user_ids, s.recommender_plan.item_ids
    )
    np.testing.assert_allclose(s.recommender_plan.matrix_, matrix)
    np.testing.assert_array_equal(s.recommender_plan.observed_mask_, observed)


@pytest.mark.parametrize("ignored", [np.inf, 2 + 3j, np.nan])
def test_implicit_feedback_ignores_rating_values_instead_of_applying_explicit_validation(ignored):
    raw = frame()
    if isinstance(ignored, complex):
        raw["rating"] = raw.rating.astype(complex)
    raw.loc[1003, "rating"] = ignored
    s = ready(raw)
    fit(s, "item_knn", feedback="implicit")
    np.testing.assert_array_equal(
        s.recommender_plan.matrix_, s.recommender_plan.observed_mask_.astype(float)
    )


@pytest.mark.parametrize("column", ["rating", "a"])
def test_nonfinite_holdout_values_never_enter_the_train_memory_or_transforms(column):
    owner = ready()
    raw = frame()
    for position in [*owner.split_plan.validation_indices, *owner.split_plan.test_indices]:
        raw.iloc[position, raw.columns.get_loc(column)] = np.inf
    candidate = ready(raw)
    fit(owner)
    fit(candidate)
    assert joblib.hash(owner.recommender_plan) == joblib.hash(candidate.recommender_plan)


@pytest.mark.parametrize("population", ["cold-users", "unknown-items"])
def test_loaded_plan_with_no_scoreable_holdout_reports_zero_population_explicitly(
    population, valid_bundle
):
    s = ready()
    raw = frame()
    for position in s.split_plan.test_indices:
        raw.iloc[
            position, raw.columns.get_loc("user" if population == "cold-users" else "item")
        ] = 555
    newcomer = Session.ingest(raw).set_roles({"user": "id", "item": "id", "rating": "target"})
    newcomer.inject_split(
        train_indices=s.split_plan.train_indices,
        validation_indices=s.split_plan.validation_indices,
        test_indices=s.split_plan.test_indices,
    )
    newcomer.recommender.load_bundle(valid_bundle, trusted=True)
    result = newcomer.recommender.evaluate(partition="test", k=3)
    assert result.n_users_scored == 0 and all(value == 0.0 for value in result.metrics.values())
    assert any("No warm users" in text for text in result.disclosures)
    holdout = newcomer.partition("test")
    assert result.n_holdout_interactions == len(set(zip(holdout.user, holdout.item)))
    json.dumps(result.to_dict(), allow_nan=False)
