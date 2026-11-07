"""Copied research-report calls must retain Python literals and real column names."""

import ast
from copy import deepcopy
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from buildml import Session
from buildml.eda.html_report import export_eda_html
from buildml.explain import Action, ActionPriority, Recommendation
from buildml.preprocess.custom import register_transform, unregister_transform


class TableRows(HTMLParser):
    """Read displayed cell text from the actual saved HTML artifact."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"th", "td"}:
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self.row is not None and self.cell is not None:
            self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def copied_call(report, boundary, path, title):
    if boundary == "report":
        destination = report.save_html(path, html_format="research")
    else:
        destination = export_eda_html(report.to_dict(), path, max_figures=0)
    parser = TableRows()
    parser.feed(destination.read_text(encoding="utf-8"))
    calls = [row[3] for row in parser.rows if len(row) == 5 and row[2] == title]
    assert len(calls) == 1
    tree = ast.parse(calls[0], mode="exec")
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
    call = tree.body[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    assert isinstance(call.func.value, ast.Name) and call.func.value.id == "session"
    return calls[0], call


def execute(code, session):
    exec(
        compile(code, "copied research recommendation", "exec"),
        {"session": session, "__builtins__": {}},
    )


def population():
    rng = np.random.default_rng(74)
    frame = pd.DataFrame(
        {
            "x": rng.normal(size=48),
            "y": np.tile([0, 1], 24),
            "words": ["red cat", "blue cat", "red blue", "cat cat"] * 12,
        }
    )
    return Session.ingest(frame).set_roles({"x": "feature", "y": "target", "words": "ignore"})


NAMES = [
    "ordinary",
    'double"quote',
    "single'quote",
    r"back\nslash",
    "line\nbreak",
    "carriage\rreturn",
    r"literal\u0041",
    "mixed'\"<&>",
    "café 漢字",
]


@pytest.fixture(
    scope="module", params=[(kind, name) for kind in ["constant", "identifier"] for name in NAMES]
)
def column_report(request):
    kind, name = request.param
    session = population()
    frame = session.to_pandas().drop(columns="words")
    frame[name] = (
        np.ones(len(frame)) if kind == "constant" else [f"record-{i}" for i in range(len(frame))]
    )
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target", name: "feature"})
    report = session.eda(include_plots=False, show=False)
    key = "next.drop_constants" if kind == "constant" else "next.exclude_ids"
    recommendation = next(item for item in report.recommendation_details if item.key == key)
    assert name in report.quality["constant_columns" if kind == "constant" else "id_like_columns"]
    return report, session, name, recommendation.title, "ignore" if kind == "constant" else "id"


@pytest.mark.parametrize("boundary", ["report", "export"])
def test_generated_column_advice_executes_for_the_exact_declared_name(
    column_report, boundary, tmp_path
):
    report, source, name, title, role = column_report
    before = source.to_pandas().copy(deep=True)
    consumer = Session.ingest(before).set_roles({"x": "feature", "y": "target", name: "feature"})
    source_history = source.history
    code, call = copied_call(report, boundary, tmp_path / "columns.html", title)
    assert call.func.attr == "set_roles"
    assert ast.literal_eval(call.args[0]) == {name: role}
    execute(code, consumer)
    assert consumer.dataset.roles[name].value == role
    assert consumer.dataset.roles["x"].value == "feature"
    assert consumer.dataset.roles["y"].value == "target"
    pd.testing.assert_frame_equal(consumer.to_pandas(), before)
    assert source.history == source_history


@pytest.fixture(scope="module")
def base_report():
    return population().eda(include_plots=False, show=False)


def action_report(base, operation, parameters):
    report = deepcopy(base)
    report.recommendation_details = [
        Recommendation(
            key="consumer.literal",
            title="Copied action",
            rationale="Explicit structured action supplied by this report's caller.",
            priority=ActionPriority.NEXT,
            action=Action("consumer.action", "Apply action", operation, parameters),
        )
    ]
    return report


@pytest.mark.parametrize("stratify", [False, True])
@pytest.mark.parametrize("validation_size", [None, 0.25])
@pytest.mark.parametrize("boundary", ["report", "export"])
def test_structured_split_calls_preserve_boolean_null_and_numeric_parameters(
    base_report, stratify, validation_size, boundary, tmp_path
):
    parameters = dict(
        test_size=0.25, validation_size=validation_size, stratify=stratify, random_state=11
    )
    report = action_report(base_report, "split", parameters)
    consumer = population()
    before = consumer.to_pandas().copy(deep=True)
    code, call = copied_call(report, boundary, tmp_path / "split.html", "Copied action")
    assert {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords} == parameters
    execute(code, consumer)
    indices = np.arange(len(before))
    train, test = train_test_split(
        indices, test_size=0.25, random_state=11, stratify=before.y if stratify else None
    )
    if validation_size is None:
        validation = []
    else:
        train, validation = train_test_split(
            train,
            test_size=validation_size,
            random_state=11,
            stratify=before.y.iloc[train] if stratify else None,
        )
    assert set(consumer.split_plan.train_indices) == set(train)
    assert set(consumer.split_plan.validation_indices) == set(validation)
    assert set(consumer.split_plan.test_indices) == set(test)
    pd.testing.assert_frame_equal(consumer.to_pandas(), before)


@pytest.mark.parametrize("drop_input", [False, True])
@pytest.mark.parametrize("boundary", ["report", "export"])
def test_copied_text_action_retains_boolean_and_none_and_produces_expected_counts(
    base_report, drop_input, boundary, tmp_path
):
    parameters = dict(
        columns=["words"], method="count", max_features=None, drop_input_columns=drop_input
    )
    report = action_report(base_report, "text_features", parameters)
    consumer = population().set_roles({"words": "feature"}).split(test_size=0.25, random_state=3)
    before = consumer.to_pandas().copy(deep=True)
    code, call = copied_call(report, boundary, tmp_path / "text.html", "Copied action")
    assert {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords} == parameters
    execute(code, consumer)
    after = consumer.to_pandas()
    for word in ["blue", "cat", "red"]:
        np.testing.assert_array_equal(
            after[f"words__{word}"], [text.split().count(word) for text in before.words]
        )
    assert ("words" not in after) == drop_input
    pd.testing.assert_series_equal(after.x, before.x)
    pd.testing.assert_series_equal(after.y, before.y)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("offset", [None, 0.0, 2.5])
@pytest.mark.parametrize("boundary", ["report", "export"])
def test_nested_action_parameters_reach_a_real_registered_transform_unchanged(
    base_report, enabled, offset, boundary, tmp_path
):
    params = dict(
        enabled=enabled,
        offset=offset,
        options={"scale": 2.0, "notes": [False, None, {"label": 'literal"<&>\\'}]},
    )
    parameters = dict(name="eda_literal_consumer", columns=["x"], params=params)
    report = action_report(base_report, "apply_custom_transform", parameters)
    consumer = population().split(test_size=0.25, random_state=3)
    before = consumer.to_pandas().copy(deep=True)
    train = before.iloc[list(consumer.split_plan.train_indices)]
    received = []

    def fit(frame, configuration):
        received.append((frame.copy(deep=True), deepcopy(configuration)))
        return dict(
            scale=configuration["options"]["scale"] if configuration["enabled"] else 1.0,
            offset=frame.x.mean() if configuration["offset"] is None else configuration["offset"],
        )

    def transform(frame, artifact):
        return pd.DataFrame(
            {"transformed": frame.x * artifact["scale"] + artifact["offset"]}, index=frame.index
        )

    register_transform("eda_literal_consumer", fit=fit, transform=transform)
    try:
        code, call = copied_call(report, boundary, tmp_path / "custom.html", "Copied action")
        assert {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords} == parameters
        execute(code, consumer)
        assert len(received) == 1 and received[0][1] == params
        pd.testing.assert_frame_equal(received[0][0], train[["x"]])
        expected = before.x.to_numpy() * (2.0 if enabled else 1.0) + (
            train.x.mean() if offset is None else offset
        )
        np.testing.assert_allclose(consumer.to_pandas().transformed, expected)
        assert consumer.custom_plan.params == params
        pd.testing.assert_series_equal(consumer.to_pandas().x, before.x)
    finally:
        unregister_transform("eda_literal_consumer")


@pytest.mark.parametrize("boundary", ["report", "export"])
def test_json_compatible_numpy_numbers_remain_plain_python_literals(
    base_report, boundary, tmp_path
):
    parameters = dict(columns=["x"], strategy="constant", fill_value=np.float64(0.25))
    report = action_report(base_report, "impute", parameters)
    frame = population().to_pandas()
    frame.loc[:2, "x"] = np.nan
    consumer = Session.ingest(frame).set_roles({"x": "feature", "y": "target", "words": "ignore"})
    consumer.split(test_size=0.25, random_state=3)
    code, call = copied_call(report, boundary, tmp_path / "numeric.html", "Copied action")
    values = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    assert values == parameters and type(values["fill_value"]) is float
    execute(code, consumer)
    expected = np.where(frame.x.isna(), 0.25, frame.x)
    np.testing.assert_allclose(consumer.to_pandas().x, expected)


@pytest.mark.parametrize(
    "name", ["ordinary", "line\nraise RuntimeError('extra')", "carriage\rreturn"]
)
@pytest.mark.parametrize("boundary", ["report", "export"])
def test_review_comments_keep_column_text_on_one_python_statement(
    base_report, name, boundary, tmp_path
):
    report = deepcopy(base_report)
    report.quality["high_cardinality_columns"] = [name]
    report.recommendation_details = [
        Recommendation(
            "next.high_cardinality",
            "Review column",
            "Review encoding choices.",
            ActionPriority.NEXT,
        )
    ]
    code, call = copied_call(report, boundary, tmp_path / "review.html", "Review column")
    assert call.func.attr == "explain" and ast.literal_eval(call.args[0]) == "encode"
    assert "\n" not in code and "\r" not in code
    execute(code, population())
