"""Saved dashboards keep report text inside data rather than HTML script markup."""

import copy
import json
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

from buildml import Session
from buildml.dashboard.offline import (
    build_offline_bundle,
    export_studio_html,
    render_offline_html,
)
from buildml.dashboard.serialize import json_safe

TEXTS = [
    pytest.param(
        "</script><script id=buildml-injected-probe>globalThis.buildmlProbe=true</script>",
        id="closing-script",
    ),
    pytest.param(
        "</ScRiPt ><img id=buildml-injected-probe src=x onerror=globalThis.buildmlProbe=true>",
        id="mixed-case-space",
    ),
    pytest.param(
        "</script\n><svg id=buildml-injected-probe onload=globalThis.buildmlProbe=true>",
        id="newline-end-tag",
    ),
    pytest.param("<!--<script>quoted text</script>-->", id="html-comment"),
    pytest.param("& <tag> \"quotes\" 'apostrophe' 日本語 🌍\u2028line\u2029end", id="unicode"),
    pytest.param(r"literal \u003c/script\u003e and backslash \\ path", id="literal-escape"),
]


class _Document(HTMLParser):
    """Use HTML raw-text parsing, independent of the producer's JSON encoder."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.scripts = []
        self.current = None
        self.injected = []
        self.titles = []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id") == "buildml-injected-probe":
            self.injected.append(tag)
        if tag == "script":
            self.current = {"attrs": attributes, "text": ""}
            self.scripts.append(self.current)
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"] += data
        if self.in_title:
            self.titles.append(data)

    def handle_entityref(self, name):
        from html import unescape

        self.handle_data(unescape(f"&{name};"))


def _embedded_bundle(document):
    parsed = _Document()
    # Match the HTML input stream; escaped JSON string characters stay intact.
    parsed.feed(document.replace("\r\n", "\n").replace("\r", "\n"))
    parsed.close()
    assert not parsed.injected
    assert len(parsed.scripts) == 2  # Vendored Plotly and the application module.
    modules = [s["text"] for s in parsed.scripts if s["attrs"].get("type") == "module"]
    assert len(modules) == 1
    marker = "window.__BUILDML_OFFLINE__ = "
    value, end = json.JSONDecoder().raw_decode(modules[0].split(marker, 1)[1])
    remainder = modules[0].split(marker, 1)[1][end:].lstrip()
    assert remainder.startswith(";")
    assert remainder[1:].lstrip().startswith("await import(appUrl);")
    assert "await import(appUrl);" in modules[0]
    return value, modules[0], "".join(parsed.titles)


@pytest.mark.parametrize("text", TEXTS)
def test_rendered_bundle_survives_html_parsing_without_creating_elements(text):
    pytest.importorskip("plotly")
    bundle = {
        "title": text,
        "report": {text: [text, None, True, 0, -1.25]},
        "meta": {"nested": {"dataset": text}},
    }
    before = copy.deepcopy(bundle)
    actual, _, title = _embedded_bundle(render_offline_html(bundle))
    assert actual == before == bundle
    assert title == text + " · Offline EDA App"


@pytest.fixture(scope="module", params=TEXTS[:4])
def report_observation(request):
    text = request.param
    frame = pd.DataFrame({text: list(range(24)), "category": [text, "plain"] * 12})
    session = Session.ingest(frame).set_roles({text: "feature", "category": "feature"})
    report = session.eda(include_plots=False, show=False, sample_rows=12)
    raw = json.loads(json.dumps(json_safe(report.to_dict()), allow_nan=False))
    assert text in raw["overview"]["columns"]
    return report, raw, text


def test_saved_session_export_keeps_original_report_and_nested_metadata(
    report_observation, tmp_path
):
    pytest.importorskip("plotly")
    _, raw, text = report_observation
    before = copy.deepcopy(raw)
    meta = {"dataset": text, "nested": {"notes": [text]}}
    destination = tmp_path / "nested" / "studio.html"
    assert export_studio_html(raw, destination, title=text, session_meta=meta) == destination
    actual, _, title = _embedded_bundle(destination.read_text(encoding="utf-8"))
    assert actual["report"] == before == raw
    assert actual["meta"]["session"] == meta
    assert actual["title"] == text
    assert title == text + " · Offline EDA App"
    assert set(actual["charts_light"]) == set(actual["charts_dark"])

    def contains_text(value):
        if isinstance(value, dict):
            return any(contains_text(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_text(item) for item in value)
        return isinstance(value, str) and text in value

    assert contains_text(actual["charts_light"])


def test_http_download_keeps_report_text_inside_data(report_observation):
    pytest.importorskip("plotly")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from buildml.dashboard.app import create_app
    from buildml.dashboard.state import DashboardState, clear_state, set_state

    report, raw, text = report_observation
    before = copy.deepcopy(raw)
    meta = {"dataset": text}
    set_state(DashboardState(report=report, report_dict=raw, title=text, session_meta=meta))
    try:
        with TestClient(create_app()) as client:
            response = client.get("/api/export/html")
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/html")
            assert "attachment" in response.headers["content-disposition"]
            actual, _, title = _embedded_bundle(response.text)
            assert actual["report"] == before == raw
            assert actual["meta"]["session"] == meta
            assert title == text + " · Offline EDA App"
    finally:
        clear_state()


def test_inline_application_modules_roundtrip_to_packaged_sources():
    pytest.importorskip("plotly")
    import buildml.dashboard.offline as offline

    bundle = build_offline_bundle({"overview": {"n_rows": 0, "n_columns": 0}})
    actual, module, _ = _embedded_bundle(render_offline_html(bundle))
    assert actual == bundle
    root = Path(offline.__file__).parent / "static" / "js"
    assets = {
        "iconsSource": "icons.js",
        "learnUiSource": "learn_ui.js",
        "gatesViewSource": "gates_view.js",
        "academyViewSource": "academy_view.js",
        "cockpitViewSource": "cockpit_view.js",
        "appSource": "app.js",
    }
    imports = {
        "icons.js": "#buildml-icons",
        "learn_ui.js": "#buildml-learn-ui",
        "gates_view.js": "#buildml-gates-view",
        "academy_view.js": "#buildml-academy-view",
        "cockpit_view.js": "#buildml-cockpit-view",
    }
    for variable, filename in assets.items():
        source = (root / filename).read_text(encoding="utf-8")
        if filename in {"gates_view.js", "academy_view.js", "cockpit_view.js", "app.js"}:
            for name, placeholder in imports.items():
                source = source.replace(f'from "./{name}"', f'from "{placeholder}"')
        encoded = module.split(f"const {variable} = ", 1)[1]
        decoded, end = json.JSONDecoder().raw_decode(encoded)
        assert encoded[end:].lstrip().startswith(";")
        assert decoded == source
