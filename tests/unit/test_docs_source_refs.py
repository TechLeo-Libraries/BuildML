"""Documentation source fallback must resolve real objects without hiding errors."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from docutils import nodes
from sphinx import addnodes

_SPEC = importlib.util.spec_from_file_location(
    "buildml_source_refs", Path(__file__).parents[2] / "docs/_ext/buildml_source_refs.py"
)
assert _SPEC and _SPEC.loader
refs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(refs)


def test_source_anchor_points_at_actual_definition():
    path, line, relative = refs.source_location("buildml.explain.prerequisites.providers_for")
    assert path.read_text(encoding="utf-8").splitlines()[line - 1].startswith("def providers_for(")
    assert relative == "buildml/explain/prerequisites.py"


def test_missing_and_external_symbols_remain_unresolved():
    assert refs.source_location("buildml.explain.prerequisites.does_not_exist") is None
    assert refs.source_location("typing.Any") is None
    assert refs.source_location("int | None") is None


def test_resolver_uses_current_build_source_and_keeps_label():
    app = SimpleNamespace(builder=SimpleNamespace(format="html", get_relative_uri=lambda _origin, page: page + ".html"))
    env = SimpleNamespace()
    node = addnodes.pending_xref("", refdomain="py", reftarget="buildml.explain.prerequisites.providers_for", refdoc="package")
    result = refs.resolve_reference(app, env, node, nodes.literal("", "providers_for"))
    assert result.astext() == "providers_for"
    assert result["refuri"].startswith("_api_source/buildml/explain/prerequisites.html#L")
    assert env.buildml_source_pages


def test_rendered_source_escapes_html_and_preserves_line_anchors(tmp_path):
    source = tmp_path / "example.py"
    source.write_text('value = "<script>"\n', encoding="utf-8")
    app = SimpleNamespace(env=SimpleNamespace(buildml_source_pages={"_api_source/example": str(source)}))
    page, context, template = next(refs.source_pages(app))
    assert page == "_api_source/example"
    assert template == "page.html"
    assert 'id="L1"' in context["body"]
    assert "&lt;script&gt;" in context["body"]
    assert "<script>" not in context["body"]


def test_definition_source_links_use_verified_objects_only():
    app = SimpleNamespace(
        builder=SimpleNamespace(format="html", get_relative_uri=lambda _origin, page: page + ".html"),
        env=SimpleNamespace(docname="package"),
    )
    document = nodes.container()
    valid = addnodes.desc_signature("", module="buildml.explain.schemas", fullname="SerializableSchema")
    missing = addnodes.desc_signature("", module="buildml.explain.schemas", fullname="MissingClass")
    document.extend([valid, missing])
    refs.link_definitions(app, document)
    assert valid[-1]["refuri"].startswith("_api_source/buildml/explain/schemas.html#L")
    assert valid[-1].astext() == "[source]"
    assert len(missing) == 0


def test_wrong_explicit_role_stays_unresolved():
    assert refs.source_location("buildml.explain.prerequisites.providers_for", "class") is None
    assert refs.source_location("buildml.explain.prerequisites.providers_for", "func") is not None


def test_non_html_builders_do_not_receive_html_source_links():
    app = SimpleNamespace(builder=SimpleNamespace(format="latex"))
    node = addnodes.pending_xref("", refdomain="py", reftarget="buildml.explain.prerequisites.providers_for")
    assert refs.resolve_reference(app, SimpleNamespace(), node, nodes.literal("", "providers_for")) is None
    refs.link_definitions(app, nodes.container())


def test_guide_links_require_matching_wrapper_and_preserve_fragment(tmp_path):
    (tmp_path / "quickstart.rst").write_text(".. include:: ../guides/quickstart.md\n")
    app = SimpleNamespace(srcdir=tmp_path, env=SimpleNamespace(found_docs={"quickstart", "missing"}))
    document = nodes.container()
    valid = addnodes.pending_xref("", refdomain=None, reftype="myst", reftarget="quickstart.md#example")
    missing = addnodes.pending_xref("", refdomain=None, reftype="myst", reftarget="missing.md")
    document.extend([valid, missing])
    refs.resolve_guide_links(app, document)
    assert valid["refdomain"] == "doc"
    assert valid["reftarget"] == "quickstart"
    assert valid["reftargetid"] == "example"
    assert missing["refdomain"] is None
    assert missing["reftarget"] == "missing.md"
