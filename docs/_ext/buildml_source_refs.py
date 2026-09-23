"""Resolve internal API references to source from this exact documentation build."""

from __future__ import annotations

import html
import importlib
import inspect
from pathlib import Path
from urllib.parse import urlsplit

from docutils import nodes
from sphinx import addnodes


def source_location(target, role=None):
    """Return a verified BuildML source path and line, or no match."""
    if not target.startswith("buildml."):
        return None
    parts = target.split(".")
    for end in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:end]))
        except (ImportError, AttributeError):
            continue
        parent = obj
        try:
            for part in parts[end:]:
                parent, obj = obj, getattr(obj, part)
            checks = {
                "class": inspect.isclass,
                "exc": lambda value: inspect.isclass(value) and issubclass(value, BaseException),
                "func": inspect.isfunction,
                "meth": lambda value: inspect.isfunction(value) or inspect.ismethod(value),
                "mod": inspect.ismodule,
            }
            if role in checks and not checks[role](obj):
                return None
            if isinstance(obj, property):
                obj = obj.fget
            if inspect.ismethoddescriptor(obj) or inspect.isdatadescriptor(obj):
                obj = parent
            path = Path(inspect.getsourcefile(obj)).resolve()
            line = inspect.getsourcelines(obj)[1]
        except (AttributeError, TypeError, OSError):
            return None
        root = Path(__file__).resolve().parents[2]
        if not path.is_relative_to(root / "buildml"):
            return None
        return path, max(1, line), path.relative_to(root).as_posix()
    return None


def resolve_reference(app, env, node, contnode):
    """Link unresolved internal names only when their source is verifiable."""
    if app.builder.format != "html" or node.get("refdomain") != "py":
        return None
    target = node["reftarget"]
    candidates = [target]
    module = node.get("py:module")
    if module:
        candidates.append(f"{module}.{target}")
    for candidate in candidates:
        location = source_location(candidate, node.get("reftype"))
        if location is None:
            continue
        path, line, relative = location
        page = "_api_source/" + relative.removesuffix(".py")
        if not hasattr(env, "buildml_source_pages"):
            env.buildml_source_pages = {}
        env.buildml_source_pages[page] = str(path)
        uri = app.builder.get_relative_uri(node["refdoc"], page) + f"#L{line}"
        reference = nodes.reference("", "", internal=True, refuri=uri)
        reference["reftitle"] = f"Source for {candidate} in this documentation build"
        reference += contnode
        return reference
    return None


def source_pages(app):
    """Render linked source as local pages rather than mutable remote links."""
    for page, filename in sorted(getattr(app.env, "buildml_source_pages", {}).items()):
        path = Path(filename)
        body = '<p>Source used by this documentation build.</p><pre>'
        body += "\n".join(
            f'<span id="L{line}">{html.escape(text)}</span>'
            for line, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        )
        body += "</pre>"
        yield page, {"title": path.name, "body": body}, "page.html"


def link_definitions(app, doctree):
    """Add source navigation using each definition's verified runtime object."""
    if app.builder.format != "html":
        return
    for signature in doctree.findall(addnodes.desc_signature):
        module, fullname = signature.get("module"), signature.get("fullname")
        if not module or not fullname:
            continue
        pending = addnodes.pending_xref(
            "", refdomain="py", reftarget=f"{module}.{fullname}", refdoc=app.env.docname
        )
        reference = resolve_reference(app, app.env, pending, nodes.inline("", "[source]"))
        if reference is not None:
            signature += nodes.Text(" ")
            signature += reference


def resolve_guide_links(app, doctree):
    """Map included Markdown links to verified Sphinx wrapper documents."""
    for node in doctree.findall(addnodes.pending_xref):
        if node.get("reftype") != "myst":
            continue
        target = urlsplit(node.get("reftarget", ""))
        if target.scheme or target.netloc or target.query:
            continue
        guide = Path(target.path)
        if guide.suffix != ".md" or len(guide.parts) != 1:
            continue
        docname = "guide-index" if guide.name == "README.md" else guide.stem
        wrapper = Path(app.srcdir) / (docname + ".rst")
        if docname not in app.env.found_docs or not wrapper.is_file():
            continue
        if f".. include:: ../guides/{guide.name}" not in wrapper.read_text(encoding="utf-8"):
            continue
        node["refdomain"] = "doc"
        node["reftarget"] = docname
        node["reftargetid"] = target.fragment


def setup(app):
    """Register source navigation after normal API reference resolution."""
    app.connect("doctree-read", resolve_guide_links)
    app.connect("doctree-read", link_definitions)
    app.connect("missing-reference", resolve_reference, priority=900)
    app.connect("html-collect-pages", source_pages)
    return {"version": "1", "parallel_read_safe": False}
