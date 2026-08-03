"""Audit public docstrings against the BuildML depth standard.

BuildML's product promise is that a reader who does not already know a
technique can still use it correctly. Docstrings are the only documentation
attached to the code itself, so they carry that promise. This script turns the
standard described in ``CONTRIBUTING.md`` into a machine check.

Two modes are available:

``--report``
    Print per-package coverage so a docstring pass can be prioritised. Always
    exits ``0``; use this while working.

``--check``
    The CI gate. Fails (exit ``1``) when either:

    * a module listed in :data:`ENFORCED_PREFIXES` has *any* finding — these
      packages have completed their depth pass and must stay clean; or
    * any other module has more findings than its recorded budget in
      :data:`BUDGET_FILE`.

    The budget file is a ratchet for a codebase mid-migration. It records how
    many findings each package currently has, so new shallow docstrings fail
    CI while the existing backlog does not block unrelated work. Regenerate it
    with ``--write-budget`` *only* after an intentional improvement; the writer
    refuses to raise a budget, so the numbers can only fall.

Rules applied to every *public* class, function, and method (a leading
underscore marks a definition private, except for the dunders in
:data:`DOCUMENTED_DUNDERS`):

``missing-docstring``
    No docstring at all.
``summary-too-short``
    The first line is not a usable sentence (fewer than
    :data:`MIN_SUMMARY_WORDS` words).
``no-description``
    Only a summary line. Anything with parameters, a non-``None`` return, or a
    ``raise`` needs a paragraph explaining what the operation means.
``missing-parameters``
    The definition takes arguments but has no ``Parameters`` section.
``undocumented-parameter``
    A named argument is absent from the ``Parameters`` section.
``missing-returns``
    The definition is annotated as returning something other than ``None`` but
    has no ``Returns`` (or ``Yields``) section.
``missing-raises``
    The body raises an exception directly but there is no ``Raises`` section.

Properties are held to a lighter bar: they need a summary and, when they can
return ``None``, an explanation of what ``None`` means. They are exempt from
``Parameters``/``Returns``.

Examples
--------
Prioritise the next package to work on::

    python scripts/audit_docstrings.py --report

Run the gate exactly as CI does::

    python scripts/audit_docstrings.py --check

Inspect the work in progress on one package::

    python scripts/audit_docstrings.py --path buildml/preprocess
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "buildml"

#: Recorded per-package finding counts. Acts as a ratchet: counts may fall,
#: never rise. Regenerate with ``--write-budget`` after real improvement.
BUDGET_FILE = ROOT / "scripts" / "docstring_budget.json"

#: Modules that have completed their depth pass and must stay at zero
#: findings. Append as each package finishes. Removing an entry is a
#: regression and should not happen.
ENFORCED_PREFIXES: tuple[str, ...] = (
    "buildml/activelearning/",
    "buildml/ai/",
    "buildml/anomaly/",
    "buildml/automl/",
    "buildml/causal/",
    "buildml/cbr/",
    "buildml/checkpoint/",
    "buildml/core/",
    "buildml/dashboard/",
    "buildml/data/",
    "buildml/dl/",
    "buildml/eda/",
    "buildml/ensemble/",
    "buildml/explain/",
    "buildml/federated/",
    "buildml/forecasting/",
    "buildml/graph/",
    "buildml/ingest/",
    "buildml/kg/",
    "buildml/metalearning/",
    "buildml/model/",
    "buildml/multitask/",
    "buildml/nlp/",
    "buildml/online/",
    "buildml/optimize/",
    "buildml/pipeline/",
    "buildml/preprocess/",
    "buildml/probabilistic/",
    "buildml/rag/",
    "buildml/ranking/",
    "buildml/recommenders/",
    "buildml/reporting/",
    "buildml/rl/",
    "buildml/selfsupervised/",
    "buildml/semisupervised/",
    "buildml/serving/",
    "buildml/session/",
    "buildml/symbolic/",
    "buildml/synthetic/",
    "buildml/tda/",
    "buildml/timeseries/",
    "buildml/unsupervised/",
)

#: Never audited: vendored 1.x code and generated caches.
SKIP_PREFIXES: tuple[str, ...] = (
    "buildml/_legacy/",
    "buildml/__pycache__/",
)

#: Dunders that form part of the documented user surface.
DOCUMENTED_DUNDERS = frozenset({"__init__", "__call__", "__enter__", "__exit__", "__iter__"})

#: Arguments that never need their own ``Parameters`` entry.
IMPLICIT_ARGS = frozenset({"self", "cls"})

#: A one-word summary ("Fit.") teaches nothing; require a real sentence.
MIN_SUMMARY_WORDS = 4

#: Session mixin methods may be thin facades that point at canonical docs on
#: ``buildml.session.*_ops``. When a public method under
#: ``buildml/session/mixins/`` carries a real summary, a description, Returns
#: (when annotated), and an explicit ops pointer, it is exempt from the full
#: Parameters / Raises essays — those live on the ops function. Properties are
#: unchanged. See CONTRIBUTING.md "Session architecture" and "Docstring
#: standard".
FACADE_MIXIN_PREFIX = "buildml/session/mixins/"

#: Patterns that mark a mixin docstring as a facade pointer to ops.
FACADE_POINTER_RE = re.compile(
    r"(?:buildml\.session\.\w+_ops|:func:`buildml\.session\.\w+_ops|"
    r"Canonical Parameters, Raises, Notes|Session facade over)",
    re.IGNORECASE,
)

SECTION_NAMES = (
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Warns",
    "Notes",
    "Examples",
    "See Also",
    "References",
    "Attributes",
    "Other Parameters",
)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
DefNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


@dataclass(frozen=True)
class Finding:
    """A single docstring violation, formatted like a compiler diagnostic."""

    path: str
    line: int
    symbol: str
    rule: str
    detail: str

    def render(self) -> str:
        """Return the ``path:line: rule: detail`` line printed to stdout."""
        return f"{self.path}:{self.line}: {self.rule}: {self.symbol} — {self.detail}"


@dataclass
class Coverage:
    """Per-package tallies used by ``--report``."""

    definitions: int = 0
    documented: int = 0
    described: int = 0
    with_parameters: int = 0
    needs_parameters: int = 0
    with_returns: int = 0
    needs_returns: int = 0
    with_examples: int = 0
    findings: int = 0

    def merge(self, other: Coverage) -> None:
        """Fold ``other`` into this tally in place."""
        self.definitions += other.definitions
        self.documented += other.documented
        self.described += other.described
        self.with_parameters += other.with_parameters
        self.needs_parameters += other.needs_parameters
        self.with_returns += other.with_returns
        self.needs_returns += other.needs_returns
        self.with_examples += other.with_examples
        self.findings += other.findings


def is_public(name: str) -> bool:
    """Return True when ``name`` is part of the importable public surface."""
    if name in DOCUMENTED_DUNDERS:
        return True
    return not name.startswith("_")


def is_property(node: DefNode) -> bool:
    """Return True when ``node`` is decorated as a property or cached property."""
    if not isinstance(node, FunctionNode):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = getattr(target, "attr", None) or getattr(target, "id", "")
        if name in {"property", "cached_property", "setter", "deleter"}:
            return True
    return False


def is_overload_or_override(node: DefNode) -> bool:
    """Return True for ``@overload`` stubs, which carry no independent docs."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (getattr(target, "attr", None) or getattr(target, "id", "")) == "overload":
            return True
    return False


def split_sections(doc: str) -> dict[str, list[str]]:
    """Split a NumPy-style docstring into ``{section: body lines}``.

    Text before the first section heading is stored under the empty-string key,
    so callers can inspect the summary and description without re-parsing.
    """
    sections: dict[str, list[str]] = {"": []}
    current = ""
    lines = doc.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        following = lines[index + 1].strip() if index + 1 < len(lines) else ""
        is_heading = (
            stripped in SECTION_NAMES
            and following
            and set(following) == {"-"}
            and len(following) >= len(stripped)
        )
        if is_heading:
            current = stripped
            sections.setdefault(current, [])
            index += 2
            continue
        sections[current].append(line)
        index += 1
    return sections


def documented_parameter_names(section: Iterable[str]) -> set[str]:
    """Extract parameter names from the body of a ``Parameters`` section.

    Handles both NumPy spellings BuildML uses — ``name:`` and ``name : type`` —
    plus the grouped shorthands that document sibling arguments on one line
    (``a / b / c:`` and NumPy's own ``a, b, c:``), and the ``*args`` /
    ``**kwargs`` forms.
    """
    names: set[str] = set()
    for raw in section:
        if not raw.strip() or raw.startswith((" " * 5, "\t")):
            continue
        line = raw.strip()
        # Bullets and block quotes always carry a space after their marker;
        # ``**kwargs:`` does not, so variadic parameters survive this filter.
        if line.startswith(("* ", "- ", "> ")):
            continue
        head = line.split(" : ")[0] if " : " in line else line.split(":")[0]
        if head == line and ":" not in line:
            continue
        for group in head.split("/"):
            for candidate in group.split(","):
                token = candidate.strip().lstrip("*")
                if token.isidentifier():
                    names.add(token)
    return names


def annotated_return(node: DefNode) -> str | None:
    """Return the source text of the return annotation, or None if absent."""
    if not isinstance(node, FunctionNode) or node.returns is None:
        return None
    return ast.unparse(node.returns)


def returns_a_value(node: DefNode) -> bool:
    """Return True when the annotation promises something other than ``None``."""
    annotation = annotated_return(node)
    if annotation is None:
        return False
    return annotation.strip() not in {"None", "'None'", '"None"'}


def raises_directly(node: DefNode) -> bool:
    """Return True when the body raises without a surrounding ``except``.

    Nested definitions are skipped so a helper's error handling is not
    attributed to its enclosing function.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for inner in ast.walk(child):
            if isinstance(inner, ast.Raise):
                return True
    return False


def argument_names(node: DefNode) -> list[str]:
    """Return every caller-supplied argument name in declaration order."""
    if not isinstance(node, FunctionNode):
        return []
    args = node.args
    collected = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg:
        collected.append(args.vararg.arg)
    if args.kwarg:
        collected.append(args.kwarg.arg)
    return [name for name in collected if name not in IMPLICIT_ARGS]


def iter_definitions(tree: ast.Module) -> Iterator[tuple[DefNode, str]]:
    """Yield ``(node, qualified name)`` for public definitions worth auditing.

    Nested functions defined inside another function are skipped: they are
    implementation detail even when their name lacks an underscore.
    """

    def walk(node: ast.AST, prefix: str, inside_function: bool) -> Iterator[tuple[DefNode, str]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                qualified = f"{prefix}{child.name}"
                is_function = isinstance(child, FunctionNode)
                if is_public(child.name) and not inside_function:
                    yield child, qualified
                yield from walk(
                    child,
                    f"{qualified}.",
                    inside_function or is_function,
                )

    yield from walk(tree, "", False)


def is_session_facade(path: str, doc: str, node: DefNode) -> bool:
    """Return True when ``node`` is an allowed Session mixin facade docstring.

    Facades still need a real summary, a description paragraph, Returns when
    the annotation promises a value, and an explicit pointer to the canonical
    ``*_ops`` function. They may omit the full Parameters / Raises essays.
    """
    if not path.startswith(FACADE_MIXIN_PREFIX):
        return False
    if is_property(node):
        return False
    if isinstance(node, ast.ClassDef):
        return False
    return bool(FACADE_POINTER_RE.search(doc))


def audit_definition(node: DefNode, symbol: str, path: str) -> list[Finding]:
    """Return every standard violation for one public definition."""
    findings: list[Finding] = []

    def add(rule: str, detail: str) -> None:
        findings.append(Finding(path, node.lineno, symbol, rule, detail))

    doc = ast.get_docstring(node)
    if not doc or not doc.strip():
        add("missing-docstring", "no docstring")
        return findings

    doc = doc.strip()
    sections = split_sections(doc)
    preamble = [line for line in sections[""] if line.strip()]
    summary = preamble[0].strip() if preamble else ""

    if len(summary.split()) < MIN_SUMMARY_WORDS:
        add("summary-too-short", f"summary {summary!r} is not a full sentence")

    prop = is_property(node)
    facade = is_session_facade(path, doc, node)
    arguments = argument_names(node)
    wants_return = returns_a_value(node)
    needs_body = bool(arguments) or wants_return or raises_directly(node)

    if needs_body and len(preamble) < 2 and not prop:
        add("no-description", "summary line only; explain what the operation does and why")

    if facade and not prop:
        # Facade contract: summary + description (above) + Returns + ops pointer.
        if wants_return and "Returns" not in sections and "Yields" not in sections:
            annotation = annotated_return(node) or "value"
            add("missing-returns", f"facade returns {annotation} with no Returns section")
        if not FACADE_POINTER_RE.search(doc):
            add("no-description", "facade missing canonical ops pointer")
        return findings

    if arguments and not prop:
        if "Parameters" not in sections:
            add("missing-parameters", f"documents none of {len(arguments)} argument(s)")
        else:
            documented = documented_parameter_names(sections["Parameters"])
            for name in arguments:
                if name not in documented:
                    add("undocumented-parameter", f"'{name}' is missing from Parameters")

    if wants_return and not prop:
        if "Returns" not in sections and "Yields" not in sections:
            annotation = annotated_return(node) or "value"
            add("missing-returns", f"returns {annotation} with no Returns section")

    if raises_directly(node) and "Raises" not in sections and not prop:
        add("missing-raises", "raises an exception with no Raises section")

    if prop:
        annotation = annotated_return(node) or ""
        optional = annotation.endswith("| None") or annotation.startswith("Optional[")
        if optional and "None" not in doc:
            add("no-description", "may return None; say what None means")

    return findings


def measure_definition(node: DefNode, doc: str | None) -> Coverage:
    """Return single-definition tallies contributing to a package report."""
    tally = Coverage(definitions=1)
    prop = is_property(node)
    arguments = argument_names(node)
    wants_return = returns_a_value(node)
    if arguments and not prop:
        tally.needs_parameters = 1
    if wants_return and not prop:
        tally.needs_returns = 1
    if not doc:
        return tally
    tally.documented = 1
    sections = split_sections(doc.strip())
    if len([line for line in sections[""] if line.strip()]) >= 2:
        tally.described = 1
    if "Parameters" in sections:
        tally.with_parameters = 1
    if "Returns" in sections or "Yields" in sections:
        tally.with_returns = 1
    if "Examples" in sections or ">>>" in doc:
        tally.with_examples = 1
    return tally


def relative(path: Path) -> str:
    """Return ``path`` as a repo-relative POSIX string."""
    return path.resolve().relative_to(ROOT).as_posix()


def iter_sources() -> Iterator[Path]:
    """Yield auditable ``buildml`` sources in stable order."""
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = relative(path)
        if any(rel.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue
        if "__pycache__" in rel:
            continue
        yield path


def audit_file(path: Path) -> tuple[list[Finding], Coverage]:
    """Audit one module, returning its findings and its coverage tally."""
    rel = relative(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        finding = Finding(rel, getattr(exc, "lineno", 1) or 1, "<module>", "parse-error", str(exc))
        return [finding], Coverage()

    findings: list[Finding] = []
    coverage = Coverage()
    for node, symbol in iter_definitions(tree):
        if is_overload_or_override(node):
            continue
        coverage.merge(measure_definition(node, ast.get_docstring(node)))
        findings.extend(audit_definition(node, symbol, rel))
    coverage.findings = len(findings)
    return findings, coverage


def package_of(rel: str) -> str:
    """Return the top-level subpackage name for a repo-relative path."""
    parts = rel.split("/")
    return parts[1] if len(parts) > 2 else "(root)"


def is_enforced(rel: str) -> bool:
    """Return True when ``rel`` must hold at zero findings."""
    return any(rel.startswith(prefix) for prefix in ENFORCED_PREFIXES)


def load_budget() -> dict[str, int]:
    """Return the recorded per-package finding ceilings.

    An empty mapping is returned when the budget file is absent, which makes a
    fresh checkout report rather than fail.
    """
    if not BUDGET_FILE.exists():
        return {}
    payload = json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in payload.get("packages", {}).items()}


def write_budget(counts: dict[str, int], *, rebaseline: bool = False) -> list[str]:
    """Record ``counts`` as the new ceilings, refusing any increase.

    Parameters
    ----------
    counts:
        Findings observed per package in this run.
    rebaseline:
        Accept increases as well as decreases. Reserved for deliberately
        re-establishing the baseline — for example after a large feature branch
        lands new public surface. The caller is expected to report what was
        ratified, since silently raising a ratchet defeats it.

    Returns
    -------
    list of str
        Packages whose observed count exceeds the previously recorded ceiling.
        When ``rebaseline`` is off this means nothing was written and the
        caller should fail; when it is on, the file was written and the list
        names what was ratified so the caller can report it.
    """
    existing = load_budget()
    regressions = [
        name for name, count in counts.items() if count > existing.get(name, count)
    ]
    if regressions and not rebaseline:
        return regressions
    merged = {**existing, **counts}
    payload = {
        "_comment": (
            "Per-package docstring finding ceilings enforced by "
            "scripts/audit_docstrings.py --check. Counts may fall, never rise. "
            "See the docstring standard in CONTRIBUTING.md."
        ),
        "packages": dict(sorted(merged.items())),
    }
    BUDGET_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return regressions


def render_report(coverage: dict[str, Coverage]) -> str:
    """Return a fixed-width coverage table sorted by remaining work."""
    header = (
        f"{'package':<16}{'defs':>6}{'doc%':>7}{'desc%':>7}"
        f"{'param%':>8}{'ret%':>7}{'ex':>5}{'issues':>8}"
    )
    lines = [header, "-" * len(header)]

    def pct(part: int, whole: int) -> str:
        return "  n/a" if whole == 0 else f"{100 * part / whole:5.0f}"

    ordered = sorted(coverage.items(), key=lambda kv: (-kv[1].findings, kv[0]))
    for name, tally in ordered:
        lines.append(
            f"{name:<16}{tally.definitions:>6}"
            f"{pct(tally.documented, tally.definitions):>7}"
            f"{pct(tally.described, tally.definitions):>7}"
            f"{pct(tally.with_parameters, tally.needs_parameters):>8}"
            f"{pct(tally.with_returns, tally.needs_returns):>7}"
            f"{tally.with_examples:>5}{tally.findings:>8}"
        )
    total = Coverage()
    for tally in coverage.values():
        total.merge(tally)
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<16}{total.definitions:>6}"
        f"{pct(total.documented, total.definitions):>7}"
        f"{pct(total.described, total.definitions):>7}"
        f"{pct(total.with_parameters, total.needs_parameters):>8}"
        f"{pct(total.with_returns, total.needs_returns):>7}"
        f"{total.with_examples:>5}{total.findings:>8}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the audit; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when a package regresses past its budget (CI gate)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print per-package coverage and always exit 0",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=None,
        metavar="PATH",
        help="audit only this file or directory (repeatable); prints every finding",
    )
    parser.add_argument(
        "--write-budget",
        action="store_true",
        help=(
            "record current counts as the new ceilings; refuses any increase. "
            "Combine with --path to rewrite only the packages you improved, "
            "leaving other packages' recorded ceilings untouched."
        ),
    )
    parser.add_argument(
        "--rebaseline",
        action="store_true",
        help=(
            "with --write-budget, accept counts that rose as well as fell. "
            "Use only when new public surface has deliberately landed undocumented; "
            "every raised package is printed."
        ),
    )
    args = parser.parse_args(argv)

    if args.path:
        selected = []
        for entry in args.path:
            target = (ROOT / entry).resolve()
            selected.extend(sorted(target.rglob("*.py")) if target.is_dir() else [target])
    else:
        selected = list(iter_sources())

    coverage: dict[str, Coverage] = defaultdict(Coverage)
    counts: dict[str, int] = defaultdict(int)
    enforced_findings: list[Finding] = []
    all_findings: list[Finding] = []
    for path in selected:
        findings, tally = audit_file(path)
        rel = relative(path)
        package = package_of(rel)
        coverage[package].merge(tally)
        counts[package] += len(findings)
        all_findings.extend(findings)
        if is_enforced(rel):
            enforced_findings.extend(findings)

    if args.path and not args.write_budget:
        for finding in all_findings:
            print(finding.render())
        print(f"{len(all_findings)} finding(s) in the selected path(s)")
        return 0

    if args.write_budget:
        regressions = write_budget(dict(counts), rebaseline=args.rebaseline)
        if regressions and not args.rebaseline:
            print(
                "refusing to raise the ratchet for: " + ", ".join(sorted(regressions)),
                file=sys.stderr,
            )
            print(
                "Document the new public surface, or pass --rebaseline to accept "
                "the higher count deliberately.",
                file=sys.stderr,
            )
            return 1
        for name in sorted(regressions):
            print(f"rebaselined {name} upward to {counts[name]}")
        print(f"recorded budgets for {len(counts)} package(s) in {BUDGET_FILE.name}")
        return 0

    if args.report or not args.check:
        print(render_report(coverage))
        return 0

    failures: list[str] = []
    for finding in enforced_findings:
        print(finding.render())
    if enforced_findings:
        failures.append(
            f"{len(enforced_findings)} finding(s) in packages required to stay clean"
        )

    budget = load_budget()
    for package, count in sorted(counts.items()):
        ceiling = budget.get(package)
        if ceiling is not None and count > ceiling:
            failures.append(
                f"{package}: {count} finding(s) exceeds its budget of {ceiling}"
            )

    if failures:
        for line in failures:
            print(line, file=sys.stderr)
        print(
            "docstring audit failed — see the docstring standard in CONTRIBUTING.md",
            file=sys.stderr,
        )
        return 1
    print(f"docstring audit passed ({sum(counts.values())} finding(s) within budget)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
