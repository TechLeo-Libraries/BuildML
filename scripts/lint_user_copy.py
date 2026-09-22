"""Lint current documentation and user-facing Python copy."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Paths skipped when scanning for user-facing copy rules.
ARCHIVAL_DOC_PREFIXES: tuple[str, ...] = ()
ARCHIVAL_DOCS: set[str] = set()
QUOTED_EXAMPLE_DOCS: set[str] = set()

COPY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "internal-or-dismissive-copy",
        re.compile(
            r"\b(?:estimator\s+zoo|empty\s+theater|perfect[- ]score\s+theater|"
            r"keep\s+this\s+method\s+as\s+a\s+thin\s+delegate|"
            r"do\s+not\s+spray\s+stubs|phase\s+coverage\s+tracker|"
            r"yank\s+them\s+on\s+pypi)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "unsupported-quality-label",
        re.compile(
            r"\b(?:research[- ]grade|production[- ]grade|enterprise[- ]grade|"
            r"professional mode|rich evaluation|deeply evaluate|teachable|"
            r"highest standard)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "marketing-boilerplate",
        re.compile(
            r"\b(?:unlock(?:ing|s)?|revolutionary|game[- ]chang(?:ing|er)|"
            r"cutting[- ]edge|best[- ]in[- ]class|seamless(?:ly)?|"
            r"harness the power|in today'?s fast[- ]paced|delve into)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "templated-heading",
        re.compile(
            r"\b(?:executive narrative|actionable recommendations)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "vague-copy",
        re.compile(
            r"\b(?:the data was handled|make complex machine[- ]learning "
            r"(?:processes|models?),? easy)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "first-or-only-library-claim",
        re.compile(
            r"(?:world'?s\s+first|first[- ]of[- ]its[- ]kind|"
            r"the\s+only\s+(?:python\s+)?(?:machine[- ]learning\s+|ml\s+)?"
            r"(?:library|toolkit|framework)|"
            r"only\s+such\s+(?:library|toolkit|framework))",
            re.IGNORECASE,
        ),
    ),
)

STALE_API = re.compile(
    r"\b(?:SupervisedLearning|MLwiz|buildml\.(?:automate|build_model|"
    r"date_features|output_dataset|preprocessing))\b"
)
LEGACY_CONTEXT = re.compile(r"\b(?:1\.x|legacy|removed|archiv|not part)\b", re.IGNORECASE)

# Soft-leakage teaching regressions after Phase A/B hard-refuse.
# Patterns are matched on a whitespace-normalized line window so claims split
# across wrapped docstring lines still fail CI.
SOFT_LEAKAGE_FALSE_CLAIM = re.compile(
    r"(?:"
    r"refuse(?:s|d)?\s+unless\s+you\s+pass\s+a\s+fold-local\s+recipe"
    r"|refuse(?:s|d)?\s+unless\s+(?:a\s+)?(?:fold-local\s+)?PreprocessRecipe"
    r"|without\s+a\s+fold-local\s+recipe,\s+BuildML\s+refuses"
    r"|already\s+ran\s+and\s+no\s+fold-local\s+recipe\s+is\s+provided"
    r"|and\s+no\s+fold-local\s+recipe\s+is\s+provided"
    r"|runs\s+without\s+a\s+fold-local\s+recipe"
    r"|without\s+a\s+fold\s+recipe,\s+treat\s+preprocess\s+honesty"
    r"|before\s+CV\s+without\s+a\s+fold\s+recipe"
    r")",
    re.IGNORECASE,
)

# UTF-8 decoded as cp1252/latin-1 left as Unicode (Â±, â€", â†', Î¸, …).
MOJIBAKE_MARKERS = re.compile(
    r"(?:\u00c2[\u0080-\u00bf]|Â±|"
    r"\u00e2\u20ac.|\u00e2\u2020\u2019|"
    r"\u00ce[\u0080-\u00bf]|\u00cf[\u0080-\u00bf])"
)

# Em dash (U+2014) is a common LLM typography tell in this project's copy.
# Prefer ASCII punctuation (: ; , . or hyphen) in docs and user-facing strings.
EM_DASH = re.compile("\u2014")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    rule: str
    text: str


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def iter_targets() -> Iterable[Path]:
    """Yield authored documentation, sources, scripts, and benchmark copy."""
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT, capture_output=True, check=False
        )
    except FileNotFoundError:
        tracked = None
    tracked_paths = (
        set(tracked.stdout.decode("utf-8").split("\0"))
        if tracked is not None and tracked.returncode == 0 else None
    )
    docs = list(ROOT.glob("*.md"))
    docs.extend((ROOT / "guides").rglob("*.md"))
    docs.extend((ROOT / "docs").rglob("*.rst"))
    docs.extend((ROOT / "docs").rglob("*.md"))
    docs.append(ROOT / "examples" / "README.md")
    docs.extend((ROOT / "proofs").rglob("README.md"))
    python = (ROOT / "buildml").rglob("*.py")
    extra = [
        path
        for directory in ("buildml", "examples", "proofs", "benchmarks", "scripts")
        for path in (ROOT / directory).rglob("*")
        if path.suffix in {".py", ".md", ".rst", ".html", ".js", ".json"}
    ]

    paths: list[Path] = []
    for path in [*docs, *python, *extra]:
        if not path.is_file():
            continue
        relative = _relative(path)
        if tracked_paths is not None and relative not in tracked_paths:
            continue
        if any(part in {"_build", "__pycache__", "node_modules", ".pytest_tmp"} for part in path.parts):
            continue
        if relative in ARCHIVAL_DOCS or relative in QUOTED_EXAMPLE_DOCS:
            continue
        if any(relative.startswith(prefix) for prefix in ARCHIVAL_DOC_PREFIXES):
            continue
        if relative.startswith("buildml/_legacy/"):
            continue
        # This file defines prohibited phrases as rule fixtures. Test fixtures
        # live under tests/, outside the authored-copy roots above.
        if relative == "scripts/lint_user_copy.py":
            continue
        paths.append(path)
    yield from sorted(set(paths), key=_relative)


def _soft_leakage_windows(lines: list[str]) -> Iterable[tuple[int, str]]:
    """Yield (start_line, normalized text) for single lines and adjacent pairs."""
    for number, line in enumerate(lines, start=1):
        yield number, line
        if number < len(lines):
            joined = f"{line} {lines[number]}"
            yield number, re.sub(r"\s+", " ", joined)


def lint_paths(paths: Iterable[Path] | None = None) -> list[Violation]:
    """Return copy violations without changing files."""
    violations: list[Violation] = []
    selected = paths if paths is not None else iter_targets()
    for path in selected:
        relative = _relative(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        # Preserve the existing typography policy on its original surfaces;
        # public-copy checks cover all additional assets below.
        typography = (
            path.suffix not in {".js", ".html", ".json"}
            and not relative.startswith(("examples/", "proofs/", "benchmarks/", "scripts/"))
        ) or path.name == "README.md"
        # Parse Python literals so adjacent strings and wrapped docstrings are
        # checked as the text users actually receive.
        if path.suffix == ".py":
            try:
                tree = ast.parse("\n".join(lines))
            except SyntaxError:
                tree = None
            for node in ast.walk(tree) if tree is not None else ():
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    normalized = re.sub(r"\s+", " ", node.value)
                    for rule, pattern in COPY_RULES:
                        if pattern.search(normalized) and not any(
                            pattern.search(line) for line in lines[node.lineno - 1:node.end_lineno]
                        ):
                            violations.append(Violation(relative, node.lineno, rule, normalized[:200]))
        seen_soft: set[tuple[int, str]] = set()
        for number, line in enumerate(lines, start=1):
            for rule, pattern in COPY_RULES:
                if pattern.search(line):
                    violations.append(Violation(relative, number, rule, line.strip()))
            if STALE_API.search(line) and not LEGACY_CONTEXT.search(line):
                violations.append(Violation(relative, number, "stale-public-api", line.strip()))
            if MOJIBAKE_MARKERS.search(line) and not (
                relative == "CHANGELOG.md" and "mojibake" in line
            ):
                violations.append(Violation(relative, number, "mojibake-text", line.strip()))
            if typography and EM_DASH.search(line):
                violations.append(Violation(relative, number, "em-dash-punctuation", line.strip()))
        for number, window in _soft_leakage_windows(lines):
            if not SOFT_LEAKAGE_FALSE_CLAIM.search(window):
                continue
            key = (number, "soft-leakage-false-claim")
            if key in seen_soft:
                continue
            seen_soft.add(key)
            violations.append(
                Violation(relative, number, "soft-leakage-false-claim", window.strip()[:200])
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    violations = lint_paths()
    for item in violations:
        print(f"{item.path}:{item.line}: {item.rule}: {item.text}")
    if violations:
        print(f"copy lint failed with {len(violations)} violation(s)", file=sys.stderr)
        return 1
    print("copy lint passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
