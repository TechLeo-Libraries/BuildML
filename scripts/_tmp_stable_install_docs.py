"""One-shot: point install callouts at stable 2.4.0 / bare pip install buildml."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files: list[Path] = []
for root in (ROOT / "guides", ROOT / "docs", ROOT / "examples"):
    if root.exists():
        files.extend(root.rglob("*.md"))
        files.extend(root.rglob("*.rst"))

replacements = [
    (
        r"Session 2\.x is on PyPI as pre-release `2\.4\.0a3`\. Pin it or use `--pre` — "
        r"plain `pip install buildml` still resolves legacy 1\.0\.9\.",
        "Install Session 2.x with `pip install buildml` (2.4.x). "
        "Legacy 1.x remains available as `pip install \"buildml==1.0.9\"`.",
    ),
    (
        r"Session 2\.x is on PyPI as pre-release `2\.4\.0a3`; plain `pip install buildml` "
        r"still resolves legacy 1\.0\.9\.",
        "Install with `pip install buildml` (Session 2.4.x).",
    ),
    (
        r"Session 2\.x is on PyPI as pre-release ``2\.4\.0a3`` "
        r"\(pin or ``--pre``; plain ``pip install buildml`` still resolves 1\.0\.9\)\.",
        "Install with ``pip install buildml`` (Session 2.4.x).",
    ),
    (
        r"Session 2\.x is on PyPI as pre-release ``2\.4\.0a3`` \(pin or ``--pre``\)\.",
        "Install with ``pip install buildml`` (Session 2.4.x).",
    ),
    (
        r'Install with `pip install "buildml==2\.4\.0a3"` \(or `--pre`\)\. '
        r"See \[installation\]\(\.\./docs/installation\.rst\)\.",
        "Install with `pip install buildml`. See [installation](../docs/installation.rst).",
    ),
    (r'pip install "buildml==2\.4\.0a3"', 'pip install buildml'),
    (r"buildml==2\.4\.0a3", "buildml"),
    (r"\b2\.4\.0a3\b", "2.4.0"),
]


def main() -> None:
    updated: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "2.4.0a3" not in text and "pre-release" not in text:
            continue
        orig = text
        for pat, repl in replacements:
            text = re.sub(pat, repl, text)
        if text != orig:
            path.write_text(text, encoding="utf-8")
            updated.append(str(path.relative_to(ROOT)))
    # migration doc status line
    mig = ROOT / "docs" / "session-facade-migration.md"
    if mig.exists():
        t = mig.read_text(encoding="utf-8")
        n = t.replace("2.4.0a3", "2.4.0")
        if n != t:
            mig.write_text(n, encoding="utf-8")
            updated.append("docs/session-facade-migration.md")
    print(f"updated={len(updated)}")
    for item in updated:
        print(item)


if __name__ == "__main__":
    main()
