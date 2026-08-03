#!/usr/bin/env python3
"""Add trusted parameter docs to load_* overlays; add missing catalog ops."""

from __future__ import annotations

import re
from pathlib import Path

OVERLAYS = Path(__file__).resolve().parents[1] / "buildml" / "explain" / "overlays"

TRUSTED_P = (
    '_p("trusted", "bool", '
    '"Must be True to deserialize pickle/joblib/torch payloads '
    '(default False).", required=False)'
)

LOAD_NAMES = {
    "load_model",
    "load_pipeline",
    "checkpoint_load",
    "reattach",
    "load_torch_bundle",
}


def patch_file(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    orig = src

    # Find each _operation("name", ...) block start and patch its parameters=
    pattern = re.compile(
        r'_operation\(\s*\n\s*"(?P<name>[^"]+)"',
        re.MULTILINE,
    )
    # Work from end so offsets stay valid — rewrite whole text via segments
    matches = list(pattern.finditer(src))
    for m in reversed(matches):
        name = m.group("name")
        if not (name.startswith("load_") or name in LOAD_NAMES):
            continue
        if name == "load_rag_bundle":
            continue  # no pickle
        # Find parameters= after this match until next _operation or end
        start = m.start()
        nxt = matches[matches.index(m) + 1].start() if matches.index(m) + 1 < len(matches) else len(src)
        # Actually matches is reversed iteration - use m.end search window
        window_end = len(src)
        for other in matches:
            if other.start() > start:
                window_end = min(window_end, other.start())
        chunk = src[start:window_end]
        if "trusted" in chunk:
            continue
        # Inject into parameters=(...)
        def inject(pm: re.Match[str]) -> str:
            body = pm.group(1).rstrip()
            if not body.strip():
                return f"parameters=({TRUSTED_P},),"
            if body.endswith(","):
                return f"parameters=({body}\n            {TRUSTED_P},),"
            return f"parameters=({body},\n            {TRUSTED_P},),"

        new_chunk, n = re.subn(
            r"parameters=\((.*?)\),",
            inject,
            chunk,
            count=1,
            flags=re.DOTALL,
        )
        if n:
            src = src[:start] + new_chunk + src[window_end:]

    if src != orig:
        path.write_text(src, encoding="utf-8", newline="\n")
        return True
    return False


def main() -> int:
    for path in sorted(OVERLAYS.glob("*.py")):
        if patch_file(path):
            print("patched", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
