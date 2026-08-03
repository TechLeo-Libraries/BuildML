#!/usr/bin/env python3
"""Patch domain checkpoint loaders to require trusted=True before joblib.load."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "buildml"

IMPORT_LINE = "from buildml.core.serialization import joblib_load_trusted"


def patch_file(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    original = src
    if "joblib.load" not in src and "torch.load" not in src:
        return False

    if "joblib_load_trusted" not in src and "joblib.load" in src:
        # Insert import after other buildml.core imports if present.
        if "from buildml.core.errors import" in src:
            src = src.replace(
                "from buildml.core.errors import",
                IMPORT_LINE + "\nfrom buildml.core.errors import",
                1,
            )
        elif "from buildml.core" in src:
            # after first buildml.core import block line
            src = IMPORT_LINE + "\n" + src
        else:
            # after __future__
            src = re.sub(
                r"(from __future__ import annotations\n)",
                r"\1\n" + IMPORT_LINE + "\n",
                src,
                count=1,
            )

    # Add trusted kwarg to load_* function signatures that lack it.
    def add_trusted_sig(match: re.Match[str]) -> str:
        sig = match.group(0)
        if "trusted" in sig:
            return sig
        # Insert before closing paren of def line(s) — handle single-line first.
        if sig.rstrip().endswith(":"):
            inner = sig[:-1]  # drop :
            if inner.rstrip().endswith(")"):
                # before last )
                idx = inner.rfind(")")
                insert = ", *, trusted: bool = False" if "*" not in inner else ", trusted: bool = False"
                # if already has * somewhere, just add trusted=
                if "*" in inner:
                    insert = ", trusted: bool = False"
                else:
                    insert = ", *, trusted: bool = False"
                return inner[:idx] + insert + ")" + ":"
        return sig

    src = re.sub(
        r"def load_\w+\([^)]*\)\s*(?:->[^:]+)?:",
        add_trusted_sig,
        src,
    )
    # Multi-line signatures: def load_...(path: ...)\n -> X:
    src = re.sub(
        r"def (load_\w+)\(\s*path:\s*str\s*\|\s*Path\s*,?\s*\)",
        r"def \1(path: str | Path, *, trusted: bool = False)",
        src,
    )

    # Replace joblib.load(x) with joblib_load_trusted(x, trusted=trusted, artifact=...)
    def repl_joblib(m: re.Match[str]) -> str:
        arg = m.group(1).strip()
        return f"joblib_load_trusted({arg}, trusted=trusted, artifact={arg!s})"

    # simpler: joblib.load(plan_path) -> joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    src = re.sub(
        r"joblib\.load\(([^)]+)\)",
        r'joblib_load_trusted(\1, trusted=trusted, artifact="joblib plan")',
        src,
    )

    # Torch loads: insert require_trusted before torch.load if trusted in signature
    if "torch.load" in src and "require_trusted_deserialize" not in src:
        if "joblib_load_trusted" in src:
            src = src.replace(
                "from buildml.core.serialization import joblib_load_trusted",
                "from buildml.core.serialization import joblib_load_trusted, require_trusted_deserialize",
            )
        else:
            src = re.sub(
                r"(from __future__ import annotations\n)",
                r"\1\nfrom buildml.core.serialization import require_trusted_deserialize\n",
                src,
                count=1,
            )
        src = re.sub(
            r"(\n\s*)([^\n]*torch\.load\()",
            r"\1require_trusted_deserialize(trusted=trusted, artifact='torch payload', path=path)\n\1\2",
            src,
        )
        # Ensure load functions with torch.load have trusted param — handled above for load_*

    if src != original:
        path.write_text(src, encoding="utf-8", newline="\n")
        return True
    return False


def main() -> int:
    changed = []
    for path in sorted(ROOT.rglob("checkpoint.py")):
        if patch_file(path):
            changed.append(path)
            print("patched", path.relative_to(ROOT.parent))
    # pipeline + persist + checkpoint bundle
    for rel in (
        "pipeline/bundle.py",
        "pipeline/persist.py",
        "checkpoint/bundle.py",
    ):
        p = ROOT / rel
        if p.exists() and patch_file(p):
            changed.append(p)
            print("patched", rel)
    print(f"changed {len(changed)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
