#!/usr/bin/env python3
"""Ensure ops modules import Session under TYPE_CHECKING for cast()."""

from __future__ import annotations

from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "buildml" / "session"
IMPORT = "from buildml.session.session import Session"


def main() -> int:
    for path in sorted(OPS.glob("*_ops.py")):
        src = path.read_text(encoding="utf-8")
        if 'cast("Session"' not in src and "cast('Session'" not in src:
            continue
        if IMPORT in src:
            continue
        if "if TYPE_CHECKING:" in src:
            src = src.replace(
                "if TYPE_CHECKING:",
                f"if TYPE_CHECKING:\n    {IMPORT}",
                1,
            )
        else:
            if "TYPE_CHECKING" not in src:
                if "from typing import" in src:
                    src = src.replace(
                        "from typing import",
                        "from typing import TYPE_CHECKING,",
                        1,
                    )
                else:
                    src = src.replace(
                        "from __future__ import annotations\n",
                        "from __future__ import annotations\n\nfrom typing import TYPE_CHECKING\n",
                        1,
                    )
            src = src.replace(
                "from __future__ import annotations\n",
                "from __future__ import annotations\n\n"
                "if TYPE_CHECKING:\n"
                f"    {IMPORT}\n",
                1,
            )
            # Avoid double future if we already replaced typing — handle carefully
        path.write_text(src, encoding="utf-8", newline="\n")
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
