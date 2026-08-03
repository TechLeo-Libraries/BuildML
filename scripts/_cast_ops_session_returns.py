#!/usr/bin/env python3
"""Cast bare ``return session`` in ops modules to satisfy warn_return_any."""

from __future__ import annotations

import re
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "buildml" / "session"


def main() -> int:
    for path in sorted(OPS.glob("*_ops.py")):
        src = path.read_text(encoding="utf-8")
        orig = src
        src = re.sub(
            r"^(\s*)return session\s*$",
            r'\1return cast("Session", session)',
            src,
            flags=re.M,
        )
        if src == orig:
            continue
        if not re.search(r"from typing import[^\n]*\bcast\b", src):
            if "from typing import" in src:
                src = src.replace("from typing import", "from typing import cast,", 1)
            else:
                src = src.replace(
                    "from __future__ import annotations\n",
                    "from __future__ import annotations\n\nfrom typing import cast\n",
                    1,
                )
        path.write_text(src, encoding="utf-8", newline="\n")
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
