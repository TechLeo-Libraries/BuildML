#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "buildml" / "session"


def main() -> int:
    for path in sorted(OPS.glob("*_ops.py")):
        src = path.read_text(encoding="utf-8")
        needs_session = (
            'cast("Session"' in src
            or "cast('Session'" in src
            or "from buildml.session.session import Session" in src
        )
        new = re.sub(
            r"\nif TYPE_CHECKING:\n    from buildml\.session\.session import Session\n",
            "\n",
            src,
        )
        new = re.sub(
            r"(from __future__ import annotations\n)\nif TYPE_CHECKING:\n"
            r"    from buildml\.session\.session import Session\n+",
            r"\1\n",
            new,
        )
        if "TYPE_CHECKING" not in new:
            if "from typing import" in new:
                new = new.replace(
                    "from typing import", "from typing import TYPE_CHECKING,", 1
                )
            else:
                new = new.replace(
                    "from __future__ import annotations\n",
                    "from __future__ import annotations\n\n"
                    "from typing import TYPE_CHECKING\n",
                    1,
                )
        if needs_session and "from buildml.session.session import Session" not in new:
            m = re.search(r"from typing import[^\n]+\n", new)
            if m:
                new = (
                    new[: m.end()]
                    + "\nif TYPE_CHECKING:\n"
                    + "    from buildml.session.session import Session\n"
                    + new[m.end() :]
                )
        if new != src:
            path.write_text(new, encoding="utf-8", newline="\n")
            print("fixed", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
