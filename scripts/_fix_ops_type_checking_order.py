#!/usr/bin/env python3
"""Fix ops modules where TYPE_CHECKING block was inserted before its import."""

from __future__ import annotations

import re
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / "buildml" / "session"

BAD = re.compile(
    r"from __future__ import annotations\n\n"
    r"if TYPE_CHECKING:\n"
    r"    from buildml\.session\.session import Session\n\n",
)


def main() -> int:
    for path in sorted(OPS.glob("*_ops.py")):
        src = path.read_text(encoding="utf-8")
        if not BAD.match(src) and "if TYPE_CHECKING:\n    from buildml.session.session import Session" not in src[:400]:
            # Still may have early TYPE_CHECKING before typing import
            pass
        # Remove early misplaced block
        new = BAD.sub("from __future__ import annotations\n\n", src)
        if "from buildml.session.session import Session" not in new:
            # Ensure TYPE_CHECKING import and Session import after typing import
            if "TYPE_CHECKING" not in new:
                if "from typing import" in new:
                    new = new.replace(
                        "from typing import",
                        "from typing import TYPE_CHECKING,",
                        1,
                    )
                else:
                    new = new.replace(
                        "from __future__ import annotations\n",
                        "from __future__ import annotations\n\nfrom typing import TYPE_CHECKING\n",
                        1,
                    )
            if "if TYPE_CHECKING:" not in new:
                # Insert after first import block
                m = re.search(r"(from typing import[^\n]+\n)", new)
                if m:
                    insert_at = m.end()
                    new = (
                        new[:insert_at]
                        + "\nif TYPE_CHECKING:\n"
                        + "    from buildml.session.session import Session\n"
                        + new[insert_at:]
                    )
            elif "from buildml.session.session import Session" not in new:
                new = new.replace(
                    "if TYPE_CHECKING:",
                    "if TYPE_CHECKING:\n    from buildml.session.session import Session",
                    1,
                )
        else:
            # Session import exists — ensure typing imports TYPE_CHECKING before use
            # Find first if TYPE_CHECKING and ensure typing import precedes it
            tc_idx = new.find("if TYPE_CHECKING:")
            typing_idx = new.find("from typing import")
            if tc_idx != -1 and (typing_idx == -1 or typing_idx > tc_idx):
                # Move the TYPE_CHECKING Session block to after typing import
                block_m = re.search(
                    r"if TYPE_CHECKING:\n(?:    .+\n)+",
                    new,
                )
                if block_m and block_m.start() < (typing_idx if typing_idx != -1 else 10**9):
                    block = block_m.group(0)
                    new = new[: block_m.start()] + new[block_m.end() :]
                    if "from typing import" in new:
                        if "TYPE_CHECKING" not in new.split("from typing import", 1)[1].split("\n", 1)[0]:
                            new = new.replace(
                                "from typing import",
                                "from typing import TYPE_CHECKING,",
                                1,
                            )
                        new = new.replace(
                            re.search(r"from typing import[^\n]+\n", new).group(0),  # type: ignore[union-attr]
                            re.search(r"from typing import[^\n]+\n", new).group(0)  # type: ignore[union-attr]
                            + "\n"
                            + block,
                            1,
                        )
                    else:
                        new = new.replace(
                            "from __future__ import annotations\n",
                            "from __future__ import annotations\n\nfrom typing import TYPE_CHECKING\n\n"
                            + block,
                            1,
                        )
        if new != src:
            path.write_text(new, encoding="utf-8", newline="\n")
            print("fixed", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
