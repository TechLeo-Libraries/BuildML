"""Sync / check teaching surfaces: Session ↔ operation index ↔ catalog ↔ AI tools.

Usage:
  python scripts/sync_teaching_surface.py --write   # regenerate operation_index.json
  python scripts/sync_teaching_surface.py --check   # CI gate (default)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from buildml.explain.sync import (  # noqa: E402
    OPERATION_INDEX_PATH,
    check_teaching_surface,
    write_operation_index,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write",
        action="store_true",
        help="Regenerate buildml/explain/generated/operation_index.json from Session.",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Fail when Session / index / catalog / AI tools drift (default).",
    )
    args = parser.parse_args(argv)

    if args.write:
        path = write_operation_index()
        report = check_teaching_surface(index_path=path)
        print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
        for warning in report.warnings:
            print(f"warning: {warning}")
        if not report.ok:
            print("Regenerated index, but remaining drift must be fixed:", file=sys.stderr)
            for error in report.errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print("teaching surface sync ok")
        return 0

    report = check_teaching_surface()
    for warning in report.warnings:
        print(f"warning: {warning}")
    if not report.ok:
        print("teaching surface drift detected:", file=sys.stderr)
        for error in report.errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "\nFix: update overlays under buildml/explain/overlays/ "
            "and/or AI tools, then run:\n"
            "  python scripts/sync_teaching_surface.py --write",
            file=sys.stderr,
        )
        return 1
    print(
        f"teaching surface sync ok "
        f"(index={OPERATION_INDEX_PATH.relative_to(ROOT)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
