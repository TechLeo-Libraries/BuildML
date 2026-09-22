"""Validate claim-register references; this does not approve the claims."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    register = json.loads((ROOT / "review/release/claims.json").read_text(encoding="utf-8"))
    seen = set()
    errors = []
    for claim in register["claims"]:
        identifier = claim["id"]
        if identifier in seen:
            errors.append(f"Duplicate claim: {identifier}")
        seen.add(identifier)
        paths = [source["path"] for source in claim["public_sources"]]
        paths += claim["implementation_paths"] + claim["test_paths"]
        paths += [selector.split("::")[0] for selector in claim.get("test_selectors", [])]
        for name in paths:
            target = (ROOT / name).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file():
                errors.append(f"{identifier}: missing or external source reference {name}")
        if not claim["assumptions_and_limits"]:
            errors.append(f"{identifier}: specify assumptions and limits")
    print(f"Checked {len(seen)} claim groups; {len(errors)} reference errors")
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
