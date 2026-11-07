"""Measure baseline-missing lines exercised by a run, without combining coverage.

The current root must contain exactly the source used for the observed run.
Unchanged lines are mapped across edits; changed/new lines are reported separately.
Profile differences mean subset reach is not a projected whole-suite percentage.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


def package_path(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    if normalized.startswith("buildml/"):
        return normalized
    marker = normalized.lower().rfind("/buildml/")
    return "buildml/" + normalized[marker + len("/buildml/") :] if marker >= 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_json", type=Path)
    parser.add_argument("observed_json", type=Path)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--current-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observed-scope", choices=("subset", "whole-suite"), default="subset")
    args = parser.parse_args()
    root = args.current_root.resolve()
    repository = args.repo_root.resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", args.baseline_ref], cwd=repository, text=True
    ).strip()
    archive_bytes = subprocess.check_output(["git", "archive", revision], cwd=repository)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
        sources = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
            and member.name.startswith("buildml/")
            and member.name.endswith(".py")
        }
    baseline_data = json.loads(args.baseline_json.read_text(encoding="utf-8-sig"))
    observed_data = json.loads(args.observed_json.read_text(encoding="utf-8-sig"))
    baseline = {
        package_path(name): information for name, information in baseline_data["files"].items()
    }
    rows, unmapped = [], []
    for name, information in observed_data["files"].items():
        path = package_path(name)
        if path is None:
            continue
        current_path = root / path
        if not current_path.is_file():
            unmapped.append(path)
            continue
        current_bytes = current_path.read_bytes()
        before = sources.get(path, b"").decode("utf-8-sig").splitlines()
        after = current_bytes.decode("utf-8-sig").splitlines()
        mapping = {}
        for block in difflib.SequenceMatcher(
            a=before, b=after, autojunk=False
        ).get_matching_blocks():
            mapping.update({block.b + i + 1: block.a + i + 1 for i in range(block.size)})
        executed = set(information["executed_lines"])
        prior_missing = set(baseline.get(path, {}).get("missing_lines", []))
        reached = sorted({mapping[line] for line in executed if line in mapping} & prior_missing)
        changed = sorted(executed - set(mapping))
        if reached or changed:
            rows.append(
                {
                    "file": path,
                    "previously_missing_lines_exercised": reached,
                    "count": len(reached),
                    "changed_or_added_executed_lines": changed,
                    "current_source_sha256": hashlib.sha256(current_bytes).hexdigest(),
                }
            )
    rows.sort(key=lambda row: -row["count"])
    result = {
        "baseline_source": revision,
        "observed_scope": args.observed_scope,
        "baseline_totals": baseline_data["totals"],
        "previously_missing_unchanged_lines_exercised": sum(row["count"] for row in rows),
        "changed_or_added_executed_lines": sum(
            len(row["changed_or_added_executed_lines"]) for row in rows
        ),
        "qualification": "Not a combined coverage percentage. Scope/profile differences and full-suite results must be assessed separately.",
        "unmapped_files": unmapped,
        "files": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Previously missing unchanged lines exercised: {result['previously_missing_unchanged_lines_exercised']}"
    )
    print(f"Changed/new executed lines: {result['changed_or_added_executed_lines']}")
    print(result["qualification"])
    if unmapped:
        print(f"Unmapped files: {len(unmapped)}; inspect the report before interpreting counts.")


if __name__ == "__main__":
    main()
