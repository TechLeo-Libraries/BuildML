"""Capture or verify the exact tracked and nonignored source under review.

A source digest identifies uncommitted candidates without pretending they are
commits. Generated evidence belongs under ignored artifacts/, outside the input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_manifest(root: Path = ROOT) -> dict:
    """Hash every existing tracked or nonignored source file in stable order."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True,
    )
    files = {}
    for name in sorted(set(result.stdout.decode("utf-8").split("\0")) - {""}):
        path = root / name
        if path.is_symlink():
            raise ValueError(f"Source snapshots require regular files: {name}")
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/release-review")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    current = source_manifest()
    if args.verify:
        saved = json.loads(args.verify.read_text(encoding="utf-8"))
        if saved["source_sha256"] != current["source_sha256"]:
            print("Source changed: create a new candidate and revalidate affected evidence.")
            return 1
        print(f"Source matches candidate {current['source_sha256']}")
        return 0
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must be inside the ignored artifacts directory.")
    destination = destination / current["source_sha256"]
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(destination / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, expected in current["files"].items():
            data = (ROOT / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"Source changed during capture: {name}")
            archive.writestr(name, data)
    (destination / "manifest.json").write_text(
        json.dumps(current, indent=2) + "\n", encoding="utf-8"
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
