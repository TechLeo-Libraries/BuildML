"""Fail closed on release identity and GitHub environment protection checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path
from urllib.request import Request, urlopen


def validate_environment(payload: dict) -> None:
    """Require an explicitly protected release environment."""
    if payload.get("name") != "pypi-release":
        raise ValueError("Expected existing pypi-release environment")
    if payload.get("can_admins_bypass") is not False:
        raise ValueError("Administrator bypass must be explicitly disabled")
    rules = payload.get("protection_rules", [])
    reviewers = [rule for rule in rules if rule.get("type") == "required_reviewers"]
    if len(reviewers) != 1:
        raise ValueError("Exactly one required-reviewers rule is required")
    rule = reviewers[0]
    if not isinstance(rule.get("prevent_self_review"), bool) or not rule.get("reviewers"):
        raise ValueError("Named reviewers and an explicit self-review policy are required")


def validate_candidate(directory: Path, commit: str, tag: str | None) -> None:
    """Check source commit, archive metadata, and the recorded artifact hashes."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected exact Git commit SHA")
    if (directory / "source-commit.txt").read_text().strip() != commit:
        raise ValueError("Candidate source commit does not match this run")
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    expected = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = Path(name.removeprefix("*")).name
        if name in expected or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid artifact hash manifest")
        expected[name] = digest
    if set(expected) != {wheels[0].name, sdists[0].name}:
        raise ValueError("Manifest must cover exactly both distribution archives")
    for path in wheels + sdists:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[path.name]:
            raise ValueError("Distribution hash mismatch")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("Ambiguous wheel metadata")
        wheel_metadata = Parser().parsestr(archive.read(names[0]).decode())
    with tarfile.open(sdists[0]) as archive:
        names = [member for member in archive.getmembers()
                 if member.name.endswith("/PKG-INFO") and member.name.count("/") == 1]
        if len(names) != 1:
            raise ValueError("Ambiguous source metadata")
        stream = archive.extractfile(names[0])
        if stream is None:
            raise ValueError("Missing source metadata")
        source_metadata = Parser().parsestr(stream.read().decode())
    version = wheel_metadata["Version"]
    if (wheel_metadata["Name"] != "buildml" or source_metadata["Name"] != "buildml"
            or not version or source_metadata["Version"] != version):
        raise ValueError("Distribution names or versions disagree")
    if tag is not None and tag != "v" + version:
        raise ValueError("Publishing requires a tag matching the distribution version")


def main() -> None:
    """Validate candidate files and, when requested, live environment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--tag")
    parser.add_argument("--environment", action="store_true")
    args = parser.parse_args()
    if args.candidate:
        validate_candidate(args.candidate, args.commit, args.tag)
    if args.environment:
        repository = os.environ["GITHUB_REPOSITORY"]
        token = os.environ["GH_TOKEN"]
        request = Request(
            f"https://api.github.com/repos/{repository}/environments/pypi-release",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28"},
        )
        with urlopen(request, timeout=30) as response:
            validate_environment(json.load(response))
    if not args.candidate and not args.environment:
        parser.error("At least one check is required")


if __name__ == "__main__":
    main()
