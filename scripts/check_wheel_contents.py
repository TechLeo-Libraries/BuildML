"""Fail if a built wheel/sdist is missing files required to import BuildML.

Editable and source checkouts hide this class of defect: the files sit on
disk next to the package even when setuptools never put them in the
artifact. CI must inspect ``dist/`` after ``python -m build``.

Usage:
  python scripts/check_wheel_contents.py          # checks dist/
  python scripts/check_wheel_contents.py path/to/dist
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

# Paths as they appear inside a wheel (no version prefix).
REQUIRED_WHEEL_MEMBERS: tuple[str, ...] = (
    "buildml/__init__.py",
    "buildml/_version.py",
    "buildml/py.typed",
    "buildml/explain/generated/__init__.py",
    "buildml/explain/generated/operation_index.json",
)

# Same contract inside an sdist, which prefixes ``buildml-<version>/``.
REQUIRED_SDIST_SUFFIXES: tuple[str, ...] = REQUIRED_WHEEL_MEMBERS
FORBIDDEN_MEMBER_MARKERS: tuple[str, ...] = ("__pycache__/", ".pyc", ".pyo")


def archive_member_names(archive: Path) -> set[str]:
    """Return member paths stored in a wheel or sdist."""
    name = archive.name
    if name.endswith(".whl"):
        with zipfile.ZipFile(archive) as handle:
            return set(handle.namelist())
    if name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as handle:
            return {member.name for member in handle.getmembers() if member.isfile()}
    raise ValueError(f"Unsupported distribution artifact: {archive}")


def missing_required_members(names: set[str], *, sdist: bool) -> list[str]:
    """Return required paths that are absent from ``names``."""
    missing: list[str] = []
    for required in REQUIRED_WHEEL_MEMBERS if not sdist else REQUIRED_SDIST_SUFFIXES:
        if required in names:
            continue
        if sdist and any(
            member == required or member.endswith("/" + required) for member in names
        ):
            continue
        missing.append(required)
    return missing


def forbidden_members(names: set[str]) -> list[str]:
    """Return bytecode / cache members that must not ship."""
    return sorted(
        name
        for name in names
        if any(marker in name.replace("\\", "/") for marker in FORBIDDEN_MEMBER_MARKERS)
    )


def discover_archives(dist_dir: Path) -> list[Path]:
    """Wheels and sdists under ``dist_dir``, sorted for stable output."""
    if not dist_dir.is_dir():
        return []
    found = [
        path
        for path in dist_dir.iterdir()
        if path.is_file() and (path.name.endswith(".whl") or path.name.endswith(".tar.gz"))
    ]
    return sorted(found)


def check_dist(dist_dir: Path) -> list[str]:
    """Return human-readable errors for every artifact under ``dist_dir``."""
    archives = discover_archives(dist_dir)
    if not archives:
        return [f"No wheel or sdist found in {dist_dir}"]
    errors: list[str] = []
    for archive in archives:
        names = archive_member_names(archive)
        sdist = archive.name.endswith(".tar.gz")
        kind = "sdist" if sdist else "wheel"
        missing = missing_required_members(names, sdist=sdist)
        if missing:
            listed = ", ".join(missing)
            errors.append(f"{archive.name} ({kind}) missing required members: {listed}")
        leaked = forbidden_members(names)
        if leaked:
            listed = ", ".join(leaked[:8])
            errors.append(f"{archive.name} ({kind}) ships cache/bytecode: {listed}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dist_dir",
        nargs="?",
        default="dist",
        help="Directory that holds built wheels/sdists (default: dist).",
    )
    args = parser.parse_args(argv)
    dist_dir = Path(args.dist_dir)
    if not dist_dir.is_absolute():
        dist_dir = (Path.cwd() / dist_dir).resolve()
    errors = check_dist(dist_dir)
    if errors:
        print("Distribution artifact check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "\nFix: add the files to [tool.setuptools.package-data] and MANIFEST.in, "
            "then rebuild with python -m build.",
            file=sys.stderr,
        )
        return 1
    archives = discover_archives(dist_dir)
    print(
        "distribution artifacts ok "
        f"({len(archives)} file(s); operation_index.json present)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
