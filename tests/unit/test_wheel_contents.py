"""Packaging contract: wheels must ship files required to import BuildML."""

from __future__ import annotations

import io
import json
import tarfile
import zipfile
from importlib.resources import files
from pathlib import Path

import pytest

from buildml.explain import sync as sync_mod
from buildml.explain.sync import (
    MISSING_OPERATION_INDEX_MESSAGE,
    load_operation_index,
    public_session_operations,
)
from scripts.check_wheel_contents import (
    REQUIRED_WHEEL_MEMBERS,
    check_dist,
    forbidden_members,
    missing_required_members,
)

ROOT = Path(__file__).resolve().parents[2]


def test_package_data_declares_generated_index() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "explain/generated/*.json" in pyproject
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "buildml/explain/generated" in manifest
    assert "operation_index.json" in manifest or "*.json" in manifest


def test_required_wheel_members_include_operation_index() -> None:
    assert "buildml/explain/generated/operation_index.json" in REQUIRED_WHEEL_MEMBERS
    assert "buildml/py.typed" in REQUIRED_WHEEL_MEMBERS


def test_missing_required_members_detects_omitted_index() -> None:
    names = {
        "buildml/__init__.py",
        "buildml/_version.py",
        "buildml/py.typed",
        "buildml/explain/generated/__init__.py",
    }
    assert missing_required_members(names, sdist=False) == [
        "buildml/explain/generated/operation_index.json"
    ]


def test_forbidden_members_detects_bytecode() -> None:
    names = {
        "buildml/explain/generated/operation_index.json",
        "buildml/explain/generated/__pycache__/__init__.cpython-313.pyc",
    }
    leaked = forbidden_members(names)
    assert leaked == ["buildml/explain/generated/__pycache__/__init__.cpython-313.pyc"]


def test_sdist_accepts_version_prefixed_members() -> None:
    names = {
        "buildml-2.6.1/buildml/__init__.py",
        "buildml-2.6.1/buildml/_version.py",
        "buildml-2.6.1/buildml/py.typed",
        "buildml-2.6.1/buildml/explain/generated/__init__.py",
        "buildml-2.6.1/buildml/explain/generated/operation_index.json",
    }
    assert missing_required_members(names, sdist=True) == []


def test_check_dist_flags_wheel_without_index(tmp_path: Path) -> None:
    wheel = tmp_path / "buildml-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as handle:
        handle.writestr("buildml/__init__.py", "")
        handle.writestr("buildml/_version.py", "")
        handle.writestr("buildml/py.typed", "")
        handle.writestr("buildml/explain/generated/__init__.py", "")
    errors = check_dist(tmp_path)
    assert len(errors) == 1
    assert "operation_index.json" in errors[0]


def test_check_dist_accepts_complete_wheel_and_sdist(tmp_path: Path) -> None:
    wheel = tmp_path / "buildml-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as handle:
        for member in REQUIRED_WHEEL_MEMBERS:
            handle.writestr(member, "{}\n")
    sdist = tmp_path / "buildml-0.0.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as handle:
        for member in REQUIRED_WHEEL_MEMBERS:
            payload = b"{}\n"
            info = tarfile.TarInfo(name=f"buildml-0.0.0/{member}")
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    assert check_dist(tmp_path) == []


def test_operation_index_is_a_package_resource() -> None:
    resource = files("buildml.explain.generated").joinpath("operation_index.json")
    assert resource.is_file()
    payload = json.loads(resource.read_text(encoding="utf-8"))
    assert payload["n_operations"] == len(public_session_operations())
    assert load_operation_index()["n_operations"] == payload["n_operations"]


def test_load_operation_index_explicit_path_raises_for_missing_file(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "absent-operation_index.json"
    with pytest.raises(FileNotFoundError):
        load_operation_index(missing)


def test_load_operation_index_default_path_names_packaging_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Missing:
        def joinpath(self, name: str) -> _Missing:
            return self

        def read_text(self, encoding: str = "utf-8") -> str:
            raise FileNotFoundError("missing resource")

    monkeypatch.setattr(sync_mod, "files", lambda package: _Missing())
    monkeypatch.setattr(sync_mod, "OPERATION_INDEX_PATH", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="package-data"):
        load_operation_index()


def test_missing_index_message_names_the_wheel_contract() -> None:
    assert "package-data" in MISSING_OPERATION_INDEX_MESSAGE
    assert "2.6.1" in MISSING_OPERATION_INDEX_MESSAGE
    assert "operation_index.json" in MISSING_OPERATION_INDEX_MESSAGE
