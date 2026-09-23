"""Regression checks for release approval and immutable artifact identity."""

import copy
import hashlib
import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "release_gate", Path(__file__).parents[2] / "scripts/check_release_gate.py"
)
assert _SPEC and _SPEC.loader
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


def protected_environment():
    return {"name": "pypi-release", "can_admins_bypass": False,
            "protection_rules": [{"type": "required_reviewers", "prevent_self_review": True,
                                  "reviewers": [{"type": "User", "reviewer": {"id": 42}}]}]}


@pytest.mark.parametrize("prevent_self_review", [True, False])
def test_protected_environment(prevent_self_review):
    payload = protected_environment()
    payload["protection_rules"][0]["prevent_self_review"] = prevent_self_review
    gate.validate_environment(payload)


@pytest.mark.parametrize("change", [
    {"name": "other"}, {"can_admins_bypass": True}, {"can_admins_bypass": None},
    {"protection_rules": []},
    {"protection_rules": [{"type": "required_reviewers", "prevent_self_review": None,
                           "reviewers": [42]}]},
    {"protection_rules": [{"type": "required_reviewers", "prevent_self_review": True,
                           "reviewers": []}]},
])
def test_unprotected_environment_rejected(change):
    payload = copy.deepcopy(protected_environment())
    payload.update(change)
    with pytest.raises(ValueError):
        gate.validate_environment(payload)


def make_candidate(root, source_version="2.6.3"):
    metadata = b"Name: buildml\nVersion: 2.6.3\n"
    wheel = root / "buildml-2.6.3-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("buildml-2.6.3.dist-info/METADATA", metadata)
    sdist = root / "buildml-2.6.3.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        data = f"Name: buildml\nVersion: {source_version}\n".encode()
        member = tarfile.TarInfo("buildml-2.6.3/PKG-INFO")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    (root / "source-commit.txt").write_text("a" * 40)
    (root / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  candidate/{path.name}\n"
        for path in [wheel, sdist]))
    return wheel


def test_candidate_identity(tmp_path):
    make_candidate(tmp_path)
    gate.validate_candidate(tmp_path, "a" * 40, "v2.6.3")


@pytest.mark.parametrize("commit,tag", [("b" * 40, "v2.6.3"), ("a" * 40, "main"),
                                      ("a" * 40, "v2.6.2")])
def test_wrong_candidate_identity(tmp_path, commit, tag):
    make_candidate(tmp_path)
    with pytest.raises(ValueError):
        gate.validate_candidate(tmp_path, commit, tag)


def test_tampered_artifact(tmp_path):
    wheel = make_candidate(tmp_path)
    wheel.write_bytes(wheel.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.validate_candidate(tmp_path, "a" * 40, "v2.6.3")


def test_mismatched_archive_versions(tmp_path):
    make_candidate(tmp_path, source_version="2.6.2")
    with pytest.raises(ValueError, match="versions disagree"):
        gate.validate_candidate(tmp_path, "a" * 40, "v2.6.3")
