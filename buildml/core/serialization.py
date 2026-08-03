"""Trusted deserialization gates for pickle / joblib / torch payloads.

BuildML persists fitted plans with joblib (and occasionally ``torch.load`` /
``torch.jit.load``). Those formats can execute arbitrary code on load. This
module does **not** make untrusted pickles safe: it makes unsafe loads an
explicit opt-in and adds controllable integrity / path checks around them.

Callers must pass ``trusted=True`` only for artifacts they created or fully
trust. Prefer JSON sidecars / parquet / ``data_only=True`` checkpoint loads when
the fitted plan is not needed.

Integrity hashes (``sha256``) detect *tampering after save*; they do **not**
protect against a malicious author who already controlled the bytes at save
time. A caller who passes ``trusted=True`` for an attacker-controlled artifact
still executes code.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from buildml.core.errors import ValidationError

# Schemes that look like remote / virtual URIs: refuse before any deserialize.
_REFUSED_URI_SCHEMES = frozenset(
    {
        "http",
        "https",
        "ftp",
        "ftps",
        "s3",
        "gs",
        "azure",
        "az",
        "hdfs",
        "ssh",
        "sftp",
        "data",
    }
)


def require_trusted_deserialize(
    *,
    trusted: bool,
    artifact: str,
    path: str | Path,
) -> None:
    """Refuse deserialization unless the caller opts in with ``trusted=True``.

    Pickle/joblib/torch payloads can execute arbitrary code. This helper does
    not sanitize those formats: it only blocks accidental loads until the
    caller asserts trust.

    Parameters
    ----------
    trusted:
        Explicit acknowledgement that ``path`` is a first-party or otherwise
        fully trusted artifact. Default callers must pass ``True`` only after
        that judgement.
    artifact:
        Short name used in the error message (for example ``\"plans.joblib\"``
        or ``\"anomaly bundle\"``).
    path:
        Filesystem location about to be deserialized.

    Raises
    ------
    ValidationError
        When ``trusted`` is false. Nothing is read from disk in that case.
    """
    if trusted:
        return
    resolved = Path(path)
    raise ValidationError(
        f"Refusing to deserialize {artifact} at '{resolved}'. "
        "Pass trusted=True only for artifacts you created or fully trust "
        "(pickle/joblib/torch can execute code on load)."
    )


def assert_local_load_path(path: str | Path, *, artifact: str = "artifact") -> Path:
    """Refuse non-local or URI-shaped load locations before any deserialize.

    BuildML loaders only accept ordinary filesystem paths. Strings that look
    like remote URIs (``https://...``, ``s3://...``) or ``file://`` URLs are
    rejected. This is a path-shape gate, not a sandbox: a local path under
    attacker control is still unsafe once ``trusted=True``.

    Parameters
    ----------
    path:
        Candidate load location.
    artifact:
        Label for error messages.

    Returns
    -------
    Path
        Normalised local path (expanduser applied; not required to exist yet).

    Raises
    ------
    ValidationError
        When ``path`` uses a refused URI scheme or is empty.
    """
    raw = os.fspath(path).strip()
    if not raw:
        raise ValidationError(f"Refusing to load {artifact}: empty path.")

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    # Windows drive letters (C:) look like schemes to urlparse: allow single-letter.
    if scheme and len(scheme) > 1:
        if scheme == "file":
            raise ValidationError(
                f"Refusing to load {artifact} from file URI '{raw}'. "
                "Pass a plain filesystem path instead."
            )
        if scheme in _REFUSED_URI_SCHEMES:
            raise ValidationError(
                f"Refusing to load {artifact} from URI scheme '{scheme}://'. "
                "Only local filesystem paths are accepted."
            )
        if "://" in raw:
            raise ValidationError(
                f"Refusing to load {artifact} from URI-shaped path '{raw}'. "
                "Only local filesystem paths are accepted."
            )

    return Path(raw).expanduser()


def sha256_file(path: str | Path) -> str:
    """Compute the hex SHA-256 digest of a file for integrity checks.

    Use when recording or verifying payload hashes in bundle metadata.
    Integrity detects tampering after save; it is not authenticity.

    Parameters
    ----------
    path:
        Filesystem file to hash.

    Returns
    -------
    str
        Lowercase hex digest.

    Raises
    ------
    ValidationError
        When ``path`` is not a regular file.
    """
    target = Path(path)
    if not target.is_file():
        raise ValidationError(f"Cannot hash missing file: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str | None, *, artifact: str = "payload") -> None:
    """Verify an optional SHA-256 digest; no-op when ``expected`` is absent.

    Integrity is not authenticity: a matching hash only shows the bytes match
    what was recorded at save time. It does not make a malicious trusted
    artifact safe.

    Parameters
    ----------
    path:
        File to hash.
    expected:
        Hex digest from the bundle manifest / ``meta.json``. ``None`` or empty
        skips verification (older bundles).
    artifact:
        Label for error messages.

    Raises
    ------
    ValidationError
        When ``expected`` is set and does not match the file on disk.
    """
    if expected is None:
        return
    needle = str(expected).strip().lower()
    if not needle:
        return
    actual = sha256_file(path)
    if actual != needle:
        raise ValidationError(
            f"Integrity check failed for {artifact} at '{Path(path)}': "
            f"sha256 mismatch (expected {needle[:12]}..., got {actual[:12]}...). "
            "The file may have been tampered with after save. "
            "Integrity is not safety from a malicious author: refuse the load."
        )


def verify_manifest_hashes(
    root: str | Path,
    hashes: dict[str, Any] | None,
    *,
    required_members: tuple[str, ...] | None = None,
) -> None:
    """Verify relative-path to sha256 entries under a bundle or checkpoint root.

    Call after reading ``MANIFEST.json``. Missing ``hashes`` (legacy artifacts)
    skips verification. When hashes are present, each member file is checked.

    Parameters
    ----------
    root:
        Bundle or checkpoint directory.
    hashes:
        Mapping of relative paths to hex digests. ``None`` / empty skips.
    required_members:
        Optional relative paths that must appear in ``hashes`` when hashing is
        present.

    Raises
    ------
    ValidationError
        On mismatch, missing required member, or non-file member path.
    """
    if not hashes:
        return
    base = Path(root)
    if required_members:
        for member in required_members:
            if member not in hashes:
                raise ValidationError(
                    f"MANIFEST hashes missing required member {member!r} under {base}."
                )
    for rel, expected in hashes.items():
        member_path = base / str(rel)
        verify_sha256(member_path, str(expected), artifact=f"manifest member '{rel}'")


def read_json_sidecar(path: str | Path, *, artifact: str = "meta.json") -> dict[str, Any]:
    """Load a JSON metadata sidecar without executing code.

    Use this for plan summaries, metrics, and format labels. Fitted estimators
    remain behind :func:`joblib_load_trusted`.

    Parameters
    ----------
    path:
        Path to a ``.json`` file.
    artifact:
        Label for error messages.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ValidationError
        When the path is refused, missing, or not a JSON object.
    """
    target = assert_local_load_path(path, artifact=artifact)
    if not target.is_file():
        raise ValidationError(f"Missing {artifact} at '{target}'.")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {artifact} at '{target}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"{artifact} at '{target}' must be a JSON object.")
    return payload


def attach_payload_sha256(meta: dict[str, Any], payload_path: str | Path) -> dict[str, Any]:
    """Return a copy of ``meta`` with ``payload_sha256`` set from the payload file.

    Call after writing the joblib/torch file so loaders can verify integrity
    when the hash is present.

    Parameters
    ----------
    meta:
        Existing metadata mapping (not mutated).
    payload_path:
        Joblib / torch file written beside the metadata.

    Returns
    -------
    dict
        New metadata dict including ``payload_sha256``.
    """
    out = dict(meta)
    out["payload_sha256"] = sha256_file(payload_path)
    return out


def joblib_load_trusted(
    path: str | Path,
    *,
    trusted: bool,
    artifact: str = "joblib payload",
    expected_sha256: str | None = None,
) -> Any:
    """Load a joblib file after path, trust, and optional integrity checks.

    Shared entry point for domain checkpoint and pipeline loaders so every
    pickle path goes through the same opt-in gate.

    Parameters
    ----------
    path:
        Path to a ``.joblib`` (or joblib-compatible) file.
    trusted:
        Must be ``True`` to proceed. See :func:`require_trusted_deserialize`.
    artifact:
        Label for error messages.
    expected_sha256:
        Optional hex digest from bundle metadata. Verified when present.

    Returns
    -------
    Any
        The object returned by ``joblib.load``.

    Raises
    ------
    ValidationError
        When the path is refused, ``trusted`` is false, or the hash mismatches.
    """
    target = assert_local_load_path(path, artifact=artifact)
    require_trusted_deserialize(trusted=trusted, artifact=artifact, path=target)
    verify_sha256(target, expected_sha256, artifact=artifact)
    import joblib

    return joblib.load(target)
