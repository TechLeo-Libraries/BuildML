"""Self-supervised bundle persistence (v2 Torch + v1 legacy migration)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from buildml._version import __version__
from buildml.core.serialization import joblib_load_trusted, require_trusted_deserialize
from buildml.core.errors import ValidationError
from buildml.selfsupervised.results import (
    SSLHeadFitResult,
    SSLHeadPlan,
    SelfSupervisedEvalResult,
    SelfSupervisedFitResult,
    SelfSupervisedPlan,
)

BUNDLE_FORMAT_V1 = "buildml.selfsupervised_bundle.v1"
BUNDLE_FORMAT_V2 = "buildml.ssl_bundle.v2"
BUNDLE_FORMAT = BUNDLE_FORMAT_V2
CHECKPOINT_BOUNDARY = (
    "Self-supervised bundles, semi-supervised bundles, Torch trainer bundles, "
    "pretrained zoo backbones, classical pipeline bundles, RAG bundles, and "
    "Session checkpoints are complementary, not interchangeable. "
    f"A self-supervised bundle ({BUNDLE_FORMAT_V2}) stores a train-fitted "
    "SelfSupervisedPlan (Torch or legacy sklearn encoder + feature contract) "
    "and optionally an SSLHeadPlan. Legacy v1 bundles remain loadable. "
    "A Session checkpoint stores data, roles, splits, history, and optional "
    "classical preprocess plans; it does not embed the SSL encoder. "
    "Reload tabular workflow via checkpoint_load; reload SSL via "
    "load_ssl_bundle. Vision/audio/speech transfer also supports "
    "load_pretrained_backbone / attach_backbone_head."
)


def save_ssl_bundle(
    path: str | Path,
    plan: SelfSupervisedPlan,
    *,
    fit_result: SelfSupervisedFitResult | None = None,
    head_plan: SSLHeadPlan | None = None,
    head_fit_result: SSLHeadFitResult | None = None,
    eval_result: SelfSupervisedEvalResult | None = None,
) -> Path:
    """Write a self-supervised bundle directory (``buildml.ssl_bundle.v2``).

    Persists the fitted :class:`~buildml.selfsupervised.results.SelfSupervisedPlan`
    separately from Session checkpoints so tabular workflow and SSL state reload
    independently.

    Parameters
    ----------
    path:
        Destination directory for ``meta.json`` and ``ssl_plan.joblib``.
    plan:
        Train-fitted self-supervised pretext plan to persist.
    fit_result, head_plan, head_fit_result, eval_result:
        Optional last operation reports for bundle metadata.

    Returns
    -------
    pathlib.Path
        The bundle directory that was written.

    Raises
    ------
    ValidationError
        When ``plan`` is ``None``.
    """
    if plan is None:
        raise ValidationError("No SelfSupervisedPlan to save.")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    bundle_format = getattr(plan, "bundle_format", BUNDLE_FORMAT_V2)
    payload = {"plan": plan, "head_plan": head_plan}
    joblib.dump(payload, destination / "ssl_plan.joblib")
    _maybe_save_torch_state(destination, plan)
    meta: dict[str, Any] = {
        "format": bundle_format,
        "buildml_version": __version__,
        "compatibility": CHECKPOINT_BOUNDARY,
        "plan": plan.to_dict(),
        "fit": None if fit_result is None else fit_result.to_dict(),
        "head": None if head_plan is None else head_plan.to_dict(),
        "head_fit": None if head_fit_result is None else head_fit_result.to_dict(),
        "eval": None if eval_result is None else eval_result.to_dict(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return destination


def load_ssl_bundle(path: str | Path, *, trusted: bool = False) -> tuple[SelfSupervisedPlan, SSLHeadPlan | None]:
    """Load a self-supervised bundle into plan (+ optional head).

    Restores v1 and v2 bundle formats, optionally rehydrating Torch encoder
    weights from ``encoder_torch.pt`` when present.

    Parameters
    ----------
    path:
        Bundle directory containing ``meta.json`` and ``ssl_plan.joblib``.
    trusted:
        Must be ``True`` to deserialize pickle/joblib/torch payloads. Pass
        only for artifacts you created or fully trust. Defaults to ``False``.

    Returns
    -------
    tuple[SelfSupervisedPlan, SSLHeadPlan | None]
        Restored pretext plan and optional supervised head plan.

    Raises
    ------
    ValidationError
        When the bundle is incomplete, uses an unsupported format, or payload
        types do not match expected plan objects.
    """
    root = Path(path)
    meta_path = root / "meta.json"
    plan_path = root / "ssl_plan.joblib"
    if not meta_path.is_file() or not plan_path.is_file():
        raise ValidationError(
            f"Incomplete self-supervised bundle at {root}. "
            f"Expected meta.json and ssl_plan.joblib."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = meta.get("format")
    if fmt not in {BUNDLE_FORMAT_V1, BUNDLE_FORMAT_V2}:
        raise ValidationError(
            f"Unsupported self-supervised bundle format {fmt!r}; "
            f"expected {BUNDLE_FORMAT_V1} or {BUNDLE_FORMAT_V2}."
        )
    loaded = joblib_load_trusted(plan_path, trusted=trusted, artifact="joblib plan")
    if isinstance(loaded, SelfSupervisedPlan):
        plan = loaded
        head = None
    elif isinstance(loaded, dict) and "plan" in loaded:
        plan = loaded["plan"]
        head = loaded.get("head_plan")
    else:
        raise ValidationError(
            "ssl_plan.joblib must contain a SelfSupervisedPlan or a payload with key 'plan'."
        )
    if not isinstance(plan, SelfSupervisedPlan):
        raise ValidationError("Loaded plan object is not a SelfSupervisedPlan")
    if head is not None and not isinstance(head, SSLHeadPlan):
        raise ValidationError("Loaded head_plan object is not an SSLHeadPlan")
    _maybe_restore_torch_state(root, plan, trusted=trusted)
    if fmt == BUNDLE_FORMAT_V1:
        plan.bundle_format = BUNDLE_FORMAT_V1  # type: ignore[attr-defined]
    return plan, head


def _maybe_save_torch_state(destination: Path, plan: SelfSupervisedPlan) -> None:
    encoder = plan.encoder_
    if hasattr(encoder, "state_dict") and callable(encoder.state_dict):
        try:
            state = encoder.state_dict()
        except ValidationError:
            return
        torch_path = destination / "encoder_torch.json"
        # Torch tensors saved separately
        from buildml.dl.extras import torch_available

        if not torch_available():
            return
        import torch

        payload_path = destination / "encoder_torch.pt"
        torch.save(state, payload_path)
        torch_path.write_text(
            json.dumps({"path": "encoder_torch.pt", "method": plan.method}, indent=2),
            encoding="utf-8",
        )


def _maybe_restore_torch_state(
    root: Path, plan: SelfSupervisedPlan, *, trusted: bool
) -> None:
    meta_path = root / "encoder_torch.json"
    pt_path = root / "encoder_torch.pt"
    if not meta_path.is_file() or not pt_path.is_file():
        return
    from buildml.dl.extras import require_torch

    torch = require_torch(feature="SSL bundle restore")
    require_trusted_deserialize(
        trusted=trusted, artifact="torch SSL encoder payload", path=pt_path
    )

    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    method = str(state.get("method", plan.method))
    if method in {
        "simclr_tabular",
        "byol_tabular",
        "vicreg_tabular",
        "mae_tabular",
        "vae_tabular",
    }:
        from buildml.selfsupervised.torch.encoder import TorchTabularSSLEncoder

        plan.encoder_ = TorchTabularSSLEncoder.from_state_dict(state)
    elif method == "vision_ssl":
        # Vision restore kept minimal — joblib encoder retained when pt missing
        pass
