"""Tabular augmentations for contrastive SSL."""

from __future__ import annotations

from typing import Any

import numpy as np

from buildml.dl.extras import require_torch


def augment_tabular_pair(
    x: Any,
    *,
    noise_std: float = 0.1,
    feature_dropout: float = 0.1,
    scale_jitter: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[Any, Any]:
    """Return two augmented views of a tabular batch (numpy or torch tensor)."""
    require_torch(feature="SSL tabular augment")
    is_torch = hasattr(x, "device")
    if is_torch:
        arr = x
    else:
        torch = require_torch(feature="SSL tabular augment")
        arr = torch.as_tensor(np.asarray(x, dtype=np.float32))

    gen = rng if rng is not None else np.random.default_rng()
    v1 = _augment_view(arr, noise_std, feature_dropout, scale_jitter, gen)
    v2 = _augment_view(arr, noise_std, feature_dropout, scale_jitter, gen)
    if not is_torch:
        return v1.cpu().numpy(), v2.cpu().numpy()
    return v1, v2


def _augment_view(
    x: Any,
    noise_std: float,
    feature_dropout: float,
    scale_jitter: float,
    rng: np.random.Generator,
) -> Any:
    torch = require_torch(feature="SSL tabular augment")
    out = x.clone()
    if scale_jitter > 0:
        factors = 1.0 + torch.as_tensor(
            rng.uniform(-scale_jitter, scale_jitter, size=(1, out.shape[1])),
            dtype=out.dtype,
            device=out.device,
        )
        out = out * factors
    if noise_std > 0:
        out = out + torch.randn_like(out) * float(noise_std)
    if feature_dropout > 0:
        mask = torch.as_tensor(
            rng.random(out.shape) >= feature_dropout,
            dtype=torch.bool,
            device=out.device,
        )
        out = out * mask.float()
    return out


def random_feature_mask(
    x: Any,
    *,
    mask_ratio: float,
    rng: np.random.Generator,
) -> tuple[Any, Any]:
    """Return (masked_input, boolean_mask) for MAE-style training."""
    torch = require_torch(feature="SSL MAE mask")
    arr = x if hasattr(x, "device") else torch.as_tensor(np.asarray(x, dtype=np.float32))
    n, d = arr.shape
    mask = torch.as_tensor(rng.random((n, d)) < mask_ratio, dtype=torch.bool, device=arr.device)
    if d > 1:
        empty = ~mask.any(dim=1)
        if empty.any():
            cols = torch.as_tensor(
                rng.integers(0, d, size=int(empty.sum().item())),
                device=arr.device,
            )
            mask[empty.nonzero(as_tuple=True)[0], cols] = True
    fill = arr.mean(dim=0, keepdim=True)
    masked = arr.clone()
    masked[mask] = fill.expand_as(arr)[mask]
    return masked, mask
