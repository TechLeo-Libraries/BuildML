"""Image decode + train-only channel normalize helpers for multimodal DL.

Supports:
- path cells (str / Path) via Pillow (included in ``buildml[torch]``)
- array cells (``numpy.ndarray`` / nested lists) without Pillow

Normalized tensors are CHW float32. Channel mean/std are fit on train only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from buildml.core.errors import MissingExtraError, ValidationError


def require_pillow(*, feature: str = "Image path loading") -> Any:
    """Import and return ``PIL.Image``, or raise :class:`MissingExtraError`."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise MissingExtraError("torch", feature) from exc
    return Image


def decode_image_cell(
    value: Any,
    *,
    size: tuple[int, int] = (32, 32),
    channels: int = 3,
) -> np.ndarray:
    """Decode one cell to a CHW float32 array in ``[0, 1]``.

    Accepts file paths, ``Path`` objects, ``numpy`` arrays, or nested lists.
    Arrays may be HWC, CHW, or HW (expanded to ``channels``).
    """
    if channels not in {1, 3}:
        raise ValidationError("image channels must be 1 or 3")
    h, w = int(size[0]), int(size[1])
    if h < 1 or w < 1:
        raise ValidationError("image size must be positive")

    if isinstance(value, (str, Path)):
        return _decode_path(Path(value), size=(h, w), channels=channels)
    if isinstance(value, np.ndarray):
        return _normalize_array(value, size=(h, w), channels=channels)
    if isinstance(value, (list, tuple)):
        return _normalize_array(np.asarray(value), size=(h, w), channels=channels)
    raise ValidationError(
        "Image cell must be a path string, Path, numpy array, or nested list; "
        f"got {type(value).__name__}"
    )


def _decode_path(path: Path, *, size: tuple[int, int], channels: int) -> np.ndarray:
    Image = require_pillow(feature="Image path multimodal loaders")
    if not path.exists():
        raise ValidationError(f"Image path does not exist: {path}")
    with Image.open(path) as img:
        if channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")
        img = img.resize((size[1], size[0]))  # PIL: (W, H)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return _to_chw(arr, channels=channels)


def _normalize_array(
    arr: np.ndarray, *, size: tuple[int, int], channels: int
) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[..., None] if channels == 1 else np.stack([arr] * channels, axis=-1)
    if arr.ndim != 3:
        raise ValidationError(
            f"Image array must be HW, HWC, or CHW; got shape {arr.shape}"
        )
    # Detect CHW vs HWC: if first dim looks like channels and last does not.
    if arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))  # CHW → HWC
    elif arr.shape[0] in {1, 3} and arr.shape[-1] in {1, 3} and arr.shape[0] < arr.shape[-1]:
        # Ambiguous small squares — prefer HWC when last dim is channel-like.
        pass
    if arr.shape[-1] == 1 and channels == 3:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 3 and channels == 1:
        arr = arr.mean(axis=-1, keepdims=True)
    elif arr.shape[-1] != channels:
        raise ValidationError(
            f"Image channels mismatch: array has {arr.shape[-1]} channels, "
            f"expected {channels}"
        )
    arr = arr.astype(np.float32, copy=False)
    if arr.max() > 1.5:  # likely 0–255
        arr = arr / 255.0
    # Resize with simple nearest-neighbor (no scipy/torchvision required).
    arr = _resize_hwc(arr, size=size)
    return _to_chw(arr, channels=channels)


def _resize_hwc(arr: np.ndarray, *, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    ys = (np.linspace(0, arr.shape[0] - 1, h)).astype(np.int64)
    xs = (np.linspace(0, arr.shape[1] - 1, w)).astype(np.int64)
    return arr[ys][:, xs]


def _to_chw(arr_hwc: np.ndarray, *, channels: int) -> np.ndarray:
    if arr_hwc.ndim != 3 or arr_hwc.shape[-1] != channels:
        raise ValidationError(
            f"Expected HWC with {channels} channels; got shape {arr_hwc.shape}"
        )
    return np.transpose(arr_hwc, (2, 0, 1)).astype(np.float32, copy=False)


def stack_image_column(
    values: Any,
    *,
    size: tuple[int, int] = (32, 32),
    channels: int = 3,
) -> np.ndarray:
    """Decode a Series/list of image cells → ``(N, C, H, W)`` float32."""
    decoded = [decode_image_cell(v, size=size, channels=channels) for v in values]
    if not decoded:
        return np.zeros((0, channels, size[0], size[1]), dtype=np.float32)
    return np.stack(decoded, axis=0)


def fit_image_channel_stats(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel mean/std on a train batch ``(N, C, H, W)``."""
    if images.ndim != 4:
        raise ValidationError(f"Expected NCHW images; got shape {images.shape}")
    if images.shape[0] < 1:
        raise ValidationError("Cannot fit image normalize stats on empty train partition")
    # Mean/std over N,H,W → shape (C,)
    mean = images.mean(axis=(0, 2, 3)).astype(np.float64)
    std = images.std(axis=(0, 2, 3)).astype(np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_image_channel_stats(
    images: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply frozen per-channel mean/std to ``(N, C, H, W)`` images."""
    mean_b = mean.reshape(1, -1, 1, 1).astype(np.float32)
    std_b = std.reshape(1, -1, 1, 1).astype(np.float32)
    return ((images.astype(np.float32) - mean_b) / std_b).astype(np.float32)


__all__ = [
    "apply_image_channel_stats",
    "decode_image_cell",
    "fit_image_channel_stats",
    "require_pillow",
    "stack_image_column",
]
