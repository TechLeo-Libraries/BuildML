"""Turn image cells into uniform tensors a network can batch.

Images arrive as file paths or as arrays already in memory, at whatever size and
orientation they were stored in. Batching needs all of them in one shape, so
this module decodes, resizes, converts to the requested channel count, scales
into ``[0, 1]``, and arranges as channels-first — the layout Torch convolutions
expect.

Channel statistics are fitted on the training partition only and applied
everywhere else. Per-channel rather than global, because photographic corpora
routinely have systematically different distributions in red, green, and blue,
and a single number would leave that structure for the first convolution to
undo.

Pillow is needed only for file paths. Arrays go through without it, which keeps
the dependency lazy.

See Also
--------
buildml.dl.multimodal : Where these tensors are consumed.
buildml.dl.zoo : Pretrained vision architectures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from buildml.core.errors import MissingExtraError, ValidationError


def require_pillow(*, feature: str = "Image path loading") -> Any:
    """Import Pillow, or explain how to install it.

    Only needed for decoding image files. Arrays already in memory go through
    without it, which is why the import is lazy.

    Parameters
    ----------
    feature:
        What the caller was doing. Appears in the error message.

    Returns
    -------
    module
        ``PIL.Image``.

    Raises
    ------
    MissingExtraError
        If Pillow is absent. It ships with ``pip install buildml[dl]``.
    """
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
    """Turn one image cell into a fixed-size channels-first array.

    Handles the whole conversion: read the file or accept the array, convert to
    the requested channel count, resize, scale into ``[0, 1]``, and transpose to
    ``(C, H, W)``.

    Parameters
    ----------
    value:
        A path string, ``Path``, NumPy array, or nested list. Arrays may be
        ``(H, W)``, ``(H, W, C)``, or ``(C, H, W)``.
    size:
        Target height and width.
    channels:
        1 for greyscale, 3 for colour. Colour input to a greyscale request is
        averaged; greyscale input to a colour request is repeated.

    Returns
    -------
    numpy.ndarray
        A ``(channels, height, width)`` float32 array in ``[0, 1]``.

    Raises
    ------
    MissingExtraError
        If a path is given and Pillow is not installed.
    ValidationError
        If ``channels`` is not 1 or 3, if the size is not positive, if the file
        does not exist, if the array shape is not interpretable, or if its
        channel count cannot be reconciled with the request.

    Notes
    -----
    **Values above 1.5 are assumed to be on a 0-255 scale and divided by 255.**
    This is a heuristic, and it is the right one nearly always — a genuine
    ``[0, 1]`` image containing a value above 1.5 is not an image. Pillow output
    is scaled unconditionally, since its range is known.

    **Array resizing is nearest-neighbour**, chosen to avoid a SciPy or
    torchvision dependency. It is blockier than bilinear on large downscales.
    Paths go through Pillow, which interpolates properly.

    See Also
    --------
    stack_image_column : The batched version.
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
    """Decode a whole column of images into one batched array.

    Applies :func:`decode_image_cell` to every cell and stacks the results.

    Parameters
    ----------
    values:
        An iterable of image cells — a pandas Series or a list.
    size:
        Target height and width.
    channels:
        1 or 3.

    Returns
    -------
    numpy.ndarray
        An ``(N, channels, height, width)`` float32 array. An empty input gives
        an array with zero rows and the right trailing shape, so downstream code
        that inspects dimensions still works.

    Raises
    ------
    MissingExtraError
        If any cell is a path and Pillow is not installed.
    ValidationError
        Propagated from any cell that cannot be decoded.

    Notes
    -----
    **Everything is decoded eagerly into memory.** A thousand 224x224 colour
    images is around 600 MB as float32. For corpora large enough to matter, a
    custom ``Dataset`` that decodes per batch is the right shape.

    See Also
    --------
    decode_image_cell : The single-cell version.
    fit_image_channel_stats : The usual next step.
    """
    decoded = [decode_image_cell(v, size=size, channels=channels) for v in values]
    if not decoded:
        return np.zeros((0, channels, size[0], size[1]), dtype=np.float32)
    return np.stack(decoded, axis=0)


def fit_image_channel_stats(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Learn each colour channel's brightness and contrast from training images.

    Computes one mean and one deviation per channel, across every training image
    and every pixel position. Standardising with these is standard practice for
    convolutional networks and helps them converge.

    Parameters
    ----------
    images:
        Training images shaped ``(N, C, H, W)``.

    Returns
    -------
    numpy.ndarray
        Per-channel means, shape ``(C,)``.
    numpy.ndarray
        Per-channel deviations, shape ``(C,)``, floored at 1.0 when near zero.

    Raises
    ------
    ValidationError
        If the input is not 4-D, or if the batch is empty.

    Notes
    -----
    **Per channel, not per pixel.** A per-pixel mean would encode where in the
    frame things tend to be bright — real structure in a corpus of centred
    product photos, and exactly the kind of structure the model should be
    learning rather than having subtracted away.

    **A constant channel would divide by zero, so its deviation becomes 1.0.**
    Real images do not produce this; synthetic or single-colour test data can.

    See Also
    --------
    apply_image_channel_stats : Applying what this learned.
    """
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
    """Rescale images using statistics learned from training images.

    Subtracts the per-channel mean and divides by the per-channel deviation,
    broadcasting across every pixel. The same constants are used for every
    partition and at inference.

    Parameters
    ----------
    images:
        Images shaped ``(N, C, H, W)``.
    mean:
        Per-channel means from :func:`fit_image_channel_stats`.
    std:
        Per-channel deviations.

    Returns
    -------
    numpy.ndarray
        Rescaled float32 images, same shape as the input.

    Notes
    -----
    Output is no longer in ``[0, 1]`` and is not meant to be — values centre
    near zero and extend either way. Reverse the transformation before trying to
    display a standardised image.
    """
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
