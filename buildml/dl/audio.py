"""Audio decode + train-only waveform normalize helpers for multimodal DL.

Supports:
- path cells (str / Path) via ``soundfile`` (included in ``buildml[torch]``)
- waveform array cells (``numpy.ndarray`` / nested lists) without soundfile

Tensors are mono ``(1, T)`` float32. Amplitude mean/std are fit on train only.

This is an honest alpha fusion branch — not a speech foundation-model stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from buildml.core.errors import MissingExtraError, ValidationError


def require_soundfile(*, feature: str = "Audio path loading") -> Any:
    """Import and return ``soundfile``, or raise :class:`MissingExtraError`."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise MissingExtraError("torch", feature) from exc
    return sf


def decode_audio_cell(
    value: Any,
    *,
    sample_rate: int = 16_000,
    max_samples: int = 16_000,
    source_sample_rate: int | None = None,
) -> np.ndarray:
    """Decode one cell to a mono ``(1, T)`` float32 waveform.

    Accepts file paths, ``Path`` objects, ``numpy`` arrays, or nested lists.
    Arrays may be ``(T,)``, ``(1, T)``, ``(C, T)``, or ``(T, C)`` with small C.
    """
    sr = int(sample_rate)
    t_max = int(max_samples)
    if sr < 1:
        raise ValidationError("sample_rate must be positive")
    if t_max < 1:
        raise ValidationError("max_samples must be positive")

    if isinstance(value, (str, Path)):
        return _decode_path(Path(value), sample_rate=sr, max_samples=t_max)
    if isinstance(value, np.ndarray):
        return _normalize_waveform(
            value,
            sample_rate=sr,
            max_samples=t_max,
            source_sample_rate=source_sample_rate,
        )
    if isinstance(value, (list, tuple)):
        return _normalize_waveform(
            np.asarray(value),
            sample_rate=sr,
            max_samples=t_max,
            source_sample_rate=source_sample_rate,
        )
    raise ValidationError(
        "Audio cell must be a path string, Path, numpy array, or nested list; "
        f"got {type(value).__name__}"
    )


def _decode_path(path: Path, *, sample_rate: int, max_samples: int) -> np.ndarray:
    sf = require_soundfile(feature="Audio path multimodal loaders")
    if not path.exists():
        raise ValidationError(f"Audio path does not exist: {path}")
    try:
        data, file_sr = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"Failed to read audio file {path}: {exc}") from exc
    # soundfile returns (T, C)
    wave = data.mean(axis=1).astype(np.float32, copy=False)
    wave = _resample_mono(wave, src_sr=int(file_sr), dst_sr=sample_rate)
    return _pad_or_truncate(wave, max_samples=max_samples)


def _normalize_waveform(
    arr: np.ndarray,
    *,
    sample_rate: int,
    max_samples: int,
    source_sample_rate: int | None,
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        wave = arr.astype(np.float32, copy=False)
    elif arr.ndim == 2:
        # Prefer channel-first when first dim is small channel count.
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            wave = arr.mean(axis=0).astype(np.float32, copy=False)
        elif arr.shape[1] <= 8 and arr.shape[1] < arr.shape[0]:
            wave = arr.mean(axis=1).astype(np.float32, copy=False)
        else:
            # Ambiguous: treat as (T, C) with C inferred as last dim if small.
            if arr.shape[-1] <= 8:
                wave = arr.mean(axis=-1).astype(np.float32, copy=False)
            else:
                wave = arr.reshape(-1).astype(np.float32, copy=False)
    else:
        raise ValidationError(
            f"Audio array must be 1D waveform or 2D (C,T)/(T,C); got shape {arr.shape}"
        )
    if source_sample_rate is not None and int(source_sample_rate) != int(sample_rate):
        wave = _resample_mono(
            wave, src_sr=int(source_sample_rate), dst_sr=int(sample_rate)
        )
    return _pad_or_truncate(wave, max_samples=max_samples)


def _resample_mono(wave: np.ndarray, *, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or wave.size == 0:
        return wave.astype(np.float32, copy=False)
    if src_sr < 1 or dst_sr < 1:
        raise ValidationError("sample rates must be positive")
    duration = wave.shape[0] / float(src_sr)
    n_out = max(1, int(round(duration * dst_sr)))
    x_old = np.linspace(0.0, 1.0, num=wave.shape[0], endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=True)
    return np.interp(x_new, x_old, wave.astype(np.float64)).astype(np.float32)


def _pad_or_truncate(wave: np.ndarray, *, max_samples: int) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if wave.shape[0] >= max_samples:
        clipped = wave[:max_samples]
    else:
        clipped = np.zeros(max_samples, dtype=np.float32)
        clipped[: wave.shape[0]] = wave
    return clipped.reshape(1, max_samples)


def stack_audio_column(
    values: Any,
    *,
    sample_rate: int = 16_000,
    max_samples: int = 16_000,
    source_sample_rate: int | None = None,
) -> np.ndarray:
    """Decode a Series/list of audio cells → ``(N, 1, T)`` float32."""
    decoded = [
        decode_audio_cell(
            v,
            sample_rate=sample_rate,
            max_samples=max_samples,
            source_sample_rate=source_sample_rate,
        )
        for v in values
    ]
    if not decoded:
        return np.zeros((0, 1, max_samples), dtype=np.float32)
    return np.stack(decoded, axis=0)


def fit_audio_waveform_stats(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit amplitude mean/std on a train batch ``(N, 1, T)``."""
    if audio.ndim != 3 or audio.shape[1] != 1:
        raise ValidationError(f"Expected N1T audio; got shape {audio.shape}")
    if audio.shape[0] < 1:
        raise ValidationError("Cannot fit audio normalize stats on empty train partition")
    mean = np.array([float(audio.mean())], dtype=np.float64)
    std = np.array([float(audio.std())], dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_audio_waveform_stats(
    audio: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply frozen amplitude mean/std to ``(N, 1, T)`` audio."""
    mean_b = float(np.asarray(mean).reshape(-1)[0])
    std_b = float(np.asarray(std).reshape(-1)[0])
    if abs(std_b) < 1e-6:
        std_b = 1.0
    return ((audio.astype(np.float32) - mean_b) / std_b).astype(np.float32)


__all__ = [
    "apply_audio_waveform_stats",
    "decode_audio_cell",
    "fit_audio_waveform_stats",
    "require_soundfile",
    "stack_audio_column",
]
