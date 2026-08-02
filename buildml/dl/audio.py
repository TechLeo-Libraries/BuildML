"""Audio decode + train-only waveform normalize helpers for multimodal DL.

Supports:
- path cells (str / Path) via ``soundfile`` (included in ``buildml[torch]``)
- waveform array cells (``numpy.ndarray`` / nested lists) without soundfile

Tensors are mono ``(1, T)`` float32. Short clips are **repeat-padded** to
``max_samples`` (not zero-filled) so global pooling remains informative.
Amplitude mean/std are fit on train only (optionally length-aware).

Repeat-pad (not length-masked pooling) is the alpha choice: the fusion
``AdaptiveAvgPool1d`` keeps a single fixed-length audio tensor for
train/export/ONNX, and tiling preserves amplitude so short clips are not
wiped by a large default window. Length-masked pooling would need lengths in
``forward`` and a wider batch/export contract.

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
    Short clips are repeat-padded to ``max_samples`` (not zero-filled).
    """
    cell, _length = _decode_audio_cell_with_length(
        value,
        sample_rate=int(sample_rate),
        max_samples=int(max_samples),
        source_sample_rate=source_sample_rate,
    )
    return cell


def _mono_wave_from_array(
    arr: np.ndarray,
    *,
    sample_rate: int,
    source_sample_rate: int | None,
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        wave = arr.astype(np.float32, copy=False)
    elif arr.ndim == 2:
        # Only accept unambiguous channel layouts: one dim is C<=8 and strictly
        # smaller than the time dim. Equal/square/both-large shapes are refused
        # rather than silently flattened into a wrong waveform.
        n0, n1 = int(arr.shape[0]), int(arr.shape[1])
        if n0 <= 8 and n0 < n1:
            wave = arr.mean(axis=0).astype(np.float32, copy=False)
        elif n1 <= 8 and n1 < n0:
            wave = arr.mean(axis=1).astype(np.float32, copy=False)
        else:
            raise ValidationError(
                f"Ambiguous 2D audio array shape {arr.shape}: expected 1D (T,), "
                "channel-first (C,T), or channel-last (T,C) with C<=8 and C<T. "
                "Pass a mono waveform or reshape explicitly."
            )
    else:
        raise ValidationError(
            f"Audio array must be 1D waveform or 2D (C,T)/(T,C); got shape {arr.shape}"
        )
    if source_sample_rate is not None and int(source_sample_rate) != int(sample_rate):
        wave = _resample_mono(
            wave, src_sr=int(source_sample_rate), dst_sr=int(sample_rate)
        )
    return wave


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
    """Pad/truncate to ``max_samples``.

    Short clips are **repeat-padded** (tiled) rather than zero-padded so a
    global ``AdaptiveAvgPool1d`` over the fixed window does not wash out the
    signal when ``max_samples`` is much larger than the source clip.
    Empty waveforms remain zeros.
    """
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if wave.shape[0] >= max_samples:
        clipped = wave[:max_samples]
    elif wave.shape[0] == 0:
        clipped = np.zeros(max_samples, dtype=np.float32)
    else:
        reps = int(np.ceil(max_samples / float(wave.shape[0])))
        clipped = np.tile(wave, reps)[:max_samples].astype(np.float32, copy=False)
    return clipped.reshape(1, max_samples)


def stack_audio_column(
    values: Any,
    *,
    sample_rate: int = 16_000,
    max_samples: int = 16_000,
    source_sample_rate: int | None = None,
    return_lengths: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Decode a Series/list of audio cells → ``(N, 1, T)`` float32.

    When ``return_lengths=True``, also returns pre-pad/truncation lengths
    ``(N,)`` (clamped to ``max_samples``) for length-aware normalize stats.
    """
    decoded: list[np.ndarray] = []
    lengths: list[int] = []
    for v in values:
        cell, length = _decode_audio_cell_with_length(
            v,
            sample_rate=sample_rate,
            max_samples=max_samples,
            source_sample_rate=source_sample_rate,
        )
        decoded.append(cell)
        lengths.append(length)
    if not decoded:
        empty = np.zeros((0, 1, max_samples), dtype=np.float32)
        if return_lengths:
            return empty, np.zeros((0,), dtype=np.int64)
        return empty
    stacked = np.stack(decoded, axis=0)
    if return_lengths:
        return stacked, np.asarray(lengths, dtype=np.int64)
    return stacked


def _decode_audio_cell_with_length(
    value: Any,
    *,
    sample_rate: int,
    max_samples: int,
    source_sample_rate: int | None,
) -> tuple[np.ndarray, int]:
    """Decode one cell and report the unpadded (pre-tile) length in samples."""
    sr = int(sample_rate)
    t_max = int(max_samples)
    if sr < 1:
        raise ValidationError("sample_rate must be positive")
    if t_max < 1:
        raise ValidationError("max_samples must be positive")

    if isinstance(value, (str, Path)):
        sf = require_soundfile(feature="Audio path multimodal loaders")
        path = Path(value)
        if not path.exists():
            raise ValidationError(f"Audio path does not exist: {path}")
        try:
            data, file_sr = sf.read(str(path), always_2d=True, dtype="float32")
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(f"Failed to read audio file {path}: {exc}") from exc
        # soundfile returns (T, C)
        wave = data.mean(axis=1).astype(np.float32, copy=False)
        wave = _resample_mono(wave, src_sr=int(file_sr), dst_sr=sr)
    elif isinstance(value, np.ndarray):
        wave = _mono_wave_from_array(
            value, sample_rate=sr, source_sample_rate=source_sample_rate
        )
    elif isinstance(value, (list, tuple)):
        wave = _mono_wave_from_array(
            np.asarray(value), sample_rate=sr, source_sample_rate=source_sample_rate
        )
    else:
        raise ValidationError(
            "Audio cell must be a path string, Path, numpy array, or nested list; "
            f"got {type(value).__name__}"
        )
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    raw_len = int(min(wave.shape[0], t_max))
    return _pad_or_truncate(wave, max_samples=t_max), raw_len


def fit_audio_waveform_stats(
    audio: np.ndarray,
    lengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit amplitude mean/std on a train batch ``(N, 1, T)``.

    When ``lengths`` is provided, stats use only the pre-pad/truncation region
    of each clip (avoids zero-pad domination if a caller still zero-fills).
    """
    if audio.ndim != 3 or audio.shape[1] != 1:
        raise ValidationError(f"Expected N1T audio; got shape {audio.shape}")
    if audio.shape[0] < 1:
        raise ValidationError("Cannot fit audio normalize stats on empty train partition")
    if lengths is None:
        mean = np.array([float(audio.mean())], dtype=np.float64)
        std = np.array([float(audio.std())], dtype=np.float64)
    else:
        lengths_a = np.asarray(lengths, dtype=np.int64).reshape(-1)
        if lengths_a.shape[0] != audio.shape[0]:
            raise ValidationError("audio lengths must align with batch dimension")
        pieces: list[np.ndarray] = []
        for i, length in enumerate(lengths_a.tolist()):
            n = int(max(0, min(int(length), audio.shape[-1])))
            if n > 0:
                pieces.append(audio[i, 0, :n].astype(np.float64, copy=False).ravel())
        if not pieces:
            raise ValidationError("Cannot fit audio normalize stats on empty waveforms")
        cat = np.concatenate(pieces)
        mean = np.array([float(cat.mean())], dtype=np.float64)
        std = np.array([float(cat.std())], dtype=np.float64)
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
