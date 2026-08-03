"""Turn audio cells into uniform waveform tensors a network can batch.

Audio arrives in inconsistent shapes: files on disk at whatever sample rate they
were recorded at, arrays already in memory, stereo or mono, of varying lengths.
Batching requires all of it to become one fixed shape, and this module does that
conversion: decode, mix down to mono, resample to a common rate, and pad or
truncate to a fixed length. The output is always ``(1, T)`` float32.

Two decisions here are worth understanding.

**Short clips are repeat-padded, not zero-padded.** The fusion audio branch ends
in global average pooling, so a half-second clip zero-padded into a five-second
window would have four and a half seconds of silence averaged into its
representation: the model would mostly learn how long each clip was. Tiling the
content keeps the pooled statistics about the audio. The alternative, masked
pooling, would require passing lengths through ``forward``, which complicates
both the batch contract and ONNX export.

**Amplitude statistics are fitted on training data only**, and optionally
length-aware so padding does not skew them.

This is an honest fusion branch, not a speech stack. For transcription or
pretrained speech representations, see :mod:`buildml.dl.speech`.

See Also
--------
buildml.dl.multimodal : Where these tensors are consumed.
buildml.dl.speech : Transcription and pretrained speech models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from buildml.core.errors import MissingExtraError, ValidationError


def require_soundfile(*, feature: str = "Audio path loading") -> Any:
    """Import soundfile, or explain how to install it.

    Only needed for decoding audio files. Waveforms already in memory as arrays
    or lists go through without it, which is why the import is lazy rather than
    at module level.

    Parameters
    ----------
    feature:
        What the caller was doing. Appears in the error message.

    Returns
    -------
    module
        The ``soundfile`` module.

    Raises
    ------
    MissingExtraError
        If soundfile is absent. It ships with ``pip install buildml[dl]``.
    """
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
    """Turn one audio cell into a fixed-length mono waveform.

    Handles the whole conversion: read the file or accept the array, mix
    channels down to mono, resample to the target rate, and pad or truncate to
    the required length.

    Parameters
    ----------
    value:
        A path string, ``Path``, NumPy array, or nested list. Arrays may be
        ``(T,)``, ``(C, T)``, or ``(T, C)``.
    sample_rate:
        Target rate in Hz. Files are resampled from whatever rate they carry.
    max_samples:
        Output length. At 16 kHz, 16000 is one second.
    source_sample_rate:
        The rate of an incoming array, which cannot be inferred the way a
        file's can. Ignored for paths.

    Returns
    -------
    numpy.ndarray
        A ``(1, max_samples)`` float32 waveform.

    Raises
    ------
    MissingExtraError
        If a path is given and soundfile is not installed.
    ValidationError
        If the file does not exist or cannot be read, if the array shape is
        ambiguous, if the cell type is unsupported, or if ``sample_rate`` or
        ``max_samples`` is not positive.

    Notes
    -----
    **Ambiguous 2-D shapes are refused rather than guessed.** Channel-first and
    channel-last are distinguished by assuming channels are few and time is
    many. A ``(2, 1000)`` array is clearly stereo; a ``(1000, 1000)`` array
    could be either, and picking wrong would silently produce a waveform that is
    not the recording. Reshape explicitly in that case.

    **Resampling is linear interpolation.** Adequate for feature extraction, and
    not the band-limited resampling a high-fidelity pipeline would use. Supply
    audio already at your target rate when quality matters.

    See Also
    --------
    stack_audio_column : The batched version.
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
    """Decode a whole column of audio into one batched array.

    Applies :func:`decode_audio_cell` to every cell and stacks the results.

    Parameters
    ----------
    values:
        An iterable of audio cells: a pandas Series or a list.
    sample_rate:
        Target rate in Hz.
    max_samples:
        Output length per clip.
    source_sample_rate:
        The rate of incoming arrays.
    return_lengths:
        Also return each clip's real length before padding.

    Returns
    -------
    numpy.ndarray
        An ``(N, 1, max_samples)`` float32 array.
    numpy.ndarray
        Only when ``return_lengths=True``: an ``(N,)`` int64 array of real
        lengths, clamped to ``max_samples``.

    Raises
    ------
    MissingExtraError
        If any cell is a path and soundfile is not installed.
    ValidationError
        Propagated from any cell that cannot be decoded.

    Notes
    -----
    **The lengths are what make normalisation statistics honest.** Without
    them, a corpus of short clips tiled into a long window would have its
    statistics computed over repeated content, weighting the shortest clips
    most heavily. :func:`fit_audio_waveform_stats` uses the lengths to measure
    only real audio.

    See Also
    --------
    decode_audio_cell : The single-cell version.
    fit_audio_waveform_stats : The consumer of the lengths.
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
    """Learn the amplitude scale of the training audio.

    Computes a single mean and standard deviation across all training samples.
    Audio amplitude varies enormously with recording conditions: the same words
    at different distances from a microphone differ by orders of magnitude :
    and standardising removes that so the network learns from structure rather
    than volume.

    Parameters
    ----------
    audio:
        Training waveforms, shaped ``(N, 1, T)``.
    lengths:
        Real length of each clip before padding. When supplied, only that
        region contributes.

    Returns
    -------
    numpy.ndarray
        A one-element mean.
    numpy.ndarray
        A one-element standard deviation, floored at 1.0 when near zero.

    Raises
    ------
    ValidationError
        If the shape is not ``(N, 1, T)``, if the batch is empty, if the
        lengths do not align with the batch, or if every clip is empty.

    Notes
    -----
    **One statistic for all samples, not one per time step.** A per-position
    mean would encode where in the clip loud moments tend to fall, which is an
    artefact of how the corpus was recorded rather than anything about the
    audio.

    **Passing ``lengths`` matters when clip durations vary widely.** Without
    them the statistics are computed over padded content, which over-weights
    whatever was tiled to fill the window.

    **Silent audio would divide by zero, so a near-zero deviation becomes 1.0.**

    See Also
    --------
    apply_audio_waveform_stats : Applying what this learned.
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
    """Rescale waveforms using statistics learned from training audio.

    Subtracts the training mean and divides by the training deviation. The same
    constants are used for every partition and at inference, which is what keeps
    a deployed model hearing what it was trained on.

    Parameters
    ----------
    audio:
        Waveforms shaped ``(N, 1, T)``.
    mean:
        The training mean from :func:`fit_audio_waveform_stats`.
    std:
        The training deviation.

    Returns
    -------
    numpy.ndarray
        Rescaled float32 waveforms, same shape as the input.

    Notes
    -----
    A near-zero deviation is replaced by 1.0 here as well as at fit time, so a
    hand-constructed statistic cannot produce infinities.
    """
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
