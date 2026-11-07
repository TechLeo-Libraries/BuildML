"""Sample times, real clip lengths and native audio consumers share one contract."""

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.dl.audio import (
    apply_audio_waveform_stats,
    decode_audio_cell,
    fit_audio_waveform_stats,
    stack_audio_column,
)


def _at_sample_times(wave, source_rate, target_rate):
    """Scalar piecewise-linear signal evaluated at j / target_rate seconds."""
    if not len(wave):
        return np.array([], dtype=np.float32)
    count = max(1, round(len(wave) * target_rate / source_rate))
    result = []
    for j in range(count):
        position = min(j * source_rate / target_rate, len(wave) - 1)
        left = math.floor(position)
        right = min(left + 1, len(wave) - 1)
        weight = position - left
        result.append((1 - weight) * float(wave[left]) + weight * float(wave[right]))
    return np.asarray(result, dtype=np.float32)


def _window(wave, size):
    return np.asarray(
        [wave[j % len(wave)] if len(wave) else 0 for j in range(size)], dtype=np.float32
    )


def _fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "kind", ["array", "list", "tuple", "channels-first", "channels-last", "float-wav", "pcm-wav"]
)
@pytest.mark.parametrize("source_rate,target_rate", [(8, 16), (16, 8), (7, 11), (11, 7), (8, 8)])
@pytest.mark.parametrize("size", [5, 32])
def test_decode_uses_recording_sample_times(kind, source_rate, target_rate, size, tmp_path):
    wave = np.array([0, 2, -1, 3, 1, -2, 0, 1, 4, -3, 2, 0, -1], dtype=np.float32) / 8
    stereo = np.column_stack([wave + 0.125, wave - 0.125])
    value = {
        "array": wave,
        "list": wave.tolist(),
        "tuple": tuple(wave),
        "channels-first": stereo.T,
        "channels-last": stereo,
    }.get(kind)
    if kind.endswith("wav"):
        sf = pytest.importorskip("soundfile")
        value = tmp_path / "recording.wav"
        sf.write(value, stereo, source_rate, subtype="FLOAT" if kind == "float-wav" else "PCM_16")
        before = _fingerprint(value)
    else:
        before = copy.deepcopy(value)
    expected_raw = _at_sample_times(wave, source_rate, target_rate)
    expected = _window(expected_raw, size)
    # A file's own metadata takes precedence over the array-only source rate.
    declared_rate = 999 if kind.endswith("wav") else source_rate
    decoded = decode_audio_cell(
        value, sample_rate=target_rate, max_samples=size, source_sample_rate=declared_rate
    )
    batch, lengths = stack_audio_column(
        [value],
        sample_rate=target_rate,
        max_samples=size,
        source_sample_rate=declared_rate,
        return_lengths=True,
    )
    assert decoded.dtype == batch.dtype == np.float32
    assert decoded.shape == (1, size) and batch.shape == (1, 1, size)
    np.testing.assert_allclose(decoded[0], expected, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(batch[0], decoded)
    assert lengths.tolist() == [min(len(expected_raw), size)]
    if kind.endswith("wav"):
        assert _fingerprint(value) == before
    else:
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("wave", [[], [0.375], [0.0, 0.0]])
@pytest.mark.parametrize("rates", [(8, 8), (8, 16), (16, 8)])
def test_empty_single_and_silent_recordings_have_defined_windows(wave, rates):
    source, target = rates
    result, lengths = stack_audio_column(
        [wave], sample_rate=target, source_sample_rate=source, max_samples=9, return_lengths=True
    )
    raw = _at_sample_times(wave, source, target)
    np.testing.assert_array_equal(result[0, 0], _window(raw, 9))
    assert lengths.tolist() == [min(len(raw), 9)]


@pytest.mark.parametrize("return_lengths", [False, True])
def test_empty_column_has_no_synthetic_observations(return_lengths):
    result = stack_audio_column([], max_samples=9, return_lengths=return_lengths)
    if return_lengths:
        result, lengths = result
        assert lengths.shape == (0,) and lengths.dtype == np.int64
    assert result.shape == (0, 1, 9) and result.dtype == np.float32


@pytest.mark.parametrize("integer", [np.int32, np.int64])
def test_numpy_integer_rates_and_counts_are_lossless(integer):
    actual = decode_audio_cell(
        [0.0, 0.5, -0.5],
        sample_rate=integer(8),
        source_sample_rate=integer(8),
        max_samples=integer(5),
    )
    np.testing.assert_array_equal(actual, [[0, 0.5, -0.5, 0, 0.5]])


@pytest.mark.parametrize("kind", ["channels-first", "channels-last", "float-wav"])
def test_finite_stereo_mix_does_not_overflow_float32_accumulation(kind, tmp_path):
    value = np.float32(np.finfo(np.float32).max * 0.75)
    stereo = np.full((12, 2), value, dtype=np.float32)
    if kind == "float-wav":
        sf = pytest.importorskip("soundfile")
        cell = tmp_path / "finite-stereo.wav"
        sf.write(cell, stereo, 8000, subtype="FLOAT")
    else:
        cell = stereo.T if kind == "channels-first" else stereo
    actual = decode_audio_cell(cell, sample_rate=8000, max_samples=12)
    np.testing.assert_array_equal(actual, np.full((1, 12), value, dtype=np.float32))


def _session(frame, backing, directory):
    engine = backing.split("-")[0]
    if engine != "pandas":
        pytest.importorskip(engine)
    if backing == "pandas":
        session = Session.ingest(frame)
    else:
        path = directory / "audio.parquet"
        frame.to_parquet(path, index=False)
        session = Session.ingest(
            path, engine=engine, mode="memory" if engine == "pandas" else "lazy"
        )
    return session.set_roles({"x": "feature", "audio": "feature", "y": "target"}).inject_split(
        train_indices=list(range(8)), validation_indices=[8, 9], test_indices=[10, 11]
    )


@pytest.mark.parametrize("backing", ["pandas", "pandas-parquet", "polars-lazy", "duckdb-lazy"])
@pytest.mark.parametrize("rates", [(8000, 16000), (16000, 8000)])
@pytest.mark.parametrize("consumer", ["normalized-fusion", "raw-fusion", "speech"])
def test_native_loaders_training_and_saved_audio_replay(backing, rates, consumer, tmp_path):
    torch = pytest.importorskip("torch")
    sf = pytest.importorskip("soundfile")
    source, target = rates
    size = 48
    paths, real, expected = [], [], []
    for i in range(12):
        # Held-out clips deliberately have a different amplitude distribution.
        wave = (np.arange(9 + i % 4, dtype=np.float32) % 5 - 2) / 16 + (0.5 if i >= 8 else i / 128)
        stereo = np.column_stack([wave + 1 / 32, wave - 1 / 32])
        path = tmp_path / f"clip-{i}.wav"
        sf.write(path, stereo, source, subtype="FLOAT")
        paths.append(str(path))
        raw = _at_sample_times(wave, source, target)
        if i < 8:
            real.extend(raw[:size].astype(float))
        expected.append(_window(raw, size))
    hashes = [_fingerprint(p) for p in paths]
    frame = pd.DataFrame({"x": np.arange(12, dtype=float), "audio": paths, "y": [0, 1] * 6})
    session = _session(frame, backing, tmp_path)
    before = session.dataset.to_pandas().copy(deep=True)
    normalized = consumer != "raw-fusion"
    mean = sum(real) / len(real)
    std = math.sqrt(sum((v - mean) ** 2 for v in real) / len(real))
    waves = np.asarray(expected, dtype=np.float32)
    expected_waves = (waves - mean) / std if normalized else waves
    if consumer == "speech":
        bundle = session.dl.make_speech_loaders(
            audio_column="audio",
            sample_rate=target,
            max_samples=size,
            normalize_audio=True,
            batch_size=8,
            shuffle_train=False,
        )
        contract = bundle.speech_contract
        assert contract.audio_mean == pytest.approx(mean, abs=1e-8)
        assert contract.audio_std == pytest.approx(std, abs=1e-8)
    else:
        bundle = session.dl.make_audio_loaders(
            audio_column="audio",
            numeric_columns=["x"],
            audio_sample_rate=target,
            audio_max_samples=size,
            normalize=False,
            normalize_audio=normalized,
            batch_size=8,
            shuffle_train=False,
        )
        contract = bundle.multimodal_contract
        if normalized:
            np.testing.assert_allclose(contract.audio_mean, [mean], atol=1e-8, rtol=0)
            np.testing.assert_allclose(contract.audio_std, [std], atol=1e-8, rtol=0)
        else:
            assert contract.audio_mean is contract.audio_std is None
    for name, positions in [("train", list(range(8))), ("validation", [8, 9]), ("test", [10, 11])]:
        batch = next(iter(bundle.loaders[name]))
        audio = batch[0] if consumer == "speech" else batch[1]
        np.testing.assert_allclose(
            audio.numpy()[:, 0], expected_waves[positions], atol=3e-6, rtol=1e-6
        )
        np.testing.assert_array_equal(batch[-1].numpy(), frame.y.iloc[positions])
    # The genuine model receives those tensors; quality on these tiny clips is not asserted.
    if consumer == "speech":
        session.dl.fit_speech(epochs=1, device="cpu")
    else:
        session.dl.fit(epochs=1, device="cpu")
    assert session.dl.evaluate(partition="validation").n_rows == 2
    module = session.dl_train_result.module.cpu().eval()
    batch = next(iter(bundle.loaders["test"]))
    with torch.no_grad():
        inputs = batch[0] if consumer == "speech" else tuple(batch[:-1])
        prediction = module(inputs)
    assert torch.isfinite(prediction).all()
    saved = session.dl.save_bundle(tmp_path / "saved-audio")
    artifact_hashes = {p.name: _fingerprint(p) for p in saved.iterdir() if p.is_file()}
    restored = _session(frame, backing, tmp_path)
    restored.dl.load_bundle(saved, copy.deepcopy(module), map_location="cpu", trusted=True)
    with torch.no_grad():
        actual = restored.dl_train_result.module.eval()(inputs)
    torch.testing.assert_close(actual, prediction, atol=0, rtol=0)
    if consumer != "speech":
        replay = restored.dl.make_multimodal_loaders(
            use_saved_preprocess=True, batch_size=8, shuffle_train=False
        )
        for name in bundle.loaders:
            for original, loaded in zip(
                next(iter(bundle.loaders[name])), next(iter(replay.loaders[name]))
            ):
                torch.testing.assert_close(loaded, original, atol=0, rtol=0)
    pd.testing.assert_frame_equal(session.dataset.to_pandas(), before)
    assert [_fingerprint(p) for p in paths] == hashes
    assert {p.name: _fingerprint(p) for p in saved.iterdir() if p.is_file()} == artifact_hashes


@pytest.mark.parametrize("parameter", ["sample_rate", "max_samples", "source_sample_rate"])
@pytest.mark.parametrize("bad", [True, 1.5, "8", np.nan, np.inf, 0, -1])
@pytest.mark.parametrize("entry", ["decode", "batch", "empty-batch"])
def test_invalid_sample_counts_are_refused(parameter, bad, entry):
    kwargs = {"sample_rate": 8, "max_samples": 9, "source_sample_rate": 8, parameter: bad}
    with pytest.raises(ValidationError, match="sample_rate|max_samples|sample rates"):
        if entry == "decode":
            decode_audio_cell([0, 1, 0], **kwargs)
        else:
            stack_audio_column([] if entry == "empty-batch" else [[0, 1, 0]], **kwargs)


@pytest.mark.parametrize("parameter", ["sample_rate", "max_samples"])
@pytest.mark.parametrize("entry", ["decode", "batch", "empty-batch"])
def test_required_counts_cannot_be_none(parameter, entry):
    with pytest.raises(ValidationError, match="sample_rate|max_samples"):
        if entry == "decode":
            decode_audio_cell([0.0], **{parameter: None})
        else:
            stack_audio_column([] if entry == "empty-batch" else [[0.0]], **{parameter: None})


@pytest.mark.parametrize("consumer", ["fusion", "speech"])
@pytest.mark.parametrize(
    "parameter,bad", [("sample_rate", "8000"), ("max_samples", None), ("source_sample_rate", 1.5)]
)
def test_invalid_loader_rates_preserve_prior_state(consumer, parameter, bad, tmp_path):
    pytest.importorskip("torch")
    frame = pd.DataFrame(
        {"x": np.arange(12, dtype=float), "audio": [[0.0, 0.5, -0.5]] * 12, "y": [0, 1] * 6}
    )
    session = _session(frame, "pandas", tmp_path)
    previous = session.dl.make_audio_loaders(
        audio_column="audio", audio_max_samples=16, batch_size=8
    )
    history = copy.deepcopy(session.history)
    kwargs = {parameter: bad}
    with pytest.raises(ValidationError, match="sample_rate|max_samples|sample rates"):
        if consumer == "fusion":
            session.dl.make_audio_loaders(
                audio_column="audio", **{"audio_" + k: v for k, v in kwargs.items()}
            )
        else:
            session.dl.make_speech_loaders(audio_column="audio", **kwargs)
    assert session._torch_loaders is previous and session.dl_train_result is None
    assert session.history == history


@pytest.mark.parametrize(
    "value",
    [
        np.array([0, np.nan]),
        np.array([0, np.inf]),
        np.array([0, -np.inf]),
        np.array([0, 1e100]),
        np.array([0, 1j]),
        np.array(["0", "1"]),
    ],
)
def test_nonreal_or_nonfinite_waveforms_are_refused(value):
    with pytest.raises(ValidationError, match="[Aa]udio|[Ww]aveform"):
        decode_audio_cell(value, max_samples=8)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("position", [0, 8, 10])
@pytest.mark.parametrize("target_rate", [8000, 4000])
def test_nonfinite_files_cannot_publish_native_loaders(normalize, position, target_rate, tmp_path):
    pytest.importorskip("torch")
    sf = pytest.importorskip("soundfile")
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    sf.write(good, np.arange(12, dtype=float) / 16, 8000, subtype="FLOAT")
    sf.write(bad, [0.0, np.nan, 0.5], 8000, subtype="FLOAT")
    paths = [str(good)] * 12
    paths[position] = str(bad)
    session = _session(
        pd.DataFrame({"x": np.arange(12, dtype=float), "audio": paths, "y": [0, 1] * 6}),
        "pandas",
        tmp_path,
    )
    history = copy.deepcopy(session.history)
    with pytest.raises(ValidationError, match="[Aa]udio|[Ww]aveform"):
        session.dl.make_audio_loaders(
            audio_column="audio",
            normalize_audio=normalize,
            audio_sample_rate=target_rate,
            audio_max_samples=16,
        )
    assert session._torch_loaders is None and session.history == history


def test_real_lengths_define_statistics_and_empty_clips_do_not_add_silence():
    batch, lengths = stack_audio_column([[1, 3], [], [5, 7, 9]], max_samples=8, return_lengths=True)
    mean, std = fit_audio_waveform_stats(batch, lengths)
    np.testing.assert_allclose(mean, [5.0], atol=0, rtol=0)
    np.testing.assert_allclose(std, [math.sqrt(8.0)], atol=1e-12, rtol=0)
    normalized = apply_audio_waveform_stats(batch, mean, std)
    np.testing.assert_allclose(normalized, (batch.astype(float) - 5) / math.sqrt(8), atol=1e-7)
    padded_mean, _ = fit_audio_waveform_stats(batch)
    assert padded_mean[0] != mean[0]
    np.testing.assert_array_equal(apply_audio_waveform_stats(batch, [0], [0]), batch)


@pytest.mark.parametrize(
    "audio,lengths,match",
    [
        (np.zeros((2, 3)), None, "N1T"),
        (np.zeros((0, 1, 8)), None, "empty train"),
        (np.zeros((2, 1, 8)), [1], "align"),
        (np.zeros((2, 1, 8)), [0, 0], "empty waveforms"),
    ],
)
def test_statistics_refuse_absent_or_misaligned_training_observations(audio, lengths, match):
    with pytest.raises(ValidationError, match=match):
        fit_audio_waveform_stats(audio, lengths)


@pytest.mark.parametrize("kind", ["missing", "corrupt", "unsupported"])
def test_unreadable_cells_report_audio_validation_errors(kind, tmp_path):
    pytest.importorskip("soundfile")
    path = tmp_path / "bad.wav"
    if kind == "corrupt":
        path.write_bytes(b"not a recording")
    with pytest.raises(ValidationError, match="Audio|audio"):
        decode_audio_cell(object() if kind == "unsupported" else path)
