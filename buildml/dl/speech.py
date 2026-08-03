"""Transcribe speech and classify audio, without pretending to train Whisper.

Two things live here, and they answer different questions.

**Transcription** turns audio into text. It runs an existing pretrained model —
a Whisper-class Hugging Face pipeline — because a model that can transcribe
general speech was trained on hundreds of thousands of hours of audio, and
nothing you do in a Session will reproduce that. A deterministic stub backend
exists so tests can exercise the plumbing offline; its output is nonsense and is
labelled as such.

**Classification** trains a small encoder on your labelled audio to predict a
category. Not what was said, but which class the clip belongs to — a speaker, a
sound type, a quality judgement. The encoder is a small 1D-CNN, trained from
scratch on your data, and it is honest about being small.

Alongside those, :func:`evaluate_asr` computes word and character error rates
from strings alone, needing no models at all.

What is explicitly out of scope is foundation-model pretraining.
:func:`refuse_foundation_model_pretrain` exists to say so clearly rather than
letting a caller discover it slowly: the data and compute involved are not
things a pip package can supply, and a method that claimed otherwise would be
lying about what it does. Fine-tuning on a small corpus is available and is
named accordingly.

See Also
--------
buildml.dl.audio : Decoding and normalising waveforms.
buildml.dl.zoo : Pretrained speech encoders for feature extraction.
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from buildml.core.errors import MissingExtraError, ValidationError
from buildml.core.types import ColumnRole
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan, assert_fit_partition
from buildml.dl.audio import (
    apply_audio_waveform_stats,
    decode_audio_cell,
    fit_audio_waveform_stats,
    stack_audio_column,
)
from buildml.dl.extras import require_torch
from buildml.dl.labels import encode_class_targets, fit_class_labels
from buildml.dl.results import LoaderReport, TorchLoaderBundle
from buildml.dl.types import FeatureContract

SpeechBackend = Literal["stub", "transformers"]
SpeechMode = Literal["classify", "asr"]


def refuse_foundation_model_pretrain(*, requested: str = "foundation_model_pretrain") -> None:
    """Refuse foundation-model pretraining, and say what to do instead.

    Always raises. Training a Whisper-scale model needs hundreds of thousands of
    hours of audio and a cluster running for weeks, and no amount of API surface
    in a pip package changes that. A method that accepted the request and
    produced something would be misrepresenting what it did.

    Parameters
    ----------
    requested:
        What was asked for. Included in the message.

    Raises
    ------
    ValidationError
        Always, naming the alternatives: fine-tuning on a small labelled corpus,
        domain adaptation, transcription with pretrained weights, or loading a
        pretrained encoder for feature extraction.

    See Also
    --------
    domain_adapt_speech_disclosures : What the supported path does claim.
    make_speech_loaders : Fine-tuning on your own labels.
    transcribe_audio_values : Using someone else's pretrained model.
    """
    raise ValidationError(
        f"Refusing {requested!r}: BuildML does not train Whisper-scale / "
        "foundation speech models from scratch, and does not run large-scale "
        "continued pretrain as a library feature (data + compute are outside "
        "a pip package). Use fit_speech_torch / domain_adapt_speech_torch for "
        "finetune-lite / domain adapt on small labeled corpora, or "
        "transcribe_speech / load_pretrained_backbone for pretrained inference "
        "and encoder hooks."
    )


def domain_adapt_speech_disclosures() -> tuple[str, ...]:
    """State plainly what speech fine-tuning here is and is not.

    Attached to results from the domain-adapt path so the claim travels with
    the numbers. Fine-tuning a small encoder on a Session-sized corpus is
    useful; it is a different thing from continued pretraining of a foundation
    model, and the two are easy to conflate.

    Returns
    -------
    tuple of str
        Two statements: that this is fine-tuning rather than pretraining, and
        where to go for genuinely pretrained weights.

    See Also
    --------
    refuse_foundation_model_pretrain : The explicit refusal.
    """
    return (
        "domain_adapt_speech_torch is finetune-lite / domain adapt on a small "
        "Session corpus — not continued pretrain of a foundation model.",
        "For Whisper-class ASR weights use transcribe_speech(backend='transformers') "
        "or load_pretrained_backbone(modality='speech'); BuildML will not "
        "pretrain those stacks from scratch.",
    )


@dataclass(slots=True)
class SpeechLoaderConfig:
    """Settings for building speech classification loaders.

    Attributes
    ----------
    batch_size:
        Clips per batch. Small by default, since waveforms are large.
    num_workers:
        Background loading processes. Worth raising when decoding audio files
        is the bottleneck.
    pin_memory:
        Page-lock batches for faster GPU transfer. Train loader only.
    shuffle_train:
        Shuffle the training loader.
    drop_last:
        Discard a final short batch.
    seed:
        Controls shuffling.
    sample_rate:
        Target rate in Hz. Clips at other rates are resampled.
    max_samples:
        Waveform length after padding or truncation. At 16 kHz, the default is
        one second — raise it for longer clips, at a cost in memory and time.
    source_sample_rate:
        The rate of incoming arrays, when supplied without one.
    normalize_audio:
        Standardise amplitude using training statistics. Usually worth keeping
        on, since recording level varies far more than anything you want the
        model to learn from.
    encoder_dim:
        Width of the encoder's output representation.

    See Also
    --------
    make_speech_loaders : Consumes this.
    """

    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    shuffle_train: bool = True
    drop_last: bool = False
    seed: int = 0
    sample_rate: int = 16_000
    max_samples: int = 16_000
    source_sample_rate: int | None = None
    normalize_audio: bool = True
    encoder_dim: int = 64

    def to_dict(self) -> dict[str, Any]:
        """Return the loader settings as JSON-safe values.

        Records how a speech run was configured, so it can be arranged the same
        way later.

        Returns
        -------
        dict
            Every field.
        """
        return asdict(self)


@dataclass(slots=True)
class SpeechContract:
    """What a speech classifier needs in order to be fed correctly.

    Attributes
    ----------
    audio_column:
        The source column.
    target_column:
        What is being predicted.
    task:
        Always ``'classification'``. Speech regression is not on this path.
    class_labels:
        The class vocabulary, indexed by predicted class id.
    sample_rate, max_samples, source_sample_rate:
        The audio geometry every clip is coerced to.
    audio_mean, audio_std:
        Train-fitted amplitude statistics. ``None`` when normalisation was off.
    encoder_dim:
        Encoder output width.
    modality:
        ``'speech_classify'``.
    disclosures:
        What this path is, carried with the contract so the claim survives
        persistence.

    Notes
    -----
    **The audio geometry is part of the contract, not a preference.** A model
    trained on one-second clips at 16 kHz has learned at that resolution;
    feeding it something else produces predictions that mean nothing in
    particular.

    See Also
    --------
    make_speech_loaders : Produces this.
    """

    audio_column: str
    target_column: str
    task: Literal["classification"] = "classification"
    class_labels: tuple[Any, ...] = ()
    sample_rate: int = 16_000
    max_samples: int = 16_000
    source_sample_rate: int | None = None
    audio_mean: float | None = None
    audio_std: float | None = None
    encoder_dim: int = 64
    modality: str = "speech_classify"
    disclosures: tuple[str, ...] = ()

    def to_feature_contract(self) -> FeatureContract:
        """Project this down to the tabular-shaped contract shared code expects.

        Loader reports and evaluation take a plain
        :class:`~buildml.dl.types.FeatureContract`. This produces one listing
        the audio column as the single feature.

        Returns
        -------
        FeatureContract
            The flattened view.

        Notes
        -----
        **The audio geometry and amplitude statistics do not survive the
        projection**, since a tabular contract has nowhere to record them. Keep
        the ``SpeechContract`` for anything that needs to rebuild loaders. The
        normalisation fields are left ``None`` because audio amplitude
        standardisation is not per-column scaling and would be misread as such.
        """
        return FeatureContract(
            feature_columns=(self.audio_column,),
            target_column=self.target_column,
            task="classification",
            class_labels=self.class_labels,
            normalize_mean=None,
            normalize_std=None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the speech contract as JSON-safe values.

        Complete and round-trippable — :meth:`from_dict` reconstructs an
        equivalent contract, which is how a saved speech model gets its audio
        geometry back.

        Returns
        -------
        dict
            Columns, task, class labels, audio geometry, amplitude statistics,
            encoder width, modality, and disclosures.

        See Also
        --------
        from_dict : The inverse.
        """
        return {
            "audio_column": self.audio_column,
            "target_column": self.target_column,
            "task": self.task,
            "class_labels": list(self.class_labels),
            "sample_rate": self.sample_rate,
            "max_samples": self.max_samples,
            "source_sample_rate": self.source_sample_rate,
            "audio_mean": self.audio_mean,
            "audio_std": self.audio_std,
            "encoder_dim": self.encoder_dim,
            "modality": self.modality,
            "disclosures": list(self.disclosures),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SpeechContract:
        """Rebuild a contract from its serialised form.

        The inverse of :meth:`to_dict`, used when reloading a trainer bundle so
        the restored model is fed audio prepared the way it was trained.

        Parameters
        ----------
        payload:
            A mapping from :meth:`to_dict`.

        Returns
        -------
        SpeechContract
            The reconstructed contract.

        Raises
        ------
        KeyError
            If ``audio_column`` or ``target_column`` is absent. Everything else
            has a defensible default; those two do not.
        """
        return cls(
            audio_column=str(payload["audio_column"]),
            target_column=str(payload["target_column"]),
            task=payload.get("task") or "classification",
            class_labels=tuple(payload.get("class_labels") or ()),
            sample_rate=int(payload.get("sample_rate") or 16_000),
            max_samples=int(payload.get("max_samples") or 16_000),
            source_sample_rate=(
                None
                if payload.get("source_sample_rate") is None
                else int(payload["source_sample_rate"])
            ),
            audio_mean=(
                None if payload.get("audio_mean") is None else float(payload["audio_mean"])
            ),
            audio_std=(
                None if payload.get("audio_std") is None else float(payload["audio_std"])
            ),
            encoder_dim=int(payload.get("encoder_dim") or 64),
            modality=str(payload.get("modality") or "speech_classify"),
            disclosures=tuple(payload.get("disclosures") or ()),
        )


@dataclass(slots=True)
class AsrEvalResult:
    """How far a set of transcriptions is from what was actually said.

    Attributes
    ----------
    n_utterances:
        How many were compared.
    wer:
        Word error rate — word-level edits divided by reference words. 0.0 is
        perfect; values above 1.0 are possible when a hypothesis inserts more
        words than the reference contains.
    cer:
        Character error rate, computed the same way over characters.
    n_ref_words, n_ref_chars:
        Reference totals, the denominators. A corpus this small makes the rates
        noisy, and this is where you notice.
    per_utterance:
        Each pair with its own rates. Where you find the individual failures
        that the corpus rate averages away.
    disclosures, limitations, warnings:
        How the metrics were computed and what they do not capture.

    Notes
    -----
    **Word error rate punishes formatting as harshly as meaning.** A transcript
    that writes "twenty five" where the reference has "25" scores two errors
    despite being correct. Normalise both sides before comparing when
    formatting is not what you are measuring.

    **Character error rate is the more forgiving view.** A near-miss on one word
    costs a few characters rather than a whole word, which makes it better for
    tracking small improvements and for languages where word boundaries are
    not obvious.

    See Also
    --------
    evaluate_asr : Produces this.
    """

    n_utterances: int
    wer: float
    cer: float
    n_ref_words: int
    n_ref_chars: int
    per_utterance: tuple[dict[str, Any], ...] = ()
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the error rates as JSON-safe values.

        Includes every per-utterance comparison, since the worst few are
        usually more actionable than the corpus average.

        Returns
        -------
        dict
            Utterance count, WER, CER, reference totals, per-utterance
            comparisons, and the three prose lists.
        """
        return {
            "n_utterances": self.n_utterances,
            "wer": self.wer,
            "cer": self.cer,
            "n_ref_words": self.n_ref_words,
            "n_ref_chars": self.n_ref_chars,
            "per_utterance": list(self.per_utterance),
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
        }


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[m]


def evaluate_asr(
    *,
    hypotheses: list[str] | tuple[str, ...],
    references: list[str] | tuple[str, ...],
    lowercase: bool = True,
) -> AsrEvalResult:
    """Measure transcription accuracy against known-correct text.

    Compares each hypothesis to its reference using Levenshtein edit distance,
    at both word and character level. Corpus rates are computed by pooling edits
    and reference lengths, which weights long utterances more heavily than
    averaging per-utterance rates would — and is the standard definition.

    Pure string comparison. No models, no audio, no downloads.

    Parameters
    ----------
    hypotheses:
        What the system produced.
    references:
        What was actually said.
    lowercase:
        Lowercase and collapse whitespace before comparing. On by default, so
        capitalisation differences do not count as errors.

    Returns
    -------
    AsrEvalResult
        Corpus rates, reference totals, and each pair's own rates.

    Raises
    ------
    ValidationError
        If the two sequences differ in length, or if either is empty.

    Notes
    -----
    **Edit distance counts substitutions, insertions, and deletions equally.**
    A transcript that drops a word and one that swaps a word both cost one
    error, though they can matter very differently downstream.

    **Text normalisation is your responsibility beyond casing.** Punctuation,
    numbers, and contractions all inflate error rates when the two sides
    disagree on convention rather than content.

    **Corpus rates pool edits rather than averaging rates.** One badly
    transcribed long utterance moves the corpus number more than one badly
    transcribed short one, which is usually what you want.

    Examples
    --------
    >>> result = evaluate_asr(
    ...     hypotheses=["the cat sat", "hello world"],
    ...     references=["the cat sat", "hello there"],
    ... )
    >>> round(result.wer, 3)
    0.2

    See Also
    --------
    transcribe_audio_values : Producing the hypotheses.
    """
    if len(hypotheses) != len(references):
        raise ValidationError(
            f"hypotheses/references length mismatch: {len(hypotheses)} vs {len(references)}"
        )
    if not hypotheses:
        raise ValidationError("evaluate_asr requires at least one utterance")
    word_edits = 0
    char_edits = 0
    n_ref_words = 0
    n_ref_chars = 0
    per: list[dict[str, Any]] = []
    for hyp_raw, ref_raw in zip(hypotheses, references, strict=True):
        if lowercase:
            hyp = " ".join(str(hyp_raw or "").strip().lower().split())
        else:
            hyp = " ".join(str(hyp_raw).split())
        if lowercase:
            ref = " ".join(str(ref_raw or "").strip().lower().split())
        else:
            ref = " ".join(str(ref_raw).split())
        hyp_words, ref_words = hyp.split(), ref.split()
        hyp_chars, ref_chars = list(hyp.replace(" ", "")), list(ref.replace(" ", ""))
        w_ed = _edit_distance(ref_words, hyp_words)
        c_ed = _edit_distance(ref_chars, hyp_chars)
        word_edits += w_ed
        char_edits += c_ed
        n_ref_words += max(len(ref_words), 1)
        n_ref_chars += max(len(ref_chars), 1)
        per.append(
            {
                "hypothesis": hyp,
                "reference": ref,
                "wer": w_ed / max(len(ref_words), 1),
                "cer": c_ed / max(len(ref_chars), 1),
            }
        )
    return AsrEvalResult(
        n_utterances=len(hypotheses),
        wer=float(word_edits / n_ref_words),
        cer=float(char_edits / n_ref_chars),
        n_ref_words=n_ref_words,
        n_ref_chars=n_ref_chars,
        per_utterance=tuple(per),
        disclosures=("WER/CER via Levenshtein edit distance.",),
        limitations=("String metrics only — not a speech quality product.",),
    )


@dataclass(slots=True)
class SpeechTranscribeResult:
    """Transcribed text, with a clear record of what produced it.

    Attributes
    ----------
    texts:
        One transcript per input clip, in order. Failures appear as
        ``'[asr-error]'`` rather than shifting the alignment.
    backend:
        ``'stub'`` or ``'transformers'``. **Check this before reading the
        text** — stub output is a deterministic fingerprint, not speech.
    model_id:
        Which model ran.
    n_rows:
        How many clips were processed.
    disclosures:
        How transcription was performed.
    limitations:
        What this path does not claim.
    warnings:
        Per-clip failures from the transformers backend.
    meta:
        Audio geometry, and the column and partition when transcribing from a
        Dataset.

    Notes
    -----
    **``backend='stub'`` means the text is meaningless.** It is derived from a
    hash of the waveform's energy so that tests get stable output offline.
    Anything downstream that treats it as a transcript is measuring noise.

    See Also
    --------
    transcribe_audio_values : Produces this.
    """

    texts: list[str]
    backend: str
    model_id: str | None
    n_rows: int
    disclosures: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the transcription run as JSON-safe values.

        Carries the backend and model alongside the text, so a stored
        transcript never loses the context that says whether it is real.

        Returns
        -------
        dict
            Texts, backend, model id, row count, the three prose lists, and
            metadata.
        """
        return {
            "texts": list(self.texts),
            "backend": self.backend,
            "model_id": self.model_id,
            "n_rows": self.n_rows,
            "disclosures": list(self.disclosures),
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "meta": dict(self.meta),
        }


def require_speech_stack(*, feature: str = "Speech transformers backend") -> Any:
    """Import transformers, or explain how to install it.

    Only the ``transformers`` transcription backend needs this. The stub backend
    and every classification path work without it, which is why the import is
    lazy.

    Parameters
    ----------
    feature:
        What the caller was doing. Appears in the error message.

    Returns
    -------
    module
        The ``transformers`` module.

    Raises
    ------
    MissingExtraError
        If transformers is absent. Install with ``pip install buildml[speech]``.

    See Also
    --------
    speech_stack_available : The boolean form.
    """
    try:
        import transformers
    except ImportError as exc:
        raise MissingExtraError("speech", feature) from exc
    return transformers


def speech_stack_available() -> bool:
    """Report whether the transformers speech backend can be used.

    Consults package metadata rather than importing, since importing
    transformers is slow enough to be worth avoiding for a capability check.

    Returns
    -------
    bool
        True when a transformers distribution is installed.

    Notes
    -----
    Installation is not the same as a working install. A broken transformers
    reports ``True`` here and fails at :func:`require_speech_stack`, which is
    the right place for that failure to surface.

    See Also
    --------
    require_speech_stack : The raising form.
    """
    return importlib.util.find_spec("transformers") is not None


def _resolve_audio_target(
    dataset: Dataset,
    audio_column: str | None,
) -> tuple[str, str]:
    roles = dataset.roles
    target_cols = [c for c, r in roles.items() if r == ColumnRole.TARGET]
    if len(target_cols) != 1:
        raise ValidationError(
            "Speech classify path requires exactly one target role column."
        )
    target = target_cols[0]
    if audio_column is not None:
        if audio_column not in dataset.columns:
            raise ValidationError(f"audio_column {audio_column!r} not in dataset")
        return audio_column, target
    feature_cols = [c for c, r in roles.items() if r == ColumnRole.FEATURE]
    if len(feature_cols) != 1:
        raise ValidationError(
            "Pass audio_column= explicitly when multiple feature columns exist."
        )
    return feature_cols[0], target


def build_tiny_speech_encoder(
    *,
    in_channels: int = 1,
    embed_dim: int = 64,
    sample_rate: int = 16_000,
) -> Any:
    """Build a small convolutional encoder that turns waveforms into vectors.

    Three strided 1D convolutions reduce the time dimension while widening the
    channels, then global average pooling collapses what remains into one
    fixed-width vector per clip. Each stride-2 layer halves the temporal
    resolution, so a 16000-sample clip reaches a manageable length by the third
    layer.

    Parameters
    ----------
    in_channels:
        Input channels. 1 for mono.
    embed_dim:
        Output width.
    sample_rate:
        Recorded on the module for reference. Does not change the computation,
        but a model trained at one rate should not be fed another.

    Returns
    -------
    torch.nn.Module
        Accepting ``(B, 1, T)`` or ``(B, T)`` and emitting ``(B, embed_dim)``.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.

    Notes
    -----
    **Trained from scratch on your data, with the ceiling that implies.** This
    has none of the acoustic knowledge a pretrained encoder brings, so it needs
    more labelled examples to reach a given accuracy and will not match
    Wav2Vec2 or HuBERT on most tasks. It is fast, dependency-free, and a
    reasonable baseline — see :mod:`buildml.dl.zoo` when you want more.

    **Global pooling discards timing entirely.** Two clips containing the same
    sounds in different orders produce similar representations, which is fine
    for classifying sound type and wrong for anything sequential.

    See Also
    --------
    build_speech_classifier : This with a head attached.
    buildml.dl.zoo.load_audio_backbone : The pretrained alternative.
    """
    torch = require_torch(feature="Tiny speech encoder")

    class TinySpeechEncoder(torch.nn.Module):
        modality = "speech_encoder"

        def __init__(self) -> None:
            super().__init__()
            self.sample_rate = int(sample_rate)
            self.embed_dim = int(embed_dim)
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, 16, kernel_size=9, stride=2, padding=4),
                torch.nn.ReLU(),
                torch.nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
                torch.nn.ReLU(),
                torch.nn.Conv1d(32, embed_dim, kernel_size=5, stride=2, padding=2),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1),
            )

        def forward(self, x: Any) -> Any:
            # x: (B, 1, T) or (B, T)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            h = self.net(x)
            return h.squeeze(-1)

    return TinySpeechEncoder()


def build_speech_classifier(
    *,
    n_classes: int,
    embed_dim: int = 64,
    sample_rate: int = 16_000,
    freeze_encoder: bool = False,
) -> Any:
    """Build a complete model for classifying audio clips.

    Combines :func:`build_tiny_speech_encoder` with a linear head sized to your
    class count. Ready to train with the ordinary training loop.

    Parameters
    ----------
    n_classes:
        Number of classes.
    embed_dim:
        Encoder output width, which is also the head's input width.
    sample_rate:
        Recorded on the encoder for reference.
    freeze_encoder:
        Train only the head. Rarely useful here — a randomly initialised
        encoder has learned nothing worth preserving, so freezing it leaves the
        head classifying random projections. It exists for the case where you
        have loaded encoder weights from elsewhere.

    Returns
    -------
    torch.nn.Module
        Accepting waveforms and emitting ``(B, n_classes)`` logits.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed.
    ValidationError
        If ``n_classes`` is below 2.

    Notes
    -----
    **Raw logits, no softmax.** ``CrossEntropyLoss`` applies its own
    log-softmax, and applying it twice degrades the gradient.

    Examples
    --------
    Size from the loader bundle::

        bundle = make_speech_loaders(dataset, split_plan, audio_column="clip")
        module = build_speech_classifier(
            n_classes=len(bundle.contract.class_labels),
        )

    See Also
    --------
    make_speech_loaders : Producing matching loaders.
    buildml.dl.train.train_supervised_module : Training it.
    """
    torch = require_torch(feature="Speech classifier")
    if n_classes < 2:
        raise ValidationError("n_classes must be >= 2 for speech classification")
    encoder = build_tiny_speech_encoder(embed_dim=embed_dim, sample_rate=sample_rate)
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    class SpeechClassifier(torch.nn.Module):
        modality = "speech_classify"

        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = torch.nn.Linear(embed_dim, n_classes)
            self.n_classes = int(n_classes)
            self.embed_dim = int(embed_dim)
            self.freeze_encoder = bool(freeze_encoder)

        def forward(self, x: Any) -> Any:
            return self.head(self.encoder(x))

    return SpeechClassifier()


def make_speech_loaders(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    audio_column: str | None = None,
    config: SpeechLoaderConfig | None = None,
) -> TorchLoaderBundle:
    """Build loaders that feed waveforms to a speech classifier.

    Finds the audio and target columns, decodes the training clips to fit
    amplitude statistics, then decodes and normalises every partition with those
    same statistics.

    Parameters
    ----------
    dataset:
        The data, with exactly one target role and a numeric target.
    split_plan:
        Which rows belong to which partition. A train partition is required.
    audio_column:
        The audio column. Inferred when exactly one feature column exists.
    config:
        Batching, audio geometry, and normalisation settings.

    Returns
    -------
    TorchLoaderBundle
        The loaders, the feature contract, the speech contract, and a report.

    Raises
    ------
    MissingExtraError
        If PyTorch is not installed. Decoding audio files also needs soundfile.
    ValidationError
        If there is not exactly one target role, if the audio column is absent
        or ambiguous, if the train partition is empty, if the target is not
        numeric class ids, or if a config value is out of range.

    Notes
    -----
    **Amplitude statistics come from training clips only**, and are computed
    length-aware so repeat-padding does not skew them toward short clips.

    **Labels must already be integers**, and they are remapped internally to
    contiguous ``0..K-1`` while the contract keeps your original ids.

    **Clips are padded or truncated to ``max_samples``.** The default of one
    second at 16 kHz is short for many tasks — a longer clip loses everything
    past the first second, silently. Raise it to match your audio.

    Examples
    --------
    Three-second clips::

        cfg = SpeechLoaderConfig(max_samples=48_000, sample_rate=16_000)
        bundle = make_speech_loaders(
            dataset, split_plan, audio_column="clip", config=cfg,
        )

    See Also
    --------
    build_speech_classifier : The matching model.
    buildml.dl.audio : The decoding and normalisation underneath.
    """
    assert_fit_partition(split_plan, "train")
    assert split_plan is not None
    require_torch(feature="Speech Torch DataLoaders")
    cfg = config or SpeechLoaderConfig()
    if cfg.batch_size < 1:
        raise ValidationError("batch_size must be >= 1")
    if cfg.max_samples < 1:
        raise ValidationError("max_samples must be >= 1")

    audio_col, target = _resolve_audio_target(dataset, audio_column)
    frame = dataset._ensure_pandas()
    train_idx = list(split_plan.indices_for("train"))
    if not train_idx:
        raise ValidationError("Train partition is empty; cannot build speech loaders")

    train_audio, train_lengths = stack_audio_column(
        frame.iloc[train_idx][audio_col].tolist(),
        sample_rate=cfg.sample_rate,
        max_samples=cfg.max_samples,
        source_sample_rate=cfg.source_sample_rate,
        return_lengths=True,
    )
    audio_mean: float | None = None
    audio_std: float | None = None
    if cfg.normalize_audio:
        mean_arr, std_arr = fit_audio_waveform_stats(train_audio, lengths=train_lengths)
        audio_mean = float(mean_arr.reshape(-1)[0])
        audio_std = float(std_arr.reshape(-1)[0])

    y_train = frame.iloc[train_idx][target]
    if not pd.api.types.is_numeric_dtype(y_train):
        raise ValidationError(
            f"Target '{target}' must be numeric class ids for the speech classify path "
            "(encode labels to integers first)."
        )
    class_labels = fit_class_labels(y_train)
    disclosures = (
        "Speech classify finetune-lite: tiny 1D-CNN encoder + linear head.",
        "Not training a Whisper-scale foundation model from scratch.",
        "Amplitude mean/std fit on train only when normalize_audio=True.",
        "Class labels remapped to contiguous 0..K-1; class_labels stores original ids.",
    )
    contract = SpeechContract(
        audio_column=audio_col,
        target_column=target,
        class_labels=class_labels,
        sample_rate=cfg.sample_rate,
        max_samples=cfg.max_samples,
        source_sample_rate=cfg.source_sample_rate,
        audio_mean=audio_mean,
        audio_std=audio_std,
        encoder_dim=cfg.encoder_dim,
        disclosures=disclosures,
    )

    torch = require_torch(feature="Speech Torch DataLoaders")
    generator = torch.Generator()
    generator.manual_seed(int(cfg.seed))
    loaders: dict[str, Any] = {}
    warnings: list[str] = list(disclosures)
    n_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}

    for name in ("train", "validation", "test"):
        idx = list(split_plan.indices_for(name))  # type: ignore[arg-type]
        if not idx:
            if name == "train":
                raise ValidationError("Train partition is empty")
            warnings.append(f"Partition '{name}' is empty; no DataLoader created.")
            continue
        part = frame.iloc[idx]
        waves = stack_audio_column(
            part[audio_col].tolist(),
            sample_rate=cfg.sample_rate,
            max_samples=cfg.max_samples,
            source_sample_rate=cfg.source_sample_rate,
        )
        if audio_mean is not None and audio_std is not None:
            waves = apply_audio_waveform_stats(
                waves,
                np.asarray([audio_mean], dtype=np.float32),
                np.asarray([audio_std], dtype=np.float32),
            )
        y = encode_class_targets(part[target], class_labels)
        x_t = torch.as_tensor(waves, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)
        dataset_t = torch.utils.data.TensorDataset(x_t, y_t)
        shuffle = bool(cfg.shuffle_train and name == "train")
        loaders[name] = torch.utils.data.DataLoader(
            dataset_t,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and name == "train",
            drop_last=cfg.drop_last and name == "train",
            generator=generator if shuffle else None,
        )
        n_counts[name] = len(idx)

    feature_contract = contract.to_feature_contract()
    report = LoaderReport(
        batch_size=cfg.batch_size,
        shuffle_train=cfg.shuffle_train,
        normalize=bool(cfg.normalize_audio),
        feature_columns=feature_contract.feature_columns,
        target_column=feature_contract.target_column,
        task="classification",
        n_train=n_counts["train"],
        n_validation=n_counts["validation"],
        n_test=n_counts["test"],
        class_labels=class_labels,
        warnings=warnings,
        split_kind=split_plan.kind,
        group_column=None,
        time_column=None,
        groups_disjoint=None,
        time_order_ok=None,
    )
    return TorchLoaderBundle(
        loaders=loaders,
        contract=feature_contract,
        report=report,
        modality="speech_classify",
        input_layout=("audio",),
        speech_contract=contract,
    )


def _stub_transcribe_one(wave: np.ndarray, *, sample_rate: int) -> str:
    """Deterministic CI-safe pseudo-transcript from waveform energy fingerprint."""
    flat = np.asarray(wave, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return "[silence]"
    energy = float(np.mean(np.square(flat)))
    digest = hashlib.sha1(
        f"{sample_rate}:{energy:.6f}:{flat[:8].tobytes().hex()}".encode()
    ).hexdigest()[:8]
    bucket = int(digest[:2], 16) % 5
    tokens = ("alpha", "bravo", "charlie", "delta", "echo")
    return f"[stub-asr] {tokens[bucket]} {digest}"


def _load_transformers_asr(model_id: str) -> Any:
    transformers = require_speech_stack(feature="Speech ASR (transformers)")
    # Prefer pipeline API — Whisper-class and compatible seq2seq ASR models.
    return transformers.pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=-1,
    )


def transcribe_audio_values(
    values: list[Any],
    *,
    backend: SpeechBackend = "stub",
    model_id: str | None = None,
    sample_rate: int = 16_000,
    max_samples: int = 16_000,
    source_sample_rate: int | None = None,
) -> SpeechTranscribeResult:
    """Turn audio into text, using a pretrained model or a testing stub.

    Decodes each cell to a waveform and runs the chosen backend over it. The
    transformers backend uses a Hugging Face automatic-speech-recognition
    pipeline; the stub produces deterministic placeholder text without any
    model.

    Parameters
    ----------
    values:
        Audio cells — paths, ``Path`` objects, or waveform arrays.
    backend:
        ``'stub'`` for offline placeholder text, ``'transformers'`` for real
        transcription.
    model_id:
        Hugging Face model id for the transformers backend. Defaults to a tiny
        testing model, which downloads once and transcribes badly — name a real
        model such as ``'openai/whisper-base'`` for actual use.
    sample_rate:
        Target rate in Hz. Whisper-class models expect 16 kHz.
    max_samples:
        Waveform length. Audio beyond this is truncated **before**
        transcription, so long recordings lose their tails.
    source_sample_rate:
        The rate of incoming arrays.

    Returns
    -------
    SpeechTranscribeResult
        Transcripts in input order, the backend and model used, and any
        per-clip failures in ``warnings``.

    Raises
    ------
    MissingExtraError
        If the transformers backend is chosen and transformers is not
        installed.
    ValidationError
        If the backend name is unrecognised, or a cell cannot be decoded.

    Notes
    -----
    **The stub backend does not transcribe anything.** It hashes the waveform's
    energy into a stable phrase so tests can run offline and deterministically.
    Never treat its output as speech.

    **A per-clip failure does not stop the run.** The transformers backend
    records the error in ``warnings`` and puts ``'[asr-error]'`` in that
    position, keeping the output aligned with the input.

    **Truncation happens before transcription and is silent.** With the default
    one-second window, a thirty-second recording is transcribed from its first
    second. Raise ``max_samples`` to cover your audio.

    **Transcription runs on CPU.** Fine for a handful of clips; slow for a
    corpus.

    Examples
    --------
    Real transcription with a named model::

        result = transcribe_audio_values(
            ["clip1.wav", "clip2.wav"],
            backend="transformers",
            model_id="openai/whisper-base",
            max_samples=16_000 * 30,
        )
        result.texts

    See Also
    --------
    transcribe_from_dataset : The Dataset-oriented version.
    evaluate_asr : Scoring the transcripts.
    """
    if backend not in {"stub", "transformers"}:
        raise ValidationError("speech backend must be 'stub' or 'transformers'")
    warnings: list[str] = []
    texts: list[str] = []
    resolved_model = model_id
    limitations = (
        "Integration/finetune path — not training a foundation model from scratch.",
        "Stub backend is for tests/smoke only; do not treat stub text as real ASR.",
        "Transformers path may download weights; keep CI on stub or tiny fixtures.",
    )
    disclosures = (
        f"Speech ASR backend={backend}.",
        "Primary product path is ASR transcription; classify uses fit_speech_torch.",
    )

    if backend == "stub":
        resolved_model = resolved_model or "buildml-stub-asr"
        for value in values:
            wave = decode_audio_cell(
                value,
                sample_rate=sample_rate,
                max_samples=max_samples,
                source_sample_rate=source_sample_rate,
            )
            texts.append(_stub_transcribe_one(wave, sample_rate=sample_rate))
        return SpeechTranscribeResult(
            texts=texts,
            backend=backend,
            model_id=resolved_model,
            n_rows=len(texts),
            disclosures=disclosures,
            limitations=limitations,
            warnings=warnings,
            meta={"sample_rate": sample_rate, "max_samples": max_samples},
        )

    # transformers path
    resolved_model = (
        resolved_model
        or "hf-internal-testing/tiny-random-WhisperForConditionalGeneration"
    )
    asr = _load_transformers_asr(resolved_model)
    for value in values:
        wave = decode_audio_cell(
            value,
            sample_rate=sample_rate,
            max_samples=max_samples,
            source_sample_rate=source_sample_rate,
        )
        mono = np.asarray(wave, dtype=np.float32).reshape(-1)
        try:
            out = asr({"array": mono, "sampling_rate": int(sample_rate)})
            if isinstance(out, dict):
                texts.append(str(out.get("text", "")).strip() or "[empty]")
            else:
                texts.append(str(out).strip() or "[empty]")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"transformers ASR failed for one cell: {exc}")
            texts.append("[asr-error]")
    return SpeechTranscribeResult(
        texts=texts,
        backend=backend,
        model_id=resolved_model,
        n_rows=len(texts),
        disclosures=disclosures + ("Uses Hugging Face transformers ASR pipeline.",),
        limitations=limitations,
        warnings=warnings,
        meta={"sample_rate": sample_rate, "max_samples": max_samples},
    )


def transcribe_from_dataset(
    dataset: Dataset,
    *,
    audio_column: str,
    backend: SpeechBackend = "stub",
    model_id: str | None = None,
    sample_rate: int = 16_000,
    max_samples: int = 16_000,
    source_sample_rate: int | None = None,
    partition: Literal["train", "validation", "test", "all"] = "all",
    split_plan: SplitPlan | None = None,
) -> SpeechTranscribeResult:
    """Transcribe an audio column, optionally limited to one partition.

    A Dataset-shaped wrapper over :func:`transcribe_audio_values`. Pulls the
    column, slices it if asked, transcribes, and records the column and
    partition in the result's metadata.

    Parameters
    ----------
    dataset:
        The data.
    audio_column:
        Which column holds audio.
    backend:
        ``'stub'`` or ``'transformers'``.
    model_id:
        Hugging Face model id for the transformers backend.
    sample_rate / max_samples / source_sample_rate:
        Audio geometry, as in :func:`transcribe_audio_values`.
    partition:
        ``'all'`` for every row, or one partition name.
    split_plan:
        Required unless ``partition='all'``.

    Returns
    -------
    SpeechTranscribeResult
        Transcripts for the selected rows, with the column and partition in
        ``meta``.

    Raises
    ------
    MissingExtraError
        If the transformers backend is chosen and transformers is not
        installed.
    ValidationError
        If the column is absent, or a partition was named without a split plan.

    Notes
    -----
    **Transcription is not fitting**, so there is no leakage concern in running
    it over any partition. Nothing is learned from the audio.

    **Transcripts arrive in partition order, not dataset order.** When
    ``partition`` names a split, index ``i`` corresponds to the ``i``-th row of
    that partition. Use the split plan's indices to map back.

    Examples
    --------
    Transcribe the test split::

        result = transcribe_from_dataset(
            dataset,
            audio_column="clip",
            partition="test",
            split_plan=split_plan,
        )

    See Also
    --------
    transcribe_audio_values : The underlying call.
    """
    if audio_column not in dataset.columns:
        raise ValidationError(f"audio_column {audio_column!r} not in dataset")
    frame = dataset._ensure_pandas()
    if partition == "all":
        values = frame[audio_column].tolist()
    else:
        if split_plan is None:
            raise ValidationError(
                "partition != 'all' requires a SplitPlan (call session.split first)."
            )
        idx = list(split_plan.indices_for(partition))
        values = frame.iloc[idx][audio_column].tolist()
    result = transcribe_audio_values(
        values,
        backend=backend,
        model_id=model_id,
        sample_rate=sample_rate,
        max_samples=max_samples,
        source_sample_rate=source_sample_rate,
    )
    result.meta["audio_column"] = audio_column
    result.meta["partition"] = partition
    return result


def resolve_audio_paths(values: list[Any]) -> list[str]:
    """Describe audio cells as readable strings for diagnostics.

    An audio column can hold paths or in-memory arrays. Printing it raw gives
    either useful filenames or pages of numbers. This keeps paths intact and
    reduces arrays to a shape summary.

    Parameters
    ----------
    values:
        Audio cells.

    Returns
    -------
    list of str
        One description per cell, in order.

    Notes
    -----
    For diagnostics only — nothing here decodes or validates the audio. A path
    that does not exist passes through unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> resolve_audio_paths(["clip.wav", np.zeros((16000,), dtype=np.float32)])
    ['clip.wav', '<waveform shape=(16000,)>']

    See Also
    --------
    buildml.dl.audio.decode_audio_cell : Actually reading a cell.
    """
    out: list[str] = []
    for value in values:
        if isinstance(value, (str, Path)):
            out.append(str(value))
        else:
            out.append(f"<waveform shape={getattr(value, 'shape', None)}>")
    return out
