"""Agent text loaders obey resource counts and match independent tensor math."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry
from buildml.core.errors import ValidationError
from buildml.dl.text import fit_vocab


def _session():
    frame = pd.DataFrame({"text": [f"common word{i} " + ("apple red" if i % 2 else "engine blue") for i in range(60)],
                          "y": [i % 2 for i in range(60)]})
    return Session.ingest(frame).set_roles({"text": "feature", "y": "target"}).split(
        test_size=0.2, validation_size=0.2, random_state=9
    ).ai.configure(provider="mock")


def _call(session, name, args, confirmed=True):
    registry = build_default_registry()
    proposal = propose_tool_execution(name, args, registry)
    return execute_tool(session, proposal, confirmed=confirmed, registry=registry)


@pytest.mark.parametrize("name", ["max_len", "max_vocab", "min_freq"])
@pytest.mark.parametrize("value", [True, np.bool_(True), 2.5, "3", float("nan"), float("inf"), 0, -1])
def test_vocab_controls_refuse_lossy_or_nonfinite_counts(name, value):
    with pytest.raises(ValidationError, match=name):
        fit_vocab(["common apple", "common engine"], **{name: value})


@pytest.mark.parametrize("name", ["max_len", "batch_size"])
@pytest.mark.parametrize("value", [True, 2.5, "3", float("nan"), 0, -1])
def test_agent_loader_rejects_bad_counts_without_attaching_state(name, value):
    if isinstance(value, int) and not isinstance(value, bool) and value <= 0:
        pytest.importorskip("torch", reason="integer range validation dispatches to Torch loaders")
    session = _session()
    outcome = _call(session, "make_text_torch_loaders", {name: value, "text_column": "text"})
    assert not outcome.executed
    assert name in outcome.error
    assert session._torch_loaders is None


@pytest.mark.parametrize("name", ["batch_size", "max_len", "max_vocab", "min_freq"])
@pytest.mark.parametrize("value", [True, np.bool_(True), 2.5, "3", float("nan"), float("inf"), 0, -1])
def test_direct_loader_refuses_invalid_counts_before_attaching_state(name, value):
    pytest.importorskip("torch", reason="public loader validates after its optional backend preflight")
    session = _session()
    with pytest.raises(ValidationError, match=name):
        session.make_text_torch_loaders(text_column="text", **{name: value})
    assert session._torch_loaders is None


def test_minimum_vocabulary_and_numpy_integer_counts():
    vocab = fit_vocab(["apple engine"], max_vocab=np.int64(2), min_freq=np.int64(1), max_len=np.int64(3))
    assert vocab.id_to_token == ("<pad>", "<unk>")
    pytest.importorskip("torch", reason="constructing a real loader requires Torch")
    session = _session()
    bundle = session.make_text_torch_loaders(text_column="text", batch_size=np.int64(4), max_len=np.int64(3))
    assert bundle.loaders["train"].batch_size == 4


@pytest.mark.parametrize("max_len,batch_size", [(1, 4), (3, 7), (6, 9)])
def test_agent_loader_training_and_metrics_match_train_only_vocabulary_and_numpy(max_len, batch_size):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    session = _session()
    original = session.dataset.frame.copy(deep=True)
    args = {"text_column": "text", "batch_size": batch_size, "max_len": max_len}
    denied = _call(session, "make_text_torch_loaders", args, confirmed=False)
    assert not denied.executed and session._torch_loaders is None
    outcome = _call(session, "make_text_torch_loaders", args)
    assert outcome.executed, outcome.error
    bundle = session._torch_loaders
    train = session.partition("train")
    counts = Counter(word for text in train.text for word in text.lower().split())
    vocabulary = ["<pad>", "<unk>"] + sorted(counts, key=lambda word: (-counts[word], word))
    assert tuple(vocabulary) == bundle.text_vocab.id_to_token
    ids = {word: i for i, word in enumerate(vocabulary)}
    for partition in ("train", "validation", "test"):
        part = session.partition(partition)
        expected = np.zeros((len(part), max_len), dtype=np.int64)
        for i, text in enumerate(part.text):
            words = text.lower().split()[:max_len]
            expected[i, :len(words)] = [ids.get(word, 1) for word in words]
        tensor, targets = bundle.loaders[partition].dataset.tensors
        np.testing.assert_array_equal(tensor.numpy(), expected)
        np.testing.assert_array_equal(targets.numpy(), part.y.to_numpy())
        assert bundle.loaders[partition].batch_size == batch_size
    torch.manual_seed(23)
    fitted = _call(session, "fit_torch", {"epochs": 2, "learning_rate": 0.01, "device": "cpu"})
    assert fitted.executed, fitted.error
    model = session.dl_train_result.module.eval()
    tokens, truth = bundle.loaders["test"].dataset.tensors
    state = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    mask = (tokens.numpy() != 0)[..., None]
    pooled = (state["embedding.weight"][tokens.numpy()] * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
    hidden = np.maximum(pooled @ state["head.0.weight"].T + state["head.0.bias"], 0)
    logits = hidden @ state["head.3.weight"].T + state["head.3.bias"]
    with torch.no_grad():
        np.testing.assert_allclose(model(tokens).numpy(), logits, rtol=2e-5, atol=2e-6)
    scored = _call(session, "evaluate_torch", {"partition": "test"})
    assert scored.executed, scored.error
    expected_accuracy = np.mean(logits.argmax(axis=1) == truth.numpy())
    assert scored.result.metrics["accuracy"] == pytest.approx(expected_accuracy)
    pd.testing.assert_frame_equal(session.dataset.frame, original)
