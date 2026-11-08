"""Offline real transformer heads must preserve sentiment-label semantics."""

import math

import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.nlp.sentiment import analyze_sentiment


@pytest.fixture(
    scope="module", params=["positive", "negative", "neutral", "LABEL_0", "custom", "favorable"]
)
def local_head(request, tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    root = tmp_path_factory.mktemp("local-sentiment-" + request.param)
    vocabulary = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "good": 4,
        "bad": 5,
        "service": 6,
        "delivery": 7,
    }
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=32,
    )
    tokenizer.save_pretrained(root)
    labels = [request.param, "negative", "neutral"]
    if request.param == "negative":
        labels = ["negative", "positive", "neutral"]
    elif request.param == "neutral":
        labels = ["neutral", "positive", "negative"]
    config = transformers.BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=12,
        max_position_embeddings=32,
        num_labels=3,
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
    )
    model = transformers.BertForSequenceClassification(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias.copy_(torch.tensor([2.0, 0.5, -1.0]))
    model.save_pretrained(root)
    frame = pd.DataFrame(
        {"text": ["good service", "bad delivery", "service delivery", "good delivery"] * 5},
        index=[4, 4, 2, 9] * 5,
    )
    s = (
        Session.ingest(frame)
        .set_roles({"text": "feature"})
        .split(test_size=0.2, validation_size=0.2, random_state=7)
    )
    return s, str(root), request.param


@pytest.mark.parametrize("partition", ["validation", "test", "all"])
@pytest.mark.parametrize("threshold", [0.05, 0.9])
def test_local_transformer_scores_follow_explicit_head_label_and_softmax(
    local_head, partition, threshold
):
    s, path, label = local_head
    original = s.to_pandas().copy(deep=True)
    if label not in {"positive", "negative", "neutral"}:
        with pytest.raises(ValidationError, match="(?i)label"):
            analyze_sentiment(
                s.dataset,
                s.split_plan,
                backend="transformer",
                transformer_model=path,
                partition=partition,
                text_column="text",
                threshold=threshold,
            )
    else:
        result = analyze_sentiment(
            s.dataset,
            s.split_plan,
            backend="transformer",
            transformer_model=path,
            partition=partition,
            text_column="text",
            threshold=threshold,
        )
        confidence = math.exp(2) / (math.exp(2) + math.exp(0.5) + math.exp(-1))
        score = confidence if label == "positive" else (-confidence if label == "negative" else 0.0)
        expected = (
            "positive" if score >= threshold else ("negative" if score <= -threshold else "neutral")
        )
        assert set(result.labels) == {expected}
        assert result.scores == pytest.approx([score] * len(result.labels), abs=1e-7)
        assert len(result.labels) == (
            len(original) if partition == "all" else len(s.partition(partition))
        )
        assert any("outside this Session" in note for note in result.disclosures)
    pd.testing.assert_frame_equal(s.to_pandas(), original)
