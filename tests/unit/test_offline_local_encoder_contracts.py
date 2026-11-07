"""Real local encoders preserve pooling, width and persistence contracts."""
import inspect

import joblib
import numpy as np
import pytest


@pytest.fixture(scope='module')
def tiny_bert(tmp_path_factory):
    transformers = pytest.importorskip('transformers')
    torch = pytest.importorskip('torch')
    destination = tmp_path_factory.mktemp('local-bert')
    vocabulary = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'red', 'blue', 'apple', 'berry', 'fruit', 'engine', 'valve']
    vocab = destination / 'vocab.txt'
    vocab.write_text('\n'.join(vocabulary), encoding='utf-8')
    parameters = inspect.signature(transformers.BertTokenizerFast.__init__).parameters
    tokenizer = transformers.BertTokenizerFast(**{
        "vocab" if "vocab" in parameters else "vocab_file": str(vocab)
    })
    assert tokenizer.convert_tokens_to_ids("red") != tokenizer.unk_token_id
    assert tokenizer.convert_tokens_to_ids("engine") != tokenizer.unk_token_id
    tokenizer.save_pretrained(destination)
    config = transformers.BertConfig(vocab_size=len(vocabulary), hidden_size=8,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=16,
        max_position_embeddings=32, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(61)
        transformers.BertModel(config).save_pretrained(destination)
    return destination


@pytest.mark.parametrize('normalized', [False, True])
def test_frozen_transformer_pooling_matches_masked_token_mean(tiny_bert, normalized, tmp_path):
    torch = pytest.importorskip('torch')
    from buildml.nlp.adapters.transformer_encoder import TransformerEncoderVectorizer
    documents = ['red apple', 'blue berry fruit', None]
    vectorizer = TransformerEncoderVectorizer(str(tiny_bert), batch_size=2,
        max_seq_tokens=16, normalize_embeddings=normalized)
    actual = vectorizer.fit_transform(documents)
    tokens = vectorizer._tokenizer(['red apple', 'blue berry fruit', ''],
        padding=True, return_tensors='pt')
    with torch.no_grad():
        hidden = vectorizer._model(**tokens).last_hidden_state
        mask = tokens['attention_mask'].unsqueeze(-1)
        expected = (hidden * mask).sum(1) / mask.sum(1)
        if normalized:
            expected = expected / expected.norm(dim=1, keepdim=True)
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-6)
    assert vectorizer.transform([]).shape == (0, 8)
    assert vectorizer.get_feature_names_out().tolist() == [f'pooled_{i}' for i in range(8)]
    path = tmp_path / 'local-vectorizer.joblib'
    joblib.dump(vectorizer, path)
    restored = joblib.load(path)  # Benign fixture written above.
    assert restored._model is None and restored._tokenizer is None
    np.testing.assert_allclose(restored.transform(documents), actual, atol=1e-6)


@pytest.fixture(scope='module')
def local_sentence_encoder(tiny_bert, tmp_path_factory):
    st = pytest.importorskip('sentence_transformers')
    destination = tmp_path_factory.mktemp('local-sentence-encoder')
    try:
        from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    except ModuleNotFoundError:
        from sentence_transformers.models import Pooling, Transformer
    encoder = Transformer(str(tiny_bert), max_seq_length=16)
    pooling = Pooling(8, pooling_mode="mean")
    st.SentenceTransformer(modules=[encoder, pooling], device='cpu').save(str(destination))
    return destination


def test_sentence_vectors_and_rag_vectors_preserve_row_order(local_sentence_encoder):
    from buildml.nlp.adapters.sentence_embedding import SentenceEmbeddingVectorizer
    from buildml.rag.embed import SentenceTransformerEmbedder
    documents = ['red apple', 'blue berry fruit', 'engine valve']
    vectorizer = SentenceEmbeddingVectorizer(str(local_sentence_encoder), device='cpu')
    vectors = vectorizer.fit_transform(documents)
    rag = SentenceTransformerEmbedder(str(local_sentence_encoder), device='cpu')
    np.testing.assert_allclose(rag.encode(documents), vectors, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(vectorizer.transform(documents[::-1]), vectors[::-1], atol=1e-6)
    assert rag.encode([]).shape == (0, 8)
    assert vectorizer.transform([]).shape == (0, 8)


@pytest.mark.parametrize('width', [4, 8, 12])
def test_text_ssl_width_truncates_or_pads_real_local_embeddings(local_sentence_encoder, width):
    st = pytest.importorskip('sentence_transformers')
    from buildml.selfsupervised.torch.text import HFTextSSLEncoder
    documents = ['red apple', 'blue berry fruit', 'engine valve']
    native = st.SentenceTransformer(str(local_sentence_encoder), device='cpu').encode(documents)
    encoder = HFTextSSLEncoder(model_name=str(local_sentence_encoder), epochs=0, latent_dim=width)
    result = encoder.fit(documents).transform(documents)
    expected = np.zeros((3, width))
    expected[:, :min(8, width)] = native[:, :min(8, width)]
    np.testing.assert_allclose(result, expected, atol=1e-6)
    state = encoder.state_dict()
    assert state['latent_dim'] == width
    assert state['model_state']



def test_text_ssl_finetunes_real_local_encoder_weights(local_sentence_encoder):
    import torch
    from sentence_transformers import SentenceTransformer

    from buildml.selfsupervised.torch.text import HFTextSSLEncoder

    documents = ["red apple", "blue berry", "engine valve", "fruit berry", "blue apple", "red valve"]
    initial = SentenceTransformer(str(local_sentence_encoder), device="cpu").state_dict()
    encoder = HFTextSSLEncoder(model_name=str(local_sentence_encoder), epochs=1,
                              batch_size=3, learning_rate=.01, random_state=7)
    encoder.fit(documents)
    final = encoder.state_dict()["model_state"]
    assert encoder._model.device.type == "cpu"
    assert any(not torch.equal(initial[key], value.cpu()) for key, value in final.items())
    assert np.isfinite(encoder.transform(documents)).all()
    assert encoder.pretext_loss_ is None
