"""Real offline CTC pipelines with analytically known logits, not ASR quality claims."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.dl.speech import transcribe_audio_values, transcribe_from_dataset


@pytest.fixture(scope='module')
def ctc_models(tmp_path_factory):
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import (
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )

    previous = {key: os.environ.get(key) for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')}
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    root = tmp_path_factory.mktemp('known-logit-ctc')
    vocabulary = {'<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, '|': 4, 'A': 5, 'B': 6}
    vocab_path = root / 'vocab.json'
    vocab_path.write_text(json.dumps(vocabulary), encoding='utf-8')
    tokenizer = Wav2Vec2CTCTokenizer(str(vocab_path), pad_token='<pad>', unk_token='<unk>',
                                    bos_token='<s>', eos_token='</s>', word_delimiter_token='|')
    processor = Wav2Vec2Processor(feature_extractor=Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0., do_normalize=True,
        return_attention_mask=True), tokenizer=tokenizer)
    paths = {}
    try:
        for name, winner in (('letter', 5), ('blank', 0)):
            with torch.random.fork_rng():
                torch.manual_seed(7)
                model = Wav2Vec2ForCTC(Wav2Vec2Config(
                    vocab_size=7, hidden_size=8, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=16,
                    conv_dim=(4, 4), conv_stride=(2, 2), conv_kernel=(5, 3),
                    num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
                    mask_time_prob=0., apply_spec_augment=False, pad_token_id=0,
                    bos_token_id=1, eos_token_id=2))
            with torch.no_grad():
                model.lm_head.weight.zero_()
                model.lm_head.bias.fill_(-5.)
                model.lm_head.bias[winner] = 5.
            model.eval()
            # Constant logits imply CTC-collapse(A,A,...)='A', or blank -> ''.
            with torch.no_grad():
                logits = model(torch.linspace(-1., 1., 1600).reshape(1, -1)).logits
            assert torch.all(logits.argmax(-1) == winner)
            path = root / name
            model.save_pretrained(path)
            processor.save_pretrained(path)
            paths[name] = str(path)
        yield paths
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def wave(length=1600):
    return np.sin(np.arange(length, dtype=np.float32) * .1)


@pytest.mark.parametrize('backend', ['transformers', 'auto', None])
@pytest.mark.parametrize('model,expected', [('letter', 'A'), ('blank', '[empty]')])
def test_real_pipeline_decodes_known_logits_and_preserves_audio(ctc_models, backend, model, expected):
    clips = [wave(500), wave(1600), wave(2300)]
    before = [clip.copy() for clip in clips]
    result = transcribe_audio_values(clips, backend=backend, model_id=ctc_models[model],
                                     sample_rate=16000, max_samples=1600)
    assert result.texts == [expected] * len(clips)
    assert result.n_rows == len(clips) and result.backend == 'transformers'
    assert result.model_id == ctc_models[model]
    assert result.meta['stub'] is False
    assert result.warnings == []
    assert any('Hugging Face' in item for item in result.disclosures)
    if backend in ('auto', None):
        assert any('resolved from auto' in item for item in result.disclosures)
    for clip, original in zip(clips, before, strict=True):
        np.testing.assert_array_equal(clip, original)


@pytest.mark.parametrize('partition', ['train', 'validation', 'test', 'all'])
def test_dataset_partition_consumers_use_actual_pipeline(ctc_models, partition):
    frame = pd.DataFrame({'clip': [wave(400 + 100*i) for i in range(10)], 'y': [0, 1]*5},
                         index=[f'row-{i}' for i in range(10)])
    s = (Session.ingest(frame).set_roles({'clip': 'feature', 'y': 'target'})
         .split(test_size=.2, validation_size=.2, random_state=3))
    result = transcribe_from_dataset(s.dataset, audio_column='clip', split_plan=s.split_plan,
        partition=partition, backend='transformers', model_id=ctc_models['letter'], max_samples=1600)
    expected_count = len(frame) if partition == 'all' else len(s.split_plan.indices_for(partition))
    assert result.texts == ['A'] * expected_count
    assert result.n_rows == expected_count and result.meta['partition'] == partition
    assert result.meta['audio_column'] == 'clip'
    pd.testing.assert_frame_equal(s.to_pandas(), frame)


def test_real_inference_failure_is_disclosed_per_clip(ctc_models):
    # One sample cannot pass a real convolutional feature extractor.
    result = transcribe_audio_values([wave(), wave()], backend='transformers',
        model_id=ctc_models['letter'], max_samples=1)
    assert result.texts == ['[asr-error]', '[asr-error]']
    assert result.n_rows == 2 and len(result.warnings) == 2
    assert all('ASR failed for one cell' in message for message in result.warnings)
    recovered = transcribe_audio_values([wave()], backend='transformers',
        model_id=ctc_models['letter'], max_samples=1600)
    assert recovered.texts == ['A'] and recovered.warnings == []
