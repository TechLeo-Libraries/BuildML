"""Frozen multimodal transforms retain train-only numerical and media contracts."""
import json

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError
from buildml.dl.multimodal import MultimodalLoaderConfig, make_multimodal_loaders


@pytest.fixture(scope='module', params=[False, True])
def original(request):
    pytest.importorskip('torch')
    pytest.importorskip('PIL')
    n = 24
    frame = pd.DataFrame({'x': np.arange(n, dtype=float), 'z': np.arange(n, dtype=float)**2,
        'text': ['train shared' if i < 12 else 'unseen shared' for i in range(n)],
        'image': [np.full((8, 8, 3), .1+i/40, dtype=np.float32) for i in range(n)],
        'audio': [(np.sin(np.arange(32)*.3)+i/10).astype(np.float32) for i in range(n)],
        'y': np.arange(n) % 2})
    s = Session.ingest(frame).set_roles({'x':'feature','z':'feature','text':'feature',
                                        'image':'feature','audio':'feature','y':'target'})
    s.inject_split(train_indices=list(range(12)),validation_indices=list(range(12,18)),test_indices=list(range(18,24)))
    cfg = MultimodalLoaderConfig(batch_size=6, shuffle_train=False, max_len=5,
        normalize=request.param,normalize_images=request.param,normalize_audio=request.param,
        image_size=(8,8),image_channels=3,audio_sample_rate=8000,audio_max_samples=32,
        audio_source_sample_rate=8000)
    bundle = make_multimodal_loaders(s.dataset,s.split_plan,config=cfg,text_column='text',
        numeric_columns=['x','z'],image_column='image',audio_column='audio',task='classification')
    return s, bundle, request.param


@pytest.mark.parametrize('serialized',[False,True])
@pytest.mark.parametrize('runtime_normalize',[False,True])
@pytest.mark.parametrize('changed_train',[False,True])
def test_saved_contract_controls_replay_despite_runtime_settings(original,serialized,runtime_normalize,changed_train):
    s, original_bundle, normalize = original
    contract = original_bundle.multimodal_contract
    frame = s.to_pandas().copy(deep=True)
    if changed_train:
        frame.loc[:11,['x','z']] += 1000
        frame.loc[:11,'text'] = 'replacement neverlearn'
        for i in range(12):
            frame.at[i,'image'] = np.full((8,8,3),.95,dtype=np.float32)
            frame.at[i,'audio'] = np.ones(32,dtype=np.float32)*100
    other = Session.ingest(frame).set_roles({'x':'feature','z':'feature','text':'feature','image':'feature','audio':'feature','y':'target'})
    other.inject_split(train_indices=list(range(12)),validation_indices=list(range(12,18)),test_indices=list(range(18,24)))
    payload = json.loads(json.dumps(contract.to_dict())) if serialized else contract
    cfg = MultimodalLoaderConfig(batch_size=3,shuffle_train=False,max_len=2,
        normalize=runtime_normalize,normalize_images=runtime_normalize,normalize_audio=runtime_normalize,
        image_size=(4,4),image_channels=1,audio_sample_rate=4000,audio_max_samples=16,
        audio_source_sample_rate=4000)
    replay = make_multimodal_loaders(other.dataset,other.split_plan,config=cfg,preprocess=payload)
    assert replay.multimodal_contract.to_dict() == contract.to_dict()
    assert replay.input_layout == ('numeric','tokens','image','audio')
    assert cfg.image_size == (4,4) and cfg.audio_max_samples == 16
    train = s.to_pandas().iloc[:12]
    numeric = frame.iloc[18:24][['x','z']].to_numpy(dtype=np.float32)
    images = np.stack(frame.image.iloc[18:24]).transpose(0,3,1,2)
    audio = np.stack(frame.audio.iloc[18:24])[:,None,:]
    if normalize:
        numeric = (numeric-train[['x','z']].to_numpy().mean(axis=0))/train[['x','z']].to_numpy().std(axis=0)
        train_images = np.stack(train.image).transpose(0,3,1,2)
        images = (images-train_images.mean(axis=(0,2,3),dtype=np.float64)[None,:,None,None])/train_images.std(axis=(0,2,3),dtype=np.float64)[None,:,None,None]
        train_audio = np.stack(train.audio)
        audio = (audio-train_audio.mean(dtype=np.float64))/train_audio.std(dtype=np.float64)
    actual = replay.loaders['test'].dataset.tensors
    np.testing.assert_allclose(actual[0].numpy(),numeric,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(actual[2].numpy(),images,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(actual[3].numpy(),audio,rtol=2e-5,atol=2e-5)
    vocab = contract.vocab
    assert 'unseen' not in vocab['token_to_id']
    expected_tokens = [vocab['unk_id'],vocab['token_to_id']['shared']]+[vocab['pad_id']]*3
    np.testing.assert_array_equal(actual[1].numpy(),np.tile(expected_tokens,(6,1)))
    for original_tensor,restored_tensor in zip(original_bundle.loaders['test'].dataset.tensors,actual,strict=True):
        np.testing.assert_allclose(restored_tensor.numpy(),original_tensor.numpy(),rtol=2e-5,atol=2e-5)



def test_missing_frozen_vocabulary_refuses_instead_of_learning_again(original):
    s, bundle, _ = original
    payload = json.loads(json.dumps(bundle.multimodal_contract.to_dict()))
    payload['vocab'] = {}
    with pytest.raises(ValidationError, match='vocab'):
        make_multimodal_loaders(s.dataset,s.split_plan,preprocess=payload)
