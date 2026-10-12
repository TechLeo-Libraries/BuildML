"""Distributed sample ordering preserves the caller loader contract."""
import pytest

from buildml.dl.ddp import _shard_loader


@pytest.mark.parametrize("shuffle", [False, True])
def test_rank_sharding_preserves_requested_shuffle(shuffle):
    torch = pytest.importorskip("torch")
    dataset = torch.utils.data.TensorDataset(torch.arange(16).reshape(8, 2))
    original = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=shuffle)
    shards = [_shard_loader(original, rank=rank, world_size=2, seed=11) for rank in range(2)]
    assert all(loader.sampler.shuffle is shuffle for loader in shards)
    indices = [list(loader.sampler) for loader in shards]
    assert sorted(indices[0] + indices[1]) == list(range(8))
    if not shuffle:
        assert indices == [[0, 2, 4, 6], [1, 3, 5, 7]]
    repeat = _shard_loader(original, rank=0, world_size=2, seed=11)
    assert list(repeat.sampler) == indices[0]
