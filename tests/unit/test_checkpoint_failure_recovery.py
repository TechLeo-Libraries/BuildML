"""Checkpoint faults retain canonical data and never advertise partial sidecars."""
import json

import numpy as np
import pandas as pd
import pytest

from buildml import Session
from buildml.core.errors import ValidationError


def _session(engine='pandas'):
    return (Session.ingest(pd.DataFrame({'x': np.arange(24), 'y': [0, 1] * 12}), engine=engine)
            .set_roles({'y': 'target'}).split(test_size=.25, random_state=3))


@pytest.mark.parametrize('value', [True, 2.5, float('nan'), float('inf'), '3', 0, -1])
def test_bad_partition_size_creates_no_checkpoint(value, tmp_path):
    session = _session()
    path = tmp_path / 'refused'
    with pytest.raises(ValidationError, match='sidecar_partition_rows'):
        session.checkpoint_save(path, sidecar_partition_rows=value)
    assert not path.exists()


@pytest.mark.parametrize('layout', ['single', 'partitioned'])
def test_partial_sidecar_write_is_cleaned_and_canonical_restore_works(layout, tmp_path, monkeypatch):
    pytest.importorskip('polars')
    from buildml.data.engines.polars_engine import PolarsEngine

    session = _session('polars')
    original = PolarsEngine.write_parquet
    calls = []

    def fail_after_write(self, table, path, **kwargs):
        original(self, table, path, **kwargs)
        calls.append(path)
        if layout == 'single' or len(calls) == 2:
            raise OSError('Injected sidecar storage failure')

    with monkeypatch.context() as patch:
        patch.setattr(PolarsEngine, 'write_parquet', fail_after_write)
        path = session.checkpoint_save(tmp_path / layout, sidecar_layout=layout,
                                       sidecar_partition_rows=5)
    metadata = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
    assert 'native_sidecar' not in metadata
    assert (path / 'data/frame.parquet').exists()
    assert not (path / 'data/native_sidecar.parquet').exists()
    assert not list((path / 'data/native_sidecar').glob('*.parquet'))
    assert len(calls) == (1 if layout == 'single' else 2)
    restored = Session.checkpoint_load(path, trusted=True)
    pd.testing.assert_frame_equal(restored.to_pandas(), session.to_pandas())
    assert restored.dataset.roles == session.dataset.roles
    assert restored.split_plan.train_indices == session.split_plan.train_indices
    assert restored.split_plan.test_indices == session.split_plan.test_indices


def test_native_reader_failure_falls_back_to_canonical_data(tmp_path, monkeypatch):
    pytest.importorskip('polars')
    from buildml.data.engines.polars_engine import PolarsEngine

    session = _session('polars')
    path = session.checkpoint_save(tmp_path / 'fallback')

    def broken_reader(*args, **kwargs):
        raise OSError('Injected native reader failure')

    monkeypatch.setattr(PolarsEngine, 'from_parquet', broken_reader)
    restored = Session.checkpoint_load(path, trusted=True)
    pd.testing.assert_frame_equal(restored.to_pandas(), session.to_pandas())
    assert restored.split_plan.train_indices == session.split_plan.train_indices
    assert restored.dataset.has_native
    assert any('eager rebuild' in message.lower() for message in restored.reattach_result.messages)


def test_switching_sidecar_layout_removes_stale_parquet_parts(tmp_path):
    pytest.importorskip('polars')
    session = _session('polars')
    path = tmp_path / 'reused'
    session.checkpoint_save(path, sidecar_layout='partitioned', sidecar_partition_rows=np.int64(5))
    assert len(list((path / 'data/native_sidecar').glob('*.parquet'))) == 5
    session.checkpoint_save(path, sidecar_layout='single')
    assert (path / 'data/native_sidecar.parquet').exists()
    assert not (path / 'data/native_sidecar').exists()
    session.checkpoint_save(path, sidecar_layout='partitioned', sidecar_partition_rows=8)
    assert not (path / 'data/native_sidecar.parquet').exists()
    assert len(list((path / 'data/native_sidecar').glob('*.parquet'))) == 3
    restored = Session.checkpoint_load(path, trusted=True)
    pd.testing.assert_frame_equal(restored.to_pandas(), session.to_pandas())
