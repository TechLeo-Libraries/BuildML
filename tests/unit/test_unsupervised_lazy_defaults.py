"""Optional defaults resolve at use, without freezing a transient probe result."""
import pytest

from buildml.unsupervised import catalog


def test_explicit_density_choice_does_not_probe_another_backend(monkeypatch):
    def unexpected():
        pytest.fail('Explicit DBSCAN selection probed HDBSCAN')
    monkeypatch.setattr(catalog, 'hdbscan_available', unexpected)
    assert catalog.resolve_density_method('dbscan') == 'dbscan'


@pytest.mark.parametrize('available,expected', [(False,'dbscan'), (True,'hdbscan')])
def test_default_density_uses_current_availability(monkeypatch, available, expected):
    monkeypatch.setattr(catalog, 'hdbscan_available', lambda:available)
    assert catalog.resolve_density_method() == expected
    assert catalog.DEFAULT_DENSITY_METHOD == expected


def test_lazy_defaults_preserve_direct_and_wildcard_imports(monkeypatch):
    monkeypatch.setattr(catalog, 'hdbscan_available', lambda:True)
    monkeypatch.setattr(catalog, 'umap_available', lambda:False)
    namespace = {}
    exec('from buildml.unsupervised.catalog import *', namespace)
    assert namespace['DEFAULT_DENSITY_METHOD'] == 'hdbscan'
    assert namespace['DEFAULT_REDUCE_VIZ'] == 'pca'
    assert 'DEFAULT_REDUCE_VIZ' in dir(catalog)
    monkeypatch.setattr(catalog, 'umap_available', lambda:True)
    assert catalog.DEFAULT_REDUCE_VIZ == 'umap'
    with pytest.raises(AttributeError):
        _ = catalog.unknown_default
