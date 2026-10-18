"""Transient probe failures do not permanently disable installed backends."""
import subprocess

import pytest

from buildml.dl import extras


@pytest.fixture(autouse=True)
def clean_cache():
    extras.clear_subprocess_import_cache()
    yield
    extras.clear_subprocess_import_cache()


@pytest.mark.parametrize('failure', [subprocess.TimeoutExpired('probe', 12), OSError('busy')])
@pytest.mark.parametrize('module', ['ngboost', 'imodels'])
def test_transient_probe_failure_can_recover_after_bounded_cooldown(monkeypatch, failure, module):
    clock = [100.]
    monkeypatch.setattr(extras, 'monotonic', lambda:clock[0])
    attempts = []
    def run(*args, **kwargs):
        attempts.append(kwargs['timeout'])
        if len(attempts) == 1:
            raise failure
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(extras.subprocess, 'run', run)
    assert not extras._subprocess_import_ok(module)
    clock[0] = 159.
    assert not extras._subprocess_import_ok(module)
    assert attempts == [90.]
    clock[0] = 160.
    assert extras._subprocess_import_ok(module)
    assert attempts == [90., 90.]
    clock[0] = 1000.
    assert extras._subprocess_import_ok(module)
    assert len(attempts) == 2


def test_completed_import_failure_stays_cached_until_explicit_clear(monkeypatch):
    attempts = []
    def run(*args, **kwargs):
        attempts.append(kwargs['timeout'])
        return subprocess.CompletedProcess(args, 1)
    monkeypatch.setattr(extras.subprocess, 'run', run)
    assert not extras._subprocess_import_ok('missing')
    assert not extras._subprocess_import_ok('missing')
    assert attempts == [12.]
    extras.clear_subprocess_import_cache()
    assert not extras._subprocess_import_ok('missing', timeout=2.5)
    assert attempts == [12., 2.5]


def test_cache_clear_resets_transient_deadlines(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('probe', 12)
    monkeypatch.setattr(extras.subprocess, 'run', timeout)
    assert not extras._subprocess_import_ok('ripser')
    assert extras._SUBPROCESS_IMPORT_TRANSIENT_UNTIL
    extras.clear_subprocess_import_cache()
    assert not extras._SUBPROCESS_IMPORT_TRANSIENT_UNTIL
    assert not extras._SUBPROCESS_IMPORT_CACHE
