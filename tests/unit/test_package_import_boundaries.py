"""Capability workers must not construct the complete Session import graph."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["buildml", "buildml.cbr.extras", "buildml.tda.extras",
                                    "buildml.probabilistic.extras"])
def test_probe_imports_do_not_load_session_or_scientific_stacks(module):
    code = '''
import importlib
import importlib.abc
import sys

class RefuseHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'buildml.session' or fullname.split('.')[0] in {'torch', 'pandas', 'sklearn'}:
            raise AssertionError('Unexpected eager import: '+fullname)

sys.meta_path.insert(0, RefuseHeavyImports())
importlib.import_module(MODULE)
import buildml
assert buildml.__version__
assert 'Session' in dir(buildml)
assert 'buildml.session' not in sys.modules
try:
    buildml.unknown_public_attribute
except AttributeError:
    pass
else:
    raise AssertionError('Unknown attribute was accepted')
'''.replace('MODULE', repr(module))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stderr


def test_lazy_session_is_the_original_public_class():
    import buildml
    from buildml import Session
    from buildml.session import Session as DirectSession

    assert Session is DirectSession is buildml.Session
    assert vars(buildml)['Session'] is Session
    namespace = {}
    exec('from buildml import *', namespace)
    assert namespace['Session'] is Session


def test_session_import_does_not_probe_optional_clustering_backends():
    code = '''
from buildml.unsupervised import extras
def unexpected():
    raise AssertionError('Session startup probed an unrelated optional backend')
extras.hdbscan_available = unexpected
extras.umap_available = unexpected
from buildml import Session
assert Session.__name__ == 'Session'
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
