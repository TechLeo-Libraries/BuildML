"""Broken native ANN imports must not terminate capability inspection."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["hnswlib", "faiss"])
@pytest.mark.parametrize("crash", [False, True])
def test_ann_native_probe_contains_child_failure(tmp_path, module, crash):
    # A disposable module reproduces an uncatchable extension import failure.
    code = ""
    if crash:
        code = (
            "import os, sys, signal\n"
            "if sys.platform != 'win32':\n"
            " import resource\n"
            " resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
            " os.kill(os.getpid(), signal.SIGILL)\n"
            "os._exit(132)\n"
        )
    (tmp_path / f"{module}.py").write_text(code)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + os.getcwd()
    result = subprocess.run(
        [sys.executable, "-c",
         f"from buildml.cbr.extras import {module}_available; "
         f"print('parent-alive', {module}_available())"],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"parent-alive {not crash}"
