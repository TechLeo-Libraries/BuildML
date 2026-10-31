"""Import retries must preserve diagnostics and never hide backend contract failures."""
import subprocess

import pytest

from scripts import backend_support_report as reporter


@pytest.mark.parametrize("test_exit", [0, 1])
def test_one_import_timeout_is_recorded_and_real_test_status_is_preserved(test_exit, monkeypatch, tmp_path):
    monkeypatch.setattr(reporter.importlib.util, "find_spec", lambda name: object())
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, 30, output=b"import started: ripser\n")
        if len(calls) == 2:
            return subprocess.CompletedProcess(command, 0, "import completed: ripser\n", "")
        assert "pytest" in command
        xml = command[command.index("--junitxml") + 1]
        body = '<failure message="numerical mismatch" />' if test_exit else ''
        from pathlib import Path
        Path(xml).write_text(f'<testsuites><testsuite><testcase name="backend">{body}</testcase></testsuite></testsuites>')
        return subprocess.CompletedProcess(command, test_exit, "contract result", "")

    monkeypatch.setattr(reporter.subprocess, "run", run)
    result = reporter.run_group("tda", tmp_path)
    assert calls == [30, 90, 300]
    assert result["status"] == ("passed" if test_exit == 0 else "failed")
    assert result["dependency_import_attempts"][0]["status"] == "timeout"
    assert result["dependency_import_attempts"][1]["status"] == "passed"
    log = (tmp_path / 'tda-dependency-import.log').read_text()
    assert "timed out" in log and "import started: ripser" in log


def test_two_import_timeouts_fail_without_running_tests(monkeypatch, tmp_path):
    monkeypatch.setattr(reporter.importlib.util, "find_spec", lambda name: object())
    calls = []
    def run(command, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=b"import started: persim\n")
    monkeypatch.setattr(reporter.subprocess, "run", run)
    result = reporter.run_group("tda", tmp_path)
    assert calls == [30, 90]
    assert result["status"] == "dependency_import_timeout"
    assert len(result["dependency_import_attempts"]) == 2


def test_native_import_crash_is_not_retried_or_reported_as_support(monkeypatch, tmp_path):
    monkeypatch.setattr(reporter.importlib.util, "find_spec", lambda name: object())
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, -11, "import started: ripser", "native crash")
    monkeypatch.setattr(reporter.subprocess, "run", run)
    result = reporter.run_group("tda", tmp_path)
    assert len(calls) == 1
    assert result["status"] == "dependency_import_failed"
    assert result["returncode"] == -11
