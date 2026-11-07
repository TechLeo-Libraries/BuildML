"""Missing, unusable and entirely skipped backends cannot become support claims."""
import subprocess

from scripts import backend_support_report as evidence


def test_missing_dependency_does_not_execute_tests(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence.importlib.util, "find_spec", lambda name: None)
    result = evidence.run_group("engines", tmp_path)
    assert result["status"] == "missing_dependency"
    assert result["missing"] == ["polars", "duckdb"]


def test_broken_transitive_dependency_is_not_supported(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence.importlib.util, "find_spec", lambda name: object())
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "missing transitive dependency")

    monkeypatch.setattr(evidence.subprocess, "run", run)
    result = evidence.run_group("engines", tmp_path)
    assert result["status"] == "dependency_import_failed"
    assert len(calls) == 1


def test_all_skipped_group_is_not_supported(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence.importlib.util, "find_spec", lambda name: object())

    def run(command, **kwargs):
        if "--junitxml" in command:
            (tmp_path / "engines.xml").write_text(
                '<testsuites><testsuite><testcase name="real_backend">'
                '<skipped message="backend unavailable"/></testcase></testsuite></testsuites>'
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(evidence.subprocess, "run", run)
    result = evidence.run_group("engines", tmp_path)
    assert result["status"] == "no_executed_tests"
    assert result["passed"] == 0
    assert result["skipped"][0]["reason"] == "backend unavailable"
