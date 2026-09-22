"""Keep authored helper output and newly added files in copy-lint coverage."""
from pathlib import Path
from types import SimpleNamespace

from scripts import lint_user_copy


def test_inventory_includes_scripts_benchmarks_and_untracked_sources(tmp_path, monkeypatch):
    names = (
        "scripts/new_helper.py",
        "benchmarks/new_benchmark.py",
        "buildml/new_module.py",
        "scripts/lint_user_copy.py",
        "tests/unit/negative_copy_fixture.py",
    )
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('message = "estimator zoo"\n', encoding="utf-8")

    def git_inventory(command, **kwargs):
        assert "--others" in command
        assert "--exclude-standard" in command
        return SimpleNamespace(returncode=0, stdout="\0".join(names).encode())

    monkeypatch.setattr(lint_user_copy, "ROOT", tmp_path)
    monkeypatch.setattr(lint_user_copy.subprocess, "run", git_inventory)
    targets = list(lint_user_copy.iter_targets())
    assert {path.relative_to(tmp_path).as_posix() for path in targets} == set(names[:3])
    violations = lint_user_copy.lint_paths(targets)
    assert {item.path for item in violations if item.rule == "internal-or-dismissive-copy"} == set(names[:3])


def test_split_string_in_script_is_checked(tmp_path: Path):
    path = tmp_path / "helper.py"
    path.write_text('message = ("keep this method as a " "thin delegate")\n', encoding="utf-8")
    assert any(
        item.rule == "internal-or-dismissive-copy"
        for item in lint_user_copy.lint_paths([path])
    )
