"""Guard public surfaces missed by the earlier copy scan."""

from pathlib import Path

from scripts.lint_user_copy import ROOT, iter_targets, lint_paths


def test_copy_scan_includes_release_docs_and_dashboard_assets() -> None:
    paths = {path.relative_to(ROOT).as_posix() for path in iter_targets()}
    assert {
        "CHANGELOG.md",
        "docs/pypi-2x-publish.md",
        "buildml/dashboard/static/js/app.js",
        "buildml/explain/generated/operation_index.json",
        "proofs/clickstream-online/script.py",
        "examples/leakage_cv_recipe.py",
    } <= paths
    assert not any("/_build/" in path for path in paths)


def test_copy_scan_checks_concatenated_runtime_strings(tmp_path: Path) -> None:
    source = tmp_path / "message.py"
    source.write_text(
        'message = ("keep this method as "\n"a thin delegate")\n',
        encoding="utf-8",
    )
    hits = lint_paths([source])
    assert any(hit.rule == "internal-or-dismissive-copy" for hit in hits)


def test_copy_scan_checks_non_python_public_copy(tmp_path: Path) -> None:
    for suffix in (".md", ".js", ".html", ".json"):
        source = tmp_path / f"message{suffix}"
        source.write_text('"Yank them on PyPI"', encoding="utf-8")
        assert any(
            hit.rule == "internal-or-dismissive-copy" for hit in lint_paths([source])
        )
