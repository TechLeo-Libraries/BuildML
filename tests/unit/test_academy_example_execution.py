"""Dashboard rendering must preserve complete runnable teaching examples."""

import pytest

from buildml.dashboard.academy_curriculum import all_lessons, build_academy_context
from buildml.dashboard.academy_curriculum.note_binder import _example_code
from buildml.explain.concepts import CONCEPT_NOTES
from scripts.check_documentation_examples import Example, execute, static_check


@pytest.mark.parametrize("task", ["classification", "regression", None])
def test_all_rendered_examples_are_standalone(task):
    context = build_academy_context({})
    if task:
        context["target"] = {"name": "uploaded_target", "task": task}
    for lesson in all_lessons():
        source = lesson.example_code(context)
        assert "your_data.csv" not in source, lesson.slug
        result = static_check(source)
        assert result["status"] == "static-pass", (lesson.slug, result)


@pytest.mark.parametrize("slug", [
    "missing-data", "missingness-mechanisms", "duplicate-records", "interaction-effects",
])
def test_demonstration_data_exercises_the_lesson(slug, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lesson = next(item for item in all_lessons() if item.slug == slug)
    namespace = {"__name__": "__main__"}
    exec(compile(lesson.example_code(build_academy_context({})), slug, "exec"), namespace)
    frame = namespace["frame"]
    if slug.startswith("missing"):
        assert 0 < frame["measurement"].isna().sum() < len(frame)
        if slug == "missingness-mechanisms":
            assert set(frame["measurement__was_missing"]) == {0, 1}
    elif slug == "duplicate-records":
        assert len(frame) == 120
        assert frame["entity_id"].duplicated().any()
        assert not frame.duplicated(["entity_id", "timestamp"]).any()
    else:
        assert namespace["a"] != namespace["b"]
        assert frame["interaction_ab"].equals(frame["measurement"] * frame["amount"])


@pytest.mark.parametrize("key", ["nlp-rule-vs-statistical-ner", "nlp-text-normalization"])
def test_catalog_example_is_complete_and_executes(key):
    note = CONCEPT_NOTES[key]
    source = _example_code(key, note, {})
    assert "\n".join(note.mini_example) in source
    assert "your_data.csv" not in source
    assert static_check(source)["status"] == "static-pass"
    result = execute(Example("dashboard:" + key, 1, "python", source, "rendered"), timeout=180)
    assert result["status"] == "passed", result
