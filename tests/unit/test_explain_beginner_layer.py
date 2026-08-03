"""Guards for the beginner-facing half of the explain system.

The point of these tests is that a beginner explanation cannot quietly regress
into the expert one. Prose is not something a type checker can police, so the
checks here are structural (every concept has a layer, every operation derives a
primer) plus a small number of readability rules that catch the specific failure
mode this system was built to fix: jargon explained with more jargon.
"""

from __future__ import annotations

import dataclasses
import json
import re

import pytest

from buildml import Session
from buildml.explain import (
    CONCEPT_NOTES,
    GLOSSARY,
    OPERATION_CATALOG,
    ConceptDifficulty,
    LearningLevel,
    concepts_at,
    learn,
    learning_path,
    primer_for,
    starting_points,
)
from buildml.explain.glossary import CONCEPT_FOR_TERM, lookup
from buildml.explain.pedagogy import derive_primer
from buildml.explain.prerequisites import PREREQUISITES, plain_prerequisite
from buildml.explain.schemas import (
    _CONCEPT_NOTE_SEQUENCES,
    _OPERATION_SPEC_SEQUENCES,
)

# Words that mean nothing to a beginner. A plain-language summary may still use
# one, but only if the same note defines it, so the reader is never left stuck.
_JARGON = re.compile(
    r"\b("
    r"stochastic|orthogonal|heteroscedastic|homoscedastic|eigen\w*|"
    r"hyperparameter|regularization|regularisation|posterior|likelihood|"
    r"gradient|convex|kernel|latent|embedding|estimator|residual|"
    r"quantile|variance|covariance|entropy|logit|softmax|manifold|"
    r"stratified|leakage|imputation|cardinality|multicollinearity"
    r")\b",
    re.IGNORECASE,
)

# A derived neighbour, as opposed to an authored sentence that mentions a call.
_RENDERED_CALL = re.compile(r"session\.([a-z_0-9]+)\(\)")


def _defined_vocabulary(note) -> set[str]:
    defined: set[str] = set()
    for entry in note.glossary:
        defined.add(entry.term.casefold())
        defined.update(alias.casefold() for alias in entry.also_called)
    return defined


# ---------------------------------------------------------------- concepts ---


def test_every_concept_carries_a_beginner_layer() -> None:
    missing = [key for key, note in CONCEPT_NOTES.items() if not note.has_beginner_layer]
    assert missing == [], f"concepts still at the old shallow standard: {missing}"


def test_beginner_layers_are_substantive() -> None:
    for key, note in CONCEPT_NOTES.items():
        assert len(note.plain_summary) >= 120, f"{key}: plain summary is too thin"
        assert note.analogy, f"{key}: no analogy"
        assert len(note.beginner_steps) >= 2, f"{key}: needs a step-by-step walkthrough"
        assert note.when_to_use, f"{key}: no guidance on when to use it"
        assert note.when_not_to_use, f"{key}: no guidance on when to avoid it"
        assert note.misconceptions, f"{key}: no misconceptions recorded"
        assert note.mini_example, f"{key}: no worked example"
        assert note.check_yourself, f"{key}: no self-check questions"
        assert note.buildml_tools, f"{key}: not connected to any BuildML operation"


def test_misconceptions_state_both_the_belief_and_the_correction() -> None:
    for key, note in CONCEPT_NOTES.items():
        for item in note.misconceptions:
            assert item.myth, f"{key}: misconception with no stated belief"
            assert item.reality, f"{key}: misconception with no correction"
            assert item.myth != item.reality


def test_plain_summaries_define_the_jargon_they_use() -> None:
    offenders: list[str] = []
    for key, note in CONCEPT_NOTES.items():
        defined = _defined_vocabulary(note)
        for match in _JARGON.finditer(note.plain_summary):
            word = match.group(0).casefold()
            if word not in defined and not any(word in term for term in defined):
                offenders.append(f"{key}: uses {word!r} without defining it")
    assert offenders == [], "\n".join(offenders)


def test_concept_navigation_points_at_real_concepts() -> None:
    for key, note in CONCEPT_NOTES.items():
        for other in (*note.prerequisite_concepts, *note.next_concepts):
            assert other in CONCEPT_NOTES, f"{key} links unknown concept {other!r}"
            assert other != key, f"{key} lists itself as a neighbour"


def test_learning_path_terminates_and_leads_with_prerequisites() -> None:
    for key in CONCEPT_NOTES:
        path = learning_path(key)
        assert path[-1] == key, f"{key}: learning path should end at the concept itself"
        assert len(path) == len(set(path)), f"{key}: learning path repeats a concept"
        for step in path:
            assert step in CONCEPT_NOTES


def test_concepts_at_partitions_the_catalog_by_difficulty() -> None:
    seen: set[str] = set()
    for rung in ConceptDifficulty:
        keys = {note.key for note in concepts_at(rung)}
        assert not (keys & seen), "a concept cannot sit on two rungs"
        seen |= keys
    assert seen == set(CONCEPT_NOTES)
    assert concepts_at(ConceptDifficulty.FOUNDATION), "no entry-level concepts"


# ---------------------------------------------------------------- glossary ---


def test_every_glossary_term_has_a_plain_meaning_and_resolves() -> None:
    for key, entry in GLOSSARY.items():
        assert entry.term
        assert len(entry.plain_meaning) >= 40, f"{key}: definition is too thin"
        assert lookup(entry.term) is entry
        for alias in entry.also_called:
            assert lookup(alias) is entry, f"{key}: alias {alias!r} does not resolve"


def test_curated_term_to_concept_mappings_point_at_real_concepts() -> None:
    unknown = {
        term: key for term, key in CONCEPT_FOR_TERM.items() if key not in CONCEPT_NOTES
    }
    assert unknown == {}


def test_every_glossary_term_leads_somewhere_deeper() -> None:
    """A definition is a dead end unless it hands the reader on."""
    orphans = []
    for entry in GLOSSARY.values():
        brief = learn(entry.term)
        if brief.concept is None and brief.operation is None:
            orphans.append(entry.term)
    assert orphans == [], f"terms with a definition but nowhere to go: {orphans}"


# --------------------------------------------------------------- operation ---


def test_every_operation_derives_a_complete_beginner_primer() -> None:
    required = (
        "plain_summary",
        "analogy",
        "why_it_exists",
        "steps",
        "prerequisites_in_plain_words",
        "when_to_use",
        "when_not_to_use",
        "what_changes",
        "how_to_read_the_result",
        "common_pitfalls",
        "mini_example",
        "related_tools",
        "learn_next",
    )
    for name in OPERATION_CATALOG:
        primer = primer_for(name)
        assert primer.operation == name
        assert primer.level is LearningLevel.BEGINNER
        for field_name in required:
            assert getattr(primer, field_name), f"{name}.{field_name} is empty"
        json.dumps(primer.to_dict())


def test_no_authored_list_is_secretly_a_single_string() -> None:
    """A missing trailing comma must not become one bullet per character.

    Every prose field is annotated as a tuple, but the catalog is hand-written
    literals and ``("only item")`` is a string. Nothing downstream would raise;
    the reader would simply be handed the alphabet.
    """
    for name, spec in OPERATION_CATALOG.items():
        for item in dataclasses.fields(spec):
            value = getattr(spec, item.name)
            if item.name in _OPERATION_SPEC_SEQUENCES:
                assert isinstance(value, tuple), f"{name}.{item.name} is not a tuple"
                assert all(isinstance(entry, str) for entry in value)
    for key, note in CONCEPT_NOTES.items():
        for item in dataclasses.fields(note):
            if item.name in _CONCEPT_NOTE_SEQUENCES:
                value = getattr(note, item.name)
                assert isinstance(value, tuple), f"{key}.{item.name} is not a tuple"


def test_primer_sections_do_not_repeat_each_other() -> None:
    """Reading the same warning twice in one briefing teaches nothing."""
    for name in OPERATION_CATALOG:
        primer = primer_for(name)
        overlap = set(primer.when_not_to_use) & set(primer.common_pitfalls)
        if overlap:
            # Allowed only where subtraction would leave the pitfalls empty.
            assert set(primer.common_pitfalls) <= set(primer.when_not_to_use), (
                f"{name}: partial duplication between avoidance and pitfalls"
            )
        spoken = set(
            re.findall(r"[a-z_0-9]+", " ".join(OPERATION_CATALOG[name].alternatives))
        )
        for tool in primer.related_tools:
            rendered = _RENDERED_CALL.fullmatch(tool)
            if rendered is None:
                continue
            bare = rendered.group(1)
            if "_" in bare:
                assert bare not in spoken, f"{name}: {bare} listed twice in related tools"


def test_primer_parameters_cover_every_required_argument() -> None:
    for name, spec in OPERATION_CATALOG.items():
        required = {item.name for item in spec.parameters if item.required}
        explained = {item.name for item in primer_for(name).key_parameters}
        assert required <= explained, f"{name}: required parameters left unexplained"


def test_primer_parameters_always_say_something_useful() -> None:
    for name in OPERATION_CATALOG:
        for meaning in primer_for(name).key_parameters:
            assert meaning.plain_meaning, f"{name}.{meaning.name} has no meaning"
            assert meaning.typical_choice, f"{name}.{meaning.name} has no guidance"


def test_reading_level_changes_depth_but_not_truth() -> None:
    beginner = primer_for("split", level=LearningLevel.BEGINNER)
    intermediate = primer_for("split", level=LearningLevel.INTERMEDIATE)
    advanced = primer_for("split", level=LearningLevel.ADVANCED)

    assert beginner.analogy and not advanced.analogy
    assert beginner.glossary and not advanced.glossary
    assert len(intermediate.glossary) <= len(beginner.glossary)
    assert len(advanced.key_parameters) >= len(beginner.key_parameters)
    assert len(advanced.common_pitfalls) >= len(beginner.common_pitfalls)
    for primer in (beginner, intermediate, advanced):
        assert primer.plain_summary == beginner.plain_summary
        assert primer.why_it_exists == beginner.why_it_exists


def test_authored_beginner_prose_overrides_derivation() -> None:
    spec = OPERATION_CATALOG["explain"]
    assert spec.plain_summary, "explain should carry hand-written beginner prose"
    assert derive_primer(spec).plain_summary == spec.plain_summary
    assert derive_primer(spec).analogy == spec.analogy


def test_every_catalog_prerequisite_has_plain_words() -> None:
    keys = {
        item.key for spec in OPERATION_CATALOG.values() for item in spec.prerequisites
    }
    missing = sorted(keys - set(PREREQUISITES))
    assert missing == [], f"prerequisites the resolver cannot evaluate: {missing}"
    for key in keys:
        sentence = plain_prerequisite(key)
        assert sentence.endswith("."), f"{key}: not a sentence"
        assert len(sentence) >= 20, f"{key}: too terse to help anyone"


# ------------------------------------------------------------------ session ---


def test_explain_before_carries_the_primer_at_every_level() -> None:
    session = Session()
    for level in LearningLevel:
        explanation = session.explain("ingest", level=level.value)
        assert explanation.beginner is not None
        assert explanation.beginner.level is level
        assert explanation.beginner.plain_summary


def test_explain_rejects_an_unknown_reading_level() -> None:
    session = Session()
    with pytest.raises(ValueError, match="Unknown learning level"):
        session.explain("ingest", level="expert")


def test_learn_resolves_concepts_operations_and_terms() -> None:
    session = Session()
    assert session.learn("leakage-boundary").kind == "concept"
    assert session.learn("split").kind == "operation"
    assert session.learn("overfitting").concept is not None
    assert session.learn().kind == "index"


def test_learn_tolerates_the_punctuation_nobody_remembers() -> None:
    canonical = learn("roc-auc")
    for spelling in ("ROC AUC", "roc auc", "Roc-Auc", "roc_auc"):
        assert learn(spelling).topic == canonical.topic


def test_learn_suggests_alternatives_for_an_unknown_topic() -> None:
    with pytest.raises(KeyError, match="Did you mean"):
        learn("splt")


def test_learn_gives_a_bounded_reading_list() -> None:
    for key in CONCEPT_NOTES:
        brief = learn(key)
        assert brief.concept is not None
        assert len(brief.related_operations) <= 12
        assert all(name in OPERATION_CATALOG for name in brief.related_operations)


def test_starting_points_lead_with_foundation_concepts() -> None:
    points = starting_points()
    assert points, "a newcomer needs somewhere to start"
    first = CONCEPT_NOTES[points[0]]
    assert first.difficulty is ConceptDifficulty.FOUNDATION
