"""Corpus ASR counts and transcript boundaries agree across public consumers."""

import pytest

from buildml import Session
from buildml.ai.executor import execute_tool, propose_tool_execution
from buildml.ai.tools import build_default_registry
from buildml.core.errors import ValidationError
from buildml.dl.speech import evaluate_asr


def invoke(surface, session, **options):
    if surface == 'direct':
        return evaluate_asr(**options)
    if surface == 'session':
        return session.dl.evaluate_asr(**options)
    registry = build_default_registry()
    proposal = propose_tool_execution('evaluate_asr', options, registry)
    outcome = execute_tool(session, proposal, confirmed=False, registry=registry)
    assert outcome.executed, outcome.error
    return outcome.result


@pytest.mark.parametrize('surface', ['direct', 'session', 'agent'])
@pytest.mark.parametrize('references,hypotheses,lowercase,words,chars,wer,cer', [
    (['a b c', 'd'], ['a b x', ''], True, 4, 4, .5, .5),
    (['red blue', 'x'], ['red blue green', 'x'], True, 3, 8, 1/3, 5/8),
    (['A B', 'c'], ['a b', 'C'], True, 3, 3, 0., 0.),
    (['A B', 'c'], ['a b', 'C'], False, 3, 3, 1., 1.),
    (['', 'a b'], ['x', 'a b'], True, 2, 2, .5, .5),
    (['', ''], ['x', ''], True, 0, 0, 1., 1.),
    (['', ''], ['', ''], True, 0, 0, 0., 0.),
])
def test_corpus_edit_totals_are_pooled_without_inventing_reference_units(
    surface, references, hypotheses, lowercase, words, chars, wer, cer
):
    result = invoke(surface, Session(), references=references, hypotheses=hypotheses,
                    lowercase=lowercase)
    assert result.n_utterances == len(references)
    assert result.n_ref_words == words and result.n_ref_chars == chars
    assert result.wer == pytest.approx(wer) and result.cer == pytest.approx(cer)


@pytest.mark.parametrize('surface', ['direct', 'session', 'agent'])
@pytest.mark.parametrize('field', ['references', 'hypotheses'])
@pytest.mark.parametrize('value', ['same', {'same': 1}, [123], [None], [['nested']]])
def test_invalid_transcript_containers_are_refused_before_recording_metrics(surface, field, value):
    s = Session()
    history = list(s._history)
    options = {'references': ['same'], 'hypotheses': ['same'], field: value}
    if surface == 'agent':
        registry = build_default_registry()
        proposal = propose_tool_execution('evaluate_asr', options, registry)
        result = execute_tool(s, proposal, confirmed=False, registry=registry)
        assert not result.executed
        assert field in result.error and 'strings' in result.error
    else:
        with pytest.raises(ValidationError, match=field):
            invoke(surface, s, **options)
    assert s._dl_asr_eval is None
    assert s._history == history
