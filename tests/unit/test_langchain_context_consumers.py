"""Real offline LangChain chains honor the recorded BuildML passage budget."""
import pytest

pytest.importorskip('langchain_classic')
pytest.importorskip('langchain_community')
from langchain_core.language_models.fake import FakeListLLM
from pydantic import Field

from buildml.core.errors import ValidationError
from buildml.rag.adapters.langchain import LangChainGroundedAdapter
from buildml.rag.results import Hit
from buildml.rag.types import GenerateConfig


class RecordingLLM(FakeListLLM):
    prompts: list[str] = Field(default_factory=list)

    def _call(self,prompt,stop=None,run_manager=None,**kwargs):
        self.prompts.append(prompt)
        return super()._call(prompt,stop=stop,run_manager=run_manager,**kwargs)


def hits():
    return [Hit(chunk_id=f'c{i}',doc_id=f'd{i}',text=f'PASSAGE_{i}_UNIQUE',rank=i+1,score=1/(i+1)) for i in range(3)]


@pytest.mark.parametrize('k',[1,2,8])
def test_real_chain_context_and_citations_match_budget(k):
    llm=RecordingLLM(responses=['An offline interface response.'])
    result=LangChainGroundedAdapter(llm).generate_from_hits('Budget question',hits(),config=GenerateConfig(k=k))
    expected=hits()[:k]
    assert len(llm.prompts)==1
    for i in range(3):
        assert (f'PASSAGE_{i}_UNIQUE' in llm.prompts[0]) is (i<k)
    assert result.prompt_context=='\n\n'.join(h.text for h in expected)
    assert [c.chunk_id for c in result.citations]==[h.chunk_id for h in expected]
    assert result.retrieve_result.hits==tuple(expected)
    assert result.retrieve_result.k==len(expected)
    assert result.faithfulness is None
    assert not result.usage


@pytest.mark.parametrize('k',[0,-1,True,1.5,'2',None])
def test_invalid_passage_budget_refused_before_llm(k):
    llm=RecordingLLM(responses=['unused'])
    adapter=LangChainGroundedAdapter(llm)
    with pytest.raises(ValidationError,match='k.*positive integer'):
        adapter.generate_from_hits('question',hits(),config=GenerateConfig(k=k))
    assert not llm.prompts


def test_empty_answer_refused():
    with pytest.raises(ValidationError,match='empty answer'):
        LangChainGroundedAdapter(RecordingLLM(responses=['   '])).generate_from_hits('question',hits())
