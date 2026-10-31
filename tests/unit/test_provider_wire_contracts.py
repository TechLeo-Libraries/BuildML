"""Offline SDK boundaries preserve conversation and tolerate malformed proposals."""

import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from buildml.ai.provider import OpenAIProvider, ProviderConfig
from buildml.ai.types import Message, ToolCall
from buildml.core.errors import ValidationError


def provider(monkeypatch, response):
    create = Mock(return_value=response)
    client = NS(chat=NS(completions=NS(create=create)))
    constructor = Mock(return_value=client)
    monkeypatch.setattr("buildml.ai.extras.require_openai", lambda **kwargs: NS(OpenAI=constructor))
    instance = OpenAIProvider(ProviderConfig(api_key="offline-test-secret", model="contract-model", max_tokens=80))
    assert constructor.call_args.kwargs["api_key"] == "offline-test-secret"
    return instance, create


@pytest.mark.parametrize("arguments,expected", [
    ('{"x": 3}', {"x": 3}), ({"x": 4}, {"x": 4}),
    ('{"broken":', {}), ('[1, 2]', {}), ('null', {}), (' ', {}), (None, {}),
])
def test_tool_proposals_parse_without_granting_permission(monkeypatch, arguments, expected):
    response = NS(choices=[NS(message=NS(content=None, tool_calls=[
        NS(id="call-1", function=NS(name="fit_classical", arguments=arguments))
    ]), finish_reason="tool_calls")], usage=NS(prompt_tokens=11, completion_tokens=7, total_tokens=18), model="answered-model")
    instance, create = provider(monkeypatch, response)
    result = instance.chat([Message(role="user", content="Fit")], tools=[{"type": "function"}])
    assert result.content == ""
    assert result.tool_calls == (ToolCall("fit_classical", expected, "call-1"),)
    assert result.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    assert result.model == "answered-model"
    assert result.finish_reason == "tool_calls"
    assert create.call_args.kwargs["tools"] == [{"type": "function"}]


def test_conversation_serialization_and_request_overrides(monkeypatch):
    response = NS(choices=[NS(message=NS(content="done", tool_calls=None), finish_reason=None)], usage=None, model="reply")
    instance, create = provider(monkeypatch, response)
    result = instance.chat([
        Message(role="assistant", content="", tool_calls=(ToolCall("evaluate", {"split": "test"}, "id-2"),)),
        Message(role="tool", content="result", tool_call_id="id-2", name="evaluate"),
    ], max_tokens=23, temperature=0.7)
    request = create.call_args.kwargs
    assert request["max_tokens"] == 23
    assert request["temperature"] == 0.7
    assert "tools" not in request
    call = request["messages"][0]["tool_calls"][0]
    assert call["id"] == "id-2"
    assert json.loads(call["function"]["arguments"]) == {"split": "test"}
    assert request["messages"][1] == {"role": "tool", "content": "result", "tool_call_id": "id-2", "name": "evaluate"}
    assert result.content == "done"
    assert result.finish_reason == "stop"
    assert result.usage == {}
    instance.chat([Message(role="user", content="next")])
    assert create.call_args.kwargs["max_tokens"] == 80
    assert create.call_args.kwargs["temperature"] == 0.0


def test_request_error_public_message_masks_configured_key(monkeypatch):
    instance, create = provider(monkeypatch, None)
    create.side_effect = RuntimeError("auth failed: offline-test-secret")
    with pytest.raises(ValidationError, match="Provider request failed") as error:
        instance.chat([Message(role="user", content="hello")])
    assert "offline-test-secret" not in str(error.value)
    assert "REDACTED" in str(error.value)
