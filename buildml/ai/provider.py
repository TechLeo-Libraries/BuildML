"""Provider protocol and implementations for AI operator."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from buildml.ai.types import Message, ToolCall
from buildml.core.errors import ValidationError

_KEY_MASK = "***REDACTED***"


@dataclass(slots=True)
class ProviderConfig:
    """Configuration for an LLM provider.

    API keys are read from environment variables by default. Never logs,
    persists, or echoes the key value.
    """

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    api_key_env: str = "BUILDML_OPENAI_API_KEY"
    base_url: str | None = None
    max_tokens: int | None = None
    temperature: float = 0.0
    timeout: float = 60.0

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get(self.api_key_env)

    def __repr__(self) -> str:
        key_status = "set" if self.api_key else "not set"
        return (
            f"ProviderConfig(provider={self.provider!r}, model={self.model!r}, "
            f"api_key={key_status}, api_key_env={self.api_key_env!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": _KEY_MASK if self.api_key else None,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }


@dataclass(slots=True)
class ProviderResponse:
    """Response from an LLM provider call."""

    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": dict(self.usage),
            "model": self.model,
        }


@runtime_checkable
class ProviderProtocol(Protocol):
    """Protocol for LLM provider implementations."""

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        """Send a chat request and return the response."""
        ...


class MockProvider:
    """Mock provider for CI testing without real API keys.

    Returns canned responses and records all calls for test assertions.
    """

    def __init__(
        self,
        *,
        default_response: str = "This is a mock response.",
        tool_responses: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.default_response = default_response
        self.tool_responses = tool_responses or {}
        self.calls: list[dict[str, Any]] = []
        self._next_tool_call: ToolCall | None = None

    def set_next_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Queue a tool call for the next response."""
        self._next_tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            call_id=f"call_{uuid.uuid4().hex[:8]}",
        )

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        self.calls.append({
            "messages": [m.to_dict() for m in messages],
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        tool_calls: tuple[ToolCall, ...] = ()
        if self._next_tool_call is not None:
            tool_calls = (self._next_tool_call,)
            self._next_tool_call = None
            return ProviderResponse(
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                usage={"prompt_tokens": 100, "completion_tokens": 50},
                model="mock-model",
            )

        return ProviderResponse(
            content=self.default_response,
            tool_calls=(),
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            model="mock-model",
        )


class OpenAIProvider:
    """OpenAI-compatible provider implementation.

    Requires the openai package and a valid API key. Never logs or persists
    the API key.
    """

    def __init__(self, config: ProviderConfig) -> None:
        from buildml.ai.extras import require_openai

        openai = require_openai(feature="OpenAI provider")

        if not config.api_key:
            raise ValidationError(
                f"API key not set. Set {config.api_key_env} environment variable "
                f"or pass api_key to ProviderConfig."
            )

        self.config = config
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        import json

        api_messages = []
        for msg in messages:
            api_msg: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                api_msg["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id:
                api_msg["tool_call_id"] = msg.tool_call_id
            if msg.name:
                api_msg["name"] = msg.name
            api_messages.append(api_msg)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        elif self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens

        if tools:
            kwargs["tools"] = tools

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            error_msg = str(exc)
            if self.config.api_key and self.config.api_key in error_msg:
                error_msg = error_msg.replace(self.config.api_key, _KEY_MASK)
            raise ValidationError(f"Provider request failed: {error_msg}") from exc

        choice = response.choices[0]
        content = choice.message.content or ""

        tool_calls: tuple[ToolCall, ...] = ()
        if choice.message.tool_calls:
            parsed_calls = []
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                parsed_calls.append(
                    ToolCall(
                        tool_name=tc.function.name,
                        arguments=args,
                        call_id=tc.id,
                    )
                )
            tool_calls = tuple(parsed_calls)

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            model=response.model,
        )
