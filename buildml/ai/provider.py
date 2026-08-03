"""Talk to a language model, or convincingly pretend to.

:class:`ProviderProtocol` is the whole interface: one ``chat`` method taking
messages and tool declarations and returning a :class:`ProviderResponse`.
Everything above this layer is written against that protocol, so swapping a
hosted model for a scripted one changes nothing else.

Two implementations ship. :class:`OpenAIProvider` calls an OpenAI-compatible
endpoint. :class:`MockProvider` returns queued responses with no network, which
is what makes the AI domain testable in CI and demonstrable offline — the tool
registry, the confirmation flow, and the egress accounting all exercise
identically against it.

API keys are handled carefully throughout. :class:`ProviderConfig` reads from an
environment variable by default, its ``repr`` reports only whether a key is set,
its ``to_dict`` masks the value, and provider errors are scrubbed before being
re-raised — an authentication failure that echoed the key back would be the
worst possible place to leak one.

See Also
--------
buildml.ai.extras : Whether a real provider is installed.
buildml.ai.types.Message : The conversation unit.
"""

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
    """Which model to call, where, and with what credential.

    Attributes
    ----------
    provider:
        Which service. ``'openai'`` covers any OpenAI-compatible endpoint.
    model:
        The model identifier.
    api_key:
        The credential. Left ``None`` to read from the environment, which is
        the safer habit — a key written into source is a key that gets
        committed.
    api_key_env:
        Which environment variable to read.
    base_url:
        An alternative endpoint, for compatible services or a local server.
    max_tokens:
        Response length cap. ``None`` uses the provider's default.
    temperature:
        Sampling randomness. Defaults to 0.0, because advice about your data
        should not vary between identical runs.
    timeout:
        Request timeout in seconds.

    Notes
    -----
    **The key never appears in output.** ``repr`` says only whether one is set,
    :meth:`to_dict` masks it, and provider errors are scrubbed. The value is
    reachable only by reading the attribute directly.

    **Temperature 0.0 is near-deterministic, not deterministic.** Providers
    make no exact-reproducibility guarantee even at zero.

    Examples
    --------
    Read the key from the environment::

        config = ProviderConfig(model="gpt-4o-mini")
        provider = OpenAIProvider(config)

    See Also
    --------
    OpenAIProvider : What consumes this.
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
        """Return the configuration as JSON-safe values, with the key masked.

        Safe to log or persist: the credential is replaced with a placeholder
        while everything needed to reproduce the setup is kept.

        Returns
        -------
        dict
            Provider, model, a masked key indicator, the environment variable
            name, base URL, token cap, temperature, and timeout.

        Notes
        -----
        ``api_key`` is ``'***REDACTED***'`` when set and ``None`` when not, so
        the record still says whether a credential was present.
        """
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
    """What came back from one model call.

    Attributes
    ----------
    content:
        The text. Empty when the model responded only with tool calls.
    tool_calls:
        Actions the model wants taken. **Proposals** — nothing has run.
    finish_reason:
        Why generation stopped. ``'stop'`` for a complete answer,
        ``'tool_calls'`` when the model wants to act, ``'length'`` when it hit
        the token cap mid-sentence.
    usage:
        Prompt, completion, and total token counts.
    model:
        Which model actually answered. Can differ from what was requested when
        a provider aliases or upgrades a name.
    raw:
        The unprocessed provider payload.

    Notes
    -----
    **``finish_reason='length'`` means the answer is truncated.** The content
    will read as though it simply ended. Raise ``max_tokens`` or shorten the
    prompt.

    See Also
    --------
    ProviderProtocol : What returns this.
    buildml.ai.types.ToolCall : What a proposal looks like.
    """

    content: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the response as JSON-safe values.

        Omits ``raw``, which is provider-shaped, potentially large, and
        duplicates everything above it.

        Returns
        -------
        dict
            Content, tool calls, finish reason, token usage, and model.
        """
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": dict(self.usage),
            "model": self.model,
        }


@runtime_checkable
class ProviderProtocol(Protocol):
    """The one method a provider must have.

    A structural protocol: anything with a matching ``chat`` satisfies it, with
    no base class to inherit. That is what lets :class:`MockProvider` stand in
    for a hosted model everywhere, and what makes adding a new backend a matter
    of writing one method.

    Notes
    -----
    **Deliberately minimal.** Streaming, retries, and caching are absent
    because they are not needed to define what a provider is; a wrapper can add
    any of them and still satisfy the protocol.

    See Also
    --------
    MockProvider : The offline implementation.
    OpenAIProvider : The hosted implementation.
    """

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        """Send the conversation and return what the model said.

        The single operation a provider must support. Everything the AI domain
        does — advising, planning, acting — is built from repeated calls to
        this.

        Parameters
        ----------
        messages:
            The conversation so far, oldest first. The model's entire memory.
        tools:
            Tool declarations the model may call. ``None`` for a text-only
            turn.
        max_tokens:
            Response length cap, overriding the provider's configured value.
        temperature:
            Sampling randomness, overriding the configured value.

        Returns
        -------
        ProviderResponse
            Content, any proposed tool calls, the finish reason, and usage.

        Raises
        ------
        ValidationError
            If the request fails. Implementations must scrub credentials from
            the message first.

        Notes
        -----
        **Declaring a tool does not authorise it.** Returned calls are
        proposals, subject to the registry and the confirmation policy.
        """
        ...


class MockProvider:
    """A provider that returns what you told it to, with no network.

    Scripted rather than random. Queue the tool calls and the text you want,
    and it hands them back in order — which turns an agent loop from something
    you observe into something you assert on.

    Every request is recorded in ``calls``, so a test can check not only what
    the agent did but what it was told and what tools it was offered.

    Attributes
    ----------
    default_response:
        Returned when no queued text remains.
    tool_responses:
        Canned outputs by tool name, for fixtures that need them.
    calls:
        Every request received, in order — messages, tools, and the sampling
        overrides.

    Notes
    -----
    **Responses are consumed in a fixed order**: the single-slot tool call
    first, then the queue, then queued text, then the default. Nothing depends
    on what the agent actually said.

    **This means the mock does not exercise model behaviour**, and cannot. It
    exercises everything around the model: validation, confirmation, egress
    accounting, transcript recording, and loop control.

    Examples
    --------
    Script a two-step run::

        provider = MockProvider()
        provider.queue_tool_calls([("describe_dataset", {})])
        provider.queue_responses(["Numeric target; try regression."])

    See Also
    --------
    OpenAIProvider : The real thing.
    """

    def __init__(
        self,
        *,
        default_response: str = "This is a mock response.",
        tool_responses: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Build a mock provider with its fallback response.

        Nothing is queued at construction; use :meth:`queue_tool_calls` and
        :meth:`queue_responses` to script a run.

        Parameters
        ----------
        default_response:
            Text returned once the queues are empty.
        tool_responses:
            Canned tool outputs by name.
        """
        self.default_response = default_response
        self.tool_responses = tool_responses or {}
        self.calls: list[dict[str, Any]] = []
        self._next_tool_call: ToolCall | None = None
        self._tool_call_queue: list[ToolCall] = []
        self._response_queue: list[str] = []

    def set_next_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Script exactly one tool call for the next turn.

        Takes priority over the FIFO queue, and is cleared once returned. For
        the common single-step test; use :meth:`queue_tool_calls` for a
        sequence.

        Parameters
        ----------
        tool_name:
            The tool to propose. Not checked against any registry — proposing
            an unregistered tool is a useful thing to test.
        arguments:
            The arguments to propose. Not validated here either.

        Notes
        -----
        Calling this twice replaces the pending call rather than queueing a
        second one.

        See Also
        --------
        queue_tool_calls : A sequence.
        """
        self._next_tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            call_id=f"call_{uuid.uuid4().hex[:8]}",
        )

    def queue_tool_calls(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        """Script a sequence of tool calls, one per turn.

        Returned in order, one per ``chat``, which is how a multi-step agent
        run gets a deterministic script.

        Parameters
        ----------
        calls:
            ``(tool_name, arguments)`` pairs, in the order to return them.

        Notes
        -----
        **One call per turn, even though the format allows several.** Keeping
        it to one makes each step separately assertable.

        Each gets a generated ``call_id``, matching what a real provider does.

        Examples
        --------
        A three-step pipeline::

            provider.queue_tool_calls([
                ("describe_dataset", {}),
                ("suggest_roles", {}),
                ("split_data", {"test_size": 0.2}),
            ])

        See Also
        --------
        queue_responses : What follows once these are exhausted.
        """
        for tool_name, arguments in calls:
            self._tool_call_queue.append(
                ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    call_id=f"call_{uuid.uuid4().hex[:8]}",
                )
            )

    def queue_responses(self, texts: list[str]) -> None:
        """Script the text replies that follow the tool calls.

        Returned one per turn once the tool-call queues are empty — the
        agent's closing summary after it has finished acting.

        Parameters
        ----------
        texts:
            Replies in order. Appended to anything already queued.

        Notes
        -----
        Once these run out, ``default_response`` is returned indefinitely, so a
        loop that runs longer than expected does not fail for lack of a script.

        See Also
        --------
        queue_tool_calls : The turns that come first.
        """
        self._response_queue.extend(texts)

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        """Return the next scripted response, recording the request.

        Satisfies :class:`ProviderProtocol` without a network call. The request
        is appended to ``calls`` before anything is returned, so a test can
        inspect what the agent sent even when the response is fixed.

        Parameters
        ----------
        messages:
            The conversation. Recorded, not read.
        tools:
            Tool declarations. Recorded, not read.
        max_tokens:
            Recorded, not applied.
        temperature:
            Recorded, not applied.

        Returns
        -------
        ProviderResponse
            The next scripted response, with fixed token counts and model name
            ``'mock-model'``.

        Notes
        -----
        **The response ignores the input entirely.** That is the point: the
        agent's behaviour becomes a function of the script, so a test failure
        means the agent changed rather than the model did.

        Token counts are constant placeholders and mean nothing.
        """
        self.calls.append({
            "messages": [m.to_dict() for m in messages],
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        if self._next_tool_call is not None:
            tool_calls = (self._next_tool_call,)
            self._next_tool_call = None
            return ProviderResponse(
                content="",
                tool_calls=tool_calls,
                finish_reason="tool_calls",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                model="mock-model",
            )

        if self._tool_call_queue:
            tool_call = self._tool_call_queue.pop(0)
            return ProviderResponse(
                content="",
                tool_calls=(tool_call,),
                finish_reason="tool_calls",
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                model="mock-model",
            )

        content = self.default_response
        if self._response_queue:
            content = self._response_queue.pop(0)

        return ProviderResponse(
            content=content,
            tool_calls=(),
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            model="mock-model",
        )


class OpenAIProvider:
    """Call an OpenAI-compatible chat endpoint.

    Translates BuildML's :class:`~buildml.ai.types.Message` and tool
    declarations into the wire format, sends the request, and parses what comes
    back into a :class:`ProviderResponse`.

    Works against any compatible endpoint via ``base_url``, including local
    servers and other hosted providers that speak the same protocol.

    Notes
    -----
    **Credentials are scrubbed from errors.** A provider exception whose text
    contains the key is rewritten before being re-raised, since authentication
    failures are exactly where a key would otherwise surface.

    **Malformed tool arguments degrade rather than crash.** When a model emits
    invalid JSON for its arguments, the call is kept with empty arguments and
    the schema check downstream reports what is missing — a more useful failure
    than a parse error.

    Examples
    --------
    Point at a local compatible server::

        config = ProviderConfig(
            model="local-model",
            base_url="http://localhost:8000/v1",
            api_key="not-used",
        )
        provider = OpenAIProvider(config)

    See Also
    --------
    MockProvider : The offline stand-in.
    buildml.ai.extras.require_openai : The dependency gate.
    """

    def __init__(self, config: ProviderConfig) -> None:
        """Build a client, failing early if it cannot be used.

        Both the package and the credential are checked here rather than at
        the first request, so a misconfiguration surfaces at construction where
        it is easy to attribute.

        Parameters
        ----------
        config:
            Model, credential, endpoint, and request settings.

        Raises
        ------
        MissingExtraError
            If the ``openai`` package is not installed. Install with
            ``pip install buildml[ai]``.
        ValidationError
            If no API key was supplied or found in the environment. The message
            names the variable to set.
        """
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
        """Send the conversation to the model and parse the reply.

        Converts messages to the wire format, applies the configured or
        overridden sampling settings, sends the request, and turns the response
        into a :class:`ProviderResponse` with any tool calls already parsed.

        Parameters
        ----------
        messages:
            The conversation, oldest first.
        tools:
            Tool declarations from
            :meth:`~buildml.ai.tools.ToolRegistry.to_openai_tools`. ``None``
            for a text-only turn.
        max_tokens:
            Response cap for this request, overriding the configuration.
        temperature:
            Sampling randomness for this request, overriding the
            configuration.

        Returns
        -------
        ProviderResponse
            Content, proposed tool calls, finish reason, token usage, and the
            model that answered.

        Raises
        ------
        ValidationError
            If the request fails — network, authentication, rate limit, or
            malformed request. The original is chained, with any occurrence of
            the API key masked.

        Notes
        -----
        **Check ``finish_reason`` before trusting the content.** ``'length'``
        means the reply was cut off mid-generation and reads as though it
        simply ended.

        **Tool calls returned here have not been validated.** They are the
        model's proposals; the registry decides whether they are permitted and
        the confirmation policy decides whether they run.

        **Unparseable tool arguments become an empty dictionary.** The failure
        then surfaces as a missing required argument, which says more than a
        JSON error would.
        """
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
                raw_args = tc.function.arguments
                args: dict[str, Any] = {}
                if isinstance(raw_args, dict):
                    args = dict(raw_args)
                elif isinstance(raw_args, str) and raw_args.strip():
                    try:
                        parsed = json.loads(raw_args)
                        if isinstance(parsed, dict):
                            args = parsed
                    except json.JSONDecodeError:
                        # Tolerant fallback: empty args; executor validates required fields.
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
