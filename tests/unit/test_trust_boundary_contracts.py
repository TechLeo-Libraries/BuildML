"""Trust permissions and public error tracebacks must respect security boundaries."""
import traceback
from types import SimpleNamespace as NS
from unittest.mock import Mock

import joblib
import numpy as np
import pytest

from buildml.ai.provider import OpenAIProvider, ProviderConfig
from buildml.ai.types import Message
from buildml.core.errors import ValidationError
from buildml.core.serialization import joblib_load_trusted
from buildml.serving.config import ServeConfig
from buildml.serving.launch import serve_bundle


@pytest.mark.parametrize("trusted", [1, 2, "false", "true", "0", "1", np.bool_(True), False, None, 0, [], {}])
def test_deserialization_requires_literal_permission(tmp_path, monkeypatch, trusted):
    path = tmp_path / "benign.joblib"
    joblib.dump({"safe": "fixture"}, path)
    loader = Mock(side_effect=AssertionError("Deserializer ran without literal permission"))
    monkeypatch.setattr(joblib, "load", loader)
    with pytest.raises(ValidationError, match="trusted=True"):
        joblib_load_trusted(path, trusted=trusted)
    loader.assert_not_called()


def test_formatted_provider_traceback_does_not_echo_secret(monkeypatch):
    secret = "offline-example-credential"
    create = Mock(side_effect=RuntimeError(f"Authentication failed: {secret}"))
    client = NS(chat=NS(completions=NS(create=create)))
    monkeypatch.setattr("buildml.ai.extras.require_openai", lambda **kwargs: NS(OpenAI=lambda **kwargs: client))
    provider = OpenAIProvider(ProviderConfig(api_key=secret))
    with pytest.raises(ValidationError) as error:
        provider.chat([Message(role="user", content="hello")])
    formatted = "".join(traceback.format_exception(error.value))
    assert secret not in formatted
    assert "REDACTED" in formatted


@pytest.mark.parametrize("trusted", ["false", "true", 1])
def test_serving_config_cannot_coerce_nonboolean_permission_to_true(monkeypatch, trusted):
    builder = Mock(side_effect=AssertionError("Serving reached artifact construction"))
    monkeypatch.setattr("buildml.serving.launch.create_serving_app", builder)
    with pytest.raises(ValidationError, match="trusted"):
        serve_bundle(config=ServeConfig(bundle="unread-fixture", trusted=False), trusted=trusted)
    builder.assert_not_called()


@pytest.mark.parametrize("trusted", ["false", "true", 1, None, np.bool_(True)])
def test_app_builder_rejects_nonboolean_permission_before_loading(monkeypatch, trusted):
    from buildml.serving.app import create_serving_app

    dependency = Mock(side_effect=AssertionError("Dependencies loaded before permission validation"))
    monkeypatch.setattr("buildml.serving.app._require_fastapi", dependency)
    with pytest.raises(ValidationError, match="trusted must be a boolean"):
        create_serving_app("unused", trusted=trusted)
    dependency.assert_not_called()


@pytest.mark.parametrize("permission", ["false", "true", 1, None])
def test_public_bind_override_rejects_nonboolean_permission(monkeypatch, permission):
    builder = Mock(side_effect=AssertionError("App built before permission validation"))
    monkeypatch.setattr("buildml.serving.launch.create_serving_app", builder)
    with pytest.raises(ValidationError, match="allow_insecure_public_bind must be a boolean"):
        serve_bundle("unused", allow_insecure_public_bind=permission)
    builder.assert_not_called()


@pytest.mark.parametrize("value,expected", [("false", False), ("true", True), (False, False), (True, True)])
def test_declarative_config_normalizes_permission_strings(value, expected):
    config = ServeConfig(bundle="unused", trusted=value, allow_insecure_public_bind=value)
    assert config.trusted is expected
    assert config.allow_insecure_public_bind is expected
