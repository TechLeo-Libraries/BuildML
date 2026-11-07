"""Serving config must preserve precedence and reject ambiguous bind options."""
import json

import pytest

from buildml.core.errors import ValidationError
from buildml.serving.config import ServeConfig


@pytest.mark.parametrize("port", [True, False, 8080.5, float("nan"), float("inf"),
                                  None, "not-a-port", "8080.5", 0, 65536])
@pytest.mark.parametrize("loader", [ServeConfig.from_mapping, lambda options: ServeConfig(**options)])
def test_invalid_port_is_never_coerced_into_another_endpoint(port, loader):
    if port is None and loader == ServeConfig.from_mapping:
        assert loader({"port": None}).port == 8080
        return
    with pytest.raises(ValidationError, match="port must"):
        loader({"port": port})


@pytest.mark.parametrize("value", ["maybe", "", 2, -1, None, [], {}])
@pytest.mark.parametrize("field", ["trusted", "docs_enabled", "allow_insecure_public_bind"])
def test_invalid_boolean_configuration_rejected(field, value):
    if value is None:
        expected = None if field == "docs_enabled" else False
        assert getattr(ServeConfig.from_mapping({field: value}), field) is expected
    else:
        with pytest.raises(ValidationError, match="boolean"):
            ServeConfig.from_mapping({field: value})


@pytest.mark.parametrize("value", ["missing-colon", ":password", ["user"], ["", "pass"],
                                   {"username": "user"}, {"password": "pass"},
                                   {"username": " ", "password": "pass"}, 12])
def test_invalid_basic_auth_config_is_not_treated_as_public(value):
    with pytest.raises(ValidationError):
        ServeConfig.from_mapping({"basic_auth": value})


@pytest.mark.parametrize("value", ["élise:秘密", ["élise", "秘密"],
                                   {"username": "élise", "password": "秘密"}])
def test_valid_basic_auth_config_is_closed_and_redacted(value):
    config = ServeConfig.from_mapping({"basic_auth": value, "bundle": "model", "api_key": "key"})
    assert config.basic_auth == ("élise", "秘密")
    assert config.auth_enabled
    assert not config.resolved_docs_enabled()
    assert config.launch_kwargs()["basic_auth"] == ("élise", "秘密")
    encoded = json.dumps(config.to_dict(), ensure_ascii=False)
    assert "秘密" not in encoded
    assert '"key"' not in encoded


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_config_file_environment_and_cli_precedence(suffix, tmp_path):
    path = tmp_path / f"serve{suffix}"
    body = {"bundle": "file-model", "port": 8123, "api_keys": ["file-key"], "trusted": False}
    path.write_text(json.dumps(body))  # JSON is also valid YAML.
    env = {"BUILDML_SERVE_CONFIG": str(path), "BUILDML_BUNDLE": "environment-model",
           "BUILDML_SERVE_BUNDLE": "lower-priority-model", "BUILDML_API_KEY": "environment-key",
           "BUILDML_SERVE_API_KEY": "lower-priority-key", "BUILDML_SERVE_TRUSTED": "yes",
           "BUILDML_SERVE_PORT": "8124", "BUILDML_SERVE_DOCS_ENABLED": "off",
           "BUILDML_SERVE_BASIC_USER": "alice", "BUILDML_SERVE_BASIC_PASSWORD": "secret"}
    config = ServeConfig.load(environ=env, cli={"port": 8125, "api_keys": [], "title": None})
    assert config.require_bundle() == "environment-model"
    assert config.port == 8125
    assert config.api_keys == ("environment-key",)
    assert config.basic_auth == ("alice", "secret")
    assert config.trusted is True
    assert config.docs_enabled is False
    assert body["port"] == 8123
    assert env["BUILDML_SERVE_PORT"] == "8124"


@pytest.mark.parametrize("body", ["[]", "42", '"text"'])
@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_non_mapping_configuration_file_rejected(body, suffix, tmp_path):
    path = tmp_path / f"serve{suffix}"
    path.write_text(body)
    with pytest.raises(ValidationError):
        ServeConfig.from_yaml(path)


def test_empty_yaml_defaults_unknown_fields_and_required_bundle(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    config = ServeConfig.from_yaml(path)
    assert config.port == 8080
    with pytest.raises(ValidationError, match="bundle"):
        config.require_bundle()
    with pytest.raises(ValidationError, match="Unknown"):
        config.merge(unknown_option=True)
    with pytest.raises(ValidationError, match="Unknown"):
        ServeConfig.from_mapping({"unknown_option": True})
