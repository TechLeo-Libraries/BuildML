"""Authentication parsing, credential rotation and failure response contracts."""
import base64

import pytest

from buildml.core.errors import ValidationError
from buildml.serving.auth import (
    APIKeyAuthMiddleware,
    ServingAuthMiddleware,
    extract_basic_credentials,
    normalize_basic_credentials,
    request_is_authorized,
)


@pytest.mark.parametrize("value", [
    "alice:pa:ss", ("alice", "pa:ss"), {"username": "alice", "password": "pa:ss"},
    {"user": "alice", "pass": "pa:ss"}, [("alice", "pa:ss")],
])
def test_equivalent_basic_configurations_preserve_password(value):
    assert normalize_basic_credentials(value) == frozenset({("alice", "pa:ss")})


@pytest.mark.parametrize("value", [
    "", "missing-colon", ":password", {}, {"user": "alice"},
    {"password": "secret"}, {"username": " ", "password": "secret"},
    [], [("alice",)], [("", "secret")], ["invalid"], 17,
])
def test_invalid_basic_configuration_rejected(value):
    with pytest.raises(ValidationError):
        normalize_basic_credentials(value)


@pytest.mark.parametrize("raw", [None, "", "Bearer wrong", "Basic", "Basic !!!",
                                     "Basic /w==", "Basic YWxpY2U=", "Basic OnNlY3JldA=="])
def test_malformed_credentials_are_unauthorized(raw):
    assert extract_basic_credentials(raw) is None
    assert not request_is_authorized(authorization=raw, api_key_header=None,
                                     api_keys=("secret",), basic_credentials=(("alice", "secret"),))


@pytest.mark.parametrize("username,password", [("alice", "pa:ss"), ("élise", "秘密")])
def test_utf8_basic_credentials_and_rotation(username, password):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    presented = f"Basic {token}"
    assert extract_basic_credentials(presented) == (username, password)
    assert request_is_authorized(authorization=presented, api_key_header="wrong",
                                 api_keys=("old", "new"),
                                 basic_credentials=(("retired", "bad"), (username, password)))
    assert not request_is_authorized(authorization=presented, api_key_header=None,
                                     api_keys=(), basic_credentials=((username, "wrong"),))


def test_api_key_precedence_and_unicode_configuration():
    options = {"api_keys": ("retired", "κλειδί"), "basic_credentials": ()}
    assert request_is_authorized(authorization="Bearer κλειδί", api_key_header=None, **options)
    assert request_is_authorized(authorization="Bearer wrong", api_key_header="κλειδί", **options)
    assert not request_is_authorized(authorization="Bearer κλειδί", api_key_header="wrong", **options)


@pytest.mark.parametrize("keys,basic,challenge", [
    (frozenset({"secret"}), frozenset(), "Bearer"),
    (frozenset(), frozenset({("élise", "秘密")}), 'Basic realm="buildml-serve"'),
    (frozenset({"secret"}), frozenset({("élise", "秘密")}), 'Bearer, Basic realm="buildml-serve"'),
])
def test_real_middleware_enforces_auth_and_reports_supported_schemes(keys, basic, challenge):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = fastapi.FastAPI()
    app.add_middleware(ServingAuthMiddleware, api_keys=keys, basic_credentials=basic)

    @app.get("/private")
    def private():
        return {"allowed": True}

    @app.get("/health")
    def health():
        return {"ok": True}

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        rejected = client.get("/private", headers={"Authorization": "Basic !!!"})
        assert rejected.status_code == 401
        assert rejected.headers["www-authenticate"] == challenge
        if keys:
            assert client.get("/private", headers={"X-API-Key": "secret"}).json() == {"allowed": True}
        if basic:
            token = base64.b64encode("élise:秘密".encode()).decode()
            assert client.get("/private", headers={"Authorization": f"Basic {token}"}).json() == {"allowed": True}


def test_legacy_middleware_open_paths_and_empty_credentials():
    with pytest.raises(ValidationError, match="requires"):
        ServingAuthMiddleware(lambda: None)
    middleware = APIKeyAuthMiddleware(lambda: None, api_keys=frozenset({"secret"}))
    assert "/docs" in middleware.open_paths
    closed = APIKeyAuthMiddleware(lambda: None, api_keys=frozenset({"secret"}), open_paths=frozenset())
    assert not closed.open_paths
