"""Declarative ServeConfig: YAML, environment, and CLI layering.

Precedence (later wins): defaults → YAML file → environment → CLI / kwargs.
When authentication is configured and ``docs_enabled`` is left unset, OpenAPI
docs default to closed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

from buildml.core.errors import MissingExtraError, ValidationError

BundleKind = Literal["pipeline", "torchscript"]

_ENV_HOST = "BUILDML_SERVE_HOST"
_ENV_PORT = "BUILDML_SERVE_PORT"
_ENV_BUNDLE = "BUILDML_BUNDLE"
_ENV_BUNDLE_ALT = "BUILDML_SERVE_BUNDLE"
_ENV_KIND = "BUILDML_SERVE_KIND"
_ENV_TITLE = "BUILDML_SERVE_TITLE"
_ENV_TRUSTED = "BUILDML_SERVE_TRUSTED"
_ENV_API_KEY = "BUILDML_API_KEY"
_ENV_API_KEY_ALT = "BUILDML_SERVE_API_KEY"
_ENV_BASIC = "BUILDML_SERVE_BASIC_AUTH"
_ENV_BASIC_USER = "BUILDML_SERVE_BASIC_USER"
_ENV_BASIC_PASSWORD = "BUILDML_SERVE_BASIC_PASSWORD"
_ENV_SSL_CERT = "BUILDML_SERVE_SSL_CERTFILE"
_ENV_SSL_KEY = "BUILDML_SERVE_SSL_KEYFILE"
_ENV_DOCS = "BUILDML_SERVE_DOCS_ENABLED"
_ENV_ALLOW_INSECURE = "BUILDML_SERVE_ALLOW_INSECURE_PUBLIC_BIND"
_ENV_MAP_LOCATION = "BUILDML_SERVE_MAP_LOCATION"
_ENV_CONFIG = "BUILDML_SERVE_CONFIG"


def _parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValidationError(f"{field_name} must be a boolean, got {value!r}")


def _parse_port(value: Any) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"port must be an integer, got {value!r}") from exc
    if port < 1 or port > 65535:
        raise ValidationError("port must be an integer in 1..65535")
    return port


def _parse_api_keys(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",")]
        return tuple(p for p in parts if p)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        cleaned = tuple(str(item).strip() for item in value if str(item).strip())
        return cleaned
    raise ValidationError("api_keys must be a string or list of strings")


def _parse_basic_auth(
    value: Any,
) -> tuple[str | None, str | None]:
    """Return ``(username, password)`` or ``(None, None)`` when unset."""
    if value is None:
        return None, None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None, None
        if ":" not in text:
            raise ValidationError(
                "basic_auth string must be 'username:password' (colon-separated)"
            )
        user, _, password = text.partition(":")
        if not user.strip():
            raise ValidationError("basic_auth username must be non-empty")
        return user.strip(), password
    if isinstance(value, Mapping):
        user = value.get("username", value.get("user"))
        password = value.get("password", value.get("pass"))
        if user is None and password is None:
            return None, None
        if user is None or password is None:
            raise ValidationError(
                "basic_auth mapping requires both username and password"
            )
        user_s = str(user).strip()
        if not user_s:
            raise ValidationError("basic_auth username must be non-empty")
        return user_s, str(password)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if len(items) != 2:
            raise ValidationError("basic_auth sequence must be [username, password]")
        user_s = str(items[0]).strip()
        if not user_s:
            raise ValidationError("basic_auth username must be non-empty")
        return user_s, str(items[1])
    raise ValidationError(
        "basic_auth must be 'user:pass', [user, pass], or {username, password}"
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - optional until serve extra
        raise MissingExtraError(
            "serve",
            "ServeConfig YAML loading (install PyYAML via buildml[serve])",
        ) from exc
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValidationError(f"ServeConfig YAML root must be a mapping: {path}")
    return dict(data)


@dataclass(slots=True)
class ServeConfig:
    """Declarative configuration for BuildML managed serving.

    Attributes
    ----------
    host, port:
        Bind target. Defaults to loopback ``127.0.0.1:8080``.
    bundle:
        Path to a pipeline bundle directory or TorchScript file.
    kind:
        ``'pipeline'`` or ``'torchscript'``.
    title:
        Display name for ``/health`` and OpenAPI (when docs are enabled).
    trusted:
        Required ``True`` to deserialize pickle/joblib/TorchScript artifacts.
    api_keys:
        Accepted API keys for Bearer / ``X-API-Key`` auth.
    basic_username, basic_password:
        Optional HTTP Basic credentials. Either API-key or Basic may authorize.
    ssl_certfile, ssl_keyfile:
        Optional local HTTPS PEM pair (both required together).
    docs_enabled:
        When ``None`` (default), OpenAPI/docs are closed automatically if any
        auth is configured; otherwise docs stay open for localhost development.
    allow_insecure_public_bind:
        Dangerous override to bind non-loopback without auth.
    map_location:
        Device placement for TorchScript loads.
    """

    host: str = "127.0.0.1"
    port: int = 8080
    bundle: str | None = None
    kind: BundleKind = "pipeline"
    title: str = "BuildML Serve"
    trusted: bool = False
    api_keys: tuple[str, ...] = ()
    basic_username: str | None = None
    basic_password: str | None = None
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    docs_enabled: bool | None = None
    allow_insecure_public_bind: bool = False
    map_location: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", str(self.host).strip() if self.host else "")
        object.__setattr__(self, "port", _parse_port(self.port))
        if self.bundle is not None:
            object.__setattr__(self, "bundle", str(self.bundle).strip() or None)
        kind = str(self.kind).strip().lower()
        if kind not in {"pipeline", "torchscript"}:
            raise ValidationError("kind must be 'pipeline' or 'torchscript'")
        object.__setattr__(self, "kind", kind)  # type: ignore[arg-type]
        object.__setattr__(self, "api_keys", _parse_api_keys(self.api_keys))
        if self.basic_username is not None:
            user = str(self.basic_username).strip()
            if not user:
                raise ValidationError("basic_username must be non-empty when set")
            object.__setattr__(self, "basic_username", user)
        if self.docs_enabled is not None and not isinstance(self.docs_enabled, bool):
            object.__setattr__(
                self, "docs_enabled", _parse_bool(self.docs_enabled, field_name="docs_enabled")
            )

    @property
    def basic_auth(self) -> tuple[str, str] | None:
        """Return ``(username, password)`` when Basic auth is fully configured."""
        if self.basic_username is None or self.basic_password is None:
            return None
        return (self.basic_username, self.basic_password)

    @property
    def auth_enabled(self) -> bool:
        """Whether any shared-secret auth mechanism is configured."""
        return bool(self.api_keys) or self.basic_auth is not None

    def resolved_docs_enabled(self) -> bool:
        """Resolve docs visibility: explicit flag, else closed when auth is on."""
        if self.docs_enabled is not None:
            return bool(self.docs_enabled)
        return not self.auth_enabled

    def require_bundle(self) -> str:
        """Return the bundle path or raise if unset."""
        if not self.bundle:
            raise ValidationError(
                "ServeConfig.bundle is required (pass --bundle, BUILDML_BUNDLE, "
                "or bundle: in YAML)"
            )
        return self.bundle

    def merge(self, **overrides: Any) -> ServeConfig:
        """Return a copy with non-``None`` overrides applied.

        Empty-string overrides for optional path fields clear them. ``api_keys``
        may be a string or sequence; ``basic_auth`` may be any form accepted by
        :func:`_parse_basic_auth`.
        """
        data = asdict(self)
        if "basic_auth" in overrides:
            user, password = _parse_basic_auth(overrides.pop("basic_auth"))
            if "basic_username" not in overrides:
                overrides["basic_username"] = user
            if "basic_password" not in overrides:
                overrides["basic_password"] = password
        for key, value in overrides.items():
            if key not in data:
                raise ValidationError(f"Unknown ServeConfig field: {key}")
            if value is None:
                continue
            data[key] = value
        return ServeConfig(**data)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view (passwords redacted)."""
        body = asdict(self)
        if body.get("basic_password") is not None:
            body["basic_password"] = "***REDACTED***"
        if body.get("api_keys"):
            body["api_keys"] = ["***REDACTED***"] * len(body["api_keys"])
        body["auth_enabled"] = self.auth_enabled
        body["docs_enabled_resolved"] = self.resolved_docs_enabled()
        return body

    def launch_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :func:`buildml.serving.launch.serve_bundle`."""
        return {
            "path": self.require_bundle(),
            "kind": self.kind,
            "host": self.host,
            "port": self.port,
            "title": self.title,
            "map_location": self.map_location,
            "api_keys": list(self.api_keys) if self.api_keys else None,
            "basic_auth": self.basic_auth,
            "docs_enabled": self.docs_enabled,
            "allow_insecure_public_bind": self.allow_insecure_public_bind,
            "ssl_certfile": self.ssl_certfile,
            "ssl_keyfile": self.ssl_keyfile,
            "trusted": self.trusted,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> ServeConfig:
        """Build a config from a plain mapping (YAML root or JSON object)."""
        if not data:
            return cls()
        raw = dict(data)
        # Accept nested basic_auth / api_key alias spellings.
        if "api_key" in raw and "api_keys" not in raw:
            raw["api_keys"] = raw.pop("api_key")
        if "basic_auth" in raw:
            user, password = _parse_basic_auth(raw.pop("basic_auth"))
            raw.setdefault("basic_username", user)
            raw.setdefault("basic_password", password)
        if "allow_insecure_bind" in raw and "allow_insecure_public_bind" not in raw:
            raw["allow_insecure_public_bind"] = raw.pop("allow_insecure_bind")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValidationError(
                f"Unknown ServeConfig field(s): {', '.join(unknown)}"
            )
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None:
                continue
            if key in {"trusted", "allow_insecure_public_bind"}:
                kwargs[key] = _parse_bool(value, field_name=key)
            elif key == "docs_enabled":
                kwargs[key] = _parse_bool(value, field_name=key)
            elif key == "port":
                kwargs[key] = _parse_port(value)
            elif key == "api_keys":
                kwargs[key] = _parse_api_keys(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ServeConfig:
        """Load ServeConfig from a YAML (or JSON-as-YAML) file."""
        root = Path(path)
        if not root.is_file():
            raise ValidationError(f"ServeConfig file not found: {root}")
        # JSON is valid YAML 1.2 for our purposes; also accept .json via yaml.
        if root.suffix.lower() == ".json":
            import json

            data = json.loads(root.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping):
                raise ValidationError(f"ServeConfig JSON root must be an object: {root}")
            return cls.from_mapping(data)
        return cls.from_mapping(_load_yaml_mapping(root))

    @classmethod
    def env_overrides(cls, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Return only environment keys that are explicitly set (for merging)."""
        env = os.environ if environ is None else environ
        data: dict[str, Any] = {}
        if _ENV_HOST in env and str(env[_ENV_HOST]).strip():
            data["host"] = env[_ENV_HOST]
        if _ENV_PORT in env and str(env[_ENV_PORT]).strip():
            data["port"] = env[_ENV_PORT]
        if _ENV_BUNDLE in env and str(env[_ENV_BUNDLE]).strip():
            data["bundle"] = env[_ENV_BUNDLE]
        elif _ENV_BUNDLE_ALT in env and str(env[_ENV_BUNDLE_ALT]).strip():
            data["bundle"] = env[_ENV_BUNDLE_ALT]
        if _ENV_KIND in env and str(env[_ENV_KIND]).strip():
            data["kind"] = env[_ENV_KIND]
        if _ENV_TITLE in env and str(env[_ENV_TITLE]).strip():
            data["title"] = env[_ENV_TITLE]
        if _ENV_TRUSTED in env and str(env[_ENV_TRUSTED]).strip():
            data["trusted"] = env[_ENV_TRUSTED]
        if _ENV_API_KEY in env and str(env[_ENV_API_KEY]).strip():
            data["api_keys"] = env[_ENV_API_KEY]
        elif _ENV_API_KEY_ALT in env and str(env[_ENV_API_KEY_ALT]).strip():
            data["api_keys"] = env[_ENV_API_KEY_ALT]
        if _ENV_BASIC in env and str(env[_ENV_BASIC]).strip():
            data["basic_auth"] = env[_ENV_BASIC]
        elif (_ENV_BASIC_USER in env and str(env[_ENV_BASIC_USER]).strip()) or (
            _ENV_BASIC_PASSWORD in env
        ):
            data["basic_auth"] = {
                "username": env.get(_ENV_BASIC_USER, ""),
                "password": env.get(_ENV_BASIC_PASSWORD, ""),
            }
        if _ENV_SSL_CERT in env and str(env[_ENV_SSL_CERT]).strip():
            data["ssl_certfile"] = env[_ENV_SSL_CERT]
        if _ENV_SSL_KEY in env and str(env[_ENV_SSL_KEY]).strip():
            data["ssl_keyfile"] = env[_ENV_SSL_KEY]
        if _ENV_DOCS in env and str(env[_ENV_DOCS]).strip():
            data["docs_enabled"] = env[_ENV_DOCS]
        if _ENV_ALLOW_INSECURE in env and str(env[_ENV_ALLOW_INSECURE]).strip():
            data["allow_insecure_public_bind"] = env[_ENV_ALLOW_INSECURE]
        if _ENV_MAP_LOCATION in env and str(env[_ENV_MAP_LOCATION]).strip():
            data["map_location"] = env[_ENV_MAP_LOCATION]
        return data

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServeConfig:
        """Load fields present in the environment (``BUILDML_*`` / ``BUILDML_SERVE_*``)."""
        return cls.from_mapping(cls.env_overrides(environ))

    @classmethod
    def load(
        cls,
        *,
        config_path: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        cli: Mapping[str, Any] | None = None,
    ) -> ServeConfig:
        """Compose defaults ← YAML ← env ← CLI.

        Parameters
        ----------
        config_path:
            Explicit YAML/JSON path. When omitted, ``BUILDML_SERVE_CONFIG`` is
            used if set.
        environ:
            Environment mapping (defaults to ``os.environ``).
        cli:
            CLI / programmatic overrides. ``None`` values are ignored so that
            argparse defaults do not clobber YAML/env.
        """
        env = os.environ if environ is None else environ
        resolved_path = config_path or env.get(_ENV_CONFIG) or None
        cfg = cls()
        if resolved_path:
            cfg = cls.from_yaml(resolved_path)
        env_data = cls.env_overrides(env)
        if env_data:
            cfg = cfg.merge(**env_data)
        if cli:
            cli_clean = {k: v for k, v in dict(cli).items() if v is not None}
            if "api_keys" in cli_clean and cli_clean["api_keys"] == []:
                del cli_clean["api_keys"]
            if cli_clean:
                cfg = cfg.merge(**cli_clean)
        return cfg


def serve_dockerfile_path() -> Path:
    """Return the first-party Dockerfile path under ``deploy/serve/``."""
    return Path(__file__).resolve().parents[2] / "deploy" / "serve" / "Dockerfile"


def serve_compose_example_path() -> Path:
    """Return the optional docker-compose example path under ``deploy/serve/``."""
    return (
        Path(__file__).resolve().parents[2]
        / "deploy"
        / "serve"
        / "docker-compose.example.yml"
    )
