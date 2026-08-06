"""Namespaced Session facades (``session.fairness.evaluate``, …).

Facades are additive views over existing flat Session methods. They do not
remove or rename the flat surface. Domain flat *actions* emit
``DeprecationWarning`` pointing at the preferred facade path; classical core
(``data`` / ``preprocess`` / ``classical`` / ``explore`` / ``audit``) stays
dual first-class without warnings.
"""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from types import MethodType
from typing import Any, cast

from buildml.session.facade_registry import (
    DEPRECATED_FLAT_ACTIONS,
    DOMAIN_FACADES,
    flat_to_facade,
    preferred_path,
)

_SHADOW_WARNED: set[int] = set()

_WARN_STACKLEVEL = 3
_ORIGINAL_ATTR = "_buildml_flat_original"
_DEPRECATED_FLAG = "_buildml_flat_deprecated"


class DomainFacade:
    """Bound view of one Session domain namespace."""

    __slots__ = ("_session", "_attr", "_spec")

    def __init__(self, session: Any, attr: str) -> None:
        """Bind this facade to a live Session and registry attribute name.

        The facade does not copy Session state; attribute access always
        delegates to the underlying flat methods on ``session``.

        Parameters
        ----------
        session:
            Live ``Session`` instance that owns the flat methods.
        attr:
            Facade attribute key from ``DOMAIN_FACADES`` (e.g. ``fairness``).
        """
        self._session = session
        self._attr = attr
        self._spec = DOMAIN_FACADES[attr]

    @property
    def domain(self) -> str:
        """Short domain key (matches capability / maturity naming)."""
        return str(self._spec["mixin_key"])

    @property
    def tier(self) -> str:
        """Stability tier: ``core``, ``domain``, or ``experimental``."""
        return str(self._spec["tier"])

    @property
    def preferred_prefix(self) -> str:
        """Human path prefix, e.g. ``session.fairness``."""
        return f"session.{self._attr}"

    def __repr__(self) -> str:
        return (
            f"<Session.{self._attr} facade "
            f"tier={self.tier!r} methods={len(self._spec['bindings'])}>"
        )

    def __dir__(self) -> list[str]:
        return sorted(self._spec["bindings"])

    def __getattr__(self, name: str) -> Any:
        flat = self._spec["bindings"].get(name)
        if flat is None:
            raise AttributeError(
                f"Session.{self._attr} has no attribute {name!r}. "
                f"Known: {', '.join(sorted(self._spec['bindings']))}"
            )
        return _resolve_flat_unwarned(self._session, flat)

    def describe(self) -> dict[str, Any]:
        """Return machine-readable bindings for this facade namespace.

        Use this when building UIs or docs that need the flat↔facade map for one
        domain without loading every capability matrix.

        Returns
        -------
        dict[str, Any]
            Facade attribute, tier, warn policy, and per-method preferred paths.
        """
        return {
            "facade": self._attr,
            "domain": self.domain,
            "tier": self.tier,
            "preferred_prefix": self.preferred_prefix,
            "warn_flat": bool(self._spec["warn_flat"]),
            "methods": {
                facade: {
                    "flat": flat,
                    "preferred_path": f"{self.preferred_prefix}.{facade}",
                }
                for facade, flat in sorted(self._spec["bindings"].items())
            },
        }


def _warn_domain_variable_shadow(session: Any, facade_attr: str) -> None:
    """Warn once when a Session is bound to a domain-named local variable.

    ``rag = Session(); rag.rag.retrieve(...)`` is awkward; prefer
    ``session = Session(); session.rag.retrieve(...)``.
    """
    sid = id(session)
    if sid in _SHADOW_WARNED:
        return
    try:
        frame = inspect.currentframe()
        if frame is None:
            return
        caller = frame.f_back.f_back if frame.f_back is not None else None
        if caller is None:
            return
        for name, value in caller.f_locals.items():
            if value is session and name in DOMAIN_FACADES:
                _SHADOW_WARNED.add(sid)
                warnings.warn(
                    f"Session instance is bound to local name {name!r}, which "
                    f"collides with the namespaced facade session.{name}. "
                    f"Prefer session = Session() (or another non-domain name) "
                    f"so calls read session.{facade_attr}.* instead of "
                    f"{name}.{facade_attr}.*. See docs/session-facade-migration.md.",
                    UserWarning,
                    stacklevel=4,
                )
                return
    except Exception:  # noqa: BLE001 — best-effort DX warning only
        return


class _FacadeProperty:
    """Instance-cached ``session.<domain>`` descriptor."""

    __slots__ = ("_attr",)

    def __init__(self, attr: str) -> None:
        """Remember which facade attribute this descriptor serves.

        Instances are created once per facade attr at Session class
        installation time and cache ``DomainFacade`` objects per Session.

        Parameters
        ----------
        attr:
            Facade attribute key installed on ``Session``.
        """
        self._attr = attr

    def __get__(
        self, obj: Any, owner: type | None = None
    ) -> DomainFacade | _FacadeProperty:
        if obj is None:
            return self
        _warn_domain_variable_shadow(obj, self._attr)
        cache = obj.__dict__.setdefault("_domain_facades", {})
        facade = cache.get(self._attr)
        if facade is None:
            facade = DomainFacade(obj, self._attr)
            cache[self._attr] = facade
        return cast(DomainFacade, facade)


def _find_class_attr(session_cls: type, name: str) -> tuple[type, Any] | None:
    for klass in session_cls.__mro__:
        if name in klass.__dict__:
            return klass, klass.__dict__[name]
    return None


def _resolve_flat_unwarned(session: Any, flat_name: str) -> Any:
    """Fetch a flat Session attribute without emitting deprecation warnings."""
    cls = type(session)
    found = _find_class_attr(cls, flat_name)
    if found is None:
        return getattr(session, flat_name)
    _owner, raw = found

    original = getattr(raw, _ORIGINAL_ATTR, None)
    if original is not None:
        raw = original
    elif isinstance(raw, (staticmethod, classmethod)):
        fn = raw.__func__
        original_fn = getattr(fn, _ORIGINAL_ATTR, None)
        if original_fn is not None:
            raw = type(raw)(original_fn)

    if isinstance(raw, property):
        return raw.__get__(session, cls)
    if isinstance(raw, staticmethod):
        return raw.__get__(session, cls)
    if isinstance(raw, classmethod):
        return raw.__get__(session, cls)
    if callable(raw):
        return MethodType(raw, session)
    return getattr(session, flat_name)


def warn_deprecated_flat(flat_name: str, *, stacklevel: int = _WARN_STACKLEVEL) -> None:
    """Emit a DeprecationWarning that points callers at the preferred facade.

    No-op when ``flat_name`` is not registered, so classical dual methods can
    share call sites safely.

    Parameters
    ----------
    flat_name:
        Flat Session method name that was invoked (e.g. ``evaluate_fairness``).
    stacklevel:
        ``warnings.warn`` stack level so the warning points at user code.
    """
    path = preferred_path(flat_name)
    if path is None:
        return
    warnings.warn(
        f"Session.{flat_name}() is deprecated for new code; prefer {path}(...). "
        "Flat aliases remain until BuildML 3.0. "
        "See docs/session-facade-migration.md.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def _wrap_action(flat_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def warning_wrapper(*args: Any, **kwargs: Any) -> Any:
        warn_deprecated_flat(flat_name, stacklevel=_WARN_STACKLEVEL)
        return original(*args, **kwargs)

    setattr(warning_wrapper, _ORIGINAL_ATTR, original)
    setattr(warning_wrapper, _DEPRECATED_FLAG, True)
    return warning_wrapper


def install_facade_properties(session_cls: type) -> None:
    """Attach ``session.<domain>`` properties for every registered facade.

    Refuses to overwrite an existing callable/property under the same name so
    teaching sync keeps seeing flat ``eda`` / ``workflow`` methods.

    Parameters
    ----------
    session_cls:
        The ``Session`` class (or test double) receiving facade descriptors.

    Raises
    ------
    RuntimeError
        When a facade attribute name collides with an existing Session member.
    """
    for attr in DOMAIN_FACADES:
        found = _find_class_attr(session_cls, attr)
        if found is not None:
            _owner, raw = found
            if isinstance(raw, _FacadeProperty):
                continue
            if callable(raw) or isinstance(raw, (property, staticmethod, classmethod)):
                raise RuntimeError(
                    f"Facade attribute {attr!r} collides with an existing Session "
                    f"member on {_owner.__name__}. Rename the facade attr in "
                    "scripts/_facade_bindings.json and regenerate."
                )
        setattr(session_cls, attr, _FacadeProperty(attr))


def install_flat_deprecations(session_cls: type) -> None:
    """Wrap deprecated flat domain actions with DeprecationWarning.

    Result/plan properties are left alone to avoid warning noise; action
    methods and capability-matrix staticmethods are wrapped once.

    Parameters
    ----------
    session_cls:
        The ``Session`` class whose mixin-owned flat actions should warn.
    """
    mapping = flat_to_facade()
    for flat_name in sorted(DEPRECATED_FLAT_ACTIONS):
        meta = mapping.get(flat_name)
        if meta is None or not meta["warn_flat"]:
            continue
        found = _find_class_attr(session_cls, flat_name)
        if found is None:
            continue
        owner, raw = found
        if getattr(raw, _DEPRECATED_FLAG, False):
            continue
        if isinstance(raw, property):
            continue
        if isinstance(raw, staticmethod):
            if getattr(raw.__func__, _DEPRECATED_FLAG, False):
                continue
            wrapped_fn = _wrap_action(flat_name, raw.__func__)
            static_wrapped: Any = staticmethod(wrapped_fn)
            setattr(static_wrapped, _ORIGINAL_ATTR, raw)
            setattr(static_wrapped, _DEPRECATED_FLAG, True)
            setattr(owner, flat_name, static_wrapped)
            continue
        if isinstance(raw, classmethod):
            if getattr(raw.__func__, _DEPRECATED_FLAG, False):
                continue
            wrapped_fn = _wrap_action(flat_name, raw.__func__)
            class_wrapped: Any = classmethod(wrapped_fn)
            setattr(class_wrapped, _ORIGINAL_ATTR, raw)
            setattr(class_wrapped, _DEPRECATED_FLAG, True)
            setattr(owner, flat_name, class_wrapped)
            continue
        if callable(raw):
            setattr(owner, flat_name, _wrap_action(flat_name, raw))


def install_session_facades(session_cls: type) -> None:
    """Install namespaced facades and flat-method deprecation wrappers.

    Called once after the ``Session`` class body is assembled from mixins.

    Parameters
    ----------
    session_cls:
        The composed ``Session`` class after mixins are attached.
    """
    install_facade_properties(session_cls)
    install_flat_deprecations(session_cls)


def list_facades() -> dict[str, Any]:
    """Catalog all namespaced facades for discovery UIs and docs.

    This is the machine-readable companion to
    ``docs/session-facade-migration.md``.

    Returns
    -------
    dict[str, Any]
        Facade count, per-attr tier/warn metadata, and policy disclosures.
    """
    return {
        "n_facades": len(DOMAIN_FACADES),
        "facades": [
            {
                "attr": attr,
                "domain": spec["mixin_key"],
                "tier": spec["tier"],
                "warn_flat": spec["warn_flat"],
                "preferred_prefix": f"session.{attr}",
                "n_methods": len(spec["bindings"]),
            }
            for attr, spec in sorted(DOMAIN_FACADES.items())
        ],
        "disclosures": (
            "Facades are additive; flat Session methods remain until BuildML 3.0. "
            "Surface size is organized via namespaces, not reduced; flat removal "
            "is a 3.0 policy decision, not unfinished 2.4 work.",
            "Classical core (data/preprocess/classical/explore/audit) is dual "
            "first-class without DeprecationWarning.",
            "Domain flat actions emit DeprecationWarning pointing at preferred paths.",
            "EDA facade is session.explore (session.eda remains the flat method). "
            "Workflow/teaching facade is session.audit (session.workflow remains).",
            "Avoid binding Session to a domain-named variable (rag = Session()); "
            "use session = Session() so calls read session.rag.* cleanly.",
        ),
    }


__all__ = [
    "DomainFacade",
    "install_session_facades",
    "list_facades",
    "preferred_path",
    "warn_deprecated_flat",
]
