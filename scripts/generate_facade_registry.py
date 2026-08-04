"""Generate ``buildml/session/facade_registry.py`` from bindings JSON."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDINGS_PATH = Path(__file__).resolve().parent / "_facade_bindings.json"
OUT_PATH = ROOT / "buildml" / "session" / "facade_registry.py"

HELPERS = r'''
_PROPERTY_FACADE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "plan",
        "result",
        "transcript",
        "assumptions",
        "spec",
        "last_report",
        "history",
        "last_dry_run",
        "last_summary",
        "last_walkthrough",
        "backbone",
        "backbone_head",
        "asr_eval",
        "speech_result",
        "train_result",
        "cv_result",
        "search_result",
        "nested_cv_result",
        "export_result",
        "ddp_result",
        "text_plan",
        "topic_plan",
        "head_plan",
        "neuro_plan",
        "imitation_plan",
        "imitation_fit_result",
        "imitation_eval_result",
        "imitation_predict_result",
        "analysis_result",
    }
)


def preferred_path(flat_name: str) -> str | None:
    """Map a flat Session method name to its preferred facade path.

    Used by discovery helpers and deprecation warnings so teaching and runtime
    point at the same ``session.<domain>.<method>`` spelling.

    Parameters
    ----------
    flat_name:
        Public flat method such as ``evaluate_fairness`` or ``fit``.

    Returns
    -------
    str or None
        ``session.<domain>.<method>`` when registered, otherwise ``None``.
    """
    for attr, spec in DOMAIN_FACADES.items():
        for facade_name, flat in spec["bindings"].items():
            if flat == flat_name:
                return f"session.{attr}.{facade_name}"
    return None


def resolve_operation_name(flat_or_facade: str) -> str:
    """Normalize flat or facade-style names to the canonical flat Session method.

    Accepts ``evaluate_fairness``, ``fairness.evaluate``, and
    ``session.fairness.evaluate``. Unrecognized spellings are returned cleaned
    so callers keep their existing unknown-name errors. Catalog keys remain
    flat; this is the dual-form input boundary for explain / AI / discovery.

    Parameters
    ----------
    flat_or_facade:
        Flat Session method or ``domain.method`` / ``session.domain.method``.

    Returns
    -------
    str
        Canonical flat method name when a facade path is recognized; otherwise
        the stripped input.
    """
    cleaned = str(flat_or_facade or "").strip()
    if not cleaned:
        return cleaned
    path = cleaned
    if path.startswith("session."):
        path = path[len("session.") :]
    if "." not in path:
        return cleaned
    attr, method = path.split(".", 1)
    spec = DOMAIN_FACADES.get(attr)
    if spec is None or method not in spec["bindings"]:
        return cleaned
    return str(spec["bindings"][method])


def flat_to_facade() -> dict[str, dict[str, Any]]:
    """Build the reverse index from flat Session methods to facade metadata.

    One lookup table for tier, warn policy, and preferred path for every flat
    Session member covered by the facade registry.

    Returns
    -------
    dict[str, dict[str, Any]]
        Flat method name → facade attr/method/tier/warn metadata.
    """
    out: dict[str, dict[str, Any]] = {}
    for attr, spec in DOMAIN_FACADES.items():
        for facade_name, flat in spec["bindings"].items():
            out[flat] = {
                "facade_attr": attr,
                "facade_method": facade_name,
                "preferred_path": f"session.{attr}.{facade_name}",
                "tier": spec["tier"],
                "warn_flat": spec["warn_flat"],
                "mixin_key": spec["mixin_key"],
            }
    return out


def _is_property_like_facade(facade_name: str) -> bool:
    if facade_name in _PROPERTY_FACADE_NAMES:
        return True
    if facade_name.endswith("_result") or facade_name.endswith("_plan"):
        return True
    return False


DEPRECATED_FLAT_ACTIONS: Final[frozenset[str]] = frozenset(
    flat
    for attr, spec in DOMAIN_FACADES.items()
    if spec["warn_flat"]
    for facade_name, flat in spec["bindings"].items()
    if not _is_property_like_facade(facade_name)
)


__all__ = [
    "DEPRECATED_FLAT_ACTIONS",
    "DOMAIN_FACADES",
    "DomainFacadeSpec",
    "flat_to_facade",
    "preferred_path",
    "resolve_operation_name",
]
'''


def main() -> None:
    domains = json.loads(BINDINGS_PATH.read_text(encoding="utf-8"))
    lines = [
        '"""Declarative Session namespaced-facade registry.',
        "",
        "Each domain exposes ``session.<attr>.*`` bindings that delegate to",
        "existing flat Session methods. Regenerate with",
        "``python scripts/generate_facade_registry.py``.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any, Final, TypedDict",
        "",
        "",
        "class DomainFacadeSpec(TypedDict):",
        '    """Typed registry row for one namespaced Session facade."""',
        "",
        "    mixin_key: str",
        "    tier: str  # core | domain | experimental",
        "    warn_flat: bool",
        "    bindings: dict[str, str]  # facade_name -> flat Session method",
        "",
        "",
        "DOMAIN_FACADES: Final[dict[str, DomainFacadeSpec]] = {",
    ]
    for attr in sorted(domains):
        spec = domains[attr]
        lines.append(f"    {attr!r}: {{")
        lines.append(f"        \"mixin_key\": {spec['mixin_key']!r},")
        lines.append(f"        \"tier\": {spec['tier']!r},")
        lines.append(f"        \"warn_flat\": {spec['warn_flat']!r},")
        lines.append("        \"bindings\": {")
        for facade_name, flat_name in sorted(
            spec["bindings"].items(), key=lambda kv: kv[0]
        ):
            lines.append(f"            {facade_name!r}: {flat_name!r},")
        lines.append("        },")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    OUT_PATH.write_text("\n".join(lines) + HELPERS, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(domains)} facades)")


if __name__ == "__main__":
    main()
