"""Port PR #10 unique files onto main as UTF-8 and apply 2.4.0 updates."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_show(path: str) -> str:
    raw = subprocess.check_output(
        ["git", "show", f"origin/hardening/safe-install-runtime-docs:{path}"],
        cwd=ROOT,
    )
    return raw.decode("utf-8")


def write_probe() -> None:
    text = git_show("scripts/verify_runtime_stability.py")
    old = (
        "report = session.evaluate_fairness("
        "sensitive_column='group', partition='test')"
    )
    new = (
        "report = session.fairness.evaluate("
        "sensitive_column='group', partition='test')"
    )
    if old not in text:
        raise SystemExit("fairness call not found in probe script")
    text = text.replace(old, new)
    out = ROOT / "scripts" / "verify_runtime_stability.py"
    out.write_text(text, encoding="utf-8", newline="\n")
    data = out.read_bytes()
    if b"\x00" in data:
        raise SystemExit("probe still contains null bytes")
    print(f"wrote {out} ({len(data)} bytes)")


def write_guide() -> None:
    text = git_show("guides/safe-install-and-runtime.md")
    pair = (
        "This guide pairs with `scripts/verify_runtime_stability.py` (subprocess\n"
        "use-case probes: `ok` / `fail` / `crash` / `skip`). That is different from\n"
        "`scripts/probe_industry_extras.py`, which only checks whether industry modules\n"
        "**import**.\n\n"
    )
    if "This guide pairs with" not in text:
        text = text.replace(
            "you rely on them.\n\n## Why staged",
            f"you rely on them.\n\n{pair}## Why staged",
        )
    text = text.replace(
        'pip install -e ".[dev,shap]"',
        'pip install "buildml[dev,shap]"\n'
        '# Or from a source checkout: pip install -e ".[dev,shap]"',
    )
    text = text.replace(
        "- Fairness (`evaluate_fairness`)",
        "- Fairness (`session.fairness.evaluate`)",
    )
    text = text.replace(
        "- SHAP (`explain_shap` via `buildml[shap]`)",
        "- SHAP (`session.explain_shap` via `buildml[shap]`)",
    )
    note = (
        "\nRun `verify_runtime_stability.py` from a BuildML source checkout "
        "(the script lives under `scripts/`).\n"
    )
    if "from a BuildML source checkout" not in text:
        text = text.replace(
            "\nVerify before adding optional native extras:\n",
            f"{note}\nVerify before adding optional native extras:\n",
        )
    out = ROOT / "guides" / "safe-install-and-runtime.md"
    out.write_text(text, encoding="utf-8", newline="\n")
    data = out.read_bytes()
    if b"\x00" in data:
        raise SystemExit("guide still contains null bytes")
    print(f"wrote {out} ({len(data)} bytes)")


if __name__ == "__main__":
    write_probe()
    write_guide()
