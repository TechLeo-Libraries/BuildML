#!/usr/bin/env python3
"""Add trusted=True to test load call sites."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tests"

SESSION_METHODS = [
    "load_anomaly_bundle",
    "load_pipeline",
    "load_model",
    "load_automl_bundle",
    "load_forecast_bundle",
    "load_ranker_bundle",
    "load_recommender_bundle",
    "load_ensemble_bundle",
    "load_nlp_bundle",
    "load_causal_bundle",
    "load_unsupervised_bundle",
    "load_semisupervised_bundle",
    "load_tda_bundle",
    "load_symbolic_bundle",
    "load_synthetic_bundle",
    "load_online_bundle",
    "load_federated_bundle",
    "load_graph_bundle",
    "load_kg_bundle",
    "load_metalearning_bundle",
    "load_multitask_bundle",
    "load_probabilistic_bundle",
    "load_decision_bundle",
    "load_cbr_bundle",
    "load_ssl_bundle",
    "load_rl_bundle",
    "load_imitation_bundle",
    "load_active_learning_bundle",
    "load_torch_bundle",
    "reattach",
]

DIRECT_FUNCS = [
    "load_pipeline_bundle",
    "load_fit_result",
    "load_anomaly_bundle",
    "load_checkpoint",
    "load_ensemble_bundle",
    "load_forecast_bundle",
    "load_unsupervised_bundle",
    "load_ranker_bundle",
    "load_recommender_bundle",
    "load_nlp_bundle",
    "load_torch_bundle",
    "load_causal_bundle",
    "load_automl_bundle",
    "load_ssl_bundle",
    "load_tda_bundle",
    "load_symbolic_bundle",
    "load_synthetic_bundle",
    "load_online_bundle",
    "load_federated_bundle",
    "load_graph_bundle",
    "load_kg_bundle",
    "load_metalearning_bundle",
    "load_multitask_bundle",
    "load_probabilistic_bundle",
    "load_decision_bundle",
    "load_cbr_bundle",
    "load_rl_bundle",
    "load_imitation_bundle",
    "load_active_learning_bundle",
    "load_semisupervised_bundle",
]


def _inject(call: str, args: str) -> str:
    if "trusted=" in args:
        return call
    if not args.strip():
        return call[:-1] + "trusted=True)"
    return call[:-1] + ", trusted=True)"


def patch(src: str) -> str:
    def repl_checkpoint(m: re.Match[str]) -> str:
        full = m.group(0)
        if "trusted=" in full:
            return full
        return f"Session.checkpoint_load({m.group(1)}, trusted=True)"

    src = re.sub(r"Session\.checkpoint_load\(([^)\n]+)\)", repl_checkpoint, src)

    for meth in SESSION_METHODS:
        pattern = re.compile(rf"\.{meth}\(([^)\n]*)\)")

        def repl(m: re.Match[str], name: str = meth) -> str:
            args = m.group(1)
            if "trusted=" in args:
                return m.group(0)
            if not args.strip():
                return f".{name}(trusted=True)"
            return f".{name}({args}, trusted=True)"

        src = pattern.sub(repl, src)

    for fn in DIRECT_FUNCS:
        pattern = re.compile(rf"(?<![\w.]){fn}\(([^)\n]*)\)")

        def repl(m: re.Match[str], name: str = fn) -> str:
            args = m.group(1)
            if "trusted=" in args:
                return m.group(0)
            if not args.strip():
                return f"{name}(trusted=True)"
            return f"{name}({args}, trusted=True)"

        src = pattern.sub(repl, src)

    src = src.replace(", trusted=True, trusted=True", ", trusted=True")
    return src


def main() -> int:
    changed = 0
    for path in sorted(ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        new = patch(text)
        if new != text:
            path.write_text(new, encoding="utf-8", newline="\n")
            changed += 1
            print(path)
    print(f"updated {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
