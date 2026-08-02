"""Rule fidelity benchmark: symbolic rules vs black-box on tabular data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from buildml import Session
from buildml.symbolic.catalog import symbolic_capability_matrix
from buildml.symbolic.extras import imodels_available, skope_rules_available


def _reference_frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    y = ((x[:, 0] + 0.5 * x[:, 1] - 0.2 * x[:, 2]) > 0).astype(int)
    return pd.DataFrame(
        {f"x{i}": x[:, i] for i in range(4)} | {"y": y},
    )


def _blackbox_accuracy(session: Session, partition: str = "test") -> float:
    train_idx = list(session.split_plan.train_indices)
    frame = session.dataset._ensure_pandas()
    features = [c for c in frame.columns if c != "y"]
    x_train = frame.loc[train_idx, features].to_numpy()
    y_train = frame.loc[train_idx, "y"].to_numpy()
    clf = RandomForestClassifier(n_estimators=40, random_state=0, max_depth=6)
    clf.fit(x_train, y_train)
    if partition == "test":
        idx = list(session.split_plan.test_indices)
    elif partition == "validation":
        idx = list(session.split_plan.validation_indices)
    else:
        idx = train_idx
    x_hold = frame.loc[idx, features].to_numpy()
    y_hold = frame.loc[idx, "y"].to_numpy()
    return float(np.mean(clf.predict(x_hold) == y_hold))


def _run_symbolic(
    backend: str,
    *,
    source: str | None = None,
    method: str | None = None,
) -> dict[str, object]:
    session = (
        Session.ingest(_reference_frame())
        .set_roles({f"x{i}": "feature" for i in range(4)} | {"y": "target"})
        .split(test_size=0.2, validation_size=0.2, stratify=True, random_state=0)
        .scale(method="standard")
    )
    kwargs: dict[str, object] = {"backend": backend, "task": "classification"}
    if source is not None:
        kwargs["source"] = source
    if method is not None:
        kwargs["method"] = method
    fit = session.fit_symbolic(**kwargs)  # type: ignore[arg-type]
    ev = session.evaluate_symbolic(partition="test")
    bb = _blackbox_accuracy(session, partition="test")
    sym_acc = float(ev.metrics.get("accuracy", 0.0))
    fidelity = sym_acc / bb if bb > 0 else None
    return {
        "backend": backend,
        "source": fit.source,
        "method": fit.method,
        "n_rules": fit.n_rules,
        "provenance": fit.provenance,
        "test_accuracy": sym_acc,
        "blackbox_accuracy": bb,
        "fidelity_ratio": fidelity,
        "rule_coverage": ev.rule_coverage,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BuildML symbolic rule fidelity benchmark")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/symbolic/results/rule_fidelity.json"),
    )
    args = parser.parse_args(argv)

    runs: list[dict[str, object]] = []
    runs.append(_run_symbolic("sklearn", source="decision_tree"))
    runs.append(_run_symbolic("sklearn", source="decision_list"))

    # Real import probes — skip industry when skope/imodels are installed-but-broken.
    if skope_rules_available():
        try:
            runs.append(_run_symbolic("industry", method="skope_rules"))
        except Exception as exc:  # noqa: BLE001 — optional industry path
            runs.append(
                {
                    "backend": "industry",
                    "method": "skope_rules",
                    "skipped": True,
                    "error": str(exc),
                }
            )
    else:
        print(
            "skope-rules unavailable/unusable — skipping industry skope_rules "
            "(sklearn symbolic paths still run).",
            flush=True,
        )
    if imodels_available():
        try:
            runs.append(_run_symbolic("industry", method="rulefit"))
        except Exception as exc:  # noqa: BLE001
            runs.append(
                {
                    "backend": "industry",
                    "method": "rulefit",
                    "skipped": True,
                    "error": str(exc),
                }
            )

    payload = {
        "benchmark": "symbolic_rule_fidelity",
        "capability_matrix": symbolic_capability_matrix(),
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "n_runs": len(runs)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
