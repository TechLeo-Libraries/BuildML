#!/usr/bin/env python3
"""Machine-checkable domain maturity index for BuildML.

Scores each claimed domain on artifact presence (catalog / Session matrix /
explain wiring / guides / proofs / tests) so uneven breadth is governed rather
than accidental. Industry wheel availability is *not* scored here — see
``scripts/probe_industry_extras.py`` and capability matrices.

Modes
-----
``--report``
    Print a table and JSON summary; always exits 0.

``--check``
    Fail (exit 1) when any domain marked ``claimed_complete`` in the registry
    is missing a required artifact, or falls below the domain-floor score.
    Used by CI as a ratchet.

``--json PATH``
    Write the full score payload to PATH (also works with ``--check``).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDML = ROOT / "buildml"
GUIDES = ROOT / "guides"
PROOFS = ROOT / "proofs"
TESTS = ROOT / "tests"

# Minimum artifact score (of 6 flags) for claimed-complete domains.
MIN_CLAIMED_SCORE = 6

# Domains that claim the "Solid domain shape" band in CONTRIBUTING.md.
# Missing required artifacts fail ``--check``.
CLAIMED_COMPLETE: dict[str, dict[str, str]] = {
    "anomaly": {"matrix": "anomaly_capability_matrix", "guide": "quickstart-anomaly"},
    "automl": {"matrix": "automl_capability_matrix", "guide": "quickstart-automl"},
    "causal": {"matrix": "causal_capability_matrix", "guide": "quickstart-causal"},
    "cbr": {"matrix": "cbr_capability_matrix", "guide": "quickstart-cbr"},
    "federated": {"matrix": "federated_capability_matrix", "guide": "quickstart-federated"},
    "forecasting": {"matrix": "forecast_capability_matrix", "guide": "quickstart-forecasting"},
    "graph": {"matrix": "graph_capability_matrix", "guide": "quickstart-graph"},
    "kg": {"matrix": "kg_capability_matrix", "guide": "quickstart-kg"},
    "metalearning": {
        "matrix": "metalearning_capability_matrix",
        "guide": "quickstart-meta-learning",
    },
    "multitask": {"matrix": "multitask_capability_matrix", "guide": "quickstart-multi-task"},
    "nlp": {"matrix": "nlp_capability_matrix", "guide": "quickstart-nlp"},
    "online": {"matrix": "online_capability_matrix", "guide": "quickstart-online-learning"},
    "probabilistic": {
        "matrix": "probabilistic_capability_matrix",
        "guide": "quickstart-probabilistic",
    },
    "ranking": {"matrix": "ranking_capability_matrix", "guide": "quickstart-ranking"},
    "recommenders": {
        "matrix": "recommender_capability_matrix",
        "guide": "quickstart-recommenders",
    },
    "rl": {"matrix": "rl_capability_matrix", "guide": "quickstart-imitation-rl"},
    "selfsupervised": {
        "matrix": "ssl_capability_matrix",
        "guide": "quickstart-selfsupervised",
        "proof_hint": "ssl",
    },
    "semisupervised": {
        "matrix": "semisupervised_capability_matrix",
        "guide": "quickstart-semisupervised",
    },
    "symbolic": {"matrix": "symbolic_capability_matrix", "guide": "quickstart-symbolic"},
    "synthetic": {"matrix": "synthetic_capability_matrix", "guide": "quickstart-synthetic"},
    "tda": {"matrix": "tda_capability_matrix", "guide": "quickstart-tda"},
    "unsupervised": {
        "matrix": "unsupervised_capability_matrix",
        "guide": "quickstart-unsupervised",
    },
    "activelearning": {
        "matrix": "activelearning_capability_matrix",
        "guide": "quickstart-active-learning",
    },
    "optimize": {"matrix": "decision_capability_matrix", "guide": "quickstart-optimize"},
    "ensemble": {"matrix": "ensemble_capability_matrix", "guide": "quickstart-ensemble"},
    "dl": {"matrix": "dl_capability_matrix", "guide": "quickstart-torch"},
    "rag": {"matrix": "rag_capability_matrix", "guide": "quickstart-rag"},
    "timeseries": {
        "matrix": "timeseries_capability_matrix",
        "guide": "quickstart-timeseries-analysis",
        "analysis_only": "1",
    },
}

# Primary-path domains without capability matrices — alternate checklist.
DEEP_NON_MATRIX: dict[str, dict[str, str]] = {
    "classical": {"guide": "quickstart-classical", "proof_hint": "classical"},
    "preprocess": {"guide": "preprocess-depth", "proof_hint": "preprocess"},
    "ai": {"guide": "quickstart-ai", "proof_hint": "ai"},
    "checkpoint": {"guide": "artifacts-checkpoints-bundles", "proof_hint": "checkpoint"},
    "explain": {"guide": "README", "proof_hint": "explain"},
}


@dataclass
class ArtifactScore:
    domain: str
    band: str
    claimed_complete: bool
    loc: int
    has_catalog: bool
    has_session_matrix: bool
    has_capability_status: bool
    has_guide: bool
    has_proof: bool
    has_test: bool
    has_explain_hooks: bool
    has_checkpoint: bool
    analysis_only: bool
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        flags = [
            self.has_catalog,
            self.has_session_matrix,
            self.has_capability_status,
            self.has_guide,
            self.has_proof,
            self.has_test,
        ]
        return sum(1 for f in flags if f)

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["score"] = self.score
        return payload


def _package_loc(pkg: Path) -> int:
    if not pkg.is_dir():
        return 0
    total = 0
    for path in pkg.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        total += sum(1 for _ in path.open(encoding="utf-8", errors="ignore"))
    return total


def _file_mentions(path: Path, needle: str) -> bool:
    if not path.is_file():
        return False
    try:
        return needle in path.read_text(encoding="utf-8")
    except OSError:
        return False


def _any_file_mentions(root: Path, needle: str, pattern: str = "**/*") -> bool:
    if not root.exists():
        return False
    for path in root.glob(pattern):
        if path.is_file() and needle in path.name:
            return True
        if path.is_file() and path.suffix in {".py", ".md", ".rst"}:
            if _file_mentions(path, needle):
                return True
    return False


def _session_has_method(method: str) -> bool:
    mixins = BUILDML / "session" / "mixins"
    if not mixins.is_dir():
        return False
    for path in mixins.glob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == method:
                return True
    return False


def _capability_status_mentions(method: str) -> bool:
    path = BUILDML / "explain" / "capability_status.py"
    return bool(method) and _file_mentions(path, method)


def score_domain(domain: str, meta: dict[str, str], *, claimed: bool) -> ArtifactScore:
    pkg = BUILDML / domain
    matrix = meta.get("matrix", "")
    guide_stem = meta.get("guide", f"quickstart-{domain}")
    proof_hint = meta.get("proof_hint", domain)
    analysis_only = meta.get("analysis_only") == "1"

    has_catalog = (pkg / "catalog.py").is_file() or any(
        path.name == "catalog.py" for path in pkg.rglob("catalog.py")
    )
    has_session = bool(matrix) and _session_has_method(matrix)
    has_status = bool(matrix) and _capability_status_mentions(matrix)
    has_guide = any(
        (GUIDES / f"{guide_stem}{suffix}").is_file()
        for suffix in (".md", ".rst")
    ) or any(GUIDES.glob(f"*{domain}*.md"))
    has_proof = _any_file_mentions(PROOFS, proof_hint, "**/*")
    has_test = _any_file_mentions(TESTS, domain, "**/*.py")
    has_explain_hooks = (pkg / "explain_hooks.py").is_file()
    has_checkpoint = (pkg / "checkpoint.py").is_file()

    missing: list[str] = []
    notes: list[str] = []
    if claimed:
        if not has_catalog:
            missing.append("catalog.py")
        if matrix and not has_session:
            missing.append(f"Session.{matrix}")
        if matrix and not has_status:
            missing.append("capability_status wiring")
        if not has_guide:
            missing.append("guides/quickstart")
        if not has_proof:
            missing.append("proofs/")
        if not has_test:
            missing.append("tests/")
        if not has_explain_hooks:
            missing.append("explain_hooks.py")
        if not has_checkpoint and not analysis_only:
            missing.append("checkpoint.py")
        if analysis_only:
            notes.append("analysis_only: checkpoint not required")

    band = "claimed_complete" if claimed else "deep_non_matrix"
    score_val = sum(
        1
        for f in (
            has_catalog,
            has_session if matrix else False,
            has_status if matrix else False,
            has_guide,
            has_proof,
            has_test,
        )
        if f
    )
    if claimed and missing:
        band = "incomplete_claimed"
    elif claimed and score_val < MIN_CLAIMED_SCORE:
        band = "below_floor"
        missing.append(f"score<{MIN_CLAIMED_SCORE}")
    elif not claimed:
        band = "deep_primary"

    return ArtifactScore(
        domain=domain,
        band=band,
        claimed_complete=claimed,
        loc=_package_loc(pkg),
        has_catalog=has_catalog,
        has_session_matrix=has_session if matrix else False,
        has_capability_status=has_status if matrix else False,
        has_guide=has_guide,
        has_proof=has_proof,
        has_test=has_test,
        has_explain_hooks=has_explain_hooks,
        has_checkpoint=has_checkpoint,
        analysis_only=analysis_only,
        missing=missing,
        notes=notes,
    )


def collect() -> list[ArtifactScore]:
    rows = [score_domain(d, m, claimed=True) for d, m in sorted(CLAIMED_COMPLETE.items())]
    rows.extend(score_domain(d, m, claimed=False) for d, m in sorted(DEEP_NON_MATRIX.items()))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail on incomplete claimed domains.")
    parser.add_argument("--report", action="store_true", help="Print human-readable table.")
    parser.add_argument("--json", type=Path, default=None, help="Write JSON artifact.")
    args = parser.parse_args(argv)
    if not args.check and not args.report and args.json is None:
        args.report = True

    rows = collect()
    failures = [
        r.domain
        for r in rows
        if r.claimed_complete and (r.missing or r.score < MIN_CLAIMED_SCORE)
    ]
    payload = {
        "min_claimed_score": MIN_CLAIMED_SCORE,
        "domains": [r.to_payload() for r in rows],
        "claimed_complete_failures": failures,
        "domain_floor": {
            "claimed_complete_requires": [
                "catalog.py",
                "Session.<domain>_capability_matrix",
                "capability_status wiring",
                "guides/quickstart",
                "proofs/ or proof mention",
                "tests/",
                "explain_hooks.py",
                "checkpoint.py (unless analysis_only)",
                f"artifact score >= {MIN_CLAIMED_SCORE}",
            ],
            "honesty": (
                "Equal LOC across domains is not required. The floor is quality "
                "shape (matrix + Session + explain + guide/proof/test), not line count."
            ),
        },
    }

    if args.report:
        print("BuildML domain maturity index")
        print(
            f"{'domain':20} {'band':20} {'score':5} {'loc':7} "
            f"{'hooks':5} {'ckpt':5} missing"
        )
        for r in rows:
            miss = ",".join(r.missing) if r.missing else "-"
            print(
                f"{r.domain:20} {r.band:20} {r.score:5d} {r.loc:7d} "
                f"{int(r.has_explain_hooks):5d} {int(r.has_checkpoint):5d} {miss}"
            )
        print(f"\nmin claimed score: {MIN_CLAIMED_SCORE}")
        print(f"claimed_complete failures: {failures or 'none'}")

    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.json}")

    if args.check and failures:
        print(
            "domain maturity check failed — claimed-complete domains missing artifacts "
            f"or below floor (score<{MIN_CLAIMED_SCORE}):",
            ", ".join(failures),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
