"""Multi-dataset EDA adaptability gauntlet (Static research HTML + Dashboard App).

Runs ≥10 analyses across real-world and synthetic frames, validates readiness
sheet / static report completeness, report-fit markers, and adaptive guidance.

Usage (from repo root, with .venv active)::

    .venv\\Scripts\\python.exe -u scripts/eda_adaptability_gauntlet.py

Artifacts land under ``.buildml-artifacts/gauntlet/`` (HTML, summary.json,
summary.md). Exit code 0 only when every case passes.
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

ARTIFACTS = ROOT / ".buildml-artifacts" / "gauntlet"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Markers that must appear in research HTML / must not appear anywhere.
_STATIC_REQUIRED = (
    "Findings register",
    "Ledger — every computed number",
    "Recommended sequence",
    "What each finding assumes",
    "bml-table--fit",
    "bml-cell-wrap",
    "Offline HTML",  # Static primary export matches App naming
    'id="bml-offline-html"',
)
_STATIC_FORBIDDEN = (
    "PDF briefing",
    'id="bml-csv"',
    "bml-csv-payload",
)
_FORBIDDEN_ANY = (
    "Unknown domain: ledger-",
    "Unknown domain: cockpit-ledger-",
)
_DEMO_COLUMNS = ("target_churn", "monthly_charges", "tenure")


@dataclass
class CaseResult:
    name: str
    kind: str
    task: str
    n_rows: int
    n_cols: int
    static_ok: bool = False
    app_ok: bool = False
    notes: list[str] = field(default_factory=list)
    findings: int = 0
    assumptions: int = 0
    ledger_groups: int = 0
    gates_items: int = 0
    academy_concepts: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.static_ok and self.app_ok and not self.errors


def _roles_for(
    frame: pd.DataFrame,
    *,
    target: str | None,
    id_col: str | None = None,
) -> dict[str, str]:
    roles: dict[str, str] = {}
    for col in frame.columns:
        if col == target:
            roles[col] = "target"
        elif col == id_col:
            roles[col] = "id"
        else:
            roles[col] = "feature"
    return roles


def _session_from_frame(
    frame: pd.DataFrame,
    *,
    target: str | None,
    id_col: str | None = None,
    stratify: bool = False,
    test_size: float = 0.25,
    seed: int = 0,
):
    from buildml import Session

    roles = _roles_for(frame, target=target, id_col=id_col)
    session = Session.ingest(frame).set_roles(roles)
    if target and stratify:
        try:
            return session.split(test_size=test_size, stratify=True, random_state=seed)
        except Exception:
            return session.split(test_size=test_size, stratify=False, random_state=seed)
    if target:
        return session.split(test_size=test_size, stratify=False, random_state=seed)
    return session


# ── Dataset builders ─────────────────────────────────────────────────────────


def ds_iris() -> tuple[pd.DataFrame, str | None, str | None, str, bool]:
    from sklearn.datasets import load_iris

    bunch = load_iris(as_frame=True)
    frame = bunch.frame.copy()
    frame = frame.rename(columns={"target": "species"})
    return frame, "species", None, "classification", True


def ds_wine() -> tuple[pd.DataFrame, str | None, str | None, str, bool]:
    from sklearn.datasets import load_wine

    bunch = load_wine(as_frame=True)
    frame = bunch.frame.copy()
    frame = frame.rename(columns={"target": "cultivar"})
    return frame, "cultivar", None, "classification", True


def ds_breast_cancer() -> tuple[pd.DataFrame, str | None, str | None, str, bool]:
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame.copy()
    frame = frame.rename(columns={"target": "malignant"})
    return frame, "malignant", None, "classification", True


def ds_diabetes() -> tuple[pd.DataFrame, str | None, str | None, str, bool]:
    from sklearn.datasets import load_diabetes

    bunch = load_diabetes(as_frame=True)
    frame = bunch.frame.copy()
    frame = frame.rename(columns={"target": "disease_progression"})
    return frame, "disease_progression", None, "regression", False


def ds_california_housing() -> tuple[pd.DataFrame, str | None, str | None, str, bool]:
    from sklearn.datasets import fetch_california_housing

    bunch = fetch_california_housing(as_frame=True)
    frame = bunch.frame.copy().sample(n=2500, random_state=11).reset_index(drop=True)
    frame = frame.rename(columns={"MedHouseVal": "median_house_value"})
    return frame, "median_house_value", None, "regression", False


def ds_titanic_like() -> tuple[pd.DataFrame, str | None, str | None, str, bool]:
    """Compact Titanic-style categorical/numeric mix (no network)."""
    rng = np.random.default_rng(42)
    n = 891
    pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    age = rng.normal(29, 14, n)
    age[rng.random(n) < 0.20] = np.nan
    fare = np.expm1(rng.normal(3.2, 0.9, n))
    embarked = rng.choice(["S", "C", "Q", None], size=n, p=[0.72, 0.19, 0.07, 0.02])
    sibsp = rng.integers(0, 5, n)
    logits = (
        -1.2
        + 1.1 * (sex == "female")
        - 0.45 * (pclass - 1)
        - 0.02 * np.nan_to_num(age, nan=29)
        + 0.015 * fare
        + rng.normal(0, 0.7, n)
    )
    survived = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    frame = pd.DataFrame(
        {
            "passenger_id": [f"P-{i:04d}" for i in range(n)],
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "sibsp": sibsp,
            "fare": fare,
            "embarked": embarked,
            "survived": survived,
        }
    )
    return frame, "survived", "passenger_id", "classification", True


def ds_synthetic_dirty() -> tuple[pd.DataFrame, str | None, str | None, str, bool]:
    from launch_synthetic_eda_studio import build_synthetic_frame

    frame = build_synthetic_frame(n_rows=800, seed=7)
    return frame, "target_churn", "customer_id", "classification", True


def ds_high_cardinality() -> tuple[pd.DataFrame, str | None, str | None, str, bool]:
    rng = np.random.default_rng(3)
    n = 1200
    frame = pd.DataFrame(
        {
            "row_id": [f"R{i}" for i in range(n)],
            "sku": [f"SKU-{rng.integers(0, 900):04d}" for _ in range(n)],
            "region": rng.choice(["NA", "EU", "APAC", "LATAM", "MEA"], size=n),
            "channel": rng.choice(["web", "store", "partner", "call"], size=n),
            "amount": np.expm1(rng.normal(4.0, 0.8, n)),
            "latency_ms": rng.gamma(2.0, 40, n),
            "label": rng.choice([0, 1], size=n, p=[0.88, 0.12]),
        }
    )
    # Force near-id cardinality on sku.
    return frame, "label", "row_id", "classification", True


def ds_wide_many_cols() -> tuple[pd.DataFrame, str | None, str | None, str, bool]:
    rng = np.random.default_rng(19)
    n = 180
    data: dict[str, Any] = {"row_id": list(range(n))}
    for i in range(40):
        data[f"feat_{i:02d}"] = rng.normal(0, 1, n)
    data["cat_a"] = rng.choice(list("ABCDEFGH"), size=n)
    data["cat_b"] = rng.choice([f"bucket-{k}" for k in range(25)], size=n)
    y = (data["feat_00"] + 0.4 * data["feat_01"] + rng.normal(0, 0.8, n) > 0).astype(int)
    data["y"] = y
    # Inject missingness across a slice of columns.
    frame = pd.DataFrame(data)
    for col in list(frame.columns)[2:12]:
        frame.loc[rng.random(n) < 0.12, col] = np.nan
    return frame, "y", "row_id", "classification", True


def ds_small_n_textish() -> tuple[pd.DataFrame, str | None, str | None, str, bool]:
    rng = np.random.default_rng(5)
    n = 36
    notes = [
        "ok",
        "Needs review",
        "NEEDS REVIEW",
        "needs  review",
        "duplicate ticket",
        "Duplicate Ticket",
        None,
    ]
    frame = pd.DataFrame(
        {
            "ticket_id": [f"T-{i}" for i in range(n)],
            "priority": rng.choice(["P1", "P2", "P3", "p1", "P2 "], size=n),
            "notes": rng.choice(notes, size=n),
            "hours_open": rng.exponential(12, n),
            "reopens": rng.integers(0, 4, n),
            "resolved": rng.choice([0, 1], size=n, p=[0.4, 0.6]),
        }
    )
    return frame, "resolved", "ticket_id", "classification", True


def ds_regression_imbalanced_tall() -> tuple[pd.DataFrame, str | None, str | None, str, bool]:
    rng = np.random.default_rng(21)
    n = 4000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(2, 1.5, n)
    cat = rng.choice(["a", "b", "c", "d"], size=n, p=[0.7, 0.15, 0.1, 0.05])
    # Heavy-tailed target with rare spikes.
    y = 10 + 2.2 * x1 - 0.8 * x2 + rng.normal(0, 1.0, n)
    y[rng.choice(n, size=25, replace=False)] *= 8
    frame = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x1 * x2 + rng.normal(0, 0.3, n),
            "group": cat,
            "const": 1,
            "target_score": y,
        }
    )
    frame.loc[rng.random(n) < 0.07, "x2"] = np.nan
    return frame, "target_score", None, "regression", False


def ds_no_target_profile() -> tuple[pd.DataFrame, str | None, str | None, str, bool]:
    rng = np.random.default_rng(8)
    n = 500
    frame = pd.DataFrame(
        {
            "metric_a": rng.normal(50, 10, n),
            "metric_b": rng.exponential(5, n),
            "segment": rng.choice(["alpha", "beta", "gamma", None], size=n),
            "flag": rng.choice([0, 1], size=n),
        }
    )
    return frame, None, None, "unsupervised", False


DATASETS: list[tuple[str, str, Callable[[], tuple[pd.DataFrame, str | None, str | None, str, bool]]]] = [
    ("iris", "sklearn", ds_iris),
    ("wine", "sklearn", ds_wine),
    ("breast_cancer", "sklearn", ds_breast_cancer),
    ("diabetes", "sklearn", ds_diabetes),
    ("california_housing_2.5k", "sklearn", ds_california_housing),
    ("titanic_like", "synthetic-realworld", ds_titanic_like),
    ("synthetic_dirty_cls", "synthetic-buildml", ds_synthetic_dirty),
    ("high_cardinality", "synthetic-buildml", ds_high_cardinality),
    ("wide_many_cols", "synthetic-buildml", ds_wide_many_cols),
    ("small_n_textish", "synthetic-buildml", ds_small_n_textish),
    ("tall_regression_spikes", "synthetic-buildml", ds_regression_imbalanced_tall),
    ("no_target_profile", "synthetic-buildml", ds_no_target_profile),
]


# ── Validators ───────────────────────────────────────────────────────────────


def _check_static_html(html: str, *, dataset_cols: set[str], name: str) -> list[str]:
    issues: list[str] = []
    for marker in _STATIC_REQUIRED:
        if marker not in html:
            issues.append(f"static missing marker: {marker}")
    for bad in _STATIC_FORBIDDEN:
        if bad in html:
            issues.append(f"static forbidden export: {bad}")
    for bad in _FORBIDDEN_ANY:
        if bad in html:
            issues.append(f"static forbidden: {bad}")
    # Overflow strategy present.
    if "bml-table-wrap" not in html:
        issues.append("static missing bml-table-wrap")
    if "overflow-wrap: anywhere" not in html and "bml-cell-wrap" not in html:
        issues.append("static missing wrap strategy")
    # Assumption evidence must not use nowrap ellipsis on teaching prose.
    if re.search(
        r"\.assumption-card__evidence[^{]*\{[^}]*white-space:\s*nowrap",
        html,
    ):
        issues.append("static CSS ellipsis-truncates assumption evidence")
    # Static shell must stay dataset-agnostic structurally (no ledger domain bleed).
    if "Unknown domain" in html:
        issues.append("static contains Unknown domain")
    return issues


def _check_app_payloads(
    report: dict[str, Any],
    sheet: dict[str, Any],
    gates: dict[str, Any],
    academy: dict[str, Any],
    *,
    dataset_cols: set[str],
    name: str,
    target: str | None,
) -> list[str]:
    issues: list[str] = []
    for bad in _FORBIDDEN_ANY:
        blob = json.dumps({"sheet": sheet, "gates": gates, "academy": academy})
        if bad in blob:
            issues.append(f"app forbidden: {bad}")

    coverage = sheet.get("coverage") or {}
    if not sheet.get("kpis"):
        issues.append("app sheet missing kpis")
    if "register" not in sheet:
        issues.append("app sheet missing register")
    if "ledger" not in sheet or not sheet["ledger"]:
        issues.append("app sheet missing ledger")
    if "assumptions" not in sheet:
        issues.append("app sheet missing assumptions")
    if coverage.get("ledger_groups", 0) < 1:
        issues.append("app coverage ledger_groups < 1")

    adapt = sheet.get("adapt") or {}
    if target and adapt.get("target_column") not in (target, None):
        # Target should match when declared.
        if adapt.get("target_column") != target:
            issues.append(
                f"adapt target mismatch: {adapt.get('target_column')!r} vs {target!r}"
            )
    if target and adapt.get("has_target") is False:
        issues.append("adapt has_target false despite declared target")

    # Teaching prose present and non-truncated structurally (full strings in JSON).
    for note in (sheet.get("assumptions") or [])[:5]:
        for key in ("means", "matters", "technical", "evidence"):
            val = note.get(key)
            if isinstance(val, str) and val.endswith("...") and len(val) < 40:
                issues.append(f"assumption {key} looks truncated: {val!r}")

    for row in (sheet.get("register") or [])[:8]:
        detail = str(row.get("detail") or "")
        evidence = str(row.get("evidence") or "")
        if detail.endswith("Sou...") or "Sou..." in evidence:
            issues.append("register evidence truncated with Sou...")

    gates_items = (
        gates.get("rows")
        or gates.get("items")
        or gates.get("gates")
        or gates.get("checklist")
        or []
    )
    if isinstance(gates, dict) and not gates_items:
        for key in ("stages", "cards"):
            if gates.get(key):
                gates_items = gates[key]
                break
    if not gates_items and not gates.get("counts"):
        issues.append("gates payload empty")

    concepts = academy.get("concepts") or academy.get("items") or academy.get("stages") or []
    if not concepts and not academy.get("curriculum"):
        issues.append("academy payload empty")

    # Demo-column overfit guard (except dirty synthetic which uses target_churn).
    if name != "synthetic_dirty_cls":
        blob = json.dumps(sheet).lower()
        for demo in _DEMO_COLUMNS:
            if demo in blob and demo not in {c.lower() for c in dataset_cols}:
                # session_sentence / focus may still mention nothing; flag hard hits.
                if f'"{demo}"' in blob or f"'{demo}'" in blob:
                    issues.append(f"app sheet hard-codes demo column: {demo}")

    # Ledger keys must not be treated as domain boards (routing contract).
    for group in sheet.get("ledger") or []:
        key = str(group.get("key") or "")
        if key.startswith("ledger-"):
            issues.append(f"ledger key already prefixed: {key}")

    return issues


def _run_case(
    name: str,
    kind: str,
    builder: Callable[[], tuple[pd.DataFrame, str | None, str | None, str, bool]],
    *,
    artifacts_dir: Path,
) -> CaseResult:
    from fastapi.testclient import TestClient

    from buildml.dashboard.app import create_app
    from buildml.dashboard.state import DashboardState, clear_state, set_state
    from buildml.eda.html_report import export_eda_html

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    result = CaseResult(name=name, kind=kind, task="?", n_rows=0, n_cols=0)
    try:
        frame, target, id_col, task, stratify = builder()
        result.task = task
        result.n_rows = int(frame.shape[0])
        result.n_cols = int(frame.shape[1])
        dataset_cols = set(map(str, frame.columns))

        session = _session_from_frame(
            frame, target=target, id_col=id_col, stratify=stratify, seed=0
        )
        report_obj = session.eda(include_plots=False, show=False)
        report = report_obj.to_dict()
        result.findings = len(report.get("findings") or [])

        # Static research HTML
        static_path = artifacts_dir / f"{name}_static.html"
        export_eda_html(report, static_path, max_figures=0)
        html = static_path.read_text(encoding="utf-8")
        static_issues = _check_static_html(html, dataset_cols=dataset_cols, name=name)
        result.static_ok = not static_issues
        result.notes.extend(static_issues)

        # Dashboard App APIs
        set_state(
            DashboardState(
                report=report_obj,
                report_dict=report,
                title=f"Gauntlet · {name}",
                session_meta={"has_split": target is not None, "gauntlet": name},
            )
        )
        try:
            client = TestClient(create_app())
            home = client.get("/")
            if home.status_code != 200:
                result.errors.append(f"home status {home.status_code}")
            if "Offline HTML" not in home.text:
                result.errors.append("home missing Offline HTML primary")
            if "PDF briefing" in home.text:
                result.errors.append("home still shows PDF briefing")
            if 'id="csv-export"' in home.text:
                result.errors.append("home still shows CSV export button")

            cockpit = client.get("/api/cockpit").json()
            sheet = cockpit.get("sheet") or {}
            result.assumptions = len(sheet.get("assumptions") or [])
            result.ledger_groups = len(sheet.get("ledger") or [])

            gates = client.get("/api/gates").json()
            academy_resp = client.get("/api/domains/academy")
            if academy_resp.status_code != 200:
                result.errors.append(f"academy status {academy_resp.status_code}")
                academy = {}
            else:
                academy = academy_resp.json()

            # Count gates / academy loosely.
            g_items = gates.get("rows") or gates.get("items") or gates.get("gates") or []
            if not g_items and isinstance(gates.get("stages"), list):
                g_items = [
                    item
                    for stage in gates["stages"]
                    for item in (stage.get("items") or stage.get("gates") or stage.get("rows") or [])
                ]
            result.gates_items = len(g_items) if isinstance(g_items, list) else 0
            a_items = academy.get("concepts") or academy.get("items") or []
            if not a_items and isinstance(academy.get("stages"), list):
                a_items = academy["stages"]
            result.academy_concepts = len(a_items) if isinstance(a_items, list) else 0

            # Ledger must 404 as domain.
            if sheet.get("ledger"):
                key = sheet["ledger"][0]["key"]
                resp = client.get(f"/api/domains/ledger-{key}")
                if resp.status_code != 404:
                    result.errors.append(f"ledger-{key} domain not 404")

            app_issues = _check_app_payloads(
                report,
                sheet,
                gates,
                academy,
                dataset_cols=dataset_cols,
                name=name,
                target=target,
            )
            result.notes.extend(app_issues)
            result.app_ok = not app_issues and not result.errors

            # Persist a compact app snapshot for inspection.
            snap = {
                "adapt": sheet.get("adapt"),
                "coverage": sheet.get("coverage"),
                "kpis": sheet.get("kpis"),
                "session_sentence": sheet.get("session_sentence"),
                "register_sample": (sheet.get("register") or [])[:3],
                "assumption_sample": (sheet.get("assumptions") or [])[:2],
            }
            (artifacts_dir / f"{name}_app.json").write_text(
                json.dumps(snap, indent=2, default=str), encoding="utf-8"
            )
        finally:
            clear_state()

        if result.static_ok and result.app_ok and not result.errors:
            result.notes.append("pass")
    except Exception as exc:  # noqa: BLE001 — collect per-case failures
        result.errors.append(f"{type(exc).__name__}: {exc}")
        result.notes.append(traceback.format_exc(limit=4))
        result.static_ok = False
        result.app_ok = False
    return result


def write_summaries(results: list[CaseResult], artifacts_dir: Path) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_cases": len(results),
        "n_passed": sum(1 for r in results if r.ok),
        "n_failed": sum(1 for r in results if not r.ok),
        "cases": [asdict(r) for r in results],
    }
    (artifacts_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# EDA adaptability gauntlet",
        "",
        f"Cases: **{payload['n_passed']}/{payload['n_cases']} passed**",
        "",
        "| Dataset | Type | Task | n×p | Static | App | Findings | Notes |",
        "|---|---|---|---:|---|---|---:|---|",
    ]
    for r in results:
        notes = "; ".join(r.notes + r.errors)[:180]
        lines.append(
            f"| {r.name} | {r.kind} | {r.task} | {r.n_rows}×{r.n_cols} | "
            f"{'OK' if r.static_ok else 'FAIL'} | {'OK' if r.app_ok else 'FAIL'} | "
            f"{r.findings} | {notes} |"
        )
    lines.append("")
    (artifacts_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def run_gauntlet(
    *,
    artifacts_dir: Path | None = None,
    quiet: bool = False,
) -> tuple[list[CaseResult], dict[str, Any]]:
    """Run all dataset cases; return results and summary payload."""
    out = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS
    out.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"Gauntlet artifacts -> {out.resolve()}")
    results: list[CaseResult] = []
    for name, kind, builder in DATASETS:
        if not quiet:
            print(f"\n=== {name} ({kind}) ===")
        result = _run_case(name, kind, builder, artifacts_dir=out)
        results.append(result)
        if not quiet:
            status = "PASS" if result.ok else "FAIL"
            print(
                f"  {status}  {result.task}  {result.n_rows}×{result.n_cols}  "
                f"findings={result.findings}  ledger={result.ledger_groups}  "
                f"static={result.static_ok} app={result.app_ok}"
            )
            for note in result.notes[:6]:
                if note != "pass":
                    print(f"    · {note}")
            for err in result.errors[:4]:
                print(f"    ! {err}")

    payload = write_summaries(results, out)
    if not quiet:
        print(f"\nSummary: {payload['n_passed']}/{payload['n_cases']} passed -> {out / 'summary.md'}")
    return results, payload


def main() -> int:
    results, payload = run_gauntlet(artifacts_dir=ARTIFACTS, quiet=False)
    if len(results) < 10:
        print("ERROR: fewer than 10 cases ran", file=sys.stderr)
        return 1
    return 0 if payload["n_passed"] == payload["n_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
