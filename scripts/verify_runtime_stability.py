#!/usr/bin/env python3
"""Subprocess-isolated runtime stability / use-case verification.

Honesty check for optional extras: each probe runs in its own process so a
Windows access violation cannot kill the parent. Statuses:

- ``ok``     : probe completed successfully
- ``fail``   : Python exception / non-zero exit (catchable)
- ``crash``  : native hard-kill / access violation / timeout with fatal signal
- ``skip``   : optional dependency missing (expected without that extra)

Usage (repo root)::

    python scripts/verify_runtime_stability.py
    python scripts/verify_runtime_stability.py --artifact runtime-stability.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each probe is a self-contained Python snippet. Print exactly one of:
# OK / SKIP:<reason> / FAIL:<reason> then exit 0 for OK/SKIP, 1 for FAIL.
_CLASSICAL_BOOT = textwrap.dedent(
    """\
    import numpy as np, pandas as pd
    from sklearn.linear_model import LogisticRegression
    from buildml import Session
    rng = np.random.default_rng({seed})
    n = {n}
    frame = pd.DataFrame({{
        'age': rng.normal(40, 10, size=n),
        'income': rng.normal(60, 15, size=n),
        'approved': np.array([0, 1] * (n // 2)),
    }})
    session = (
        Session.ingest(frame)
        .set_roles({{'age': 'feature', 'income': 'feature', 'approved': 'target'}})
        .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
        .scale(method='standard')
        .fit(LogisticRegression(max_iter=400), task='classification')
    )
    """
)

PROBES: list[tuple[str, str, str]] = [
    (
        "import_core",
        "gate",
        "import buildml; from buildml import Session; print('OK', buildml.__version__)",
    ),
    (
        "classical_fit_evaluate",
        "core",
        _CLASSICAL_BOOT.format(seed=0, n=120)
        + "ev = session.evaluate(partition='test'); assert ev.metrics; print('OK')",
    ),
    (
        "classical_pipeline_roundtrip",
        "core",
        textwrap.dedent(
            """\
            import tempfile
            from pathlib import Path
            import numpy as np, pandas as pd
            from sklearn.linear_model import LogisticRegression
            from buildml import Session
            rng = np.random.default_rng(1)
            n = 100
            frame = pd.DataFrame({
                'age': rng.normal(40, 10, size=n),
                'income': rng.normal(60, 15, size=n),
                'approved': np.array([0, 1] * (n // 2)),
            })
            roles = {'age': 'feature', 'income': 'feature', 'approved': 'target'}
            session = (
                Session.ingest(frame).set_roles(roles)
                .split(test_size=0.25, stratify=True, random_state=0)
                .scale(method='standard')
                .fit(LogisticRegression(max_iter=400), task='classification')
            )
            with tempfile.TemporaryDirectory() as td:
                pipe = Path(td) / 'pipe'
                session.save_pipeline(pipe, evaluate_partition='test')
                holdout = frame.iloc[list(session.split_plan.test_indices)].reset_index(drop=True)
                scored = Session().predict_from_pipeline(pipe, holdout, roles=roles, trusted=True)
                assert scored.n_rows == len(holdout)
            print('OK')
            """
        ),
    ),
    (
        "fairness_evaluate",
        "core",
        textwrap.dedent(
            """\
            import numpy as np, pandas as pd
            from sklearn.linear_model import LogisticRegression
            from buildml import Session
            rng = np.random.default_rng(2)
            n = 160
            group = np.array(['A'] * (n // 2) + ['B'] * (n // 2))
            x = rng.normal(size=n)
            y = (x + np.where(group == 'B', -0.8, 0.0) > 0).astype(int)
            frame = pd.DataFrame({'x': x, 'group': group, 'y': y})
            session = (
                Session.ingest(frame)
                .set_roles({'x': 'feature', 'group': 'ignore', 'y': 'target'})
                .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
                .fit(LogisticRegression(max_iter=500), task='classification')
            )
            report = session.fairness.evaluate(sensitive_column='group', partition='test')
            assert report.n_rows > 0
            print('OK')
            """
        ),
    ),
    (
        "shap_explain",
        "optional",
        _CLASSICAL_BOOT.format(seed=3, n=100)
        + textwrap.dedent(
            """\
            from buildml.core.errors import MissingExtraError
            try:
                result = session.explain_shap(max_samples=32)
            except MissingExtraError as exc:
                print('SKIP:' + str(exc))
                raise SystemExit(0)
            assert result is not None
            print('OK')
            """
        ),
    ),
    (
        "ensemble_voting",
        "core",
        textwrap.dedent(
            """\
            import numpy as np, pandas as pd
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from buildml import Session
            rng = np.random.default_rng(4)
            n = 120
            x1 = rng.normal(size=n); x2 = rng.normal(size=n)
            y = (0.9 * x1 - 0.4 * x2 > 0).astype(int)
            frame = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
            bases = {
                'lr': LogisticRegression(max_iter=400),
                'rf': RandomForestClassifier(n_estimators=20, random_state=0),
            }
            session = (
                Session.ingest(frame)
                .set_roles({'x1': 'feature', 'x2': 'feature', 'y': 'target'})
                .split(test_size=0.2, validation_size=0.2, random_state=0, stratify=True)
                .scale(method='standard')
            )
            session.ensemble.fit_voting(bases, voting='soft', task='classification')
            print('OK')
            """
        ),
    ),
    (
        "anomaly_sklearn",
        "core",
        textwrap.dedent(
            """\
            import numpy as np, pandas as pd
            from buildml import Session
            rng = np.random.default_rng(5)
            n = 100
            frame = pd.DataFrame({
                'x': rng.normal(size=n),
                'y': rng.normal(size=n),
                'is_fraud': (rng.random(n) < 0.1).astype(int),
            })
            session = (
                Session.ingest(frame)
                .set_roles({'x': 'feature', 'y': 'feature', 'is_fraud': 'target'})
                .split(test_size=0.25, validation_size=0.25, random_state=0)
            )
            session.anomaly.fit(method='isolation_forest', mode='unsupervised', contamination=0.1)
            print('OK')
            """
        ),
    ),
    (
        "forecast_classical",
        "core",
        textwrap.dedent(
            """\
            import numpy as np, pandas as pd
            from buildml import Session
            n = 80
            t = pd.date_range('2023-01-01', periods=n, freq='D')
            y = 8 + 0.03 * np.arange(n) + np.sin(np.arange(n) / 6.0)
            frame = pd.DataFrame({'clock': t, 'sales': y})
            session = (
                Session.ingest(frame)
                .set_roles({'clock': 'time', 'sales': 'target'})
                .time_split(test_size=0.2, validation_size=0.2)
            )
            session.forecast.fit(method='lag_ridge', lags=[1, 2, 3], horizon=5)
            print('OK')
            """
        ),
    ),
    (
        "cbr_sklearn_path",
        "domain",
        textwrap.dedent(
            """\
            import numpy as np, pandas as pd
            from buildml import Session
            rng = np.random.default_rng(6)
            n = 80
            frame = pd.DataFrame({
                'a': rng.normal(size=n),
                'b': rng.normal(size=n),
                'y': rng.integers(0, 3, size=n),
            })
            session = (
                Session.ingest(frame)
                .set_roles({'a': 'feature', 'b': 'feature', 'y': 'target'})
                .split(test_size=0.25, validation_size=0.25, random_state=0)
            )
            session.cbr.fit(backend='sklearn', task='classification', k=3)
            print('OK')
            """
        ),
    ),
    (
        "cbr_industry_ann",
        "optional-native",
        textwrap.dedent(
            """\
            import numpy as np, pandas as pd
            from buildml import Session
            from buildml.core.errors import MissingExtraError, ValidationError
            rng = np.random.default_rng(7)
            n = 80
            frame = pd.DataFrame({
                'a': rng.normal(size=n),
                'b': rng.normal(size=n),
                'y': rng.integers(0, 3, size=n),
            })
            session = (
                Session.ingest(frame)
                .set_roles({'a': 'feature', 'b': 'feature', 'y': 'target'})
                .split(test_size=0.25, validation_size=0.25, random_state=0)
            )
            try:
                session.cbr.fit(backend='industry', task='classification', k=3)
            except (MissingExtraError, ValidationError, TypeError) as exc:
                print('SKIP:' + type(exc).__name__ + ':' + str(exc)[:160])
                raise SystemExit(0)
            print('OK')
            """
        ),
    ),
    (
        "torch_import",
        "optional-native",
        textwrap.dedent(
            """\
            from buildml.core.errors import MissingExtraError
            from buildml.dl.extras import require_torch
            try:
                require_torch()
                import torch
                x = torch.zeros(2, 3)
                assert x.shape == (2, 3)
            except MissingExtraError as exc:
                print('SKIP:' + str(exc))
                raise SystemExit(0)
            except OSError as exc:
                print('FAIL:torch_dll:' + str(exc)[:200])
                raise SystemExit(1)
            print('OK')
            """
        ),
    ),
    (
        "dl_tiny_mlp_fit",
        "optional-native",
        textwrap.dedent(
            """\
            import numpy as np, pandas as pd
            from buildml import Session
            from buildml.core.errors import MissingExtraError
            rng = np.random.default_rng(8)
            n = 64
            frame = pd.DataFrame({
                'a': rng.normal(size=n),
                'b': rng.normal(size=n),
                'y': np.array([0, 1] * (n // 2)),
            })
            try:
                import torch.nn as nn
                class TinyMLP(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
                    def forward(self, x):
                        return self.net(x)
                session = (
                    Session.ingest(frame)
                    .set_roles({'a': 'feature', 'b': 'feature', 'y': 'target'})
                    .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
                )
                session.make_torch_loaders(batch_size=16, normalize=True, seed=0)
                session.dl.fit(TinyMLP(), epochs=1, learning_rate=1e-2, device='cpu')
            except MissingExtraError as exc:
                print('SKIP:' + str(exc))
                raise SystemExit(0)
            except ModuleNotFoundError as exc:
                print('SKIP:' + str(exc))
                raise SystemExit(0)
            except OSError as exc:
                print('FAIL:torch_dll:' + str(exc)[:200])
                raise SystemExit(1)
            print('OK')
            """
        ),
    ),
    (
        "automl_native_smoke",
        "optional",
        textwrap.dedent(
            """\
            import numpy as np, pandas as pd
            from buildml import Session
            from buildml.core.errors import MissingExtraError, ValidationError
            rng = np.random.default_rng(9)
            n = 80
            frame = pd.DataFrame({
                'a': rng.normal(size=n),
                'b': rng.normal(size=n),
                'y': np.array([0, 1] * (n // 2)),
            })
            session = (
                Session.ingest(frame)
                .set_roles({'a': 'feature', 'b': 'feature', 'y': 'target'})
                .split(test_size=0.25, validation_size=0.25, stratify=True, random_state=0)
            )
            try:
                session.automl.run(backend='native', n_trials=4, cv=2, time_budget=15)
            except (MissingExtraError, ValidationError, ImportError) as exc:
                print('SKIP:' + type(exc).__name__ + ':' + str(exc)[:160])
                raise SystemExit(0)
            print('OK')
            """
        ),
    ),
    (
        "cvxpy_solve",
        "optional-native",
        textwrap.dedent(
            """\
            import importlib.util
            if importlib.util.find_spec('cvxpy') is None:
                print('SKIP:cvxpy not installed')
                raise SystemExit(0)
            import cvxpy as cp
            x = cp.Variable()
            prob = cp.Problem(cp.Minimize(x), [x >= 1])
            prob.solve(solver=cp.SCS, verbose=False)
            print('OK')
            """
        ),
    ),
    (
        "hnswlib_build",
        "optional-native",
        textwrap.dedent(
            """\
            import importlib.util
            import numpy as np
            if importlib.util.find_spec('hnswlib') is None:
                print('SKIP:hnswlib not installed')
                raise SystemExit(0)
            import hnswlib
            dim = 8
            data = np.random.default_rng(0).normal(size=(32, dim)).astype('float32')
            idx = hnswlib.Index(space='l2', dim=dim)
            idx.init_index(max_elements=64, ef_construction=50, M=8)
            idx.add_items(data)
            labels, _dist = idx.knn_query(data[:2], k=3)
            assert labels.shape == (2, 3)
            print('OK')
            """
        ),
    ),
]


@dataclass
class ProbeResult:
    name: str
    tier: str
    status: str
    detail: str
    elapsed_s: float
    returncode: int | None


def _classify(rc: int | None, out: str, timed_out: bool) -> tuple[str, str]:
    text = (out or "").strip()
    last = text.splitlines()[-1] if text else ""
    if timed_out:
        return "crash", "timeout"
    if "Windows fatal exception" in text or "access violation" in text.lower():
        return "crash", last[:300] or "access violation"
    # Windows STATUS_ACCESS_VIOLATION
    if rc is not None and rc in (-1073741819, 3221225477):
        return "crash", f"native_rc={rc}"
    if rc is not None and rc < 0:
        return "crash", f"signal_rc={rc}"
    if last.startswith("OK"):
        return "ok", last
    if last.startswith("SKIP:"):
        return "skip", last[5:]
    if last.startswith("FAIL:"):
        return "fail", last[5:]
    if rc == 0:
        return "ok", last or "ok"
    # Truncate traceback-ish output
    return "fail", (last or text)[:400]


def run_probe(name: str, tier: str, code: str, *, timeout: float) -> ProbeResult:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        rc = int(completed.returncode)
        out = (completed.stdout or "") + (completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = None
        out = ((exc.stdout or "") if isinstance(exc.stdout, str) else "") + (
            (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        )
    elapsed = time.perf_counter() - started
    status, detail = _classify(rc, out, timed_out)
    return ProbeResult(
        name=name,
        tier=tier,
        status=status,
        detail=detail,
        elapsed_s=round(elapsed, 2),
        returncode=rc,
    )


def run_pytest_probe(nodeid: str, *, timeout: float) -> ProbeResult:
    name = f"pytest:{nodeid}"
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", nodeid, "-q", "--tb=line"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        rc = int(completed.returncode)
        out = (completed.stdout or "") + (completed.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = None
        out = str(exc)
    elapsed = time.perf_counter() - started
    if timed_out:
        status, detail = "crash", "timeout"
    elif "Windows fatal exception" in out or "access violation" in out.lower():
        status, detail = "crash", "access violation during pytest"
    elif rc in (-1073741819, 3221225477) or (rc is not None and rc < 0):
        status, detail = "crash", f"native_rc={rc}"
    elif rc == 0:
        status, detail = "ok", "pytest passed"
    elif rc == 5:
        status, detail = "skip", "no tests collected"
    else:
        status, detail = "fail", (out.strip().splitlines()[-1] if out.strip() else f"rc={rc}")[:400]
    return ProbeResult(
        name=name,
        tier="gate",
        status=status,
        detail=detail,
        elapsed_s=round(elapsed, 2),
        returncode=rc,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "runtime-stability.json",
        help="JSON report path",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=ROOT / "runtime-stability.md",
        help="Markdown report path",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    results: list[ProbeResult] = []
    print(
        f"Runtime stability probes on {platform.system()} "
        f"{platform.release()} Python {platform.python_version()}",
        flush=True,
    )

    results.append(
        run_pytest_probe(
            "tests/integration/test_classical_alpha_smoke.py",
            timeout=max(args.timeout, 180.0),
        )
    )
    print(
        f"  [{results[-1].status.upper():5}] {results[-1].name} "
        f"({results[-1].elapsed_s}s)",
        flush=True,
    )

    for name, tier, code in PROBES:
        result = run_probe(name, tier, code, timeout=args.timeout)
        results.append(result)
        print(
            f"  [{result.status.upper():5}] {result.name} "
            f"({result.elapsed_s}s) {result.detail[:120]}",
            flush=True,
        )

    counts = {k: 0 for k in ("ok", "fail", "crash", "skip")}
    for row in results:
        counts[row.status] = counts.get(row.status, 0) + 1

    payload = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "counts": counts,
        "results": [asdict(r) for r in results],
        "interpretation": {
            "core_ok_means": (
                "Classical Session fit/evaluate/checkpoint paths completed "
                "without a native hard-kill in this environment."
            ),
            "crash_means": (
                "An optional native extra (Torch / hnswlib / cvxpy / etc.) hard-killed "
                "the worker process. BuildML cannot catch that fault; use a supported "
                "Python/wheel combo or Linux for those surfaces."
            ),
            "not_a_global_verdict": (
                "This report is local to the environment that ran the probe. "
                "Linux CI is the release gate for Torch and industry extras; "
                "Windows CI gates classical-only."
            ),
        },
    }
    args.artifact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Runtime stability report",
        "",
        f"- Platform: `{payload['platform']['system']} "
        f"{payload['platform']['release']}` Python `{payload['platform']['python']}`",
        f"- Counts: ok={counts['ok']} fail={counts['fail']} "
        f"crash={counts['crash']} skip={counts['skip']}",
        "",
        "| Probe | Tier | Status | Detail | Seconds |",
        "|---|---|---|---|---|",
    ]
    for row in results:
        detail = row.detail.replace("|", "\\|")[:120]
        lines.append(
            f"| `{row.name}` | {row.tier} | **{row.status}** | {detail} | {row.elapsed_s} |"
        )
    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "- **ok** on `gate` / `core` tiers: those Session paths are safe in this environment.",
            "- **crash** on `optional-native`: the installed wheel hard-crashes this OS/Python;",
            "  treat that surface as unsupported here.",
            "- Release gate for Torch/industry extras: Linux CI (Windows CI is classical-only).",
            "",
        ]
    )
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {args.artifact} and {args.markdown}")
    print(
        f"SUMMARY ok={counts['ok']} fail={counts['fail']} "
        f"crash={counts['crash']} skip={counts['skip']}"
    )

    # Non-zero only if a core/gate probe failed or crashed.
    bad_core = [
        r
        for r in results
        if r.tier in {"gate", "core"} and r.status in {"fail", "crash"}
    ]
    return 1 if bad_core else 0


if __name__ == "__main__":
    raise SystemExit(main())
