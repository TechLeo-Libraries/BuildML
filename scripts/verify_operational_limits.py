"""Isolated scale, memory, serving-soak and optional CUDA probes; JSON evidence."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_MEMORY_JOB = None


def limit_worker_memory(budget_mib: int) -> None:
    """Bound only this disposable worker's allocations on Linux or Windows."""
    if sys.platform != "win32":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (budget_mib * 1024 ** 2,) * 2)
        return
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64),
                    ("Flags", wintypes.DWORD), ("MinWorkingSet", ctypes.c_size_t),
                    ("MaxWorkingSet", ctypes.c_size_t), ("ActiveProcesses", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("Priority", wintypes.DWORD),
                    ("Scheduling", wintypes.DWORD)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("Basic", BasicLimits), ("IOCounters", ctypes.c_uint64 * 6),
                    ("ProcessMemory", ctypes.c_size_t), ("JobMemory", ctypes.c_size_t),
                    ("PeakProcessMemory", ctypes.c_size_t), ("PeakJobMemory", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    global _MEMORY_JOB
    _MEMORY_JOB = kernel.CreateJobObjectW(None, None)
    limits = ExtendedLimits()
    limits.Basic.Flags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    limits.ProcessMemory = budget_mib * 1024 ** 2
    if not _MEMORY_JOB or not kernel.SetInformationJobObject(_MEMORY_JOB, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.AssignProcessToJobObject(_MEMORY_JOB, kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())


def memory_mib() -> float | None:
    """Return measured process peak RSS, where the platform exposes it."""
    if sys.platform != "win32":
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024 ** 2 if sys.platform == "darwin" else 1024)
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    values = Counters()
    values.cb = ctypes.sizeof(values)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    if get_memory(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb):
        return values.PeakWorkingSetSize / 1024 ** 2
    return None


def scale_probe(rows: int, budget_mib: int | None) -> dict:
    """Compare predictions with an independent pipeline and measure peak memory."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from buildml import Session

    if budget_mib is not None:
        limit_worker_memory(budget_mib)
    rng = np.random.default_rng(42)
    started = time.perf_counter()
    x = rng.normal(size=(rows, 12))
    y = (x[:, 0] + .3 * x[:, 1] > 0).astype(int)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(12)])
    frame["target"] = y
    session = Session.ingest(frame).set_roles(
        {**{f"x{i}": "feature" for i in range(12)}, "target": "target"}
    ).split(test_size=.2, random_state=42)
    train, test = session.partition("train"), session.partition("test")
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    baseline.fit(train.drop(columns="target"), train.target)
    expected = baseline.predict(test.drop(columns="target"))
    session.scale(method="standard").fit(LogisticRegression(max_iter=500), task="classification")
    actual = np.asarray(session.predict(partition="test"))
    if not np.array_equal(actual, expected):
        raise AssertionError("Scale probe predictions differ from sklearn")
    return {"status": "passed", "rows": rows, "features": 12,
            "seconds": time.perf_counter() - started, "peak_rss_mib": memory_mib(),
            "predictions_equal": True, "memory_budget_mib": budget_mib,
            "scope": "whole worker, including both baseline and BuildML; in-memory training"}


def soak_probe(seconds: float) -> dict:
    """Exercise concurrent ASGI scoring and confirm parity throughout a bounded soak."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from starlette.testclient import TestClient

    from buildml import Session
    from buildml.serving.app import clear_serving_state, create_serving_app

    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x": rng.normal(size=500)})
    frame["y"] = (frame.x > 0).astype(int)
    session = Session.ingest(frame).set_roles({"x": "feature", "y": "target"}).split(test_size=.2, random_state=3)
    session.fit(LogisticRegression(), task="classification")
    with tempfile.TemporaryDirectory() as area:
        session.save_pipeline(Path(area) / "pipeline")
        app = create_serving_app(Path(area) / "pipeline", kind="pipeline", api_keys="local-probe", trusted=True)
        payload = {"rows": [{"x": -1.}, {"x": 1.}]}
        headers = {"Authorization": "Bearer local-probe"}
        expected = session.fit_result.estimator.predict(pd.DataFrame(payload["rows"])).tolist()
        before = memory_mib()
        deadline = time.monotonic() + seconds

        def requests():
            count, errors, latencies = 0, [], []
            with TestClient(app) as client:
                assert client.post("/predict", json=payload).status_code == 401
                while time.monotonic() < deadline:
                    started = time.perf_counter()
                    response = client.post("/predict", json=payload, headers=headers)
                    body = response.json()
                    if response.status_code != 200 or body.get("predictions") != expected:
                        errors.append({"code": response.status_code, "body": body})
                    count += 1
                    latencies.append(time.perf_counter() - started)
            return count, errors, latencies

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                outcomes = list(pool.map(lambda _: requests(), range(4)))
            latencies = [v for _, _, values in outcomes for v in values]
            errors = [v for _, values, _ in outcomes for v in values]
            return {"status": "passed" if not errors else "failed", "seconds": seconds,
                    "workers": 4, "requests": sum(r[0] for r in outcomes), "errors": errors[:10],
                    "latency_p95_ms": float(np.quantile(latencies, .95) * 1000),
                    "peak_rss_before_mib": before, "peak_rss_after_mib": memory_mib(),
                    "scope": "concurrent ASGI in-process soak; not an internet deployment or long-term SLA"}
        finally:
            clear_serving_state()


def gpu_probe() -> dict:
    """Measure real CUDA training throughput and device memory when available."""
    import torch
    if not torch.cuda.is_available():
        return {"status": "unavailable", "reason": "installed Torch has no usable CUDA device"}
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    x, y = torch.randn(8192, 32, device="cuda"), torch.randn(8192, 1, device="cuda")
    model = torch.nn.Sequential(torch.nn.Linear(32, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not torch.isfinite(loss):
        raise AssertionError("CUDA training produced nonfinite loss")
    import numpy as np
    import pandas as pd

    from buildml import Session

    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(8192, 32)), columns=[f"x{i}" for i in range(32)])
    frame["target"] = 2 * frame.x0 - frame.x1
    session = Session.ingest(frame).set_roles(
        {**{f"x{i}": "feature" for i in range(32)}, "target": "target"}
    ).split(test_size=.2, validation_size=.2, random_state=42)
    session.dl.make_loaders(task="regression", batch_size=256, normalize=True, seed=42)
    torch.cuda.synchronize()
    training_started = time.perf_counter()
    session.dl.fit(epochs=5, hidden=(128, 64), dropout=0, device="cuda")
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    metrics = session.dl.evaluate(partition="test", device="cuda")
    if not all(np.isfinite(v) for v in metrics.metrics.values() if isinstance(v, (int, float))):
        raise AssertionError("BuildML CUDA evaluation contains nonfinite metrics")
    return {"status": "passed", "device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "steps": 100, "examples_per_second": 819200 / elapsed,
            "seconds": elapsed, "peak_device_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "buildml_training_seconds": training_seconds,
            "buildml_epochs": session.dl_train_result.n_epochs_ran,
            "buildml_test_metrics": metrics.metrics,
            "scope": "Torch CUDA runtime baseline plus BuildML regression training/evaluation; single GPU"}


def main() -> int:
    """Run fault-isolated probes and retain stdout/stderr for failed configurations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["scale", "soak", "gpu"])
    parser.add_argument("--rows", type=int, nargs="+", default=[100000, 1000000])
    parser.add_argument("--memory-budget-mib", type=int)
    parser.add_argument("--soak-seconds", type=float, default=120)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("artifacts/operational-limits.json"))
    args = parser.parse_args()
    if args.worker:
        try:
            result = {"scale": lambda: scale_probe(args.rows[0], args.memory_budget_mib),
                      "soak": lambda: soak_probe(args.soak_seconds), "gpu": gpu_probe}[args.worker]()
        except MemoryError as exc:
            if args.worker != "scale" or args.memory_budget_mib is None:
                raise
            result = {"status": "memory_limited", "memory_budget_mib": args.memory_budget_mib,
                      "reason": str(exc), "scope": "bounded worker allocation refused; parent survives"}
        print(json.dumps(result))
        return 0 if result["status"] in {"passed", "memory_limited"} else 1
    probes = [("scale", ["--rows", str(n)]) for n in args.rows]
    probes.append(("soak", ["--soak-seconds", str(args.soak_seconds)]))
    if args.memory_budget_mib:
        probes.append(("memory_pressure", ["--rows", str(max(args.rows)), "--memory-budget-mib", str(args.memory_budget_mib)]))
    if args.gpu:
        probes.append(("gpu", []))
    results = []
    for name, options in probes:
        worker = "scale" if name == "memory_pressure" else name
        environment = os.environ.copy()
        environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", worker, *options],
                                    cwd=ROOT, env=environment, capture_output=True, text=True,
                                    timeout=max(300, args.soak_seconds + 60))
            lines = result.stdout.strip().splitlines()
            payload = json.loads(lines[-1]) if lines else {"status": "failed", "stderr": result.stderr[-3000:]}
            payload.update(probe=name, returncode=result.returncode)
        except subprocess.TimeoutExpired:
            payload = {"probe": name, "status": "timeout"}
        results.append(payload)
        print(f"{name}: {payload['status']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    return int(any(r["status"] not in (
        {"passed", "memory_limited"} if r["probe"] == "memory_pressure" else {"passed"}
    ) for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
