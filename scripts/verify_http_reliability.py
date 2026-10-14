"""Bounded real HTTP load, process restart and prediction-parity evidence.

This measures one loopback deployment, not an uptime SLA. Traffic pauses during
deliberate shutdowns; recovery time and downtime are reported separately.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(bundle, port, stop_file):
    from buildml.serving.launch import serve_bundle

    logging.disable(logging.CRITICAL)
    handle = serve_bundle(bundle, port=port, trusted=True, api_keys="probe-key")
    try:
        while not Path(stop_file).exists() and handle.is_running:
            time.sleep(.1)
    finally:
        handle.stop()


def verify(seconds, out):
    import httpx
    import numpy as np
    import pandas as pd
    import psutil
    from sklearn.linear_model import LinearRegression

    from buildml import Session

    if seconds <= 0:
        raise ValueError("seconds must be positive")
    started = time.monotonic()
    result = {"status": "running", "requested_load_seconds": seconds,
              "scope": "real loopback HTTP; traffic paused during deliberate recovery; no deployment SLA",
              "epochs": [], "recoveries": [], "errors": []}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(result, indent=2, allow_nan=False))

    with tempfile.TemporaryDirectory(prefix="buildml-http-reliability-") as tmp:
        frame = pd.DataFrame({"x": np.arange(80, dtype=float)})
        frame["y"] = 2 * frame.x + 3
        session = (Session.ingest(frame).set_roles({"x": "feature", "y": "target"})
                   .split(test_size=.2, random_state=7).fit(LinearRegression(), task="regression"))
        bundle = session.save_pipeline(Path(tmp) / "bundle")
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        process = None
        try:
            for epoch, (clients, batch) in enumerate([(1, 1), (4, 16), (8, 64)]):
                stop_file = Path(tmp) / f"stop-{epoch}"
                launch_started = time.monotonic()
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", "--bundle", str(bundle),
                     "--port", str(port), "--stop-file", str(stop_file)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                with httpx.Client(timeout=2) as client:
                    deadline = launch_started + 60
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"server exited during startup: {process.returncode}")
                        try:
                            response = client.get(url + "/health")
                            if response.status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() >= deadline:
                            raise TimeoutError("server did not become healthy within 60 seconds")
                        time.sleep(.05)
                recovery = time.monotonic() - launch_started
                if epoch:
                    result["recoveries"].append({"after": "graceful" if epoch == 1 else "forced",
                                                 "startup_seconds": recovery})
                deadline = time.monotonic() + seconds / 3

                def traffic(worker_id, epoch=epoch, batch=batch, deadline=deadline):
                    rng = random.Random(epoch * 100 + worker_id)
                    samples, errors, count = [], [], 0
                    rows = [{"x": float(i - batch // 2)} for i in range(batch)]
                    expected = [2 * row["x"] + 3 for row in rows]
                    with httpx.Client(base_url=url, timeout=10) as client:
                        while time.monotonic() < deadline:
                            t0 = time.perf_counter()
                            try:
                                if count % 101 == 0:
                                    response = client.post("/predict", json={"rows": rows})
                                    assert response.status_code == 401, response.text
                                elif count % 103 == 0:
                                    response = client.post("/predict", headers={"X-API-Key": "probe-key"},
                                                           json={"rows": [{"wrong": 1}]})
                                    assert response.status_code in {400, 422}, response.text
                                else:
                                    response = client.post("/predict", headers={"X-API-Key": "probe-key"},
                                                           json={"rows": rows})
                                    assert response.status_code == 200, response.text
                                    np.testing.assert_allclose(response.json()["predictions"], expected,
                                                               rtol=1e-10, atol=1e-10)
                            except Exception as exc:
                                if len(errors) < 20:
                                    errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")
                                # Fail quickly instead of masking failures with a success rate.
                                return {"requests": count, "samples": samples, "errors": errors}
                            count += 1
                            elapsed = (time.perf_counter() - t0) * 1000
                            if len(samples) < 2048:
                                samples.append(elapsed)
                            else:
                                index = rng.randrange(count)
                                if index < len(samples):
                                    samples[index] = elapsed
                    return {"requests": count, "samples": samples, "errors": errors}

                rss = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as executor:
                    futures = [executor.submit(traffic, i) for i in range(clients)]
                    next_progress = time.monotonic()
                    while not all(f.done() for f in futures):
                        server = psutil.Process(process.pid)
                        tree = [server, *server.children(recursive=True)]
                        rss.append({"elapsed_seconds": time.monotonic() - started,
                                    "mib": sum(p.memory_info().rss for p in tree) / 1024 ** 2})
                        if time.monotonic() >= next_progress:
                            print(f"HTTP epoch={epoch} clients={clients} batch={batch} RSS={rss[-1]['mib']:.2f}MiB", flush=True)
                            next_progress = time.monotonic() + 60
                        time.sleep(min(1, max(.01, seconds / 30)))
                    outcomes = [f.result() for f in futures]
                errors = [error for item in outcomes for error in item["errors"]]
                samples = [value for item in outcomes for value in item["samples"]]
                result["epochs"].append({"clients": clients, "batch_size": batch,
                                          "requests": sum(item["requests"] for item in outcomes),
                                          "latency_p95_ms": float(np.percentile(samples, 95)) if samples else None,
                                          "latency_sampling": "reservoir of up to 2048 requests per client",
                                          "rss_samples": rss, "errors": errors})
                result["errors"].extend(errors)
                save()
                if errors:
                    raise RuntimeError("HTTP scoring or request-contract failure")
                stop_started = time.monotonic()
                if epoch == 1:
                    # Windows venv launchers can have a separate interpreter child.
                    children = psutil.Process(process.pid).children(recursive=True)
                    for child in children:
                        child.kill()
                    process.kill()
                    _, alive = psutil.wait_procs(children, timeout=15)
                    if alive:
                        raise RuntimeError("server interpreter child did not terminate")
                else:
                    stop_file.touch()
                process.wait(timeout=15)
                close_deadline = time.monotonic() + 5
                while True:
                    with socket.socket() as sock:
                        sock.settimeout(.2)
                        closed = sock.connect_ex(("127.0.0.1", port)) != 0
                    if closed:
                        break
                    if time.monotonic() >= close_deadline:
                        raise RuntimeError("server port remained open after shutdown")
                    time.sleep(.05)
                result["epochs"][-1]["shutdown_seconds"] = time.monotonic() - stop_started
                result["epochs"][-1]["shutdown"] = "forced" if epoch == 1 else "graceful"
            result["status"] = "passed"
        except Exception as exc:
            result["status"] = "failed"
            result["errors"].append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if process is not None and process.poll() is None:
                for child in psutil.Process(process.pid).children(recursive=True):
                    child.kill()
                process.kill()
                process.wait(timeout=15)
            result["elapsed_seconds"] = time.monotonic() - started
            save()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--out", default="artifacts/http-reliability.json")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--bundle")
    parser.add_argument("--port", type=int)
    parser.add_argument("--stop-file")
    args = parser.parse_args()
    if args.worker:
        worker(args.bundle, args.port, args.stop_file)
    else:
        result = verify(args.seconds, args.out)
        print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                          "requests": sum(e["requests"] for e in result["epochs"]),
                          "recoveries": result["recoveries"], "evidence": args.out}, allow_nan=False))
