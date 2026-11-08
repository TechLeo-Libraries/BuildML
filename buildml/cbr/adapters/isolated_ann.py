"""Keep native ANN build/query faults in a persistent child process."""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from typing import Any

import numpy as np

from buildml.core.errors import ValidationError


class IsolatedAnnIndex:
    """Serializable ANN handle with serialized, timeout-bounded worker calls.

    Native objects live only in the worker. Pickling stores vectors/configuration,
    not the process; restoring builds lazily on first use. A worker failure is
    reported rather than substituted with another retrieval algorithm.
    """

    def __init__(self, vectors: np.ndarray, *, metric: str, ef_construction: int, m: int):
        """Start a native worker and retain a parent snapshot for restoration.

        The parent keeps serializable vectors while the child owns native
        objects, so a native process fault cannot take down the calling Session.

        Parameters
        ----------
        vectors:
            Finite case vectors validated by the industry adapter.
        metric:
            Euclidean or cosine distance convention.
        ef_construction:
            HNSW construction exploration budget.
        m:
            HNSW connections per node.

        Raises
        ------
        ValidationError
            If worker initialization fails or exceeds its response deadline.
        """
        self.vectors = np.asarray(vectors, dtype=np.float32).copy()
        self.metric, self.ef_construction, self.m = metric, ef_construction, m
        self.timeout = 60.0
        self._process: Any = None
        self._lock = threading.Lock()
        self._responses: queue.Queue = queue.Queue()
        self._start()

    @property
    def ntotal(self) -> int:
        """Number of indexed cases, used to bound neighbour requests."""
        return len(self.vectors)

    def _start(self) -> None:
        self._process = subprocess.Popen(
            [sys.executable, "-m", "buildml.cbr.adapters.ann_worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        responses, stream = self._responses, self._process.stdout

        def read_responses():
            try:
                for line in stream:
                    responses.put(line)
            except (OSError, ValueError):
                # Closing a failed worker can race its daemon stream reader.
                pass
            finally:
                responses.put(None)

        threading.Thread(target=read_responses, daemon=True).start()
        self._call({"op": "build", "vectors": self.vectors.tolist(), "metric": self.metric,
                    "ef_construction": self.ef_construction, "m": self.m})

    def _call(self, payload: dict) -> dict:
        try:
            self._process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
            self._process.stdin.flush()
            line = self._responses.get(timeout=self.timeout)
            if line is None:
                raise RuntimeError("native ANN worker exited unexpectedly")
            result = json.loads(line)
            if not isinstance(result, dict):
                raise RuntimeError("native ANN worker returned an invalid response")
            if "error" in result:
                raise RuntimeError(result["error"])
            return result
        except (OSError, ValueError, RuntimeError, queue.Empty) as exc:
            self.close()
            raise ValidationError(
                "Isolated ANN worker failed or timed out; the parent process is intact. "
                f"Refit the index or choose backend='sklearn'. Detail: {exc}"
            ) from exc

    def query(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search with the worker, returning distances and case positions.

        A restored handle rebuilds its native index before searching. Calls are
        serialized because the worker owns one mutable native index.

        Parameters
        ----------
        queries:
            Query vectors with the same feature dimension as the cases.
        k:
            Number of neighbours, bounded by case count through the adapter.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Distances and aligned case-memory positions for each query.

        Raises
        ------
        ValidationError
            If the worker fails, times out or returns an invalid response.
        """
        with self._lock:
            if self._process is None:
                self._responses = queue.Queue()
                self._start()
            result = self._call({"op": "query", "queries": np.asarray(queries).tolist(), "k": k})
            return np.asarray(result["distances"]), np.asarray(result["labels"], dtype=int)

    def add(self, vectors: np.ndarray, start_id: int) -> None:
        """Extend the worker and parent snapshot with aligned case positions.

        Update the snapshot only after the worker acknowledges retention, so
        failed native additions cannot silently alter the retained case count.

        Parameters
        ----------
        vectors:
            Additional case vectors already validated by the adapter.
        start_id:
            First new case position; must equal the current case count.

        Raises
        ------
        ValidationError
            If positions are misaligned or the worker fails.
        """
        if start_id != self.ntotal:
            raise ValidationError("ANN retained IDs must continue the case-memory positions.")
        with self._lock:
            if self._process is None:
                self._responses = queue.Queue()
                self._start()
            self._call({"op": "add", "vectors": vectors.tolist(), "start_id": start_id})
            self.vectors = np.vstack([self.vectors, vectors])

    def close(self) -> None:
        """Release the worker without leaving a native helper running."""
        process, self._process = self._process, None
        if process is not None:
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            if process.stdout:
                process.stdout.close()

    def __getstate__(self):
        return {"vectors": self.vectors, "metric": self.metric,
                "ef_construction": self.ef_construction, "m": self.m, "timeout": self.timeout}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._process = None
        self._lock = threading.Lock()
        self._responses = queue.Queue()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
