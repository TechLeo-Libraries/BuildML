"""Internal JSON protocol for isolated ANN; native objects remain in this process."""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def main() -> None:
    """Build once, then query/extend until the parent closes its input stream.

    Read newline-delimited JSON from stdin and acknowledge operations on stdout.
    Native failures become protocol errors whenever Python can catch them;
    process-level faults are detected by the parent handle.

    Raises
    ------
    OSError
        If the communication stream cannot be read or written.
    """
    os.environ["BUILDML_ALLOW_CBR_INDUSTRY_WINDOWS"] = "1"
    from buildml.cbr.adapters.industry_ann import (
        add_vectors_to_ann_index,
        build_ann_index,
        query_ann_index,
    )
    index = library = metric = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["op"]
            if operation == "build":
                metric = request["metric"]
                index, library = build_ann_index(np.asarray(request["vectors"]), metric=metric,
                                                ef_construction=request["ef_construction"], m=request["m"])
                response = {"ready": True, "library": library}
            elif operation == "query":
                distances, labels = query_ann_index(index, library, np.asarray(request["queries"]),
                                                   k=request["k"], metric=metric)
                response = {"distances": distances.tolist(), "labels": labels.tolist()}
            elif operation == "add":
                vectors = np.asarray(request["vectors"], dtype=np.float32)
                if library == "hnswlib":
                    index.resize_index(request["start_id"] + len(vectors))
                add_vectors_to_ann_index(index, library, vectors, start_id=request["start_id"])
                response = {"added": len(vectors)}
            else:
                raise ValueError(f"Unknown ANN worker operation: {operation}")
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
