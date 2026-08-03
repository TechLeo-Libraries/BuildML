"""Helpers for normalizing analyzer payloads into JSON-safe dashboard shapes."""

from __future__ import annotations

from typing import Any


def flagged_column_names(flagged: Any) -> list[str]:
    """Pull column names out of a flag list, whichever shape it arrived in.

    Analyzers report flagged columns two ways. Some return plain names :
    ``['age', 'income']``: and some return rows with the measurement attached :
    ``[{'column': 'age', 'ks_stat': 0.3}]``. Both are reasonable for their
    purpose, and a caller that only wants names should not have to know which
    one it has.

    Parameters
    ----------
    flagged:
        A list of names, a list of dicts with a ``column`` key, or a mixture.
        Anything that is not a list yields an empty result rather than raising,
        since a missing section is a normal state.

    Returns
    -------
    list of str
        The names, in order. Entries that are neither a string nor a dict with a
        ``column`` are skipped.

    Notes
    -----
    **Duplicates are preserved.** A column flagged by two analyzers appears
    twice; deduplicate if that matters.

    Examples
    --------
    >>> flagged_column_names(["age", "income"])
    ['age', 'income']
    >>> flagged_column_names([{"column": "age", "ks_stat": 0.3}])
    ['age']
    >>> flagged_column_names(None)
    []
    """
    names: list[str] = []
    if not isinstance(flagged, list):
        return names
    for item in flagged:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("column") is not None:
            names.append(str(item["column"]))
    return names


def json_safe(value: Any) -> Any:
    """Convert anything into something ``json.dumps`` will accept.

    Analyzer output is full of values that look like numbers and are not:
    ``numpy.int64``, ``numpy.float32``, pandas timestamps, NumPy booleans. Every
    one of them raises ``TypeError`` from the JSON encoder, and the failure
    happens at serialisation time: after the whole response has been assembled
   : with a message naming a type rather than a field.

    Recurses through dicts and lists, converting leaves. NumPy and pandas
    scalars are unwrapped through their ``item()`` method, which yields the
    native Python equivalent. Anything left over becomes its string
    representation.

    Parameters
    ----------
    value:
        Anything at all.

    Returns
    -------
    Any
        A structure of dicts, lists, strings, numbers, booleans, and ``None``.

    Notes
    -----
    **The fallback to ``str`` never fails, and that is the point.** A dashboard
    that returns an odd-looking value is better than one that returns a 500. The
    cost is that a type nobody anticipated shows up as text rather than
    announcing itself.

    **Dict keys are stringified**, since JSON objects have no other kind of key.
    An integer-keyed dict comes back with string keys.

    **Deeply nested structures recurse.** Analyzer output is shallow; a
    pathological input could hit the recursion limit.

    Examples
    --------
    >>> json_safe({"n": 5, "ok": True, "sub": [1, None]})
    {'n': 5, 'ok': True, 'sub': [1, None]}
    >>> import numpy as np
    >>> json_safe({"count": np.int64(7)})
    {'count': 7}
    >>> json_safe(np.float64(0.5))
    0.5
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except Exception:
            return str(value)
    return str(value)
