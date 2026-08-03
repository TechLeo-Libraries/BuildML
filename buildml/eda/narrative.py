"""The old narrative interface, kept working on top of the new findings.

BuildML's EDA once produced a list of sentences. It now produces findings with
severity, affected columns, and evidence, which is strictly more useful: and
code written against the old shape still exists.

Rather than maintaining two generators, this derives the strings from the
findings. One source of truth, and the sentences cannot drift away from the
structured claims they summarise.

See Also
--------
buildml.eda.findings : The structured form, and what new code should use.
"""

from __future__ import annotations

from typing import Any

from buildml.eda.findings import build_findings, narrative_view


def build_narrative(report_sections: dict[str, Any]) -> list[str]:
    """Build findings, then return just their sentences.

    Two steps in one call: run :func:`~buildml.eda.findings.build_findings` over
    the sections, then flatten to detail strings. Provided so older callers keep
    working without reaching into the findings module.

    Parameters
    ----------
    report_sections:
        The analyzer outputs, as passed to ``build_findings``.

    Returns
    -------
    list of str
        One sentence per finding, in generation order.

    Notes
    -----
    **New code should call ``build_findings`` directly.** The severity and the
    evidence are what let a reader tell an observation from a problem, and they
    are exactly what this discards.

    See Also
    --------
    buildml.eda.findings.build_findings : The structured version.
    """
    return narrative_view(build_findings(report_sections))
