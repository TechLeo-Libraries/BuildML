"""Generate a viewable BUILDML STATIC EDA HTML preview.

Reuses the synthetic dirty-classification frame from the Teaching Studio
launcher, runs research-format EDA, and writes a self-contained HTML file.
The sample frame is for preview only; the exporter itself is dataset-agnostic.

Usage (from repo root, with the project venv active)::

    .venv\\Scripts\\python.exe -u scripts/generate_static_eda_preview.py

Open the printed path in a browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

ARTIFACTS = ROOT / ".buildml-artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    from buildml import Session
    from launch_synthetic_eda_studio import build_synthetic_frame

    frame = build_synthetic_frame(n_rows=800, seed=7)
    csv_path = ARTIFACTS / "synthetic_dirty_classification.csv"
    frame.to_csv(csv_path, index=False)

    session = (
        Session.ingest(frame)
        .set_roles(
            {
                "customer_id": "id",
                "age": "feature",
                "income": "feature",
                "tenure_years": "feature",
                "monthly_spend": "feature",
                "risk_score": "feature",
                "city": "feature",
                "segment": "feature",
                "constant_flag": "feature",
                "signup_channel": "feature",
                "target_churn": "target",
            }
        )
        .split(test_size=0.25, stratify=True, random_state=7)
    )

    destination = ARTIFACTS / "static_eda_cockpit.html"
    report = session.eda(
        include_plots=True,
        max_plots=12,
        export_html=destination,
        html_format="research",
        show=False,
    )
    print(f"Wrote BUILDML STATIC EDA preview: {destination.resolve()}")
    print(f"Findings: {len(report.findings)} · Recommendations: {len(report.recommendation_details)}")
    print(f"Source CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
