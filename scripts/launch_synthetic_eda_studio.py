"""Build a rich dirty-classification synthetic dataset and launch Teaching Studio.

Usage (from repo root, with .venv active)::

    .venv\\Scripts\\python.exe -u scripts/launch_synthetic_eda_studio.py

Stops with Ctrl+C in this terminal (calls handle.stop()).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Unbuffered status lines when launched under redirected IO.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / ".buildml-artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def build_synthetic_frame(n_rows: int = 800, seed: int = 7) -> pd.DataFrame:
    """Dirty binary classification table with realistic quality issues."""
    rng = np.random.default_rng(seed)
    # Mild train/test drift via row order (later rows shift).
    age = rng.normal(42, 12, n_rows)
    age[n_rows // 2 :] += 4.5
    income = np.expm1(rng.normal(10.2, 0.55, n_rows))
    income[n_rows // 2 :] *= 1.18
    tenure = rng.gamma(2.2, 3.5, n_rows)
    # Correlated features.
    spend = 0.55 * income / 1000 + 0.35 * tenure + rng.normal(0, 2.5, n_rows)
    risk_score = 0.04 * age + 0.6 * (spend / (spend.std() + 1e-6)) + rng.normal(0, 0.8, n_rows)

    cities = np.array(["north", "south", "east", "west", "central", ""])
    city = rng.choice(cities, size=n_rows, p=[0.28, 0.22, 0.18, 0.14, 0.10, 0.08])
    # High-cardinality category.
    segment = np.array([f"seg-{rng.integers(0, 180):03d}" for _ in range(n_rows)])
    # Constants / id-like.
    constant_flag = np.full(n_rows, "always_on")
    customer_id = np.array([f"CUST-{10_000 + i}" for i in range(n_rows)])
    # Outliers.
    income[rng.choice(n_rows, size=8, replace=False)] *= 18
    spend[rng.choice(n_rows, size=6, replace=False)] += 80

    # Imbalanced target with signal + noise.
    logits = (
        -2.4
        + 0.035 * (age - 40)
        + 0.00002 * (income - income.mean())
        + 0.12 * (spend - spend.mean())
        + 0.18 * risk_score
        + rng.normal(0, 0.9, n_rows)
    )
    prob = 1 / (1 + np.exp(-logits))
    target = (rng.random(n_rows) < np.clip(prob * 0.22, 0.03, 0.28)).astype(int)

    frame = pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "income": income,
            "tenure_years": tenure,
            "monthly_spend": spend,
            "risk_score": risk_score,
            "city": city,
            "segment": segment,
            "constant_flag": constant_flag,
            "signup_channel": rng.choice(["web", "mobile", "partner", None], size=n_rows),
            "target_churn": target,
        }
    )
    # Missingness (MAR-ish on age/income/city).
    miss_age = rng.random(n_rows) < 0.08
    miss_income = rng.random(n_rows) < 0.11
    miss_spend = rng.random(n_rows) < 0.05
    frame.loc[miss_age, "age"] = np.nan
    frame.loc[miss_income, "income"] = np.nan
    frame.loc[miss_spend, "monthly_spend"] = np.nan
    frame.loc[rng.random(n_rows) < 0.06, "city"] = None
    return frame


def main() -> None:
    from buildml import Session
    from buildml.dashboard.launch import DashboardLaunchError

    frame = build_synthetic_frame()
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
    handle = None
    last_error: Exception | None = None
    for port in range(8765, 8795):
        try:
            handle = session.eda_app(
                host="127.0.0.1",
                port=port,
                open_browser=True,
                title="BuildML Synthetic Dirty Classification",
                blocking=False,
            )
            break
        except DashboardLaunchError as exc:
            last_error = exc
            continue
    if handle is None:
        raise RuntimeError(f"Could not bind EDA App near 8765: {last_error}")
    url_path = ARTIFACTS / "eda_studio_url.txt"
    url_path.write_text(handle.url + "\n", encoding="utf-8")
    print(f"Synthetic CSV: {csv_path}")
    print(f"EDA App URL: {handle.url}")
    print(f"URL also written to: {url_path}")
    print("Stop with Ctrl+C in this terminal (calls handle.stop()).")
    print("Keeping process alive so the background server stays up...")
    try:
        while handle.is_running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        handle.stop()
        print("Stopped EDA App.")


if __name__ == "__main__":
    main()
