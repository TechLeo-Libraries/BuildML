"""License-clear synthetic / sklearn datasets for proof projects.

Real OpenML / remote downloads can be added later; synthetics here are
deterministic, documented, and free of proprietary licensing risk.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def load_credit_approval_synthetic(
    *,
    n: int = 1200,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Synthetic credit/loan approval table with missingness and mixed types."""
    rng = np.random.default_rng(seed)
    age = rng.normal(38, 12, size=n).clip(18, 80)
    income = rng.lognormal(10.8, 0.55, size=n)
    debt_ratio = rng.beta(2.0, 5.0, size=n)
    employment_years = rng.exponential(5.0, size=n).clip(0, 40)
    region = rng.choice(["N", "S", "E", "W"], size=n, p=[0.3, 0.25, 0.25, 0.2])
    product = rng.choice(["personal", "auto", "home"], size=n, p=[0.5, 0.3, 0.2])
    # Latent score → approval (non-linear, region shift).
    region_bias = {"N": 0.1, "S": -0.05, "E": 0.0, "W": -0.1}
    logit = (
        -1.2
        + 0.035 * (age - 35)
        + 0.7 * np.log1p(income) / 10
        - 2.5 * debt_ratio
        + 0.08 * employment_years
        + np.array([region_bias[r] for r in region])
        + rng.normal(0, 0.35, size=n)
    )
    approved = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    frame = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "debt_ratio": debt_ratio,
            "employment_years": employment_years,
            "region": region,
            "product": product,
            "approved": approved,
        }
    )
    # Inject train-like missingness patterns (MCAR-ish).
    miss_age = rng.random(n) < 0.08
    miss_income = rng.random(n) < 0.05
    frame.loc[miss_age, "age"] = np.nan
    frame.loc[miss_income, "income"] = np.nan
    meta = {
        "name": "credit_approval_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(n),
        "target": "approved",
        "positive_rate": float(approved.mean()),
        "notes": "Synthetic credit underwriting labels; not a real FCRA dataset.",
    }
    return frame, meta


def load_telco_churn_synthetic(
    *,
    n: int = 1600,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Telco-style churn table (synthetic stand-in for IBM Telco Churn)."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, size=n)
    monthly = rng.normal(65, 20, size=n).clip(20, 150)
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        size=n,
        p=[0.55, 0.25, 0.2],
    )
    internet = rng.choice(["dsl", "fiber", "none"], size=n, p=[0.35, 0.45, 0.2])
    support_tickets = rng.poisson(1.2, size=n)
    contract_map = {"month-to-month": 0.35, "one-year": -0.15, "two-year": -0.4}
    internet_map = {"dsl": 0.05, "fiber": 0.15, "none": -0.2}
    logit = (
        -0.8
        - 0.03 * tenure
        + 0.015 * monthly
        + np.array([contract_map[c] for c in contract])
        + np.array([internet_map[i] for i in internet])
        + 0.18 * support_tickets
        + rng.normal(0, 0.4, size=n)
    )
    churn = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    frame = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "contract": contract,
            "internet_service": internet,
            "support_tickets": support_tickets,
            "churn": churn,
        }
    )
    meta = {
        "name": "telco_churn_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(n),
        "target": "churn",
        "positive_rate": float(churn.mean()),
        "notes": "Synthetic telco churn; mirrors IBM Telco schema themes only.",
    }
    return frame, meta


def load_intrusion_anomaly_synthetic(
    *,
    n_normal: int = 2000,
    n_attack: int = 120,
    seed: int = 11,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Network-flow-like anomaly table with rare attack rows."""
    rng = np.random.default_rng(seed)
    normal = pd.DataFrame(
        {
            "duration": rng.exponential(2.0, n_normal),
            "src_bytes": rng.lognormal(5.0, 0.8, n_normal),
            "dst_bytes": rng.lognormal(4.5, 0.9, n_normal),
            "count": rng.poisson(8, n_normal),
            "srv_count": rng.poisson(6, n_normal),
            "same_srv_rate": rng.beta(5, 2, n_normal),
            "dst_host_count": rng.integers(1, 255, n_normal),
            "is_attack": np.zeros(n_normal, dtype=int),
        }
    )
    attack = pd.DataFrame(
        {
            "duration": rng.exponential(0.4, n_attack),
            "src_bytes": rng.lognormal(8.5, 0.6, n_attack),
            "dst_bytes": rng.lognormal(2.0, 1.2, n_attack),
            "count": rng.poisson(40, n_attack),
            "srv_count": rng.poisson(35, n_attack),
            "same_srv_rate": rng.beta(2, 5, n_attack),
            "dst_host_count": rng.integers(50, 255, n_attack),
            "is_attack": np.ones(n_attack, dtype=int),
        }
    )
    frame = pd.concat([normal, attack], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    meta = {
        "name": "network_intrusion_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(len(frame)),
        "n_attack": int(n_attack),
        "target": "is_attack",
        "notes": "KDD-inspired synthetic flows; not the full KDD Cup 1999 corpus.",
    }
    return frame, meta


def load_store_sales_synthetic(
    *,
    n_days: int = 730,
    seed: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Daily store sales with trend, weekly seasonality, and promo spikes."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    t = np.arange(n_days)
    weekly = 8 * np.sin(2 * np.pi * t / 7)
    annual = 15 * np.sin(2 * np.pi * t / 365.25)
    promo = (rng.random(n_days) < 0.08).astype(float)
    sales = (
        100
        + 0.04 * t
        + weekly
        + annual
        + 25 * promo
        + rng.normal(0, 4.0, n_days)
    ).clip(5, None)
    frame = pd.DataFrame(
        {
            "date": dates,
            "promo": promo.astype(int),
            "sales": sales,
        }
    )
    meta = {
        "name": "store_sales_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(n_days),
        "freq": "D",
        "target": "sales",
        "time_column": "date",
        "notes": "Synthetic retail demand; time_split required for honest eval.",
    }
    return frame, meta


def load_customer_segments_synthetic(
    *,
    n_per: int = 250,
    seed: int = 19,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Customer RFM-like features with latent segment labels (for external val)."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [20, 2, 50],
            [80, 8, 200],
            [40, 12, 30],
            [60, 4, 120],
        ],
        dtype=float,
    )
    frames = []
    for i, center in enumerate(centers):
        block = rng.normal(center, [8, 1.5, 25], size=(n_per, 3))
        frames.append(
            pd.DataFrame(
                {
                    "recency_days": block[:, 0].clip(1, 365),
                    "frequency": block[:, 1].clip(0, None),
                    "monetary": block[:, 2].clip(1, None),
                    "true_segment": i,
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    meta = {
        "name": "customer_segments_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(len(frame)),
        "n_segments": 4,
        "external_label": "true_segment",
        "notes": "Latent segments used only for external cluster validation.",
    }
    return frame, meta


def load_support_kb_corpus() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Small support knowledge-base corpus + retrieval relevance judgments."""
    docs = [
        {
            "doc_id": "billing-refund",
            "text": (
                "To request a refund, open Billing > Invoices, select the charge, "
                "and choose Request refund. Refunds post within 5–10 business days. "
                "Subscriptions canceled mid-cycle are prorated."
            ),
            "metadata": {"topic": "billing"},
        },
        {
            "doc_id": "password-reset",
            "text": (
                "Reset a forgotten password from the login screen using Forgot password. "
                "Enter the account email; the reset link expires after 30 minutes. "
                "SSO users must reset credentials with their identity provider."
            ),
            "metadata": {"topic": "account"},
        },
        {
            "doc_id": "data-export",
            "text": (
                "Workspace admins can export project data from Settings > Data export. "
                "Exports are ZIP archives of CSV files. Large workspaces may take up to "
                "an hour; you receive an email when the export is ready."
            ),
            "metadata": {"topic": "admin"},
        },
        {
            "doc_id": "rate-limits",
            "text": (
                "API rate limits are 120 requests per minute on the standard plan and "
                "1200 on enterprise. HTTP 429 responses include a Retry-After header. "
                "Burst traffic should use exponential backoff."
            ),
            "metadata": {"topic": "api"},
        },
        {
            "doc_id": "leakage-eval",
            "text": (
                "Never index labeled evaluation answers into the retrieval corpus. "
                "Doing so contaminates offline RAG metrics and overstates production quality."
            ),
            "metadata": {"topic": "ml-hygiene"},
        },
    ]
    judgments = {
        "how do I get a refund for a charge": ["billing-refund"],
        "forgot password reset link expired": ["password-reset"],
        "export workspace csv data": ["data-export"],
        "api returns 429 too many requests": ["rate-limits"],
        "evaluation contamination indexed answers": ["leakage-eval"],
    }
    return docs, judgments
