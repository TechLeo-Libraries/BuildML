"""License-clear synthetic and real public datasets for proof projects.

Synthetics here are deterministic and free of proprietary licensing risk.
Real loaders wrap sklearn built-ins (offline) and optional OpenML fetches
(network / cache) with explicit provenance metadata for results.json.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _real_meta(
    *,
    name: str,
    source: str,
    license_provenance: str,
    n_rows: int,
    n_features: int,
    task: str,
    **extra: Any,
) -> dict[str, Any]:
    """Standard provenance envelope for REAL_PUBLIC_DATASET proofs."""
    meta: dict[str, Any] = {
        "name": name,
        "dataset_identity": name,
        "source": source,
        "license": license_provenance,
        "provenance": license_provenance,
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "task": task,
        "real_public_dataset": True,
        "evidence_tier": "REAL_PUBLIC_DATASET",
    }
    meta.update(extra)
    return meta


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
        + rng.normal(0, 0.55, size=n)
    )
    # Bernoulli labels (not hard-threshold) leave irreducible classification error.
    approved = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
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
    """Network-flow-like anomaly table with rare, partially overlapping attacks.

    Attack margins are softer than well-separated blobs, and a small fraction of
    rows receive flipped labels so unsupervised detectors cannot trivially hit
    perfect holdout scores.
    """
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
    # Milder shift vs normal so IsolationForest / HBOS leave residual error.
    attack = pd.DataFrame(
        {
            "duration": rng.exponential(1.1, n_attack),
            "src_bytes": rng.lognormal(6.4, 0.9, n_attack),
            "dst_bytes": rng.lognormal(3.5, 1.0, n_attack),
            "count": rng.poisson(18, n_attack),
            "srv_count": rng.poisson(14, n_attack),
            "same_srv_rate": rng.beta(3, 3, n_attack),
            "dst_host_count": rng.integers(20, 255, n_attack),
            "is_attack": np.ones(n_attack, dtype=int),
        }
    )
    frame = pd.concat([normal, attack], ignore_index=True)
    # ~4% label flips keep irreducible evaluation error for tuned detectors.
    flip = rng.random(len(frame)) < 0.04
    frame.loc[flip, "is_attack"] = 1 - frame.loc[flip, "is_attack"]
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    meta = {
        "name": "network_intrusion_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(len(frame)),
        "n_attack": int(int(frame["is_attack"].sum())),
        "target": "is_attack",
        "difficulty": "partial_overlap_with_label_noise",
        "notes": (
            "KDD-inspired synthetic flows with soft attack margins and ~4% "
            "label flips; not the full KDD Cup 1999 corpus."
        ),
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
            [30, 3, 70],
            [70, 7, 160],
            [45, 9, 55],
            [55, 5, 110],
        ],
        dtype=float,
    )
    frames = []
    for i, center in enumerate(centers):
        # Wider within-cluster spread so ARI/NMI cannot trivially hit ~1.0.
        block = rng.normal(center, [18, 3.5, 45], size=(n_per, 3))
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
    # Boundary / mixed-membership rows blur segment edges.
    n_boundary = max(1, int(0.12 * len(frame)))
    boundary_idx = rng.choice(len(frame), size=n_boundary, replace=False)
    for idx in boundary_idx:
        a, b = rng.choice(4, size=2, replace=False)
        blend = 0.5 * centers[a] + 0.5 * centers[b] + rng.normal(0, [6, 1.2, 15], size=3)
        frame.loc[idx, "recency_days"] = float(np.clip(blend[0], 1, 365))
        frame.loc[idx, "frequency"] = float(max(blend[1], 0))
        frame.loc[idx, "monetary"] = float(max(blend[2], 1))
        frame.loc[idx, "true_segment"] = int(a if rng.random() < 0.5 else b)
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
    """Adversarial support KB corpus + retrieval judgments.

    Includes near-duplicate distractors and paraphrased queries so lexical
    hashing cannot trivially score perfect MRR/nDCG on every query.
    """
    docs = [
        {
            "doc_id": "billing-refund",
            "text": (
                "To request a refund, open Billing > Invoices, select the charge, "
                "and choose Request refund. Refunds post within 5-10 business days. "
                "Subscriptions canceled mid-cycle are prorated."
            ),
            "metadata": {"topic": "billing"},
        },
        {
            "doc_id": "billing-credit-note",
            "text": (
                "Credit notes are accounting adjustments, not cash refunds. "
                "Open Billing > Credits to apply a credit note to a future invoice. "
                "Do not use Request refund when you only need a credit note."
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
            "doc_id": "password-mfa",
            "text": (
                "Multi-factor authentication codes are separate from password resets. "
                "If login fails after a password change, check the authenticator app "
                "or SMS OTP before requesting another reset link."
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
            "doc_id": "data-import",
            "text": (
                "CSV import uses Settings > Data import. Imports never create ZIP "
                "exports. Mapping columns incorrectly will skip rows silently."
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
            "doc_id": "rate-limits-webhooks",
            "text": (
                "Webhook delivery retries are not API rate limits. Failed webhooks "
                "retry with backoff for 24 hours. HTTP 429 on the REST API is unrelated."
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
        {
            "doc_id": "leakage-train-test",
            "text": (
                "Train/test leakage in tabular ML is different from RAG corpus "
                "contamination. Keep holdout rows out of fitting; keep judgment "
                "answers out of the index."
            ),
            "metadata": {"topic": "ml-hygiene"},
        },
        {
            "doc_id": "shipping-sla",
            "text": (
                "Standard shipping arrives in 3-5 business days. Express shipping "
                "is overnight in-region only. Tracking updates every four hours."
            ),
            "metadata": {"topic": "logistics"},
        },
        {
            "doc_id": "noise-marketing",
            "text": (
                "Marketing campaigns can mention refunds, passwords, exports, and "
                "API limits in promotional copy without being support procedures."
            ),
            "metadata": {"topic": "noise"},
        },
    ]
    # Paraphrases + distractor pressure: relevant ids are intentional, not
    # keyword-identical to a single doc.
    judgments = {
        "customer wants money back for a double charge": ["billing-refund"],
        "need a credit on next invoice not cash": ["billing-credit-note"],
        "cannot sign in and the email link died": ["password-reset"],
        "otp fails after changing credentials": ["password-mfa"],
        "download all project tables as spreadsheet archive": ["data-export"],
        "upload csv mappings into the workspace": ["data-import"],
        "rest client keeps getting too many requests": ["rate-limits"],
        "outbound event retries after failure": ["rate-limits-webhooks"],
        "why offline rag scores look perfect but prod fails": ["leakage-eval"],
        "holdout rows used during model fitting": ["leakage-train-test"],
        "when does express parcel arrive overnight": ["shipping-sla"],
    }
    return docs, judgments


# Per-queue sentence pools. Each queue has vocabulary that overlaps the others
# (dates, ids, "team", "support") so a text classifier has to learn a real
# discriminative signal instead of latching onto one giveaway keyword.
_TICKET_QUEUES: dict[str, dict[str, tuple[str, ...]]] = {
    "billing": {
        "opening": (
            "Invoice INV-{ref} charged {amount} twice on the same card.",
            "The renewal quote said {amount} but the invoice came to almost double.",
            "We were billed {amount} for seats that were removed last cycle.",
            "A proration credit of {amount} never appeared on invoice INV-{ref}.",
        ),
        "detail": (
            "Finance has already flagged the discrepancy in their reconciliation.",
            "The line item has no description, so nobody here can approve it.",
            "Our purchase order caps monthly spend well below that figure.",
            "The same charge was disputed in the previous quarter as well.",
        ),
        "ask": (
            "Please reverse the duplicate and reissue a corrected invoice.",
            "We need a written breakdown before the payment run on the 28th.",
            "Can you confirm the refund amount and the expected posting date?",
        ),
    },
    "shipping": {
        "opening": (
            "Order ORD-{ref} was promised for the 3rd and arrived nine days late.",
            "The courier marked shipment ORD-{ref} as delivered but nothing arrived.",
            "Two of the four cartons in shipment ORD-{ref} were crushed in transit.",
            "Tracking for ORD-{ref} has not updated since it left the depot.",
        ),
        "detail": (
            "The packaging was soaked and the outer seal was already broken.",
            "A customs hold added four days that nobody notified us about.",
            "The warehouse signature on the manifest does not match anyone here.",
            "The replacement carton shipped to a previous address on file.",
        ),
        "ask": (
            "Please send a replacement on an expedited service at your cost.",
            "We need a revised delivery date we can give the site team today.",
            "Can you open a carrier claim and share the reference with us?",
        ),
    },
    "account": {
        "opening": (
            "Single sign-on stopped working for the whole workspace this morning.",
            "The onboarding portal rejects the invite link for every new hire.",
            "Two-factor codes are refused even though the clock is synced.",
            "An admin was removed from the workspace and cannot be restored.",
        ),
        "detail": (
            "The error page shows no code, only a generic try again later message.",
            "Password resets arrive but the link has already expired on arrival.",
            "The identity provider logs show the assertion being accepted.",
            "Role permissions reverted to read-only after the last release.",
        ),
        "ask": (
            "Please restore access for the affected users before the audit.",
            "Can you confirm whether the change came from a release on your side?",
            "We need a workaround today; the team is blocked from every project.",
        ),
    },
    "hardware": {
        "opening": (
            "The hinge on unit HW-{ref} snapped within a week of light use.",
            "Unit HW-{ref} overheats and shuts down under a normal workload.",
            "The display on unit HW-{ref} flickers whenever the lid is moved.",
            "Battery life on unit HW-{ref} dropped to under an hour after a month.",
        ),
        "detail": (
            "The plastic around the mount was already stressed out of the box.",
            "Diagnostics report no fault, yet the device powers off regardless.",
            "A second identical unit shows exactly the same behaviour.",
            "The firmware update did not change anything measurable.",
        ),
        "ask": (
            "Please advise on a warranty replacement rather than another repair.",
            "Can you confirm whether this is a known batch defect?",
            "We need a loan unit while the assessment is in progress.",
        ),
    },
}

# Queue-agnostic sentences. A share of rows is built only from these, so those
# tickets genuinely do not say which queue they belong to. That gives the corpus
# an irreducible error floor and keeps the proof from reporting a perfect score
# that no real routing problem would ever produce.
_TICKET_GENERIC: dict[str, tuple[str, ...]] = {
    "opening": (
        "Following up on the case raised by our team earlier this week.",
        "We opened a ticket about this on the 12th and have had no reply.",
        "This is the third time we are writing about the same problem.",
        "Reference REF-{ref}: the situation has not changed since Monday.",
    ),
    "detail": (
        "Nobody on your side has confirmed who owns this now.",
        "The account manager suggested we escalate through support instead.",
        "The last update we received simply asked us to wait.",
        "Our team has already sent the details twice.",
    ),
    "ask": (
        "Please confirm who is handling this and by when.",
        "We need an update today so the team can plan around it.",
        "Can you escalate this to whoever is responsible?",
    ),
}

_TICKET_CHANNELS: tuple[str, ...] = ("web", "email", "phone", "chat")


def load_support_tickets_synthetic(
    *,
    n: int = 900,
    seed: int = 11,
    ambiguous_rate: float = 0.18,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Synthetic labeled support tickets for document classification.

    Each row is a short multi-sentence ticket plus the queue it was routed to,
    which makes it a single-label document-classification problem with an honest
    amount of vocabulary overlap between classes.

    Parameters
    ----------
    n:
        Number of tickets, balanced across the four queues.
    seed:
        RNG seed. The frame is shuffled with the same seed.
    ambiguous_rate:
        Share of tickets composed entirely from queue-agnostic sentences. Those
        rows carry a label no reader could recover from the text, which puts a
        deliberate ceiling on achievable accuracy — roughly
        ``1 - ambiguous_rate * 3/4`` for four balanced queues.
    """
    if not 0.0 <= float(ambiguous_rate) < 1.0:
        raise ValueError("ambiguous_rate must be in [0.0, 1.0).")
    rng = np.random.default_rng(seed)
    queues = list(_TICKET_QUEUES)
    rows: list[dict[str, Any]] = []
    n_ambiguous = 0
    for index in range(n):
        queue = queues[index % len(queues)]
        ambiguous = bool(rng.random() < float(ambiguous_rate))
        pools = _TICKET_GENERIC if ambiguous else _TICKET_QUEUES[queue]
        n_ambiguous += int(ambiguous)
        ref = int(rng.integers(10_000, 99_999))
        amount = f"${int(rng.integers(2, 40)) * 50:,}"
        opening = str(rng.choice(pools["opening"])).format(ref=ref, amount=amount)
        detail = str(rng.choice(pools["detail"]))
        ask = str(rng.choice(pools["ask"]))
        parts = [opening, detail, ask]
        if rng.random() < 0.35:
            parts.insert(2, str(rng.choice(pools["detail"])))
        rows.append(
            {
                "ticket_id": f"T{100_000 + index}",
                "body": " ".join(parts),
                "channel": str(rng.choice(_TICKET_CHANNELS)),
                "queue": queue,
            }
        )
    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    meta = {
        "name": "support_tickets_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(len(frame)),
        "target": "queue",
        "text_column": "body",
        "classes": sorted(queues),
        "ambiguous_rate_requested": float(ambiguous_rate),
        "ambiguous_rows": int(n_ambiguous),
        "expected_accuracy_ceiling": round(
            1.0 - (n_ambiguous / max(1, n)) * (1.0 - 1.0 / len(queues)), 4
        ),
        "notes": (
            "Synthetic support tickets composed from per-queue sentence pools, "
            "plus a share of deliberately queue-agnostic tickets that no reader "
            "could route from the text alone. Not a real customer-support corpus."
        ),
    }
    return frame, meta


def load_mortgage_default_synthetic(
    *,
    n: int = 1400,
    seed: int = 31,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Synthetic mortgage default table (distinct from consumer loan approval)."""
    rng = np.random.default_rng(seed)
    ltv = rng.beta(5, 3, size=n) * 1.2
    dti = rng.beta(2.5, 4.0, size=n)
    credit_score = rng.normal(680, 55, size=n).clip(480, 850)
    rate = rng.normal(6.2, 1.1, size=n).clip(2.5, 12.0)
    term_years = rng.choice([15, 20, 30], size=n, p=[0.15, 0.2, 0.65])
    property_type = rng.choice(["sfr", "condo", "townhome"], size=n, p=[0.6, 0.25, 0.15])
    type_bias = {"sfr": -0.1, "condo": 0.15, "townhome": 0.05}
    logit = (
        -1.5
        + 2.8 * (ltv - 0.8)
        + 2.2 * dti
        - 0.008 * (credit_score - 650)
        + 0.12 * (rate - 5.5)
        + 0.02 * (term_years - 20)
        + np.array([type_bias[t] for t in property_type])
        + rng.normal(0, 0.6, size=n)
    )
    # Bernoulli labels leave irreducible error vs hard-threshold synthetic labels.
    defaulted = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    frame = pd.DataFrame(
        {
            "ltv": ltv,
            "dti": dti,
            "credit_score": credit_score,
            "note_rate": rate,
            "term_years": term_years.astype(float),
            "property_type": property_type,
            "defaulted": defaulted,
        }
    )
    miss = rng.random(n) < 0.06
    frame.loc[miss, "credit_score"] = np.nan
    meta = {
        "name": "mortgage_default_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(n),
        "target": "defaulted",
        "positive_rate": float(defaulted.mean()),
        "notes": "Synthetic mortgage default; not a real servicing / HMDA extract.",
    }
    return frame, meta


def load_claim_severity_synthetic(
    *,
    n: int = 1100,
    seed: int = 29,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Synthetic insurance claim severity (regression target)."""
    rng = np.random.default_rng(seed)
    vehicle_age = rng.integers(0, 20, size=n).astype(float)
    driver_age = rng.normal(42, 14, size=n).clip(18, 90)
    prior_claims = rng.poisson(0.6, size=n).astype(float)
    urban = rng.binomial(1, 0.55, size=n).astype(float)
    deductible = rng.choice([250, 500, 1000], size=n).astype(float)
    severity = (
        800
        + 45 * vehicle_age
        + 12 * (70 - driver_age).clip(0, None)
        + 350 * prior_claims
        + 220 * urban
        - 0.15 * deductible
        + rng.lognormal(4.5, 0.55, size=n)
    ).clip(100, None)
    frame = pd.DataFrame(
        {
            "vehicle_age": vehicle_age,
            "driver_age": driver_age,
            "prior_claims": prior_claims,
            "urban": urban,
            "deductible": deductible,
            "severity": severity,
        }
    )
    meta = {
        "name": "claim_severity_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(n),
        "target": "severity",
        "notes": "Synthetic P&C severity; not a real claims extract.",
    }
    return frame, meta


def load_payment_rail_anomaly_synthetic(
    *,
    n_normal: int = 1800,
    n_attack: int = 100,
    seed: int = 41,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Payment-rail anomaly table (ACH / card authorization style)."""
    rng = np.random.default_rng(seed)
    normal_hours = rng.integers(0, 24, n_normal)
    attack_hours = rng.integers(0, 24, n_attack)  # overlap hours with normals
    normal = pd.DataFrame(
        {
            "amount_z": rng.normal(0, 1, n_normal),
            "hour_sin": np.sin(2 * np.pi * normal_hours / 24),
            "hour_cos": np.cos(2 * np.pi * normal_hours / 24),
            "merchant_risk": rng.beta(2, 8, n_normal),
            "device_age_days": rng.exponential(180, n_normal).clip(1, 2000),
            "velocity_1h": rng.poisson(2, n_normal).astype(float),
            "is_attack": np.zeros(n_normal, dtype=int),
        }
    )
    # Milder attack shifts + shared hour support so detectors leave residual error.
    attack = pd.DataFrame(
        {
            "amount_z": rng.normal(1.2, 1.0, n_attack),
            "hour_sin": np.sin(2 * np.pi * attack_hours / 24),
            "hour_cos": np.cos(2 * np.pi * attack_hours / 24),
            "merchant_risk": rng.beta(3.5, 4.0, n_attack),
            "device_age_days": rng.exponential(90, n_attack).clip(1, 1500),
            "velocity_1h": rng.poisson(5, n_attack).astype(float),
            "is_attack": np.ones(n_attack, dtype=int),
        }
    )
    frame = pd.concat([normal, attack], ignore_index=True)
    # Label noise: flipped authorizations so perfect F1/AP/ROC is not theater.
    flip_n = max(1, int(0.04 * len(frame)))
    flip_idx = rng.choice(len(frame), size=flip_n, replace=False)
    frame.loc[flip_idx, "is_attack"] = 1 - frame.loc[flip_idx, "is_attack"].to_numpy()
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    meta = {
        "name": "payment_rail_anomaly_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(len(frame)),
        "n_attack": int((frame["is_attack"] == 1).sum()),
        "target": "is_attack",
        "notes": (
            "Synthetic payment authorizations with overlapping attack margins "
            "and ~4% label noise; not a card-network extract."
        ),
    }
    return frame, meta


def load_iot_sensor_anomaly_synthetic(
    *,
    n_normal: int = 1600,
    n_fault: int = 90,
    seed: int = 43,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Factory IoT sensor anomaly table."""
    rng = np.random.default_rng(seed)
    normal = pd.DataFrame(
        {
            "temp_c": rng.normal(55, 3, n_normal),
            "vibration": rng.normal(0.4, 0.08, n_normal),
            "current_a": rng.normal(12, 1.2, n_normal),
            "pressure": rng.normal(101, 2, n_normal),
            "rpm": rng.normal(1800, 40, n_normal),
            "is_fault": np.zeros(n_normal, dtype=int),
        }
    )
    # Milder fault shifts so unsupervised detectors leave residual error.
    fault = pd.DataFrame(
        {
            "temp_c": rng.normal(62, 4, n_fault),
            "vibration": rng.normal(0.65, 0.15, n_fault),
            "current_a": rng.normal(14.5, 1.8, n_fault),
            "pressure": rng.normal(98, 2.5, n_fault),
            "rpm": rng.normal(1740, 70, n_fault),
            "is_fault": np.ones(n_fault, dtype=int),
        }
    )
    frame = pd.concat([normal, fault], ignore_index=True)
    flip_n = max(1, int(0.04 * len(frame)))
    flip_idx = rng.choice(len(frame), size=flip_n, replace=False)
    frame.loc[flip_idx, "is_fault"] = 1 - frame.loc[flip_idx, "is_fault"].to_numpy()
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    meta = {
        "name": "iot_sensor_anomaly_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(len(frame)),
        "n_fault": int((frame["is_fault"] == 1).sum()),
        "target": "is_fault",
        "notes": (
            "Synthetic industrial sensors with overlapping fault margins and "
            "~4% label noise; not a real SCADA extract."
        ),
    }
    return frame, meta


def load_energy_load_synthetic(
    *,
    n_hours: int = 24 * 120,
    seed: int = 17,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Hourly energy load with daily/weekly seasonality."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    t = np.arange(n_hours)
    daily = 80 * np.sin(2 * np.pi * (t % 24) / 24 - 0.8)
    weekly = 25 * np.sin(2 * np.pi * t / (24 * 7))
    temp = 15 + 10 * np.sin(2 * np.pi * t / (24 * 365.25)) + rng.normal(0, 2, n_hours)
    load = (
        400 + daily + weekly + 3.5 * (22 - temp).clip(0, None) + rng.normal(0, 12, n_hours)
    ).clip(50, None)
    frame = pd.DataFrame({"ts": times, "temp_c": temp, "load_mw": load})
    meta = {
        "name": "energy_load_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(n_hours),
        "freq": "h",
        "target": "load_mw",
        "time_column": "ts",
        "notes": "Synthetic grid load; time_split required.",
    }
    return frame, meta


def load_attrition_tabular_synthetic(
    *,
    n: int = 1200,
    seed: int = 23,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """HR attrition table for ensemble / classical proofs."""
    rng = np.random.default_rng(seed)
    tenure = rng.exponential(4.0, size=n).clip(0.1, 30)
    salary = rng.lognormal(10.9, 0.4, size=n)
    overtime = rng.binomial(1, 0.28, size=n).astype(float)
    satisfaction = rng.beta(3, 2, size=n)
    promotions = rng.poisson(0.4, size=n).astype(float)
    dept = rng.choice(["eng", "sales", "ops", "hr"], size=n)
    dept_bias = {"eng": -0.1, "sales": 0.2, "ops": 0.05, "hr": -0.05}
    logit = (
        -0.4
        - 0.08 * tenure
        - 0.15 * np.log1p(salary) / 10
        + 1.1 * overtime
        - 1.8 * satisfaction
        - 0.25 * promotions
        + np.array([dept_bias[d] for d in dept])
        + rng.normal(0, 0.45, size=n)
    )
    # Bernoulli draw (not hard threshold) so seeds keep usable class balance.
    p_leave = 1.0 / (1.0 + np.exp(-logit))
    left = (rng.random(n) < p_leave).astype(int)
    frame = pd.DataFrame(
        {
            "tenure_years": tenure,
            "salary": salary,
            "overtime": overtime,
            "satisfaction": satisfaction,
            "promotions": promotions,
            "department": dept,
            "left": left,
        }
    )
    meta = {
        "name": "attrition_tabular_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(n),
        "target": "left",
        "positive_rate": float(left.mean()),
        "notes": "Synthetic HR attrition; not a real employee extract.",
    }
    return frame, meta


def load_catalog_interactions_synthetic(
    *,
    n_users: int = 90,
    n_items: int = 70,
    seed: int = 13,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """E-commerce catalog interactions for recommender proofs."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        liked = rng.choice(n_items, size=max(10, n_items // 5), replace=False)
        for item in liked:
            rows.append(
                {
                    "user_id": f"u{u}",
                    "item_id": f"sku{item}",
                    "rating": float(rng.integers(3, 6)),
                    "category_code": float(item % 9),
                    "price_band": float((item * 2) % 5),
                }
            )
    frame = pd.DataFrame(rows)
    meta = {
        "name": "catalog_interactions_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(len(frame)),
        "n_users": n_users,
        "n_items": n_items,
        "notes": "Synthetic catalog clicks/ratings; not a real retail extract.",
    }
    return frame, meta


def load_ad_ltr_judgments_synthetic(
    *,
    n_queries: int = 70,
    n_ads: int = 10,
    seed: int = 27,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sponsored ad LTR judgments."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_queries):
        q_center = float(q % 5)
        for ad in range(n_ads):
            relevance_feat = float(rng.normal(q_center, 0.7))
            bid = float(rng.uniform(0.2, 3.0))
            ctr_prior = float(rng.beta(2, 8))
            score = 2.5 - abs(relevance_feat - q_center) + 0.3 * bid + 0.8 * ctr_prior
            rel = float(max(0, min(4, int(round(score)))))
            rows.append(
                {
                    "query_id": f"q{q}",
                    "ad_id": f"ad{ad}",
                    "rel_feat": relevance_feat,
                    "bid": bid,
                    "ctr_prior": ctr_prior,
                    "relevance": rel,
                }
            )
    frame = pd.DataFrame(rows)
    meta = {
        "name": "ad_ltr_judgments_synthetic",
        "license": "synthetic/public-domain (generated in-repo)",
        "n_rows": int(len(frame)),
        "notes": "Synthetic ad judgments; not a real auction log.",
    }
    return frame, meta


# ---------------------------------------------------------------------------
# Real public datasets (sklearn built-ins offline; OpenML optional / cached)
# ---------------------------------------------------------------------------


def load_sklearn_breast_cancer() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Wisconsin breast cancer (sklearn bundled; offline, no network)."""
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame.copy()
    # sklearn uses "target"; rename for proof clarity.
    frame = frame.rename(columns={"target": "malignant"})
    feature_cols = [c for c in frame.columns if c != "malignant"]
    # Sanitize spaces in feature names for stable Session column keys.
    rename = {c: c.replace(" ", "_") for c in feature_cols}
    frame = frame.rename(columns=rename)
    feature_cols = [rename[c] for c in feature_cols]
    meta = _real_meta(
        name="sklearn_breast_cancer",
        source=(
            "sklearn.datasets.load_breast_cancer "
            "(UCI ML Repository: Breast Cancer Wisconsin Diagnostic)"
        ),
        license_provenance=(
            "UCI / sklearn redistributed sample; see sklearn.datasets docs "
            "and UCI Breast Cancer Wisconsin (Diagnostic) citation. "
            "Non-commercial research redistributed with sklearn."
        ),
        n_rows=int(len(frame)),
        n_features=int(len(feature_cols)),
        task="binary_classification",
        target="malignant",
        feature_columns=feature_cols,
        openml=False,
        offline_safe=True,
        citation=(
            "Wolberg, Street, Mangasarian — Breast Cancer Wisconsin (Diagnostic), "
            "UCI Machine Learning Repository."
        ),
    )
    return frame, meta


def load_sklearn_wine() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Wine recognition (sklearn bundled; offline) with cultivar labels."""
    from sklearn.datasets import load_wine

    bunch = load_wine(as_frame=True)
    frame = bunch.frame.copy()
    frame = frame.rename(columns={"target": "cultivar"})
    feature_cols = [c for c in frame.columns if c != "cultivar"]
    meta = _real_meta(
        name="sklearn_wine",
        source="sklearn.datasets.load_wine (UCI Wine recognition data)",
        license_provenance=(
            "UCI / sklearn redistributed sample; see sklearn.datasets docs "
            "and UCI Wine dataset citation."
        ),
        n_rows=int(len(frame)),
        n_features=int(len(feature_cols)),
        task="multiclass_classification_or_clustering",
        target="cultivar",
        external_label="cultivar",
        feature_columns=feature_cols,
        n_classes=int(frame["cultivar"].nunique()),
        openml=False,
        offline_safe=True,
        citation="Aeberhard & Forina — Wine, UCI Machine Learning Repository.",
    )
    return frame, meta


def load_sklearn_diabetes() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Diabetes disease progression regression (sklearn bundled; offline)."""
    from sklearn.datasets import load_diabetes

    bunch = load_diabetes(as_frame=True)
    frame = bunch.frame.copy()
    frame = frame.rename(columns={"target": "progression"})
    feature_cols = [c for c in frame.columns if c != "progression"]
    meta = _real_meta(
        name="sklearn_diabetes",
        source="sklearn.datasets.load_diabetes (Efron et al. diabetes study)",
        license_provenance=(
            "sklearn redistributed sample of the diabetes dataset used in "
            "Efron et al. (2004) Least Angle Regression; see sklearn.datasets docs."
        ),
        n_rows=int(len(frame)),
        n_features=int(len(feature_cols)),
        task="regression",
        target="progression",
        feature_columns=feature_cols,
        openml=False,
        offline_safe=True,
        citation=(
            "Efron, Hastie, Johnstone, Tibshirani (2004), Least Angle Regression, "
            "Annals of Statistics."
        ),
    )
    return frame, meta


def load_sklearn_digits_subset(
    *,
    n_class: int = 5,
    max_rows: int = 400,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Digits image features (sklearn bundled subset; offline, CI-friendly)."""
    from sklearn.datasets import load_digits

    bunch = load_digits(n_class=n_class, as_frame=True)
    frame = bunch.frame.copy()
    frame = frame.rename(columns={"target": "digit"})
    if len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    feature_cols = [c for c in frame.columns if c != "digit"]
    meta = _real_meta(
        name="sklearn_digits_subset",
        source=(
            f"sklearn.datasets.load_digits(n_class={n_class}) "
            f"subsampled to max_rows={max_rows}"
        ),
        license_provenance=(
            "UCI / sklearn redistributed Optical Recognition of Handwritten Digits; "
            "see sklearn.datasets docs."
        ),
        n_rows=int(len(frame)),
        n_features=int(len(feature_cols)),
        task="multiclass_classification",
        target="digit",
        feature_columns=feature_cols,
        n_class=int(n_class),
        openml=False,
        offline_safe=True,
    )
    return frame, meta


def load_openml_adult(
    *,
    data_id: int = 1590,
    as_frame: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """UCI Adult income via OpenML (cached after first fetch; may need network).

    Raises
    ------
    RuntimeError
        When OpenML/sklearn cannot load the dataset (offline, no cache, etc.).
    """
    try:
        from sklearn.datasets import fetch_openml
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"sklearn.fetch_openml unavailable: {exc}") from exc
    try:
        bunch = fetch_openml(
            data_id=data_id,
            as_frame=as_frame,
            parser="auto",
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "OpenML Adult (data_id=1590) unavailable "
            f"(network/cache failure): {type(exc).__name__}: {exc}"
        ) from exc
    frame = bunch.frame.copy() if hasattr(bunch, "frame") else pd.DataFrame(bunch.data)
    # Canonical Adult target / sensitive names across OpenML versions.
    target_col = "class" if "class" in frame.columns else "income"
    if target_col not in frame.columns and getattr(bunch, "target", None) is not None:
        frame[target_col] = bunch.target
    if target_col not in frame.columns:
        raise RuntimeError("Adult frame missing target column 'class'/'income'.")
    if "sex" not in frame.columns:
        raise RuntimeError("Adult frame missing sensitive column 'sex'.")
    # Drop rows with string missing markers common in Adult.
    frame = frame.replace("?", np.nan).dropna(axis=0).reset_index(drop=True)
    y_raw = frame[target_col].astype(str).str.strip()
    # Positive class: >50K income.
    frame["income_gt_50k"] = (y_raw.str.contains(">50", regex=False)).astype(int)
    frame = frame.drop(columns=[target_col])
    sensitive = "sex"
    feature_cols = [
        c for c in frame.columns if c not in {"income_gt_50k", sensitive}
    ]
    meta = _real_meta(
        name="openml_adult_1590",
        source=(
            f"sklearn.datasets.fetch_openml(data_id={data_id}) "
            "(UCI Adult / Census Income)"
        ),
        license_provenance=(
            "UCI Adult (Census Income) via OpenML data_id=1590; "
            "public research redistribution. Cite UCI / Kohavi & Becker."
        ),
        n_rows=int(len(frame)),
        n_features=int(len(feature_cols)),
        task="binary_classification",
        target="income_gt_50k",
        sensitive_column=sensitive,
        feature_columns=feature_cols,
        openml=True,
        openml_data_id=int(data_id),
        offline_safe=False,
        notes=(
            "Requires OpenML cache or network on first fetch. "
            "Proofs should fall back when load fails."
        ),
    )
    return frame, meta


def load_openml_credit_g() -> tuple[pd.DataFrame, dict[str, Any]]:
    """German Credit (credit-g) via OpenML; sensitive stand-in from personal_status."""
    try:
        from sklearn.datasets import fetch_openml
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"sklearn.fetch_openml unavailable: {exc}") from exc
    try:
        bunch = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "OpenML credit-g unavailable "
            f"(network/cache failure): {type(exc).__name__}: {exc}"
        ) from exc
    frame = bunch.frame.copy()
    target_col = "class" if "class" in frame.columns else str(bunch.target.name)
    if target_col not in frame.columns:
        frame[target_col] = bunch.target
    frame = frame.replace("?", np.nan).dropna(axis=0).reset_index(drop=True)
    # good/bad → binary (UCI/OpenML: 'good'/'bad' or 1=good, 2=bad)
    y = frame[target_col].astype(str).str.lower().str.strip()
    frame["credit_good"] = y.isin(["good", "1"]).astype(int)
    # Map personal_status → binary sex proxy disclosed as OpenML attribute encoding.
    if "personal_status" in frame.columns:
        ps = frame["personal_status"].astype(str).str.lower()
        # credit-g codes: male* / female*
        frame["sex_standin"] = np.where(ps.str.contains("female"), "female", "male")
        sensitive = "sex_standin"
    elif "sex" in frame.columns:
        frame["sex_standin"] = frame["sex"].astype(str)
        sensitive = "sex_standin"
    else:
        raise RuntimeError("credit-g missing personal_status/sex for fairness stand-in.")
    drop_cols = {target_col, "personal_status", "sex", "credit_good", sensitive}
    feature_cols = [c for c in frame.columns if c not in drop_cols]
    # Keep sensitive column; drop raw personal_status to avoid leakage into features.
    keep = feature_cols + [sensitive, "credit_good"]
    frame = frame[keep].copy()
    meta = _real_meta(
        name="openml_credit_g",
        source="sklearn.datasets.fetch_openml(name='credit-g', version=1)",
        license_provenance=(
            "UCI Statlog German Credit via OpenML credit-g; public research "
            "redistribution. Sensitive stand-in derived from personal_status."
        ),
        n_rows=int(len(frame)),
        n_features=int(len(feature_cols)),
        task="binary_classification",
        target="credit_good",
        sensitive_column=sensitive,
        feature_columns=feature_cols,
        openml=True,
        offline_safe=False,
        notes=(
            "sex_standin is derived from OpenML personal_status encoding "
            "(male*/female*); observational fairness only."
        ),
    )
    return frame, meta


def load_breast_cancer_fairness_proxy() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Offline fairness fallback: breast cancer + disclosed constructed proxy.

    When Adult / credit-g cannot be fetched, CI still exercises the fairness
    Session surface on a real sklearn table. The sensitive column is a
    **constructed median-split proxy** on ``mean_radius`` (imaging intensity
    stand-in) — **not** a protected demographic class. Disclosed in metadata.
    """
    frame, base = load_sklearn_breast_cancer()
    radius = frame["mean_radius"].to_numpy(dtype=float)
    median = float(np.median(radius))
    frame = frame.copy()
    frame["radius_intensity_proxy"] = np.where(radius >= median, "high", "low")
    feature_cols = [
        c
        for c in frame.columns
        if c not in {"malignant", "radius_intensity_proxy", "mean_radius"}
    ]
    # Exclude the proxy-generating feature from predictors (honest disclosure).
    meta = _real_meta(
        name="sklearn_breast_cancer_fairness_proxy",
        source=str(base["source"]),
        license_provenance=str(base["license"]),
        n_rows=int(len(frame)),
        n_features=int(len(feature_cols)),
        task="binary_classification",
        target="malignant",
        sensitive_column="radius_intensity_proxy",
        feature_columns=feature_cols,
        openml=False,
        offline_safe=True,
        proxy_disclosure=(
            "radius_intensity_proxy is a constructed median-split of mean_radius "
            "for offline CI fairness API exercise only. It is NOT a legal "
            "protected attribute. Prefer OpenML Adult / credit-g when available."
        ),
        fallback_reason="openml_adult_or_credit_g_unavailable",
        notes=str(base.get("citation", "")),
    )
    return frame, meta


def load_fairness_public_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prefer Adult → credit-g → disclosed breast-cancer proxy (always offline-safe)."""
    errors: list[str] = []
    for loader, label in (
        (load_openml_adult, "openml_adult"),
        (load_openml_credit_g, "openml_credit_g"),
    ):
        try:
            frame, meta = loader()
            meta = dict(meta)
            meta["loader_selected"] = label
            meta["loader_errors"] = errors
            return frame, meta
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    frame, meta = load_breast_cancer_fairness_proxy()
    meta = dict(meta)
    meta["loader_selected"] = "breast_cancer_fairness_proxy"
    meta["loader_errors"] = errors
    return frame, meta
