"""Shared helpers for BuildML proof projects."""

from __future__ import annotations

from proofs._lib.datasets import (
    load_ad_ltr_judgments_synthetic,
    load_attrition_tabular_synthetic,
    load_catalog_interactions_synthetic,
    load_claim_severity_synthetic,
    load_credit_approval_synthetic,
    load_customer_segments_synthetic,
    load_energy_load_synthetic,
    load_intrusion_anomaly_synthetic,
    load_iot_sensor_anomaly_synthetic,
    load_mortgage_default_synthetic,
    load_payment_rail_anomaly_synthetic,
    load_store_sales_synthetic,
    load_support_kb_corpus,
    load_support_tickets_synthetic,
    load_telco_churn_synthetic,
)
from proofs._lib.env import (
    TORCH_STATUS,
    extra_available,
    probe_torch,
    skip_reason,
)
from proofs._lib.compare import (
    compute_deltas,
    extract_buildml_test_metrics,
    load_buildml_results,
    write_comparison,
)
from proofs._lib.harness import (
    ProofContext,
    assert_disjoint_partitions,
    assert_no_test_in_selection,
    json_safe,
    metrics_round,
    new_proof_context,
    set_global_seed,
    write_results,
)

__all__ = [
    "TORCH_STATUS",
    "ProofContext",
    "assert_disjoint_partitions",
    "assert_no_test_in_selection",
    "compute_deltas",
    "extra_available",
    "extract_buildml_test_metrics",
    "json_safe",
    "load_buildml_results",
    "load_ad_ltr_judgments_synthetic",
    "load_attrition_tabular_synthetic",
    "load_catalog_interactions_synthetic",
    "load_claim_severity_synthetic",
    "load_credit_approval_synthetic",
    "load_customer_segments_synthetic",
    "load_energy_load_synthetic",
    "load_intrusion_anomaly_synthetic",
    "load_iot_sensor_anomaly_synthetic",
    "load_mortgage_default_synthetic",
    "load_payment_rail_anomaly_synthetic",
    "load_store_sales_synthetic",
    "load_support_kb_corpus",
    "load_support_tickets_synthetic",
    "load_telco_churn_synthetic",
    "metrics_round",
    "new_proof_context",
    "probe_torch",
    "set_global_seed",
    "skip_reason",
    "write_comparison",
    "write_results",
]
