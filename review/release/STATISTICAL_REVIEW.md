# Statistical and EDA acceptance review

Status: independent source review completed; all five findings have verified dispositions. Human acceptance of the intended evaluation design remains required before release approval.

## Independence and scope

This reviewer inspected the repaired EDA orchestration, target, association and drift analyzers, probabilistic fitting/calibration helpers, teaching claims and regression tests. The reviewer did not author or edit the reviewed implementation or its tests. The coordinator authored the conformal repair after the finding was reported. This document is the reviewer's only tracked edit.

The review supports specific behavior in the local Windows/Python 3.12 environment. It does not substitute for the user's final methodological review or the release platform/dependency matrix. EDA outputs are exploratory screens; their suitability depends on the sampling design and intended use.

## Findings and dispositions

| ID | Severity | Finding | Disposition |
| --- | --- | --- | --- |
| STAT-01 | High | `conformal_quantile` silently clamped an unattainable order statistic to the maximum observed score. With five calibration scores and alpha 0.01 it returned a finite cutoff, contradicting the stated finite-sample coverage construction. | Repaired by coordinator: unsupported finite cutoffs raise an actionable validation error. Independently inspected and exercised boundary regression tests. Closed. |
| STAT-02 | Medium | Nonfinite calibration scores could yield a NaN cutoff without validation. Direct reproduction with `[1, NaN]` returned NaN. | Repaired by coordinator: rejects NaN and both infinities. Independently verified all three regression cases. Closed. |
| STAT-03 | Medium | Drift `feature_columns_analyzed` reported all selected columns, including numeric columns skipped for insufficient finite observations and categoricals outside the 30-column cap. | Coordinator repaired the tested-column list and added explicit skipped reasons. Independently inspected and probed a frame with one sparse numeric feature and 31 categorical features: sparse and capped fields were excluded from tested provenance with correct reasons. Closed. |
| STAT-04 | Low | EDA orchestration prose asserted that a KS test on ten million rows provides nothing a sample would not. Sampling changes sensitivity and can omit rare effects. | Coordinator replaced this with the performance/precision tradeoff and accurately distinguished full-partition drift. Independently read the replacement. Closed. |
| STAT-05 | Medium | Train-only calibration membership alone does not establish the conformal assumptions: fitting uses the current Session frame, potentially transformed using the eventual calibration rows; classification carving uses labels to retain classes in both sets. | Scope qualified in `guides/probabilistic-deep.md` and independently reviewed. Closed as documentation qualification, not implementation of a new calibration design. Human reviewer must assess exchangeability and preprocessing independence for the intended evidence workflow. No empirical undercoverage claim is made for these paths by this review. |

The finite-sample order-statistic construction and requirement for a predictor fitted separately from calibration observations are described by [Angelopoulos and Bates, sections 1 and Appendix D](https://arxiv.org/pdf/2107.07511). The boundary defect is a code-level mismatch with that construction, not merely an editorial concern.

## Executed evidence

Command, using the repository virtual environment:

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/test_eda_audit_regressions.py tests/unit/test_probabilistic_slice.py tests/unit/test_probabilistic_m2_depth.py tests/unit/test_conformal_acceptance.py -q --disable-warnings --basetemp=artifacts/statistical-acceptance-temp
```

Result: **23 passed**, eight warnings, in 6.21 seconds. Evidence: `artifacts/statistical-acceptance-tests.log`. An initial run encountered access denial to the shared pytest temporary directory; the successful run used a separate workspace temporary directory. The initial environmental error is not counted as a product failure or as a passing check.

The same 23-test selection passed again after the drift/prose changes, using `artifacts/statistical-acceptance-final-temp`; evidence: `artifacts/statistical-acceptance-final-tests.log`. A separate direct drift probe confirmed both insufficient-observation and categorical-cap provenance repairs.

The coordinator's durable `tests/unit/test_drift_acceptance.py` regression was then independently read and executed: **1 passed**, one warning. Evidence: `artifacts/statistical-drift-review-tests.log`. The reviewer verified the final sampling paragraph and conformal prerequisite qualification directly in source; these prose dispositions are based on inspection, not attributed to the test count.

The tests cover continuous-feature retention, explicit role overrides, fractional target inference, missing-label denominators, partition scope, unavailable drift, nonfinite handling, report export disclosure, native probabilistic interval/evaluation and persistence behavior, and the newly repaired conformal boundary. They do not establish coverage for optional MAPIE/NGBoost backends or prove empirical calibration across deployment populations.

## Accepted behavior and interpretive limits

- Train-only EDA explicitly selects training rows for exploration, while drift remains a disclosed aggregate train/test comparison. It is not a guarantee that repeated inspection of held-out drift cannot influence model development.
- Target type inference is documented as a screening heuristic; integer cardinality can differ from the scientific task. Users must check the inferred type.
- Missing and infinite numeric values are separated in quality reporting; numerical analysis excludes infinities without mutating the Session's source data.
- Target association p-values are unadjusted, and the documentation states this. Their rankings are exploratory, not confirmatory evidence.
- Mutual information label-encodes categorical inputs and treats the resulting dense matrix with the estimator's defaults. The current documentation discloses encoding sensitivity. A categorical-aware estimator/mask would strengthen this screen; current output must not be presented as validated feature importance.
- Drift is univariate and threshold-based. Absence of flags does not establish population equivalence, absence of multivariate shift, or suitability for deployment.

## Human acceptance questions

1. Are the chosen train/calibration/test units independent enough for the claimed evaluation, or do groups, repeated subjects or time ordering require another design?
2. Are feature engineering and model choices fixed before calibration and final evaluation?
3. Are reported EDA findings clearly separated from confirmatory results, with multiplicity and missingness addressed where substantive claims depend on them?
4. Does the evidence describe marginal coverage and its assumptions rather than promising coverage for every subgroup or individual prediction?

Final human acceptance and release approval remain unsigned until the candidate evidence is frozen and the user has reviewed the intended evaluation design. These findings have been resolved within the stated review scope.
