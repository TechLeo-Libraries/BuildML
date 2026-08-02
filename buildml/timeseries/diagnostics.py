"""ACF/PACF and stationarity diagnostics."""

from __future__ import annotations

import numpy as np

from buildml.core.errors import ValidationError
from buildml.timeseries.extras import require_statsmodels, statsmodels_available
from buildml.timeseries.results import TSDiagnosticsResult


def compute_diagnostics(
    y: np.ndarray,
    *,
    acf_lags: int = 40,
    pacf_lags: int = 40,
    adf_regression: str = "c",
    kpss_regression: str = "c",
    target_column: str = "target",
    time_column: str = "time",
) -> TSDiagnosticsResult:
    """Compute ACF/PACF and optional ADF/KPSS stationarity tests."""
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.shape[0])
    if n < 4:
        raise ValidationError("Need at least 4 points for diagnostics")
    max_lag = min(int(acf_lags), n - 2)
    max_pacf = min(int(pacf_lags), n // 2 - 1)
    if max_lag < 1:
        raise ValidationError("Series too short for requested acf_lags")

    warnings: list[str] = []
    disclosures: list[str] = [
        f"Diagnostics on n={n} points; acf_lags={max_lag}, pacf_lags={max_pacf}.",
        "ADF/KPSS are complementary: ADF tests unit root; KPSS tests stationarity around a trend.",
    ]

    if statsmodels_available():
        require_statsmodels(feature="ACF/PACF and stationarity tests")
        from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

        acf_vals, acf_conf = acf(y, nlags=max_lag, alpha=0.05, fft=True)
        acf_vals = np.asarray(acf_vals, dtype=float)
        acf_confint = tuple(
            (float(row[0]), float(row[1])) for row in np.asarray(acf_conf, dtype=float)
        )
        pacf_vals, pacf_conf = pacf(y, nlags=max_pacf, alpha=0.05, method="ywm")
        pacf_vals = np.asarray(pacf_vals, dtype=float)
        pacf_confint = tuple(
            (float(row[0]), float(row[1])) for row in np.asarray(pacf_conf, dtype=float)
        )
        disclosures.append("ACF/PACF via statsmodels (FFT ACF, Yule-Walker PACF).")

        adf_stat = adf_p = adf_lags = None
        adf_crit: dict[str, float] = {}
        try:
            adf_out = adfuller(y, regression=adf_regression, autolag="AIC")
            adf_stat = float(adf_out[0])
            adf_p = float(adf_out[1])
            adf_lags = int(adf_out[2])
            adf_crit = {str(k): float(v) for k, v in adf_out[4].items()}
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"ADF test failed: {exc}")

        kpss_stat = kpss_p = None
        kpss_crit: dict[str, float] = {}
        try:
            kpss_out = kpss(y, regression=kpss_regression, nlags="auto")
            kpss_stat = float(kpss_out[0])
            kpss_p = float(kpss_out[1])
            kpss_crit = {str(k): float(v) for k, v in kpss_out[3].items()}
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"KPSS test failed: {exc}")

        disclosures.append(
            "ADF/KPSS via statsmodels; interpret jointly — conflicting signals often "
            "indicate trend-stationarity or structural breaks."
        )
    else:
        acf_vals = _numpy_acf(y, max_lag)
        acf_confint = ()
        pacf_vals = _numpy_pacf(y, max_pacf)
        pacf_confint = ()
        adf_stat = adf_p = adf_lags = None
        adf_crit = {}
        kpss_stat = kpss_p = None
        kpss_crit = {}
        warnings.append(
            "statsmodels not installed: using numpy ACF/PACF only. "
            "ADF/KPSS unavailable — install buildml[timeseries]."
        )
        disclosures.append("Core fallback ACF via normalized autocovariance.")

    return TSDiagnosticsResult(
        target_column=target_column,
        time_column=time_column,
        n_points=n,
        acf_lags=max_lag,
        pacf_lags=max_pacf,
        acf_values=tuple(float(v) for v in acf_vals.tolist()),
        acf_confint=acf_confint,
        pacf_values=tuple(float(v) for v in pacf_vals.tolist()),
        pacf_confint=pacf_confint,
        adf_statistic=adf_stat,
        adf_pvalue=adf_p,
        adf_used_lags=adf_lags,
        adf_critical_values=adf_crit,
        kpss_statistic=kpss_stat,
        kpss_pvalue=kpss_p,
        kpss_critical_values=kpss_crit,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


def _numpy_acf(y: np.ndarray, nlags: int) -> np.ndarray:
    y = y - np.mean(y)
    n = len(y)
    acf = np.correlate(y, y, mode="full")[n - 1 : n + nlags]
    acf /= acf[0] if acf[0] != 0 else 1.0
    return acf


def _numpy_pacf(y: np.ndarray, nlags: int) -> np.ndarray:
    """Levinson-Durbin PACF approximation without statsmodels."""
    if nlags < 1:
        return np.array([1.0], dtype=float)
    acf = _numpy_acf(y, nlags)
    pacf = np.zeros(nlags + 1, dtype=float)
    pacf[0] = 1.0
    phi = np.zeros((nlags, nlags), dtype=float)
    pacf[1] = acf[1]
    phi[0, 0] = acf[1]
    for k in range(2, nlags + 1):
        num = acf[k] - sum(phi[k - 2, j] * acf[k - j - 1] for j in range(k - 1))
        den = 1.0 - sum(phi[k - 2, j] * acf[j + 1] for j in range(k - 1))
        pacf[k] = num / den if den != 0 else 0.0
        phi[k - 1, k - 1] = pacf[k]
        for j in range(k - 2):
            phi[k - 1, j] = phi[k - 2, j] - pacf[k] * phi[k - 2, k - j - 2]
    return pacf[: nlags + 1]
