"""Rolling statistics and spectral features."""

from __future__ import annotations

import numpy as np

from buildml.core.errors import ValidationError
from buildml.timeseries.extras import scipy_available
from buildml.timeseries.results import TSFeatureResult


def compute_features(
    y: np.ndarray,
    *,
    rolling_window: int = 7,
    spectral_n_fft: int | None = None,
    target_column: str = "target",
) -> TSFeatureResult:
    """Compute rolling mean/std and optional spectral dominant period."""
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.shape[0])
    window = int(rolling_window)
    if window < 2:
        raise ValidationError("rolling_window must be >= 2")
    if n < window:
        raise ValidationError(
            f"Need n >= rolling_window ({window}); have n={n}"
        )

    warnings: list[str] = []
    disclosures: list[str] = [
        f"Rolling features window={window}, n={n}.",
    ]

    # pandas-free rolling via cumsum
    csum = np.cumsum(np.insert(y, 0, 0.0))
    csum2 = np.cumsum(np.insert(y * y, 0, 0.0))
    roll_mean = np.zeros(n, dtype=float)
    roll_std = np.zeros(n, dtype=float)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        s = csum[i + 1] - csum[start]
        s2 = csum2[i + 1] - csum2[start]
        mu = s / count
        var = max(0.0, s2 / count - mu * mu)
        roll_mean[i] = mu
        roll_std[i] = np.sqrt(var)

    freq: tuple[float, ...] = ()
    power: tuple[float, ...] = ()
    dominant: float | None = None

    if scipy_available():
        from scipy.signal import welch

        n_fft = spectral_n_fft or min(256, max(8, 1 << (n - 1).bit_length()))
        f, pxx = welch(y, nperseg=min(n, n_fft), detrend="constant")
        if len(f) > 1:
            # Skip DC component
            idx = int(np.argmax(pxx[1:])) + 1
            if f[idx] > 0:
                dominant = float(1.0 / f[idx])
            freq = tuple(float(v) for v in f.tolist())
            power = tuple(float(v) for v in pxx.tolist())
            disclosures.append(
                "Spectral density via scipy.signal.welch; dominant_period = 1/f_peak (excl. DC)."
            )
    else:
        warnings.append(
            "scipy not available for spectral features; rolling stats only. "
            "Install buildml[timeseries] (includes scipy)."
        )

    return TSFeatureResult(
        target_column=target_column,
        n_points=n,
        rolling_window=window,
        rolling_mean=tuple(float(v) for v in roll_mean.tolist()),
        rolling_std=tuple(float(v) for v in roll_std.tolist()),
        spectral_frequencies=freq,
        spectral_power=power,
        dominant_period=dominant,
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )
