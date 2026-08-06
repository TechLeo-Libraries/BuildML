from pathlib import Path

p = Path("scripts/verify_runtime_stability.py")
t = p.read_text(encoding="utf-8")
repls = {
    "session.fit_voting(bases, voting='soft', task='classification')": (
        "session.ensemble.fit_voting(bases, voting='soft', task='classification')"
    ),
    "session.fit_anomaly(method='isolation_forest', mode='unsupervised', contamination=0.1)": (
        "session.anomaly.fit(method='isolation_forest', mode='unsupervised', contamination=0.1)"
    ),
    "session.fit_forecast(method='lag_ridge', lags=[1, 2, 3], horizon=5)": (
        "session.forecast.fit(method='lag_ridge', lags=[1, 2, 3], horizon=5)"
    ),
    "session.fit_cbr(backend='sklearn', task='classification', k=3)": (
        "session.cbr.fit(backend='sklearn', task='classification', k=3)"
    ),
    "session.fit_cbr(backend='industry', task='classification', k=3)": (
        "session.cbr.fit(backend='industry', task='classification', k=3)"
    ),
}
for old, new in repls.items():
    if old not in t:
        raise SystemExit(f"missing: {old}")
    t = t.replace(old, new)
p.write_text(t, encoding="utf-8", newline="\n")
print("ok")
