from pathlib import Path

p = Path("scripts/verify_runtime_stability.py")
t = p.read_text(encoding="utf-8")
t = t.replace("session.fit_torch(TinyMLP()", "session.dl.fit(TinyMLP()")
t = t.replace(
    "session.run_automl(backend='native'",
    "session.automl.run(backend='native'",
)
p.write_text(t, encoding="utf-8", newline="\n")
print("done")
