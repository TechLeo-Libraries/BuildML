# BuildML

BuildML keeps a machine-learning job in one Session: the table, what each
column is for, the train / validation / test split, the preparation that
learned from train only, and the model. If you try to prepare or fit before
a split, it stops you.

If you already use pandas and scikit-learn, this is the workflow around
them. You are not looking for another estimator zoo. You are looking for a
place that remembers what you did and will not quietly leak the holdout.

```bash
pip install buildml
```

Python 3.10 through 3.13. That install is BuildML **2.5.0**, the current
stable Session line. The public entry point is `buildml.Session`.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from buildml import Session

frame = pd.DataFrame(
    {
        "age": [21, None, 35, 40, 29, 33, 52, 47],
        "income": [40, 55, 60, 80, 50, 70, 90, 65],
        "approved": [0, 1, 0, 1, 0, 1, 1, 0],
    }
)

session = Session.ingest(frame)
session.set_roles(
    {"age": "feature", "income": "feature", "approved": "target"}
)
session.split(test_size=0.25, stratify=True, random_state=42)
session.impute(strategy="median")
session.scale(method="standard")
session.fit(LogisticRegression(max_iter=500), task="classification")

print(session.evaluate(partition="test").metrics)
```

Roles say how each column may be used. `split` creates the holdout.
`impute` and `scale` learn from training rows and apply frozen numbers
everywhere else. `evaluate` scores the partition you name.

When rows are not interchangeable (the same customer twice, a time order),
use `group_split`, `time_split`, or pass memberships you designed yourself
with `inject_split`.

Docs: [buildml.readthedocs.io](https://buildml.readthedocs.io/)

---

## What the Session protects

sklearn will fit on whatever frame you hand it. BuildML will not.
Fit-capable preparation and `fit` learn from train. Validation and test
receive the frozen plans. Cross-validation and search draw folds from the
training partition only. The test holdout is not used for ranking.

```python
from sklearn.tree import DecisionTreeClassifier
from buildml.preprocess import PreprocessRecipe

cv = session.cv_score(
    LogisticRegression(max_iter=500),
    cv=5,
    preprocess=PreprocessRecipe(impute="median", scale="standard"),
)
search = session.grid_search(
    DecisionTreeClassifier(random_state=0),
    param_grid={"max_depth": [2, 4, 6], "min_samples_leaf": [1, 5]},
    cv=5,
)
```

Pass a `PreprocessRecipe` when encoding, binning, or scaling should be
refit inside each fold, on data that has not already been prepared on the
whole training partition. If you already ran Session-global `impute` /
`encode` / `scale`, CV and search refuse by default. That is deliberate.
Opt in with `allow_session_global_preprocess=True` only when you mean to
accept leakage-biased scores.

The Session also remembers the run. You can ask what a step means, what is
blocked, or write a walkthrough for someone else.

```python
session.explain("split")           # this Session, right now
session.learn("leakage")           # the idea, in reading order
session.workflow()                 # done / available / blocked
```

`beginner` is the default reading level. It does not assume machine-learning
vocabulary. Teaching copy explains the contract. It does not inspect your
data or certify that a choice fits the domain.

---

## Where to go next

| I want to… | Open |
| --- | --- |
| Run a few real loops (imbalance, groups, time) | [First Session](https://buildml.readthedocs.io/en/latest/usage.html) |
| Understand roles, leakage, and partitions | [Concepts](https://buildml.readthedocs.io/en/latest/concepts.html) |
| Follow the order as a decision path | [Workflow guide](https://buildml.readthedocs.io/en/latest/workflow-guide.html) |
| Work a full classical tutorial | [Classical quickstart](guides/quickstart-classical.md) |
| See every domain guide | [Guides](guides/README.md) |
| Confirm a domain actually runs | [Proof suite](proofs/README.md) |

Classical `session.fit` / `session.evaluate` stay first-class. Domain
work uses namespaced facades (`session.anomaly.*`, `session.forecast.*`,
…). Flat domain aliases still run and warn until BuildML 3.0. The
stability note is in [`docs/stability.md`](docs/stability.md).

---

## Optional extras

`pip install buildml` stays light: numpy, pandas, pyarrow, scikit-learn.
Plotting, Torch, RAG, the local EDA app, and industry backends are extras.
Install what the job needs. The
[installation guide](https://buildml.readthedocs.io/en/latest/installation.html)
lists them by job.

```bash
pip install "buildml[torch]"
pip install "buildml[dashboard]"
```

`buildml[production]` is a best-effort bundle of domain depth. It is not a
promise that every nested industry wheel installs on every machine. On
Python 3.13, especially Windows, some pins are skipped when upstream
wheels are missing. Ask the domain (`session.automl.capability_matrix()`)
or, from a checkout, run `python scripts/probe_industry_extras.py`.

---

## Save, reload, and trust

A checkpoint stores the data workflow (table, roles, split, history,
optional preprocess plans). A pipeline bundle stores fitted plans and the
estimator. They do not embed each other.

```python
session.checkpoint_save("artifacts/checkpoint")
restored = Session.checkpoint_load("artifacts/checkpoint")

session.save_pipeline("artifacts/pipeline", evaluate_partition="test")
```

Pickle / joblib / torch loaders default to `trusted=False` and refuse until
you pass `trusted=True` for an artifact you created or fully trust. A hash
in the manifest can catch tampering after save. It cannot make an
attacker-controlled file safe. Prefer JSON sidecars, parquet, or
`checkpoint_load(..., data_only=True)` when provenance is unclear.

The AI operator (`buildml[ai]`) is propose, then confirm, then execute,
with a closed tool list. Pattern checks on prompts are a best-effort layer,
not a proof against injection. Details live in
[artifacts](guides/artifacts-checkpoints-bundles.md) and
[AI operator safety](guides/ai-operator-safety.md).

---

## Proof suite

[`proofs/`](proofs/README.md) is end-to-end evidence that Session domains
run with honest splits and holdout metrics. It is not a smoke folder.

From a source checkout, after the extras that project needs:

```bash
python -m proofs._lib.run_all --tier all
python proofs/loan-approval-classical/script.py
```

Each major domain has a deep project. Named products compose more than one
Session surface. Where a twin exists, it writes `comparison.json` on the
same split.

---

## BuildML 1.x legacy boundary

BuildML 1.x (`SupervisedLearning` and the old module layout) lives under
`buildml/_legacy/` for reference only. It is not imported from the 2.x
package root. There is no compatibility shim that re-exports 1.x APIs from
`import buildml`.

If you still need that line: `pip install "buildml==1.0.9"`.

---

## Author and license

**Leonard Onyiriuba**: [LinkedIn](https://www.linkedin.com/in/chukwubuikem-leonard-onyiriuba/) · leonard.c.onyiriuba@gmail.com

Issues: [GitHub](https://github.com/TechLeo-Libraries/BuildML/issues)

Apache License 2.0.
