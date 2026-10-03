# Session domain methods

BuildML groups specialized methods by domain, such as `session.anomaly`
and `session.fairness`. These namespaces make related operations easier to
find. The following complete example fits and scores an anomaly model:

```python
from sklearn.datasets import make_blobs
import pandas as pd
from buildml import Session

values, _ = make_blobs(n_samples=100, centers=1, random_state=42)
session = Session.ingest(pd.DataFrame(values, columns=["x", "y"]))
session.split(test_size=0.2, random_state=42)
session.anomaly.fit(method="isolation_forest")
result = session.anomaly.score(partition="test")
print(result)
```

## Compatibility with existing code

Domain namespaces were introduced in BuildML 2.4.0. Older flat domain
methods, such as `session.fit_anomaly`, remain available in the 2.x series
and emit a `DeprecationWarning` identifying the replacement. They are
scheduled for removal in BuildML 3.0.

Classical methods such as `session.ingest`, `session.split`, `session.fit`,
and `session.evaluate` remain supported without deprecation warnings.
Their namespace equivalents are also supported.

| Operation group | Namespace |
| --- | --- |
| Data ingestion, roles, and partitions | `session.data` |
| Preprocessing | `session.preprocess` |
| Classical models | `session.classical` |
| Exploratory data analysis | `session.explore` |
| Workflow and teaching | `session.audit` |
| Specialized domains | For example, `session.anomaly`, `session.rag`, or `session.fairness` |

The EDA namespace is called `explore` because `session.eda()` is already
an operation. Similarly, `audit` groups workflow methods without replacing
`session.workflow()`.

## Discover available methods

This example lists namespaces, their capabilities, and the methods in the
fairness namespace. Discovery does not fit a model or require a dataset.

```python
from buildml import Session

session = Session()
print(Session.list_facades())
print(Session.list_capabilities())
print(Session.describe_method("fairness.evaluate"))
print(session.fairness.describe())
print(session.list_active_domains())
```

Discovery results identify each method's stability tier:

| Tier | Meaning |
| --- | --- |
| `core` | Data, preprocessing, classical modeling, EDA, and workflow methods; both flat and namespace forms are supported |
| `domain` | Specialized methods; use the namespace form |
| `experimental` | APIs that may change as their backends and behavior develop; review release notes before upgrading |

## Explain a method

The teaching APIs accept both namespace paths and legacy flat names.
This example retrieves explanations without executing either operation:

```python
from buildml import Session

session = Session()
fairness_explanation = session.explain("fairness.evaluate")
forecast_explanation = session.explain("session.forecast.fit")
```

Catalog entries and operation history retain their canonical flat names,
so an explanation requested for `fairness.evaluate` identifies its
operation as `evaluate_fairness`.
