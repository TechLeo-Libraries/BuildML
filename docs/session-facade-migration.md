# Session namespaced facades

**Status:** shipped in BuildML `2.4.0`  
**Flat domain alias removal:** BuildML `3.0`

## Why facades exist

One `Session` entry point, with a surface you can navigate by domain:

```python
session.fairness.evaluate(sensitive_column="group")
session.anomaly.fit(method="isolation_forest")
session.rag.retrieve(query="...")
```

Flat domain methods still work during the deprecation window and emit
`DeprecationWarning` pointing at the preferred facade. Classical core stays
dual and first-class **without** warnings.

## Version timeline

| Version | What lands |
| --- | --- |
| `2.4.0a2` | Discovery helpers (`list_capabilities`, `describe_method`, …) |
| **`2.4.0`** | Namespaced facades for all domains; flat domain actions warn; docs / guides / examples / proofs teach facades first |
| `2.4.0` (stable intent) | Facades remain preferred; flat aliases still present |
| **`2.5.0`** | Same facade policy continues on the stable Session line |
| **`3.0`** | Flat domain aliases eligible for removal (classical core policy re-evaluated then) |

## Product rules for 2.4+

1. **Classical core does not warn.**  
   Flat `ingest` / `set_roles` / `split` / preprocess / `fit` / `evaluate` / …
   remain first-class. Facades exist as dual paths:
   `session.data.*`, `session.preprocess.*`, `session.classical.*`,
   `session.explore.*` (EDA), `session.audit.*` (workflow / teaching).
2. **Domain industry methods warn on flat actions.**  
   Prefer `session.<domain>.*`. Result / plan properties do not warn.
3. **No functionality removed in `2.4.x` / `2.5.x`.** Deprecation means warnings
   + docs preference only.
4. **Name collisions avoided.**  
   - EDA facade attr is `session.explore` (flat method remains `session.eda`)  
   - Workflow / teaching facade attr is `session.audit` (flat method remains
     `session.workflow`)

## Product decisions

| Decision | Meaning |
| --- | --- |
| **Surface stays large until 3.0** | Facades organize the Session surface; they do not shrink it. Flat domain aliases remain until BuildML **3.0**. Use `Session.list_facades()` / `list_capabilities()` / `describe_method()` to navigate. |
| **Classical dual stays first-class** | Flat classical chains stay without `DeprecationWarning`. |
| **Catalog keys stay flat** | `OPERATION_CATALOG` and teaching sync keep **canonical flat** Session method names as keys. Explain / AI / discovery accept both flat and facade forms (`fairness.evaluate`, `session.fairness.evaluate`) and resolve to the flat key. |
| **Variable naming** | Bind `session = Session()` (or another non-domain name). Binding `rag = Session()` produces awkward `rag.rag.*`; runtime emits a one-shot `UserWarning` when a Session is bound to a domain-named local and a facade is accessed. |

## Stability tiers

Discovery payloads (`list_capabilities`, `describe_method`, `list_facades`) expose:

| Tier | Meaning | Examples |
| --- | --- | --- |
| `core` | Primary product path; dual flat+facade, no flat warnings | data, preprocess, classical, explore, audit |
| `domain` | Specialized Session surface; prefer facade | fairness, anomaly, rag, forecast, … |
| `experimental` | More likely to move before stable 2.x | ai, rl, tda, metalearning |

## Discovery APIs

```python
Session.list_facades()
Session.list_capabilities()           # includes facades + preferred_facade / stability_tier
Session.describe_method("evaluate_fairness")
Session.describe_method("fairness.evaluate")
session.fairness.describe()           # bindings for one namespace
session.list_active_domains()
```

Explain and AI tooling accept flat **and** facade preferred paths:

```python
session.explain("fairness.evaluate")          # → operation == "evaluate_fairness"
session.explain("session.forecast.fit")       # → operation == "fit_forecast"
from buildml.session.facade_registry import resolve_operation_name
resolve_operation_name("rag.retrieve")        # → "rag_retrieve"
```

Tool allowlists may set `session_method` / `catalog_operation` to either form;
teaching sync canonicalizes before parity checks. Emitted catalog names and
history IDs remain flat.

## Registry / regeneration

- Bindings source: `scripts/_facade_bindings.json`
- Generated module: `buildml/session/facade_registry.py`
- Runtime: `buildml/session/facades.py` (`DomainFacade`, deprecation install)
- Regenerate: `python scripts/generate_facade_registry.py`
- Content rewrite helper: `python scripts/migrate_session_facades.py --check|--write`

The migrator rewrites:

- Dot calls / attrs (`session.<flat>(` → `session.<domain>.<method>(`)
- Narrative backticks with optional args (`` `<flat>(...)` `` → `` `session.<domain>.<method>(...)` ``)
- RST double-backticks
- Unquoted teaching tokens in `buildml/explain/{concepts,beginner,overlays,…}`
  (exact catalog key strings like `"fit_anomaly"` stay flat; ambiguous English
  verbs like rank / recommend are never rewritten as bare tokens)

## Included in 2.4.0

- All Session domains have namespaced facades
- Domain flat actions deprecated with tests
- Docs, guides, explain teachings, examples, and proof scripts prefer facades
  (classical chains may stay flat)
- Discovery shows preferred paths + tiers
- Explain / AI / discovery accept facade and flat operation names
- Migrator `--check` clean on examples / guides / docs / explain / proofs
- Domain-variable shadowing warned at runtime

## Documented exceptions

- **Classical teaching** may keep flat `session.fit` / `session.evaluate` chains
  (dual first-class, no deprecation warnings).
- **`OPERATION_CATALOG` keys remain flat** by design. Dual-form acceptance lives
  at resolver boundaries (`get_operation`, `session.explain`, AI tool binding
  checks, `describe_method`).
- **`Session.*_capability_matrix()`** static calls remain valid; prefer
  `session.<domain>.capability_matrix()` when a live session exists.
- **Package catalog discovery** (`from buildml.<domain> import
  <domain>_capability_matrix`) remains a valid non-Session dual.
- **Unit / integration tests** may call flat aliases to assert `DeprecationWarning`.
- **Historical CHANGELOG** entries describing the past are not rewritten.
- **Generated Sphinx `_build/`** may lag sources until docs are rebuilt; fix
  sources under `docs/*.rst` / `docs/*.md`, not `_build`.

## Deprecation window after 2.4.x

Facades are the supported domain API for 2.4.x / 2.5.x. Flat domain aliases stay
supported-but-deprecated until **3.0**; that removal is out of the 2.x scope.

PyPI 2.x publish readiness is separate: see
[`docs/pypi-2x-publish.md`](pypi-2x-publish.md).

## Maintainer checklist (before tagging a 2.x build)

- [x] Facades are the supported public API for domains
- [x] Flat domain actions emit `DeprecationWarning` (removal deferred to 3.0)
- [x] Classical core remains dual first-class (no flat warnings)
- [x] Teaching / docs / examples / proofs prefer facades (migrator `--check` clean)
- [x] Discovery + explain / AI accept flat and facade operation forms
- [x] Flat-alias removal deferred to **BuildML 3.0**
