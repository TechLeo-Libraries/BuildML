# BuildML examples

Small scripts that mirror guide snippets for copy-paste outside Markdown.
These are **not** a CI gate; prefer `tests/` for behavioral guarantees.

**Install (GitHub 2.x):**

```bash
pip install "git+https://github.com/TechLeo-Libraries/BuildML.git"
# or from a checkout: pip install -e ".[dev]"
```

| Script | Guide |
| --- | --- |
| `classical_loan_loop.py` | [classical-end-to-end](../guides/classical-end-to-end.md) |
| `leakage_cv_recipe.py` | [leakage-cv-recipes](../guides/leakage-cv-recipes.md) |
| `rag_hashing_loop.py` | [rag-deep](../guides/rag-deep.md) |
| `probabilistic_bayesian_ridge.py` | [quickstart-probabilistic](../guides/quickstart-probabilistic.md) |
| `causal_aipw_ate.py` | [quickstart-causal](../guides/quickstart-causal.md) |
| `symbolic_rules_loop.py` | [quickstart-symbolic](../guides/quickstart-symbolic.md) |
| `imitation_rl_loop.py` | [quickstart-imitation-rl](../guides/quickstart-imitation-rl.md) |
| `tda_loop.py` | [quickstart-tda](../guides/quickstart-tda.md) (`buildml[tda]`) |
| `recommender_item_knn_loop.py` | [quickstart-recommenders](../guides/quickstart-recommenders.md) |
| `nlp_text_classifier_loop.py` | [quickstart-nlp](../guides/quickstart-nlp.md) |

Run from the repo root (or any cwd) after install:

```bash
python examples/classical_loan_loop.py
python examples/leakage_cv_recipe.py
python examples/rag_hashing_loop.py
python examples/probabilistic_bayesian_ridge.py
python examples/causal_aipw_ate.py
python examples/symbolic_rules_loop.py
python examples/imitation_rl_loop.py
python examples/tda_loop.py
python examples/recommender_item_knn_loop.py
python examples/nlp_text_classifier_loop.py
```
