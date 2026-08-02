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

Run from the repo root (or any cwd) after install:

```bash
python examples/classical_loan_loop.py
python examples/leakage_cv_recipe.py
python examples/rag_hashing_loop.py
```
