# BuildML guides

User-facing tutorials and reference copy for BuildML 2.x. These Markdown files
are the canonical source for quickstarts and the glossary. They are readable
directly on GitHub and included in the published Sphinx site at
[buildml.readthedocs.io](https://buildml.readthedocs.io/).

## Suggested learning path

1. [Sphinx installation and usage](../docs/index.rst) — install, first loop, learning path index
2. [Concepts](../docs/concepts.rst) and [workflow guide](../docs/workflow-guide.rst) — vocabulary and stage decisions
3. **[Classical quickstart](quickstart-classical.md)** — chapter-style tutorial (classification, imbalance, regression, splits, teaching APIs)
4. Optional extras on the same Session: [Torch](quickstart-torch.md), [RAG](quickstart-rag.md), [AI operator](quickstart-ai.md)
5. [Glossary](glossary.md) — terms used across Session, reports, and extras

For maintainer engineering notes, see [maintainers/](../maintainers/README.md).

## Quickstarts

| Guide | Extra | Summary |
| --- | --- | --- |
| [Classical](quickstart-classical.md) | core | Split, preprocess, fit, evaluate, CV, teaching surfaces, bundles |
| [Torch](quickstart-torch.md) | `buildml[torch]` | Tabular + text loaders, built-in models, fold-local CV |
| [RAG](quickstart-rag.md) | `buildml[rag]` | Ingest, retrieve, grounded generate, evaluate, bundle |
| [AI operator](quickstart-ai.md) | `buildml[ai]` | Advisor, plan, confirmed execute across classical/RAG/Torch |

## Reference

- [Glossary](glossary.md) — terms used across Session, reports, and extras
