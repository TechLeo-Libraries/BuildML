# BuildML editorial standards

These rules apply to operation explanations, findings, recommendations, console text, and HTML
reports. They are a product contract: a report should help a reader decide what to inspect or do
next without overstating what the data establishes.

## Canonical product terminology

- `Session` is the public 2.x workflow facade; do not call it an automator, assistant, or agent.
- `Dataset` is the canonical tabular state. `DataFrame` names a concrete Pandas value.
- Use `partition` for stored `train`, `validation`, and `test` membership. Use `split` for the
  operation or complete membership plan.
- Use `train-fitted` only for values learned exclusively from the training partition.
- Use `operation history`, not provenance or audit trail: history records Session calls but cannot
  prove source lineage or methodological validity.
- A `checkpoint` stores data workflow state. A `model bundle` stores a fitted estimator and feature
  contract. Never imply that either artifact contains the other.
- `available` means API prerequisites pass. It does not mean recommended or suitable.
- A `self-contained report` has no network dependency. It is not necessarily a complete account of
  external work.

The canonical definitions are in [glossary.md](../glossary.md). Catalog concept keys live in
`buildml.explain.CONCEPT_NOTES`; documentation should name the matching key when it expands one.

## Voice and structure

- Name the object and action: "Median imputation replaced missing `age` values" is preferable to
  "The data was handled."
- Lead with the observed result, then interpretation, then action. Keep those three parts distinct.
- Use direct, restrained language. Do not use "unlock," "revolutionary," "seamless," "robust," or
  "intelligent" without a measured definition.
- Address the reader only for a concrete action: "Review the six unmatched categories."
- Use present tense for stable definitions, past tense for completed operations, and conditional
  language for uncertain consequences.
- Define an abbreviation on first use in each standalone report. Metric labels may retain standard
  abbreviations after the definition.
- Remove boilerplate openings, praise, and consulting labels. State the object, computation,
  population, and limitation instead of calling material "research-grade," "professional," "rich,"
  "deep," or "comprehensive."

## Definitions, evidence, and findings

An **observation** is a computed or recorded fact. A **finding** interprets one or more observations.
A **recommendation** proposes a response. Do not collapse them into one sentence.

Every finding should state:

1. what was observed;
2. the population or partition (`train`, `validation`, or `test`);
3. the measure and value, including units or denominator;
4. why it may matter;
5. a limitation when the evidence is conditional or noisy.

Use severity for likely workflow impact, not visual emphasis:

- `info`: context with no implied problem;
- `low`: worth noting, unlikely to change the current result alone;
- `medium`: investigate before relying on the affected analysis;
- `high`: address before model selection or performance claims;
- `critical`: the current result is invalid or unsafe to use.

Never convert a heuristic cutoff into a fact. Write "VIF above 5 is a collinearity review flag" rather
than "VIF above 5 proves multicollinearity." Include sample count for statistical tests and curves.
Name the missing capability when an analysis is skipped.

## Recommendations and automatic choices

Label the origin of each choice:

- `automatic`: BuildML selected it from deterministic rules;
- `recommended`: BuildML proposes it, but has not changed state;
- `explicit`: the caller supplied it.

An automatic choice must expose the inputs and rule that selected it. A recommendation must include
the evidence keys it relies on, a rationale, a priority, and caveats. Recommendations do not mutate a
Session. An executable action may show the matching Session operation and parameters, but reports
must not imply that it has run.

Avoid universal prescriptions. For example:

- Good: "Median imputation is a reasonable baseline for the skewed numeric columns listed here.
  Compare it with native missing-value handling."
- Bad: "Always impute missing numeric values with the median."

## Leakage and partition language

Call a learned operation **train-fitted** only when statistics, categories, synthetic samples, model
parameters, or thresholds were learned without validation/test rows.

- `validation` supports model, feature, and threshold choices.
- `test` estimates performance after those choices are fixed.
- Do not call test performance "unseen" if test results influenced an earlier decision.
- State when ordinary random splitting assumes independent, exchangeable rows. Grouped and temporal
  data usually need externally supplied partitions.
- When displaying full-dataset EDA after a split, distinguish descriptive full-data observations
  from evidence used to select a model.

## Metrics and uncertainty

- Show the metric name, direction, partition, and sample count together.
- For classification, state the positive class and class prevalence when interpreting precision,
  recall, PR curves, calibration, or thresholds.
- For regression, report the target unit for MAE/RMSE and explain when a transformed target changes
  that unit.
- Do not describe a model as "accurate" from accuracy alone. Compare with a relevant baseline and
  inspect error costs.
- Avoid causal language for correlations, permutation importance, model coefficients, and SHAP-like
  attributions.
- Prefer ranges, fold spread, or repeat spread where available. Rounded score differences smaller
  than the observed variation are ties for editorial purposes.

## Before/after explanations

A state-change explanation must identify:

- fields or rows changed;
- state deliberately preserved;
- the decision origin and parameters;
- any fitted values and the partition used to learn them;
- invalidated downstream artifacts, especially feature schemas and fitted models.

"No state change" is valid and should be explicit for inspection and export methods. Do not imply
that mutating a returned DataFrame changes the Session when the method returns a copy.

## Tables, figures, and HTML

- Each table needs a visible or programmatic caption and explicit column headings.
- Escape all data-derived text. Raw HTML is accepted only from trusted BuildML component renderers.
- Figure alternative text should name the chart and its analytical purpose, not repeat "image of."
- Do not encode meaning by color alone. Use labels, symbols, or text with sufficient contrast.
- Navigation must use landmarks and in-page headings. Include a skip link and visible keyboard focus.
- Reports must work from a local file with networking disabled. Embed CSS, JavaScript, and required
  small assets; do not load fonts, libraries, trackers, or images from a CDN.
- JavaScript may enhance navigation or themes, but the report content must remain readable without it.

## Operation catalog entries

Each public `Session` method has one catalog entry. It must cover definition, purpose, pipeline role,
mechanism, parameters, inputs, outputs, prerequisites, usual ordering, alternatives, selection
rationale, assumptions, failure modes, leakage risks, anti-patterns, state changes, result reading,
next considerations, and concept links.

Catalog text should answer operation-specific questions. "This operation processes data efficiently"
is not acceptable. When a method has no call-time parameter or no state change, say so through its
inputs/state-change text rather than inventing a parameter.

The catalog completeness test is the change gate: adding a public callable to `Session` requires a
substantive entry in the same change.

## Copy-lint scope

`scripts/lint_user_copy.py` checks current public documentation and user-facing Python. Archival
plans, historical release notes, and `buildml/_legacy/` are excluded because they describe removed
interfaces. This standards file is also excluded from phrase checks because it quotes prohibited
examples. Exclusions are not permission to link archival examples as current guidance.
