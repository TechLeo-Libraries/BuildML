// Stage 00 (framing & provenance) and stage 05 (interpretation & handoff).
// These bracket the numeric stages: the questions that must be answered before
// any statistic means anything, and the ones that decide what it is worth.

import { fmt, plural, list } from './eda-format.js';

export const CONCEPTS_ENDS = {

  /* ── Stage 0 · framing & provenance ─────────────────────────────── */

  'problem-framing': {
    stage: 0, title: 'problem-framing',
    prose: [
      'Before a single column is profiled, one sentence has to exist: who takes what action, when, on the basis of what output. That sentence decides the unit of analysis, the target, the prediction moment, the metric and the baseline — five choices that no amount of later analysis can repair, because each of them changes what the data would have to be.',
      'Prediction and inference are different jobs with different requirements. A prediction problem needs an output that generalises and tolerates uninterpretable features; an inference problem needs coefficients you can defend, which forbids collinearity, demands attention to confounders, and rules out most of the reduction toolkit. Deciding which one you are doing is the first fork in the road.',
      'Framing also sets what "good" means before you can be tempted. The action defines the two errors, the errors define their costs, the costs define the metric and the threshold. Written down first, they are a specification; written down last, they are a rationalisation of whatever the model happened to produce.',
    ],
    read: c => [
      `State the decision in one sentence and check that this frame could support it: ${fmt.n(c.rows)} rows at ${c.colCount} columns, ${c.target ? `target ${c.target.name} (${c.target.task})` : 'no declared target'}.`,
      'Name the moment of prediction. Every later leakage question is decided by that single point in time.',
      'Name the incumbent — the rule, model or human judgement in use today. If there is none, say so; if there is one, it is your bar.',
    ],
    decide: () => 'Write the decision sentence, the prediction moment, the metric and the incumbent baseline down before profiling anything.',
    session: c => `${c.ds.label}: ${fmt.n(c.rows)} rows, ${c.colCount} columns, ${c.target ? `${c.target.name} declared as a ${c.target.task} target` : 'no target declared'}. The decision this supports is not recorded in the data and must come from you — the sheet can describe the frame, not the intent behind it.`,
    example: c => `session.frame(\n  decision="who is contacted",\n  predicted_at="account open",\n  target="${(c.target && c.target.name) || '<target>'}",\n  metric="average_precision",\n  incumbent="rules v3")`,
    pitfalls: () => [
      'Profiling first and framing later, which lets the available columns choose the problem.',
      'Treating a prediction problem as an inference one, then reading coefficients from a model built for accuracy.',
      'Leaving the incumbent unnamed, so no score can be judged as good or bad.',
    ],
  },

  'unit-of-analysis': {
    stage: 0, title: 'unit-of-analysis',
    prose: [
      'The unit of analysis is what one row means: one customer, one customer-month, one transaction, one sensor reading. It is the frame\u2019s grain, and it determines what can be predicted at all — a per-transaction target cannot be modelled on a per-customer table without either aggregating the target or expanding the features.',
      'Grain is verifiable, not assumed. Name the key you believe uniquely identifies a row and count distinct keys against total rows. Equality confirms the grain; fewer distinct keys means the true grain is finer than you thought, and every share, mean and rate you have computed is weighted by whatever repeated.',
      'Changing grain is a modelling act with consequences. Aggregating transactions to customers loses within-customer variation and requires choosing summary statistics; expanding customers to customer-months multiplies rows and introduces dependence between them, which then constrains the split.',
    ],
    read: c => [
      `Count rows against distinct candidate keys: ${fmt.n(c.rows)} rows here, with ${(c.idLike || []).length} near-unique column(s)${(c.idLike || []).length ? ` (${list(c.idLike, 3)})` : ''}.`,
      'Say the grain out loud as "one row is one ___". If the sentence needs an "and", the key is composite and should be declared as such.',
      'Check that the target is defined at the same grain as the rows. A mismatch here invalidates everything downstream.',
    ],
    decide: () => 'Declare the grain as a named key, enforce uniqueness on it, and confirm the target is defined at that same grain.',
    session: c => `${(c.idLike || []).length ? `${list(c.idLike, 3)} ${c.idLike.length === 1 ? 'is' : 'are'} near-unique across ${fmt.n(c.rows)} rows and could serve as the key` : `No near-unique column was observed, so no candidate key is visible in the frame`}${c.groupCol ? `, while ${c.groupCol.name} identifies ${fmt.n(c.groupCol.groups)} repeating ${plural(c.groupCol.groups, 'group')} — the grain is finer than the entity` : ''}. The intended grain was not declared in this session.`,
    example: c => `session.set_grain(\n  key=["${(c.idLike && c.idLike[0]) || '<key>'}"],\n  enforce_unique=True)\n\n# rows == distinct keys, or the\n# grain is not what you think`,
    pitfalls: () => [
      'Assuming one row per entity because the id looks unique in this extract.',
      'Computing a rate over a table whose grain is finer than the denominator you have in mind.',
      'Aggregating to a coarser grain without recording which summary statistics were chosen.',
    ],
  },

  'population-and-sampling-frame': {
    stage: 0, title: 'population-and-sampling-frame',
    prose: [
      'The rows in the file are not the population; they are whatever the extract\u2019s filters let through. Somewhere upstream there is a WHERE clause, a date range, a status flag, a join that dropped non-matches, and each of them narrows the population the model will be honest about. A model trained on approved applications cannot speak about rejected ones.',
      'Selection bias is the general name for the resulting gap, and survivorship is its most common shape: entities that ended before the extract window are simply absent, so the surviving rows are systematically different from the ones you will meet in production. The distribution looks clean precisely because the awkward cases were removed before you saw them.',
      'The repair is documentary rather than statistical. Write down the inclusion rules, the window, and what was excluded; then state the population the model may be applied to. Where the deployment population is wider than the training one, that difference is a known limitation, not a surprise to be discovered by drift monitoring later.',
    ],
    read: c => [
      `Compare rows examined with rows available: ${fmt.n(c.rows)} of ${fmt.n(c.rowsTotal)}${c.sampled ? ' — this extract is sampled, so tail behaviour is under-represented' : ' — no sampling was applied at this step'}.`,
      'Ask for the extract query. Every filter in it is a boundary on what the model may claim.',
      'Ask what happened to the rows that are absent — closed accounts, failed joins, rejected applications — and whether the model will meet them in production.',
    ],
    decide: () => 'Record the inclusion rules and the window, then state explicitly which population the model is licensed to score.',
    session: c => `${fmt.n(c.rows)} of ${fmt.n(c.rowsTotal)} rows were examined${c.sampled ? ' under sampling' : ' with no sampling at this step'}, covering ${c.ds.session || 'this session'}. Upstream inclusion rules are not visible to the sheet: whatever was filtered before the extract is invisible here and cannot be inferred from the rows that survived.`,
    example: () => `session.provenance(\n  window=("2025-01", "2025-12"),\n  included="status in (active,\n            closed)",\n  excluded="test accounts")`,
    pitfalls: () => [
      'Reading the extract as the population and quoting its base rate as the world\u2019s.',
      'Ignoring survivorship: the entities that failed early are usually the ones you were asked to predict.',
      'Scoring a wider population in production than the one the training rows came from.',
    ],
  },

  'target-definition': {
    stage: 0, title: 'target-definition',
    prose: [
      'A target is constructed, not found. "Churned" means no activity for some number of days, measured from some anchor, within some horizon — three arbitrary choices, each of which changes the base rate and the difficulty of the problem. Two teams with the same table and different definitions will report incompatible results and both be right.',
      'Most labels are proxies for the thing you care about. A cancellation flag proxies dissatisfaction; a click proxies interest; a diagnosis code proxies illness. The gap between the proxy and the concept is where the model\u2019s later failures come from, and it belongs in the write-up rather than in a footnote.',
      'Censoring is the specific trap. A row whose horizon has not fully elapsed has not yet had the chance to be positive, so labelling it negative manufactures a negative class from ignorance. Either exclude those rows, or model time-to-event, but do not quietly call them zeros.',
    ],
    read: c => [
      c.target ? `Read the label\u2019s balance as a consequence of its definition: ${c.target.classes ? c.target.classes.map(k => `${k.label} ${fmt.n(k.count)}`).join(', ') : `continuous, median ${c.target.stats ? fmt.compact(c.target.stats.median) : 'n/a'}`}.` : 'No target is declared, so its definition is the next thing to write down.',
      'State the anchor, the horizon and the rule in one sentence, then ask whether every row has had the full horizon to declare itself.',
      'Name the concept the label proxies and the gap between them.',
    ],
    decide: () => 'Write the label rule with its anchor and horizon, exclude or model censored rows, and record what the proxy stands in for.',
    session: c => c.target
      ? `${c.target.name} is used as the ${c.target.task} target${c.target.classes ? ` with ${c.target.classes.map(k => `${fmt.n(k.count)} ${k.label}`).join(' and ')}` : c.target.stats ? ` spanning ${fmt.compact(c.target.stats.min)} to ${fmt.compact(c.target.stats.max)}` : ''}. How it was constructed — anchor, horizon, rule — is not recorded in the frame, so the observed balance cannot be interpreted without it.`
      : 'No target is declared in this session, so nothing here is a supervised problem yet.',
    example: c => `session.define_target(\n  name="${(c.target && c.target.name) || '<target>'}",\n  rule="no_activity_days > 60",\n  anchor="period_end",\n  horizon_days=90,\n  drop_censored=True)`,
    pitfalls: () => [
      'Labelling censored rows negative, which invents a majority class out of incomplete follow-up.',
      'Changing the horizon mid-project and comparing scores across the change.',
      'Forgetting that the proxy and the concept differ, then explaining model failures as data quality.',
    ],
  },

  'provenance-and-lineage': {
    stage: 0, title: 'provenance-and-lineage',
    prose: [
      'Every column arrived from somewhere: a source system, a join, a derivation, a manual upload. Lineage is that chain written down, and without it you cannot answer the two questions that decide whether a column may be used — when is this value known, and will it still be produced next month in the same units.',
      'Derived columns are the ones to trace hardest. A column that is a function of other columns duplicates information (a conditioning problem) and can encode the target (a leakage problem); a column computed by an upstream job inherits that job\u2019s schedule, which may run after the moment of prediction even though the value looks static in the file.',
      'Lineage also covers versions. The frame is a snapshot; the source keeps moving. Pin the extract — a query, a date, a commit, a hash — so that the analysis can be reproduced, and so that a later difference in results can be attributed to a change in the data rather than argued about.',
    ],
    read: c => [
      `For each of the ${c.colCount} columns, name the source system and whether it is raw or derived. Anything you cannot attribute is a candidate for exclusion.`,
      'Trace every derived column to its inputs and check whether any input is the target or is recorded after it.',
      'Pin the snapshot: query, window, extraction time, row count. Those four make the analysis reproducible.',
    ],
    decide: () => 'Attribute every column to a source and a timing, and pin the extract so the same frame can be rebuilt.',
    session: c => `${c.colCount} columns were profiled from ${c.ds.label} on ${c.ds.engine} ${c.ds.version}. Source system, derivation and refresh schedule are not carried in the frame; the sheet reports what the columns contain, not where they came from or when they are computed.`,
    example: () => `session.lineage({\n  "tenure_months": "derived: \n     period_end - open_date",\n  "risk_band": "upstream job,\n     nightly"})`,
    pitfalls: () => [
      'Using a derived column without checking whether its inputs include the outcome.',
      'Assuming an upstream job runs before the prediction moment because the column is populated in the extract.',
      'Rebuilding the extract later with a different window and comparing results as if they were the same data.',
    ],
  },

  'sensitive-attributes': {
    stage: 0, title: 'sensitive-attributes',
    prose: [
      'Some columns carry legal and ethical weight regardless of predictive value: identity attributes, health, and anything that identifies a person directly. The first pass is inventory — which columns are personal, which are protected, which are neither — and it belongs at the start, because it constrains what may be stored, joined, modelled and shown.',
      'Removing the column is not removing the attribute. Postcode proxies for ethnicity and income; device type proxies for wealth; occupation and name proxy for several things at once. A model without the protected column can still be a function of it, which is why the check is on the model\u2019s behaviour across groups, not only on the feature list.',
      'That behavioural check needs the attribute retained for evaluation even when it is excluded from the features — you cannot measure performance per group without the group. Keeping it for measurement while barring it from the matrix is a deliberate, documentable arrangement, not a contradiction.',
    ],
    read: c => [
      `Inventory the ${c.colCount} columns into personal, protected and neither, and record who approved each use.`,
      'For each excluded protected attribute, list the columns that could proxy for it and plan a per-group performance check.',
      'Check free-text and identifier columns for personal data hiding inside them.',
    ],
    decide: () => 'Inventory sensitive columns up front, bar them from the matrix where required, and retain them for per-group evaluation.',
    session: c => `No sensitivity classification is attached to any of the ${c.colCount} columns in this session. The sheet cannot infer which columns are personal or protected; that inventory is a human judgement recorded before modelling, and its absence is itself a finding.`,
    example: () => `session.classify_columns(\n  personal=["email", "postcode"],\n  protected=["age_band"],\n  evaluate_by=["age_band"],\n  exclude_from_features=True)`,
    pitfalls: () => [
      'Dropping a protected attribute and declaring the model fair without checking proxies.',
      'Dropping it so completely that per-group performance can no longer be measured.',
      'Leaving personal data in free-text columns that are then embedded into features.',
    ],
  },

  /* ── Stage 5 · interpretation & handoff ─────────────────────────── */

  'feature-importance-methods': {
    stage: 5, title: 'feature-importance-methods',
    prose: [
      'Importance is not one quantity. Tree gain measures how much a feature reduced impurity during training and is biased toward high-cardinality columns; permutation importance measures how much a metric degrades when a column is shuffled and is therefore about this model on this data; SHAP attributes each individual prediction and averages up. Three methods, three questions, three orderings.',
      'Collinearity breaks all of them in the same way. When two features carry the same information, gain splits arbitrarily between them and permutation shows neither as important — shuffling one leaves the other to carry the signal. A near-zero importance can therefore mean irrelevant or merely redundant, and only the correlation structure tells you which.',
      'Importance is also not effect. It says a column mattered to the model, not which direction the target moves, by how much, or whether intervening on it would change anything. For direction you need effect plots; for intervention you need a design the data usually cannot supply.',
    ],
    read: c => [
      `Compute permutation importance on held-out rows rather than training rows, and read it beside the correlation structure — ${c.hasCorr ? `${c.corrPairs.filter(p => Math.abs(p.r) >= 0.8).length} pair(s) here exceed |0.8|` : 'no correlation screen was supplied, so redundancy is unmapped'}.`,
      'Repeat the permutation several times and read the spread; a single pass is one noisy draw.',
      'Compare two methods. Where they disagree, the disagreement is the finding — usually redundancy or cardinality bias.',
    ],
    decide: () => 'Report permutation importance with its spread on held-out data, note the redundant groups, and never present importance as effect size.',
    session: c => `${c.eligible} eligible ${plural(c.eligible, 'feature')} would enter an importance calculation${(c.corrPairs || []).filter(p => Math.abs(p.r) >= 0.8).length ? `, of which ${(c.corrPairs || []).filter(p => Math.abs(p.r) >= 0.8).length} correlated ${plural((c.corrPairs || []).filter(p => Math.abs(p.r) >= 0.8).length, 'pair')} will share credit unpredictably` : ''}. No model has been fitted in this session, so no importance exists yet — the univariate screens on the cockpit are not importances.`,
    example: () => `session.permutation_importance(\n  on="validation",\n  n_repeats=10,\n  metric="average_precision")\n# read mean and std together`,
    pitfalls: () => [
      'Reading tree gain as importance without accounting for its cardinality bias.',
      'Concluding a redundant feature is useless because permuting it changed nothing.',
      'Presenting an importance ranking as a set of effect sizes to a business audience.',
    ],
  },

  'effect-shapes': {
    stage: 5, title: 'effect-shapes',
    prose: [
      'Once you know which features matter, the next question is how. A partial-dependence plot averages the model\u2019s prediction as one feature is swept across its range, holding the rest as observed; an ICE plot draws the same sweep per row, and the spread between those curves is where interactions become visible — parallel curves mean no interaction, crossing curves mean plenty.',
      'Both methods evaluate the model at combinations that may not exist in the data. Sweeping income to £500k for a row whose other attributes make that implausible asks the model to extrapolate, and the resulting curve says more about the estimator than the world. Read the feature\u2019s density alongside the curve and distrust the ends.',
      'These are descriptions of a fitted model, not of the population. Two models with equal accuracy can produce different shapes on the same data, and neither shape is a causal statement — the honest reading is "this model responds to this feature like so", which is exactly what you need when explaining a decision.',
    ],
    read: c => [
      `Plot the sweep against the feature\u2019s own distribution: for ${((c.numeric || []).filter(n => n.role !== 'id')[0] || {}).name || 'each numeric feature'} the curve is only trustworthy between the quartiles where rows actually live.`,
      'Read ICE curves for the spread, not the average. A flat mean with wide spread is an interaction, not an absence of effect.',
      'Check monotonicity against domain expectation; a non-monotone shape where theory demands monotone is a data or specification problem.',
    ],
    decide: () => 'Publish effect shapes only over the supported range, with the feature\u2019s density beneath, and label them as model behaviour rather than causal effect.',
    session: c => `${(c.numeric || []).length} numeric ${plural((c.numeric || []).length, 'feature')} could be swept once a model exists${(c.skewed || []).length ? `, though ${(c.skewed || []).length} of them are skewed, so the supported range is much narrower than min-to-max` : ''}. Nothing is fitted in this session, so no effect shape is available.`,
    example: c => `session.partial_dependence(\n  features=["${((c.analysable || [])[0] || {}).name || '<feature>'}"],\n  kind="both",\n  clip_to_percentiles=(1, 99))`,
    pitfalls: () => [
      'Reading a partial-dependence curve out where the data has no rows.',
      'Averaging ICE curves that cross, then reporting the flat mean as no effect.',
      'Describing a partial-dependence shape to stakeholders in causal language.',
    ],
  },

  'learning-curves-and-capacity': {
    stage: 5, title: 'learning-curves-and-capacity',
    prose: [
      'A learning curve plots training and validation score against training-set size. Its shape diagnoses which problem you have: a large persistent gap between the two, with training near perfect, is variance — the model is memorising, and more data or more constraint will help. Both curves low and converged is bias — the model is too simple, or the features do not carry the signal, and more rows will change nothing.',
      'That distinction decides where to spend. Variance is answered by regularisation, simpler models, more data or better splits; bias by richer features, interactions, or a more flexible estimator. Diagnosing wrongly costs weeks: collecting more data for a bias problem is the classic expensive mistake.',
      'A validation curve is the same instrument aimed at a hyper-parameter rather than a sample size, and it shows the same two regimes on either side of the optimum. Read both curves before tuning: they tell you whether tuning is the lever at all.',
    ],
    read: c => [
      `Read the gap and the level together. At ${fmt.n(c.rows)} rows over ${c.eligible} features (${Math.round(c.rows / Math.max(1, c.eligible))} rows per feature) variance is the likelier regime${c.rows < 2000 ? ', and the curve should still be rising at the right edge' : ''}.`,
      'Check whether the validation curve is still climbing at the largest sample. If it is, more data is worth buying.',
      'Plot with error bars from repeated folds; a gap smaller than the fold spread is not a gap.',
    ],
    decide: () => 'Diagnose bias versus variance from the curve before choosing between more data, more features and more regularisation.',
    session: c => `${fmt.n(c.rows)} rows and ${c.eligible} eligible ${plural(c.eligible, 'feature')} — about ${Math.round(c.rows / Math.max(1, c.eligible))} rows per feature — put this frame ${c.rows / Math.max(1, c.eligible) < 20 ? 'in the regime where variance dominates and the curve is worth drawing before any tuning' : 'in a comfortable regime where the curve mostly diagnoses feature quality'}. No model is fitted here, so no curve exists yet.`,
    example: () => `session.learning_curve(\n  sizes=[0.1, 0.25, 0.5, 0.75, 1.0],\n  cv=5,\n  metric="average_precision")`,
    pitfalls: () => [
      'Collecting more data for a bias problem, where the curves have already converged.',
      'Reading a single fold\u2019s curve as a trend when the fold spread is wider than the gap.',
      'Tuning hyper-parameters before knowing which regime the model is in.',
    ],
  },

  'causal-caution': {
    stage: 5, title: 'causal-caution',
    prose: [
      'Everything on the readiness sheet is associational. Correlation, mutual information, importance and effect shapes all describe how variables move together in the observed data; none of them answers what would happen if you changed a value, because nothing here was changed — it was recorded.',
      'The gap has a specific mechanism: confounding. When a third variable drives both feature and target, the association between them is real, reproducible and useless for intervention. Adjusting for the confounder can remove the bias, but only if you know it exists and measured it — and adjusting for the wrong variable (one on the causal path, or a collider) introduces bias rather than removing it.',
      'What EDA can honestly deliver is a description, a set of candidate mechanisms, and a list of the confounders that would have to be measured. What it cannot deliver is an intervention estimate; that needs an experiment, a natural experiment, or an explicit causal design defended on its own terms.',
    ],
    read: c => [
      c.corrPairs && c.corrPairs.length ? `For each strong association — the top pair here is ${c.corrPairs[0].a} × ${c.corrPairs[0].b} at r=${c.corrPairs[0].r.toFixed(2)} — name a plausible common cause before naming a mechanism.` : 'For every association you plan to describe, name a plausible common cause first.',
      'Check whether the relationship reverses within subgroups; if it does, the aggregate was a confounded artefact.',
      'Ask whether anyone will act on the feature itself. If so, the question is causal and this evidence cannot answer it.',
    ],
    decide: () => 'Report associations as associations, list the confounders that would need measuring, and refuse intervention claims the design cannot support.',
    session: c => `${(c.corrPairs || []).length} correlation ${plural((c.corrPairs || []).length, 'pair')}${(c.mi || []).length ? ` and ${c.mi.length} mutual-information ${plural(c.mi.length, 'estimate')}` : ''} were computed on observational rows. No randomisation, assignment mechanism or intervention is recorded anywhere in this frame, so every relationship on the sheet is associational by construction.`,
    example: () => `session.explain(\n  "associations", moment="after")\n\n# document confounders that\n# would need measuring before\n# any causal claim`,
    pitfalls: () => [
      'Turning a feature importance into a recommendation to change that feature.',
      'Adjusting for a variable on the causal path and reporting the attenuated association as the true one.',
      'Reading a subgroup reversal as noise rather than as evidence of confounding.',
    ],
  },

  'handoff-and-monitoring': {
    stage: 5, title: 'handoff-and-monitoring',
    prose: [
      'The analysis is only finished when someone else can act on it and check it. That means four artefacts: the assumptions ledger (what was taken on trust), the decision log (what was chosen and why), the reproducible extract (query, window, hash) and the residual risks (what could still be wrong and how you would find out). Each of them is short; their absence is what makes results unauditable.',
      'Monitoring is the same list pointed at the future. Watch the input distributions against the training baseline, the prediction distribution, the realised metric once labels arrive, and the operating threshold\u2019s behaviour. Input drift arrives immediately, label-based decay arrives one horizon later, and only both together tell you whether to retrain.',
      'Two failure modes belong in the handoff rather than the model. Feedback loops: a model that decides who is contacted changes the data the next model trains on. And silent pipeline change: a renamed column or a unit change upstream degrades predictions with no error anywhere. Both are process risks, so both need an owner named in the document.',
    ],
    read: c => [
      `Hand over the numbers with their scope attached: ${fmt.n(c.rows)} of ${fmt.n(c.rowsTotal)} rows, ${fmt.pct(c.completeness)} cell completeness, ${c.colCount} columns profiled on ${c.ds.engine} ${c.ds.version}.`,
      'List every threshold you configured — contamination, VIF cut, drift bands, decision cut — since each is a choice a reader must be able to revisit.',
      'Name the owner and the review date for each monitored quantity; unowned monitoring is not monitoring.',
    ],
    decide: () => 'Ship the ledger, the decision log, the pinned extract and the monitoring plan with named owners — the analysis is not done until those exist.',
    session: c => `This session profiled ${fmt.n(c.rows)} of ${fmt.n(c.rowsTotal)} rows across ${c.colCount} columns at ${fmt.pct(c.completeness)} cell completeness, with configured thresholds for ${c.vifThreshold ? `VIF (${c.vifThreshold.toFixed(1)})` : 'collinearity'}${c.anomalies ? ` and anomaly contamination (${fmt.pct(c.anomalies.contamination, 0)})` : ''}. Those thresholds and the assumptions ledger are the handoff; nothing downstream can be audited without them.`,
    example: () => `session.export_ledger(\n  path="readiness.md",\n  include=["assumptions",\n           "thresholds",\n           "extract_hash"])\n\nsession.monitor(\n  inputs=True, metric="ap",\n  owner="risk-analytics")`,
    pitfalls: () => [
      'Handing over a score without the thresholds and assumptions that produced it.',
      'Monitoring inputs only, so a real performance decay waits for a complaint.',
      'Ignoring the feedback loop: the model\u2019s own decisions shape the next training set.',
    ],
  },
};
