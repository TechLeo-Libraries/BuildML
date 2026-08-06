// Additions to stage 03 (validation) and stage 04 (evaluation).

import { fmt, plural, list } from './eda-format.js';

export const CONCEPTS_ADD_B = {

  /* ── Stage 3 additions ──────────────────────────────────────────── */

  'pipeline-order': {
    stage: 3, title: 'pipeline-order',
    prose: [
      'There are two kinds of step and the difference decides everything. A stateless step depends only on the row in front of it — parsing a date, taking a log, computing a ratio — and can run anywhere. A fitted step learns parameters from a set of rows — an imputer\u2019s median, a scaler\u2019s mean and variance, an encoder\u2019s level map, a selector\u2019s ranking, PCA\u2019s components, a threshold — and must learn them from training rows only.',
      'So the order is fixed: load, then stateless cleaning and role declaration, then split, then every fitted step inside the training partition, then fit, then evaluate. Anything fitted above the split line has seen rows it should not have, and the resulting score is optimistic by an amount that cannot be measured after the fact.',
      'The practical enforcement is a pipeline object rather than a sequence of notebook cells. Wrapping the fitted steps with the estimator means cross-validation refits all of them per fold automatically, which turns a discipline you have to remember into a property of the code.',
    ],
    read: c => [
      'Walk the code and mark every call that fits something. Each mark must sit below the split.',
      `Check that the frame the split is drawn on is the cleaned one, not the transformed one — here ${c.colCount} columns and ${fmt.n(c.rows)} rows were profiled before any split existed.`,
      'Confirm the cross-validator receives the pipeline, not an already-transformed matrix.',
    ],
    decide: () => 'Put every fitted step inside a pipeline object below the split, and let the cross-validator refit it per fold.',
    session: c => `Every statistic on this sheet was computed on the full frame of ${fmt.n(c.rows)} rows: medians, correlations, VIF, mutual information, anomaly scores. That makes them descriptive observations, not train-fitted evidence — any of them that goes on to drive a transform must be recomputed inside the fold.`,
    example: () => `session.pipeline([\n  ("impute", "median"),\n  ("encode", "one_hot"),\n  ("scale", "robust"),\n  ("model", "gbm")])\nsession.cross_validate(folds=5)`,
    pitfalls: () => [
      'Imputing or scaling the whole frame in cell three and splitting in cell nine.',
      'Selecting features on the full frame and then cross-validating the model only.',
      'Fitting the encoder once "for convenience" and reusing it across folds.',
    ],
  },

  'nested-validation': {
    stage: 3, title: 'nested-validation',
    prose: [
      'Choosing and estimating are different jobs and cannot share a set of rows. Every choice you make against a score — hyper-parameters, feature sets, model families, thresholds — fits that score a little, so the number for the winner is optimistic. With enough choices the optimism is large, and it grows with the number of candidates rather than their quality.',
      'Nested cross-validation separates the two. An inner loop, run within each outer training fold, picks the configuration; the outer loop scores whatever the inner loop chose on rows the inner loop never saw. The outer scores then estimate the performance of your whole selection procedure, which is the thing you will actually deploy.',
      'It costs k_outer × k_inner fits, which is why the cheaper approximation is a three-way split — train, validation, test — where validation absorbs every choice and test is opened once. Either arrangement is acceptable; using one set for both jobs is not.',
    ],
    read: c => [
      `Count the choices you plan to make against a score and ask whether ${fmt.n(Math.round(c.rows * 0.2))} rows — a 20% slice here — can absorb them all.`,
      'Check that the reported number comes from rows that influenced nothing: no tuning, no threshold, no feature selection.',
      'Read the outer-fold spread as the honest uncertainty on the whole procedure.',
    ],
    decide: () => 'Separate choosing from estimating — nested CV or a three-way split — and report the outer score with its spread.',
    session: c => `${fmt.n(c.rows)} rows leave about ${fmt.n(Math.round(c.rows * 0.6))} for training, ${fmt.n(Math.round(c.rows * 0.2))} for choices and ${fmt.n(Math.round(c.rows * 0.2))} for the final estimate under a 60/20/20 arrangement${c.rows < 3000 ? ' — thin enough that nested cross-validation is the better instrument than a fixed three-way split' : ''}.`,
    example: () => `session.nested_cv(\n  outer_folds=5,\n  inner_folds=3,\n  search={"max_depth": [2,3,5]})\n# outer scores estimate the\n# whole procedure`,
    pitfalls: () => [
      'Reporting the best cross-validation score from a hyper-parameter search as the expected performance.',
      'Tuning on the test set once "just to check".',
      'Running many candidates on a small validation set and trusting the maximum.',
    ],
  },

  'sample-size-and-power': {
    stage: 3, title: 'sample-size-and-power',
    prose: [
      'Sample size sets the resolution of every conclusion. For a proportion the standard error is about √(p(1−p)/n), so a metric near 0.5 has an error of roughly 5 points at 100 rows, 1.6 at 1,000 and 0.5 at 10,000 — and a confidence interval about twice that either way. Any difference smaller than the interval is not a difference.',
      'For imbalanced problems the binding constraint is the minority count, not the row count. Fifty positives in a hundred thousand rows gives you a recall estimate built on fifty events, and after a 20% test split, on ten. The rule of thumb of at least ten to twenty events per feature exists to keep that arithmetic in view.',
      'Do the calculation before the analysis, not after. If the smallest effect worth acting on is two points and the data can only resolve five, the honest output is a description and a request for more data — not a model with a confident-looking number attached.',
    ],
    read: c => [
      `Compute the resolution you have: at ${fmt.n(c.rows)} rows a proportion metric carries roughly ±${(1.96 * Math.sqrt(0.25 / Math.max(1, c.rows)) * 100).toFixed(1)} points at 95%, before splitting.`,
      c.target && c.target.classes ? `Read the minority count (${fmt.n(Math.min(...c.target.classes.map(k => k.count)))}) and divide across folds — that is the sample every recall estimate rests on.` : 'Identify the scarcest quantity your metric depends on, and count it.',
      `Compare events per feature: ${c.target && c.target.classes ? Math.round(Math.min(...c.target.classes.map(k => k.count)) / Math.max(1, c.eligible)) : Math.round(c.rows / Math.max(1, c.eligible))} here, against a floor of about ten.`,
    ],
    decide: () => 'State the smallest effect worth acting on, compute whether this sample can resolve it, and say so plainly when it cannot.',
    session: c => `${fmt.n(c.rows)} rows and ${c.eligible} eligible ${plural(c.eligible, 'feature')}${c.target && c.target.classes ? `, with a minority class of ${fmt.n(Math.min(...c.target.classes.map(k => k.count)))} rows — about ${Math.round(Math.min(...c.target.classes.map(k => k.count)) / Math.max(1, c.eligible))} events per feature` : ''}. ${c.rows < 1000 ? 'At this size a metric interval is several points wide, so small differences between candidates cannot be resolved.' : 'That supports a metric interval of a point or two, which bounds what comparisons are meaningful.'}`,
    example: () => `session.metric_interval(\n  metric="average_precision",\n  method="bootstrap",\n  n=1000)\n# read the width first`,
    pitfalls: () => [
      'Comparing two models whose scores differ by less than the interval width.',
      'Counting rows when the metric depends on events.',
      'Running a power calculation after seeing the result, to justify it.',
    ],
  },

  'multiple-comparisons': {
    stage: 3, title: 'multiple-comparisons',
    prose: [
      'Screening many columns manufactures findings. At a 5% threshold, twenty independent tests on pure noise produce one significant result on average; a forty-column frame tested pairwise gives 780 pairs and around 39 spurious hits. The strongest correlation in a wide frame is therefore expected to be substantial even when nothing is related.',
      'The corrections trade the two error types. Bonferroni divides the threshold by the number of tests and is safe and blunt; Benjamini–Hochberg controls the false-discovery rate and keeps more power, which is usually the right choice for exploratory screening. Either way the number of tests must be counted honestly, including the ones you ran and discarded.',
      'The alternative discipline is to treat screening output as a shortlist rather than a result. A relationship that matters should survive a fresh sample, a subgroup check, or a held-out fold — and the sheet\u2019s job is to hand over candidates, not conclusions.',
    ],
    read: c => [
      `Count the tests implied by the frame: ${c.colCount} columns give ${fmt.n((c.colCount * (c.colCount - 1)) / 2)} pairs, and ${(c.corrPairs || []).length} pair(s) were scored here.`,
      'Expect the maximum of many noisy draws to look impressive; compare it against what noise alone would produce at this width.',
      'Re-check every shortlisted relationship on a held-out fold or a different period before reporting it.',
    ],
    decide: () => 'Count the tests, apply a false-discovery correction to any p-value you report, and treat screening results as candidates until they replicate.',
    session: c => `${(c.corrPairs || []).length} correlation ${plural((c.corrPairs || []).length, 'pair')}${(c.mi || []).length ? `, ${c.mi.length} mutual-information ${plural(c.mi.length, 'estimate')}` : ''}${(c.vif || []).length ? ` and ${c.vif.length} VIF ${plural(c.vif.length, 'value')}` : ''} were computed across ${c.colCount} columns. No multiplicity correction was applied, so the extremes of these rankings are partly a consequence of how many things were measured.`,
    example: () => `session.correlations(\n  method="spearman",\n  p_adjust="fdr_bh")\n\n# and re-check survivors on a\n# held-out period`,
    pitfalls: () => [
      'Reporting the strongest of hundreds of correlations as a discovery.',
      'Counting only the tests you kept.',
      'Correcting with Bonferroni on a large exploratory screen and concluding nothing is there.',
    ],
  },

  'reproducibility': {
    stage: 3, title: 'reproducibility',
    prose: [
      'Randomness enters an analysis at more points than people expect: the split, the fold assignment, any resampling, tree feature subsampling, initialisation, and the shuffling of neighbours in MI estimators. Unseeded, every run gives a different number, and a difference between two runs cannot be attributed to the change you made.',
      'Seeding makes runs comparable but also hides variance. The honest arrangement is to seed for reproducibility and then repeat across several seeds to measure how much the answer moves — if the conclusion flips between seed 0 and seed 1, the conclusion is noise regardless of how reproducible each run is.',
      'Reproducibility extends past seeds to the environment: library versions, the extract query, the data snapshot. Pin all three. A result that cannot be regenerated cannot be reviewed, and a result that changes when someone upgrades a package should change loudly, not silently.',
    ],
    read: c => [
      `Record the seed alongside every reported number, including this profile of ${fmt.n(c.rows)} rows${c.sampled ? ' — which is itself a sample and therefore seed-dependent' : ''}.`,
      'Re-run with three different seeds and read the spread of the conclusion, not just the metric.',
      'Pin the environment and the extract: versions, query, window, row count, hash.',
    ],
    decide: () => 'Seed every random step, repeat across seeds to measure sensitivity, and pin the data snapshot and library versions with the result.',
    session: c => `This profile ran on ${c.ds.engine} ${c.ds.version} over ${fmt.n(c.rows)} of ${fmt.n(c.rowsTotal)} rows${c.sampled ? ' under sampling, so the row selection is itself a random draw' : ''}. Seeds for the sampling${c.anomalies ? ' and for the anomaly screen' : ''} are not reported on the sheet, so the numbers are not yet reproducible from the document alone.`,
    example: () => `session.configure(\n  random_state=17,\n  n_jobs=1)\n\nsession.snapshot(\n  hash=True, versions=True)`,
    pitfalls: () => [
      'Comparing two runs that differ in a seed as well as in the change under test.',
      'Seeding once and treating one seed\u2019s result as the truth.',
      'Reporting numbers from a data snapshot nobody can rebuild.',
    ],
  },

  'shift-taxonomy': {
    stage: 3, title: 'shift-taxonomy',
    prose: [
      'Three shifts have three different remedies. Covariate shift: the feature distribution moves while the relationship to the target holds — often fixable by reweighting or retraining on recent rows. Label shift: the base rate moves while the features given the label hold — fixable by adjusting the prior or the threshold. Concept shift: the relationship itself changes, so past data is no longer evidence about the present and the model needs rebuilding.',
      'Only the first is detectable without labels, which is why input monitoring catches covariate shift immediately and misses concept shift entirely. Concept shift shows up as a drop in a realised metric, which arrives one label-horizon later — sometimes months.',
      'Distinguishing them decides the response. Retraining on recent data fixes covariate and label shift and merely tracks concept shift a step behind. Naming which one you are seeing, from the pattern of what moved and what did not, is what turns a drift alert into an action.',
    ],
    read: c => [
      c.drifted && c.drifted.length ? `Read which columns moved (${list(c.drifted, 6)}) and whether the target rate moved with them; features-only means covariate, target-only means label.` : 'Compare feature distributions and the target rate separately; which one moved names the shift.',
      'Check whether the relationship changed by refitting on the recent window and comparing coefficients or importances.',
      'Note the label horizon, since it sets how late concept shift can possibly be detected.',
    ],
    decide: () => 'Name the shift from the pattern — covariate, label, or concept — and match the remedy: reweight, re-threshold, or rebuild.',
    session: c => c.drifted && c.drifted.length
      ? `${c.drifted.length} ${plural(c.drifted.length, 'column')} met the configured drift thresholds (${list(c.drifted, 6)}). Whether the target rate moved with them is not tested here, so the shift cannot yet be classified as covariate, label or concept.`
      : 'No column met the configured drift thresholds, and no target-rate comparison was made across memberships — so this is silence on all three shift types rather than evidence of stability.',
    example: () => `session.explain(\n  "split", moment="after")\n\n# compare feature distributions\n# and the target rate separately`,
    pitfalls: () => [
      'Monitoring inputs only and calling it drift detection.',
      'Retraining on recent data as a reflex, without asking which shift occurred.',
      'Treating a threshold-crossing on one column as a population change before checking the split.',
    ],
  },

  /* ── Stage 4 additions ──────────────────────────────────────────── */

  'confusion-matrix': {
    stage: 4, title: 'confusion-matrix',
    prose: [
      'Every classification metric is a ratio of the same four counts: true positives, false positives, false negatives, true negatives — at one threshold. Recall is TP/(TP+FN), the share of real positives caught. Precision is TP/(TP+FP), the share of positive calls that were right. Specificity is TN/(TN+FP). Accuracy is the diagonal over the total, which is why it flatters an imbalanced problem.',
      'Reading the raw counts prevents most metric mistakes, because it makes the base rate and the operating point visible at once. "Recall 0.6" is abstract; "caught 18 of 30 positives, at the cost of 240 false alarms" is a decision anyone can weigh.',
      'The matrix is defined at one threshold, so it must always be reported with that threshold stated. Two matrices for the same model at different cuts describe two different products, and comparing a metric from one against a metric from the other is not a comparison at all.',
    ],
    read: c => [
      c.target && c.target.classes && c.target.classes.length === 2 ? `Read the four counts, not the ratios: at a ${fmt.pct(c.target.classes[1].count / c.rows, 1)} base rate a small false-positive rate still means a large absolute number of false alarms.` : 'Build the matrix at the threshold you intend to ship, and read the counts before any ratio.',
      'Convert each cell into its operational consequence — a contact, an investigation, a missed case — and weigh those.',
      'State the threshold every time the matrix is shown.',
    ],
    decide: () => 'Report the four counts at the shipped threshold alongside every headline metric.',
    session: c => c.target && c.target.classes
      ? `${c.target.name} is ${c.target.task} with ${c.target.classes.map(k => `${fmt.n(k.count)} ${k.label}`).join(' and ')}. Across ${fmt.n(c.rows)} rows, a test slice of 20% would contain roughly ${fmt.n(Math.round(Math.min(...c.target.classes.map(k => k.count)) * 0.2))} minority rows — the entire basis of a recall figure.`
      : 'No classification target is declared, so there is no confusion matrix to build; the regression analogue is a residual distribution.',
    example: () => `session.confusion_matrix(\n  threshold=0.20,\n  normalize=None)\n# counts, then rates`,
    pitfalls: () => [
      'Quoting recall without the false-positive count that bought it.',
      'Comparing metrics computed at two different thresholds.',
      'Normalising the matrix and losing sight of the absolute volumes someone has to handle.',
    ],
  },

  'ranking-curves': {
    stage: 4, title: 'ranking-curves',
    prose: [
      'Both curves sweep the threshold; they differ in what they plot. ROC plots recall against the false-positive rate, and the FPR denominator is the negative class — so on an imbalanced problem a large absolute number of false alarms is a small rate, and the curve looks generous. Precision-recall plots precision against recall, both conditioned on the positive class, so it reflects the imbalance honestly.',
      'The consequence is a rule: with a rare positive class, report average precision (the area under the PR curve) as the headline and use ROC AUC only as a secondary summary. AUC also has a clean interpretation worth keeping — the probability that a random positive is scored above a random negative — which is exactly what you want when the product is a ranked worklist.',
      'Neither curve says anything about calibration, and neither commits you to a threshold. They describe separability across all cuts; the operating point is a separate decision, and reporting the curve without the chosen point leaves the reader unable to picture the product.',
    ],
    read: c => [
      c.target && c.target.classes && c.target.classes.length === 2 ? `At a ${fmt.pct(c.target.classes[1].count / c.rows, 1)} positive rate, read the PR curve first and treat ROC AUC as secondary.` : 'Choose the curve from the base rate: PR when the positive class is rare, ROC when the classes are comparable.',
      'Read the PR curve\u2019s baseline — it sits at the base rate, not at 0.5 — before judging the area.',
      'Mark the intended operating point on the curve, so the reader sees the product and not just the summary.',
    ],
    decide: () => 'Use average precision as the headline for rare positives, keep ROC AUC as a secondary summary, and always mark the operating point.',
    session: c => c.target && c.target.classes && c.target.classes.length === 2
      ? `With ${fmt.pct(c.target.classes[1].count / c.rows, 1)} positives, a precision-recall curve\u2019s no-skill baseline sits at ${fmt.pct(c.target.classes[1].count / c.rows, 2)} while ROC\u2019s sits at 0.5 — which is why the two curves will give very different impressions of the same model.`
      : 'No binary target is declared here, so neither curve applies yet; multiclass and regression have their own diagnostics.',
    example: () => `session.evaluate(\n  metrics=["average_precision",\n           "roc_auc"],\n  curves=["pr", "roc"],\n  mark_threshold=0.20)`,
    pitfalls: () => [
      'Reporting a high ROC AUC on a rare-positive problem as evidence the model is usable.',
      'Judging a PR curve without noting that its baseline is the base rate.',
      'Presenting a curve with no operating point, so nobody can tell what would be shipped.',
    ],
  },

  'multiclass-and-averaging': {
    stage: 4, title: 'multiclass-and-averaging',
    prose: [
      'With more than two classes, every metric needs an averaging rule and the rule changes the answer. Macro averaging takes the unweighted mean across classes, so a rare class counts as much as a common one. Micro pools all the counts first, which makes the result dominated by the largest class and equal to accuracy in the single-label case. Weighted averages by class support, which sits between the two.',
      'Choose the rule from what matters. If every class matters equally — a diagnostic taxonomy — macro is right, and it will look worse, correctly, when a rare class is handled badly. If overall throughput is what matters, micro is defensible. Reporting one without naming it is the actual error.',
      'The per-class table is more informative than any average, and the multiclass confusion matrix more informative still: it shows which classes are confused with which, which is where the fix lives — usually a label definition problem or two classes that are not really distinct.',
    ],
    read: c => [
      c.target && c.target.classes && c.target.classes.length > 2 ? `Read the per-class table for all ${c.target.classes.length} classes before any average, and name the averaging rule in the caption.` : 'If the target is binary today, ask whether it is a collapsed multiclass problem, since the collapse itself is a modelling choice.',
      'Read the off-diagonal of the multiclass matrix for the pairs being confused; that names the label problem.',
      'Compare macro against micro. A large gap means the rare classes are being handled badly.',
    ],
    decide: () => 'Report the per-class table plus one named averaging rule chosen from whether every class matters equally.',
    session: c => c.target && c.target.classes
      ? `${c.target.name} has ${c.target.classes.length} ${plural(c.target.classes.length, 'class')} (${c.target.classes.map(k => `${k.label} ${fmt.n(k.count)}`).join(', ')})${c.target.classes.length > 2 ? ' — so every metric on this problem needs an averaging rule stated explicitly.' : ', so averaging rules do not yet arise.'}`
      : 'No classification target is declared, so no averaging decision arises.',
    example: () => `session.evaluate(\n  metrics=["f1"],\n  average="macro",\n  per_class=True)`,
    pitfalls: () => [
      'Quoting an F1 without saying which average produced it.',
      'Using micro averaging and reporting it as evidence that rare classes are handled.',
      'Skipping the per-class table, which is where the diagnosis actually is.',
    ],
  },

  'residual-diagnostics': {
    stage: 4, title: 'residual-diagnostics',
    prose: [
      'For a continuous target, a single error figure is a summary of a distribution you should look at. Plot residuals against the prediction: a horizontal band of constant width is what you want. A fan means the error grows with the level (heteroscedasticity), a curve means the model is missing a shape, and a tilt means systematic bias across the range.',
      'Then plot residuals against each feature and against time. Structure against a feature names a missing term — a non-linearity or an interaction. Structure against time means the relationship is drifting, and a model fitted across the whole period is an average of two regimes.',
      'Read the tails as well as the shape. Which rows have the largest residuals, and what do they have in common? A handful of extreme errors concentrated in one segment is a different problem from error spread evenly, and only one of them is fixed by a better model.',
    ],
    read: c => [
      c.target && c.target.task === 'regression' ? `Plot residuals against prediction and check for a fan — with target skew ${c.target.stats ? c.target.stats.skew.toFixed(2) : 'unknown'} a widening band is likely.` : 'For a classification target the analogue is a calibration curve plus error rates by score band.',
      'Plot residuals against each feature and against time, and treat any visible structure as a missing term.',
      'Rank rows by absolute residual and read the top twenty for a common factor.',
    ],
    decide: () => 'Read the residual distribution and its structure before accepting any error figure, and report error per band when the spread is not constant.',
    session: c => c.target && c.target.task === 'regression' && c.target.stats
      ? `${c.target.name} runs ${fmt.compact(c.target.stats.min)} to ${fmt.compact(c.target.stats.max)} with skew ${c.target.stats.skew.toFixed(2)}, so residual spread is likely to vary across the range and a single RMSE will describe neither end well. No model is fitted here, so no residuals exist yet.`
      : 'No continuous target is declared, so residual diagnostics do not apply; calibration and the confusion matrix are the classification analogues.',
    example: () => `session.residuals(\n  against=["prediction",\n           "time"],\n  bands=5)\n# error per band, not one number`,
    pitfalls: () => [
      'Reporting RMSE on a heteroscedastic problem as if it applied uniformly.',
      'Never plotting residuals against time and missing a regime change.',
      'Treating a few huge residuals as outliers rather than as a segment the model cannot handle.',
    ],
  },

  'uncertainty-intervals': {
    stage: 4, title: 'uncertainty-intervals',
    prose: [
      'A metric computed on a test set is one draw. The bootstrap turns it into a distribution: resample the evaluation rows with replacement, recompute the metric, repeat a few hundred to a thousand times, and read the 2.5th and 97.5th percentiles as a 95% interval. It requires no distributional assumption and works for any metric you can compute.',
      'Two rules keep it honest. Resample at the level of independence — by group where rows repeat an entity, by period where they are ordered — because resampling correlated rows individually understates the interval. And for a comparison between two models, bootstrap the difference on the same resamples; an interval on the difference that includes zero means you cannot tell them apart.',
      'Cross-validation gives a cheaper spread: the fold-to-fold variation. It is not a confidence interval — the folds share training data and are not independent — but reporting the mean with the fold standard deviation is far better than reporting the mean alone.',
    ],
    read: c => [
      `Bootstrap at the level of independence: ${c.groupCol ? `by ${c.groupCol.name}, since rows repeat entities` : c.timeCol ? `by period, since rows are time-ordered` : 'by row, since no group or time structure is declared'}.`,
      'Read the interval width before the point estimate, and refuse comparisons narrower than it.',
      `Remember which sample each number rests on: ${fmt.n(c.completeRows)} complete cases versus ${fmt.n(c.rows)} rows analysed.`,
    ],
    decide: () => 'Bootstrap every headline metric at the level of independence, report the interval, and bootstrap the difference when comparing models.',
    session: c => `Every figure on this sheet is a point estimate from ${fmt.n(c.rows)} rows${c.completeRows !== c.rows ? ` (${fmt.n(c.completeRows)} for anything needing complete cases)` : ''} with no interval attached. ${c.groupCol ? `Because ${c.groupCol.name} repeats, any resampling must be done at group level.` : c.timeCol ? 'Because the rows carry a time order, resampling should respect periods.' : 'No group or temporal dependence is declared, so row-level resampling is defensible.'}`,
    example: () => `session.metric_interval(\n  metric="average_precision",\n  method="bootstrap",\n  n=1000,\n  group_by=None)`,
    pitfalls: () => [
      'Comparing two point estimates whose intervals overlap almost entirely.',
      'Bootstrapping rows individually when they repeat an entity.',
      'Reading the cross-validation fold spread as a confidence interval.',
    ],
  },

  'slice-evaluation': {
    stage: 4, title: 'slice-evaluation',
    prose: [
      'An aggregate metric is an average over subpopulations that may behave very differently. A model at 0.85 overall can be at 0.92 on the segment that supplies most of the rows and 0.55 on the one that supplies the decisions you care about — and the aggregate will never show it. Slicing is how you find out.',
      'Define the slices before you look: by segment, region, channel, source system, time period, and any protected attribute you retained for measurement. Report the metric per slice with its support and its interval, and treat a small slice\u2019s dramatic number with the suspicion it deserves.',
      'Where slices differ materially, you have three options and should name which you chose: accept and document the difference, add features or capacity to close it, or restrict the model\u2019s scope to the population it serves adequately. Silence is the one option that is not available.',
    ],
    read: c => [
      `Define slices from the categorical columns available — ${(c.categorical || []).length} here — plus period and any retained protected attribute.`,
      'Report support alongside the metric per slice, and read the interval before reacting to a gap.',
      'Check whether the worst slice is the one the decision matters most for; that is the finding, not the average.',
    ],
    decide: () => 'Publish the metric per predefined slice with support and intervals, and name your response to every material gap.',
    session: c => (c.categorical || []).length
      ? `${c.categorical.length} categorical ${plural(c.categorical.length, 'column')} could define evaluation slices${c.timeCol ? ', alongside periods of ' + c.timeCol.name : ''}. Every figure on this sheet is pooled across all ${fmt.n(c.rows)} rows, so no slice-level variation is visible here.`
      : `No categorical column is available to slice on${c.timeCol ? `, though ${c.timeCol.name} supports slicing by period` : ''}. All figures here are pooled across ${fmt.n(c.rows)} rows.`,
    example: c => `session.evaluate(\n  metrics=["average_precision"],\n  by=["${((c.categorical || [])[0] || {}).name || '<segment>'}"],\n  min_support=100,\n  interval="bootstrap")`,
    pitfalls: () => [
      'Reporting only the aggregate and discovering the weak slice in production.',
      'Reacting to a slice of forty rows whose interval spans half the metric range.',
      'Slicing after the fact until a flattering cut appears.',
    ],
  },
};
