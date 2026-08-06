// Teaching depth for the stage 3–4 core concepts. Same shape as eda-depth-a.js.

import { fmt, plural, list } from './eda-format.js';

export const DEPTH_B = {

  'data-splitting': {
    more: [
      'A split is three memberships, not two. Train fits parameters; validation chooses between things you could have done — features, hyper-parameters, thresholds, whole models; test estimates how the finished object behaves on data it has never influenced. Collapsing validation into test is the most common shortcut, and it converts your only unbiased estimate into another number you have optimised against.',
      'Size follows from what you need the estimate to do. A test set of n rows gives a proportion metric a standard error of roughly √(p(1−p)/n) — at 500 rows and a metric near 0.5 that is about 2 percentage points, so any difference smaller than about 4 points between two models is inside the noise. Small frames therefore want cross-validation for choosing and a held-out set only for the final honest number.',
    ],
    read: c => [
      `Read the row count against the split you intend: a 20% test set here is about ${fmt.n(Math.round(c.rows * 0.2))} rows.`,
      'Ask what makes rows non-exchangeable before choosing random: time order, entity groups, geography, source system. Any one of them changes the strategy.',
      'Confirm every fitted step sits after the split in the code, not before it.',
    ],
    decide: c => `Draw the split with the structure the data has — ${c.timeCol ? 'time-ordered' : c.groupCol ? 'grouped' : 'stratified random'} here — and touch the test set exactly once.`,
  },

  'stratification': {
    more: [
      'Stratified sampling partitions the rows into strata and samples within each, so every stratum\u2019s share is preserved in every membership. For a binary target this removes the variance that comes from the class split alone; without it, the positive count in a small test set is a binomial draw and can land far from the base rate by luck.',
      'The constraint is stratum size. Every stratum needs at least as many rows as folds — with 5-fold cross-validation, a class or a level with four rows cannot be represented in all folds, and the splitter will either error or silently disappoint. Stratifying on several columns multiplies the strata and hits that wall quickly.',
    ],
    read: c => [
      c.target && c.target.classes ? `Read the smallest class count — ${fmt.n(Math.min(...c.target.classes.map(k => k.count)))} rows — and divide by the fold count to see how thin each fold gets.` : 'With no classification target, decide whether to stratify on quartiles of the continuous target instead.',
      'List the columns you want to stratify on and multiply their level counts. That product is your stratum count; compare it with the row count.',
      'After splitting, compare the target rate across memberships. Any drift there means the stratification did not apply.',
    ],
    decide: () => 'Stratify on the target by default, add at most one more column, and verify the resulting shares rather than assuming them.',
  },

  'cross-validation': {
    more: [
      'The loop is mechanical: split into k parts, hold each out in turn, fit on the other k − 1, score on the held-out part, and end with k scores. The mean estimates performance and the spread estimates how much that estimate can move — which is why reporting the mean alone throws away half of what the procedure produced. k = 5 or 10 is conventional; leave-one-out has the lowest bias and the highest variance and is rarely worth it.',
      'The variant must match the data\u2019s structure: stratified k-fold for imbalanced classification, group k-fold when rows share an entity, time-series splits when order matters. And every fitted step — imputer, encoder, scaler, selector, PCA, threshold — must be inside the pipeline object so it refits on each fold. A single preprocessing pass before the loop invalidates every fold at once.',
    ],
    read: c => [
      `Read the fold spread alongside the mean; at ${fmt.n(c.rows)} rows each fold holds about ${fmt.n(Math.round(c.rows / 5))} rows, so the spread will be wide.`,
      'Check the splitter class: plain KFold on grouped or ordered data reports a score that cannot be reproduced.',
      'Confirm the pipeline, not the fitted matrix, is what gets passed to the cross-validator.',
    ],
    decide: () => 'Cross-validate the whole pipeline with the splitter the data\u2019s structure demands, and report mean with spread.',
  },

  'dataset-drift': {
    more: [
      'Drift is measured by comparing two distributions of the same column. The usual instruments are a Kolmogorov–Smirnov statistic for continuous columns, a chi-square or population-stability index for categorical ones, and PSI has the familiar 0.1 / 0.25 rule-of-thumb bands. Every one of them depends on a threshold you configured, so a flag is a statement about your threshold as much as about the data.',
      'Sample size drives false alarms in both directions. On a small test set, real shifts go undetected; on a very large one, trivial differences reach significance. And a flag on many columns at once almost always indicates the split rather than the world — check the split first, then the ingestion, then the population.',
    ],
    read: c => [
      c.drifted && c.drifted.length ? `Read which columns flagged (${list(c.drifted, 6)}) and look for a common cause before treating them separately.` : 'Read the absence of flags as the absence of evidence at your chosen threshold, not as stability.',
      'Compare the two distributions directly rather than trusting the statistic — an overlaid histogram usually explains the flag in one glance.',
      'Check whether the flagged columns share a source, a period, or a join, which turns many findings into one.',
    ],
    decide: () => 'Diagnose the split, the pipeline and the ingestion before touching the data, and never re-split until the flags disappear.',
  },

  'leakage': {
    more: [
      'The formal test is a timing question: at the moment you would make the prediction, is this value already known and already stable? Any column failing that test leaks, however innocent its name. The classic families are post-outcome fields (a resolution code, a settlement date, a churn-survey response), identifiers that index the answer, aggregates computed over the full period, and duplicated rows straddling the split.',
      'Pipeline leakage is the subtler half. Any statistic fitted before the split — a median, a scaler\u2019s mean, an encoder\u2019s level map, a selector\u2019s ranking, PCA\u2019s components — carries test-row information into training. It is invisible in the data and visible only in the code, which is why the review is a code review.',
    ],
    read: c => [
      c.leakage && c.leakage.length ? `Start with the flagged suspects (${c.leakage.map(x => x.name).slice(0, 4).join(', ')}) and establish the recording time of each.` : 'Walk the schema column by column and write down when each value becomes known; the heuristic screen found nothing, which is not the same as clean.',
      'Treat any single feature with implausible predictive power as a leak until proven otherwise.',
      'Read the pipeline in order and mark every fit call. Anything fitted above the split line is a leak.',
    ],
    decide: () => 'Build the feature set from what is known at prediction time only, and move every fitted step inside the fold.',
  },

  'temporal-structure': {
    more: [
      'Time changes the split from a sampling question into an ordering one. The honest arrangement trains on a past window and tests on a future one, optionally with a gap the length of your prediction horizon so that no training row could have observed the test period. Rolling-origin evaluation repeats that forward, giving several honest estimates instead of one.',
      'Two failure modes are specific to time. Features built from windows that extend past the prediction moment leak the future — a "customer lifetime total" computed over all data is the canonical example. And the final period in an extract is usually incomplete, so its lower counts read as a downward trend that is really the reporting lag.',
    ],
    read: c => [
      c.timeCol ? `Read the span (${c.timeCol.min} to ${c.timeCol.max}) and the coverage gaps — ${fmt.n(c.timeCol.gaps || 0)} here — before deciding the cut point.` : 'Confirm there is genuinely no time column: a created_at, a batch id or even row order can encode the ordering implicitly.',
      'Plot record counts per period and check the last one for truncation.',
      'For every engineered feature, name the window it uses and confirm the window closes before the prediction moment.',
    ],
    decide: () => 'Split forward in time with a gap the length of the horizon, and audit every window feature for the future it can see.',
  },

  'group-structure': {
    more: [
      'The assumption every random split makes is that rows are independent draws. Repeated entities break it: several rows from one customer share unobserved traits, so a model that has seen some of them recognises the rest. The measured effect is large — group-aware splits routinely score several points lower than row-level ones on the same data, and the lower number is the true one.',
      'Finding the grouping is the work. The candidates are any identifier that repeats, any column with far fewer distinct values than rows, and any composite of them; the right level is the one that shares information, which is a domain judgement (household or customer, aircraft or flight, patient or visit). Once chosen, the constraint has to travel into cross-validation too, not just the initial split.',
    ],
    read: c => [
      c.groupCol ? `Read rows per group — about ${(c.rows / Math.max(1, c.groupCol.groups)).toFixed(1)} here — since anything above 1 means a row-level split straddles entities.` : 'Count distinct values per candidate id column and look for any that repeats; near-uniqueness in this extract does not prove one row per entity.',
      'Check the distribution of group sizes: a few huge groups can dominate a fold on their own.',
      'Confirm the group column is excluded from the features while still driving the split.',
    ],
    decide: () => 'Split and cross-validate by group whenever rows repeat an entity, and accept the lower score as the honest one.',
  },

  'diagnostic-uncertainty': {
    more: [
      'Every reported number has a sampling distribution. A metric on n test rows, a correlation on n complete cases, an importance from one fit — each would come out differently on another draw of the same size. The cheap general instrument is the bootstrap: resample the evaluation rows with replacement a few hundred times, recompute, and read the 2.5th and 97.5th percentiles as an interval.',
      'Formatting is not precision. Six decimal places on an MI estimate or a fourth digit on an AUC implies a stability that the sample size does not support; round to the precision your interval justifies and state the interval. Where a number drives a decision, the honest form is "0.78, 95% CI 0.74–0.82", not "0.7814".',
    ],
    read: c => [
      `Ask what sample each number came from: ${fmt.n(c.completeRows)} complete cases here versus ${fmt.n(c.rows)} rows analysed, and the two are not interchangeable.`,
      'Bootstrap any metric that drives a decision and read the width of the interval before the point estimate.',
      c.anomalies ? `Check whether a rate was configured rather than discovered — the anomaly screen was told to flag about ${fmt.pct(c.anomalies.contamination, 0)}.` : 'Check which numbers are thresholds you chose rather than quantities you measured.',
    ],
    decide: () => 'Attach an interval to every number that drives a decision, and round to the precision that interval supports.',
  },

  'outlier-screens': {
    more: [
      'The three screens ask three different questions. The IQR fence flags values outside q1 − 1.5·IQR to q3 + 1.5·IQR and is distribution-free; the z-score fence flags |value − mean| > 3σ and assumes symmetry, so on a skewed column it condemns the entire tail; a multivariate detector such as Isolation Forest flags rows that are unusual as combinations, and those rows may have no unusual single value at all.',
      'A flag is a question, and there are only four answers: a recording error (fix or drop), a rare true event (keep, it may be the point), a distinct subpopulation (segment or add a feature), or a sentinel (convert to missing). Deleting flagged rows without reaching one of those four is how the rare events you were hired to find get erased.',
    ],
    read: c => [
      `Read univariate rates per column — the worst here is ${(() => { const o = (c.numeric || []).filter(n => n.outlierRate > 0).sort((a, b) => b.outlierRate - a.outlierRate)[0]; return o ? `${o.name} at ${fmt.pct(o.outlierRate, 2)}` : 'none'; })()} — and remember the fence is a rule, not a verdict.`,
      'Compare the univariate and multivariate flag sets. Rows in one and not the other are the interesting ones.',
      'Inspect a handful of flagged rows in full. The explanation is almost always visible in the other columns.',
    ],
    decide: () => 'Resolve every flag to error, rare event, subpopulation or sentinel — and screen before splitting only for errors, never for shape.',
  },

  'class-imbalance': {
    more: [
      'Imbalance is a property of the base rate, and its first casualty is accuracy: at a 5% positive rate the always-negative predictor scores 95% and catches nothing. The metrics that survive are the ones conditioned on the classes — recall (of the actual positives, how many did we catch), precision (of our positive calls, how many were right), and average precision, which summarises that trade-off across all thresholds.',
      'The repairs are not equivalent. Class weights change the loss and keep every row. Random undersampling discards majority data. SMOTE synthesises minority rows and must run inside the fold, on training data only, or it leaks. All three change the base rate the model assumes, which decalibrates its probabilities — fine for ranking, wrong if the probability is the product.',
    ],
    read: c => [
      c.target && c.target.classes ? `Read the base rate first — ${c.target.classes.map(k => `${k.label} ${fmt.pct(k.count / c.rows, 1)}`).join(', ')} — and compute the majority-class accuracy as your floor.` : 'With no classification target, imbalance does not apply; the analogous question for a continuous target is tail concentration.',
      'Read precision and recall as a pair at a stated threshold; either one alone can be made perfect trivially.',
      'Ask whether the product is a ranking or a probability, since that decides whether resampling is acceptable at all.',
    ],
    decide: () => 'Report a ranking metric plus precision and recall at a stated threshold, and prefer class weights to resampling unless you can show the gain.',
  },

  'target-distribution': {
    more: [
      'For a continuous target the questions are spread, shape and whether error should scale with magnitude. If a £10 error matters equally at £100 and £10,000, absolute error is the right frame; if it is the percentage that matters, the target belongs on a log scale and the metric with it. That choice comes before the model, because it decides what "good" means.',
      'Heteroscedasticity is the shape that catches people: when the spread of the target grows with its level, a single global error figure describes neither end. Check by binning the target and reading the spread per bin — a fan shape means you should either transform, model the variance, or report error per band.',
    ],
    read: c => [
      c.target && c.target.stats ? `Read median against mean and the skew (${c.target.stats.skew.toFixed(2)}) to see how much the tail will dominate a squared-error loss.` : 'With no continuous target this does not apply; the analogous check for a classification target is the base rate.',
      'Bin the target and read the spread within each bin to test whether error is likely to scale with level.',
      'Check the boundaries: a target that cannot go below zero, or is capped, changes which model families are appropriate.',
    ],
    decide: () => 'Choose the error frame — absolute, squared, or relative — from the target\u2019s shape, and report at least one metric in the target\u2019s own units.',
  },

  'metric-selection': {
    more: [
      'Work backwards from the decision. Name the action the prediction triggers, name the two ways it can be wrong, and name the cost of each. The metric is whatever summarises those costs — which is why the same model can be excellent for triage (ranking, so AUC or average precision) and useless for pricing (probabilities, so log loss and a calibration curve).',
      'Fix the metric before the first fit, in writing, along with the population it is computed over and the threshold if there is one. Choosing afterwards is not a small sin: with a handful of metrics and a handful of models, something will look good by chance, and the number you publish will be the maximum of a set of noisy draws.',
    ],
    read: c => [
      c.target ? `Start from the task — ${c.target.task} on ${c.target.name} — and list the candidate metrics it admits before picking one.` : 'No target is declared, so the metric question is still open; declaring the target is the prerequisite.',
      'For each candidate metric, write the sentence "this number goes down when the model makes ___ mistakes". If you cannot, it is the wrong metric.',
      'Check that every number you plan to compare is computed on the same rows.',
    ],
    decide: () => 'Write one headline metric plus two or three diagnostics down before modelling, and do not change them afterwards.',
  },

  'thresholds-and-costs': {
    more: [
      'Where the cut belongs follows from the cost matrix. If a false negative costs C_FN and a false positive costs C_FP, the expected-cost-minimising cut is C_FP / (C_FP + C_FN) — with a false negative ten times as expensive as a false positive, that is about 0.09, nowhere near the default 0.5. The default is only correct when the errors cost the same and the classes are balanced.',
      'Choose the cut on validation data by sweeping thresholds and reading the metric you defined, then freeze it and report the confusion matrix at that value. If the model is retrained on a different base rate, or resampled, the cut has to be re-derived — a frozen threshold on a shifted score distribution is a silently different operating point.',
    ],
    read: c => [
      c.target && c.target.classes && c.target.classes.length === 2 ? `Read the base rate (${fmt.pct(c.target.classes[1].count / c.rows, 1)} positive) and compute the cost-implied cut before looking at any sweep.` : 'A single-cut decision only arises for a binary target; for multiclass the analogous choice is the decision rule over the score vector.',
      'Read the precision-recall curve at the candidate cut, not the AUC, since the cut is a point and the AUC is a summary of all of them.',
      'Check how flat the metric is near the chosen cut; a sharp peak means the choice will not survive new data.',
    ],
    decide: () => 'Derive the cut from stated costs, tune it on validation only, and publish the confusion matrix at the frozen value.',
  },

  'baselines': {
    more: [
      'Three baselines cost nothing and are all worth reporting: the trivial predictor (majority class, or the target median), the single-feature model (one obvious column, one shallow rule), and the incumbent — whatever rule or human process the business uses today. The last is the one that decides whether the project is worth shipping, and it is the one teams most often skip.',
      'Baselines also diagnose. A model barely above the trivial predictor means the features carry little signal; a model far above the incumbent on offline data and level with it in practice usually means leakage or a population mismatch. Both readings require having computed the baselines on the same rows and the same metric.',
    ],
    read: c => [
      c.target ? `Compute the trivial baseline on this target first — ${c.target.task === 'regression' && c.target.stats ? `predicting the median ${fmt.compact(c.target.stats.median)}` : 'predicting the majority class'} — and treat it as the floor.` : 'No target, so no baseline yet; this is the first number to compute once one exists.',
      'Write the incumbent rule down and score it on the same test rows.',
      'Compare the model\u2019s lift against the baseline interval, not against the baseline point.',
    ],
    decide: () => 'Report every model score next to the trivial and incumbent baselines on identical rows, or do not report it.',
  },

  'calibration': {
    more: [
      'Calibration is checked by grouping predictions into score bands and comparing the mean predicted probability with the observed rate in each band. Plotted, that is the reliability curve, and perfect calibration is the diagonal; above it the model is under-confident, below it over-confident. The summary numbers are Brier score and expected calibration error, and both are separate from any ranking metric.',
      'Two fixes, both fitted on held-out data: Platt scaling (a logistic fit on the scores, smooth, few parameters) and isotonic regression (monotone, non-parametric, needs more rows and can overfit small sets). Neither changes the ranking, so AUC will not move — which is precisely the point that a good AUC never implied calibration in the first place.',
    ],
    read: c => [
      c.target && c.target.classes ? `Bin the scores and compare predicted with observed rates against the ${c.target.classes.length === 2 ? fmt.pct(c.target.classes[1].count / c.rows, 1) : ''} base rate.` : 'Calibration applies to probabilistic classification only; for a continuous target the analogue is a residual-versus-prediction plot.',
      'Check whether the pipeline resampled or weighted the classes; if so, expect the probabilities to be shifted by construction.',
      'Read the curve, not just the Brier score — a single number hides which band is wrong.',
    ],
    decide: () => 'If a probability is the product, calibrate on held-out data and publish the reliability curve alongside the ranking metric.',
  },
};
