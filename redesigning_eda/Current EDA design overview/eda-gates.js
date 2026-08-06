// The readiness gates: every question that must be settled before a frame is
// modelled, one row each, with the status derived from this session's numbers.
//
// Four statuses, and the distinction between them is the point of the sheet:
//   clear — the frame's own evidence settles it
//   open  — something measurable is unresolved, and the sheet says what
//   human — the data cannot answer it; a person must record a decision
//   na    — the question does not arise for this frame

import { fmt, plural, list, names } from './eda-format.js';

const S = { clear: 'clear', open: 'open', human: 'human', na: 'na' };

export const GATE_LABELS = {
  clear: 'settled by the frame',
  open: 'open — measurable',
  human: 'needs a recorded decision',
  na: 'not applicable',
};

// [id, stage, question, concept slug, resolver(ctx) -> {status, evidence, closes}]
const GATES = [
  ['00.1', 0, 'Is there one written sentence saying who acts on this model\u2019s output, and when?', 'problem-framing', c => ({
    status: S.human,
    evidence: `${fmt.n(c.rows)} rows and ${c.colCount} columns were profiled; no decision statement is attached to the session.`,
    closes: 'One sentence: who acts, on what output, at what moment.',
  })],
  ['00.2', 0, 'Is it written down what one row represents, and does a key prove it is unique?', 'unit-of-analysis', c => ({
    status: (c.idLike || []).length ? S.open : S.human,
    evidence: (c.idLike || []).length
      ? `${list(c.idLike, 3)} ${c.idLike.length === 1 ? 'is' : 'are'} near-unique across ${fmt.n(c.rows)} rows, but no uniqueness assertion ran.`
      : 'No near-unique column was observed, so no candidate key is visible in the frame.',
    closes: 'A named key with a uniqueness check that passes.',
  })],
  ['00.3', 0, 'Do we know which rows the extract filtered out before we saw it?', 'population-and-sampling-frame', c => ({
    status: S.human,
    evidence: `${fmt.n(c.rows)} of ${fmt.n(c.rowsTotal)} rows examined${c.sampled ? ' under sampling' : ''}; upstream filters are invisible here.`,
    closes: 'The extract query, the window, and what was excluded.',
  })],
  ['00.4', 0, 'Is the label\u2019s exact rule recorded — what counts as positive, measured from when, over how long?', 'target-definition', c => ({
    status: c.target ? S.human : S.open,
    evidence: c.target
      ? `${c.target.name} is used as a ${c.target.task} target; how it was constructed is not recorded.`
      : 'No target is declared, so this is not yet a supervised problem.',
    closes: c.target ? 'The label rule with its anchor, horizon and censoring policy.' : 'A declared target column.',
  })],
  ['00.5', 0, 'For each column, do we know its source system and whether its value exists at prediction time?', 'provenance-and-lineage', c => ({
    status: S.human,
    evidence: `${c.colCount} columns profiled on ${c.ds.engine} ${c.ds.version}; no lineage or refresh schedule travels with them.`,
    closes: 'Source, derivation and known-at time per column.',
  })],
  ['00.6', 0, 'Has someone listed which columns are personal or legally protected?', 'sensitive-attributes', c => ({
    status: S.human,
    evidence: `No sensitivity classification is attached to any of the ${c.colCount} columns.`,
    closes: 'An inventory naming personal, protected and neither — plus which are kept for evaluation only.',
  })],

  ['01.1', 1, 'Was each column\u2019s type set deliberately, rather than guessed by the CSV loader?', 'dtypes-and-storage', c => ({
    status: S.human,
    evidence: (() => { const k = {}; c.cols.forEach(x => { k[x.dtype] = (k[x.dtype] || 0) + 1; }); return `${Object.keys(k).map(t => `${k[t]} ${t}`).join(', ')} — as loaded, not as asserted.`; })(),
    closes: 'An explicit dtype per column at load time.',
  })],
  ['01.2', 1, 'Does each column with gaps have a chosen fill strategy, fitted on training rows only?', 'missing-data', c => ({
    status: c.missingCells === 0 ? S.clear : S.open,
    evidence: c.missingCells === 0
      ? `No missing cells across ${c.colCount} columns.`
      : `${fmt.n(c.missingCells)} cells missing across ${c.missing.length} ${plural(c.missing.length, 'column')}; worst is ${c.missing[0].name} at ${fmt.pct(c.missing[0].missingRate, 1)}.`,
    closes: 'One in-fold strategy per gappy column, plus indicators where the gap may be informative.',
  })],
  ['01.3', 1, 'For each gappy column, do we know whether the gaps are random or systematic?', 'missingness-mechanisms', c => ({
    status: (c.missing || []).length ? S.open : S.na,
    evidence: (c.missing || []).length
      ? `${c.missing.length} ${plural(c.missing.length, 'column')} carry gaps; no mechanism was inferred for any of them.`
      : 'Nothing is missing in this extract.',
    closes: 'MCAR / MAR / MNAR recorded per column, on evidence.',
  })],
  ['01.4', 1, 'Have exact duplicate rows and repeated keys been counted and resolved?', 'duplicate-records', c => {
    const d = c.duplicates;
    const ok = d && !d.rows && !d.keyDupes;
    return {
      status: d ? (ok ? S.clear : S.open) : S.open,
      evidence: !d ? 'No duplicate screen ran on this frame.'
        : ok ? `No exact duplicates and no repeated keys across ${fmt.n(c.rows)} rows.`
        : `${fmt.n(d.rows)} exact duplicate ${plural(d.rows, 'row')}${d.keyDupes ? ` and ${fmt.n(d.keyDupes)} repeated ${plural(d.keyDupes, 'key')} on ${d.keyColumn}` : ''}.`,
      closes: 'A stated grain, deduplication before the split, and near-duplicate checks after string cleaning.',
    };
  }],
  ['01.5', 1, 'Have columns with one dominant value been dropped, coarsened, or knowingly kept?', 'constant-and-near-constant', c => {
    const n = (c.constants || []).length + (c.nearConstant || []).length;
    return {
      status: n ? S.open : S.clear,
      evidence: n
        ? `${(c.constants || []).length} constant and ${(c.nearConstant || []).length} near-constant ${plural(n, 'column')}${(c.constants || []).length ? ` (${list(c.constants, 3)})` : ''}.`
        : 'No constant or near-constant columns were observed.',
      closes: 'Constants ignored, near-constants coarsened or knowingly kept.',
    };
  }],
  ['01.6', 1, 'Does each category column with many levels have an encoding plan and an unseen-level policy?', 'high-cardinality', c => ({
    status: (c.highCard || []).length ? S.open : S.clear,
    evidence: (c.highCard || []).length
      ? `${c.highCard.length} categorical ${plural(c.highCard.length, 'column')} exceed 20 levels; one-hot would add about ${fmt.n(c.highCard.reduce((s, x) => s + x.distinct, 0))} columns.`
      : 'No categorical column exceeds 20 observed levels.',
    closes: 'Group-rare, in-fold target encoding or attribute replacement — plus an unseen-level policy.',
  })],
  ['01.7', 1, 'Does each numeric column have an allowed min/max, with codes like -999 turned into missing?', 'measurement-units-and-ranges', c => {
    const negs = (c.numeric || []).filter(n => n.negatives > 0);
    return {
      status: negs.length ? S.open : S.human,
      evidence: negs.length
        ? `${negs.length} numeric ${plural(negs.length, 'column')} contain negatives (${names(negs, 3)}); legitimacy is a domain question.`
        : 'No negative values observed; no range assertion has been declared either.',
      closes: 'A min/max assertion per numeric column and sentinels converted to missing.',
    };
  }],
  ['01.8', 1, 'Were text columns trimmed and case-folded before their category levels were counted?', 'text-hygiene', c => {
    const mixed = (c.cols || []).filter(x => x.mixedType).length;
    const varia = (c.cols || []).filter(x => x.caseVariants).length;
    return {
      status: mixed + varia ? S.open : S.human,
      evidence: mixed + varia
        ? `${mixed} mixed-type and ${varia} case-variant ${plural(mixed + varia, 'column')}; level counts on this sheet use raw strings.`
        : 'No mixed-type or case-variant columns were observed; no normalisation was applied either.',
      closes: 'Strip, normalise, case-fold, recount — and the same transform at prediction time.',
    };
  }],
  ['01.9', 1, 'For every join that built this table, do we know it added or dropped no rows unexpectedly?', 'join-integrity', c => ({
    status: S.human,
    evidence: `${fmt.n(c.rows)} rows are present; no join history travels with the frame.`,
    closes: 'Expected cardinality and a minimum match rate per join.',
  })],
  ['01.10', 1, 'Are contradictions between columns tested — end before start, parts not summing to total?', 'cross-field-consistency', c => ({
    status: S.human,
    evidence: `${c.colCount} columns were profiled independently; no cross-field constraint was tested.`,
    closes: 'One assertion per known relationship, run on every extract.',
  })],
  ['01.11', 1, 'Were date columns parsed with an explicit format and timezone, and does the span look right?', 'datetime-parsing', c => ({
    status: c.timeCol ? S.open : S.human,
    evidence: c.timeCol
      ? `${c.timeCol.name} spans ${c.timeCol.min} to ${c.timeCol.max}; format and timezone are not recorded.`
      : 'No column is typed as a date, which is not the same as the rows being order-free.',
    closes: 'An explicit parse with format and tz, and a span checked against reality.',
  })],
  ['01.12', 1, 'Do we know how coarsely each number was recorded, and whether values pile up at a cap?', 'precision-and-heaping', c => ({
    status: S.human,
    evidence: `${(c.numeric || []).length} numeric ${plural((c.numeric || []).length, 'column')} were summarised; digit preference and edge pile-ups were not tested.`,
    closes: 'True precision per column and edge masses classified as caps or tails.',
  })],

  ['02.1', 2, 'Has each numeric column been read as a distribution — quartiles and histogram — not just a mean?', 'univariate-distributions', c => ({
    status: !(c.numeric || []).length ? S.na : ((c.unprofiled || []).length ? S.open : S.clear),
    evidence: !(c.numeric || []).length
      ? 'This frame has no numeric columns.'
      : (c.unprofiled || []).length
        ? `${c.numeric.length} numeric ${plural(c.numeric.length, 'column')} present, but ${c.unprofiled.length} of them carry no quartiles or range in this profile.`
        : `${c.numeric.length} numeric ${plural(c.numeric.length, 'column')} summarised with quartiles and ranges.`,
    closes: 'Quartiles plus a histogram per column, with the shape named.',
  })],
  ['02.2', 2, 'For each skewed column, is there a decision to transform it or not, and why?', 'skew-and-transforms', c => ({
    status: (c.skewed || []).length ? S.open : S.clear,
    evidence: (c.skewed || []).length
      ? `${c.skewed.length} ${plural(c.skewed.length, 'column')} exceed |skew| 1: ${c.skewed.slice(0, 3).map(x => `${x.name} (${x.skew.toFixed(2)})`).join(', ')}.`
      : 'No numeric column exceeds |skew| 1.',
    closes: 'Transform or not, per column, decided against the model family and fitted in-fold.',
  })],
  ['02.3', 2, 'Have columns that are just re-expressions of other columns been removed?', 'derived-and-redundant-columns', c => {
    const near = (c.corrPairs || []).filter(p => Math.abs(p.r) >= 0.95);
    return {
      status: !c.hasCorr ? S.open : (near.length ? S.open : S.clear),
      evidence: !c.hasCorr
        ? 'No pairwise correlations were supplied in this profile, so redundancy has not been screened.'
        : near.length
          ? `${near.length} ${plural(near.length, 'pair')} correlate above |0.95| (${near.slice(0, 2).map(p => `${p.a} × ${p.b}`).join('; ')}).`
          : `No pair correlates above |0.95|; strongest is ${c.corrPairs[0].r.toFixed(3)}.`,
      closes: 'The derived member removed, the measured one kept.',
    };
  }],
  ['02.4', 2, 'Are features independent enough that a model\u2019s coefficients can be trusted?', 'variance-inflation', c => {
    const over = (c.vif || []).filter(v => v.vif >= c.vifThreshold);
    return {
      status: over.length ? S.open : ((c.vif || []).length ? S.clear : S.na),
      evidence: !(c.vif || []).length ? 'No eligible numeric feature set to compute VIF against.'
        : over.length ? `${over.length} ${plural(over.length, 'feature')} above the ${c.vifThreshold.toFixed(1)} threshold, led by ${over[0].name} at ${over[0].vif.toFixed(2)}.`
        : `All features below the ${c.vifThreshold.toFixed(1)} threshold; highest is ${c.vif[0].vif.toFixed(2)}.`,
      closes: 'One member removed at a time, recomputed, until coefficients can be read.',
    };
  }],
  ['02.5', 2, 'Has each feature\u2019s relationship to the target been checked for curves and reversals, not just straight lines?', 'non-linearity-and-binning', c => ({
    status: S.open,
    evidence: c.hasCorr
      ? `Every relationship figure here is linear or monotone (${c.corrPairs.length} pairs, ${(c.mi || []).length} MI estimates); a saturating or reversing shape would not appear.`
      : `No pairwise correlations were supplied in this profile${(c.mi || []).length ? `, only ${c.mi.length} MI estimates` : ''}; shape has not been examined at all.`,
    closes: 'Target mean per decile for each candidate feature, with the shape named.',
  })],
  ['02.6', 2, 'Was each headline relationship re-checked inside subgroups, in case it reverses?', 'confounding-and-subgroups', c => ({
    status: (c.categorical || []).length ? S.open : S.na,
    evidence: (c.categorical || []).length
      ? `All figures are pooled across ${fmt.n(c.rows)} rows; ${c.categorical.length} categorical ${plural(c.categorical.length, 'column')} were available as stratifiers and none was used.`
      : 'No categorical column is available to stratify by.',
    closes: 'Each headline relationship recomputed within at least one plausible confounder.',
  })],
  ['02.7', 2, 'Are there enough rows per feature — after encoding — for a model to learn rather than memorise?', 'sparsity-and-dimensionality', c => {
    const ratio = c.rows / Math.max(1, c.eligible);
    return {
      status: ratio >= 10 ? S.clear : S.open,
      evidence: `${fmt.n(c.rows)} rows over ${c.eligible} eligible features — about ${Math.round(ratio)} rows per feature before encoding.`,
      closes: 'Above about ten after encoding, reached by removing redundancy and coarsening categories first.',
    };
  }],
  ['02.8', 2, 'Is there a recorded decision on whether to scale features, based on the model chosen?', 'feature-scaling', c => ({
    status: (c.numeric || []).length ? S.human : S.na,
    evidence: (() => {
      if (!(c.numeric || []).length) return 'No numeric columns, so no scaling decision arises.';
      const s = (c.profiledNumeric || []).map(n => Math.abs(n.max - n.min)).filter(x => isFinite(x) && x > 0);
      return s.length
        ? `Numeric ranges differ by a factor of about ${fmt.compact(Math.max(...s) / Math.min(...s))}; no scaler is declared.`
        : `${c.numeric.length} numeric ${plural(c.numeric.length, 'column')} present but unranged in this profile; no scaler is declared.`;
    })(),
    closes: 'A named scaler fitted in-fold, or a recorded decision that the model does not need one.',
  })],

  ['03.1', 3, 'Does the train/test split respect time order and repeated entities, rather than splitting at random?', 'data-splitting', c => ({
    status: (c.timeCol || c.groupCol) ? S.open : S.human,
    evidence: c.timeCol ? `${c.timeCol.name} orders the rows, so a random split would train on the future.`
      : c.groupCol ? `${c.groupCol.name} repeats across ${fmt.n(c.rows)} rows, so a row-level split would straddle entities.`
      : 'No time or group structure is declared; row independence is assumed rather than verified.',
    closes: 'A split drawn on the structure the data has, with the test set touched once.',
  })],
  ['03.2', 3, 'Will each split keep the target\u2019s class balance, and was that verified after splitting?', 'stratification', c => {
    if (!c.target) return { status: S.na, evidence: 'No target is declared.', closes: 'A declared target.' };
    const small = c.target.classes ? Math.min(...c.target.classes.map(k => k.count)) : null;
    return {
      status: S.open,
      evidence: small != null
        ? `Smallest class holds ${fmt.n(small)} rows; a 20% unstratified test set would carry about ${fmt.n(Math.round(small * 0.2))}.`
        : `${c.target.name} is continuous, so stratification would need binning.`,
      closes: 'Stratified split with the resulting shares verified per membership.',
    };
  }],
  ['03.3', 3, 'Is every step that learns from data — imputer, encoder, scaler — fitted after the split, not before?', 'pipeline-order', c => ({
    status: S.open,
    evidence: `Every statistic here — medians, correlations, VIF, MI${c.anomalies ? ', anomaly scores' : ''} — was computed on the full ${fmt.n(c.rows)} rows.`,
    closes: 'A pipeline object below the split that the cross-validator refits per fold.',
  })],
  ['03.4', 3, 'Has every column been confirmed knowable at prediction time, with no post-outcome values?', 'leakage', c => ({
    status: (c.leakage || []).length ? S.open : S.human,
    evidence: (c.leakage || []).length
      ? `${c.leakage.length} ${plural(c.leakage.length, 'suspect')} flagged by heuristic: ${c.leakage.slice(0, 3).map(x => x.name).join(', ')}.`
      : `No suspect flagged${(c.idLike || []).length ? `, though ${list(c.idLike, 2)} must stay out of the matrix` : ''}; column timing was not verified.`,
    closes: 'A known-at time per column and every fitted step below the split.',
  })],
  ['03.5', 3, 'Do the split and every window feature look only backwards in time?', 'temporal-structure', c => ({
    status: c.timeCol ? S.open : S.na,
    evidence: c.timeCol
      ? `${c.timeCol.name} spans ${c.timeCol.min} to ${c.timeCol.max}${c.timeCol.gaps ? ` with ${fmt.n(c.timeCol.gaps)} coverage ${plural(c.timeCol.gaps, 'gap')}` : ''}.`
      : 'No time column is declared in this frame.',
    closes: 'A forward split with a horizon-length gap, and every window closed before the prediction moment.',
  })],
  ['03.6', 3, 'If rows repeat the same entity, does the split keep that entity on one side?', 'group-structure', c => ({
    status: c.groupCol ? S.open : ((c.idLike || []).length ? S.human : S.human),
    evidence: c.groupCol
      ? `${c.groupCol.name} identifies ${fmt.n(c.groupCol.groups)} groups — about ${(c.rows / Math.max(1, c.groupCol.groups)).toFixed(1)} rows each.`
      : `No group column is declared${(c.idLike || []).length ? `; ${list(c.idLike, 2)} looks near-unique in this extract` : ''}.`,
    closes: 'A group-aware split and cross-validation, or a recorded finding that rows are independent.',
  })],
  ['03.7', 3, 'Are the rows used to pick the model different from the rows used to report its score?', 'nested-validation', c => ({
    status: S.human,
    evidence: `${fmt.n(c.rows)} rows would leave about ${fmt.n(Math.round(c.rows * 0.2))} per 20% slice${c.rows < 3000 ? ', thin enough to prefer nested cross-validation' : ''}.`,
    closes: 'Nested CV or a three-way split, with the reported number from rows that influenced nothing.',
  })],
  ['03.8', 3, 'Is this sample large enough to detect a difference small enough to matter?', 'sample-size-and-power', c => ({
    status: c.rows < 1000 ? S.open : S.human,
    evidence: `At ${fmt.n(c.rows)} rows a proportion metric carries roughly ±${(1.96 * Math.sqrt(0.25 / Math.max(1, c.rows)) * 100).toFixed(1)} points before splitting.`,
    closes: 'The smallest actionable effect stated, and the interval width compared against it.',
  })],
  ['03.9', 3, 'Given how many statistics were screened, are the strongest results corrected for chance?', 'multiple-comparisons', c => {
    const tests = (c.corrPairs || []).length + (c.mi || []).length;
    return {
      status: tests > 20 ? S.open : S.clear,
      evidence: `${tests} screening ${plural(tests, 'statistic')} were computed across ${c.colCount} columns with no multiplicity correction.`,
      closes: 'A false-discovery correction on reported p-values, and survivors re-checked out of sample.',
    };
  }],
  ['03.10', 3, 'Could someone else re-run this and get the same numbers — seed, library versions, data snapshot?', 'reproducibility', c => ({
    status: S.open,
    evidence: `Run on ${c.ds.engine} ${c.ds.version} over ${fmt.n(c.rows)} of ${fmt.n(c.rowsTotal)} rows${c.sampled ? ' under sampling' : ''}; no seed is reported.`,
    closes: 'A seed per random step, a sensitivity check across seeds, and a pinned extract.',
  })],
  ['03.11', 3, 'For each drift flag, was the split and the pipeline ruled out before blaming the data?', 'dataset-drift', c => ({
    status: (c.drifted || []).length ? S.open : S.clear,
    evidence: (c.drifted || []).length
      ? `${c.drifted.length} ${plural(c.drifted.length, 'column')} met the configured thresholds: ${list(c.drifted, 5)}.`
      : 'No column met the configured thresholds — an absence of flags at your threshold.',
    closes: 'The split, the ingestion and the population ruled out in that order, and the shift named.',
  })],
  ['03.12', 3, 'Is each outlier explained as an error, a rare true event, a subgroup, or a sentinel code?', 'outlier-screens', c => {
    const uni = (c.numeric || []).filter(n => n.outlierRate > 0).length;
    return {
      status: (uni || c.anomalies) ? S.open : S.clear,
      evidence: c.anomalies
        ? `${fmt.n(c.anomalies.flagged)} of ${fmt.n(c.anomalies.scored)} scored rows marked at a ${fmt.pct(c.anomalies.contamination, 0)} configured contamination${uni ? `, plus ${uni} ${plural(uni, 'column')} with IQR-fence values` : ''}.`
        : uni ? `${uni} numeric ${plural(uni, 'column')} carry values beyond the IQR fences.`
        : 'No univariate or multivariate outlier flags were raised.',
      closes: 'Each flag classified as error, rare event, subpopulation or sentinel.',
    };
  }],

  ['04.1', 4, 'Was the scoring metric written down before the first model was fitted?', 'metric-selection', c => ({
    status: c.target ? S.human : S.na,
    evidence: c.target ? `${c.target.name} is ${c.target.task}; no metric is declared in this session.` : 'No target is declared.',
    closes: 'One headline metric plus diagnostics, with the population and threshold stated.',
  })],
  ['04.2', 4, 'Do we know what the dumbest predictor, and today\u2019s existing process, would score?', 'baselines', c => ({
    status: c.target ? S.open : S.na,
    evidence: c.target
      ? (c.target.task === 'regression' && c.target.stats
        ? `Predicting the median (${fmt.compact(c.target.stats.median)}) is the floor and has not been scored.`
        : `Predicting the majority class scores ${c.target.classes ? fmt.pct(Math.max(...c.target.classes.map(k => k.count)) / c.rows, 1) : 'the base rate'} and catches nothing.`)
      : 'No target is declared.',
    closes: 'Trivial and incumbent baselines scored on the same rows and the same metric.',
  })],
  ['04.3', 4, 'Given the class balance, is the metric one that a majority-class guess cannot win?', 'class-imbalance', c => ({
    status: (c.target && c.target.task !== 'regression' && c.target.classes) ? S.open : S.na,
    evidence: (c.target && c.target.classes)
      ? `${c.target.classes.map(k => `${k.label} ${fmt.pct(k.count / c.rows, 1)}`).join(', ')} — accuracy is uninformative at this balance.`
      : 'No classification target in this frame.',
    closes: 'A ranking metric plus precision and recall at a stated threshold; class weights preferred to resampling.',
  })],
  ['04.4', 4, 'Does the cut-off come from the relative cost of a false alarm versus a miss, not from 0.5?', 'thresholds-and-costs', c => ({
    status: (c.target && c.target.classes && c.target.classes.length === 2) ? S.open : S.na,
    evidence: (c.target && c.target.classes && c.target.classes.length === 2)
      ? `Positive rate ${fmt.pct(c.target.classes[1].count / c.rows, 1)}; the default 0.5 cut assumes equal costs and an even base rate.`
      : 'No binary target, so no single cut arises.',
    closes: 'A cut derived from C_FP/(C_FP+C_FN), tuned on validation and frozen.',
  })],
  ['04.5', 4, 'Does every number someone will act on carry an uncertainty range?', 'uncertainty-intervals', c => ({
    status: S.open,
    evidence: `Every figure on the sheet is a point estimate from ${fmt.n(c.rows)} rows${c.completeRows !== c.rows ? ` (${fmt.n(c.completeRows)} complete cases for some)` : ''}.`,
    closes: 'Bootstrap intervals at the level of independence, and rounding that matches them.',
  })],
  ['04.6', 4, 'Will performance be reported per segment, not only as one overall figure?', 'slice-evaluation', c => ({
    status: (c.categorical || []).length ? S.open : S.human,
    evidence: (c.categorical || []).length
      ? `${c.categorical.length} categorical ${plural(c.categorical.length, 'column')} could define slices${c.timeCol ? ', plus periods' : ''}; all figures here are pooled.`
      : 'No categorical column is available to slice by.',
    closes: 'Predefined slices with support and intervals, and a named response to every material gap.',
  })],
  ['04.7', 4, 'If the output is used as a probability, has it been checked against observed rates?', 'calibration', c => ({
    status: (c.target && c.target.task !== 'regression' && c.target.classes) ? S.human : S.na,
    evidence: (c.target && c.target.classes)
      ? `${c.target.name} would produce scores; whether they are used as probabilities is a product decision, not a data one.`
      : 'No probabilistic classification target here.',
    closes: 'Either a recorded decision that ranking suffices, or a reliability curve on held-out data.',
  })],

  ['05.1', 5, 'Is the feature-importance method named, run on held-out rows, and reported with its variability?', 'feature-importance-methods', c => ({
    status: S.open,
    evidence: `${c.eligible} eligible ${plural(c.eligible, 'feature')} would enter the calculation${(c.corrPairs || []).filter(p => Math.abs(p.r) >= 0.8).length ? `, of which ${(c.corrPairs || []).filter(p => Math.abs(p.r) >= 0.8).length} correlated ${plural((c.corrPairs || []).filter(p => Math.abs(p.r) >= 0.8).length, 'pair')} will share credit` : ''}.`,
    closes: 'Permutation importance on held-out rows, repeated, with redundant groups noted.',
  })],
  ['05.2', 5, 'Are feature-effect curves drawn only across the range where rows actually exist?', 'effect-shapes', c => ({
    status: (c.numeric || []).length ? S.open : S.na,
    evidence: (c.numeric || []).length
      ? `${c.numeric.length} numeric ${plural(c.numeric.length, 'feature')} could be swept${(c.skewed || []).length ? `, ${c.skewed.length} of them skewed enough to narrow the supported range sharply` : ''}.`
      : 'No numeric features to sweep.',
    closes: 'Partial-dependence and ICE curves clipped to the data\u2019s own percentiles, with density shown.',
  })],
  ['05.3', 5, 'Do we know whether more data or better features is the lever, before spending on either?', 'learning-curves-and-capacity', c => ({
    status: S.open,
    evidence: `About ${Math.round(c.rows / Math.max(1, c.eligible))} rows per feature${c.rows / Math.max(1, c.eligible) < 20 ? ' — the regime where variance usually dominates' : ''}; no curve has been drawn.`,
    closes: 'A learning curve with fold error bars, read before any hyper-parameter search.',
  })],
  ['05.4', 5, 'Is every finding stated as an association, without implying that changing the feature changes the outcome?', 'causal-caution', c => ({
    status: S.open,
    evidence: `${(c.corrPairs || []).length} correlation ${plural((c.corrPairs || []).length, 'pair')}${(c.mi || []).length ? ` and ${c.mi.length} MI ${plural(c.mi.length, 'estimate')}` : ''} from observational rows; no assignment mechanism is recorded.`,
    closes: 'Associations reported as associations, with the confounders that would need measuring listed.',
  })],
  ['05.5', 5, 'Do the assumptions, chosen thresholds and monitoring owners exist as a written handoff?', 'handoff-and-monitoring', c => ({
    status: S.open,
    evidence: `Configured thresholds include ${c.vifThreshold ? `VIF ${c.vifThreshold.toFixed(1)}` : 'collinearity'}${c.anomalies ? ` and contamination ${fmt.pct(c.anomalies.contamination, 0)}` : ''}; no owner or review date is attached.`,
    closes: 'Assumptions ledger, decision log, pinned extract and a monitoring plan with named owners.',
  })],
];

export function gates(c, findings, academyHref) {
  const rows = GATES.map(([id, stage, question, concept, resolve]) => {
    const r = resolve(c);
    const cited = findings.filter(f => f.concept === concept);
    return {
      key: id, id, stage, question, concept,
      conceptHref: `${academyHref}#${concept}`,
      status: r.status, statusLabel: GATE_LABELS[r.status],
      evidence: r.evidence, closes: r.closes,
      findings: cited.map(f => ({ key: f.key, label: f.key })),
      isClear: r.status === S.clear, isOpen: r.status === S.open,
      isHuman: r.status === S.human, isNa: r.status === S.na,
    };
  });
  const count = s => rows.filter(g => g.status === s).length;
  const answerable = rows.filter(g => g.status !== S.na).length;
  return {
    rows,
    counts: { clear: count(S.clear), open: count(S.open), human: count(S.human), na: count(S.na), total: rows.length, answerable },
    settledPct: fmt.pct(count(S.clear) / Math.max(1, answerable), 0),
  };
}
