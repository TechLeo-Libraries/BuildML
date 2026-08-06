// EDA report engine — derives every number, sentence, colour and chart
// coordinate from a dataset descriptor. Nothing here is dataset-specific.

const A = {
  ink: 'var(--color-text)',
  muted: 'var(--color-neutral-600)',
  rule: 'var(--color-divider)',
  line: 'var(--color-accent-700)',
  fill: 'var(--color-accent-200)',
  solid: 'var(--color-accent-800)',
  deep: 'var(--color-accent-900)',
  paper: 'var(--color-bg)',
};

import { fmt, plural, list, truncate, names } from './eda-format.js';
import { CONCEPTS, STAGES } from './eda-concepts.js';
import { gates as buildGates } from './eda-gates.js';

export { fmt };

/* ── scales ───────────────────────────────────────────────────────── */

export function niceScale(max, count = 3) {
  if (!(max > 0) || !isFinite(max)) return { max: 1, ticks: [0, 0.5, 1] };
  const exp = Math.floor(Math.log10(max));
  const base = Math.pow(10, exp);
  const steps = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
  let top = base * 10;
  for (const s of steps) { if (base * s >= max * 1.001) { top = base * s; break; } }
  const ticks = [];
  for (let i = count; i >= 0; i--) ticks.push((top * i) / count);
  return { max: top, ticks };
}

const tickLabel = (v, kind) => {
  if (kind === 'pct') return `${Math.round(v * 100)}%`;
  if (kind === 'mi') return v === 0 ? '0' : v.toFixed(3).replace(/^0/, '');
  if (v === 0) return '0';
  return fmt.compact(v);
};

/* ── charts (percentage geometry, drawn as HTML boxes) ───────────── */

const HORIZONTAL_AT = 8;
const SCROLL_AT = 18;

const shapeOf = (d, threshold) => {
  const past = !!d.past;
  const hot = threshold != null && d.value >= threshold;
  return { fill: past ? A.solid : (hot ? A.fill : 'transparent'), stroke: past ? A.solid : A.line };
};

export function barChart(items, opts = {}) {
  const n = items.length;
  if (!n) return null;
  const horizontal = opts.horizontal != null ? opts.horizontal : n > HORIZONTAL_AT;
  const sc = niceScale(Math.max(...items.map(d => d.value), opts.threshold || 0));
  const ticks = sc.ticks.map(t => ({
    key: `t${t}`, label: tickLabel(t, opts.tickKind),
    topPct: (1 - t / sc.max) * 100, leftPct: (t / sc.max) * 100,
  }));
  if (horizontal) {
    return {
      orientation: 'h', count: n, ticks,
      rows: items.map((d, i) => Object.assign({
        key: `${d.label}-${i}`, name: truncate(d.label, 24), display: d.display,
        widthPct: Math.max(0.4, (d.value / sc.max) * 100),
      }, shapeOf(d, opts.threshold))),
      threshold: opts.threshold != null ? { leftPct: (opts.threshold / sc.max) * 100, label: opts.thresholdLabel } : null,
      scroll: n > SCROLL_AT, maxHeight: n > SCROLL_AT ? '324px' : 'none',
    };
  }
  const cap = n > 7 ? 9 : 13;
  return {
    orientation: 'v', count: n, ticks,
    bars: items.map((d, i) => Object.assign({
      key: `${d.label}-${i}`, label: truncate(d.label, cap), display: d.display,
      heightPct: Math.max(0.6, (d.value / sc.max) * 100),
    }, shapeOf(d, opts.threshold))),
    threshold: opts.threshold != null ? { topPct: (1 - opts.threshold / sc.max) * 100, label: opts.thresholdLabel } : null,
    scroll: false, maxHeight: 'none',
  };
}

function groupedBars(levels) {
  const sc = niceScale(Math.max(...levels.flatMap(l => [l.train, l.test])), 2);
  return {
    count: levels.length,
    ticks: sc.ticks.map(t => ({ key: `t${t}`, label: `${Math.round(t * 100)}%`, topPct: (1 - t / sc.max) * 100 })),
    groups: levels.map((l, i) => ({
      key: `${l.label}-${i}`, label: truncate(l.label, 9),
      trainPct: Math.max(0.6, (l.train / sc.max) * 100),
      testPct: Math.max(0.6, (l.test / sc.max) * 100),
    })),
  };
}

function histogram(counts, opts = {}) {
  const sc = niceScale(Math.max(...counts), 2);
  return {
    count: counts.length,
    ticks: sc.ticks.map(t => ({ key: `t${t}`, label: tickLabel(t), topPct: (1 - t / sc.max) * 100 })),
    bins: counts.map((c, i) => {
      const past = opts.cutIndex != null && i >= opts.cutIndex;
      return {
        key: `b${i}`, heightPct: Math.max(0.6, (c / sc.max) * 100), past,
        fill: past ? A.solid : 'transparent', stroke: past ? A.solid : A.line,
      };
    }),
    cut: opts.cutIndex != null ? { leftPct: (opts.cutIndex / counts.length) * 100, label: opts.cutLabel || 'cut' } : null,
    left: opts.leftLabel || '', right: opts.rightLabel || '', footnote: opts.footnote || '',
  };
}

/* ── severity ─────────────────────────────────────────────────────── */

const SEV_RANK = { crit: 0, high: 1, med: 2, low: 3, info: 4 };
const sevStyle = sev => {
  if (sev === 'crit') return { cls: 'tag', style: `background:${A.deep};color:${A.paper};text-transform:uppercase;letter-spacing:.06em` };
  if (sev === 'high') return { cls: 'tag', style: `background:${A.solid};color:${A.paper};text-transform:uppercase;letter-spacing:.06em` };
  if (sev === 'med') return { cls: 'tag tag-accent', style: 'text-transform:uppercase;letter-spacing:.06em' };
  if (sev === 'low') return { cls: 'tag tag-outline', style: 'text-transform:uppercase;letter-spacing:.06em' };
  return { cls: 'tag tag-neutral', style: 'text-transform:uppercase;letter-spacing:.06em' };
};

/* ── sample-size caution ──────────────────────────────────────────── */

function sampleNote(denominator, unit = 'rows') {
  if (denominator == null) return '';
  if (denominator < 30) return `Unreliable — drawn from ${fmt.n(denominator)} ${unit}. Treat the shape as anecdote, not distribution.`;
  if (denominator < 100) return `Small sample — ${fmt.n(denominator)} ${unit}. Differences between adjacent bars are probably noise.`;
  return '';
}

/* ── dataset context ──────────────────────────────────────────────── */

function context(ds) {
  const cols = ds.columns;
  const rows = ds.rows;
  const target = ds.target || null;
  const features = cols.filter(c => c.role === 'feature');
  const constants = cols.filter(c => c.distinct === 1).map(c => c.name);
  const idLike = cols.filter(c => c.role === 'id' || (c.role === 'feature' && c.distinct / Math.max(1, rows) > 0.98)).map(c => c.name);
  const missing = cols.filter(c => (c.missingRate || 0) > 0).sort((a, b) => b.missingRate - a.missingRate);
  const missingCells = Math.round(cols.reduce((s, c) => s + (c.missingRate || 0) * rows, 0));
  const completeness = 1 - missingCells / Math.max(1, rows * cols.length);
  const completeRows = ds.completeRows != null ? ds.completeRows : rows;
  const mi = cols.filter(c => c.mi != null && c.role === 'feature').sort((a, b) => b.mi - a.mi);
  const vif = cols.filter(c => c.vif != null).sort((a, b) => b.vif - a.vif);
  const vifThreshold = ds.vifThreshold != null ? ds.vifThreshold : 5;
  const drifted = cols.filter(c => c.drift).map(c => c.name);
  const eligible = features.filter(c => c.distinct > 1 && !idLike.includes(c.name)).length;
  // Numeric-ness comes from the declared dtype, never from whether summary
  // statistics happen to be present — a profile that omits quartiles still has
  // numeric columns, and the sheet must say so rather than claim there are none.
  const NUM_DTYPES = ['integer', 'float', 'numeric', 'int', 'double', 'decimal'];
  const numeric = cols.filter(x => NUM_DTYPES.includes(String(x.dtype || '').toLowerCase()))
    .map(x => Object.assign({}, x, {
      outlierRate: x.outlierRate || 0, zeros: x.zeros || 0, negatives: x.negatives || 0,
      hasStats: x.q != null || x.median != null,
    }));
  // Columns whose distribution was never summarised: an open question, not an
  // absence of numeric data.
  const unprofiled = numeric.filter(x => !x.hasStats);
  const profiledNumeric = numeric.filter(x => x.hasStats);
  // Numeric columns that are legitimate objects of analysis: identifiers are
  // numeric and must never be the column an example tells you to sweep, bin,
  // fence or describe.
  const analysable = numeric.filter(x => x.role !== 'id' && !idLike.includes(x.name));
  const categorical = cols.filter(x => x.dtype === 'categorical' || x.dtype === 'string' || x.dtype === 'boolean');
  const highCard = categorical.filter(x => x.distinct > 20).sort((a, b) => b.distinct - a.distinct);
  const skewed = numeric.filter(x => x.skew != null && Math.abs(x.skew) > 1).sort((a, b) => Math.abs(b.skew) - Math.abs(a.skew));
  const nearConstant = cols.filter(x => x.topShare != null && x.topShare >= 0.95 && x.distinct > 1)
    .map(x => ({ name: x.name, topShare: x.topShare }));
  const corrPairs = (ds.corrPairs || []).slice().sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
  const outlierCols = numeric.filter(x => x.outlierRate > 0).sort((a, b) => b.outlierRate - a.outlierRate);
  const spans = numeric.map(x => Math.abs((x.max || 0) - (x.min || 0))).filter(x => isFinite(x) && x > 0);
  return {
    ds, rows, rowsTotal: ds.rowsTotal != null ? ds.rowsTotal : rows, cols, colCount: cols.length,
    target, features, eligible, constants, idLike, missing, missingCells, completeness, completeRows,
    mi, vif, vifThreshold, drifted,
    anomalies: ds.anomalies || null,
    sampled: (ds.rowsTotal != null && ds.rowsTotal !== rows),
    numeric, profiledNumeric, unprofiled, analysable, categorical, highCard, skewed, nearConstant, corrPairs, outlierCols,
    hasCorr: corrPairs.length > 0,
    duplicates: ds.duplicates || null,
    leakage: ds.leakage || [],
    timeCol: ds.timeColumn || null,
    groupCol: ds.groupColumn || null,
    memoryMB: ds.memoryMB || null,
    mixed: cols.filter(x => x.mixedType),
    caseVar: cols.filter(x => x.caseVariants),
    rangeRatio: spans.length ? Math.max(...spans) / Math.min(...spans) : 1,
  };
}

/* ── findings, derived ────────────────────────────────────────────── */

const REGISTRY = [
  {
    key: 'validation.drift', sev: 'high', evidence: 'drift.flagged_columns', concept: 'dataset-drift',
    when: c => c.drifted.length > 0,
    detail: c => `${c.drifted.length} eligible ${plural(c.drifted.length, 'column')} met the configured drift flag thresholds. Observed: ${list(c.drifted)}.`,
  },
  {
    key: 'quality.completeness', sev: 'med', evidence: 'quality.missing_cells', concept: 'missing-data',
    when: c => c.missingCells > 0,
    detail: c => `${fmt.n(c.missingCells)} cells are missing; observed cell completeness is ${fmt.pct(c.completeness)}.`
      + (c.missing.length ? ` Worst column: ${c.missing[0].name} at ${fmt.pct(c.missing[0].missingRate)}.` : ''),
  },
  {
    key: 'quality.identifiers', sev: 'med', evidence: 'quality.id_like_columns', concept: 'column-roles',
    when: c => c.idLike.length > 0,
    detail: c => `Identifier-like columns have near-unique observed values and are not valid default predictors. Observed: ${list(c.idLike)}.`,
  },
  {
    key: 'quality.constants', sev: 'med', evidence: 'quality.constant_columns', concept: 'column-roles',
    when: c => c.constants.length > 0,
    detail: c => `Constant columns contain no observed variation. Observed: ${list(c.constants)}.`,
  },
  {
    key: 'relationships.vif', sev: 'med', evidence: 'multivariate.vif', concept: 'variance-inflation',
    when: c => c.vif.length > 0 && c.vif[0].vif >= c.vifThreshold,
    detail: c => {
      const over = c.vif.filter(v => v.vif >= c.vifThreshold);
      return `'${c.vif[0].name}' has VIF=${fmt.dec(c.vif[0].vif)} among complete-case eligible numeric features`
        + (over.length > 1 ? `; ${over.length} features sit above the ${c.vifThreshold.toFixed(1)} threshold.` : '.');
    },
  },
  {
    key: 'relationships.mi_leader', sev: 'info', evidence: 'bivariate.mutual_information', concept: 'mutual-information',
    when: c => c.mi.length > 0 && c.target,
    detail: c => `'${c.mi[0].name}' had the highest estimated mutual information with the target (${fmt.dec(c.mi[0].mi, 7)}).`,
  },
  {
    key: 'outliers.multivariate', sev: 'info', evidence: 'outliers.multivariate', concept: 'diagnostic-uncertainty',
    when: c => !!c.anomalies,
    detail: c => `Isolation Forest marked ${fmt.n(c.anomalies.flagged)} of ${fmt.n(c.anomalies.scored)} scored rows (${fmt.pct(c.anomalies.flagged / c.anomalies.scored)}) as anomalies.`,
  },
  {
    key: 'target.summary', sev: 'info', evidence: 'target.class_balance', concept: c => (c.target && c.target.task === 'regression' ? 'target-distribution' : 'class-imbalance'),
    when: c => !!c.target,
    detail: c => {
      const t = c.target;
      if (t.task === 'regression') {
        return `Observed regression target '${t.name}': median ${fmt.compact(t.stats.median)}, range ${fmt.compact(t.stats.min)} to ${fmt.compact(t.stats.max)}, skew ${t.stats.skew.toFixed(2)}.`;
      }
      const pos = t.classes && t.classes.length === 2 ? t.classes[t.classes.length - 1] : null;
      return `Observed ${t.task} classification for '${t.name}'`
        + (pos ? `; positive rate ${fmt.pct(pos.count / c.rows)}.` : ` across ${t.classes.length} observed classes.`);
    },
  },
  {
    key: 'target.missing', sev: 'crit', evidence: 'overview.roles', concept: 'column-roles',
    when: c => !c.target,
    detail: () => 'No target column is declared, so no supervised relationship, class balance or leakage screen could be computed.',
  },
  {
    key: 'quality.duplicates', sev: 'med', evidence: 'integrity.duplicate_rows', concept: 'duplicate-records',
    when: c => c.duplicates && (c.duplicates.rows > 0 || c.duplicates.keyDupes > 0),
    detail: c => {
      const d = c.duplicates;
      const bits = [];
      if (d.rows) bits.push(`${fmt.n(d.rows)} exact duplicate ${plural(d.rows, 'row')} (${fmt.pct(d.rows / c.rows, 2)})`);
      if (d.keyDupes) bits.push(`${fmt.n(d.keyDupes)} repeated ${plural(d.keyDupes, 'key')} on '${d.keyColumn}'`);
      return `${bits.join(' and ')} were observed; the table's grain is not one row per observation.`;
    },
  },
  {
    key: 'quality.high_cardinality', sev: 'med', evidence: 'quality.cardinality', concept: 'high-cardinality',
    when: c => c.highCard.length > 0,
    detail: c => `${c.highCard.length} categorical ${plural(c.highCard.length, 'column')} exceed 20 observed levels (${names(c.highCard, 3)}); one-hot encoding them all would add ${fmt.n(c.highCard.reduce((a, b) => a + b.distinct, 0))} columns.`,
  },
  {
    key: 'quality.near_constant', sev: 'low', evidence: 'quality.low_variance', concept: 'constant-and-near-constant',
    when: c => c.nearConstant.length > 0,
    detail: c => `${c.nearConstant.length} ${plural(c.nearConstant.length, 'column')} are near-constant — one level holds 95% or more of observed rows (${c.nearConstant.slice(0, 3).map(x => `${x.name} ${fmt.pct(x.topShare, 1)}`).join(', ')}).`,
  },
  {
    key: 'quality.text_hygiene', sev: 'low', evidence: 'integrity.string_variants', concept: 'text-hygiene',
    when: c => c.mixed.length > 0 || c.caseVar.length > 0,
    detail: c => {
      const bits = [];
      if (c.mixed.length) bits.push(`${c.mixed.length} ${plural(c.mixed.length, 'column')} hold mixed types (${names(c.mixed, 3)})`);
      if (c.caseVar.length) bits.push(`${fmt.n(c.caseVar.reduce((a, b) => a + b.caseVariants, 0))} case or whitespace variants across ${c.caseVar.length} ${plural(c.caseVar.length, 'column')}`);
      return `${bits.join('; ')}. Level counts on this sheet are computed on the raw strings.`;
    },
  },
  {
    key: 'quality.ranges', sev: 'low', evidence: 'univariate.ranges', concept: 'measurement-units-and-ranges',
    when: c => c.numeric.some(n => n.negatives > 0) || c.numeric.some(n => n.zeros / Math.max(1, c.rows) > 0.3),
    detail: c => {
      const negs = c.numeric.filter(n => n.negatives > 0);
      const zeros = c.numeric.filter(n => n.zeros / Math.max(1, c.rows) > 0.3);
      const bits = [];
      if (negs.length) bits.push(`${negs.length} numeric ${plural(negs.length, 'column')} contain negative values (${names(negs, 3)})`);
      if (zeros.length) bits.push(`${zeros.length} ${plural(zeros.length, 'column')} are more than 30% zeros (${names(zeros, 3)})`);
      return `${bits.join('; ')}. Whether those are legitimate is a domain question this sheet cannot settle.`;
    },
  },
  {
    key: 'distribution.skew', sev: 'low', evidence: 'univariate.skew', concept: 'skew-and-transforms',
    when: c => c.skewed.length > 0,
    detail: c => `${c.skewed.length} numeric ${plural(c.skewed.length, 'column')} have |skew| above 1, led by ${c.skewed[0].name} at ${c.skewed[0].skew.toFixed(2)}.`,
  },
  {
    key: 'relationships.correlated_pairs', sev: 'med', evidence: 'bivariate.correlation', concept: 'correlation',
    when: c => c.corrPairs.some(p => Math.abs(p.r) >= 0.8),
    detail: c => {
      const strong = c.corrPairs.filter(p => Math.abs(p.r) >= 0.8);
      return `${strong.length} feature ${plural(strong.length, 'pair')} correlate at |r| ≥ 0.8, strongest ${strong[0].a} × ${strong[0].b} at r=${strong[0].r.toFixed(3)}.`;
    },
  },
  {
    key: 'relationships.scaling', sev: 'info', evidence: 'univariate.ranges', concept: 'feature-scaling',
    when: c => c.numeric.length > 1 && c.rangeRatio > 100,
    detail: c => `Numeric ranges differ by a factor of about ${fmt.compact(c.rangeRatio)}, which matters for distance-based and regularised models and not at all for trees.`,
  },
  {
    key: 'outliers.univariate', sev: 'info', evidence: 'outliers.iqr', concept: 'outlier-screens',
    when: c => c.outlierCols.length > 0,
    detail: c => `${c.outlierCols.length} numeric ${plural(c.outlierCols.length, 'column')} have values beyond the IQR fences, led by ${c.outlierCols[0].name} at ${fmt.pct(c.outlierCols[0].outlierRate, 2)}.`,
  },
  {
    key: 'validation.leakage', sev: 'high', evidence: 'validation.leakage_suspects', concept: 'leakage',
    when: c => c.leakage.length > 0,
    detail: c => `${c.leakage.length} leakage ${plural(c.leakage.length, 'suspect')} flagged by heuristic: ${c.leakage.slice(0, 3).map(x => `${x.name} (${x.reason})`).join(', ')}. A heuristic screen is not proof of absence.`,
  },
  {
    key: 'validation.temporal', sev: 'med', evidence: 'overview.time_column', concept: 'temporal-structure',
    when: c => !!c.timeCol,
    detail: c => `Rows carry a time in '${c.timeCol.name}' spanning ${c.timeCol.min} to ${c.timeCol.max}${c.timeCol.gaps ? ` with ${fmt.n(c.timeCol.gaps)} observed coverage ${plural(c.timeCol.gaps, 'gap')}` : ''}; a random split would train on the future.`,
  },
  {
    key: 'validation.grouping', sev: 'med', evidence: 'overview.group_column', concept: 'group-structure',
    when: c => !!c.groupCol,
    detail: c => `'${c.groupCol.name}' identifies ${fmt.n(c.groupCol.groups)} ${plural(c.groupCol.groups, 'group')} across ${fmt.n(c.rows)} rows (about ${(c.rows / Math.max(1, c.groupCol.groups)).toFixed(1)} rows each), so rows are not independent.`,
  },
  {
    key: 'validation.sampling', sev: 'med', evidence: 'overview.sampling', concept: 'diagnostic-uncertainty',
    when: c => c.sampled,
    detail: c => `This sheet describes a ${fmt.pct(c.rows / c.rowsTotal, 1)} sample — ${fmt.n(c.rows)} of ${fmt.n(c.rowsTotal)} rows — so tail behaviour and rare levels may be under-represented.`,
  },
  {
    key: 'evaluation.metric', sev: 'info', evidence: 'target.summary', concept: 'metric-selection',
    when: c => !!c.target,
    detail: c => c.target.task === 'regression'
      ? `Metric choice is open: MAE reads in units of ${c.target.name}, RMSE weights the tail that skew ${c.target.stats ? c.target.stats.skew.toFixed(2) : ''} produces.`
      : `Metric choice is open: accuracy is not informative at this base rate, and no metric has been fixed for '${c.target.name}' yet.`,
  },
  {
    key: 'evaluation.baseline', sev: 'info', evidence: 'target.baseline', concept: 'baselines',
    when: c => !!c.target,
    detail: c => {
      const t = c.target;
      if (t.task === 'regression' && t.stats) return `Baseline to beat: predicting the median ${fmt.compact(t.stats.median)} for every row.`;
      const major = t.classes ? [...t.classes].sort((a, b) => b.count - a.count)[0] : null;
      return major ? `Baseline to beat: predicting "${major.label}" for every row scores ${fmt.pct(major.count / c.rows, 1)} accuracy.` : 'Baseline not computable for this target shape.';
    },
  },
  {
    key: 'eda.scope', sev: 'info', evidence: 'overview', concept: 'data-splitting',
    when: () => true,
    detail: c => `EDA examined ${fmt.n(c.rows)} of ${fmt.n(c.rowsTotal)} rows across ${c.colCount} columns.`
      + (c.rows < 100 ? ` At this size every estimate below carries wide uncertainty.` : ''),
  },
];

const RECIPE = {
  'validation.drift': { title: 'Investigate flagged train/test shifts', call: () => 'session.explain("split", moment="after")', when: 'before_modeling' },
  'quality.identifiers': { title: 'Keep identifiers outside feature matrices', call: c => `session.set_roles({${c.idLike.slice(0, 2).map(n => `"${n}": "id"`).join(', ')}})`, when: 'before_modeling' },
  'quality.constants': { title: 'Exclude constant columns', call: c => `session.set_roles({${c.constants.slice(0, 2).map(n => `"${n}": "ignore"`).join(', ')}})`, when: 'before_modeling' },
  'quality.completeness': { title: 'Define a train-fitted missing-data strategy', call: () => 'session.impute(strategy="median")', when: 'before_modeling' },
  'relationships.vif': { title: 'Review correlated feature groups', call: () => 'session.explain("reduce_dimensions")', when: 'next' },
  'target.missing': { title: 'Declare a target before anything else', call: () => 'session.set_roles({"<column>": "target"})', when: 'before_modeling' },
  'target.summary': { title: 'Choose metrics that survive this target shape', call: () => 'session.explain("evaluate", moment="before")', when: 'next' },
  'validation.leakage': { title: 'Remove leakage suspects from the feature set', call: c => `session.set_roles({"${c.leakage[0].name}": "ignore"})`, when: 'before_modeling' },
  'quality.duplicates': { title: 'Fix the table grain before splitting', call: c => `session.deduplicate(subset=["${(c.duplicates && c.duplicates.keyColumn) || (c.idLike[0] || '<key>')}"])`, when: 'before_modeling' },
  'validation.temporal': { title: 'Split forward in time, not at random', call: c => `session.split(strategy="time_ordered", time_column="${c.timeCol.name}")`, when: 'before_modeling' },
  'validation.grouping': { title: 'Split by entity so groups do not straddle', call: c => `session.split(strategy="group", group_column="${c.groupCol.name}")`, when: 'before_modeling' },
  'quality.high_cardinality': { title: 'Group rare levels before encoding', call: c => `session.group_rare_levels(columns=["${c.highCard[0].name}"], min_frequency=0.01)`, when: 'before_modeling' },
  'relationships.correlated_pairs': { title: 'Resolve near-duplicate feature pairs', call: () => 'session.correlations(method="spearman", threshold=0.8)', when: 'next' },
  'distribution.skew': { title: 'Consider an in-fold transform for skewed columns', call: c => `session.transform(method="log1p", columns=["${c.skewed[0].name}"])`, when: 'next' },
  'quality.text_hygiene': { title: 'Clean strings, then recount levels', call: () => 'session.clean_strings(strip=True, case="lower")', when: 'before_modeling' },
  'quality.near_constant': { title: 'Inspect the rare level before dropping', call: c => `session.value_counts("${c.nearConstant[0].name}")`, when: 'next' },
  'outliers.univariate': { title: 'Inspect IQR-fence outliers before excluding any', call: c => `session.outliers(method="iqr", columns=["${c.outlierCols[0].name}"])`, when: 'next' },
  'quality.ranges': { title: 'Confirm ranges and sentinel values with the source', call: () => 'session.replace_sentinels([-999, -1])', when: 'before_modeling' },
  'evaluation.baseline': { title: 'Record the trivial baseline before any model', call: () => 'session.baseline(strategy="most_frequent")', when: 'next' },
  'validation.sampling': { title: 'Re-run the screens on the full frame before deciding', call: () => 'session.eda(sample=None)', when: 'next' },
};

const ORDER = [
  'target.missing', 'validation.leakage', 'quality.duplicates', 'validation.drift', 'validation.temporal',
  'validation.grouping', 'quality.identifiers', 'quality.constants', 'quality.completeness',
  'quality.text_hygiene', 'quality.ranges', 'quality.high_cardinality', 'relationships.correlated_pairs',
  'relationships.vif', 'distribution.skew', 'quality.near_constant', 'outliers.univariate',
  'validation.sampling', 'evaluation.baseline', 'target.summary',
];

/* ── figures ──────────────────────────────────────────────────────── */

function figures(c) {
  const out = [];

  out.push(c.mi.length ? {
    key: 'mi', n: '5.1', caption: `mutual information vs ${c.target ? c.target.name : 'target'}`,
    kind: 'bars', chart: barChart(c.mi.map(m => ({ label: m.name, value: m.mi, display: fmt.short(m.mi, 4) })), { tickKind: 'mi' }),
    note: 'MI does not establish direction or causality',
    concept: 'mutual-information', sample: sampleNote(c.completeRows, 'complete cases'),
  } : {
    key: 'mi', n: '5.1', caption: 'mutual information vs target', kind: 'empty',
    empty: c.target ? 'No eligible feature produced a mutual-information estimate on this frame.' : 'Nothing found — no target column is declared, so there is no relationship to score. Declaring one would rank every eligible feature here.',
    concept: 'mutual-information',
  });

  out.push(c.vif.length ? {
    key: 'vif', n: '5.2', caption: 'variance inflation, eligible numeric features',
    kind: 'bars', chart: barChart(c.vif.map(v => ({ label: v.name, value: v.vif, display: v.vif.toFixed(2) })), { threshold: c.vifThreshold, thresholdLabel: `threshold ${c.vifThreshold.toFixed(1)}` }),
    note: c.vif.filter(v => v.vif >= c.vifThreshold).length
      ? `${c.vif.filter(v => v.vif >= c.vifThreshold).length} above threshold. VIF is sensitive to the included feature set`
      : 'Nothing above threshold. VIF is sensitive to the included feature set',
    concept: 'variance-inflation', sample: sampleNote(c.completeRows, 'complete cases'),
  } : {
    key: 'vif', n: '5.2', caption: 'variance inflation', kind: 'empty',
    empty: 'Nothing found — this frame has fewer than two eligible numeric features, so there is no linear dependence to measure.',
    concept: 'variance-inflation',
  });

  const t = c.target;
  if (t && t.task === 'regression' && t.histogram) {
    out.push({
      key: 'target', n: '5.3', caption: `distribution of ${t.name}`, kind: 'hist',
      chart: histogram(t.histogram, {
        leftLabel: fmt.compact(t.stats.min), rightLabel: fmt.compact(t.stats.max),
        footnote: `median ${fmt.compact(t.stats.median)} · skew ${t.stats.skew.toFixed(2)}`,
      }),
      note: `Continuous target — no class balance applies`, concept: 'target-distribution', sample: sampleNote(c.rows),
    });
  } else if (t && t.classes) {
    const total = t.classes.reduce((s, k) => s + k.count, 0);
    out.push({
      key: 'target', n: '5.3', caption: `class balance of ${t.name}`, kind: 'bars',
      chart: barChart(t.classes.map(k => ({ label: k.label, value: k.count, display: `${fmt.n(k.count)} · ${fmt.pct(k.count / total, 1)}` })), { horizontal: true }),
      note: `A constant "${[...t.classes].sort((a, b) => b.count - a.count)[0].label}" predictor scores ${fmt.pct(Math.max(...t.classes.map(k => k.count)) / total, 1)} accuracy`,
      concept: 'class-imbalance', sample: sampleNote(Math.min(...t.classes.map(k => k.count)), 'rows in the smallest class'),
    });
  } else {
    out.push({
      key: 'target', n: '5.3', caption: 'target summary', kind: 'empty',
      empty: 'Nothing found — no target column is declared. This slot would carry class balance for a classification target, or a distribution for a continuous one.',
      concept: 'column-roles',
    });
  }

  const d = c.ds.driftDetail;
  out.push(d && d.levels ? {
    key: 'drift', n: '5.4', caption: `drift: ${d.column} shares`, kind: 'grouped',
    chart: groupedBars(d.levels), note: `One of ${c.drifted.length} flagged ${plural(c.drifted.length, 'column')}. Check split construction first`,
    concept: 'dataset-drift', sample: sampleNote(d.testRows, 'test rows'),
  } : {
    key: 'drift', n: '5.4', caption: 'drift: train vs test shares', kind: 'empty',
    empty: c.drifted.length
      ? `${c.drifted.length} ${plural(c.drifted.length, 'column')} ${c.drifted.length === 1 ? 'is' : 'are'} flagged, but no per-level shares were recorded for them, so there is nothing to draw here.`
      : 'Nothing found — no column met the configured drift thresholds on this split.',
    concept: 'dataset-drift',
  });

  out.push(c.anomalies && c.anomalies.histogram ? {
    key: 'anomaly', n: '5.5', caption: 'anomaly score distribution', kind: 'hist',
    chart: histogram(c.anomalies.histogram, {
      cutIndex: c.anomalies.cutIndex, cutLabel: 'cut', leftLabel: 'low score', rightLabel: 'high',
      footnote: `${fmt.n(c.anomalies.flagged)} of ${fmt.n(c.anomalies.scored)} scored rows marked · ${fmt.pct(c.anomalies.flagged / c.anomalies.scored)}`,
    }),
    note: 'Screening signals, not confirmed errors', concept: 'diagnostic-uncertainty',
    sample: sampleNote(c.anomalies.scored, 'scored rows'),
  } : {
    key: 'anomaly', n: '5.5', caption: 'anomaly score distribution', kind: 'empty',
    empty: 'Nothing found — no multivariate outlier screen ran on this frame.',
    concept: 'diagnostic-uncertainty',
  });

  return out;
}

/* ── ledger ───────────────────────────────────────────────────────── */

const chunk = (items, per) => {
  if (items.length <= per) return [{ key: 'c0', items }];
  const half = Math.ceil(items.length / 2);
  return [{ key: 'c0', items: items.slice(0, half) }, { key: 'c1', items: items.slice(half) }];
};

function ledger(c, findings) {
  const sevCount = s => findings.filter(f => f.sev === s).length;
  const g = [];
  g.push({
    key: 'frame', title: 'Frame', wide: false, cols: chunk([
      ['rows analysed', fmt.n(c.rows)],
      ['rows in frame', fmt.n(c.rowsTotal)],
      ['columns', String(c.colCount)],
      ['eligible features', String(c.eligible)],
      ['complete rows', fmt.n(c.completeRows)],
      ['missing cells', fmt.n(c.missingCells)],
      ['cell completeness', fmt.pct(c.completeness)],
      ['sampling', c.sampled ? `${fmt.pct(c.rows / c.rowsTotal, 1)} of frame` : 'none'],
    ].map(([k, v], i) => ({ key: `f${i}`, k, v, mono: false })), 99),
  });

  const missItems = c.missing.map((m, i) => ({ key: `m${i}`, k: m.name, v: fmt.pct(m.missingRate), mono: true }));
  const complete = c.colCount - c.missing.length;
  if (complete > 0) missItems.push({ key: 'm-rest', k: `${complete} other ${plural(complete, 'column')}`, v: '0.000%', mono: true, muted: true });
  g.push({ key: 'missing', title: 'Missing rate by column', wide: c.missing.length > 12, cols: chunk(missItems, 12) });

  g.push({
    key: 'roles', title: 'Roles & severity', wide: false, cols: chunk([
      ['feature', String(c.features.length)],
      ['identifier-like', String(c.idLike.length)],
      ['target', c.target ? '1' : '0'],
      ['constant', String(c.constants.length)],
      ['findings · crit / high', `${sevCount('crit')} / ${sevCount('high')}`],
      ['findings · med / low', `${sevCount('med')} / ${sevCount('low')}`],
      ['findings · info', String(sevCount('info'))],
    ].map(([k, v], i) => ({ key: `r${i}`, k, v, mono: false })), 99),
  });

  if (c.mi.length) {
    g.push({
      key: 'mi', title: `Mutual information vs ${c.target ? c.target.name : 'target'}`, wide: c.mi.length > 12,
      cols: chunk(c.mi.map((m, i) => ({ key: `i${i}`, k: m.name, v: fmt.dec(m.mi, 6), mono: true })), 12),
    });
  }
  if (c.vif.length) {
    g.push({
      key: 'vif', title: 'Variance inflation (complete case)', wide: c.vif.length > 12,
      cols: chunk(c.vif.map((v, i) => ({ key: `v${i}`, k: v.name, v: fmt.dec(v.vif, 5), mono: true })), 12),
    });
  }

  const t = c.target;
  const screens = [['target column', t ? t.name : 'not declared'], ['task', t ? t.task : '—']];
  if (t && t.classes) {
    t.classes.forEach(k => screens.push([`class · ${k.label}`, `${fmt.n(k.count)} · ${fmt.pct(k.count / c.rows, 1)}`]));
  } else if (t && t.stats) {
    screens.push(['median', fmt.compact(t.stats.median)], ['min / max', `${fmt.compact(t.stats.min)} / ${fmt.compact(t.stats.max)}`], ['skew', t.stats.skew.toFixed(2)]);
  }
  if (c.anomalies) {
    screens.push(['anomalies / scored', `${fmt.n(c.anomalies.flagged)} / ${fmt.n(c.anomalies.scored)}`], ['anomaly rate', fmt.pct(c.anomalies.flagged / c.anomalies.scored)]);
  }
  screens.push(['drift flags', String(c.drifted.length)], ['VIF threshold', c.vifThreshold.toFixed(1)]);
  g.push({ key: 'screens', title: 'Target & screens', wide: false, cols: chunk(screens.map(([k, v], i) => ({ key: `s${i}`, k, v, mono: false })), 99) });

  return g;
}

/* ── report ───────────────────────────────────────────────────────── */

export function buildReport(ds, opts = {}) {
  const c = context(ds);
  const academyHref = opts.academyHref || 'EDA Sheet - Academy.dc.html';
  const cockpitHref = opts.cockpitHref || 'EDA Sheet - Cockpit.dc.html';

  const findings = REGISTRY.filter(r => r.when(c))
    .map(r => {
      const slug = typeof r.concept === 'function' ? r.concept(c) : r.concept;
      const s = sevStyle(r.sev);
      return {
        key: r.key, anchor: `f-${r.key.replace(/\./g, '-')}`, sev: r.sev, sevLabel: r.sev,
        sevClass: s.cls, sevStyle: s.style, detail: r.detail(c), evidence: r.evidence,
        concept: slug, conceptHref: `${academyHref}#${slug}`,
      };
    })
    .sort((a, b) => SEV_RANK[a.sev] - SEV_RANK[b.sev]);

  const conceptOrder = [];
  findings.forEach(f => { if (!conceptOrder.includes(f.concept)) conceptOrder.push(f.concept); });

  const assumptions = conceptOrder.map(slug => ({
    key: slug, slug, href: `${academyHref}#${slug}`,
    text: CONCEPTS[slug].prose[0].split('. ').slice(0, 2).join('. ') + '.',
  }));

  const recommendations = ORDER.filter(k => findings.some(f => f.key === k) && RECIPE[k])
    .map((k, i) => ({
      key: k, n: String(i + 1), when: RECIPE[k].when, title: RECIPE[k].title,
      call: RECIPE[k].call(c), basis: k,
    }));

  const figs = figures(c).map(f => Object.assign({}, f, {
    conceptHref: `${academyHref}#${f.concept}`,
    isBars: f.kind === 'bars' && f.chart, isHist: f.kind === 'hist', isGrouped: f.kind === 'grouped',
    isEmpty: f.kind === 'empty',
    isV: f.kind === 'bars' && f.chart && f.chart.orientation === 'v',
    isH: f.kind === 'bars' && f.chart && f.chart.orientation === 'h',
  }));

  const blocking = findings.filter(f => f.sev === 'crit' || f.sev === 'high').length;

  const entry = slug => {
    const def = CONCEPTS[slug];
    const cited = findings.filter(f => f.concept === slug);
    const readTexts = (typeof def.read === 'function' ? def.read(c) : (def.read || []));
    const decideText = typeof def.decide === 'function' ? def.decide(c) : (def.decide || '');
    const search = [slug, def.title, ...def.prose, def.session(c), def.example(c), decideText,
      ...readTexts, ...def.pitfalls(c)].join(' ').toLowerCase();
    return {
      search, citeCount: cited.length,
      key: slug, slug, title: def.title, stage: def.stage,
      prose: def.prose.map((p, i) => ({ key: `p${i}`, text: p })),
      session: def.session(c), example: def.example(c),
      pitfalls: def.pitfalls(c).map((p, i) => ({ key: `x${i}`, text: p })),
      read: (typeof def.read === 'function' ? def.read(c) : (def.read || []))
        .map((t, i) => ({ key: `d${i}`, text: t })),
      decide: typeof def.decide === 'function' ? def.decide(c) : (def.decide || ''),
      hasDecide: !!def.decide,
      cited: cited.length > 0,
      citations: cited.map(f => ({ key: f.key, label: `cited by ${f.key} →`, href: `${cockpitHref}#${f.anchor}` })),
    };
  };

  const citedStages = STAGES.map(s => ({
    key: s.n, n: s.n, label: s.label, blurb: s.blurb,
    entries: conceptOrder.filter(slug => CONCEPTS[slug].stage === s.key).map(entry),
  })).filter(s => s.entries.length);

  // Every concept is taught in full, always — grouped by stage, cited entries
  // first inside their stage so the session's own findings lead the section.
  const allSlugs = Object.keys(CONCEPTS);
  const allStages = STAGES.map(s => {
    const inStage = allSlugs.filter(slug => CONCEPTS[slug].stage === s.key);
    const entries = [
      ...inStage.filter(slug => conceptOrder.includes(slug)),
      ...inStage.filter(slug => !conceptOrder.includes(slug)),
    ].map(entry).map(e => Object.assign({}, e, { uncited: !e.cited }));
    const citedN = entries.filter(e => e.cited).length;
    return {
      key: s.n, n: s.n, label: s.label, blurb: s.blurb, entries,
      count: `${entries.length} ${plural(entries.length, 'concept')} · ${citedN} cited here`,
    };
  }).filter(s => s.entries.length);

  return {
    label: ds.label, session: ds.session, engineName: ds.engine, version: ds.version,
    academyHref, cockpitHref,
    readiness: blocking ? 'Blocked' : (findings.some(f => f.sev === 'med') ? 'Caution' : 'Clear'),
    readinessNote: blocking
      ? `${blocking} blocking ${plural(blocking, 'finding')}`
      : (findings.some(f => f.sev === 'med') ? `${findings.filter(f => f.sev === 'med').length} findings to resolve` : 'no findings raised'),
    scope: `${fmt.n(c.rows)} / ${fmt.n(c.rowsTotal)}`,
    scopeNote: `rows analysed · ${c.colCount} columns`,
    completeness: fmt.pct(c.completeness),
    completenessNote: `${fmt.n(c.missingCells)} missing cells`,
    runtime: ds.engine,
    runtimeNote: ds.runtimeNote || '',
    findings, assumptions, ledger: ledger(c, findings), recommendations, figures: figs,
    conceptCount: Object.keys(CONCEPTS).length, citedCount: conceptOrder.length,
    stageCount: allStages.length,
    citedStages, allStages,
    gates: buildGates(c, findings, academyHref),
    academyKicker: `${ds.engine} ${ds.version} · companion sheet · ${Object.keys(CONCEPTS).length} concepts, ${conceptOrder.length} cited in this session`,
    nav: STAGES.map(s => ({
      key: s.n, n: s.n, label: s.label,
      items: Object.keys(CONCEPTS).filter(slug => CONCEPTS[slug].stage === s.key)
        .map(slug => ({ key: slug, slug, href: `#${slug}`, cited: conceptOrder.includes(slug) })),
    })).filter(s => s.items.length),
  };
}
