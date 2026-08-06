// The concept curriculum. Every entry is written once and re-reads the
// current session's numbers through its session()/example()/pitfalls(ctx).

import { fmt, plural, list, names } from './eda-format.js';

const some = xs => xs && xs.length;

export const CONCEPTS = {

  /* ── Stage 1 · data quality ─────────────────────────────────────── */

  'column-roles': {
    stage: 1, title: 'column-roles',
    prose: [
      'A role states how a column participates in modeling — feature, target, identifier, ignore — and is independent of its dtype. An integer identifier and an integer count have the same storage and opposite meanings, so the declaration has to come from you.',
      'Roles left undeclared do not stay neutral: whatever is in the frame ends up in the matrix. An identifier there lets the model memorise rows; a column recorded after the outcome lets it read the answer.',
    ],
    session: c => {
      const bits = [];
      if (some(c.idLike)) bits.push(`${list(c.idLike)} ${c.idLike.length === 1 ? 'is' : 'are'} near-unique across ${fmt.n(c.rows)} rows`);
      if (some(c.constants)) bits.push(`${list(c.constants)} ${c.constants.length === 1 ? 'has' : 'have'} a single observed value`);
      return `${bits.length ? `${bits.join('; ')}. ` : 'No identifier-like or constant columns were observed. '}Of ${c.colCount} columns, ${c.features.length} are marked feature, ${c.idLike.length} identifier-like and ${c.target ? 1 : 0} target, leaving ${c.eligible} eligible features.`;
    },
    example: c => {
      const lines = [];
      (c.idLike || []).slice(0, 2).forEach(n => lines.push(`  "${n}": "id",`));
      (c.constants || []).slice(0, 2).forEach(n => lines.push(`  "${n}": "ignore",`));
      if (c.target) lines.push(`  "${c.target.name}": "target",`);
      if (!lines.length) lines.push('  # nothing to reassign');
      return `session.set_roles({\n${lines.join('\n')}\n})\n\nsession.roles()\n# → ${c.eligible} eligible features`;
    },
    pitfalls: () => [
      'Assuming near-uniqueness is always an identifier — a genuine continuous measurement is also near-unique.',
      'Assuming a constant column is harmless. It is, until the pipeline scales it and divides by zero variance.',
      'Declaring roles after the split has already been drawn on the unfiltered frame.',
    ],
  },

  'dtypes-and-storage': {
    stage: 1, title: 'dtypes-and-storage',
    prose: [
      'A dtype is a storage decision, not a semantic one. Booleans stored as strings, categories stored as integers and dates stored as text all load without complaint and then silently change what every downstream operation means.',
      'Storage also sets the cost. Wide frames spend most of their memory on object columns; converting them to categoricals often cuts the footprint by an order of magnitude without touching a single value.',
    ],
    session: c => {
      const counts = {};
      c.cols.forEach(col => { counts[col.dtype] = (counts[col.dtype] || 0) + 1; });
      const summary = Object.keys(counts).map(k => `${counts[k]} ${k}`).join(', ');
      return `${c.colCount} columns: ${summary}.${c.memoryMB ? ` The frame occupies about ${c.memoryMB.toFixed(1)} MB in ${c.ds.engine}.` : ''}`;
    },
    example: c => `session.schema()\n\n# confirm each dtype is the\n# one you intended, not the\n# one the loader guessed\nsession.cast({\n  "${(c.categorical[0] && c.categorical[0].name) || '<column>'}": "category"})`,
    pitfalls: () => [
      'Letting the CSV loader infer types, then treating the inference as documentation.',
      'Numeric-looking identifiers: a zero-padded code loses its padding the moment it becomes an integer.',
      'Dates left as strings, which sort lexically and make any temporal check meaningless.',
    ],
  },

  'missing-data': {
    stage: 1, title: 'missing-data',
    prose: [
      'A missing cell is an absence of record, not a value of zero. Imputation replaces it with a rule — a median, a most-frequent category, a model — learned from the rows you are allowed to learn from. That makes the matrix dense enough to fit; it does not recover what was never written down.',
      'The rule must be fitted on training rows only. A median computed over the whole frame carries information from the test rows into training, and the score you get back is optimistic by an amount you cannot measure.',
    ],
    session: c => c.missingCells === 0
      ? `No missing cells were observed across ${c.colCount} columns, so no imputation rule is needed. That is a property of this extract, not a guarantee about future loads.`
      : `${fmt.n(c.missingCells)} cells are missing across ${c.missing.length} ${plural(c.missing.length, 'column')}; observed cell completeness is ${fmt.pct(c.completeness)}. ${c.missing[0].name} is worst at ${fmt.pct(c.missing[0].missingRate)}, so its observed shares describe the ${fmt.n(Math.round(c.rows * (1 - c.missing[0].missingRate)))} rows that answered rather than the population.`,
    example: c => {
      const nums = c.missing.filter(m => m.dtype === 'float' || m.dtype === 'integer').slice(0, 3).map(m => m.name);
      const cats = c.missing.filter(m => m.dtype === 'categorical' || m.dtype === 'boolean' || m.dtype === 'string').slice(0, 2).map(m => m.name);
      if (!nums.length && !cats.length) return 'session.explain(\n  "impute", moment="before")\n\n# nothing to fill in this frame';
      let out = '';
      if (nums.length) out += `session.impute(\n  strategy="median",\n  columns=[${nums.map(n => `"${n}"`).join(',\n           ')}])\n\n`;
      if (cats.length) out += `session.impute(\n  strategy="most_frequent",\n  columns=[${cats.map(n => `"${n}"`).join(',\n           ')}])\n\n`;
      return `${out}# fitted on train rows only`;
    },
    pitfalls: c => [
      'Computing the fill value over the full frame — leakage, and the most common form of it.',
      some(c.missing) ? `Treating a ${fmt.pct(c.missing[0].missingRate, 1)} gap as random; mechanisms were not inferred here.` : 'Assuming a complete extract means a complete source.',
      c.rows > c.completeRows ? `Dropping incomplete rows silently: complete-case analysis discards ${fmt.n(c.rows - c.completeRows)} of ${fmt.n(c.rows)} rows.` : 'Dropping incomplete rows without counting them first.',
    ],
  },

  'missingness-mechanisms': {
    stage: 1, title: 'missingness-mechanisms',
    prose: [
      'Three mechanisms behave differently. Missing completely at random loses precision only; missing at random can be repaired using the other columns; missing not at random encodes the very thing you are predicting, and no imputation rule recovers it.',
      'You cannot distinguish them from the missing counts alone. What you can do is test whether missingness in one column predicts the target or correlates with another column — and, when it does, keep the indicator instead of pretending the gap was noise.',
    ],
    session: c => some(c.missing)
      ? `${c.missing.length} ${plural(c.missing.length, 'column')} carry gaps, led by ${c.missing[0].name} at ${fmt.pct(c.missing[0].missingRate, 1)}. No mechanism was inferred for any of them — the sheet reports the rate, not the reason.`
      : 'Nothing is missing in this extract, so there is no mechanism to classify.',
    example: c => `# keep the fact of absence:\nsession.add_missing_indicator(\n  ["${(c.missing[0] && c.missing[0].name) || '<column>'}"])\n\nsession.explain(\n  "impute", moment="before")`,
    pitfalls: () => [
      'Imputing first and asking about the mechanism afterwards — the evidence is gone.',
      'Filling a "not recorded" category with the mode, which invents a positive answer.',
      'Discarding missing-indicator columns as noise when the absence itself is predictive.',
    ],
  },

  'duplicate-records': {
    stage: 1, title: 'duplicate-records',
    prose: [
      'Duplicate rows inflate the apparent sample size, bias every share towards whatever was double-counted, and — if a duplicate pair straddles a split — leak the answer directly from train to test.',
      'Exact duplicates are the easy case. Key duplicates, where the identifier repeats with different payloads, are the interesting one: they mean the grain of the table is not the grain you assumed.',
    ],
    session: c => {
      const d = c.duplicates;
      if (!d) return 'No duplicate screen ran on this frame.';
      if (!d.rows && !d.keyDupes) return `No exact duplicate rows and no repeated keys were observed across ${fmt.n(c.rows)} rows, so one row appears to be one observation.`;
      return `${fmt.n(d.rows)} exact duplicate ${plural(d.rows, 'row')} (${fmt.pct(d.rows / c.rows, 2)})${d.keyDupes ? ` and ${fmt.n(d.keyDupes)} repeated ${plural(d.keyDupes, 'key')} on ${d.keyColumn}` : ''} were observed.`;
    },
    example: c => `session.duplicates(\n  subset=["${(c.idLike && c.idLike[0]) || '<key>'}"])\n\n# decide the grain, then dedupe\n# before the split, never after`,
    pitfalls: () => [
      'De-duplicating after splitting, which leaves the copies on opposite sides.',
      'Dropping duplicates when the repetition is real — two identical transactions can both be genuine.',
      'Checking only exact duplicates and missing near-duplicates that differ by whitespace or case.',
    ],
  },

  'constant-and-near-constant': {
    stage: 1, title: 'constant-and-near-constant',
    prose: [
      'A constant column carries no information; a near-constant one carries almost none while still costing a coefficient, a split candidate and a scaling step. Both survive most pipelines quietly.',
      'Near-constant is the more dangerous shape, because the rare level is often the interesting one. Dropping the column throws the signal away; keeping it unbalanced means most folds never see the minority value.',
    ],
    session: c => {
      const nc = c.nearConstant || [];
      if (!some(c.constants) && !some(nc)) return 'No constant or near-constant columns were observed.';
      const parts = [];
      if (some(c.constants)) parts.push(`${list(c.constants)} ${c.constants.length === 1 ? 'has' : 'have'} one observed value`);
      if (some(nc)) parts.push(`${nc.map(x => `${x.name} at ${fmt.pct(x.topShare, 1)} one level`).join(', ')}`);
      return `${parts.join('; ')}.`;
    },
    example: c => `session.set_roles({${(c.constants || []).slice(0, 2).map(n => `\n  "${n}": "ignore",`).join('')}\n})\n\n# for near-constants, check the\n# rare level before dropping`,
    pitfalls: () => [
      'Auto-dropping low-variance columns without looking at what the rare value means.',
      'Scaling a constant column — zero variance, division by zero, silent NaNs.',
      'Assuming a column constant in this extract is constant in production.',
    ],
  },

  'high-cardinality': {
    stage: 1, title: 'high-cardinality',
    prose: [
      'A categorical column with hundreds of levels spreads the data thin: most levels appear a handful of times, one-hot encoding explodes the matrix width, and any level absent from a training fold has no learned representation at prediction time.',
      'The usual repairs are grouping the tail into "other", target encoding fitted inside the fold, or replacing the identity with an attribute of it — a city becomes a region, a code becomes a family.',
    ],
    session: c => {
      const hc = c.highCard || [];
      return hc.length
        ? `${hc.length} categorical ${plural(hc.length, 'column')} exceed 20 observed levels: ${hc.slice(0, 4).map(x => `${x.name} (${fmt.n(x.distinct)})`).join(', ')}${hc.length > 4 ? `, and ${hc.length - 4} more` : ''}. One-hot encoding all of them would add ${fmt.n(hc.reduce((s, x) => s + x.distinct, 0))} columns.`
        : `No categorical column exceeds 20 observed levels${c.categorical.length ? `; the widest is ${c.categorical.slice().sort((a, b) => b.distinct - a.distinct)[0].name}` : ''}.`;
    },
    example: c => `session.group_rare_levels(\n  columns=["${(c.highCard && c.highCard[0] && c.highCard[0].name) || (c.categorical[0] && c.categorical[0].name) || '<column>'}"],\n  min_frequency=0.01,\n  other_label="other")`,
    pitfalls: () => [
      'One-hot encoding a high-cardinality column and then wondering why the model overfits.',
      'Target encoding fitted on the full frame — a textbook leak.',
      'Grouping the tail without recording which levels went into "other".',
    ],
  },

  'categorical-encoding': {
    stage: 1, title: 'categorical-encoding',
    prose: [
      'Most estimators need numbers, so every category becomes one through a mapping. That mapping is fitted, which puts it inside the training fold alongside every other fitted step.',
      'The choice matters more than it looks. One-hot is safe and wide; ordinal is compact and asserts an order that may not exist; target encoding is powerful and leaks the moment it sees a validation row.',
    ],
    session: c => c.categorical.length
      ? `${c.categorical.length} categorical ${plural(c.categorical.length, 'column')} need an encoding: ${names(c.categorical.slice(0, 4))}${c.categorical.length > 4 ? `, and ${c.categorical.length - 4} more` : ''}. No encoding is declared in this session yet.`
      : 'This frame has no categorical columns, so no encoding decision is pending.',
    example: c => `session.encode(\n  strategy="one_hot",\n  columns=[${c.categorical.slice(0, 2).map(x => `"${x.name}"`).join(', ') || '"<column>"'}],\n  handle_unknown="ignore")`,
    pitfalls: () => [
      'Ordinal-encoding an unordered category and letting the model read an order into it.',
      'Fitting the encoder before the split.',
      'No policy for unseen levels at prediction time.',
    ],
  },

  'measurement-units-and-ranges': {
    stage: 1, title: 'measurement-units-and-ranges',
    prose: [
      'Every numeric column has a plausible range, and values outside it are usually recording artefacts rather than rare events: negative durations, ages above 120, percentages above one, sentinel values like -999 standing in for "unknown".',
      'Units are the quieter version of the same problem. A column that switches from minutes to seconds partway through a merge produces a distribution that no amount of scaling can rescue.',
    ],
    session: c => {
      const negs = (c.numeric || []).filter(n => n.negatives > 0);
      const zeros = (c.numeric || []).filter(n => n.zeros / Math.max(1, c.rows) > 0.3);
      const bits = [];
      if (some(negs)) bits.push(`${negs.length} numeric ${plural(negs.length, 'column')} contain negative values (${names(negs, 3)})`);
      if (some(zeros)) bits.push(`${zeros.length} are more than 30% zeros (${names(zeros, 3)})`);
      return bits.length ? `${bits.join('; ')}. Whether those are legitimate is a domain question the sheet cannot answer.` : 'No negative values or zero-dominated numeric columns were observed; ranges look plausible on their face.';
    },
    example: c => `session.assert_range(\n  "${((c.analysable || [])[0] || {}).name || '<feature>'}",\n  min=0)\n\n# sentinels are not values:\nsession.replace_sentinels(\n  [-999, -1])`,
    pitfalls: () => [
      'Treating a sentinel like -999 as a number and averaging it into the mean.',
      'Clipping out-of-range values before asking why they are out of range.',
      'Merging two sources without confirming both use the same unit.',
    ],
  },

  'text-hygiene': {
    stage: 1, title: 'text-hygiene',
    prose: [
      'String columns fragment silently. Trailing whitespace, inconsistent case and mixed types split what should be one level into several, each with a fraction of the support.',
      'The fix is boring and must happen before the level counts are trusted: strip, case-fold, normalise, then recount. Cardinality measured on unclean strings is a measurement of the mess, not of the data.',
    ],
    session: c => {
      const mixed = (c.cols || []).filter(x => x.mixedType);
      const varia = (c.cols || []).filter(x => x.caseVariants);
      const bits = [];
      if (some(mixed)) bits.push(`${mixed.length} ${plural(mixed.length, 'column')} hold mixed types (${names(mixed, 3)})`);
      if (some(varia)) bits.push(`${varia.reduce((s, x) => s + x.caseVariants, 0)} case or whitespace variants were observed across ${varia.length} ${plural(varia.length, 'column')}`);
      return bits.length ? `${bits.join('; ')}. Level counts elsewhere on the sheet are computed on the raw strings.` : 'No mixed-type or case-variant string columns were observed.';
    },
    example: () => `session.clean_strings(\n  strip=True,\n  case="lower",\n  normalize_unicode=True)\n\n# then recount levels`,
    pitfalls: () => [
      'Counting cardinality before cleaning, then designing an encoding around the wrong number.',
      'Case-folding identifiers that are legitimately case-sensitive.',
      'Cleaning training strings and forgetting the same transform at prediction time.',
    ],
  },

  /* ── Stage 2 · relationships ────────────────────────────────────── */

  'univariate-distributions': {
    stage: 2, title: 'univariate-distributions',
    prose: [
      'Before any relationship, each column has a shape: centre, spread, tails, gaps and repeated values. Two columns with identical means can behave completely differently in a model, and the summary that reveals it is quartiles rather than the mean.',
      'Read the quartiles together with min and max. A median far from the mean says skew; a q3 far below the max says a tail; a q1 equal to the min says a floor or a sentinel.',
    ],
    session: c => {
      const ns = c.numeric || [];
      if (!ns.length) return 'This frame has no numeric columns to summarise.';
      const withStats = ns.filter(n => n.hasStats && isFinite(n.max - n.min));
      if (!withStats.length) return `${ns.length} numeric ${plural(ns.length, 'column')} are present, but this profile supplied no quartiles, minima or maxima for any of them — so their shapes are unexamined, not symmetric.`;
      const widest = withStats.filter(n => n.role !== 'id').sort((a, b) => (b.max - b.min) - (a.max - a.min))[0] || withStats[0];
      return `${ns.length} numeric ${plural(ns.length, 'column')} were summarised. ${widest.name} spans the widest range, ${fmt.compact(widest.min)} to ${fmt.compact(widest.max)}, with a median of ${fmt.compact(widest.median)}.`;
    },
    example: c => `session.describe()\n\n# per-column quartiles, not\n# just the mean:\nsession.describe(\n  columns=["${((c.analysable || [])[0] || {}).name || '<feature>'}"],\n  percentiles=[.01, .25, .5, .75, .99])`,
    pitfalls: () => [
      'Reading the mean of a skewed column as typical.',
      'Missing a spike of repeated values because only the summary was inspected.',
      'Summarising after imputation and mistaking the narrowed variance for the real one.',
    ],
  },

  'skew-and-transforms': {
    stage: 2, title: 'skew-and-transforms',
    prose: [
      'Skew means the tail dominates. Under squared-error loss a handful of extreme rows can steer the whole fit, and a linear model asked to span three orders of magnitude spends its capacity on the tail.',
      'A log or Box-Cox transform makes the distribution more symmetric, at the cost of interpretability: coefficients and errors then live in transformed units, and reporting them as business quantities is a mistake.',
    ],
    session: c => {
      const sk = (c.skewed || []);
      return sk.length
        ? `${sk.length} numeric ${plural(sk.length, 'column')} have |skew| above 1: ${sk.slice(0, 4).map(x => `${x.name} (${x.skew.toFixed(2)})`).join(', ')}${sk.length > 4 ? `, and ${sk.length - 4} more` : ''}.`
        : ((c.numeric || []).some(n => n.skew != null)
          ? 'No numeric column has |skew| above 1; distributions are near enough symmetric to leave alone.'
          : `No skew statistic was supplied for any of the ${(c.numeric || []).length} numeric ${plural((c.numeric || []).length, 'column')}, so symmetry is unknown rather than confirmed.`);
    },
    example: c => `session.transform(\n  method="log1p",\n  columns=["${(c.skewed && c.skewed[0] && c.skewed[0].name) || '<column>'}"])\n\n# fitted in-fold; report error\n# in original units`,
    pitfalls: () => [
      'Log-transforming a column containing zeros or negatives without an offset.',
      'Reporting RMSE in log units as if it were the business quantity.',
      'Transforming for a tree model, which does not care about monotone rescaling.',
    ],
  },

  'correlation': {
    stage: 2, title: 'correlation',
    prose: [
      'Pearson correlation measures linear co-movement only, so a strong curved relationship can score near zero. Spearman answers the monotone question instead, and disagreement between the two is itself informative.',
      'Correlation between features is a redundancy signal; correlation with the target is a screening signal. Neither is evidence of a mechanism, and a coefficient near ±1 between two features usually means one of them is a re-expression of the other.',
    ],
    session: c => {
      const ps = c.corrPairs || [];
      if (!ps.length) return 'No pairwise correlations were recorded for this frame.';
      const top = ps[0];
      const strong = ps.filter(p => Math.abs(p.r) >= 0.8);
      return `${ps.length} feature ${plural(ps.length, 'pair')} were scored; the strongest is ${top.a} × ${top.b} at r=${top.r.toFixed(3)}${strong.length ? `, and ${strong.length} ${plural(strong.length, 'pair')} exceed |0.8|` : ''}.`;
    },
    example: () => `session.correlations(\n  method="spearman",\n  threshold=0.8)\n\n# compare with pearson: the\n# gap is the non-linearity`,
    pitfalls: () => [
      'Reading a near-zero Pearson coefficient as "no relationship".',
      'Computing correlations on imputed data and treating them as observed.',
      'Dropping one of every correlated pair mechanically, without asking which is measured more reliably.',
    ],
  },

  'mutual-information': {
    stage: 2, title: 'mutual-information',
    prose: [
      'MI measures how much knowing a feature reduces uncertainty about the target. It assumes no functional form, so it catches a relationship that rises and then falls — but it returns a single non-negative number, so it says nothing about direction.',
      'It is estimated, not computed: nearest-neighbour and binning estimators both introduce variance, and small differences between adjacent features are usually inside that noise.',
    ],
    session: c => {
      if (!some(c.mi)) return 'No mutual-information estimates are available — either no target is declared or no eligible feature survived the screen.';
      const top = c.mi[0], second = c.mi[1];
      const gap = second ? Math.abs(top.mi - second.mi) / Math.max(top.mi, 1e-12) : 1;
      return `${top.name} leads at ${fmt.dec(top.mi, 6)}${second ? ` with ${second.name} at ${fmt.dec(second.mi, 6)}` : ''}.${second && gap < 0.15 ? ' The gap is small enough that their order should not be treated as settled.' : ''} The weakest scored feature, ${c.mi[c.mi.length - 1].name}, sits at ${fmt.dec(c.mi[c.mi.length - 1].mi, 6)}.`;
    },
    example: c => {
      const rows = (c.mi || []).slice(0, 4).map(m => `#   ${m.name.padEnd(16).slice(0, 16)}${fmt.dec(m.mi, 6)}`);
      return `session.explain(\n  "features", moment="before")\n\n# screening aid, not ranking:\n${rows.join('\n') || '#   nothing scored'}`;
    },
    pitfalls: () => [
      'Reading rank order as importance. MI scores each feature alone; a weak feature can be decisive in combination.',
      'Reading it as causal, or as a direction of effect.',
      'Selecting features by MI on the full frame — the ranking is then fitted on test rows.',
    ],
  },

  'variance-inflation': {
    stage: 2, title: 'variance-inflation',
    prose: [
      'VIF asks how well the other numeric features predict this one. When they predict it well, the model cannot tell which of them is responsible for an effect, and each coefficient\u2019s variance inflates accordingly. Prediction quality may be untouched; interpretation is what degrades.',
      'Because every value is computed against the rest of the set, VIF is a property of the set and not of the column. Remove one member and every remaining number changes.',
    ],
    session: c => {
      if (!some(c.vif)) return 'No VIF estimates are available: this frame has no eligible numeric feature set to compute them against.';
      const over = c.vif.filter(v => v.vif >= c.vifThreshold);
      return over.length
        ? `${over.slice(0, 4).map(v => `${v.name} at ${v.vif.toFixed(3)}`).join(', ')}${over.length > 4 ? ` and ${over.length - 4} more` : ''} ${over.length === 1 ? 'sits' : 'sit'} above the ${c.vifThreshold.toFixed(1)} threshold, computed on ${fmt.n(c.completeRows)} complete cases.`
        : `Every numeric feature sits below the ${c.vifThreshold.toFixed(1)} threshold — the highest is ${c.vif[0].name} at ${c.vif[0].vif.toFixed(3)}.`;
    },
    example: c => {
      const over = (c.vif || []).filter(v => v.vif >= c.vifThreshold);
      const drop = over[1] || (c.vif || [])[1];
      return `session.explain(\n  "reduce_dimensions")\n\n${drop ? `session.set_roles(\n  {"${drop.name}": "ignore"})\n# recompute; the rest fall` : '# nothing above threshold'}`;
    },
    pitfalls: () => [
      'Dropping every feature above the threshold at once, when the values were computed with all of them present.',
      'Treating VIF as a prediction problem. Trees and regularised models tolerate collinearity; interpretation does not.',
      'Comparing complete-case VIF with post-imputation VIF as if they measured the same frame.',
    ],
  },

  'interaction-effects': {
    stage: 2, title: 'interaction-effects',
    prose: [
      'Univariate screens rank features one at a time, so they are blind to a pair that only matters jointly. A feature with near-zero MI can be decisive in combination with another, which is why screening should never be the same step as selection.',
      'Trees and gradient boosting find interactions themselves; linear models need them written down. Either way the decision is a modeling choice, and the EDA\u2019s job is to say which pairs are worth trying.',
    ],
    session: c => some(c.mi)
      ? `The screens on this sheet are univariate: ${c.mi.length} ${plural(c.mi.length, 'feature')} were scored one at a time. Nothing here rules out a pair — ${c.mi[0].name} × ${(c.mi[1] && c.mi[1].name) || 'another feature'} is untested as a combination.`
      : 'No univariate screen ran, so there is nothing to caveat about interactions yet.',
    example: c => `session.explain(\n  "interactions", moment="before")\n\n# candidate pair, in-fold:\nsession.add_interaction(\n  "${(c.mi[0] && c.mi[0].name) || '<a>'}",\n  "${(c.mi[1] && c.mi[1].name) || '<b>'}")`,
    pitfalls: () => [
      'Dropping a low-MI feature before any model has seen it in combination.',
      'Adding every pairwise interaction and multiplying the width of the matrix by the square of your patience.',
      'Reading a tree\u2019s importance for an interaction as evidence about either feature alone.',
    ],
  },

  'dimensionality-reduction': {
    stage: 2, title: 'dimensionality-reduction',
    prose: [
      'Reduction trades interpretability for conditioning. PCA produces uncorrelated components that no longer correspond to anything anyone measured; selection keeps the original columns and simply discards some.',
      'Both are fitted steps. Components learned on the full frame carry test-set structure into training, and the choice of how many to keep is itself a hyper-parameter that needs its own validation.',
    ],
    session: c => {
      const over = (c.vif || []).filter(v => v.vif >= c.vifThreshold);
      return `${c.eligible} eligible ${plural(c.eligible, 'feature')} for ${fmt.n(c.rows)} rows — a ratio of about ${Math.round(c.rows / Math.max(1, c.eligible))} rows per feature.${over.length ? ` ${over.length} of them are collinear enough to be candidates for reduction.` : ''}`;
    },
    example: () => `session.explain(\n  "reduce_dimensions")\n\nsession.pca(\n  n_components=0.95,\n  fit_on="train")`,
    pitfalls: () => [
      'Fitting PCA on the full frame before splitting.',
      'Reducing first and then reporting feature importances as if they named real columns.',
      'Reaching for reduction when the real problem is a handful of duplicated columns.',
    ],
  },

  'feature-scaling': {
    stage: 2, title: 'feature-scaling',
    prose: [
      'Scaling changes the units a model sees. Distance-based methods, regularised linear models and neural networks all need comparable ranges; trees do not care at all.',
      'The scaler is fitted, so it belongs in the fold. Standardising with a mean and standard deviation computed over the full frame is the same leak as a full-frame median, only harder to spot.',
    ],
    session: c => {
      const ns = c.numeric || [];
      if (!ns.length) return 'No numeric columns, so no scaling decision arises.';
      const spans = ns.map(n => Math.abs(n.max - n.min)).filter(x => isFinite(x) && x > 0);
      if (!spans.length) return `${ns.length} numeric ${plural(ns.length, 'column')} are present but this profile supplied no ranges for them, so their relative scales are unknown. Scaling matters for distance-based and regularised models and not at all for trees.`;
      const ratio = Math.max(...spans) / Math.min(...spans);
      return `Numeric ranges differ by a factor of about ${fmt.compact(ratio)} across ${ns.length} ${plural(ns.length, 'column')}. That matters for distance-based and regularised models and not at all for trees.`;
    },
    example: () => `session.scale(\n  method="standard",\n  fit_on="train")\n\n# robust scaling if the tails\n# are doing the talking`,
    pitfalls: () => [
      'Fitting the scaler before the split.',
      'Standardising a heavily skewed column and expecting symmetry.',
      'Scaling one-hot columns along with continuous ones without deciding whether that is what you meant.',
    ],
  },

  /* ── Stage 3 · validation ───────────────────────────────────────── */

  'data-splitting': {
    stage: 3, title: 'data-splitting',
    prose: [
      'A split assigns each row a membership that controls what it may do: train a model, guide a choice, or assess the result. The value of a test set comes entirely from having been untouched, which is a discipline rather than a property of the data.',
      'Descriptive EDA over the full frame — this sheet — describes observed rows. It is not train-fitted transform evidence, and nothing in it should choose a transform without being recomputed inside the training fold.',
    ],
    session: c => `${fmt.n(c.rows)} of ${fmt.n(c.rowsTotal)} rows were examined across ${c.colCount} columns${c.sampled ? ', so these are sampled observations and may not reproduce full-data tail behaviour' : ', with no sampling'}. Every number here is a full-frame observation, not a training-fold estimate.`,
    example: c => `session.explain(\n  "split", moment="before")\n\nsession.split(\n  test_size=0.2${c.target && c.target.task !== 'regression' ? `,\n  stratify="${c.target.name}"` : ''})`,
    pitfalls: c => [
      'Reusing the test set to choose between candidates — that makes it a validation set.',
      'Random splitting when rows share a group or a time order.',
      c.rows < 300 ? `Splitting ${fmt.n(c.rows)} rows at all: a 20% test set is ${fmt.n(Math.round(c.rows * 0.2))} rows, and cross-validation is the safer instrument.` : 'Reading full-frame EDA as evidence about a fitted pipeline.',
    ],
  },

  'stratification': {
    stage: 3, title: 'stratification',
    prose: [
      'Stratifying holds a distribution constant across split memberships. For a rare class it is the difference between a test set that contains the positives and one that happens not to.',
      'It is not free of judgement: you choose what to stratify on. Stratifying the target is standard; stratifying a rare category as well can be necessary and can also make the split infeasible when the levels are too thin.',
    ],
    session: c => {
      const t = c.target;
      if (!t) return 'No target is declared, so there is nothing to stratify on yet.';
      if (t.task === 'regression') return `${t.name} is continuous, so stratification would need binning — quartiles of the target are the usual choice, especially with skew ${t.stats ? t.stats.skew.toFixed(2) : 'unknown'}.`;
      const small = t.classes ? t.classes.slice().sort((a, b) => a.count - b.count)[0] : null;
      return small ? `The smallest class of ${t.name} holds ${fmt.n(small.count)} rows; a 20% unstratified test set would contain about ${fmt.n(Math.round(small.count * 0.2))} of them, and could contain far fewer.` : `${t.name} is ${t.task}.`;
    },
    example: c => `session.split(\n  test_size=0.2,\n  stratify="${(c.target && c.target.name) || '<target>'}")`,
    pitfalls: () => [
      'Forgetting to stratify with an imbalanced target and blaming the variance on the model.',
      'Stratifying on so many columns that some strata hold one row.',
      'Stratifying a continuous target without binning it.',
    ],
  },

  'cross-validation': {
    stage: 3, title: 'cross-validation',
    prose: [
      'Cross-validation reuses the data by rotating which part is held out, giving a spread of scores instead of one number. The spread is the point: a single hold-out estimate has no error bar.',
      'Every fitted step must move inside the loop. A pipeline that imputes, encodes and scales before cross-validating reports a score that no future data will reproduce.',
    ],
    session: c => `At ${fmt.n(c.rows)} ${plural(c.rows, 'row')} and ${c.eligible} eligible ${plural(c.eligible, 'feature')}, ${c.rows < 2000 ? 'k-fold cross-validation is a better instrument than a single hold-out' : 'a single hold-out is defensible, and k-fold still buys you an error bar'}.`,
    example: c => `session.cross_validate(\n  folds=5${c.target && c.target.task !== 'regression' ? ',\n  stratified=True' : ''},\n  pipeline="full")\n# every fitted step in-fold`,
    pitfalls: () => [
      'Preprocessing outside the loop — the most common way to invent a good score.',
      'Reporting the mean fold score without the spread.',
      'Using k-fold when rows are grouped or ordered in time.',
    ],
  },

  'dataset-drift': {
    stage: 3, title: 'dataset-drift',
    prose: [
      'Drift is a measured distribution change between two defined populations — here, the train and test memberships. It is a statement about the comparison, so the first thing to question is how the comparison was built, not the data.',
      'Three ordinary causes precede anything alarming: a split drawn without stratification, a time order the split ignored, and a group of related rows landing unevenly. Only after those are ruled out does drift mean the populations genuinely differ.',
    ],
    session: c => some(c.drifted)
      ? `${c.drifted.length} eligible ${plural(c.drifted.length, 'column')} met the configured thresholds: ${list(c.drifted, 6)}. This is the blocking finding on the sheet.`
      : 'No column met the configured drift thresholds. The thresholds are configured rather than derived, so this is an absence of flags rather than proof of stability.',
    example: c => `session.explain(\n  "split", moment="after")\n\n${some(c.drifted) ? `# inspect before touching data:\n${c.drifted.slice(0, 3).map(n => `#   ${n}`).join('\n')}\n# re-split if the split is the cause` : '# no flags on this split'}`,
    pitfalls: () => [
      'Correcting the data before checking the split — the usual cause is the split.',
      'Re-splitting repeatedly until the flags disappear, which fits the split to the test set.',
      'Treating a flag on a small test set as evidence of a real shift.',
    ],
  },

  'leakage': {
    stage: 3, title: 'leakage',
    prose: [
      'Leakage is any path by which information unavailable at prediction time reaches the model. It shows up as a validation score that is too good, and it survives every honest split because the problem is the column, not the partition.',
      'The three usual sources: identifiers that index the answer, columns recorded after the outcome, and any statistic fitted before the split. The first two are schema questions; the third is a pipeline question.',
    ],
    session: c => {
      const lk = c.leakage || [];
      if (lk.length) return `${lk.length} leakage ${plural(lk.length, 'suspect')} were flagged by heuristic: ${lk.map(x => `${x.name} (${x.reason})`).slice(0, 3).join(', ')}${lk.length > 3 ? `, and ${lk.length - 3} more` : ''}. A heuristic screen is not a proof of absence.`;
      return `No leakage suspect was flagged${some(c.idLike) ? `, though ${list(c.idLike)} ${c.idLike.length === 1 ? 'is' : 'are'} identifier-like and must stay out of the matrix` : ''}. Timing of each column relative to the outcome was not verified.`;
    },
    example: c => `session.explain(\n  "leakage", moment="before")\n\nsession.set_roles({\n  "${(c.leakage && c.leakage[0] && c.leakage[0].name) || (c.idLike && c.idLike[0]) || '<column>'}": "ignore"})`,
    pitfalls: () => [
      'Trusting a suspiciously high score instead of hunting for its cause.',
      'Keeping a column because it is predictive, without asking when it becomes known.',
      'Fitting any statistic — median, encoder, scaler, selector — before the split.',
    ],
  },

  'temporal-structure': {
    stage: 3, title: 'temporal-structure',
    prose: [
      'If rows carry a time, the split must respect it: training on the future to predict the past inflates every score and cannot be reproduced in deployment. Random splitting destroys the ordering silently.',
      'Time also brings coverage questions. A gap in the record, an incomplete final period, or a seasonal cycle shorter than the training window all change what the model can be said to have learned.',
    ],
    session: c => {
      const t = c.timeCol;
      if (!t) return 'No time column is declared in this frame, so no temporal check was possible. That is not the same as the rows being order-free.';
      return `${t.name} spans ${t.min} to ${t.max}${t.gaps ? ` with ${fmt.n(t.gaps)} observed ${plural(t.gaps, 'gap')} in coverage` : ' with no observed gaps'}. Any split on this frame should be drawn forward in time.`;
    },
    example: c => `session.split(\n  strategy="time_ordered",\n  time_column="${(c.timeCol && c.timeCol.name) || '<timestamp>'}",\n  test_size=0.2)`,
    pitfalls: () => [
      'Random splitting time-ordered rows.',
      'Engineering a feature from a window that extends past the prediction moment.',
      'Including a partial final period and reading its lower counts as a trend.',
    ],
  },

  'group-structure': {
    stage: 3, title: 'group-structure',
    prose: [
      'When several rows belong to the same entity — customer, flight, patient, device — they are not independent. Splitting at row level puts the same entity on both sides, and the model recognises it rather than generalising.',
      'The repair is to split by group, which usually costs accuracy on paper and buys a number you can trust.',
    ],
    session: c => {
      const g = c.groupCol;
      if (g) return `${g.name} identifies ${fmt.n(g.groups)} ${plural(g.groups, 'group')} across ${fmt.n(c.rows)} rows — about ${(c.rows / Math.max(1, g.groups)).toFixed(1)} rows per group, so a row-level split would straddle groups.`;
      return `No group column is declared. ${some(c.idLike) ? `${list(c.idLike)} ${c.idLike.length === 1 ? 'is' : 'are'} near-unique, so ${c.idLike.length === 1 ? 'it does' : 'they do'} not indicate repeated entities in this extract.` : 'Row independence was assumed rather than verified.'}`;
    },
    example: c => `session.split(\n  strategy="group",\n  group_column="${(c.groupCol && c.groupCol.name) || '<entity_id>'}",\n  test_size=0.2)`,
    pitfalls: () => [
      'Assuming one row per entity because the identifier looks unique in this extract.',
      'Group-splitting on the wrong level — household versus customer, aircraft versus flight.',
      'Cross-validating without carrying the group constraint into the folds.',
    ],
  },

  'diagnostic-uncertainty': {
    stage: 3, title: 'diagnostic-uncertainty',
    prose: [
      'Every number on the readiness sheet is an estimate from one sample: metrics, curves, importances and anomaly labels alike. Reported precision is a formatting choice, not a claim about stability.',
      'Anomaly detectors are the sharpest case. An Isolation Forest returns a score, a configured contamination turns that score into a label, and the label marks rows that are unusual in the feature space — not rows that are wrong.',
    ],
    session: c => c.anomalies
      ? `${fmt.n(c.anomalies.flagged)} of ${fmt.n(c.anomalies.scored)} scored rows were marked, a ${fmt.pct(c.anomalies.flagged / c.anomalies.scored)} rate against a ${fmt.pct(c.anomalies.contamination, 0)} configured contamination. Scoring ran on complete cases, so ${fmt.n(c.rows - c.anomalies.scored)} rows were never eligible.`
      : 'No multivariate outlier screen ran on this frame, so no row carries an anomaly label either way.',
    example: c => c.anomalies
      ? `session.explain(\n  "outliers", moment="after")\n\nsession.rows(anomaly=True)\n# ${fmt.n(c.anomalies.flagged)} rows to inspect`
      : 'session.explain(\n  "outliers", moment="before")',
    pitfalls: c => [
      c.anomalies ? 'Deleting flagged rows as errors — the observed rate matches the configured contamination, so the detector was told how many to find.' : 'Assuming no screen means no outliers.',
      'Reading five decimal places as five significant figures.',
      c.anomalies && c.rows > c.anomalies.scored ? `Forgetting that complete-case scoring excluded ${fmt.n(c.rows - c.anomalies.scored)} rows.` : 'Forgetting that the screen sees only the columns it was given.',
    ],
  },

  'outlier-screens': {
    stage: 3, title: 'outlier-screens',
    prose: [
      'Univariate screens — IQR fences, z-scores — flag values far from the centre of one column. Multivariate screens flag rows that are unusual as combinations, which is a different question and often a different set of rows.',
      'Neither answers whether the row is wrong. An outlier is a candidate for investigation: a recording error, a rare true event, or evidence that the column means two different things.',
    ],
    session: c => {
      const os = (c.numeric || []).filter(n => n.outlierRate > 0).sort((a, b) => b.outlierRate - a.outlierRate);
      if (!os.length) return (c.numeric || []).some(n => n.outlierRate != null && n.hasStats)
        ? 'No numeric column produced IQR-fence outliers in this extract.'
        : `No outlier rates were supplied for the ${(c.numeric || []).length} numeric ${plural((c.numeric || []).length, 'column')}, so no screen has run — this is silence, not a clean result.`;
      return `${os.length} numeric ${plural(os.length, 'column')} have values beyond the IQR fences, led by ${os[0].name} at ${fmt.pct(os[0].outlierRate, 2)}${c.anomalies ? `. The multivariate screen marked ${fmt.n(c.anomalies.flagged)} rows, which need not be the same rows.` : '.'}`;
    },
    example: c => `session.outliers(\n  method="iqr",\n  columns=["${((c.analysable || [])[0] || {}).name || '<column>'}"],\n  factor=1.5)\n\n# inspect, then decide`,
    pitfalls: () => [
      'Winsorising by default and erasing the rare events you were hired to predict.',
      'Applying a z-score fence to a skewed column, where it flags the whole tail.',
      'Removing outliers before splitting, so the test set no longer resembles reality.',
    ],
  },

  /* ── Stage 4 · evaluation ───────────────────────────────────────── */

  'class-imbalance': {
    stage: 4, title: 'class-imbalance',
    prose: [
      'When classes are unequal, accuracy stops being informative: it can be maximised by predicting the majority every time. The default 0.5 threshold inherits the same problem — it assumes the two errors cost the same and the base rate is even.',
      'Threshold choice is a business decision expressed as a number. Precision, recall and their balance describe a chosen operating point; ranking metrics describe separability at any point.',
    ],
    session: c => {
      const t = c.target;
      if (!t || t.task === 'regression' || !t.classes) return 'This frame has no classification target, so class balance does not apply to it.';
      const sorted = [...t.classes].sort((a, b) => b.count - a.count);
      return `${t.name} is ${t.task} with ${t.classes.map(k => `${fmt.n(k.count)} ${k.label}`).join(' and ')}. Always predicting "${sorted[0].label}" scores ${fmt.pct(sorted[0].count / c.rows, 1)} accuracy and catches none of the rest.`;
    },
    example: c => {
      const t = c.target;
      const rate = t && t.classes && t.classes.length === 2 ? t.classes[1].count / c.rows : 0.5;
      return `session.evaluate(\n  metrics=["average_precision",\n           "recall",\n           "roc_auc"],\n  threshold=${Math.max(0.05, Math.min(0.5, rate)).toFixed(2)})`;
    },
    pitfalls: c => [
      c.target && c.target.classes && c.target.classes.length === 2 ? `Reporting accuracy against a ${fmt.pct(c.target.classes[1].count / c.rows, 1)} base rate.` : 'Reporting accuracy without stating the base rate.',
      'Resampling to balance the classes and then reading predicted probabilities as calibrated.',
      'Choosing a threshold on the test set.',
    ],
  },

  'target-distribution': {
    stage: 4, title: 'target-distribution',
    prose: [
      'A continuous target has no classes to balance, so the questions change: what is its spread, how skewed is it, and does the error you care about scale with its magnitude? A single mean-squared error hides all three.',
      'Skew matters most. A long right tail means a few rows dominate the loss, and a model can look competent on the bulk while being useless where the cost is.',
    ],
    session: c => {
      const t = c.target;
      if (!t || t.task !== 'regression' || !t.stats) return 'This frame has no continuous target, so its distribution does not apply.';
      const s = t.stats;
      return `${t.name} runs from ${fmt.compact(s.min)} to ${fmt.compact(s.max)} with a median of ${fmt.compact(s.median)} and skew ${s.skew.toFixed(2)}. ${Math.abs(s.skew) > 1 ? 'That is a long tail: a handful of rows will dominate any squared-error loss.' : 'The distribution is close enough to symmetric that squared error is not obviously distorted.'}`;
    },
    example: () => `session.evaluate(\n  metrics=["mae",\n           "rmse",\n           "r2"])\n# MAE reads in target units;\n# RMSE punishes the tail`,
    pitfalls: () => [
      'Reporting R² alone — it says nothing about whether the error size is acceptable.',
      'Log-transforming the target and reporting error in log units as a business quantity.',
      'Averaging error across a skewed target where the expensive rows are all in the tail.',
    ],
  },

  'metric-selection': {
    stage: 4, title: 'metric-selection',
    prose: [
      'A metric is a statement about which mistakes matter. Choosing one before modeling forces the question into the open; choosing one afterwards invites picking whichever number looks best.',
      'No single metric is sufficient. Report one headline figure tied to the decision the model supports, plus the diagnostics that show where the error concentrates.',
    ],
    session: c => {
      const t = c.target;
      if (!t) return 'No target is declared, so no metric can be selected yet.';
      if (t.task === 'regression') return `${t.name} is continuous with skew ${t.stats ? t.stats.skew.toFixed(2) : 'unknown'} — MAE reads in target units, RMSE weights the tail, and the choice between them is a statement about which errors you are willing to accept.`;
      const bal = t.classes ? Math.min(...t.classes.map(k => k.count)) / c.rows : 0.5;
      return `${t.name} is ${t.task} with a minority share of ${fmt.pct(bal, 1)} — average precision and recall at a stated threshold describe this better than accuracy.`;
    },
    example: () => `session.explain(\n  "evaluate", moment="before")\n\n# fix the metric before the\n# first fit, and write down why`,
    pitfalls: () => [
      'Optimising one metric and reporting another.',
      'Comparing metrics computed on different row subsets.',
      'Letting a leaderboard metric stand in for the decision the model actually supports.',
    ],
  },

  'thresholds-and-costs': {
    stage: 4, title: 'thresholds-and-costs',
    prose: [
      'A classifier outputs a score; a decision needs a cut. The cut encodes the relative cost of a false positive and a false negative, which is a business fact, not a statistical one.',
      'Because the cut is a choice, it must be chosen on validation data and then held fixed. Tuning it on the test set converts your only unbiased estimate into another fitted parameter.',
    ],
    session: c => {
      const t = c.target;
      if (!t || t.task === 'regression' || !t.classes || t.classes.length !== 2) return 'No binary target, so no single threshold decision arises here.';
      const pos = t.classes[1];
      return `With a ${fmt.pct(pos.count / c.rows, 1)} positive rate, the default 0.5 cut will predict the positive class rarely. Moving it towards the base rate trades precision for recall — the exchange rate is yours to set.`;
    },
    example: c => `session.threshold_sweep(\n  metric="f_beta",\n  beta=2,\n  on="validation")\n\n# then freeze the chosen cut`,
    pitfalls: () => [
      'Leaving the threshold at 0.5 because it is the default.',
      'Tuning the threshold on the test set.',
      'Reporting precision and recall without saying at which cut.',
    ],
  },

  'baselines': {
    stage: 4, title: 'baselines',
    prose: [
      'A score means nothing without something to beat. The majority-class predictor, the median predictor, and last period\u2019s value are all free, and any one of them is a stiffer competitor than teams expect.',
      'Compute the baseline before the model, not after. Afterwards it becomes a thing you justify around rather than a bar you clear.',
    ],
    session: c => {
      const t = c.target;
      if (!t) return 'No target is declared, so no baseline can be computed.';
      if (t.task === 'regression' && t.stats) return `Predicting the median of ${t.name} (${fmt.compact(t.stats.median)}) for every row is the baseline any model must beat, and with skew ${t.stats.skew.toFixed(2)} it is a stronger one than it looks.`;
      const major = t.classes ? [...t.classes].sort((a, b) => b.count - a.count)[0] : null;
      return major ? `Predicting "${major.label}" for every row scores ${fmt.pct(major.count / c.rows, 1)} accuracy. That is the number to beat, and it catches nothing.` : `${t.name} is ${t.task}.`;
    },
    example: () => `session.baseline(\n  strategy="most_frequent")\n\n# record it next to every\n# later score`,
    pitfalls: () => [
      'Reporting a model score with no baseline beside it.',
      'Using a baseline computed on different rows than the model.',
      'Forgetting the domain baseline — the rule the business already uses.',
    ],
  },

  'calibration': {
    stage: 4, title: 'calibration',
    prose: [
      'A calibrated probability means what it says: among rows scored 0.3, about 30% are positive. Ranking metrics can be excellent while calibration is poor, which is fine for triage and wrong for anything that multiplies the probability by a cost.',
      'Resampling and class weighting both distort calibration by changing the base rate the model assumes. If probabilities are the product, calibrate explicitly and check the curve.',
    ],
    session: c => {
      const t = c.target;
      if (!t || t.task === 'regression' || !t.classes) return 'No probabilistic classification target here, so calibration does not apply.';
      return `If ${t.name} probabilities will be used as probabilities — expected value, prioritisation, pricing — they need calibrating against the observed ${t.classes.length === 2 ? fmt.pct(t.classes[1].count / c.rows, 1) : ''} base rate, not just ranking well.`;
    },
    example: () => `session.calibrate(\n  method="isotonic",\n  on="validation")\n\nsession.reliability_curve()`,
    pitfalls: () => [
      'Reading raw scores from a resampled model as probabilities.',
      'Calibrating on the training fold.',
      'Assuming a high AUC implies calibrated output.',
    ],
  },
};

export const STAGES = [
  { n: '01', key: 1, label: 'Data quality', blurb: 'what the frame is before anything is fitted' },
  { n: '02', key: 2, label: 'Relationships', blurb: 'how columns relate to the target and to each other' },
  { n: '03', key: 3, label: 'Validation', blurb: 'what the evidence is allowed to certify' },
  { n: '04', key: 4, label: 'Evaluation', blurb: 'what a score is worth' },
];
