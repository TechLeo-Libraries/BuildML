// Additions to stage 01 (data quality) and stage 02 (relationships).

import { fmt, plural, list, names } from './eda-format.js';

export const CONCEPTS_ADD_A = {

  /* ── Stage 1 additions ──────────────────────────────────────────── */

  'join-integrity': {
    stage: 1, title: 'join-integrity',
    prose: [
      'Most frames are the output of joins, and a join is the single most reliable way to change a dataset without noticing. The arithmetic is the check: record the row count before and the row count you predict, then compare with what you got. One-to-one keeps the count; one-to-many multiplies it; many-to-many multiplies it by a number nobody predicted.',
      'Two silent failures dominate. Fan-out inflates rows, which weights the duplicated entities more heavily in every subsequent mean, share and fit. A dropped inner join removes rows whose key had no match, and those non-matching rows are almost never random — they are the new accounts, the foreign records, the ones with the awkward encoding.',
      'The diagnostics are cheap: match rate per side, count of unmatched keys, and the distribution of matches per key. A left join with a 92% match rate is a finding, not a detail, because the 8% left an entire population out of the frame with no error raised.',
    ],
    read: c => [
      `Compare the row count you expected with the ${fmt.n(c.rows)} you have, and treat any unexplained difference as a bug.`,
      'Read the match rate per side and inspect ten unmatched keys; the reason is usually a type or formatting mismatch.',
      'Read matches per key: a maximum above one on a supposedly one-to-one join is fan-out.',
    ],
    decide: () => 'Assert the expected cardinality of every join, verify the match rate, and account for every row the join added or removed.',
    session: c => `${fmt.n(c.rows)} rows are present at ${c.colCount} columns${c.duplicates && c.duplicates.rows ? `, including ${fmt.n(c.duplicates.rows)} exact duplicate ${plural(c.duplicates.rows, 'row')} — a common fan-out signature` : ''}. No join history is recorded in this frame: whether these rows are the join\u2019s intended output cannot be verified from the result alone.`,
    example: c => `session.assert_join(\n  left_key="${(c.idLike && c.idLike[0]) || '<key>'}",\n  cardinality="1:1",\n  min_match_rate=0.99)`,
    pitfalls: () => [
      'Using an inner join where a left join was meant, so unmatched rows vanish silently.',
      'Joining on keys of different dtypes and getting a zero match rate that looks like empty data.',
      'Accepting a row-count change because the numbers "look about right".',
    ],
  },

  'cross-field-consistency': {
    stage: 1, title: 'cross-field-consistency',
    prose: [
      'Single-column checks cannot see contradictions that live between columns: an end date before a start date, a total that is not the sum of its parts, a child count above a household size, a status of "closed" with an open balance. Each is individually plausible and jointly impossible, which is why they survive profiling.',
      'These constraints come from the domain and have to be written down as assertions — one line each, run on every extract. They are the cheapest quality instrument there is, they fail loudly, and each violation localises the problem to a specific pair of columns and a specific set of rows.',
      'Violations are also informative about lineage. A total that disagrees with its components usually means the two were computed at different times or from different sources, which tells you something no single-column statistic could: that the frame is a merge of two views of the world.',
    ],
    read: c => [
      `Write one assertion per known relationship between the ${c.colCount} columns, then count violating rows for each.`,
      'For each violation, inspect whether it clusters by period or by source — clustering means an ingestion change rather than data entry.',
      'Check derived totals against their components before trusting either.',
    ],
    decide: () => 'Encode domain constraints as assertions that run on every extract, and resolve violations at the source rather than by clipping.',
    session: c => `${c.colCount} columns were profiled independently; no cross-field constraint was declared or tested in this session. ${(c.numeric || []).length} numeric ${plural((c.numeric || []).length, 'column')}${c.timeCol ? ' and one time column' : ''} are present, so relationships of the total-equals-parts and end-after-start kind are possible and untested.`,
    example: c => `session.assert_rows(\n  "end_date >= start_date")\nsession.assert_rows(\n  "${((c.analysable || [])[0] || {}).name || '<total>'} >= 0")`,
    pitfalls: () => [
      'Profiling columns one at a time and concluding the frame is consistent.',
      'Fixing a contradiction by overwriting one side without knowing which side is wrong.',
      'Writing the assertions once and not running them on the next extract.',
    ],
  },

  'datetime-parsing': {
    stage: 1, title: 'datetime-parsing',
    prose: [
      'Dates fail in ways numbers do not. Ambiguous formats resolve differently by locale, so 03/04 is two different days and only the rows above the twelfth reveal which; strings sort lexically, so "2025-1-9" lands before "2025-10-1"; epoch integers are seconds in one system and milliseconds in another, and misreading them lands the whole column in 1970.',
      'Timezones are the second layer. A naive timestamp with no zone cannot be compared with one that has it; a UTC column and a local column differ by a shifting offset; and daylight-saving transitions duplicate one hour and delete another every year, which puts a real spike and a real hole in any hourly aggregate.',
      'The consequence is that every temporal check downstream — ordering, splits, gaps, lags, seasonality — is only as good as the parse. Confirm the parse first by reading the resulting minimum and maximum against reality, then look for the impossible: future timestamps, epoch zero, and midnight-only values that mean the time component was never recorded.',
    ],
    read: c => [
      c.timeCol ? `Read the parsed span (${c.timeCol.min} to ${c.timeCol.max}) against what the business says the window should be.` : 'Identify every column that could be a date, including integers that look like epochs and strings that look like ISO dates.',
      'Check the dtype: an object dtype on a date column means every comparison is lexical.',
      'Count values at exactly midnight and exactly epoch zero; both indicate a lost or defaulted component.',
    ],
    decide: () => 'Parse every date explicitly with a stated format and timezone, verify the span, and store as datetime rather than string.',
    session: c => c.timeCol
      ? `${c.timeCol.name} spans ${c.timeCol.min} to ${c.timeCol.max}${c.timeCol.gaps ? ` with ${fmt.n(c.timeCol.gaps)} coverage ${plural(c.timeCol.gaps, 'gap')}` : ' with no observed gaps'}. Timezone and parse format are not recorded, so the ordering it supplies is trusted rather than verified.`
      : `No column is typed as a date in this frame. ${(c.cols || []).filter(x => x.dtype === 'string').length} string ${plural((c.cols || []).filter(x => x.dtype === 'string').length, 'column')} could contain unparsed timestamps; nothing here proves the rows are order-free.`,
    example: c => `session.cast({\n  "${(c.timeCol && c.timeCol.name) || '<date>'}": "datetime"},\n  format="%Y-%m-%d",\n  tz="UTC")`,
    pitfalls: () => [
      'Letting the parser infer a day-first or month-first format from a sample that cannot distinguish them.',
      'Mixing naive and zone-aware timestamps in one comparison.',
      'Reading epoch milliseconds as seconds, which puts every row near 1970.',
    ],
  },

  'precision-and-heaping': {
    stage: 1, title: 'precision-and-heaping',
    prose: [
      'Recorded precision is a property of the instrument, not the quantity. Ages given in whole years, prices rounded to the nearest pound, sensor readings quantised to a grid — each imposes a granularity that shows up as a small number of distinct values and a histogram made of spikes rather than a curve.',
      'Heaping is the human version: self-reported numbers cluster on multiples of five and ten, so a distribution of stated ages or stated incomes has visible teeth. That pattern is evidence about how the data was collected, and it caps how finely any model can discriminate — a threshold at 27.5 is meaningless when every value is a multiple of 5.',
      'Truncation and censoring are the boundary cases. Truncation means values beyond a limit are absent from the frame entirely; censoring means they are present but recorded as the limit, which puts a pile-up at the edge. A mass exactly at a round maximum is almost always a cap rather than a coincidence.',
    ],
    read: c => [
      `Read distinct count against row count per numeric column: a low ratio means coarse recording rather than a small sample (${(c.numeric || []).length} numeric column(s) here).`,
      'Look at the last digit distribution. An excess of 0s and 5s is heaping and tells you the values were reported, not measured.',
      'Check for a mass at exactly the minimum or maximum, which indicates a cap or a floor rather than a tail.',
    ],
    decide: () => 'Record the true precision of each numeric column, treat edge pile-ups as censoring, and do not model distinctions finer than the instrument.',
    session: c => {
      const ns = c.numeric || [];
      const coarse = ns.filter(n => n.distinct && n.distinct < Math.min(30, c.rows / 20));
      return coarse.length
        ? `${coarse.length} numeric ${plural(coarse.length, 'column')} hold fewer than 30 distinct values over ${fmt.n(c.rows)} rows (${names(coarse, 3)}), so they are coarsely recorded whatever their dtype suggests.`
        : `No numeric column looks unusually coarse across ${fmt.n(c.rows)} rows. Digit preference and edge pile-ups were not tested, so heaping and censoring remain unexamined.`;
    },
    example: c => `session.describe(\n  columns=["${((c.analysable || [])[0] || {}).name || '<feature>'}"],\n  percentiles=[.01, .5, .99])\n\n# then read distinct counts and\n# the last-digit histogram`,
    pitfalls: () => [
      'Reporting a median to two decimals on a column recorded in whole units.',
      'Reading a pile-up at the maximum as a heavy tail rather than a cap.',
      'Treating heaped self-reported values as precise measurements in a distance-based model.',
    ],
  },

  'nested-and-multivalued': {
    stage: 1, title: 'nested-and-multivalued',
    prose: [
      'Not every cell holds one value. Delimited lists ("wifi;parking;pool"), JSON blobs, arrays and free text all pack a structure into a single column, and profiling treats each distinct combination as its own level — which is why such a column reports thousands of levels and no useful cardinality.',
      'The choice is how to flatten. Explode to multiple rows and the grain changes, so every rate and the split strategy change with it. Pivot to indicator columns and the width grows by the number of distinct elements while the grain is preserved. Summarise to counts and lengths and you keep one column at the cost of the detail.',
      'Free text is the extreme case and is a separate discipline: length, language, and token statistics first, representation second. What matters at the EDA stage is recognising that the column is not categorical, so no encoder should be pointed at it as though it were.',
    ],
    read: c => [
      `Inspect the widest string columns by hand — ${(c.categorical || []).length} categorical and ${(c.cols || []).filter(x => x.dtype === 'string').length} string column(s) here — looking for separators, braces and sentence-length values.`,
      'Count elements per cell: the distribution of that count tells you whether exploding multiplies rows by two or by twenty.',
      'Decide the grain consequence before flattening, since exploding changes what one row means.',
    ],
    decide: () => 'Flatten multi-valued columns deliberately — explode, pivot to indicators, or summarise — and state the grain consequence of the choice.',
    session: c => {
      const wide = (c.highCard || []);
      return wide.length
        ? `${wide.length} categorical ${plural(wide.length, 'column')} exceed 20 levels (${wide.slice(0, 3).map(x => `${x.name}, ${fmt.n(x.distinct)} levels`).join('; ')}); a level count that high on a short frame is often packed values rather than genuine categories.`
        : 'No categorical column carries an unusually high level count, so no packed-value signature is visible. Nested structures inside string columns were not parsed.';
    },
    example: c => `session.explode(\n  "${((c.highCard || [])[0] || (c.categorical || [])[0] || {}).name || '<column>'}",\n  sep=";")\n# grain changes: re-declare it\n\n# or keep grain:\nsession.indicators(\n  "<column>", sep=";", top_n=20)`,
    pitfalls: () => [
      'Encoding a delimited list as a category, producing one level per unique combination.',
      'Exploding without re-declaring the grain, so every subsequent rate is wrong.',
      'Pointing a one-hot encoder at a free-text column.',
    ],
  },

  /* ── Stage 2 additions ──────────────────────────────────────────── */

  'categorical-association': {
    stage: 2, title: 'categorical-association',
    prose: [
      'Correlation does not apply to categories, so association needs its own instruments. For two categorical columns, a chi-square test on the contingency table says whether the levels are related and Cramér\u2019s V (0 to 1, scale-free) says how strongly. For a categorical against a numeric column, the correlation ratio η — the share of the numeric column\u2019s variance explained by group membership — plays the same role.',
      'Both are affected by sparsity. A contingency table with many small cells makes chi-square unreliable (the conventional rule is an expected count of at least five per cell), and Cramér\u2019s V is biased upward when either column has many levels. On a wide categorical column, group the tail first and then measure.',
      'Read them for the same two purposes as correlation: redundancy between features, and a screen against the target. A V near 1 between two categorical features means one is a relabelling of the other; a high η between a category and the target names a grouping worth keeping intact through encoding.',
    ],
    read: c => [
      `Build the contingency table before the statistic and read the small cells: ${(c.categorical || []).length} categorical column(s) here${(c.highCard || []).length ? `, ${(c.highCard || []).length} of them wide enough to make cells thin` : ''}.`,
      'Group rare levels before measuring, then recompute — the statistic on an ungrouped tail measures the tail.',
      'Compare η across the categorical features to see which grouping explains most of the numeric target\u2019s variance.',
    ],
    decide: () => 'Measure categorical association with Cramér\u2019s V and η after grouping rare levels, and use the result for redundancy first, screening second.',
    session: c => (c.categorical || []).length
      ? `${c.categorical.length} categorical ${plural(c.categorical.length, 'column')} are present (${names(c.categorical, 4)}); no pairwise categorical association was computed in this session, so the correlation figures on the cockpit cover numeric pairs only.`
      : 'This frame has no categorical columns, so categorical association does not arise; the numeric correlations cover every pair.',
    example: c => `session.associations(\n  method="cramers_v",\n  columns=[${(c.categorical || []).slice(0, 2).map(x => `"${x.name}"`).join(', ') || '"<a>", "<b>"'}],\n  group_rare=0.01)`,
    pitfalls: () => [
      'Ordinal-encoding categories in order to run Pearson on them.',
      'Reading a chi-square p-value from a table full of cells with expected counts below five.',
      'Comparing Cramér\u2019s V across columns with very different level counts without correction.',
    ],
  },

  'non-linearity-and-binning': {
    stage: 2, title: 'non-linearity-and-binning',
    prose: [
      'A single coefficient assumes the relationship is a straight line. Real ones bend: they saturate, they have thresholds, they reverse. The cheap way to see the shape without fitting anything is to bin the feature into deciles and plot the target\u2019s mean per bin — a monotone staircase, a plateau or a U is visible immediately, and each implies a different treatment.',
      'Binning is a diagnostic that people then keep as a transform, which costs more than it looks. Discretising throws away within-bin variation, makes the model insensitive to movement inside a bin, and hands you a new arbitrary choice — the cut points, which if chosen using the target are another fitted step that must live inside the fold.',
      'The alternatives keep the resolution: splines and polynomials for linear models, or a tree-based estimator which finds the thresholds itself. Reach for explicit bins when the shape is genuinely a threshold effect that stakeholders need to see, not as a default treatment for skew.',
    ],
    read: c => [
      `Plot the target mean per decile of each candidate feature — ${(c.numeric || []).length} numeric column(s) here — and name the shape before choosing a treatment.`,
      c.corrPairs && c.corrPairs.length ? 'Compare Pearson with Spearman: a gap means monotone-but-curved, and a low value on both with a visible decile pattern means non-monotone.' : 'Compare Pearson with Spearman on any pair you care about; the gap is the curvature.',
      'Check the bin counts as well as the bin means. A dramatic mean over twelve rows is noise.',
    ],
    decide: () => 'Diagnose the shape with deciles, then keep resolution with splines or trees unless a threshold effect genuinely needs to be shown.',
    session: c => (c.numeric || []).length
      ? `${c.numeric.length} numeric ${plural(c.numeric.length, 'column')} were summarised with quartiles${(c.corrPairs || []).length ? ` and ${c.corrPairs.length} pair(s) scored linearly` : ''}. Every relationship measure on this sheet assumes linearity or monotonicity, so a saturating or reversing shape would not appear in any of them.`
      : 'No numeric columns, so shape does not arise; the relationship questions here are categorical.',
    example: c => `session.target_by_bin(\n  "${((c.analysable || [])[0] || {}).name || '<feature>'}",\n  bins=10)\n\n# read the shape, then prefer\n# splines over hard cuts`,
    pitfalls: () => [
      'Concluding "no relationship" from a near-zero Pearson coefficient on a U-shaped feature.',
      'Choosing bin edges using the target on the full frame.',
      'Binning by default and losing the resolution the model needed.',
    ],
  },

  'confounding-and-subgroups': {
    stage: 2, title: 'confounding-and-subgroups',
    prose: [
      'An aggregate relationship can reverse inside every subgroup. That is Simpson\u2019s paradox, and it is not a curiosity: it happens whenever a third variable both drives the target and is distributed unevenly across the feature\u2019s levels. The aggregate number is arithmetically correct and directionally wrong.',
      'The detection procedure is mechanical. Take the relationship you plan to report, pick the most plausible confounders — segment, region, period, source, product — and recompute within each. If the sign flips or the magnitude changes materially, the aggregate was a mixture and reporting it alone is misleading.',
      'What to do about it depends on the job. For prediction, include the confounder as a feature and let the model condition on it. For description, report the stratified view rather than the pooled one. For any claim about intervention, stop: this is the point where EDA hands over to a causal design.',
    ],
    read: c => [
      c.corrPairs && c.corrPairs.length ? `Take the strongest pair (${c.corrPairs[0].a} × ${c.corrPairs[0].b}, r=${c.corrPairs[0].r.toFixed(2)}) and recompute it within each level of a plausible confounder.` : 'Take each relationship you intend to report and recompute it within subgroups.',
      `Use the categorical columns as your stratifiers — ${(c.categorical || []).length} available here — starting with the one closest to a segment or a source.`,
      'Watch the subgroup sizes: a reversal inside a group of thirty rows is noise, not a paradox.',
    ],
    decide: () => 'Stratify every headline relationship by at least one plausible confounder before reporting it, and report the stratified view when the sign moves.',
    session: c => `Every relationship figure on this sheet is pooled across all ${fmt.n(c.rows)} rows. ${(c.categorical || []).length ? `${c.categorical.length} categorical ${plural(c.categorical.length, 'column')} (${names(c.categorical, 3)}) are available as stratifiers and none was used` : 'No categorical column is available as a stratifier'}, so no subgroup reversal could have been detected here.`,
    example: c => `session.correlations(\n  method="spearman",\n  by="${((c.categorical || [])[0] || {}).name || '<segment>'}")\n\n# compare each subgroup with\n# the pooled coefficient`,
    pitfalls: () => [
      'Reporting a pooled relationship that reverses in every segment.',
      'Stratifying until some subgroup shows the result you wanted.',
      'Controlling for a variable that sits on the causal path, which biases rather than corrects.',
    ],
  },

  'derived-and-redundant-columns': {
    stage: 2, title: 'derived-and-redundant-columns',
    prose: [
      'Wide frames accumulate columns that are functions of other columns: a total beside its components, an amount beside the same amount in another currency, a ratio beside its numerator and denominator, a band beside the continuous value it was cut from. Each duplicates information while consuming a coefficient, a split candidate and a slot in every importance ranking.',
      'The tells are specific and easy to check. A perfect or near-perfect correlation, an infinite VIF, a pair whose difference or ratio is constant, or a categorical whose levels map one-to-one onto another column\u2019s ranges. Any of these means one member can be removed with no loss of information at all.',
      'Choose which to keep on grounds other than the statistics: prefer the column that is measured rather than computed, closer to the source, more completely populated, and more interpretable to whoever reads the model. The derived column can always be recreated; a lost measurement cannot.',
    ],
    read: c => [
      `Read the top of the correlation ranking for pairs above |0.95| — ${(c.corrPairs || []).filter(p => Math.abs(p.r) >= 0.95).length} here — and check each pair for an exact functional relationship.`,
      `Look for infinite or extreme VIF values, which name exact linear dependencies (${(c.vif || []).length} feature(s) scored).`,
      'Compute the difference and the ratio of each suspicious pair; a constant result proves the redundancy.',
    ],
    decide: () => 'Delete or consolidate every column that is a function of others, keeping the measured and better-populated member.',
    session: c => {
      const near = (c.corrPairs || []).filter(p => Math.abs(p.r) >= 0.95);
      return near.length
        ? `${near.length} feature ${plural(near.length, 'pair')} correlate above |0.95| (${near.slice(0, 3).map(p => `${p.a} × ${p.b} at ${p.r.toFixed(3)}`).join('; ')}), which is the signature of one column being a re-expression of another.`
        : `No feature pair correlates above |0.95|${(c.corrPairs || []).length ? `; the strongest is ${c.corrPairs[0].a} × ${c.corrPairs[0].b} at ${c.corrPairs[0].r.toFixed(3)}` : ''}. Exact non-linear derivations would not show up in this screen.`;
    },
    example: c => `session.set_roles({\n  "${((c.corrPairs || [])[0] || {}).b || '<derived>'}": "ignore"})\n\n# keep the measured column,\n# drop the computed one`,
    pitfalls: () => [
      'Keeping a total alongside all of its components and reading the coefficients.',
      'Dropping whichever member of a pair appears first, rather than the derived one.',
      'Missing a non-linear derivation, which correlation will not flag.',
    ],
  },

  'time-feature-engineering': {
    stage: 2, title: 'time-feature-engineering',
    prose: [
      'A timestamp is not a feature; what you build from it is. The useful families are calendar parts (day of week, month, holiday flags), elapsed times (age of account, days since last event), lags (the value k periods ago) and rolling aggregates (mean, count or max over a trailing window). Each answers a different question and each carries a different risk.',
      'Cyclical parts need care in representation. Encoding month as 1–12 tells a linear model that December is eleven units from January when it is one; the standard repair is a sine and cosine pair, or letting a tree handle it, or treating it as a category.',
      'The risk that dominates is leakage through the window. Every aggregate must be computed over a window that closes at or before the prediction moment — a trailing window, never a centred or full-period one. "Total spend" computed over all data is the canonical leak: it includes the future for every row except the last.',
    ],
    read: c => [
      c.timeCol ? `Confirm the time column parses and orders correctly (${c.timeCol.name}, ${c.timeCol.min} to ${c.timeCol.max}) before deriving anything from it.` : 'Establish whether any column encodes time at all; without one, none of these features can be built honestly.',
      'For each derived feature, write down the window and check that it closes before the prediction moment.',
      'Check that lag features have enough history: the first k periods of any lag are missing by construction.',
    ],
    decide: () => 'Derive calendar, elapsed, lag and trailing-window features with explicitly closed windows, and encode cyclical parts cyclically.',
    session: c => c.timeCol
      ? `${c.timeCol.name} spans ${c.timeCol.min} to ${c.timeCol.max}${c.timeCol.gaps ? ` with ${fmt.n(c.timeCol.gaps)} coverage ${plural(c.timeCol.gaps, 'gap')}, so trailing windows will be uneven` : ''}. No derived time features exist in this session; the ${c.colCount} columns profiled here are as loaded.`
      : `No time column is declared, so no temporal feature could be derived or checked. ${c.colCount} columns were profiled as loaded.`,
    example: c => `session.add_time_features(\n  time_column="${(c.timeCol && c.timeCol.name) || '<timestamp>'}",\n  parts=["dow", "month"],\n  cyclical=True,\n  rolling={"window": "28d",\n           "closed": "left"})`,
    pitfalls: () => [
      'Aggregating over the full period instead of a trailing window.',
      'Encoding month or hour as a plain integer for a linear model.',
      'Building a lag feature and forgetting that the earliest rows now have no value.',
    ],
  },

  'sparsity-and-dimensionality': {
    stage: 2, title: 'sparsity-and-dimensionality',
    prose: [
      'Width relative to depth decides how much a model can learn. The working heuristic is rows per feature: comfortable above about twenty, thin below ten, and below one you are in the regime where a linear model can fit the training data exactly and learn nothing. One-hot encoding is what usually pushes a modest frame over the line.',
      'Sparsity compounds it. In a mostly-zero matrix each column\u2019s effective sample is the count of its non-zero rows, not the row count, so a one-hot column for a level with eight rows has eight rows behind its coefficient however large the frame is. Distance also degrades: in high dimensions the nearest and farthest neighbours become similar distances apart, which is what breaks k-NN and clustering long before it breaks trees.',
      'The responses are ordered by preference: remove redundancy, coarsen high-cardinality columns, then select, then reduce, then regularise. Reaching for PCA before deleting the duplicated columns solves a harder problem than the one you have.',
    ],
    read: c => [
      `Read rows per feature before and after encoding: ${fmt.n(c.rows)} rows over ${c.eligible} eligible features is about ${Math.round(c.rows / Math.max(1, c.eligible))} now${(c.highCard || []).length ? `, and one-hot encoding the wide categoricals would add roughly ${fmt.n((c.highCard || []).reduce((s, x) => s + x.distinct, 0))} columns` : ''}.`,
      'Read the non-zero count per encoded column, not the frame\u2019s row count, as each coefficient\u2019s real sample size.',
      'Ask whether the intended model is distance-based; if so, dimensionality is a first-order problem rather than a note.',
    ],
    decide: () => 'Keep rows per feature above about ten after encoding — by removing redundancy and coarsening categories first, reduction last.',
    session: c => `${fmt.n(c.rows)} rows carry ${c.eligible} eligible ${plural(c.eligible, 'feature')}, about ${Math.round(c.rows / Math.max(1, c.eligible))} rows per feature before encoding.${(c.highCard || []).length ? ` One-hot encoding the ${(c.highCard || []).length} wide categorical ${plural((c.highCard || []).length, 'column')} would add about ${fmt.n((c.highCard || []).reduce((s, x) => s + x.distinct, 0))} columns and cut that ratio sharply.` : ''}`,
    example: c => `session.group_rare_levels(\n  min_frequency=0.01)\nsession.explain(\n  "reduce_dimensions")\n\n# ${Math.round(c.rows / Math.max(1, c.eligible))} rows per feature now`,
    pitfalls: () => [
      'Counting rows per feature before encoding and declaring the frame comfortable.',
      'Running PCA while duplicated and derived columns are still in the matrix.',
      'Using k-NN or k-means on a wide sparse matrix and trusting the distances.',
    ],
  },
};
