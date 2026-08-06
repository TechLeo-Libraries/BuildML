// Teaching depth for the stage 1–2 core concepts: extra prose paragraphs that
// carry the mechanism and the arithmetic, a "how to read it" procedure, and the
// single decision the concept forces. Merged onto eda-concepts-core.js.

import { fmt } from './eda-format.js';

export const DEPTH_A = {

  'column-roles': {
    more: [
      'The mechanics are simple and the discipline is not. A role table is a mapping from column name to one of four values, applied before anything else happens, and every later step reads it: the split sees only rows, the imputer sees only feature columns, the metric sees only the target. Nothing infers a role for you, and an unassigned column defaults to whatever the pipeline\u2019s selector happens to sweep up — usually "every numeric column", which is exactly how an account number becomes a feature.',
      'Two roles deserve their own scrutiny. An identifier is any column whose value indexes the row rather than describing it — account numbers, ticket ids, hashes, row order. A target is the thing you predict, and its definition (which rows are positive, over what horizon, measured when) is a modelling decision recorded here rather than a fact about the file.',
    ],
    read: c => [
      `Compare distinct count to row count. A ratio above roughly 0.95 — here, ${fmt.n(Math.round(c.rows * 0.95))} of ${fmt.n(c.rows)} — means the column is near-unique and is either an identifier or a continuous measurement; the name and dtype tell you which.`,
      'Check every column against the moment of prediction: would this value exist, for this row, before the outcome is known? If not, its role is ignore regardless of how predictive it looks.',
      'Count the eligible features at the end and sanity-check it against the frame width. A large gap means columns were silently dropped; no gap at all usually means nothing was reviewed.',
    ],
    decide: c => `Write the role of all ${c.colCount} columns down explicitly — feature, target, id, ignore — and treat any column you cannot classify as ignore until someone can.`,
  },

  'dtypes-and-storage': {
    more: [
      'Inference is the failure point. A CSV loader reads a sample of rows, guesses a type per column, and moves on: a column of integers with one stray "N/A" becomes object; a column of "0012" codes becomes int64 and loses its padding; a column of "TRUE"/"FALSE"/"" becomes object and evaluates truthy for every non-empty string, including "FALSE". None of these raise an error, and each one changes the meaning of every aggregate computed downstream.',
      'Storage also decides what operations are even available. Categorical dtype gives you a fixed level set, which is what lets you detect an unseen level at prediction time; datetime dtype gives you ordering and differencing; float gives you NaN as a first-class missing marker where int does not. Choosing the dtype is therefore choosing which checks you can run later.',
    ],
    read: () => [
      'Read the schema next to a sample of five raw rows. Any column whose dtype is object but whose values look numeric or boolean was inferred wrongly, not stored deliberately.',
      'For each numeric column, ask whether arithmetic on it is meaningful. Adding two postcodes is not; averaging two ratings may not be either.',
      'Check memory per column. An object column costing far more than its neighbours is usually a low-cardinality string that should be categorical.',
    ],
    decide: () => 'Declare every dtype explicitly at load time and re-read the schema afterwards; never let inference stand as documentation.',
  },

  'missing-data': {
    more: [
      'Imputation is a fitted transform, and that is the whole of the discipline around it. "Fit" means computing the fill value — a median, a mode, a regression — from a set of rows; "transform" means writing it into the gaps. Fit on the training rows, transform everywhere. Fit on everything and the test rows have contributed to a number that then appears in the training matrix, which is leakage in its most ordinary form.',
      'Choose the strategy from the column\u2019s shape and the gap\u2019s size, not from habit. Median survives skew where mean does not; most-frequent is defensible for a category with a dominant level and misleading for a flat one; a model-based fill borrows accuracy from the other columns and borrows their errors too. Above roughly 40% missing, no fill is honest and the column is better represented by its own presence indicator.',
    ],
    read: c => [
      `Read the per-column rate before the total. ${fmt.pct(c.completeness)} cell completeness across the frame can mean a light dusting everywhere or one column that is almost empty — here the worst column is ${c.missing[0] ? `${c.missing[0].name} at ${fmt.pct(c.missing[0].missingRate, 1)}` : 'none'}.`,
      `Count complete rows separately: ${fmt.n(c.completeRows)} of ${fmt.n(c.rows)} rows have no gaps at all, which is the sample size any complete-case analysis actually runs on.`,
      'Ask what each gap means in the source system — not answered, not applicable, not yet recorded, or lost in transit. Those four call for four different treatments.',
    ],
    decide: c => c.missingCells === 0
      ? 'Nothing to fill here, so record the strategy you would use if the next extract is not complete.'
      : 'Pick one strategy per column, fit it inside the training fold, and add a missing-indicator wherever the gap could itself be informative.',
  },

  'missingness-mechanisms': {
    more: [
      'The three names describe where the gap\u2019s probability lives. MCAR: the chance of being missing is unrelated to anything, so you lose precision and nothing else. MAR: the chance depends on other observed columns, so those columns can reconstruct it — this is the case imputation is designed for. MNAR: the chance depends on the unobserved value itself, so the rows that would have been extreme are exactly the rows that are blank, and no fill computed from the survivors can recover them.',
      'You test toward the answer rather than proving it. Build a binary indicator for "missing in column X", then check whether it predicts the target and whether it correlates with the other features. A strong association rules out MCAR and makes the indicator worth keeping as a feature in its own right; no association is weak evidence for MCAR and no evidence at all against MNAR.',
    ],
    read: c => [
      'Compare the target rate among rows missing a column against rows that have it. A visible difference means the gap carries signal.',
      'Compare the distributions of the other features between the missing and present groups. Systematic differences point to MAR and tell you which columns to impute from.',
      c.missing && c.missing.length ? `Ask the source owner about the worst column (${c.missing[0].name}) before modelling it — the mechanism is usually documented in a process, not in the data.` : 'Ask whether the extract filtered out incomplete rows upstream; a clean frame can be the residue of a silent drop.',
    ],
    decide: () => 'Classify each gappy column MCAR / MAR / MNAR on the evidence you have, keep an indicator wherever the answer is not MCAR, and record the classification as an assumption.',
  },

  'duplicate-records': {
    more: [
      'Duplication has an arithmetic you can check. Declare the key you believe identifies a row, then compare three numbers: total rows, distinct keys, and distinct full rows. Rows above distinct-full-rows means exact copies. Distinct-full-rows above distinct-keys means the key repeats with different payloads — the table\u2019s grain is finer than you assumed, and the fix is to redefine the key or aggregate to the grain, not to drop rows.',
      'Duplicates also arrive from joins. A many-to-many merge multiplies rows silently, and the symptom is a row count that grew when you expected it to stay put. Check the count before and after every merge; a change you did not predict is a bug until explained.',
    ],
    read: c => [
      `Count exact duplicates first — here ${c.duplicates ? fmt.n(c.duplicates.rows) : 'none'} of ${fmt.n(c.rows)} rows — then key duplicates, then near-duplicates after string cleaning.`,
      'For each repeated key, inspect two of the colliding rows side by side. Whether they differ, and in which columns, tells you the real grain.',
      'Check whether duplicates cluster in one source, one period or one segment. Concentrated duplication is an ingestion incident, not noise.',
    ],
    decide: () => 'State the grain of the table in one sentence, enforce it with a uniqueness check, and de-duplicate before drawing any split.',
  },

  'constant-and-near-constant': {
    more: [
      'Quantify it as the share of the most common level. At 100% the column is constant and carries zero information. Above about 99% it is near-constant: with 5-fold cross-validation on a few thousand rows, several folds will contain no minority value at all, so any coefficient or split learned on it is fitted to a handful of rows and will not reproduce.',
      'The decision is not "drop or keep" but "drop, keep, or reframe". A rare flag that matters is often better expressed at a coarser grain — combined with other rare flags into an "any exception" indicator — which concentrates the support instead of spreading it.',
    ],
    read: c => [
      `Read the top-level share, not the variance: variance depends on scale, share does not. Anything above 99% — and the near-constant screen here found ${(c.nearConstant || []).length} — is effectively a constant with a rumour attached.`,
      'Multiply the minority share by the fold size. If the answer is under about ten rows per fold, the column cannot be learned from reliably.',
      'Ask what the rare level means before dropping it. "Fraud", "recall" and "died" are all rare and all the point of the exercise.',
    ],
    decide: () => 'Drop true constants outright; for near-constants either coarsen the column or keep it and accept that its estimate is thin — but never scale a zero-variance column.',
  },

  'high-cardinality': {
    more: [
      'The cost is concrete. One-hot encoding a column with k levels adds k − 1 columns, and each new column\u2019s support is the count of that level: with 400 levels over 5,000 rows the median level has a dozen rows behind it. Wide and sparse is not merely inefficient, it changes what regularisation does, and it guarantees unseen levels at prediction time.',
      'The three repairs trade different things. Frequency grouping keeps the matrix narrow and destroys the identity of the tail. Target encoding keeps it narrow and encodes signal, at the price of needing an in-fold fit and a smoothing prior. Attribute replacement — city to region, SKU to product family — keeps interpretability and needs a domain table that someone has to maintain.',
    ],
    read: c => [
      `Rank categorical columns by distinct count and read the level-frequency curve, not just the total: ${(c.highCard || []).length} column(s) here exceed 20 levels.`,
      'Compute the share of rows in the top ten levels. A high share means the tail is grouping-friendly; a flat curve means grouping throws away most of the column.',
      'Estimate the one-hot width you would create and compare it to the row count. More columns than rows is a decision, not an accident.',
    ],
    decide: () => 'For each high-cardinality column choose one of group-rare, in-fold target encoding, or attribute replacement — and write down the unseen-level policy at the same time.',
  },

  'categorical-encoding': {
    more: [
      'Every encoder is a fitted mapping, so it inherits two obligations: it must be fitted inside the fold, and it must have an answer for a level it has never seen. One-hot handles the unknown by producing all zeros if you ask it to (handle_unknown="ignore") and by crashing if you do not. Ordinal maps the unknown to a sentinel that sits inside the numeric range and quietly asserts something false.',
      'Ordinal encoding is only correct where an order genuinely exists and the spacing is defensible — small/medium/large, yes; postcode, no. Target encoding is the strongest and the most dangerous: the encoded value is a statistic of the target, so any fit that has seen the validation rows has copied the answer into the feature.',
    ],
    read: c => [
      `List each categorical column with its level count and decide the encoder per column, not per frame: ${c.categorical.length} column(s) need a decision here.`,
      'For any column you plan to ordinal-encode, write the intended order out. If you cannot, the order does not exist.',
      'Check that the encoder is inside the same pipeline object as the estimator, so cross-validation refits it per fold.',
    ],
    decide: () => 'Choose an encoder per categorical column, fit it in-fold, and set an explicit handle_unknown policy for every one.',
  },

  'measurement-units-and-ranges': {
    more: [
      'Range checks are cheap and specific: for each numeric column write the minimum and maximum the world allows, then count violations. Negative counts, ages over 120, percentages above 1, timestamps in 1970 (a zero epoch) and values like −999, 9999 or 99999 are sentinels or artefacts, and averaging them corrupts every statistic that follows.',
      'Units break silently across joins and across time. Two systems reporting "duration" in seconds and minutes produce a bimodal distribution with a 60× gap; a currency column mixing cents and units does the same at 100×. The tell is a histogram with two separated modes of similar shape, or a distribution whose spread is much larger than the domain allows.',
    ],
    read: c => [
      `For each of the ${(c.numeric || []).length} numeric columns, read min and max against a plausible bound before reading the mean.`,
      'Look for repeated extreme values. A single value appearing hundreds of times at the edge of the range is a sentinel, not a measurement.',
      'Check for suspicious multiplicative gaps (60×, 100×, 1000×) between modes — that is a unit mix, not a subpopulation.',
    ],
    decide: () => 'Assert an allowed range per numeric column, convert sentinels to missing before any statistic is computed, and confirm one unit per column at every merge.',
  },

  'text-hygiene': {
    more: [
      'The pipeline is fixed and the order matters: strip whitespace, normalise unicode (NFKC), case-fold, collapse internal whitespace, then recount levels. Doing it in this order is what turns " Gold", "gold" and "Gold " into one level. Counting first and cleaning later means every cardinality number, every encoding decision and every rare-level threshold was computed on the mess.',
      'Cleaning is itself a transform that must be reproduced at prediction time, byte for byte. A model trained on case-folded categories and served raw strings sees nothing but unseen levels — a failure that passes every offline test.',
    ],
    read: c => [
      'Compare level counts before and after a strip-and-lowercase pass. The difference is the fragmentation you were about to encode.',
      'Sort levels alphabetically and read the neighbours: near-identical adjacent strings are the same level.',
      `Check for mixed types inside a column — ${(c.cols || []).filter(x => x.mixedType).length} column(s) flagged here — since a stray number in a string column defeats both string and numeric handling.`,
    ],
    decide: () => 'Normalise every string column with one documented pipeline, recount levels afterwards, and ship the same normalisation to prediction time.',
  },

  'univariate-distributions': {
    more: [
      'Read six numbers per column and one picture. Min, q1, median, q3, max and the count of distinct values tell you centre, spread, both tails and granularity; the histogram tells you shape — unimodal, bimodal, zero-spiked, truncated. Mean and standard deviation alone can describe a distribution you would never recognise.',
      'Specific shapes carry specific diagnoses. A spike at exactly zero suggests structural zeros rather than measurement. A hard edge with a pile-up against it suggests clipping or a data-entry limit. A gap in the middle suggests two populations merged. Digit preference — everything ending in 0 or 5 — suggests self-reported values.',
    ],
    read: c => [
      (() => { const w = (c.numeric || []).filter(n => n.hasStats && n.role !== 'id' && isFinite(n.max - n.min)).sort((a, b) => (b.max - b.min) - (a.max - a.min))[0]; return w ? `Compare median with mean per column: a large gap is skew. The widest-ranging feature here is ${w.name}.` : 'Compare median with mean per column: a large gap is skew. This profile supplied no ranges, so start by computing them.'; })(),
      'Compare q3 with max. A max many multiples above q3 is a tail; a max equal to q3 is a ceiling.',
      'Check q1 against min. Equality means a floor, a sentinel, or a large mass at one value.',
    ],
    decide: () => 'Summarise every numeric column with quartiles plus a histogram, and name the shape before choosing any transform.',
  },

  'skew-and-transforms': {
    more: [
      'Skew is scale-free — roughly, the average cubed deviation in standard-deviation units — so |skew| above 1 is the conventional flag and above 2 is severe. The practical consequence is that squared-error loss weights each row by the square of its error, so the tail rows dominate both the fit and the reported metric.',
      'Pick the transform to fit the data\u2019s support. log1p handles zeros and needs non-negative values; Box-Cox estimates its own exponent and needs strictly positive input; Yeo-Johnson accepts negatives; a rank or quantile transform is the blunt instrument that always works and destroys the metric\u2019s meaning. All of them are fitted, all of them belong in the fold, and none of them help a tree.',
    ],
    read: c => [
      `Read |skew| per column against the 1.0 flag — ${(c.skewed || []).length} column(s) here exceed it — then look at the histogram to see whether the skew is a tail or a second mode.`,
      'Check the minimum before choosing: a zero rules out plain log, a negative rules out Box-Cox.',
      'Ask what the model is. For gradient boosting and random forests a monotone transform changes nothing at all.',
    ],
    decide: () => 'Transform skewed inputs only for models that care, fit the transform in-fold, and report error in original units.',
  },

  'correlation': {
    more: [
      'The two coefficients answer different questions. Pearson measures linear co-movement and is sensitive to outliers and to scale; Spearman correlates the ranks, so it catches any monotone relationship and shrugs at outliers. Compute both: |Spearman| much larger than |Pearson| means a monotone but curved relationship, and the reverse usually means a few extreme points are manufacturing the linear fit.',
      'Read the matrix twice with different intent. Feature-to-target is a screening signal and a very weak one — it is univariate, linear (or monotone), and blind to combinations. Feature-to-feature is a redundancy map: pairs above about |0.8| are candidates for consolidation, and |0.95| usually means one column is a re-expression of the other.',
    ],
    read: c => [
      `Sort pairs by |r| and read the top of the list, not the matrix as a whole — ${(c.corrPairs || []).length} pair(s) scored here.`,
      'For every strong pair, decide which member is measured more reliably and closer to the source, and keep that one.',
      'Scatter-plot any pair whose Pearson and Spearman disagree; the shape is the finding.',
    ],
    decide: () => 'Use correlation to consolidate redundant features, never to select features on its own, and recompute it inside the fold if it drives a decision.',
  },

  'mutual-information': {
    more: [
      'MI is defined as the reduction in entropy of the target given the feature — zero when they are independent, unbounded above, and reported in nats or bits. That makes it invariant to monotone transforms and blind to direction: a feature that raises the target then lowers it scores highly, and so does its mirror image.',
      'Because it is estimated from finite data, the estimator\u2019s bias matters as much as the ranking. Nearest-neighbour estimators depend on k; binned estimators depend on the bin count; both inflate the scores of high-cardinality features, which is why an identifier can top an MI table. Differences smaller than the spread you get from re-estimating with a different seed are not differences.',
    ],
    read: c => [
      c.mi && c.mi.length ? `Read the gap between neighbours, not their order: ${c.mi[0].name} at ${fmt.dec(c.mi[0].mi, 6)} versus ${c.mi[1] ? `${c.mi[1].name} at ${fmt.dec(c.mi[1].mi, 6)}` : 'no second feature'}.` : 'With no target declared there is nothing to score; the ranking arrives only once a target exists.',
      'Re-estimate with another seed or another k. Whatever moves was noise.',
      'Cross-check the top of the table against the leakage screen — an implausibly strong single feature is usually a leak, not a discovery.',
    ],
    decide: () => 'Treat MI as a shortlist for inspection, keep low scorers until a model has seen them in combination, and refit any MI-based selection inside the fold.',
  },

  'variance-inflation': {
    more: [
      'The definition is mechanical: regress feature j on all the other numeric features, take the R² of that regression, and VIF = 1 / (1 − R²). R² of 0.9 gives VIF 10; 0.99 gives 100. The convention is that 5 warrants a look and 10 warrants action, but both are conventions, not tests — and the standard error of a coefficient grows with the square root of its VIF, which is the number that actually matters.',
      'Perfect collinearity has its own signature: an infinite or absurdly large VIF means one feature is an exact linear combination of others — a total alongside its parts, a set of one-hot columns with no dropped reference level, a duplicated column under a new name. That is an algebra problem, not a modelling one, and the fix is to remove the redundancy rather than to regularise it.',
    ],
    read: c => [
      `Read the whole set at once and remove one member at a time: ${(c.vif || []).length} feature(s) were scored here against a ${(c.vifThreshold || 5).toFixed(1)} threshold.`,
      'Look for near-infinite values first — they name an exact dependency you can delete without loss.',
      'Ask whether the model needs interpretable coefficients at all. If it is a tree or a purely predictive pipeline, high VIF is a note, not a defect.',
    ],
    decide: () => 'Drop or combine one collinear feature at a time and recompute; only interpret coefficients once the set is conditioned.',
  },

  'interaction-effects': {
    more: [
      'An interaction means the effect of one feature depends on the level of another — the difference in target between two values of A is not the same at every value of B. Every univariate screen on this sheet is blind to it by construction, which is why a feature with near-zero MI can be indispensable in a pair.',
      'Two ways to look before committing. Group the target by a coarse bin of A crossed with a coarse bin of B and read the cell means: a table whose rows are not parallel is an interaction. Or fit a shallow tree and read its second split, which is exactly where an interaction lives. Then add the term deliberately — products for linear models, nothing at all for gradient boosting, which finds them itself.',
    ],
    read: c => [
      'Cross two candidate features into a small grid of target means and check whether the pattern in one row repeats in the others.',
      'Fit a depth-2 tree and read which pairs it chooses; those are your candidates.',
      c.mi && c.mi.length > 1 ? `Start from the pair the univariate screen ranks first and second (${c.mi[0].name}, ${c.mi[1].name}) — but do not stop there, since the interesting pairs are usually not both strong alone.` : 'Start from domain knowledge; with no univariate ranking there is nothing to seed the pairing.',
    ],
    decide: () => 'Never eliminate a feature on univariate evidence alone; nominate two or three interaction candidates and test them in-fold.',
  },

  'dimensionality-reduction': {
    more: [
      'PCA rotates the feature space onto orthogonal directions of maximum variance, so component 1 is the direction along which rows differ most. Two consequences follow: variance is not the same as relevance (a high-variance direction can be pure noise, and a low-variance one can carry the target), and every component is a weighted sum of all original columns, so nothing you report afterwards names a column anyone measured.',
      'Selection is the alternative that keeps names. Filter methods score columns independently, wrapper methods search subsets against a model, embedded methods (L1, tree importances) select while fitting. All three are fitted steps and all three must run inside the fold — a selector that saw the validation rows has chosen features using the answer.',
    ],
    read: c => [
      `Read rows per feature first: ${fmt.n(c.rows)} rows over ${c.eligible} eligible features is about ${Math.round(c.rows / Math.max(1, c.eligible))} rows each. Under about 10 the frame is thin and reduction is worth considering.`,
      'Look at the cumulative explained-variance curve for an elbow, and treat the component count as a hyper-parameter with its own validation, not as a setting.',
      'Check whether the problem is really a handful of duplicated or derived columns; if so, delete those instead.',
    ],
    decide: () => 'Prefer selection when you need to explain the model and PCA when you need conditioning; fit either inside the fold and report accordingly.',
  },

  'feature-scaling': {
    more: [
      'Which models care is not a matter of taste. Distance and kernel methods (k-NN, SVM, k-means), any regularised linear model (the penalty is applied to coefficients, so it depends on units), PCA and neural networks all need comparable ranges. Decision trees, random forests and gradient boosting split on order alone and are invariant to any monotone rescaling.',
      'Pick the scaler from the distribution. Standardisation subtracts mean and divides by standard deviation, and both statistics are dragged by outliers. Min-max maps to a fixed interval and is entirely determined by the two most extreme rows. Robust scaling uses median and IQR and is the sane default whenever the tails are heavy.',
    ],
    read: c => [
      `Compare the ranges across numeric columns — they differ by a factor of about ${(() => { const s = (c.numeric || []).map(n => Math.abs(n.max - n.min)).filter(x => isFinite(x) && x > 0); return s.length ? fmt.compact(Math.max(...s) / Math.min(...s)) : '1'; })()} here — and decide whether the model in mind is scale-sensitive at all.`,
      'Check skew and outliers before choosing standard over robust scaling.',
      'Decide explicitly whether one-hot and binary columns go through the scaler; either answer is defensible, silence is not.',
    ],
    decide: () => 'Scale only for models that need it, choose the scaler from the tails, and fit it on training rows inside the pipeline.',
  },
};
