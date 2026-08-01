# BuildML quality bar (locked)

Applies to every capability created so far and going forward.

## Rule

Do **not** ship thin convenience wrappers. Each public feature must be:

1. **Extensive** — covers the professional conversation around that workflow turn  
2. **Adaptive** — chooses strategies from data shape, dtypes, cardinality, scale, and roles  
3. **Deep** — returns structured insight, not just a single metric or plot  
4. **Creative / high-impact** where visual — layouts and chart choices should teach and reveal, not decorate  
5. **Documented** — purpose, args, returns, examples, leakage notes, scale notes  
6. **Tested** — unit + integration coverage for happy path and edge cases  

## EDA standard (non-negotiable)

Must include at minimum:

- quality/completeness/pattern/id-like screens  
- univariate diagnostics (including normality/entropy where applicable)  
- bivariate correlations + mutual information vs target  
- multivariate collinearity (VIF/clusters) + PCA summary  
- target-aware statistical screens  
- outlier screens (univariate + multivariate)  
- train/test drift when a split exists  
- adaptive visualization plan + optional figure/HTML export  
- evidence-linked findings and recommendations

A thin `describe()` wrapper is a failed implementation.

## Modeling / evaluation standard

Must include rich metrics, diagnostics (confusion/residuals), probability tools (calibration/thresholds where applicable), learning curves, importance, and multi-model comparison — not accuracy-only helpers.

## Packaging note

Heavy backends (seaborn, imbalanced-learn, etc.) may live in extras. **Analytical depth stays in core structured reports**; visuals and samplers become available with extras.
