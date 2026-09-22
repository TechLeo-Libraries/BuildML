# ruff: noqa: E501
"""Beginner layers for probabilistic modeling and uncertainty."""

from __future__ import annotations

from buildml.explain.beginner._builder import ADVANCED, CORE, BeginnerLayer, _index, _layer

PROBABILISTIC_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "probabilistic-uncertainty",
        plain=(
            "Most models give you one number and no sense of how sure they are. A probabilistic model gives "
            "you a number *and* a spread: 'about 42, give or take 6'. That extra half of the answer is "
            "often the part a decision actually needs."
        ),
        analogy=(
            "A delivery estimate of 'Tuesday' versus 'Tuesday, but anywhere from Monday to Friday'. Same "
            "central guess, completely different planning consequences."
        ),
        steps=(
            "Choose a model that can express uncertainty: Bayesian ridge, a Gaussian process, or a naive Bayes classifier.",
            "Fit it on training rows as usual.",
            "Predict: you get a central value plus a standard deviation, or a full probability per class.",
            "Optionally add conformal intervals for marginal coverage under exchangeability between calibration and future rows.",
            "Report the interval alongside the point estimate everywhere it will be used.",
        ),
        use=(
            "When downstream decisions depend on risk, not just on the expected value.",
            "When you need to know which predictions the model is guessing at, so a human can review those.",
        ),
        avoid=(
            "Do not use it when a point estimate genuinely suffices and the extra complexity buys nothing.",
            "Do not treat a model's own uncertainty as trustworthy without checking coverage: many models are overconfident.",
        ),
        myths=(
            (
                "A model's predicted standard deviation is its true uncertainty.",
                "It is the uncertainty implied by the model's assumptions. If those assumptions are wrong, the interval is wrong too: often too narrow.",
            ),
            (
                "Uncertainty estimates require Bayesian statistics.",
                "Conformal methods can calibrate many models without a parametric error distribution, but their marginal coverage guarantees require assumptions such as exchangeability.",
            ),
        ),
        example=(
            "session.probabilistic.fit(estimator='bayesian_ridge')",
            "pred = session.probabilistic.predict(partition='test')",
            "print(pred.predictions[:5], pred.std[:5])",
            "intervals = session.probabilistic.predict_interval(alpha=0.1)",
        ),
        check=(
            "What decision changes if the interval is wide rather than narrow?",
            "Of your 90% intervals, do about 90% actually contain the true value?",
        ),
        tools=("fit_probabilistic", "predict_probabilistic", "predict_interval", "evaluate_probabilistic"),
        terms=("probabilistic model", "prediction interval", "conformal prediction", "calibration"),
        difficulty=CORE,
    ),
    _layer(
        "probabilistic-bayesian-ridge",
        plain=(
            "Bayesian ridge is a linear regression that returns a spread as well as a value. It gets there "
            "by treating the coefficients as uncertain quantities rather than fixed numbers, and it tunes "
            "its own regularization strength from the data instead of needing you to pick it."
        ),
        analogy=(
            "A surveyor who reports 'the boundary is at 12.4 metres, plus or minus 30 centimetres' instead "
            "of just '12.4'. The tolerance comes from how consistent the measurements were."
        ),
        steps=(
            "Prepare numeric features and scale them, as with any linear model.",
            "Fit: the method estimates both the coefficients and how uncertain each one is.",
            "Predict with `return_std` to get the mean and the predictive standard deviation per row.",
            "Notice that rows unlike the training data get wider intervals, which is the behaviour you want.",
            "Check coverage on held-out rows before relying on the widths.",
        ),
        use=(
            "As a drop-in replacement for ridge regression when you want uncertainty for free.",
            "On small datasets, where automatic regularization tuning avoids a separate search.",
        ),
        avoid=(
            "Do not expect it to capture non-linear relationships; it is still a linear model underneath.",
            "Do not trust the intervals when the residuals are strongly non-Gaussian or heteroscedastic: the model assumes constant Gaussian noise.",
        ),
        myths=(
            (
                "Bayesian means slow and complicated.",
                "This one has a closed-form solution and runs about as fast as ordinary ridge regression.",
            ),
            (
                "The intervals are guaranteed to be correct.",
                "Posterior intervals depend on the model assumptions. Conformal calibration can provide marginal coverage under exchangeability, but does not guarantee coverage for each row or under distribution shift.",
            ),
        ),
        example=(
            "session.scale(method='standard')",
            "session.probabilistic.fit(estimator='bayesian_ridge')",
            "pred = session.probabilistic.predict(partition='validation')",
            "print(pred.predictions[:5], pred.std[:5])",
        ),
        check=(
            "Are your residuals roughly constant in spread across the prediction range?",
            "Do intervals widen for rows far from your training data?",
        ),
        tools=("fit_probabilistic", "predict_probabilistic", "predict_interval", "scale"),
        terms=("probabilistic model", "linear model", "regularization", "prediction interval"),
        difficulty=CORE,
    ),
    _layer(
        "probabilistic-gaussian-process",
        plain=(
            "A Gaussian process is a flexible model that fits smooth curves and, crucially, becomes "
            "appropriately unsure wherever it has little data. Its uncertainty is not a bolt-on: it comes "
            "directly from how far a point is from what it has seen."
        ),
        analogy=(
            "Drawing a curve through scattered dots. Where dots are dense, the curve is pinned down. Where "
            "there is a gap, many curves fit equally well, and the honest answer is a fan rather than a line."
        ),
        steps=(
            "Scale your features: the kernel measures distances and is entirely at the mercy of units.",
            "Choose a kernel, which encodes what kind of smoothness you expect. The default RBF is a reasonable start.",
            "Fit; the method learns the kernel's parameters from the data.",
            "Predict with `return_std` for regression, or use the classifier variant for probabilities.",
            "Watch the runtime: cost grows roughly with the cube of the row count.",
        ),
        use=(
            "On small datasets, up to a few thousand rows, where the uncertainty behaviour is worth the cost.",
            "For interpolation problems: sensor readings, spatial data, expensive experiments.",
        ),
        avoid=(
            "Do not use it on large datasets; it will not finish, and the sparse approximations are a different tool.",
            "Do not use it on high-dimensional data where distances stop being meaningful.",
        ),
        myths=(
            (
                "Gaussian processes are the most accurate models available.",
                "They are excellent on small, smooth, low-dimensional problems. On large tabular data gradient boosting beats them comfortably and finishes.",
            ),
            (
                "These are deep Gaussian processes.",
                "BuildML exposes scikit-learn's classical GPs. Deep GPs are a different, much heavier family that is not part of this surface.",
            ),
        ),
        example=(
            "session.scale(method='standard')",
            "session.probabilistic.fit(estimator='gaussian_process_regressor', random_state=0)",
            "pred = session.probabilistic.predict(partition='validation')",
            "print(pred.predictions[:5], pred.std[:5])   # std grows away from training data",
        ),
        check=(
            "How many training rows do you have, and can you afford the cubic cost?",
            "Does the predicted spread widen where your data is sparse?",
        ),
        tools=("fit_probabilistic", "predict_probabilistic", "scale", "predict_interval"),
        terms=("Gaussian process", "probabilistic model", "scaling", "prediction interval"),
        difficulty=ADVANCED,
    ),
    _layer(
        "probabilistic-split-conformal",
        plain=(
            "Split conformal prediction calibrates intervals using errors on a held-out slice of training "
            "rows. When calibration and future rows can be treated as interchangeable (exchangeability), "
            "an error cutoff adjusted for calibration sample size provides coverage across future cases "
            "overall (marginal coverage). This is not a guarantee for each row or subgroup, and it may "
            "fail when the deployment population changes."
        ),
        analogy=(
            "A commute app uses past delays to suggest a time allowance. That allowance is useful only "
            "if future journeys resemble the calibration journeys; a new road closure can invalidate it. "
            "It describes performance across journeys, not certainty about tomorrow's trip."
        ),
        steps=(
            "Carve a calibration slice out of the training rows: never from validation or test.",
            "Fit your model on the remaining training rows.",
            "Measure the absolute residuals on the calibration slice.",
            "Take the quantile of those residuals matching your desired coverage.",
            "The interval is the prediction plus and minus that quantile; for classification, it is the set of classes whose scores clear the corresponding cut-off.",
        ),
        use=(
            "When you can justify exchangeability and need a marginal coverage target with its assumptions stated.",
            "On top of any model, including gradient boosting and neural networks that have no native uncertainty.",
        ),
        avoid=(
            "With a tiny calibration slice, quantiles are coarse and uncertain; 40 residuals permit a 95th-percentile estimate but provide limited tail information.",
            "Do not use it when the deployment distribution differs from calibration: the guarantee assumes exchangeability and quietly breaks under drift.",
        ),
        myths=(
            (
                "Conformal intervals are tight.",
                "Coverage does not imply tightness. Basic absolute-residual split conformal gives every row the same width, so it can be wide for easy rows. Adaptive variants require additional modeling choices.",
            ),
            (
                "The guarantee holds per row.",
                "The guarantee is marginal under exchangeability. Individual rows and subgroups can have lower coverage, and finite evaluation samples need not match the requested rate.",
            ),
        ),
        example=(
            "session.probabilistic.fit(",
            "    estimator='bayesian_ridge', conformal=True, conformal_calibration_fraction=0.2,",
            ")",
            "intervals = session.probabilistic.predict_interval(alpha=0.1, partition='test')",
            "print(intervals.lower[:5], intervals.upper[:5])",
        ),
        check=(
            "How many rows are in your calibration slice?",
            "On test, what fraction of true values fell inside your 90% intervals?",
        ),
        tools=("fit_probabilistic", "predict_interval", "evaluate_probabilistic"),
        terms=("conformal prediction", "prediction interval", "calibration", "train"),
        difficulty=ADVANCED,
    ),
    _layer(
        "probabilistic-bundle-boundary",
        plain=(
            "The probabilistic plan: the fitted model plus any conformal calibration it needs: saves as "
            "its own bundle. The calibration quantiles are part of the model: without them the intervals "
            "cannot be reproduced."
        ),
        analogy=(
            "A measuring instrument and its calibration certificate. The instrument alone still produces "
            "numbers; it just cannot tell you how much to trust them."
        ),
        steps=(
            "Fit a probabilistic model, optionally with conformal calibration.",
            "Call `session.probabilistic.save_bundle(path)`: the calibration state travels with the estimator.",
            "Reload with `session.probabilistic.load_bundle(path)`.",
            "Call `session.probabilistic.predict_interval` on new rows and get the same widths you validated.",
            "Checkpoint separately for the data state.",
        ),
        use=(
            "When intervals are part of the deliverable and must reproduce exactly.",
            "For any scheduled job that reports uncertainty to a downstream consumer.",
        ),
        avoid=(
            "Do not save only the estimator and recompute calibration later on different rows; the widths will differ.",
            "Do not expect a Session checkpoint to hold the probabilistic plan.",
        ),
        myths=(
            (
                "Calibration can be recomputed on demand.",
                "It can, on a different slice, giving different quantiles. Reproducibility of the interval requires the original calibration state.",
            ),
            (
                "Intervals are metadata, not model state.",
                "They are model state. The quantile is a fitted number exactly like a coefficient.",
            ),
        ),
        example=(
            "session.probabilistic.save_bundle('artifacts/demand-uncertainty')",
            "job = Session.ingest(new_frame).probabilistic.load_bundle('artifacts/demand-uncertainty', trusted=True)",
            "job.probabilistic.predict_interval(alpha=0.1, partition='all')",
        ),
        check=(
            "Does your bundle include the conformal calibration state?",
            "Would today's intervals match the ones you validated last month?",
        ),
        tools=("save_probabilistic_bundle", "load_probabilistic_bundle", "predict_interval", "checkpoint_save"),
        terms=("bundle", "checkpoint", "conformal prediction", "calibration"),
        difficulty=CORE,
    ),
    _layer(
        "probabilistic-mapie",
        plain=(
            "MAPIE is a dedicated conformal-prediction library. With the optional extra installed, BuildML "
            "can use its more sophisticated variants: cross-conformal and jackknife+: which reuse all "
            "your training rows for calibration instead of setting a slice aside."
        ),
        analogy=(
            "Instead of reserving one day's deliveries to measure delays, you rotate through the whole "
            "month, so every day contributes to the estimate and you waste nothing."
        ),
        steps=(
            "Install `pip install buildml[probabilistic-industry]`.",
            "Choose a method: `split` is the simple one, `cv_plus` and `jackknife_plus` reuse all training rows through folds.",
            "Fit through the MAPIE backend.",
            "Request intervals at your target coverage.",
            "Check the empirical coverage on held-out rows: the guarantee is asymptotic, not exact.",
        ),
        use=(
            "When your training set is small and giving up a calibration slice hurts.",
            "When you want the extra methods and the extra dependency is acceptable.",
        ),
        avoid=(
            "Do not use the cross-conformal variants when refitting is expensive; they train the model once per fold.",
            "Do not install the extra just for basic split conformal: BuildML does that natively.",
        ),
        myths=(
            (
                "Cross-conformal intervals are strictly better.",
                "They use data more efficiently and cost k model fits. On plentiful data, split conformal is simpler and equally adequate.",
            ),
            (
                "Conformal guarantees survive drift.",
                "They rest on exchangeability between calibration and deployment rows. Under drift, coverage silently degrades.",
            ),
        ),
        example=(
            "# pip install \"buildml[probabilistic-industry]\"",
            "session.probabilistic.fit(",
            "    backend='mapie', estimator='cv_plus',",
            ")",
            "print(session.probabilistic.evaluate(partition='test').metrics)",
        ),
        check=(
            "How many model fits will your chosen conformal variant require?",
            "Is your deployment population exchangeable with your calibration rows?",
        ),
        tools=("fit_probabilistic", "predict_interval", "evaluate_probabilistic"),
        terms=("conformal prediction", "cross-validation", "prediction interval", "extra"),
        difficulty=ADVANCED,
    ),
    _layer(
        "probabilistic-ngboost",
        plain=(
            "NGBoost is gradient boosting that predicts a whole distribution instead of a single value. "
            "Each row gets its own mean and its own spread, so the model can say 'this one is easy and this "
            "one is genuinely uncertain'."
        ),
        analogy=(
            "A forecaster who says 'tomorrow: 18 degrees, very confident' for a settled week and '18 "
            "degrees, but it could be anywhere from 10 to 26' when a front is coming through."
        ),
        steps=(
            "Install `pip install buildml[probabilistic-industry]`.",
            "Choose a distribution: Normal for regression, Bernoulli for binary classification.",
            "Fit: boosting optimizes both the location and the spread parameters together.",
            "Predict to get per-row distributional parameters, not a single number.",
            "Score with a proper scoring rule such as negative log-likelihood, not only with MAE.",
        ),
        use=(
            "When uncertainty genuinely varies row by row and a constant-width interval would be misleading.",
            "When you already trust gradient boosting for the point prediction and want uncertainty in the same model.",
        ),
        avoid=(
            "Do not use it when a constant-width conformal interval is adequate; it is slower and has more knobs.",
            "Do not assume the distributional assumption fits: a Normal predictive distribution on skewed, bounded, or count data will be wrong in a specific direction.",
        ),
        myths=(
            (
                "Per-row uncertainty is automatically better calibrated.",
                "It is more expressive, and expressiveness without calibration checking is just a more confident way to be wrong. Verify coverage.",
            ),
            (
                "NGBoost replaces conformal prediction.",
                "They complement each other. NGBoost models varying spread; conformal supplies the guarantee. Wrapping one in the other is a reasonable combination.",
            ),
        ),
        example=(
            "# pip install \"buildml[probabilistic-industry]\"",
            "session.probabilistic.fit(backend='ngboost', estimator='ngboost_regressor', random_state=0)",
            "pred = session.probabilistic.predict(partition='validation')",
            "print(pred.predictions[:5], pred.std[:5])   # std varies per row",
        ),
        check=(
            "Does the predicted spread actually vary meaningfully across rows?",
            "Is your target's shape compatible with the distribution you chose?",
        ),
        tools=("fit_probabilistic", "predict_probabilistic", "predict_interval", "evaluate_probabilistic"),
        terms=("probabilistic model", "gradient boosting", "prediction interval", "calibration", "extra"),
        difficulty=ADVANCED,
    ),
)

__all__ = ["PROBABILISTIC_BEGINNER"]
