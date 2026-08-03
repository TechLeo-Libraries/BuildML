# ruff: noqa: E501
"""Beginner layers for online / incremental learning."""

from __future__ import annotations

from buildml.explain.beginner._builder import CORE, BeginnerLayer, _index, _layer

ONLINE_BEGINNER: dict[str, BeginnerLayer] = _index(
    _layer(
        "online-partial-fit",
        plain=(
            "Online learning updates a model with new data instead of retraining it from scratch. You feed "
            "it a chunk of rows, it adjusts, and it is immediately ready to predict again. In BuildML this "
            "is incremental scikit-learn fitting over training chunks — not a distributed streaming platform."
        ),
        analogy=(
            "Adding notes to a notebook you already keep, rather than rewriting the whole notebook every "
            "time you learn something new."
        ),
        steps=(
            "Split as usual — the chunks come from the training partition.",
            "Fit an initial model on the first chunk to establish the plan.",
            "Feed subsequent chunks with `partial_fit_online`; each call nudges the model.",
            "Predict at any point with `predict_online`; the model is always usable.",
            "Evaluate on held-out rows the incremental updates never touched.",
        ),
        use=(
            "When data arrives continuously and full retraining is too slow or too expensive.",
            "When the dataset is larger than memory and you want to stream it through in chunks.",
        ),
        avoid=(
            "Do not use it when you can comfortably retrain from scratch; batch retraining is simpler and usually more accurate.",
            "Do not use it with estimators that lack `partial_fit` — most tree ensembles cannot update incrementally at all.",
        ),
        myths=(
            (
                "Online learning is just batch learning done in pieces.",
                "The model sees each chunk once, in order, so chunk ordering affects the result. Batch fitting sees everything at once, repeatedly.",
            ),
            (
                "Online models never need retraining.",
                "They accumulate drift and can slowly wander. Periodic full retraining remains the safety net.",
            ),
        ),
        example=(
            "session.fit_online(estimator=SGDClassifier(loss='log_loss'), chunk_size=1000)",
            "for chunk in later_chunks:",
            "    session.partial_fit_online(chunk)",
            "session.evaluate_online(partition='validation')",
        ),
        check=(
            "Does your estimator actually support incremental updates?",
            "Would the result change if the chunks arrived in a different order?",
        ),
        tools=("fit_online", "partial_fit_online", "predict_online", "evaluate_online"),
        terms=("online learning", "batch", "estimator", "drift"),
        difficulty=CORE,
    ),
    _layer(
        "online-class-discovery",
        plain=(
            "An incremental classifier has to know the complete list of possible classes before its very "
            "first update, because it allocates internal structure for each one. A class it meets later "
            "cannot be added. So you either declare the list up front or let BuildML read it from the "
            "training targets."
        ),
        analogy=(
            "Printing ballot papers. Every candidate has to be on the sheet before voting starts — you "
            "cannot write one in halfway through the count."
        ),
        steps=(
            "Work out the full set of labels your problem can produce, including rare ones.",
            "Pass them explicitly as `classes=[...]` on the first fit when you know them.",
            "If you do not pass them, BuildML discovers them from the training partition's labels.",
            "Confirm the discovered list contains every class you expect — a class absent from your first chunk is a real risk.",
            "If a genuinely new class appears later, you need a fresh model, not another update.",
        ),
        use=(
            "Always for incremental classification. There is no way around it; the only choice is explicit or discovered.",
            "Explicitly whenever a rare class might be missing from the early chunks.",
        ),
        avoid=(
            "Do not rely on discovery when your first chunk is small or time-ordered — the rare class may simply not be there yet.",
            "Do not discover classes from unlabelled rows; only labelled targets count.",
        ),
        myths=(
            (
                "The model will pick up new classes as it sees them.",
                "It will not. An unseen class at initialization time cannot be predicted, and the update will either error or silently ignore those rows.",
            ),
            (
                "This is a quirk of BuildML.",
                "It is how scikit-learn's `partial_fit` works for classifiers. BuildML surfaces it explicitly instead of letting it surprise you mid-stream.",
            ),
        ),
        example=(
            "session.fit_online(",
            "    estimator=SGDClassifier(loss='log_loss'),",
            "    classes=['low', 'medium', 'high', 'critical'],   # all of them, up front",
            ")",
            "print(session.online_plan.classes)",
        ),
        check=(
            "Is every class in your problem present in your declared or discovered list?",
            "What happens in your pipeline if an unknown label arrives next month?",
        ),
        tools=("fit_online", "partial_fit_online", "predict_online"),
        terms=("online learning", "target", "categorical", "estimator"),
        difficulty=CORE,
    ),
    _layer(
        "online-drift-disclose",
        plain=(
            "As chunks stream past, the data can change. BuildML records a lightweight comparison between "
            "each chunk and the very first one — mostly shifts in column means — so you get a warning "
            "signal. It is a smoke detector, not a full drift monitoring product."
        ),
        analogy=(
            "A dashboard warning light. It tells you to look under the bonnet. It does not tell you what is "
            "wrong or how serious it is."
        ),
        steps=(
            "Fit an initial chunk; its statistics become the reference.",
            "Each subsequent chunk is compared against that reference.",
            "BuildML notes columns whose mean has shifted noticeably and attaches the note to the update record.",
            "Read those notes as a prompt to investigate, not as a verdict.",
            "When a note fires, check whether performance actually moved before reacting.",
        ),
        use=(
            "On any long-running incremental job, as cheap early warning.",
            "When you have no dedicated drift monitoring in place and want something rather than nothing.",
        ),
        avoid=(
            "Do not treat it as a drift product; it does not do distributional tests, effect sizes, or label-aware performance tracking.",
            "Do not act on a single chunk's note — chunk-to-chunk variation is large, especially with small chunks.",
        ),
        myths=(
            (
                "A drift note means the model has degraded.",
                "It means an input distribution moved. Whether performance followed is a separate question that needs labels to answer.",
            ),
            (
                "No drift notes means everything is fine.",
                "Mean-shift detection misses variance changes, correlation changes, and changes in the relationship between features and target.",
            ),
        ),
        example=(
            "session.partial_fit_online(chunk, disclose_drift=True)",
            "for note in session.online_plan.update_history[-1].drift_notes:",
            "    print(note)",
        ),
        check=(
            "Do you have any labels arriving that would let you measure real performance drift?",
            "Is your chunk size large enough for a mean shift to be meaningful?",
        ),
        tools=("partial_fit_online", "evaluate_online", "eda"),
        terms=("drift", "online learning", "distribution", "disclosure"),
        difficulty=CORE,
    ),
    _layer(
        "online-bundle-boundary",
        plain=(
            "An online model's bundle stores more than the estimator: it also stores the cursor (how far "
            "through the stream you are) and the update history. Those are what make a resumed stream "
            "continue correctly rather than restart."
        ),
        analogy=(
            "A bookmark plus a reading log, not just the book. Without the bookmark you start over; without "
            "the log you do not know what you already read."
        ),
        steps=(
            "Run at least one incremental update so a plan exists.",
            "Call `save_online_bundle(path)` to store the estimator, the cursor, and the history.",
            "Reload with `load_online_bundle(path)` when the job restarts.",
            "Continue with `partial_fit_online` from the recorded position.",
            "Keep checkpoints separately for the underlying data state.",
        ),
        use=(
            "For scheduled jobs that must survive restarts without replaying the whole stream.",
            "When you need an audit trail of how many updates the deployed model has absorbed.",
        ),
        avoid=(
            "Do not resume from a stale bundle against a stream that has moved on — check the cursor first.",
            "Do not expect a Session checkpoint to hold the online plan; it does not.",
        ),
        myths=(
            (
                "Saving the estimator is enough to resume.",
                "Without the cursor you will re-feed rows the model already absorbed, over-weighting them.",
            ),
            (
                "The update history is only for logging.",
                "It carries the drift notes and update counts that let you judge whether the deployed model is still the one you validated.",
            ),
        ),
        example=(
            "session.save_online_bundle('artifacts/online-model')",
            "job = Session.ingest(new_chunk).load_online_bundle('artifacts/online-model')",
            "print(job.online_plan.cursor, len(job.online_plan.update_history))",
            "job.partial_fit_online(new_chunk)",
        ),
        check=(
            "Where does your cursor point, and where does your stream currently stand?",
            "How many updates has the deployed model absorbed since it was last validated?",
        ),
        tools=("save_online_bundle", "load_online_bundle", "partial_fit_online", "checkpoint_save"),
        terms=("bundle", "checkpoint", "online learning", "history"),
        difficulty=CORE,
    ),
)

__all__ = ["ONLINE_BEGINNER"]
