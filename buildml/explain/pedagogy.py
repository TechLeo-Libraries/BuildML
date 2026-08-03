# ruff: noqa: E501
"""Derive the beginner primer that fronts every operation explanation.

The catalog holds 287 operations. Hand-writing a beginner briefing for each one
would guarantee drift: the prose would age out of step with the parameters,
prerequisites, and concept links that are already maintained elsewhere. So the
primer is *derived*: from the operation's kind, its prerequisites, its
parameters, the state it changes, and the beginner layer of the concepts it
links to.

Derivation is not simplification. Everything a primer says is sourced from
material a maintainer already keeps correct, and any operation can override any
section by authoring the matching field on its :class:`OperationSpec`.
"""

from __future__ import annotations

import re
from functools import cache, lru_cache

from buildml.explain.catalog import get_operation
from buildml.explain.concepts import CONCEPT_NOTES
from buildml.explain.glossary import detect_terms
from buildml.explain.prerequisites import plain_prerequisite, providers_for
from buildml.explain.schemas import (
    GlossaryTerm,
    LearningLevel,
    OperationKind,
    OperationPrimer,
    OperationSpec,
    ParameterMeaning,
    ParameterSpec,
    PrerequisiteStatus,
)


@lru_cache(maxsize=1)
def _kind_framing() -> dict[OperationKind, tuple[str, str, str]]:
    """Kind → (opening sentence, why-it-exists sentence, fallback analogy)."""
    return {
        OperationKind.INGEST: (
            "This is how data gets into BuildML in the first place.",
            "Nothing else can run until the session is holding a dataset it understands.",
            "Unpacking the shopping before you can start cooking.",
        ),
        OperationKind.CONFIGURE: (
            "This sets something up so that later steps know what to do.",
            "It records a decision once, in one place, instead of leaving every later step to guess.",
            "Setting the rules of the game before anyone starts playing.",
        ),
        OperationKind.INSPECT: (
            "This looks at what you have and reports back. Nothing is changed.",
            "Deciding what to do next is much easier when you can see what you are actually holding.",
            "Reading the label before you open the tin.",
        ),
        OperationKind.SPLIT: (
            "This decides which rows are used for learning and which are held back for honest testing.",
            "Without a boundary between what a model learns from and what judges it, every score you produce flatters you.",
            "Setting exam questions aside before the revision starts.",
        ),
        OperationKind.TRANSFORM: (
            "This changes your data, learning what to do from the training rows only.",
            "Models need numbers in a workable shape, and the rules for reshaping must come from training data alone.",
            "Preparing ingredients to a recipe written before you saw tonight's guests.",
        ),
        OperationKind.MODEL: (
            "This learns something from your training rows and keeps the result on the session.",
            "It is the step where patterns in your data become an artifact you can apply to new rows.",
            "Studying past cases until you can handle a new one.",
        ),
        OperationKind.DIAGNOSTIC: (
            "This measures or investigates something about what you have already built.",
            "A number without a diagnosis is not evidence; this is where you find out whether to trust the model.",
            "Sending the prototype for testing rather than admiring it.",
        ),
        OperationKind.PERSIST: (
            "This writes state to disk, or reads it back.",
            "Work that only exists inside one running process is work you will do again.",
            "Filing the paperwork so tomorrow does not start from nothing.",
        ),
        OperationKind.EXPORT: (
            "This hands your data or model to something outside BuildML.",
            "At some point the result has to leave the session and be useful somewhere else.",
            "Packing the parcel for someone who does not work here.",
        ),
    }


_KIND_FALLBACK_AVOID: dict[OperationKind, tuple[str, ...]] = {
    OperationKind.INGEST: (
        "Do not ingest again mid-workflow expecting roles, splits, and fitted plans to survive.",
        "Do not use it as a reload shortcut when what you actually want is a checkpoint.",
    ),
    OperationKind.CONFIGURE: (
        "Do not configure once and assume it survives a change to the underlying columns.",
        "Do not treat configuration as a substitute for checking that the setting is correct.",
    ),
    OperationKind.INSPECT: (
        "Do not use what you learn from holdout rows to choose features or settings.",
        "Do not treat a summary statistic as a finding without checking how it was computed.",
    ),
    OperationKind.SPLIT: (
        "Do not split after fitting preprocessing; the preprocessing has already seen everything.",
        "Do not re-split to chase a better score.",
    ),
    OperationKind.TRANSFORM: (
        "Do not fit the transformation before the split.",
        "Do not apply a transformation you cannot reproduce at prediction time.",
    ),
    OperationKind.MODEL: (
        "Do not fit on rows you intend to evaluate against.",
        "Do not judge the result on the training score.",
    ),
    OperationKind.DIAGNOSTIC: (
        "Do not repeatedly consult the test partition while still making choices.",
        "Do not read a single headline metric as the whole answer.",
    ),
    OperationKind.PERSIST: (
        "Do not assume one artifact type contains another; each has its own load-time contract.",
        "Do not overwrite an artifact you may need to reproduce later.",
    ),
    OperationKind.EXPORT: (
        "Do not export holdout-derived artifacts you have not finished evaluating.",
        "Do not assume the receiving system reproduces BuildML's preprocessing for you.",
    ),
}

_READ_ONLY_KINDS = {OperationKind.INSPECT, OperationKind.DIAGNOSTIC, OperationKind.EXPORT}

_KEY_PARAMETER_LIMIT: dict[LearningLevel, int] = {
    LearningLevel.BEGINNER: 6,
    LearningLevel.INTERMEDIATE: 12,
    LearningLevel.ADVANCED: 32,
}
_PITFALL_LIMIT: dict[LearningLevel, int] = {
    LearningLevel.BEGINNER: 5,
    LearningLevel.INTERMEDIATE: 8,
    LearningLevel.ADVANCED: 24,
}
_GLOSSARY_LIMIT: dict[LearningLevel, int] = {
    LearningLevel.BEGINNER: 6,
    LearningLevel.INTERMEDIATE: 4,
    LearningLevel.ADVANCED: 0,
}

# Parameters whose plain meaning is worth spelling out wherever they appear.
# name -> (plain meaning, effect of increasing, effect of decreasing, typical choice)
_PARAMETER_MEANINGS: dict[str, tuple[str, str, str, str]] = {
    "partition": (
        "Which slice of your data to work on: the rows the model learned from, the rows you tune against, or the rows you saved for the final honest check.",
        "",
        "",
        "Use 'validation' while you are still deciding things and 'test' only once, at the end.",
    ),
    "path": (
        "Where on disk to write the artifact, or where to read it from.",
        "",
        "",
        "A directory inside your project, such as 'artifacts/<name>'.",
    ),
    "random_state": (
        "The seed for anything random in this step. Fixing it makes the result repeatable.",
        "A different seed gives a different but equally valid result.",
        "",
        "Set it to any fixed integer so your run can be reproduced.",
    ),
    "seed": (
        "The seed for anything random in this step. Fixing it makes the result repeatable.",
        "A different seed gives a different but equally valid result.",
        "",
        "Set it to any fixed integer so your run can be reproduced.",
    ),
    "backend": (
        "Which underlying library does the work. The BuildML call stays the same; the engine behind it changes.",
        "",
        "",
        "Start with the native backend and switch only when you need something it does not offer.",
    ),
    "method": (
        "Which algorithm to use for this step.",
        "",
        "",
        "Start with the default, then compare alternatives on held-out data.",
    ),
    "estimator": (
        "The underlying model object that does the learning.",
        "",
        "",
        "A scikit-learn compatible estimator; leave unset to accept BuildML's default choice.",
    ),
    "base_estimator": (
        "The model used as the building block inside a larger procedure.",
        "",
        "",
        "Something fast and well-behaved, since it may be fitted many times.",
    ),
    "task": (
        "Whether you are predicting a category or a number. 'auto' lets BuildML infer it from the target column.",
        "",
        "",
        "Leave on 'auto' unless the inference is getting it wrong.",
    ),
    "columns": (
        "Which columns this step applies to. Leaving it unset usually means every eligible column.",
        "",
        "",
        "Name them explicitly once your dataset is stable, so a new column cannot silently join in.",
    ),
    "test_size": (
        "How much of your data to hold back for testing, as a fraction or a row count.",
        "A larger test set gives a steadier score but leaves less to learn from.",
        "A smaller test set leaves more for training but makes the score noisier.",
        "0.2 is a common starting point on a few thousand rows.",
    ),
    "validation_size": (
        "How much to hold back for tuning decisions, kept separate from the final test rows.",
        "More validation rows make comparisons between candidates more reliable.",
        "Fewer validation rows make model selection noisier and more prone to luck.",
        "Add one whenever you will compare more than a couple of options.",
    ),
    "stratify": (
        "Whether to keep each class in the same proportion across the split.",
        "",
        "",
        "Turn it on for classification, especially when one class is rare.",
    ),
    "cv": (
        "How many folds to split the training data into for cross-validation.",
        "More folds use more of the data for training each time, at more compute.",
        "Fewer folds are quicker but give a noisier estimate.",
        "5 is the usual compromise.",
    ),
    "cv_strategy": (
        "How cross-validation folds are formed: plain, stratified by class, grouped by entity, or ordered by time.",
        "",
        "",
        "Match it to your data: grouped when rows repeat per entity, time-ordered when order matters.",
    ),
    "scoring_metric": (
        "The single number used to rank candidates during a search.",
        "",
        "",
        "Choose the metric that reflects your actual decision cost, not the conventional default.",
    ),
    "n_iter": (
        "How many candidate configurations to try.",
        "More candidates explore more of the space and take proportionally longer.",
        "Fewer candidates finish sooner and may miss the good region entirely.",
        "Start small to check the plumbing, then raise it for the real run.",
    ),
    "n_trials": (
        "How many configurations the search is allowed to evaluate.",
        "More trials search harder and cost more time.",
        "Fewer trials risk stopping before the search has learned anything.",
        "Enough that the best score has visibly stopped improving.",
    ),
    "epochs": (
        "How many complete passes to make over the training data.",
        "More passes fit the training data better and eventually start memorizing it.",
        "Fewer passes may stop before the model has learned what it could.",
        "Use early stopping on a validation split instead of guessing.",
    ),
    "batch_size": (
        "How many rows are processed together before the model updates itself.",
        "Larger batches train faster per pass and use more memory.",
        "Smaller batches update more often, which is noisier but sometimes generalizes better.",
        "32 to 256 for tabular data, bounded by the memory you have.",
    ),
    "learning_rate": (
        "How big a step the model takes each time it corrects itself.",
        "A larger rate learns quickly and can overshoot or diverge entirely.",
        "A smaller rate is stable but may need many more passes.",
        "Around 0.001 for neural networks; lower it if the loss bounces around.",
    ),
    "device": (
        "Where the computation runs: the processor or the graphics card.",
        "",
        "",
        "Leave it on automatic unless you are pinning a specific device.",
    ),
    "alpha": (
        "How strongly the model is penalized for being complicated.",
        "More penalty gives a simpler, steadier model that may underfit.",
        "Less penalty lets the model fit detail, including noise.",
        "Tune it on validation rather than guessing.",
    ),
    "max_iter": (
        "The cap on how long the fitting procedure is allowed to keep going.",
        "A higher cap gives the solver room to converge.",
        "A lower cap may stop before the answer settles, usually with a warning.",
        "Raise it if you see a convergence warning.",
    ),
    "k": (
        "How many items to consider or return.",
        "A larger k smooths over more neighbours or returns a longer list.",
        "A smaller k is more local and more sensitive to individual rows.",
        "Odd values avoid ties for two-class voting.",
    ),
    "threshold": (
        "The cut-off that turns a score into a yes-or-no decision.",
        "A higher threshold flags fewer rows, catching less and being wrong less often.",
        "A lower threshold flags more rows, catching more and raising more false alarms.",
        "Choose it from the cost of each mistake, not from 0.5 by habit.",
    ),
    "n_jobs": (
        "How many processor cores to use.",
        "More cores finish sooner and use more of the machine.",
        "One core is slower and easier to debug.",
        "-1 uses everything available.",
    ),
    "verbose": (
        "How much progress information to print.",
        "",
        "",
        "Turn it up when something is slow and you want to see where.",
    ),
    "shuffle_train": (
        "Whether to reorder the training rows before each pass.",
        "",
        "",
        "Keep it on unless row order carries meaning you must preserve.",
    ),
    "groups": (
        "The column identifying which entity each row belongs to, so rows from one entity never straddle a split.",
        "",
        "",
        "Set it whenever the same customer, patient, or device appears in several rows.",
    ),
    "time_column": (
        "The column holding the timestamp, used to keep the past and the future apart.",
        "",
        "",
        "Required for anything where order matters; a random split would leak the future.",
    ),
    "target_column": (
        "The column holding the answer you want to predict.",
        "",
        "",
        "Usually set once through roles rather than per call.",
    ),
    "text_column": (
        "Which column holds the documents.",
        "",
        "",
        "Name it explicitly when more than one column contains free text.",
    ),
    "normalize": (
        "Whether to put values on a comparable scale before the computation.",
        "",
        "",
        "Leave it on for anything distance-based.",
    ),
    "strategy": (
        "Which approach to take for this step.",
        "",
        "",
        "Start with the default and change it once you know what it is doing.",
    ),
    "mode": (
        "Which of several behaviours this call should perform.",
        "",
        "",
        "Pick the mode matching the question you are asking, not the one with the most output.",
    ),
    "export_html": (
        "Where to write a self-contained HTML report, if you want one.",
        "",
        "",
        "Leave unset for interactive work; set it when sharing results.",
    ),
    "export_figures": (
        "Where to write generated figures, if you want them on disk.",
        "",
        "",
        "Leave unset unless you need the images as files.",
    ),
    "refit": (
        "Whether to retrain on all the training data once the best configuration is known.",
        "",
        "",
        "Usually yes, so the final model uses everything it is allowed to.",
    ),
    "preprocess": (
        "Whether this step should apply the session's fitted preprocessing.",
        "",
        "",
        "Keep it consistent between fitting and scoring, or the numbers will not match.",
    ),
    "n_estimators": (
        "How many individual models make up the ensemble.",
        "More members usually help a little and cost proportionally more time.",
        "Fewer members train faster and may be less stable.",
        "A few hundred is a reasonable default for tree ensembles.",
    ),
    "max_depth": (
        "How many decisions deep each tree may go.",
        "Deeper trees capture finer patterns and memorize more readily.",
        "Shallower trees are more general and may miss real structure.",
        "Start shallow and increase only if validation improves.",
    ),
    "n_components": (
        "How many combined dimensions to keep.",
        "Keeping more retains more of the original information.",
        "Keeping fewer compresses harder and discards more.",
        "Choose by how much variance you need to retain, not by round numbers.",
    ),
    "n_clusters": (
        "How many groups to split the rows into.",
        "More groups make each one tighter and less meaningful.",
        "Fewer groups are easier to describe and blur real distinctions.",
        "There is no correct value; compare a few and interpret them.",
    ),
    "n_bins": (
        "How many buckets each value is divided into.",
        "More buckets keep more detail and give each bucket fewer rows.",
        "Fewer buckets are coarser and better populated.",
        "Watch how many rows land in each bucket before increasing it.",
    ),
    "n_rounds": (
        "How many rounds the procedure runs for.",
        "More rounds keep refining and cost more time.",
        "Fewer rounds may stop before things settle.",
        "Watch the round history and stop once it flattens.",
    ),
    "n_episodes": (
        "How many episodes to run.",
        "More episodes give more experience and take longer.",
        "Fewer episodes may not cover enough situations to learn from.",
        "Enough that the reward curve has visibly levelled off.",
    ),
    "horizon": (
        "How many steps into the future to forecast.",
        "A longer horizon is more useful and much less accurate.",
        "A shorter horizon is easier to get right and may be too late to act on.",
        "Match it to how far ahead the decision actually needs to be made.",
    ),
    "contamination": (
        "What share of rows you expect to be unusual.",
        "A higher value flags more rows as anomalies.",
        "A lower value flags fewer and misses more.",
        "Set it from how many alerts you can actually investigate.",
    ),
    "sample_rows": (
        "How many rows to look at, when looking at all of them would be slow.",
        "More rows give a more representative picture and take longer.",
        "Fewer rows are quicker and may miss rare patterns.",
        "Enough that the summary stops changing when you raise it.",
    ),
    "attach": (
        "Whether the result is stored on the session or just returned to you.",
        "",
        "",
        "Attach it when later steps depend on it.",
    ),
    "name": (
        "A label for this artifact, used when you have several.",
        "",
        "",
        "Something you will recognize in three months.",
    ),
}

_PARAMETER_PATTERNS: tuple[tuple[re.Pattern[str], str, str, str], ...] = (
    (
        re.compile(r"^n_(?!jobs$)\w+"),
        "How many of these to use.",
        "A larger number does more work and costs more time.",
        "A smaller number is quicker and may be too coarse.",
    ),
    (
        re.compile(r"^max_\w+"),
        "The upper limit for this setting.",
        "Raising the cap allows more, at more cost.",
        "Lowering the cap constrains the procedure, sometimes helpfully.",
    ),
    (
        re.compile(r"^min_\w+"),
        "The lower limit for this setting.",
        "Raising the floor filters out more of the small cases.",
        "Lowering the floor keeps more of them, including the noise.",
    ),
    (
        re.compile(r"\w*_?state$|^seed$|_seed$"),
        "The seed controlling randomness, so the run can be repeated exactly.",
        "",
        "",
    ),
    (
        re.compile(r"^export_\w+|_path$|^path$"),
        "Where to write the output, if you want it on disk.",
        "",
        "",
    ),
    (
        re.compile(r"^\w*column$|^\w*columns$"),
        "Which column or columns this step should use.",
        "",
        "",
    ),
    (
        re.compile(r"^(use_|include_|allow_|enable_|with_)\w+"),
        "A switch turning this behaviour on or off.",
        "",
        "",
    ),
)


def parameter_meaning(parameter: ParameterSpec) -> ParameterMeaning:
    """Translate one catalog parameter into plain language.

    A type and a one-line description tell a beginner what to type, not what
    will happen. This adds the two things they actually need: which direction to
    move the value, and what a reasonable starting value is.

    Resolution is a three-step fallback. Parameters that recur across the
    catalog: ``random_state``, ``test_size``, ``threshold``: have a curated
    reading. Anything else is matched against family patterns (``n_*``,
    ``max_*``, ``*_column``). Failing both, the catalog description is used,
    which is terse but always accurate.

    Parameters
    ----------
    parameter:
        The catalog specification for one argument of one operation.

    Returns
    -------
    ~buildml.explain.schemas.ParameterMeaning
        The plain reading: what the knob controls, the effect of raising and
        lowering it where that is meaningful, and a typical choice.
    """
    curated = _PARAMETER_MEANINGS.get(parameter.name)
    if curated is not None:
        plain, up, down, typical = curated
        return ParameterMeaning(
            name=parameter.name,
            plain_meaning=plain,
            effect_of_increase=up,
            effect_of_decrease=down,
            typical_choice=typical or _default_hint(parameter),
        )
    for pattern, plain, up, down in _PARAMETER_PATTERNS:
        if pattern.match(parameter.name):
            return ParameterMeaning(
                name=parameter.name,
                plain_meaning=plain,
                effect_of_increase=up,
                effect_of_decrease=down,
                typical_choice=_default_hint(parameter),
            )
    return ParameterMeaning(
        name=parameter.name,
        plain_meaning=parameter.description,
        typical_choice=_default_hint(parameter),
    )


def _default_hint(parameter: ParameterSpec) -> str:
    if parameter.required:
        return f"Required; expects {parameter.type_name}."
    if parameter.choices:
        return "One of: " + ", ".join(parameter.choices) + "."
    if parameter.default is not None:
        return f"Defaults to {parameter.default!r}."
    return "Optional; leave unset to accept BuildML's default behaviour."


def _ordered_parameters(spec: OperationSpec, limit: int) -> tuple[ParameterSpec, ...]:
    """Required knobs first, then the ones a beginner is most likely to touch."""
    required = [item for item in spec.parameters if item.required]
    curated = [
        item
        for item in spec.parameters
        if not item.required and item.name in _PARAMETER_MEANINGS
    ]
    rest = [
        item
        for item in spec.parameters
        if not item.required and item.name not in _PARAMETER_MEANINGS
    ]
    return tuple((required + curated + rest)[:limit])


def _dedupe(*groups: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for item in group:
            text = item.strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return tuple(out)


def _linked_notes(spec: OperationSpec) -> tuple:
    return tuple(
        CONCEPT_NOTES[key] for key in spec.concept_links if key in CONCEPT_NOTES
    )


def _analogy(spec: OperationSpec) -> str:
    if spec.analogy:
        return spec.analogy
    for note in _linked_notes(spec):
        if note.analogy:
            return note.analogy
    return _kind_framing()[spec.kind][2]


def _plain_summary(spec: OperationSpec) -> str:
    if spec.plain_summary:
        return spec.plain_summary
    lead = _kind_framing()[spec.kind][0]
    return f"{lead} {spec.definition} {spec.purpose}".strip()


def _decapitalize(text: str) -> str:
    """Lower the first letter unless it opens an acronym such as RAG or NLP."""
    if len(text) > 1 and text[1].isupper():
        return text
    return text[0].lower() + text[1:]


def _why_it_exists(spec: OperationSpec) -> str:
    role = spec.pipeline_role.strip().rstrip(".")
    why = _kind_framing()[spec.kind][1]
    if role:
        return f"{why} In a BuildML workflow this call is the {_decapitalize(role)}."
    return why


def _steps(spec: OperationSpec) -> tuple[str, ...]:
    if spec.beginner_steps:
        return spec.beginner_steps
    steps: list[str] = []
    for item in spec.prerequisites:
        if item.status is PrerequisiteStatus.REQUIRED:
            steps.append(f"First make sure: {plain_prerequisite(item.key)}")
    steps.append(f"Call `session.{spec.name}(...)` with the arguments you need.")
    steps.extend(spec.mechanism)
    for change in spec.state_changes:
        steps.append(f"When it finishes: {change}")
    for reading in spec.result_reading[:2]:
        steps.append(f"Then read the result: {reading}")
    return _dedupe(tuple(steps))


def _prerequisites_in_plain_words(spec: OperationSpec) -> tuple[str, ...]:
    lines: list[str] = []
    for item in spec.prerequisites:
        sentence = plain_prerequisite(item.key)
        providers = providers_for(item.key)
        if providers:
            sentence = f"{sentence} Establish it with {' or '.join(providers[:3])}."
        if item.status is not PrerequisiteStatus.REQUIRED:
            sentence = f"{sentence} ({item.status.value})"
        lines.append(sentence)
    if not lines:
        lines.append("Nothing has to be true first; this call stands on its own.")
    return tuple(lines)


def _when_to_use(spec: OperationSpec) -> tuple[str, ...]:
    """Say when to reach for this, not where it sits in the running order.

    ``usual_ordering`` is already rendered in the expert appropriateness
    section, and its entries frequently describe role requirements rather than
    suitability. It is therefore only a fallback, used when the operation has no
    selection rationale of its own.
    """
    if spec.when_to_use:
        return spec.when_to_use
    if spec.selection_rationale:
        return _dedupe(spec.selection_rationale)
    if spec.usual_ordering:
        return _dedupe(spec.usual_ordering)
    return (f"When you need to {_decapitalize(spec.purpose)}",)


def _when_not_to_use(spec: OperationSpec) -> tuple[str, ...]:
    if spec.when_not_to_use:
        return spec.when_not_to_use
    derived = _dedupe(spec.anti_patterns, spec.leakage_risks)
    if derived:
        return derived
    return _KIND_FALLBACK_AVOID[spec.kind]


def _what_changes(spec: OperationSpec) -> tuple[str, ...]:
    if spec.state_changes:
        return spec.state_changes
    if spec.kind in _READ_ONLY_KINDS:
        return ("Nothing on the session changes; this call only reads and reports.",)
    return ("No session state change is recorded for this operation.",)


def _how_to_read(spec: OperationSpec) -> tuple[str, ...]:
    if spec.result_reading:
        return spec.result_reading
    if spec.outputs:
        return tuple(f"The call returns {item}" for item in spec.outputs)
    return ("Inspect the returned object and the session history entry it added.",)


def _pitfalls(spec: OperationSpec, avoid: tuple[str, ...], limit: int) -> tuple[str, ...]:
    """List the mistakes not already stated as reasons to avoid the operation.

    Both sections draw on ``anti_patterns`` and ``leakage_risks``, so without
    subtraction a beginner reads the same warning twice in one briefing and
    learns nothing from the repetition. When subtraction would empty the list,
    the unfiltered one is kept: a repeated warning beats a missing one.
    """
    everything = _dedupe(spec.leakage_risks, spec.failure_modes, spec.anti_patterns)
    already = set(avoid)
    remaining = tuple(item for item in everything if item not in already)
    return (remaining or everything)[:limit]


def _mini_example(spec: OperationSpec) -> tuple[str, ...]:
    """Prefer authored code, then real concept code that calls this operation."""
    if spec.mini_example:
        return spec.mini_example
    marker = f"{spec.name}("
    for note in _linked_notes(spec):
        if any(marker in line for line in note.mini_example):
            return note.mini_example
    return _derived_call(spec)


def _derived_call(spec: OperationSpec) -> tuple[str, ...]:
    """Build a signature-accurate call sketch when no real snippet exists."""
    arguments: list[str] = []
    for parameter in spec.parameters:
        if parameter.required:
            arguments.append(f"{parameter.name}=<{parameter.type_name}>")
        elif parameter.name in _PARAMETER_MEANINGS and parameter.default is not None:
            arguments.append(f"{parameter.name}={parameter.default!r}")
        if len(arguments) >= 4:
            break
    call = f"session.{spec.name}({', '.join(arguments)})"
    lines = [call]
    providers = ()
    for item in spec.prerequisites:
        if item.status is PrerequisiteStatus.REQUIRED:
            providers = providers_for(item.key)
            if providers:
                break
    if providers:
        lines.insert(0, f"# after: session.{providers[0]}(...)")
    if spec.next_considerations:
        lines.append(f"# next: {spec.next_considerations[0]}")
    elif spec.result_reading:
        lines.append(f"# read: {spec.result_reading[0]}")
    return tuple(lines)


_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")

#: Underscored operation names inside prose, e.g. "use group_split". Single-word
#: names are deliberately not matched: "fit" and "split" are ordinary English.
_IDENTIFIER_IN_PROSE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def _as_call(item: str) -> str:
    """Render a bare operation name as a call so mixed lists read consistently."""
    return f"session.{item}()" if _IDENTIFIER.match(item) else item


def _related_tools(spec: OperationSpec) -> tuple[str, ...]:
    """Name the neighbours, without repeating any the alternatives already named.

    The authored alternatives read as advice: "use ``group_split`` when
    entities define boundaries": while the derived neighbours are bare calls.
    Listing ``session.group_split()`` underneath that sentence adds a line and
    no information, so anything the advice already mentions is dropped.
    """
    provider_tools: list[str] = []
    for item in spec.prerequisites:
        provider_tools.extend(providers_for(item.key)[:2])
    concept_tools: list[str] = []
    for note in _linked_notes(spec):
        concept_tools.extend(note.buildml_tools)
    advice = " ".join(spec.alternatives)
    spoken = set(_IDENTIFIER_IN_PROSE.findall(advice))
    tools = _dedupe(tuple(provider_tools), tuple(concept_tools))
    rendered = tuple(
        _as_call(item) for item in tools if item != spec.name and item not in spoken
    )
    return (_dedupe(spec.alternatives, rendered))[:10]


def _learn_next(spec: OperationSpec) -> tuple[str, ...]:
    concept_reads = tuple(
        f"Concept '{note.key}': {note.title}" for note in _linked_notes(spec)
    )
    follow_on: list[str] = []
    for note in _linked_notes(spec):
        for key in note.next_concepts:
            other = CONCEPT_NOTES.get(key)
            if other is not None:
                follow_on.append(f"Concept '{other.key}': {other.title}")
    return _dedupe(spec.next_considerations, concept_reads, tuple(follow_on))[:8]


def _glossary(
    spec: OperationSpec,
    texts: tuple[str, ...],
    limit: int,
) -> tuple[GlossaryTerm, ...]:
    if limit <= 0:
        return ()
    declared: list[GlossaryTerm] = []
    seen: set[str] = set()
    for note in _linked_notes(spec):
        for entry in note.glossary:
            if entry.term.lower() not in seen:
                declared.append(entry)
                seen.add(entry.term.lower())
    detected = detect_terms(texts, limit=limit, exclude=seen)
    ordered = [*detected, *declared]
    return tuple(ordered[:limit])


def derive_primer(
    spec: OperationSpec,
    *,
    level: LearningLevel | str | None = LearningLevel.BEGINNER,
) -> OperationPrimer:
    """Build the beginner-first briefing for one operation specification.

    Each section prefers prose the operation authored for it and falls back to
    material already maintained elsewhere: the operation's kind supplies the
    framing, its prerequisites supply the "what must be true first" list, its
    parameters supply the knob readings, and its linked concept notes supply the
    analogy, the worked example, and the further reading. Nothing here is
    invented, so the primer cannot claim something the expert sections deny.

    Parameters
    ----------
    spec:
        The catalog entry to brief. Use :func:`primer_for` when you have a name
        rather than a specification.
    level:
        How much scaffolding to render. ``'beginner'`` (the default) includes
        the analogy and glossary and caps the parameter and pitfall lists;
        ``'advanced'`` drops the scaffolding and widens the lists. The level
        never changes which facts are stated.

    Returns
    -------
    ~buildml.explain.schemas.OperationPrimer
        The briefing, ready to render beside the expert explanation.

    Raises
    ------
    ValueError
        ``level`` is not one of the three reading levels.

    See Also
    --------
    primer_for : The cached form, taking an operation name.
    buildml.explain.learn : Teach the concept behind an operation.
    """
    tier = LearningLevel.coerce(level)
    plain = _plain_summary(spec)
    why = _why_it_exists(spec)
    steps = _steps(spec)
    reading = _how_to_read(spec)
    avoid = _when_not_to_use(spec)
    parameters = tuple(
        parameter_meaning(item)
        for item in _ordered_parameters(spec, _KEY_PARAMETER_LIMIT[tier])
    )
    return OperationPrimer(
        operation=spec.name,
        level=tier,
        plain_summary=plain,
        analogy=_analogy(spec) if tier is LearningLevel.BEGINNER else "",
        why_it_exists=why,
        steps=steps if tier is not LearningLevel.ADVANCED else steps[:3],
        prerequisites_in_plain_words=_prerequisites_in_plain_words(spec),
        when_to_use=_when_to_use(spec),
        when_not_to_use=avoid,
        key_parameters=parameters,
        what_changes=_what_changes(spec),
        how_to_read_the_result=reading,
        common_pitfalls=_pitfalls(spec, avoid, _PITFALL_LIMIT[tier]),
        glossary=_glossary(spec, (plain, why, *steps, *reading), _GLOSSARY_LIMIT[tier]),
        mini_example=_mini_example(spec),
        related_tools=_related_tools(spec),
        learn_next=_learn_next(spec),
    )


@cache
def primer_for(
    operation: str,
    *,
    level: LearningLevel | str | None = LearningLevel.BEGINNER,
) -> OperationPrimer:
    """Look up an operation by name and return its briefing, from cache.

    Derivation walks the catalog entry and every concept note it links, which is
    wasted work when the same operation is explained repeatedly: and it is,
    since the resolver calls this on every ``explain``. Results are immutable, so
    caching them per name and level is safe.

    Parameters
    ----------
    operation:
        A catalog operation name, such as ``'split'`` or ``'fit'``.
    level:
        ``'beginner'`` (the default), ``'intermediate'``, or ``'advanced'``.

    Returns
    -------
    ~buildml.explain.schemas.OperationPrimer
        The briefing for that operation at that level.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        No catalog operation has that name.
    ValueError
        ``level`` is not one of the three reading levels.

    Examples
    --------
    >>> from buildml.explain import primer_for
    >>> primer_for("split").operation
    'split'
    """
    return derive_primer(get_operation(operation), level=level)


__all__ = [
    "derive_primer",
    "parameter_meaning",
    "primer_for",
]
