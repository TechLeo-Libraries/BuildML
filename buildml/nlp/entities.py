"""Find the named things inside documents: dates, amounts, references, people.

Entity extraction pulls structured facts out of prose. It is what turns "invoice
INV-4821 for £340 was raised on 2024-03-11" into three fields you can query.

Two backends with opposite failure modes, and choosing between them is really a
question of which error you can tolerate.

The **rules** backend matches regular expressions and phrase lists. It is
deterministic, needs no download, and is precision-first: it finds email
addresses, URLs, IP addresses, phone numbers, monetary amounts, percentages,
dates, times, and reference identifiers, all of which have recognisable shapes.
It deliberately does not guess at free-form person or place names, because rules
cannot do that without a false-positive rate that makes the output useless.

The **spaCy** backend runs a statistical model that will find names it has never
seen — genuinely impossible for rules. In exchange it produces confident false
positives on text unlike its training data, and it needs a model download.

Overlapping matches are resolved by keeping the longest span, then by label
priority. That is what stops a monetary amount being shadowed by the bare number
inside it.
"""

from __future__ import annotations

import re
from collections import Counter

from buildml.core.errors import ValidationError
from buildml.data.dataset import Dataset
from buildml.data.splits import SplitPlan
from buildml.nlp.catalog import resolve_entity_backend
from buildml.nlp.features import PartitionOrAll, documents_for, resolve_text_column
from buildml.nlp.lexicons import RULE_ENTITY_LABELS, RULE_ENTITY_PATTERNS
from buildml.nlp.results import Entity, NlpEntityResult

MAX_TOP_MENTIONS = 10

_COMPILED_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern)) for label, pattern in RULE_ENTITY_PATTERNS
)
# Earlier labels win ties so a MONEY match is not shadowed by a bare number.
_RULE_PRIORITY: dict[str, int] = {label: index for index, label in enumerate(RULE_ENTITY_LABELS)}


def compile_gazetteers(
    gazetteers: dict[str, list[str]] | None,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Turn phrase lists into matchers for terms only you know about.

    A gazetteer is a list of known names — your products, your regions, your
    internal system identifiers. No general model knows them, and no regular
    expression describes them, but you can enumerate them, and enumeration is
    enough.

    Longer phrases are tried first, so "New York City" wins over "New York"
    rather than being split by it. Matching is case-insensitive and bounded to
    whole words, so "cat" cannot match inside "catalogue".

    Parameters
    ----------
    gazetteers:
        Label to phrases. Labels are upper-cased for consistency with the
        built-in ones; duplicate phrases within a label are collapsed.
        ``None`` or empty returns no matchers.

    Returns
    -------
    tuple
        ``(label, compiled_pattern)`` pairs, ready for
        :func:`extract_rule_entities`.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        A label was given with no usable phrases. Silently ignoring it would
        leave you wondering why that label never appears in the output.

    Notes
    -----
    Gazetteers are exact matching, so a misspelling in a document will not be
    found. For names with predictable variation, include the variants.

    Examples
    --------
    >>> rules = compile_gazetteers({"product": ["Widget Pro", "Widget"]})
    >>> len(rules)
    1

    See Also
    --------
    extract_rule_entities : Apply these alongside the built-in patterns.
    """
    if not gazetteers:
        return ()
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for label, phrases in gazetteers.items():
        cleaned = sorted(
            {str(phrase).strip() for phrase in phrases if str(phrase).strip()},
            key=len,
            reverse=True,
        )
        if not cleaned:
            raise ValidationError(
                f"Gazetteer for label {label!r} is empty; remove it or add phrases."
            )
        alternation = "|".join(re.escape(phrase) for phrase in cleaned)
        compiled.append(
            (
                str(label).upper(),
                re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE),
            )
        )
    return tuple(compiled)


def _resolve_overlaps(spans: list[Entity]) -> list[Entity]:
    """Keep the longest, then highest-priority, span for each overlapping region."""
    ordered = sorted(
        spans,
        key=lambda item: (
            item.start,
            -(item.end - item.start),
            _RULE_PRIORITY.get(item.label, len(_RULE_PRIORITY)),
        ),
    )
    kept: list[Entity] = []
    last_end = -1
    for span in ordered:
        if span.start < last_end:
            continue
        kept.append(span)
        last_end = span.end
    return kept


def extract_rule_entities(
    document: str,
    *,
    labels: tuple[str, ...] | None = None,
    gazetteers: tuple[tuple[str, re.Pattern[str]], ...] = (),
) -> tuple[Entity, ...]:
    """Find structured mentions in one string, using patterns and phrase lists.

    Runs the built-in patterns and any gazetteers over the text, then resolves
    overlaps so each region of the document yields at most one mention.

    Parameters
    ----------
    document:
        The raw text. Offsets in the result index into this string exactly as
        passed, which is what makes redaction and highlighting possible.
    labels:
        Restrict to these labels. Worth using when you only care about one kind
        of thing — scanning for personal data, say — since it also avoids the
        cost of patterns you will discard.
    gazetteers:
        Compiled phrase matchers from :func:`compile_gazetteers`.

    Returns
    -------
    tuple of ~buildml.nlp.results.Entity
        Non-overlapping mentions in document order, each tagged ``'rules'`` or
        ``'gazetteer'`` by origin.

    Notes
    -----
    **Overlaps resolve to the longest span first**, then by label priority. So
    "£1,200.00" is one monetary mention rather than a currency symbol plus a
    number, and a gazetteer phrase containing a date is not fragmented by it.

    **Precision over recall, deliberately.** These patterns will miss unusual
    formats — an unfamiliar date layout, a phone number written oddly. What
    they find is almost always right, which is the property that makes the
    output usable without review.

    See Also
    --------
    extract_entities : The dataset-level entry point, with corpus aggregates.
    """
    text = "" if document is None else str(document)
    wanted = None if labels is None else {str(item).upper() for item in labels}
    found: list[Entity] = []
    for label, pattern in _COMPILED_RULES:
        if wanted is not None and label not in wanted:
            continue
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if not value:
                continue
            found.append(
                Entity(
                    text=value,
                    label=label,
                    start=int(match.start()),
                    end=int(match.start()) + len(value),
                    source="rules",
                )
            )
    for label, pattern in gazetteers:
        if wanted is not None and label not in wanted:
            continue
        for match in pattern.finditer(text):
            found.append(
                Entity(
                    text=match.group(0),
                    label=label,
                    start=int(match.start()),
                    end=int(match.end()),
                    source="gazetteer",
                )
            )
    return tuple(_resolve_overlaps(found))


def extract_entities(
    dataset: Dataset,
    split_plan: SplitPlan | None,
    *,
    partition: PartitionOrAll = "test",
    backend: str | None = "rules",
    text_column: str | None = None,
    labels: list[str] | None = None,
    gazetteers: dict[str, list[str]] | None = None,
    spacy_model: str = "en_core_web_sm",
    max_documents: int = 25,
    batch_size: int = 32,
) -> NlpEntityResult:
    """Extract entity mentions from a partition's documents.

    Scans each document, records every mention with its exact character span,
    and aggregates what was found across the corpus.

    Parameters
    ----------
    dataset:
        The dataset holding the documents.
    split_plan:
        The split defining partitions. Required unless ``partition`` is
        ``'all'``.
    partition:
        Which rows to scan. Extraction learns nothing, so any partition is
        safe.
    backend:
        ``'rules'`` for deterministic pattern matching, or ``'spacy'`` for
        statistical recognition that generalises to unseen names.
    text_column:
        Which column holds the documents. Inferred when omitted.
    labels:
        Restrict to these entity types. For the rules backend, a label that
        neither a built-in pattern nor a gazetteer produces is an error rather
        than an empty result — otherwise a typo looks like "nothing found".
    gazetteers:
        Label to phrases for domain names no general model knows. See
        :func:`compile_gazetteers`.
    spacy_model:
        Which spaCy pipeline to load. Match it to your text's language: an
        English model on French text finds very little and does not say so.
    max_documents:
        Cap on documents kept in the per-document output. Corpus aggregates
        still cover everything scanned.
    batch_size:
        How many documents spaCy processes at once. Larger batches are faster
        and use more memory.

    Returns
    -------
    ~buildml.nlp.results.NlpEntityResult
        Per-document mentions with spans, corpus label counts, the most
        frequent surface form per label, and the caveats.

    Raises
    ------
    ~buildml.core.errors.ValidationError
        The backend is unknown; ``max_documents`` is negative; a requested
        label cannot be produced; a gazetteer is empty; the text column cannot
        be resolved; or the partition is empty.
    ~buildml.core.errors.MissingExtraError
        spaCy was requested without ``buildml[nlp-industry]``.

    Notes
    -----
    **Check ``top_mentions`` before trusting the counts.** It is the fastest
    way to spot a pattern firing on the wrong thing — an ``ORG`` list full of
    ordinary sentence openers tells you immediately that something is wrong,
    where a label count alone would not.

    **spaCy's labels are remapped onto the same vocabulary** as the rules
    backend, so results are comparable across backends. The remapping is
    approximate: spaCy's categories do not correspond one-to-one.

    **Spans index the raw document**, before any normalisation, which is what
    makes redaction against the original text correct.

    Examples
    --------
    >>> result = extract_entities(  # doctest: +SKIP
    ...     dataset, split_plan, gazetteers={"product": ["Widget Pro"]}
    ... )
    >>> result.label_counts  # doctest: +SKIP

    See Also
    --------
    extract_rule_entities : Scan a single string.
    compile_gazetteers : Build matchers for your own names.
    """
    backend_key = resolve_entity_backend(backend)
    if max_documents < 0:
        raise ValidationError("max_documents must be >= 0.")

    column = resolve_text_column(dataset, text_column)
    documents, frame = documents_for(
        dataset, split_plan, partition, column, operation="extract_entities"
    )
    wanted = tuple(str(item).upper() for item in labels) if labels else None
    if wanted and backend_key == "rules":
        unknown = [
            item
            for item in wanted
            if item not in set(RULE_ENTITY_LABELS)
            and item not in {str(key).upper() for key in (gazetteers or {})}
        ]
        if unknown:
            raise ValidationError(
                f"labels={unknown} are not produced by the rules backend. "
                f"Built-in labels: {list(RULE_ENTITY_LABELS)}. Add a gazetteer or "
                "use backend='spacy'."
            )

    compiled_gazetteers = compile_gazetteers(gazetteers)
    warnings: list[str] = []
    disclosures: list[str] = []

    if backend_key == "rules":
        per_document = [
            extract_rule_entities(
                document, labels=wanted, gazetteers=compiled_gazetteers
            )
            for document in documents
        ]
        disclosures.extend(
            [
                f"Rules backend: {len(RULE_ENTITY_LABELS)} regex label(s) "
                f"({list(RULE_ENTITY_LABELS)}) plus "
                f"{len(compiled_gazetteers)} gazetteer label(s).",
                "Precision-first by design: free-form person and place names are "
                "not guessed. Use backend='spacy' for statistical NER.",
                "Overlapping matches are resolved by longest span, then rule order.",
            ]
        )
    else:
        from buildml.nlp.adapters.spacy_pipeline import extract_spacy_entities

        raw = extract_spacy_entities(
            documents,
            model=spacy_model,
            labels=wanted,
            batch_size=batch_size,
        )
        per_document = [
            tuple(
                Entity(text=text, label=label, start=start, end=end, source="spacy")
                for text, label, start, end in mentions
            )
            for mentions in raw
        ]
        if compiled_gazetteers:
            merged: list[tuple[Entity, ...]] = []
            for document, mentions in zip(documents, per_document, strict=False):
                extra = [
                    span
                    for span in extract_rule_entities(
                        document, labels=wanted, gazetteers=compiled_gazetteers
                    )
                    if span.source == "gazetteer"
                ]
                merged.append(tuple(_resolve_overlaps([*mentions, *extra])))
            per_document = merged
        disclosures.extend(
            [
                f"spaCy backend: pipeline {spacy_model!r}; ONTONOTES labels are "
                "remapped onto BuildML's label vocabulary.",
                "The spaCy model was trained outside this Session; its training "
                "data is not covered by the Session split.",
            ]
        )

    label_counter: Counter[str] = Counter()
    mention_counter: dict[str, Counter[str]] = {}
    total = 0
    for mentions in per_document:
        for span in mentions:
            label_counter[span.label] += 1
            mention_counter.setdefault(span.label, Counter())[span.text.lower()] += 1
            total += 1

    if total == 0:
        warnings.append(
            "No entities were extracted. Confirm the text column, add gazetteers, "
            "or install buildml[nlp-industry] for statistical NER."
        )
    limit = min(max_documents, len(per_document)) if max_documents else 0
    if limit and len(per_document) > limit:
        warnings.append(
            f"Returned per-document mentions for the first {limit} of "
            f"{len(per_document)} documents (raise max_documents for more); "
            "label_counts still cover the whole partition."
        )

    return NlpEntityResult(
        partition=str(partition),
        backend=backend_key,
        n_rows=len(documents),
        n_entities=total,
        label_counts={key: int(label_counter[key]) for key in sorted(label_counter)},
        document_entities=tuple(per_document[:limit]) if limit else (),
        document_row_labels=tuple(frame.index[:limit]) if limit else (),
        top_mentions={
            label: tuple(counter.most_common(MAX_TOP_MENTIONS))
            for label, counter in sorted(mention_counter.items())
        },
        disclosures=tuple(disclosures),
        warnings=tuple(warnings),
    )


__all__ = [
    "MAX_TOP_MENTIONS",
    "compile_gazetteers",
    "extract_entities",
    "extract_rule_entities",
]
