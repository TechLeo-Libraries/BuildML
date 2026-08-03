"""Teaching-surface sync: Session API index, catalog, and AI tool drift checks.

Session public callables are the source of truth for *which* operations exist.
Human overlays under ``buildml.explain.overlays`` supply teaching prose.
``operation_index.json`` is generated from Session signatures and checked in CI.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

INDEX_SCHEMA_VERSION = 1
GENERATED_DIR = Path(__file__).resolve().parent / "generated"
OPERATION_INDEX_PATH = GENERATED_DIR / "operation_index.json"

# AI tools are an intentional allowlist, not a full Session mirror.
# Every tool must still resolve to a real Session method / catalog op.
# Teaching-critical Phase C + Pass R surfaces must appear in the default registry.
REQUIRED_AI_TOOL_SESSION_METHODS: frozenset[str] = frozenset(
    {
        "rag_retrieve",
        "rag_generate",
        "rag_ingest_corpus",
        "rag_embed_and_index",
        "make_torch_loaders",
        "make_text_torch_loaders",
        "fit_torch",
        "evaluate_torch",
        "cross_validate_torch",
        "fit",
        "evaluate",
        "fit_clusters",
        "evaluate_clusters",
        "fit_voting",
        "fit_stacking",
        "fit_blending",
        "evaluate_ensemble",
        "run_automl",
        "evaluate_automl",
        "fit_forecast",
        "evaluate_forecast",
        "fit_anomaly",
        "evaluate_anomaly",
        "fit_semisupervised",
        "evaluate_semisupervised",
        "fit_ssl_pretext",
        "finetune_ssl_head",
        "evaluate_ssl",
        "fit_active_learner",
        "suggest_query",
        "evaluate_active_learning",
        "fit_online",
        "partial_fit_online",
        "evaluate_online",
        "fit_multitask",
        "evaluate_multitask",
        "fit_metalearning",
        "adapt_to_task",
        "evaluate_metalearning",
        "fit_federated",
        "evaluate_federated",
        "fit_probabilistic",
        "evaluate_probabilistic",
        "predict_interval",
        "declare_causal_assumptions",
        "fit_causal",
        "evaluate_causal",
        "estimate_causal",
        "set_graph",
        "fit_graph",
        "evaluate_graph",
        "predict_graph",
        "fit_symbolic",
        "evaluate_symbolic",
        "predict_symbolic",
        "fit_neuro_symbolic",
        "evaluate_neuro_symbolic",
        "predict_neuro_symbolic",
        "profile_text_corpus",
        "detect_language",
        "fit_text_classifier",
        "predict_text",
        "evaluate_text_classifier",
        "interpret_text_prediction",
        "fit_topics",
        "assign_topics",
        "extract_keyphrases",
        "analyze_sentiment",
        "extract_entities",
        "summarize_text",
        "fit_cbr",
        "retrieve_cases",
        "evaluate_cbr",
        "predict_cbr",
        "retain_cbr",
        "fit_imitation",
        "predict_imitation_action",
        "evaluate_imitation",
        "fit_rl",
        "act_rl",
        "evaluate_rl",
        "fit_tda",
        "transform_tda",
        "predict_tda",
        "evaluate_tda",
        "fit_recommender",
        "recommend",
        "evaluate_recommender",
        "fit_ranker",
        "rank",
        "evaluate_ranker",
        "fit_kg",
        "score_triples",
        "predict_links",
        "query_kg",
        "evaluate_kg",
        "fit_decision_policy",
        "apply_decisions",
        "evaluate_decisions",
        "fit_synthesizer",
        "sample_synthetic",
        "evaluate_synthetic",
        "evolutionary_search",
        "split",
        "set_roles",
        "explain",
        "workflow",
        "eda",
        "walkthrough",
        "load_pretrained_backbone",
        "pack_torchserve",
        "prepare_tensorrt_export",
        "emit_k8s_ddp_job",
        "domain_adapt_speech_torch",
        "attach_backbone_head",
        "evaluate_asr",
        "emit_k8s_serve_deployment",
    }
)

# Tools that intentionally have no Session method (builtins / status helpers).
AI_TOOL_BUILTINS: frozenset[str] = frozenset({"describe_dataset", "ai_status"})

# Session methods that must NOT appear in the AI tool registry.
# serve_bundle starts a network listener: CLI / Session-primary only.
EXPLICITLY_NON_AI_SESSION_METHODS: frozenset[str] = frozenset({"serve_bundle"})


@dataclass(frozen=True, slots=True)
class ParameterIndex:
    """One parameter of a Session method, captured as comparable text.

    Everything is a string, including the annotation and the default. That is
    deliberate: the index is written to JSON and diffed against a regenerated
    version, and comparing live type objects across Python versions is not
    stable enough to build a CI gate on. Text is.

    Attributes
    ----------
    name:
        The parameter name.
    kind:
        How it can be passed: ``'POSITIONAL_OR_KEYWORD'``, ``'KEYWORD_ONLY'``,
        ``'VAR_KEYWORD'``, and so on, from ``inspect``. Part of the contract:
        making a positional parameter keyword-only breaks callers.
    annotation:
        The type annotation as source text, empty when unannotated.
    default:
        The default's ``repr``, or ``None`` when there is no default. Note that
        a parameter defaulting to ``None`` records the string ``'None'``, so the
        two cases stay distinguishable.
    required:
        Whether a caller must supply it. ``*args`` and ``**kwargs`` are never
        required.

    See Also
    --------
    OperationIndexEntry : The operation these belong to.
    """

    name: str
    kind: str
    annotation: str
    default: str | None
    required: bool

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dict for the JSON index.

        No conversion is needed: every field is already a string or a bool,
        which is the reason they were stored that way. The method exists so the
        index writer can treat parameters and operations uniformly.

        Returns
        -------
        dict
            The five fields, all JSON-safe already.
        """
        return {
            "name": self.name,
            "kind": self.kind,
            "annotation": self.annotation,
            "default": self.default,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class OperationIndexEntry:
    """One Session operation as the teaching surface sees it.

    A machine-readable description of a public method: its name, its
    parameters, and the first line of its docstring. This is what the catalog,
    the AI tool registry, and the documentation are checked against, so an
    operation cannot quietly change shape while its teaching content goes on
    describing the old one.

    Attributes
    ----------
    name:
        The method name, and the key everything else references.
    qualname:
        The qualified name, which shows where an inherited method came from.
    doc_summary:
        The first non-empty docstring line. Extracted so the index can be read
        as a listing of what a Session does.
    parameters:
        The parameters, in declaration order, excluding ``self`` and ``cls``.
    is_classmethod:
        Whether it is a classmethod: ``Session.from_csv`` and similar.
    is_staticmethod:
        Whether it is a staticmethod.

    See Also
    --------
    build_operation_index : What produces these.
    """

    name: str
    qualname: str
    doc_summary: str
    parameters: tuple[ParameterIndex, ...]
    is_classmethod: bool
    is_staticmethod: bool

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dict, converting the parameters too.

        The parameter tuple becomes a list of dicts, since JSON has no tuples
        and reading one back would produce a list regardless. Doing the
        conversion here keeps the written form and the round-tripped form
        identical, so a comparison between them never reports a false
        difference.

        Returns
        -------
        dict
            The entry with ``parameters`` as a list of dicts.
        """
        return {
            "name": self.name,
            "qualname": self.qualname,
            "doc_summary": self.doc_summary,
            "parameters": [item.to_dict() for item in self.parameters],
            "is_classmethod": self.is_classmethod,
            "is_staticmethod": self.is_staticmethod,
        }


@dataclass(slots=True)
class DriftReport:
    """Structured teaching-surface drift findings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether anything blocking was found.

        Errors fail the build; warnings do not. A check that could not run :
        because an optional extra is missing: records a warning, so a partial
        environment does not block a merge over something it cannot verify.

        Returns
        -------
        bool
            ``True`` when there are no errors, regardless of warnings.
        """
        return not self.errors

    def raise_if_failed(self) -> None:
        """Raise with every error listed, or return quietly.

        For use as a test assertion. Every error is included in the message
        rather than only the first, because drift usually comes in batches: one
        renamed parameter shows up in the catalog check, the index check, and
        the AI tool check at once, and fixing them one CI run at a time is
        needlessly slow.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If there are errors. The message lists them, one per line.

        Notes
        -----
        **Warnings are not raised** and not shown here. Inspect ``warnings``
        directly if a skipped check matters.
        """
        if self.errors:
            joined = "\n".join(f"- {item}" for item in self.errors)
            raise AssertionError(f"Teaching-surface drift detected:\n{joined}")


def public_session_operations(session_cls: type | None = None) -> dict[str, Callable[..., Any]]:
    """List the Session's public methods, which define what operations exist.

    The Session class is the source of truth for the operation surface. Every
    other list: the catalog, the AI tool registry, the generated index: is
    checked against this one rather than maintained beside it.

    Public means not starting with an underscore. A simple rule, and one that
    means adding a public method to Session automatically obliges you to add its
    teaching content, because the parity check will fail until you do.

    Parameters
    ----------
    session_cls:
        The class to inspect. Defaults to the real
        :class:`~buildml.session.Session`; pass a stand-in for testing.

    Returns
    -------
    dict
        Method name to callable. Includes inherited methods and properties that
        happen to be callable.

    Notes
    -----
    **Renaming a public method is a breaking change here as well as for
    users**: the catalog, the index, and possibly a tool registry all reference
    the old name and will fail until updated.

    See Also
    --------
    build_operation_index : Turning these into the index.
    check_session_catalog_parity : The check this feeds.
    """
    if session_cls is None:
        from buildml.session import Session

        session_cls = Session
    return {
        name: member
        for name, member in inspect.getmembers(session_cls, predicate=callable)
        if not name.startswith("_")
    }


def _annotation_str(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    if isinstance(annotation, str):
        return annotation
    return inspect.formatannotation(annotation)


def _default_str(default: Any) -> str | None:
    if default is inspect.Parameter.empty:
        return None
    try:
        return repr(default)
    except Exception:
        return type(default).__name__


def _doc_summary(func: Callable[..., Any]) -> str:
    doc = inspect.getdoc(func) or ""
    for line in doc.splitlines():
        text = line.strip()
        if text:
            return text
    return ""


def build_operation_index(session_cls: type | None = None) -> dict[str, Any]:
    """Read the live Session's signatures into a comparable index.

    Introspects every public method and records its parameters, kinds,
    defaults, and docstring summary. The result is the current shape of the API,
    and comparing it against the checked-in copy is how signature drift gets
    caught before it reaches a release.

    A method whose signature cannot be read: which happens with some decorated
    callables: is indexed with no parameters rather than skipped. Present and
    incomplete is more useful than absent, since absence would look like the
    method had been removed.

    Parameters
    ----------
    session_cls:
        The class to index. Defaults to the real ``Session``.

    Returns
    -------
    dict
        ``schema_version``, ``source``, ``n_operations``, and ``operations`` :
        the entries keyed by name, sorted for a stable diff.

    Notes
    -----
    **Sorting is what makes this diffable.** Dict ordering would otherwise
    follow ``inspect``'s enumeration, and an unrelated edit could reshuffle the
    file and bury the real change.

    **``self`` and ``cls`` are excluded**, so the parameters listed are the ones
    a caller actually passes.

    See Also
    --------
    write_operation_index : Persisting this.
    check_operation_index_fresh : Comparing it against the checked-in copy.
    """
    if session_cls is None:
        from buildml.session import Session

        session_cls = Session
    operations = public_session_operations(session_cls)
    entries: dict[str, Any] = {}
    for name in sorted(operations):
        member = operations[name]
        raw = inspect.getattr_static(session_cls, name)
        is_classmethod = isinstance(raw, classmethod)
        is_staticmethod = isinstance(raw, staticmethod)
        try:
            signature = inspect.signature(member)
        except (TypeError, ValueError):
            signature = inspect.Signature()
        parameters: list[ParameterIndex] = []
        for param_name, param in signature.parameters.items():
            if param_name in {"self", "cls"}:
                continue
            parameters.append(
                ParameterIndex(
                    name=param_name,
                    kind=param.kind.name,
                    annotation=_annotation_str(param.annotation),
                    default=_default_str(param.default),
                    required=param.default is inspect.Parameter.empty
                    and param.kind
                    not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ),
                )
            )
        entry = OperationIndexEntry(
            name=name,
            qualname=getattr(member, "__qualname__", name),
            doc_summary=_doc_summary(member),
            parameters=tuple(parameters),
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
        )
        entries[name] = entry.to_dict()
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "source": "buildml.session.Session",
        "n_operations": len(entries),
        "operations": entries,
    }


def write_operation_index(
    path: Path | None = None,
    *,
    session_cls: type | None = None,
) -> Path:
    """Write the index to disk, formatted so diffs stay readable.

    Regenerates the checked-in ``operation_index.json``. Run it after changing a
    public Session signature; the freshness check fails until you do, and its
    error message says so.

    Indented and key-sorted, with a trailing newline. That formatting is not
    cosmetic: an index written compactly would show every signature change as
    one enormous modified line, and reviewing it would be impossible.

    Parameters
    ----------
    path:
        Where to write. Defaults to the packaged location. Parent directories
        are created.
    session_cls:
        The class to index. Defaults to the real ``Session``.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    OSError
        If the file cannot be written.

    Notes
    -----
    **Commit the result.** The point of the file is that CI can compare against
    it; an uncommitted regeneration fixes the check locally and nowhere else.

    Examples
    --------
    From the repository root::

        python scripts/sync_teaching_surface.py --write

    See Also
    --------
    check_operation_index_fresh : The check this satisfies.
    """
    destination = path or OPERATION_INDEX_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_operation_index(session_cls)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_operation_index(path: Path | None = None) -> dict[str, Any]:
    """Read the checked-in index back from JSON.

    The counterpart to :func:`write_operation_index`. Used by the freshness
    check, and by anything that wants the operation surface without importing
    Session: documentation tooling, for instance, which can then describe the
    API without paying for the import.

    Parameters
    ----------
    path:
        The index file. Defaults to the packaged location.

    Returns
    -------
    dict
        The index as written, with ``schema_version``, ``source``,
        ``n_operations``, and ``operations``.

    Raises
    ------
    FileNotFoundError
        If the file is absent.
    json.JSONDecodeError
        If it is not valid JSON.

    Notes
    -----
    **This may be stale.** It reflects the last regeneration, not the live
    class. :func:`check_operation_index_fresh` is what tells you which.

    See Also
    --------
    write_operation_index : Producing the file.
    """
    from typing import cast

    source = path or OPERATION_INDEX_PATH
    return cast(dict[str, Any], json.loads(source.read_text(encoding="utf-8")))


def _catalog_names(catalog: Mapping[str, Any] | None = None) -> set[str]:
    if catalog is None:
        from buildml.explain.catalog import OPERATION_CATALOG

        catalog = OPERATION_CATALOG
    return set(catalog)


def check_session_catalog_parity(
    *,
    session_cls: type | None = None,
    catalog: Mapping[str, Any] | None = None,
) -> DriftReport:
    """Check that the catalog covers exactly the Session's public operations.

    Both directions are errors, for different reasons. A Session method missing
    from the catalog is an operation with no teaching content: it exists, users
    can call it, and nothing explains it. A catalog entry with no Session method
    is teaching content for something that does not exist, which is worse: a
    reader follows the documentation and gets an ``AttributeError``.

    Parameters
    ----------
    session_cls:
        The class to check. Defaults to the real ``Session``.
    catalog:
        The operation catalog. Defaults to the real one.

    Returns
    -------
    DriftReport
        Errors naming the operations on each side of the mismatch. Clean when
        the two sets match.

    Notes
    -----
    **This checks names only.** Whether the catalog's parameters match the
    signature is :func:`check_catalog_parameters_vs_signatures`.

    See Also
    --------
    check_teaching_surface : Running this with the rest.
    """
    report = DriftReport()
    session_names = set(public_session_operations(session_cls))
    catalog_names = _catalog_names(catalog)
    missing = sorted(session_names - catalog_names)
    extra = sorted(catalog_names - session_names)
    if missing:
        report.errors.append(f"Catalog missing Session operations: {missing}")
    if extra:
        report.errors.append(f"Catalog has unknown operations (not on Session): {extra}")
    return report


def check_catalog_parameters_vs_signatures(
    *,
    session_cls: type | None = None,
    catalog: Mapping[str, Any] | None = None,
) -> DriftReport:
    """Check that documented parameters and real parameters are the same set.

    Both directions again, and both are real failures. A catalog parameter that
    does not exist means the documentation describes an argument nobody can
    pass. A signature parameter missing from the catalog means an argument
    exists and nothing explains it: which is how a parameter ends up
    permanently undiscovered.

    ``self`` and ``cls`` are excluded on both sides.

    Parameters
    ----------
    session_cls:
        The class to check. Defaults to the real ``Session``.
    catalog:
        The operation catalog. Defaults to the real one.

    Returns
    -------
    DriftReport
        Errors naming the operation and the offending parameters. A method whose
        signature cannot be read produces a warning rather than an error, since
        the check could not run.

    Notes
    -----
    **Names only.** Types, defaults, and ordering are not compared here; the
    generated index covers required flags and kinds.

    See Also
    --------
    check_session_catalog_parity : Whether the operations themselves line up.
    """
    report = DriftReport()
    if session_cls is None:
        from buildml.session import Session

        session_cls = Session
    if catalog is None:
        from buildml.explain.catalog import OPERATION_CATALOG

        catalog = OPERATION_CATALOG
    for name, spec in catalog.items():
        member = getattr(session_cls, name, None)
        if member is None or not callable(member):
            report.errors.append(f"{name}: not a public Session callable")
            continue
        try:
            signature = inspect.signature(member)
        except (TypeError, ValueError):
            report.warnings.append(f"{name}: signature unavailable for parameter check")
            continue
        available = {
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.name not in {"self", "cls"}
            and parameter.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        }
        documented = {parameter.name for parameter in getattr(spec, "parameters", ())}
        unknown = sorted(documented - available)
        missing = sorted(available - documented)
        if unknown:
            report.errors.append(f"{name}: catalog parameters not in signature: {unknown}")
        if missing:
            report.errors.append(
                f"{name}: signature parameters missing from catalog: {missing}"
            )
    return report


def check_operation_index_fresh(
    *,
    path: Path | None = None,
    session_cls: type | None = None,
) -> DriftReport:
    """Check the committed index still matches the code it describes.

    The gate that keeps the generated file honest. If a signature changed and
    nobody regenerated, this fails and its message names the command to run.

    Parameter names, required flags, and kinds are compared; annotations and
    defaults are not. That is a considered line. Changing a parameter from
    required to optional, or from positional to keyword-only, changes what
    callers can do and must be visible. Tightening an annotation from ``Any`` to
    ``str | None`` does not break anyone, and failing CI over it would train
    people to regenerate reflexively without reading the diff.

    Parameters
    ----------
    path:
        The index file. Defaults to the packaged location.
    session_cls:
        The class to compare against. Defaults to the real ``Session``.

    Returns
    -------
    DriftReport
        Errors for a missing file, a schema version mismatch, operations present
        on one side only, or a parameter contract that moved.

    Notes
    -----
    **The remedy is in the error message**: run the sync script and commit the
    result.

    See Also
    --------
    write_operation_index : The fix.
    """
    report = DriftReport()
    source = path or OPERATION_INDEX_PATH
    if not source.is_file():
        report.errors.append(f"Missing generated operation index: {source}")
        return report
    on_disk = load_operation_index(source)
    expected = build_operation_index(session_cls)
    if on_disk.get("schema_version") != INDEX_SCHEMA_VERSION:
        report.errors.append(
            f"operation_index schema_version {on_disk.get('schema_version')!r} "
            f"!= {INDEX_SCHEMA_VERSION}"
        )
    disk_ops = on_disk.get("operations") or {}
    expected_ops = expected["operations"]
    if set(disk_ops) != set(expected_ops):
        missing = sorted(set(expected_ops) - set(disk_ops))
        extra = sorted(set(disk_ops) - set(expected_ops))
        if missing:
            report.errors.append(f"operation_index missing Session ops: {missing}")
        if extra:
            report.errors.append(f"operation_index has stale ops: {extra}")
    # Compare parameter names / required flags (stable teaching contract).
    for name in sorted(set(disk_ops) & set(expected_ops)):
        disk_params = {
            item["name"]: (item.get("required"), item.get("kind"))
            for item in disk_ops[name].get("parameters") or []
        }
        live_params = {
            item["name"]: (item.get("required"), item.get("kind"))
            for item in expected_ops[name].get("parameters") or []
        }
        if disk_params != live_params:
            report.errors.append(
                f"operation_index[{name}] parameters drifted; "
                "run: python scripts/sync_teaching_surface.py --write"
            )
    return report


def check_ai_tools_vs_catalog(
    *,
    session_cls: type | None = None,
    catalog: Mapping[str, Any] | None = None,
) -> DriftReport:
    """Check the AI tool registry against Session and the catalog, both ways.

    The AI tools are an allowlist, not a mirror. Not every Session method should
    be callable by an agent, and the registry is curated: so unlike the catalog
    this is not checked for exact parity. What is checked is that every listed
    tool resolves to something real, and that two specific lists are respected.

    ``REQUIRED_AI_TOOL_SESSION_METHODS`` names the operations an agent must be
    able to reach. Without this, a refactor can quietly drop a tool and the
    agent simply becomes less capable, with no failure anywhere.

    ``EXPLICITLY_NON_AI_SESSION_METHODS`` names operations an agent must not
    reach. ``serve_bundle`` is the example: it opens a network listener, and an
    agent deciding on its own to start a server is not a decision anyone
    delegated. Those methods still need catalog entries, because a human calling
    them deserves documentation.

    Parameters
    ----------
    session_cls:
        The class to check. Defaults to the real ``Session``.
    catalog:
        The operation catalog. Defaults to the real one.

    Returns
    -------
    DriftReport
        Errors for a tool pointing at a missing method or catalog operation, a
        required method absent from the registry, a forbidden method present in
        it, or a forbidden method missing from Session or the catalog. Builtin
        tools that unexpectedly set ``session_method`` produce a warning.

    Notes
    -----
    **Builtins are handled separately.** They do not map to Session methods, so
    only their catalog references are checked.

    See Also
    --------
    buildml.ai.tools.build_default_registry : The registry being checked.
    """
    report = DriftReport()
    if session_cls is None:
        from buildml.session import Session

        session_cls = Session
    catalog_names = _catalog_names(catalog)
    session_names = set(public_session_operations(session_cls))
    from buildml.ai.tools import build_default_registry

    registry = build_default_registry()
    tool_session_methods: set[str] = set()
    for tool in registry.tools:
        if tool.name in AI_TOOL_BUILTINS:
            if tool.session_method is not None:
                report.warnings.append(
                    f"AI builtin tool {tool.name!r} unexpectedly sets session_method"
                )
            if tool.catalog_operation and tool.catalog_operation not in catalog_names:
                report.errors.append(
                    f"AI tool {tool.name!r} catalog_operation "
                    f"{tool.catalog_operation!r} not in catalog"
                )
            continue
        method = tool.session_method or tool.name
        tool_session_methods.add(method)
        if method not in session_names:
            report.errors.append(f"AI tool {tool.name!r} maps to missing Session.{method}")
        catalog_op = tool.catalog_operation or method
        if catalog_op not in catalog_names:
            report.errors.append(
                f"AI tool {tool.name!r} catalog_operation {catalog_op!r} not in catalog"
            )
    missing_required = sorted(REQUIRED_AI_TOOL_SESSION_METHODS - tool_session_methods)
    if missing_required:
        report.errors.append(
            "Default AI registry missing teaching-critical Session methods: "
            f"{missing_required}"
        )
    leaked_non_ai = sorted(EXPLICITLY_NON_AI_SESSION_METHODS & tool_session_methods)
    if leaked_non_ai:
        report.errors.append(
            "Default AI registry must not include CLI/Session-primary methods: "
            f"{leaked_non_ai}"
        )
    for name in sorted(EXPLICITLY_NON_AI_SESSION_METHODS):
        if name not in session_names:
            report.errors.append(
                f"EXPLICITLY_NON_AI_SESSION_METHODS lists missing Session.{name}"
            )
        elif name not in catalog_names:
            report.errors.append(
                f"EXPLICITLY_NON_AI Session.{name} missing from catalog "
                "(still needs teaching overlay)"
            )
    return report


def check_dashboard_teaching_concepts() -> DriftReport:
    """Check that the studio's concept links all point at real glossary entries.

    Teaching panels offer concept chips: "what is a variance inflation factor",
    "what is a leakage boundary": and each is a key into ``CONCEPT_NOTES``. A
    renamed or removed note leaves a chip pointing nowhere, and the failure
    surfaces as a dead link in a UI that a developer may not open.

    Returns
    -------
    DriftReport
        An error listing every unknown key. When the dashboard extra is not
        installed the check cannot run, and that is recorded as a warning rather
        than an error, so a minimal environment does not fail CI over something
        it has no way to verify.

    Notes
    -----
    **The studios are built against an empty report**, which is enough to
    enumerate the concept keys without needing real analysis output.

    See Also
    --------
    buildml.explain.concepts : The glossary being referenced.
    """
    report = DriftReport()
    from buildml.explain.concepts import CONCEPT_NOTES

    try:
        from buildml.dashboard.teaching import build_teaching_studios
    except Exception as exc:  # pragma: no cover - optional dashboard extra
        report.warnings.append(f"dashboard teaching import skipped: {exc}")
        return report

    studios = build_teaching_studios({})
    unknown: set[str] = set()
    for studio in studios.values():
        for key in studio.get("concepts") or ():
            if key not in CONCEPT_NOTES:
                unknown.add(str(key))
    if unknown:
        report.errors.append(
            "dashboard teaching.py references unknown concept keys: "
            f"{sorted(unknown)}"
        )
    return report


def check_teaching_surface(
    *,
    session_cls: type | None = None,
    catalog: Mapping[str, Any] | None = None,
    index_path: Path | None = None,
) -> DriftReport:
    """Run every drift check and combine the results into one report.

    The single entry point, used by the CI test. Runs all five checks and
    collects their findings rather than stopping at the first failure, because
    one underlying change usually breaks several checks at once and seeing them
    together is what makes the cause obvious.

    Parameters
    ----------
    session_cls:
        The class to check. Defaults to the real ``Session``.
    catalog:
        The operation catalog. Defaults to the real one.
    index_path:
        The generated index. Defaults to the packaged location.

    Returns
    -------
    DriftReport
        Every error and warning from every check. ``ok`` is ``True`` only when
        nothing blocking was found.

    Notes
    -----
    **Every check runs**, even after one fails.

    Examples
    --------
    As a test::

        def test_teaching_surface_in_sync():
            check_teaching_surface().raise_if_failed()

    See Also
    --------
    DriftReport.raise_if_failed : Turning the report into an assertion.
    """
    report = DriftReport()
    for partial in (
        check_session_catalog_parity(session_cls=session_cls, catalog=catalog),
        check_catalog_parameters_vs_signatures(session_cls=session_cls, catalog=catalog),
        check_operation_index_fresh(path=index_path, session_cls=session_cls),
        check_ai_tools_vs_catalog(session_cls=session_cls, catalog=catalog),
        check_dashboard_teaching_concepts(),
    ):
        report.errors.extend(partial.errors)
        report.warnings.extend(partial.warnings)
    return report
