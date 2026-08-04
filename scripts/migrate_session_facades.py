#!/usr/bin/env python3
"""Rewrite teaching/content Session flat domain calls to namespaced facades.

Reads ``scripts/_facade_bindings.json`` and rewrites only ``warn_flat`` domain
bindings (classical CORE stays dual and is not force-migrated).

Usage:
  python scripts/migrate_session_facades.py --check PATH [PATH ...]
  python scripts/migrate_session_facades.py --write PATH [PATH ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINDINGS_PATH = Path(__file__).resolve().parent / "_facade_bindings.json"

TEXT_SUFFIXES = frozenset({".py", ".md", ".rst", ".txt"})
SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "node_modules",
        "dist",
        "build",
        "_build",
        "artifacts",
        ".buildml-artifacts",
        "htmlcov",
        ".eggs",
    }
)

# Flat names that collide with common third-party APIs (e.g. DataFrame.rank).
# Underscored domain flats are treated as unique; only short generic names need
# a Session-like receiver before rewrite.
AMBIGUOUS_FLATS = frozenset(
    {
        "rank",
        "recommend",
    }
)

PROPERTY_FACADE_NAMES = frozenset(
    {
        "plan",
        "result",
        "transcript",
        "assumptions",
        "spec",
        "last_report",
        "history",
        "last_dry_run",
        "last_summary",
        "last_walkthrough",
        "backbone",
        "backbone_head",
        "asr_eval",
        "speech_result",
        "train_result",
        "cv_result",
        "search_result",
        "nested_cv_result",
        "export_result",
        "ddp_result",
        "text_plan",
        "topic_plan",
        "head_plan",
        "neuro_plan",
        "imitation_plan",
        "imitation_fit_result",
        "imitation_eval_result",
        "imitation_predict_result",
        "analysis_result",
    }
)

_PROTECTED_FLAT_CALL = re.compile(
    r"""(?x)
    (?:
        describe_method
        | propose_tool_execution
        | getattr
        | hasattr
        | setattr
        | delattr
    )
    \s*\(\s*["']
    """
)

_SESSION_ASSIGN = re.compile(r"(?m)^\s*[A-Za-z_][\w]*\s*=\s*.*\bsession\b|^\s*session\s*=")
_SESSION_IDENT = re.compile(r"\bsession\b", re.IGNORECASE)
_FENCE_RE = re.compile(
    r"(?ms)^(?P<indent>[ \t]{0,3})(?P<fence>```+|~~~+)[^\n]*\n"
    r"(?P<body>.*?)^(?P=indent)(?P=fence)[ \t]*$",
)

_SESSION_LIKE = re.compile(
    r"(?:"
    r"session"
    r"|[A-Za-z_]\w*session"
    r"|[A-Za-z_]\w*_session"
    r"|sess"
    r"|[A-Za-z_]\w*_sess"
    r")$",
    re.IGNORECASE,
)

# Receivers that must never be rewritten (schema ids, package attrs).
_PACKAGE_LIKE = re.compile(
    r"^(?:buildml|sklearn|torch|numpy|np|pd|pandas|pyod|scipy|transformers)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Binding:
    flat: str
    facade_attr: str
    facade_method: str
    is_property: bool
    ambiguous: bool

    @property
    def preferred(self) -> str:
        return f"session.{self.facade_attr}.{self.facade_method}"


@dataclass
class Hit:
    path: Path
    line: int
    flat: str
    preferred: str
    kind: str
    snippet: str


@dataclass
class FileResult:
    path: Path
    original: str
    rewritten: str
    hits: list[Hit] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.original != self.rewritten


def _is_property_like(facade_name: str) -> bool:
    if facade_name in PROPERTY_FACADE_NAMES:
        return True
    return facade_name.endswith("_result") or facade_name.endswith("_plan")


def load_warn_flat_bindings(path: Path = BINDINGS_PATH) -> list[Binding]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[Binding] = []
    seen: dict[str, Binding] = {}
    for facade_attr, spec in raw.items():
        if not spec.get("warn_flat"):
            continue
        for facade_method, flat in spec.get("bindings", {}).items():
            binding = Binding(
                flat=flat,
                facade_attr=facade_attr,
                facade_method=facade_method,
                is_property=_is_property_like(facade_method),
                ambiguous=flat in AMBIGUOUS_FLATS
                or ("_" not in flat and len(flat) <= 12),
            )
            prev = seen.get(flat)
            if prev is not None and (
                prev.facade_attr != binding.facade_attr
                or prev.facade_method != binding.facade_method
            ):
                raise SystemExit(
                    f"Duplicate warn_flat flat name {flat!r}: "
                    f"{prev.preferred} vs {binding.preferred}"
                )
            seen[flat] = binding
            out.append(binding)
    out.sort(key=lambda b: len(b.flat), reverse=True)
    return out


def _should_skip_dir(name: str) -> bool:
    if name in SKIP_DIR_NAMES:
        return True
    return name.endswith(".egg-info")


def iter_target_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = raw if raw.is_absolute() else (ROOT / raw)
        path = path.resolve()
        if not path.exists():
            print(f"warning: path does not exist: {path}", file=sys.stderr)
            continue
        if path.is_file():
            if path.suffix.lower() in TEXT_SUFFIXES:
                files.append(path)
            continue
        for p in path.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if any(_should_skip_dir(part) for part in p.parts):
                continue
            # proofs/: only script.py, baseline*.py, README.md
            # (enforced by caller path filters when needed; here accept all under given roots)
            files.append(p)
    return sorted(set(files), key=lambda p: str(p).lower())


def filter_proofs_files(files: list[Path]) -> list[Path]:
    """When migrating proofs/, keep script.py / baseline*.py / README.md only."""
    kept: list[Path] = []
    for path in files:
        try:
            rel = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            kept.append(path)
            continue
        if not rel.startswith("proofs/"):
            kept.append(path)
            continue
        name = path.name
        if name == "script.py" or name == "README.md" or (
            name.startswith("baseline") and name.endswith(".py")
        ):
            kept.append(path)
    return kept


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _snippet(text: str, index: int, width: int = 88) -> str:
    line_start = text.rfind("\n", 0, index) + 1
    line_end = text.find("\n", index)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end].strip()
    if len(line) > width:
        return line[: width - 3] + "..."
    return line


def _protected_at(text: str, index: int) -> bool:
    window_start = max(0, index - 96)
    if _PROTECTED_FLAT_CALL.search(text[window_start:index]) is not None:
        return True
    # Skip Python / prose comments on the same line.
    line_start = text.rfind("\n", 0, index) + 1
    line_prefix = text[line_start:index]
    if "#" in line_prefix:
        return True
    return False


def _receiver_before_dot(text: str, dot_index: int) -> str | None:
    """Identifier immediately before ``.``; None if chained (e.g. ``).method``)."""
    i = dot_index - 1
    if i < 0:
        return None
    if not (text[i].isalnum() or text[i] == "_"):
        return None
    while i >= 0 and (text[i].isalnum() or text[i] == "_"):
        i -= 1
    return text[i + 1 : dot_index]


def _has_session_nearby(text: str) -> bool:
    return bool(_SESSION_IDENT.search(text))


def _already_facaded(text: str, dot_index: int, facade_attr: str) -> bool:
    """True when ``.<facade_attr>.<flat>`` (already migrated / nested)."""
    receiver = _receiver_before_dot(text, dot_index)
    return receiver == facade_attr


def _peel_session_receiver(out: list[str]) -> None:
    """Rewrite trailing ``Session`` identifier in the emit buffer to ``session``."""
    if not out or not out[-1].endswith("Session"):
        return
    prev = out[-1]
    cut = len(prev) - len("Session")
    if cut >= 0 and prev[cut:] == "Session" and (
        cut == 0 or not (prev[cut - 1].isalnum() or prev[cut - 1] == "_")
    ):
        out[-1] = prev[:cut] + "session"


def _allow_dot_rewrite(
    binding: Binding,
    receiver: str | None,
    *,
    kind: str,
    allow_session_class_matrix: bool,
) -> bool:
    """Decide whether a ``.flat`` / ``.flat(`` match should be rewritten."""
    if receiver is not None and _PACKAGE_LIKE.match(receiver):
        return False
    if receiver == "Session":
        if (
            binding.facade_method == "capability_matrix"
            or binding.flat.endswith("_capability_matrix")
        ) and not allow_session_class_matrix:
            return False
        return True
    if receiver is None:
        # Chained ).fit_anomaly( — safe for non-ambiguous domain flats.
        return not binding.ambiguous
    if _SESSION_LIKE.match(receiver):
        return True
    # Unknown / package / domain receivers: do not rewrite. This avoids
    # ``buildml.federated.results.export_round_history`` and similar module
    # paths. In-repo teaching uses ``session`` / ``*_session`` receivers.
    return False


def _rewrite_region(
    text: str,
    bindings: list[Binding],
    *,
    path: Path,
    allow_session_class_matrix: bool,
    rewrite_bare_backticks: bool,
    rewrite_bare_tokens: bool = False,
    line_offset: int = 0,
) -> tuple[str, list[Hit]]:
    hits: list[Hit] = []
    if not bindings:
        return text, hits
    by_flat = {b.flat: b for b in bindings}
    flat_alt = "|".join(re.escape(b.flat) for b in bindings)

    # Dot-call: .flat(   (covers session.X(, Session.X(, ).X()
    call_re = re.compile(rf"\.(?P<flat>{flat_alt})(?P<call>\s*\()")
    # Dot-attr without call: .flat\b
    attr_re = re.compile(rf"\.(?P<flat>{flat_alt})(?![A-Za-z0-9_])(?!\s*\()")
    # Prose ticks: `flat` or `flat(...)` (args may include nested quotes, not backticks)
    tick_re = re.compile(rf"`(?P<flat>{flat_alt})(?P<args>\([^`]*)?`")
    # RST double-backticks: ``flat`` / ``flat(...)``
    rst_tick_re = re.compile(rf"``(?P<flat>{flat_alt})(?P<args>\([^`]*)?``")
    # Unquoted teaching tokens: fit_anomaly / fit_anomaly(...) in concept strings
    token_re = re.compile(
        rf"(?<![\w.])(?P<flat>{flat_alt})(?P<args>\([^)]*\))?(?!\w)"
    )

    out: list[str] = []
    pos = 0
    n = len(text)
    session_nearby = _has_session_nearby(text)

    while pos < n:
        candidates: list[tuple[int, str, re.Match[str]]] = []
        m_call = call_re.search(text, pos)
        if m_call:
            candidates.append((m_call.start(), "call", m_call))
        m_attr = attr_re.search(text, pos)
        if m_attr:
            candidates.append((m_attr.start(), "attr", m_attr))
        if rewrite_bare_backticks:
            m_rst = rst_tick_re.search(text, pos)
            if m_rst:
                candidates.append((m_rst.start(), "rst_tick", m_rst))
            m_tick = tick_re.search(text, pos)
            if m_tick:
                candidates.append((m_tick.start(), "tick", m_tick))
        if rewrite_bare_tokens:
            m_tok = token_re.search(text, pos)
            if m_tok:
                candidates.append((m_tok.start(), "token", m_tok))
        if not candidates:
            out.append(text[pos:])
            break
        # Prefer longer / more specific matches at the same index (rst before tick,
        # call before attr, ticks before bare tokens).
        kind_rank = {"call": 0, "attr": 1, "rst_tick": 2, "tick": 3, "token": 4}
        candidates.sort(key=lambda t: (t[0], kind_rank.get(t[1], 9)))
        start, kind, match = candidates[0]
        out.append(text[pos:start])

        flat = match.group("flat")
        binding = by_flat[flat]
        flat_idx = match.start("flat")
        if _protected_at(text, flat_idx):
            out.append(match.group(0))
            pos = match.end()
            continue

        if kind == "token":
            args = match.group("args") or ""
            match_end = match.end()
            # Ambiguous flats (rank/recommend) are English verbs too — never
            # rewrite as bare tokens. Backtick / session.dot paths still migrate.
            if binding.ambiguous or flat in AMBIGUOUS_FLATS:
                out.append(match.group(0))
                pos = match_end
                continue
            # Only rewrite teaching prose inside Python string literals.
            if not _inside_python_string(text, flat_idx):
                out.append(match.group(0))
                pos = match_end
                continue
            if _is_sole_catalog_key(
                text, match.start(), match.end(), flat=flat, args=args
            ):
                out.append(match.group(0))
                pos = match_end
                continue
            # Skip tokens already inside markdown/RST ticks (handled by tick paths).
            if start > 0 and text[start - 1] == "`":
                out.append(match.group(0))
                pos = match_end
                continue
            # Skip on-disk bundle member names (cbr_plan.joblib, etc.).
            if match_end < n and text[match_end : match_end + 7] == ".joblib":
                out.append(match.group(0))
                pos = match_end
                continue
            replacement = f"{binding.preferred}{args}"
            hits.append(
                Hit(
                    path=path,
                    line=line_offset + _line_of(text, start),
                    flat=flat,
                    preferred=binding.preferred,
                    kind="token",
                    snippet=_snippet(text, start),
                )
            )
            out.append(replacement)
            pos = match_end
            continue

        if kind in {"call", "attr"}:
            dot_index = start
            if _already_facaded(text, dot_index, binding.facade_attr):
                out.append(match.group(0))
                pos = match.end()
                continue
            receiver = _receiver_before_dot(text, dot_index)
            if not _allow_dot_rewrite(
                binding,
                receiver,
                kind=kind,
                allow_session_class_matrix=allow_session_class_matrix,
            ):
                out.append(match.group(0))
                pos = match.end()
                continue

            if kind == "call":
                call = match.group("call")
                if receiver == "Session":
                    _peel_session_receiver(out)
                replacement = f".{binding.facade_attr}.{binding.facade_method}{call}"
                hits.append(
                    Hit(
                        path=path,
                        line=line_offset + _line_of(text, start),
                        flat=flat,
                        preferred=binding.preferred,
                        kind="call",
                        snippet=_snippet(text, start),
                    )
                )
                out.append(replacement)
                pos = match.end()
                continue

            # attr
            if receiver == "Session":
                _peel_session_receiver(out)
            replacement = f".{binding.facade_attr}.{binding.facade_method}"
            hits.append(
                Hit(
                    path=path,
                    line=line_offset + _line_of(text, start),
                    flat=flat,
                    preferred=binding.preferred,
                    kind="attr",
                    snippet=_snippet(text, start),
                )
            )
            out.append(replacement)
            pos = match.end()
            continue

        # backtick / RST prose: `fit_anomaly` / `fit_anomaly(...)` → facade path
        args = match.group("args") or ""
        if binding.ambiguous and not args and not session_nearby:
            out.append(match.group(0))
            pos = match.end()
            continue
        # Skip schema-ish ticks that are not API references (rare bare flat).
        if kind == "tick" and match.group(0).startswith("`buildml."):
            out.append(match.group(0))
            pos = match.end()
            continue
        if kind == "rst_tick":
            replacement = f"``{binding.preferred}{args}``"
            hit_kind = "rst_backtick"
        else:
            replacement = f"`{binding.preferred}{args}`"
            hit_kind = "backtick"
        hits.append(
            Hit(
                path=path,
                line=line_offset + _line_of(text, start),
                flat=flat,
                preferred=binding.preferred,
                kind=hit_kind,
                snippet=_snippet(text, start),
            )
        )
        out.append(replacement)
        pos = match.end()

    return "".join(out), hits


def _split_fences(text: str) -> list[tuple[str, str, int]]:
    """Return list of (kind, chunk, start_index) where kind is 'fence' or 'prose'."""
    parts: list[tuple[str, str, int]] = []
    pos = 0
    for fence in _FENCE_RE.finditer(text):
        if fence.start() > pos:
            parts.append(("prose", text[pos : fence.start()], pos))
        parts.append(("fence", fence.group(0), fence.start()))
        pos = fence.end()
    if pos < len(text):
        parts.append(("prose", text[pos:], pos))
    if not parts:
        parts.append(("prose", text, 0))
    return parts


def _rewrite_fence_block(
    block: str,
    bindings: list[Binding],
    *,
    path: Path,
    line_offset: int,
) -> tuple[str, list[Hit]]:
    """Rewrite a full fenced block, preserving opening/closing fence lines."""
    lines = block.splitlines(keepends=True)
    if len(lines) < 2:
        return block, []
    # opening fence is lines[0]; closing is last non-empty fence line
    # Find closing fence: last line matching fence pattern
    close_idx = len(lines) - 1
    while close_idx > 0 and lines[close_idx].strip() == "":
        close_idx -= 1
    opening = lines[0]
    closing = lines[close_idx]
    body = "".join(lines[1:close_idx])
    allow = _has_session_nearby(body)
    new_body, hits = _rewrite_region(
        body,
        bindings,
        path=path,
        allow_session_class_matrix=allow,
        rewrite_bare_backticks=False,
        rewrite_bare_tokens=False,
        line_offset=line_offset + 1,
    )
    return opening + new_body + closing + "".join(lines[close_idx + 1 :]), hits


def _rel_posix(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def _allow_prose_ticks(path: Path) -> bool:
    """True when file embeds markdown-style `` `flat` `` API prose in text/strings."""
    rel = _rel_posix(path)
    if rel is None or path.suffix.lower() != ".py":
        return False
    return rel.startswith("buildml/explain/") or rel.startswith("examples/")


def _allow_bare_teaching_tokens(path: Path) -> bool:
    """Rewrite unquoted ``fit_anomaly`` tokens in explain teaching strings.

    Catalog keys / tool allowlists that are *exactly* the flat name stay flat.
    """
    rel = _rel_posix(path)
    if rel is None or path.suffix.lower() != ".py":
        return False
    if not rel.startswith("buildml/explain/"):
        return False
    # sync.py / generated registries keep flat callable keys; teaching modules
    # (concepts, overlays, layered prose, prerequisites) prefer facade paths in
    # narrative while leaving exact \"flat\" catalog strings untouched.
    if rel in {
        "buildml/explain/prerequisites.py",
        "buildml/explain/capability_status.py",
        "buildml/explain/glossary.py",
        "buildml/explain/pedagogy.py",
        "buildml/explain/history.py",
    }:
        return True
    parts = rel.split("/")
    if len(parts) < 3:
        return False
    return parts[2] in {
        "concepts",
        "beginner",
        "intermediate",
        "advanced",
        "walkthroughs",
        "overlays",
    }


def _is_sole_catalog_key(
    text: str, start: int, end: int, *, flat: str, args: str
) -> bool:
    """True when match is the entire contents of a ``\"flat\"`` / ``'flat'`` literal."""
    if args:
        return False
    i = start - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    if i < 0 or text[i] not in "'\"":
        return False
    quote = text[i]
    j = end
    while j < len(text) and text[j] in " \t":
        j += 1
    if j >= len(text) or text[j] != quote:
        return False
    return text[start:end] == flat


def _inside_python_string(text: str, index: int) -> bool:
    """True when ``index`` lies inside a ``'...'`` / ``\"...\"`` / triple-quoted string."""
    i = 0
    n = len(text)
    while i < n and i <= index:
        ch = text[i]
        if ch == "#":
            nl = text.find("\n", i)
            if nl < 0:
                return False
            if index <= nl:
                return False
            i = nl + 1
            continue
        if ch in "'\"":
            quote = ch
            start = i
            if text.startswith(quote * 3, i):
                i += 3
                end = text.find(quote * 3, i)
                if end < 0:
                    return start < index
                if start < index < end:
                    return True
                i = end + 3
                continue
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    end = i
                    i += 1
                    if start < index < end:
                        return True
                    break
                i += 1
            else:
                return start < index
            continue
        i += 1
    return False


def rewrite_text(text: str, bindings: list[Binding], *, path: Path) -> tuple[str, list[Hit]]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".rst"}:
        hits: list[Hit] = []
        out: list[str] = []
        for kind, chunk, start in _split_fences(text):
            line_offset = _line_of(text, start) - 1
            if kind == "fence":
                new_chunk, chunk_hits = _rewrite_fence_block(
                    chunk, bindings, path=path, line_offset=line_offset
                )
            else:
                # Prose: session/Session refs + backticks; matrix rewrite if session mentioned
                new_chunk, chunk_hits = _rewrite_region(
                    chunk,
                    bindings,
                    path=path,
                    allow_session_class_matrix=_has_session_nearby(chunk)
                    or _has_session_nearby(text),
                    rewrite_bare_backticks=True,
                    rewrite_bare_tokens=False,
                    line_offset=line_offset,
                )
            out.append(new_chunk)
            hits.extend(chunk_hits)
        return "".join(out), hits

    allow = _has_session_nearby(text)
    # Explain + example modules embed markdown-style `flat` / `flat(...)` prose in
    # strings/docstrings; rewrite those. Tool allowlist string keys stay flat
    # (exact \"flat\" literals / no backticks around the registry name).
    return _rewrite_region(
        text,
        bindings,
        path=path,
        allow_session_class_matrix=allow,
        rewrite_bare_backticks=_allow_prose_ticks(path),
        rewrite_bare_tokens=_allow_bare_teaching_tokens(path),
    )


def process_file(path: Path, bindings: list[Binding]) -> FileResult:
    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return FileResult(path=path, original="", rewritten="", hits=[])
    rewritten, hits = rewrite_text(original, bindings, path=path)
    return FileResult(path=path, original=original, rewritten=rewritten, hits=hits)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _area_for(path: Path) -> str:
    rel = _rel(path)
    if rel in {"README.md", "CONTRIBUTING.md"}:
        return rel
    parts = rel.split("/")
    if parts[0] == "buildml" and len(parts) > 2 and parts[1] == "explain":
        return f"buildml/explain/{parts[2]}"
    return parts[0]


def scan_residuals(text: str, path: Path, bindings: list[Binding]) -> list[Hit]:
    by_flat = {b.flat: b for b in bindings}
    flat_alt = "|".join(re.escape(b.flat) for b in bindings)
    if not flat_alt:
        return []
    call_re = re.compile(rf"\.(?P<flat>{flat_alt})\s*\(")
    residuals: list[Hit] = []
    for match in call_re.finditer(text):
        flat = match.group("flat")
        binding = by_flat[flat]
        if _already_facaded(text, match.start(), binding.facade_attr):
            continue
        if _protected_at(text, match.start("flat")):
            continue
        receiver = _receiver_before_dot(text, match.start())
        if receiver is not None and _PACKAGE_LIKE.match(receiver):
            continue
        if receiver is not None and receiver != "Session" and not (
            _SESSION_LIKE.match(receiver)
        ):
            continue
        if binding.ambiguous and receiver is None:
            continue
        residuals.append(
            Hit(
                path=path,
                line=_line_of(text, match.start()),
                flat=flat,
                preferred=binding.preferred,
                kind="residual_call",
                snippet=_snippet(text, match.start()),
            )
        )
    # Narrative ticks (fence-aware for md/rst)
    suffix = path.suffix.lower()
    regions: list[tuple[str, int]] = []
    if suffix in {".md", ".rst"}:
        for kind, chunk, start in _split_fences(text):
            if kind == "prose":
                regions.append((chunk, start))
    else:
        regions.append((text, 0))
    tick_re = re.compile(rf"`(?P<flat>{flat_alt})(?P<args>\([^`]*)?`")
    rst_tick_re = re.compile(rf"``(?P<flat>{flat_alt})(?P<args>\([^`]*)?``")
    for chunk, region_start in regions:
        for rx, kind in ((rst_tick_re, "residual_rst_backtick"), (tick_re, "residual_backtick")):
            for match in rx.finditer(chunk):
                # Avoid double-counting the inner single-tick of an RST ``flat``.
                if kind == "residual_backtick" and match.start() > 0 and chunk[match.start() - 1] == "`":
                    continue
                flat = match.group("flat")
                binding = by_flat[flat]
                if binding.ambiguous and not match.group("args") and not _has_session_nearby(
                    chunk
                ):
                    continue
                if _protected_at(chunk, match.start("flat")):
                    continue
                abs_index = region_start + match.start()
                residuals.append(
                    Hit(
                        path=path,
                        line=_line_of(text, abs_index),
                        flat=flat,
                        preferred=binding.preferred,
                        kind=kind,
                        snippet=_snippet(text, abs_index),
                    )
                )
    return residuals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Report without writing")
    mode.add_argument("--write", action="store_true", help="Apply rewrites in place")
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories")
    parser.add_argument("--bindings", type=Path, default=BINDINGS_PATH)
    parser.add_argument(
        "--report-residuals",
        action="store_true",
        help="List remaining flat domain calls after processing",
    )
    parser.add_argument(
        "--proofs-filter",
        action="store_true",
        help="When paths include proofs/, only script.py / baseline*.py / README.md",
    )
    args = parser.parse_args(argv)

    bindings = load_warn_flat_bindings(args.bindings)
    files = iter_target_files(args.paths)
    if args.proofs_filter:
        files = filter_proofs_files(files)
    if not files:
        print("No target files found.", file=sys.stderr)
        return 1

    results = [process_file(path, bindings) for path in files]
    changed = [r for r in results if r.changed]
    area_counts: dict[str, int] = {}
    for r in changed:
        area = _area_for(r.path)
        area_counts[area] = area_counts.get(area, 0) + 1

    total_hits = sum(len(r.hits) for r in results)
    print(
        f"Scanned {len(files)} files; "
        f"{len(changed)} would change; "
        f"{total_hits} replacements"
    )
    for area in sorted(area_counts):
        print(f"  {area}: {area_counts[area]} file(s)")

    if args.check:
        for r in changed:
            print(f"\n== {_rel(r.path)} ({len(r.hits)} hit(s)) ==")
            for h in r.hits[:25]:
                print(f"  L{h.line}: {h.flat} -> {h.preferred}  ({h.snippet})")
            if len(r.hits) > 25:
                print(f"  ... +{len(r.hits) - 25} more")

    if args.write:
        for r in changed:
            r.path.write_text(r.rewritten, encoding="utf-8")
        print(f"\nWrote {len(changed)} file(s).")

    if args.report_residuals or args.write or args.check:
        residual_hits: list[Hit] = []
        for r in results:
            text = r.rewritten if (args.check or args.write) else r.original
            # After write, re-read from disk for accuracy
            if args.write:
                text = r.path.read_text(encoding="utf-8")
            residual_hits.extend(scan_residuals(text, r.path, bindings))

        if residual_hits:
            print(f"\nResidual flat domain calls: {len(residual_hits)}")
            for h in residual_hits[:100]:
                print(
                    f"  {_rel(h.path)}:{h.line}: [{h.kind}] {h.flat} -> {h.preferred}"
                    f"  ({h.snippet})"
                )
            if len(residual_hits) > 100:
                print(f"  ... +{len(residual_hits) - 100} more")
        else:
            print("\nResidual flat domain calls: 0")

    if args.check and changed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
