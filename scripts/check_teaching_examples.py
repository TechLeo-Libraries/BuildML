"""Bind authored teaching examples to the installed Session API without executing them.

Examples are contextual fragments: their data and models may be supplied by the
reader. This checks complete argument binding, including required arguments,
and reports unparseable fragments rather than silently skipping them.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import sys
from pathlib import Path
from types import UnionType
from typing import Literal, Union, get_args, get_origin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _literal_choices(annotation):
    if get_origin(annotation) is Literal:
        return get_args(annotation)
    if annotation is type(None):
        return (None,)
    if get_origin(annotation) in {Union, UnionType}:
        choices = [_literal_choices(part) for part in get_args(annotation)]
        if all(part is not None for part in choices):
            return tuple(value for part in choices for value in part)
    return None


def check_examples() -> tuple[int, list[str]]:
    from buildml import Session
    from buildml.session.facade_registry import DOMAIN_FACADES

    checked = 0
    errors = []
    type_names = {}
    for module_name, module in tuple(sys.modules.items()):
        if module_name.startswith("buildml.") and module_name.endswith((".types", ".results")):
            type_names.update(vars(module))
    for path in sorted((ROOT / "buildml/explain/beginner").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.keyword) or node.arg != "example":
                continue
            location = f"{path.relative_to(ROOT)}:{node.lineno}"
            try:
                lines = ast.literal_eval(node.value)
                tree = ast.parse(lines if isinstance(lines, str) else "\n".join(lines))
            except (ValueError, TypeError, SyntaxError) as exc:
                errors.append(f"{location}: {exc}")
                continue
            fresh_sessions = {
                target.id
                for statement in tree.body
                if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
                and ast.unparse(statement.value).startswith(("Session.ingest(", "Session()."))
                for target in statement.targets if isinstance(target, ast.Name)
            }
            restored_splits = {
                ast.unparse(call.func).split(".")[0]
                for call in ast.walk(tree) if isinstance(call, ast.Call)
                and ast.unparse(call.func).endswith((".inject_split", ".split", ".group_split", ".time_split"))
            }
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                parts = ast.unparse(call.func).split(".")
                if parts[0] not in {"session", "job", "service", "restored", "resumed", "review", "svc", "app", "serving", "audit", "other", "later"} or len(parts) not in {2, 3}:
                    continue
                method = parts[-1] if len(parts) == 2 else DOMAIN_FACADES.get(parts[1], {}).get("bindings", {}).get(parts[2])
                checked += 1
                try:
                    function = getattr(Session, method or "")
                    signature = inspect.signature(function)
                    receiver = [object()] if next(iter(signature.parameters), None) in {"self", "cls"} else []
                    signature.bind(*receiver, *[object() for _ in call.args], **{kw.arg: object() for kw in call.keywords if kw.arg})
                    if parts[0] in fresh_sessions - restored_splits and "partition" in signature.parameters:
                        keywords = {kw.arg: kw.value for kw in call.keywords}
                        partition = keywords.get("partition")
                        selected = partition.value if isinstance(partition, ast.Constant) else signature.parameters["partition"].default
                        if selected in {"train", "validation", "test"} and not {"support_frame", "candidates", "frame"} & keywords.keys():
                            errors.append(f"{location}: {ast.unparse(call)} selects a named partition on a fresh Session without restoring a split")
                    annotations = {}
                    for name, annotation in inspect.get_annotations(function).items():
                        try:
                            annotations[name] = eval(annotation, {**type_names, **function.__globals__}) if isinstance(annotation, str) else annotation
                        except NameError:
                            pass
                    for keyword in call.keywords:
                        annotation = annotations.get(keyword.arg)
                        choices = _literal_choices(annotation)
                        if choices is not None and isinstance(keyword.value, ast.Constant) and keyword.value.value not in choices:
                            errors.append(f"{location}: {ast.unparse(call)}: invalid {keyword.arg}={keyword.value.value!r}; expected {choices}")
                except (AttributeError, TypeError) as exc:
                    errors.append(f"{location}: {ast.unparse(call)}: {exc}")
            # Check direct result attributes against typed return records. This
            # catches plausible but nonexistent fields such as pred.mean.
            result_types = {}
            for statement in tree.body:
                if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
                    parts = ast.unparse(statement.value.func).split(".")
                    if parts[0] == "session" and len(parts) in {2, 3}:
                        method = parts[-1] if len(parts) == 2 else DOMAIN_FACADES.get(parts[1], {}).get("bindings", {}).get(parts[2])
                        function = getattr(Session, method or "", None)
                        try:
                            annotation = inspect.get_annotations(function).get("return")
                            record = eval(annotation, {**type_names, **function.__globals__}) if isinstance(annotation, str) else annotation
                        except (NameError, TypeError):
                            record = None
                        if dataclasses.is_dataclass(record):
                            for target in statement.targets:
                                if isinstance(target, ast.Name):
                                    result_types[target.id] = record
                for attribute in ast.walk(statement):
                    if isinstance(attribute, ast.Attribute) and isinstance(attribute.value, ast.Name):
                        record = result_types.get(attribute.value.id)
                        if record is not None and attribute.attr not in {f.name for f in dataclasses.fields(record)} and not hasattr(record, attribute.attr):
                            errors.append(f"{location}: {ast.unparse(attribute)} is not a field or method of {record.__name__}")
    return checked, errors


if __name__ == "__main__":
    checked, errors = check_examples()
    print(f"Checked {checked} teaching calls; {len(errors)} errors")
    print("\n".join(errors))
    raise SystemExit(bool(errors))
