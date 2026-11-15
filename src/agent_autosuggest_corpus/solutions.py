# SPDX-License-Identifier: MIT

"""Solution availability for corpus generation and replay.

Lake workspaces own the solution catalogue. Every service turn reads the catalogue and then reads
each listed solution in full. Standalone generation uses an explicit fixture so its recommendations
remain replayable without a workspace.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_autosuggest_corpus.methodologies import CapabilityError, Post, Query, make_query

FIXTURE = Path(__file__).parent / "fixtures" / "solutions.json"


@dataclass(frozen=True)
class Registry:
    """The solution facts used for one generation or replay."""

    entries: list[dict[str, Any]]
    source: str
    unavailable: bool = False
    note: str = ""

    def entry(self, identifier: str | None) -> dict[str, Any] | None:
        return next((item for item in self.entries if item.get("id") == identifier), None)

    def context(self) -> str:
        value: dict[str, Any] = {
            "source": self.source,
            "solutions": self.entries,
        }
        if self.unavailable:
            value["unavailable"] = True
        if self.note:
            value["note"] = self.note
        return (
            "\n\n## Solutions available for this example\n\n"
            "This registry is authoritative. Choose only ids and levels present here. A solution "
            "whose `runnable` value is false must be declined as not supported yet. If the list "
            "is empty, do not invent a solution.\n\n```json\n"
            + json.dumps(value, indent=2, sort_keys=True)
            + "\n```\n"
        )


def load_fixture(path: Path = FIXTURE) -> Registry:
    """Load the standalone solution registry fixture."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CapabilityError("invalid_fixture", f"cannot read {path}: {error}") from error
    entries = value.get("solutions") if isinstance(value, dict) else None
    _validate_entries(entries, "fixture", path)
    return Registry(entries, f"fixture:{path}")


def read(query: Query) -> Registry:
    """Read and hydrate the live solution catalogue for one turn."""
    try:
        value = query("list_solutions", {})
    except CapabilityError as error:
        if error.code != "unknown_capability":
            raise
        note = "list_solutions is not deployed on this workspace engine; no solutions are available"
        return Registry([], "solution registry unavailable", True, note)

    listed = value.get("solutions") if isinstance(value, dict) else None
    _validate_entries(listed, "list_solutions")
    entries: list[dict[str, Any]] = []
    for listed_entry in listed:
        identifier = str(listed_entry.get("id") or "")
        detail = query("get_solution", {"solution": identifier})
        entry = detail.get("solution") if isinstance(detail, dict) else None
        if not isinstance(entry, dict) or entry.get("id") != identifier:
            raise CapabilityError(
                "invalid_response",
                f"get_solution did not return solution {identifier}",
            )
        entries.append(entry)
    return Registry(entries, "list_solutions/get_solution")


def for_turn(
    turn: dict[str, Any],
    query: Query | None = None,
    fixture: Path = FIXTURE,
    post: Post | None = None,
) -> Registry:
    """Read solution facts at the start of a service turn."""
    if query is not None:
        return read(query)

    lake_url = os.environ.get("LAKE_URL", "")
    token = str(turn.get("lake_token") or "")
    customer_id = str(turn.get("customer_id") or "")
    if lake_url and token and customer_id:
        return read(make_query(lake_url, token, customer_id, post))
    return load_fixture(fixture)


def enforce(example: dict[str, Any], registry: Registry) -> dict[str, Any]:
    """Stamp catalogue facts and move non-runnable suggestions to declined."""
    project = dict(example.get("project_profile") or {})
    suggested: list[dict[str, Any]] = []
    declined: list[dict[str, Any]] = []

    for original in project.get("suggested") or []:
        item = _with_catalogue(dict(original), registry)
        entry = registry.entry(item.get("solution"))
        if entry is not None and entry.get("runnable") is not True:
            declined.append(_unsupported(item, entry))
            continue
        suggested.append(item)

    for original in project.get("declined") or []:
        item = _with_catalogue(dict(original), registry)
        entry = registry.entry(item.get("solution"))
        if entry is not None and entry.get("runnable") is not True:
            item = _unsupported(item, entry)
        declined.append(item)

    project["suggested"] = suggested
    project["declined"] = declined
    return {
        **example,
        "project_profile": project,
        "solution_registry_source": registry.source,
    }


def defects(example: dict[str, Any], registry: Registry) -> list[str]:
    """Return solution contract violations in a generated or replayed example."""
    problems: list[str] = []
    project = example.get("project_profile") or {}
    for index, item in enumerate(project.get("suggested") or []):
        where = f"suggested[{index}]"
        entry = _check_item(item, where, registry, problems)
        if entry is not None and entry.get("runnable") is not True:
            problems.append(f"{where} uses non-runnable {item.get('solution')}")

    for index, item in enumerate(project.get("declined") or []):
        where = f"declined[{index}]"
        entry = _check_item(item, where, registry, problems)
        if entry is not None and entry.get("runnable") is not True:
            _check_unsupported(item, where, entry, problems)
        if entry is not None and entry.get("runnable") is True:
            reason = str(item.get("reason") or "")
            names_solution = f"needs {item.get('solution')}" in reason
            if item.get("answer") == "not supported yet" and names_solution:
                problems.append(
                    f"{where} must not call runnable {item.get('solution')} unsupported"
                )
        if item.get("answer") == "transform first":
            reason = str(item.get("reason") or "")
            if not reason.lower().startswith("transform first:"):
                problems.append(f"{where} must name the required transform")
    return problems


def _validate_entries(
    entries: Any,
    capability: str,
    path: Path | None = None,
) -> None:
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        message = (
            f"{path} does not contain a solutions list"
            if path is not None
            else f"{capability} did not return a solutions list"
        )
        raise CapabilityError(
            "invalid_response" if path is None else "invalid_fixture",
            message,
        )
    identifiers = [item.get("id") for item in entries]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise CapabilityError(
            "invalid_response" if path is None else "invalid_fixture",
            f"{capability} contains a solution without an id",
        )
    if len(set(identifiers)) != len(identifiers):
        raise CapabilityError(
            "invalid_response" if path is None else "invalid_fixture",
            f"{capability} contains duplicate solution ids",
        )


def _with_catalogue(item: dict[str, Any], registry: Registry) -> dict[str, Any]:
    entry = registry.entry(item.get("solution"))
    if entry is None:
        return item
    item["solution_runnable"] = entry.get("runnable") is True
    levels = entry.get("capability_levels") or []
    if not levels and entry.get("status") == "sketch":
        item.pop("level", None)
        item["methodologies"] = _methodologies_for_level(entry, 1)
        return item
    level = item.get("level")
    if isinstance(level, int) and not isinstance(level, bool):
        item["methodologies"] = _methodologies_for_level(entry, level)
    return item


def _methodologies_for_level(entry: dict[str, Any], level: int) -> list[str]:
    composition = entry.get("composition") or []
    selected = {
        str(item.get("methodology"))
        for item in composition
        if isinstance(item, dict)
        and isinstance(item.get("minimum_level"), int)
        and item.get("minimum_level") <= level
    }
    ordered = [
        str(identifier)
        for identifier in entry.get("pipeline_order") or []
        if str(identifier) in selected
    ]
    remaining = [
        str(item.get("methodology"))
        for item in composition
        if isinstance(item, dict) and str(item.get("methodology")) in selected
    ]
    for identifier in remaining:
        if identifier not in ordered:
            ordered.append(identifier)
    return ordered


def _primary_methodology(entry: dict[str, Any], level: int) -> str | None:
    for item in entry.get("composition") or []:
        if not isinstance(item, dict):
            continue
        if item.get("role") != "primary":
            continue
        if not isinstance(item.get("minimum_level"), int):
            continue
        if item.get("minimum_level") <= level:
            return str(item.get("methodology"))
    return None


def _unsupported(item: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    solution = str(entry.get("id") or item.get("solution") or "")
    issue = str(entry.get("issue") or "").strip()
    issue_text = f", {issue}" if issue else ""
    promotion = str(entry.get("promotion_condition") or "").strip()
    promotion_text = f"; promotion requires: {promotion}" if promotion else ""
    return {
        **item,
        "solution": solution,
        "solution_runnable": False,
        "reason": f"needs {solution} (not supported yet{issue_text}){promotion_text}",
        "answer": "not supported yet",
    }


def _check_item(
    item: dict[str, Any],
    where: str,
    registry: Registry,
    problems: list[str],
) -> dict[str, Any] | None:
    solution = item.get("solution")
    entry = registry.entry(solution)
    if entry is None:
        problems.append(f"{where} names {solution!r}, which is not in the solution registry")
        return None
    runnable = entry.get("runnable") is True
    if item.get("solution_runnable") is not runnable:
        problems.append(f"{where} must record solution_runnable={runnable} for {solution}")
    level = item.get("level")
    levels = {
        candidate.get("level")
        for candidate in entry.get("capability_levels") or []
        if isinstance(candidate, dict)
    }
    if not levels and entry.get("status") == "sketch":
        if level is not None:
            problems.append(f"{where} must not invent a capability level for sketch {solution}")
        expected = _methodologies_for_level(entry, 1)
        if item.get("methodologies") != expected:
            problems.append(f"{where} must record the sketch methodology order {expected}")
        primary = _primary_methodology(entry, 1)
        if item.get("methodology") != primary:
            problems.append(f"{where} must record primary methodology {primary} for {solution}")
        return entry
    if not isinstance(level, int) or isinstance(level, bool) or level not in levels:
        problems.append(f"{where} must choose a registry level for {solution}")
        return entry
    expected = _methodologies_for_level(entry, level)
    if item.get("methodologies") != expected:
        problems.append(
            f"{where} must record the level-{level} methodology order {expected} for {solution}"
        )
    primary = _primary_methodology(entry, level)
    if item.get("methodology") != primary:
        problems.append(f"{where} must record primary methodology {primary} for {solution}")
    return entry


def _check_unsupported(
    item: dict[str, Any],
    where: str,
    entry: dict[str, Any],
    problems: list[str],
) -> None:
    solution = str(entry.get("id") or "")
    reason = str(item.get("reason") or "")
    if item.get("answer") != "not supported yet":
        problems.append(f"{where} must answer 'not supported yet' for {solution}")
    if f"needs {solution}" not in reason:
        problems.append(f"{where} must say it needs {solution}")
    issue = str(entry.get("issue") or "").strip()
    if issue and issue not in reason:
        problems.append(f"{where} must name {issue}")
    promotion = str(entry.get("promotion_condition") or "").strip()
    if promotion and promotion not in reason:
        problems.append(f"{where} must repeat the promotion condition for {solution}")
