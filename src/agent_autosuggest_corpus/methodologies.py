# SPDX-License-Identifier: MIT

"""Methodology availability for corpus generation and replay.

The workspace registry is authoritative and is read once for every lake turn. Standalone corpus
generation has no workspace, so it uses an explicit fixture instead. The only compatibility
assumption in code is Zetta's documented ``unknown_capability`` behavior: an older engine can run
M-1 and nothing else.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).parent / "fixtures" / "methodologies.json"

LEGACY_M1 = {
    "id": "M-1",
    "name": "Churn / GBDT classification",
    "family": "classification",
    "model_kinds": ["gbdt"],
    "solutions": [],
    "inputs": [],
    "outputs": [],
    "metrics": [],
    "runnable": True,
    "partial": False,
}

Query = Callable[[str, dict[str, Any]], dict[str, Any]]
Post = Callable[[str, str, dict[str, Any]], dict[str, Any]]


class CapabilityError(RuntimeError):
    """A lake workspace capability could not answer."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Registry:
    """The methodology facts used for one generation or replay."""

    entries: list[dict[str, Any]]
    source: str
    old_engine: bool = False
    note: str = ""

    def entry(self, identifier: str | None) -> dict[str, Any] | None:
        return next((item for item in self.entries if item.get("id") == identifier), None)

    def context(self) -> str:
        value: dict[str, Any] = {
            "source": self.source,
            "methodologies": self.entries,
        }
        if self.old_engine:
            value["old_engine"] = True
        if self.note:
            value["note"] = self.note
        return (
            "\n\n## Methodologies available for this example\n\n"
            "This registry is authoritative. Use its M-n ids and never suggest an entry whose "
            "`runnable` value is false.\n\n```json\n"
            + json.dumps(value, indent=2, sort_keys=True)
            + "\n```\n"
        )


def _urllib_post(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        try:
            return json.loads(error.read().decode())
        except json.JSONDecodeError:
            return {
                "ok": False,
                "error": {
                    "code": "store_error",
                    "message": f"HTTP {error.code}",
                },
            }


def make_query(
    lake_url: str,
    token: str,
    customer_id: str,
    post: Post | None = None,
) -> Query:
    """Build the read path from this agent through lake to a workspace."""
    send = post or _urllib_post
    url = f"{lake_url.rstrip('/')}/v1/workspace_query"

    def query(capability: str, inputs: dict[str, Any]) -> dict[str, Any]:
        body = {
            "customer_id": customer_id,
            "capability": capability,
            "input": inputs,
        }
        answer = send(url, token, body)
        if not isinstance(answer, dict) or not answer.get("ok"):
            error = answer.get("error") if isinstance(answer, dict) else None
            error = error if isinstance(error, dict) else {}
            raise CapabilityError(
                str(error.get("code") or "store_error"),
                str(error.get("message") or "workspace_query failed"),
            )
        value = answer.get("value")
        if not isinstance(value, dict):
            raise CapabilityError(
                "invalid_response",
                "workspace_query returned no object value",
            )
        return value

    return query


def load_fixture(path: Path = FIXTURE) -> Registry:
    """Load the deliberate standalone-generation fixture."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CapabilityError("invalid_fixture", f"cannot read {path}: {error}") from error
    entries = value.get("methodologies") if isinstance(value, dict) else None
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise CapabilityError(
            "invalid_fixture",
            f"{path} does not contain a methodologies list",
        )
    return Registry(entries, f"fixture:{path}")


def read(query: Query) -> Registry:
    """Read the live registry, recognizing only the documented old-engine response."""
    try:
        value = query("list_methodologies", {})
    except CapabilityError as error:
        if error.code != "unknown_capability":
            raise
        note = (
            "list_methodologies is unknown to this older workspace engine; "
            "only M-1 is runnable"
        )
        return Registry([dict(LEGACY_M1)], "old-engine fallback", True, note)

    entries = value.get("methodologies") if isinstance(value, dict) else None
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise CapabilityError(
            "invalid_response",
            "list_methodologies did not return a methodologies list",
        )
    return Registry(entries, "list_methodologies")


def for_turn(
    turn: dict[str, Any],
    query: Query | None = None,
    fixture: Path = FIXTURE,
    post: Post | None = None,
) -> Registry:
    """Read methodology facts at the start of a service turn."""
    if query is not None:
        return read(query)

    lake_url = os.environ.get("LAKE_URL", "")
    token = str(turn.get("lake_token") or "")
    customer_id = str(turn.get("customer_id") or "")
    if lake_url and token and customer_id:
        return read(make_query(lake_url, token, customer_id, post))
    return load_fixture(fixture)


def enforce(example: dict[str, Any], registry: Registry) -> dict[str, Any]:
    """Stamp registry facts and move non-runnable suggestions to declined."""
    project = dict(example.get("project_profile") or {})
    suggested: list[dict[str, Any]] = []
    declined: list[dict[str, Any]] = []

    for original in project.get("suggested") or []:
        item = _with_status(dict(original), registry)
        entry = registry.entry(item.get("methodology"))
        if entry is not None and entry.get("runnable") is not True:
            declined.append(_unsupported(item, entry))
            continue
        suggested.append(item)

    for original in project.get("declined") or []:
        item = _with_status(dict(original), registry)
        entry = registry.entry(item.get("methodology"))
        if entry is not None and entry.get("runnable") is not True:
            item = _unsupported(item, entry)
        declined.append(item)

    project["suggested"] = suggested
    project["declined"] = declined
    return {
        **example,
        "project_profile": project,
        "methodology_registry_source": registry.source,
    }


def defects(example: dict[str, Any], registry: Registry) -> list[str]:
    """Return methodology contract violations in one generated or replayed example."""
    problems: list[str] = []
    project = example.get("project_profile") or {}
    for index, item in enumerate(project.get("suggested") or []):
        _check_item(item, f"suggested[{index}]", registry, problems)
        entry = registry.entry(item.get("methodology"))
        if entry is not None and entry.get("runnable") is not True:
            problems.append(f"suggested[{index}] uses non-runnable {item.get('methodology')}")

    for index, item in enumerate(project.get("declined") or []):
        where = f"declined[{index}]"
        _check_item(item, where, registry, problems)
        entry = registry.entry(item.get("methodology"))
        if entry is not None and entry.get("runnable") is not True:
            _check_unsupported(item, where, entry, problems)
        if entry is not None and entry.get("runnable") is True:
            reason = str(item.get("reason") or "")
            names_methodology = f"needs {item.get('methodology')}" in reason
            if item.get("answer") == "not supported yet" and names_methodology:
                problems.append(
                    f"{where} must not call runnable {item.get('methodology')} unsupported"
                )
        if item.get("answer") == "transform first":
            reason = str(item.get("reason") or "")
            if not reason.lower().startswith("transform first:"):
                problems.append(f"{where} must name the required transform")
    return problems


def _with_status(item: dict[str, Any], registry: Registry) -> dict[str, Any]:
    entry = registry.entry(item.get("methodology"))
    if entry is not None:
        item["runnable"] = entry.get("runnable") is True
    return item


def _unsupported(item: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    methodology = str(entry.get("id") or item.get("methodology") or "")
    issue = str(entry.get("issue") or "").strip()
    issue_text = f", {issue}" if issue else ""
    return {
        **item,
        "methodology": methodology,
        "runnable": False,
        "reason": f"needs {methodology} (not supported yet{issue_text})",
        "answer": "not supported yet",
    }


def _check_item(
    item: dict[str, Any],
    where: str,
    registry: Registry,
    problems: list[str],
) -> None:
    methodology = item.get("methodology")
    entry = registry.entry(methodology)
    if entry is None:
        problems.append(f"{where} names {methodology!r}, which is not in the methodology registry")
        return
    runnable = entry.get("runnable") is True
    if item.get("runnable") is not runnable:
        problems.append(f"{where} must record runnable={runnable} for {methodology}")


def _check_unsupported(
    item: dict[str, Any],
    where: str,
    entry: dict[str, Any],
    problems: list[str],
) -> None:
    methodology = str(entry.get("id") or "")
    reason = str(item.get("reason") or "")
    if item.get("answer") != "not supported yet":
        problems.append(f"{where} must answer 'not supported yet' for {methodology}")
    if f"needs {methodology}" not in reason:
        problems.append(f"{where} must say it needs {methodology}")
    issue = str(entry.get("issue") or "").strip()
    if issue and issue not in reason:
        problems.append(f"{where} must name {issue}")
