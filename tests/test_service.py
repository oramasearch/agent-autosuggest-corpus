"""The corpus generator as a lake library agent: one checked example per request."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_autosuggest_corpus import methodologies, service
from agent_autosuggest_corpus.guard import EvalGuard

EXAMPLE = json.loads(Path("corpus/autosuggest-sft-v2.jsonl").read_text().splitlines()[0])
METHODOLOGIES = {
    "methodologies": [
        {
            "id": "M-1",
            "name": "Churn / GBDT classification",
            "family": "classification",
            "model_kinds": ["gbdt"],
            "runnable": True,
            "partial": False,
        },
        {
            "id": "M-3",
            "name": "Segmentation / clustering",
            "runnable": False,
            "partial": False,
            "issue": "zetta#1042",
        },
    ]
}


class Scripted:
    def __init__(self, *contents: str) -> None:
        self.contents = list(contents)
        self.sent: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.sent.append(kwargs)
        message = SimpleNamespace(content=self.contents.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeQuery:
    def __init__(self, value: dict[str, Any] | None = None, error: str | None = None) -> None:
        self.value = value
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, capability: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((capability, inputs))
        if self.error:
            raise methodologies.CapabilityError(self.error, "not available")
        return self.value or {}


@pytest.fixture
def guard(tmp_path: Path) -> EvalGuard:
    profile = tmp_path / "evals" / "held-out" / "profile.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        json.dumps(
            {
                "profile_id": "held-out-profile",
                "industry": "held-out industry",
                "tables": [{"database": "held", "name": "outcomes"}],
            }
        )
    )
    return EvalGuard(profile.parents[1])


def test_a_runnable_methodology_is_kept_from_the_live_registry(guard: EvalGuard) -> None:
    client = Scripted(json.dumps(EXAMPLE))
    query = FakeQuery(METHODOLOGIES)
    answer = service.run_turn(
        {"request": "a contract bakery"},
        client,
        guard,
        "m",
        methodology_query=query,
    )

    assert answer["ok"], answer
    assert query.calls == [("list_methodologies", {})]
    assert "a contract bakery" in client.sent[0]["messages"][1]["content"]
    assert '\"id\": \"M-1\"' in client.sent[0]["messages"][0]["content"]
    suggestion = answer["result"]["example"]["project_profile"]["suggested"][0]
    assert suggestion["methodology"] == "M-1"
    assert suggestion["runnable"] is True
    assert answer["result"]["example"]["methodology_registry_source"] == "list_methodologies"


def test_an_unsupported_methodology_is_declined_with_its_issue(guard: EvalGuard) -> None:
    candidate = copy.deepcopy(EXAMPLE)
    unsupported = copy.deepcopy(candidate["project_profile"]["suggested"][0])
    unsupported.update(
        {
            "intent": "segment employees into intervention cohorts",
            "solution": "S-9",
            "methodology": "M-3",
        }
    )
    candidate["project_profile"]["suggested"].append(unsupported)
    answer = service.run_turn(
        {"request": "employee retention"},
        Scripted(json.dumps(candidate)),
        guard,
        "m",
        methodology_query=FakeQuery(METHODOLOGIES),
    )

    assert answer["ok"], answer
    project = answer["result"]["example"]["project_profile"]
    assert [item["methodology"] for item in project["suggested"]] == ["M-1"]
    declined = next(item for item in project["declined"] if item["methodology"] == "M-3")
    assert declined["answer"] == "not supported yet"
    assert declined["runnable"] is False
    assert "needs M-3" in declined["reason"]
    assert "zetta#1042" in declined["reason"]


def test_an_old_engine_falls_back_to_m1_and_says_so(guard: EvalGuard) -> None:
    client = Scripted(json.dumps(EXAMPLE))
    query = FakeQuery(error="unknown_capability")
    answer = service.run_turn(
        {"request": "employee retention"},
        client,
        guard,
        "m",
        methodology_query=query,
    )

    assert answer["ok"], answer
    assert query.calls == [("list_methodologies", {})]
    assert '\"old_engine\": true' in client.sent[0]["messages"][0]["content"]
    assert "only M-1" in answer["reply"]


def test_an_example_from_the_eval_set_is_never_answered(guard: EvalGuard) -> None:
    leaked = {
        **EXAMPLE,
        "profile_id": "held-out-profile",
        "industry": "held-out industry",
    }
    answer = service.run_turn(
        {"request": "x"},
        Scripted(json.dumps(leaked), json.dumps(leaked)),
        guard,
        "m",
        methodology_query=FakeQuery(METHODOLOGIES),
    )
    assert not answer["ok"]
    assert "CONTAMINATED" in answer["error"]


def test_without_the_eval_set_it_writes_nothing(tmp_path: Path) -> None:
    answer = service.run_turn(
        {"request": "x"},
        Scripted(),
        EvalGuard(tmp_path),
        "m",
        methodology_query=FakeQuery(METHODOLOGIES),
    )
    assert not answer["ok"]
    assert "eval set" in answer["error"]
