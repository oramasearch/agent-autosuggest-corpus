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

EXAMPLE = json.loads(Path("corpus/autosuggest-sft-v3.jsonl").read_text().splitlines()[0])
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
SOLUTIONS = json.loads(
    Path("src/agent_autosuggest_corpus/fixtures/solutions.json").read_text()
)


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
    def __init__(self, error: str | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, capability: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((capability, inputs))
        if self.error:
            raise methodologies.CapabilityError(self.error, "not available")
        if capability == "list_methodologies":
            return METHODOLOGIES
        if capability == "list_solutions":
            return SOLUTIONS
        if capability == "get_solution":
            identifier = inputs["solution"]
            entry = next(item for item in SOLUTIONS["solutions"] if item["id"] == identifier)
            return {"solution": entry}
        return {}


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
    query = FakeQuery()
    answer = service.run_turn(
        {"request": "a contract bakery"},
        client,
        guard,
        "m",
        methodology_query=query,
    )

    assert answer["ok"], answer
    assert query.calls == [
        ("list_methodologies", {}),
        ("list_solutions", {}),
        ("get_solution", {"solution": "S-1"}),
        ("get_solution", {"solution": "S-2"}),
    ]
    assert "a contract bakery" in client.sent[0]["messages"][1]["content"]
    assert '\"id\": \"M-1\"' in client.sent[0]["messages"][0]["content"]
    suggestion = answer["result"]["example"]["project_profile"]["suggested"][0]
    assert suggestion["methodology"] == "M-1"
    assert suggestion["runnable"] is True
    assert answer["result"]["example"]["methodology_registry_source"] == "list_methodologies"
    assert suggestion["solution"] == "S-1"
    assert suggestion["level"] == 1
    assert suggestion["methodologies"] == ["M-2", "M-1", "M-3"]
    assert answer["result"]["example"]["solution_registry_source"] == (
        "list_solutions/get_solution"
    )


def test_an_old_engine_has_m1_but_no_solutions_and_says_so(guard: EvalGuard) -> None:
    candidate = copy.deepcopy(EXAMPLE)
    candidate["project_profile"] = {"suggested": [], "declined": []}
    client = Scripted(json.dumps(candidate))
    query = FakeQuery(error="unknown_capability")
    answer = service.run_turn(
        {"request": "employee retention"},
        client,
        guard,
        "m",
        methodology_query=query,
    )

    assert answer["ok"], answer
    assert query.calls == [("list_methodologies", {}), ("list_solutions", {})]
    assert '\"old_engine\": true' in client.sent[0]["messages"][0]["content"]
    assert "only M-1" in answer["reply"]
    assert "does not deploy the solution registry" in answer["reply"]
    assert answer["result"]["example"]["project_profile"] == {
        "suggested": [],
        "declined": [],
    }


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
        methodology_query=FakeQuery(),
    )
    assert not answer["ok"]
    assert "CONTAMINATED" in answer["error"]


def test_without_the_eval_set_it_writes_nothing(tmp_path: Path) -> None:
    answer = service.run_turn(
        {"request": "x"},
        Scripted(),
        EvalGuard(tmp_path),
        "m",
        methodology_query=FakeQuery(),
    )
    assert not answer["ok"]
    assert "eval set" in answer["error"]


def test_healthz_reports_the_installed_version(tmp_path: Path, monkeypatch) -> None:
    import tomllib
    from importlib.metadata import version

    declared = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    monkeypatch.setenv("EVAL_PROFILES", str(tmp_path))
    status, body = service.health()
    assert status == 503 and not body["ok"]
    assert body["version"] == version("agent-autosuggest-corpus") == declared


def test_a_turn_calls_its_model_through_lakes_gateway_with_the_turns_key(
        guard: EvalGuard, monkeypatch) -> None:
    """lake#248: the model is lake's `/openai/v1`, with the turn's own key, never a provider key."""
    import httpx

    from agent_autosuggest_corpus import model

    seen: list[httpx.Request] = []

    def gateway(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(EXAMPLE)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(service, "gateway_client", lambda url, token: model.gateway_client(
        url, token, http_client=httpx.Client(transport=httpx.MockTransport(gateway))))
    answer = service.run_turn(
        {"request": "a contract bakery", "lake_url": "http://lake:8080",
         "lake_token": "lat.task.secret"},
        None, guard, "m", methodology_query=FakeQuery())
    assert answer["ok"], answer
    assert seen and all(str(request.url) == "http://lake:8080/openai/v1/chat/completions"
                        for request in seen)
    assert all(request.headers["authorization"] == "Bearer lat.task.secret" for request in seen)

    monkeypatch.delenv("LAKE_URL", raising=False)
    calls = len(seen)
    refused = service.run_turn({"request": "a contract bakery"}, None, guard, "m",
                               methodology_query=FakeQuery())
    assert not refused["ok"] and "lake_token" in refused["error"]
    assert len(seen) == calls
