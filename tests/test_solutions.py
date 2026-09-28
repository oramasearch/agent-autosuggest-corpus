"""Solution registry transport, filtering, and corpus replay."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from agent_autosuggest_corpus import methodologies, solutions
from agent_autosuggest_corpus.agent import replay

EXAMPLE = json.loads(Path("corpus/autosuggest-sft-v3.jsonl").read_text().splitlines()[0])


class FakeQuery:
    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries = entries
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, capability: str, inputs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((capability, inputs))
        if capability == "list_solutions":
            return {"solutions": [{"id": entry["id"]} for entry in self.entries]}
        if capability == "get_solution":
            entry = next(item for item in self.entries if item["id"] == inputs["solution"])
            return {"solution": entry}
        return {}


def test_live_registry_lists_then_gets_every_solution() -> None:
    entries = solutions.load_fixture().entries
    query = FakeQuery(entries)

    registry = solutions.read(query)

    assert registry.entries == entries
    assert registry.source == "list_solutions/get_solution"
    assert query.calls == [
        ("list_solutions", {}),
        ("get_solution", {"solution": "S-1"}),
        ("get_solution", {"solution": "S-2"}),
    ]


def test_unknown_solution_capability_means_no_solutions() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def query(capability: str, inputs: dict[str, Any]) -> dict[str, Any]:
        calls.append((capability, inputs))
        raise methodologies.CapabilityError("unknown_capability", "not deployed")

    registry = solutions.read(query)

    assert registry.entries == []
    assert registry.unavailable
    assert calls == [("list_solutions", {})]


def test_enforcement_stamps_composition_and_declines_non_runnable_solution() -> None:
    candidate = copy.deepcopy(EXAMPLE)
    unsupported = copy.deepcopy(candidate["project_profile"]["suggested"][0])
    unsupported.update(
        {
            "intent": "prioritize leads",
            "solution": "S-2",
            "level": 3,
        }
    )
    candidate["project_profile"]["suggested"].append(unsupported)

    enforced = solutions.enforce(candidate, solutions.load_fixture())
    project = enforced["project_profile"]

    assert project["suggested"][0]["methodologies"] == ["M-2", "M-1", "M-3"]
    declined = next(item for item in project["declined"] if item["solution"] == "S-2")
    assert declined["level"] == 3
    assert declined["methodologies"] == ["M-2", "M-1", "M-3", "M-6"]
    assert declined["answer"] == "not supported yet"
    assert "needs S-2" in declined["reason"]
    assert "zetta/issues/1102" in declined["reason"]


def test_a_sketch_is_declined_without_an_invented_level() -> None:
    sketch = {
        "id": "B-1",
        "name": "Dynamic Pricing & Discount Governance",
        "composition": [
            {"methodology": "M-1", "role": "primary", "minimum_level": 1}
        ],
        "pipeline_order": [],
        "capability_levels": [],
        "status": "sketch",
        "runnable": False,
        "issue": "https://github.com/oramasearch/zetta/issues/1102",
        "promotion_condition": "controlled price experiments are permitted",
    }
    candidate = copy.deepcopy(EXAMPLE)
    item = candidate["project_profile"]["suggested"][0]
    item["solution"] = "B-1"
    candidate["project_profile"] = {"suggested": [item], "declined": []}
    registry = solutions.Registry([sketch], "test")

    enforced = solutions.enforce(candidate, registry)
    declined = enforced["project_profile"]["declined"][0]

    assert "level" not in declined
    assert declined["methodologies"] == ["M-1"]
    assert "controlled price experiments are permitted" in declined["reason"]
    assert solutions.defects(enforced, registry) == []


def test_replay_rejects_a_primary_methodology_outside_the_solution_composition(
    tmp_path: Path,
) -> None:
    candidate = copy.deepcopy(EXAMPLE)
    candidate["project_profile"]["suggested"][0]["methodology"] = "M-3"
    corpus = tmp_path / "wrong-primary.jsonl"
    corpus.write_text(json.dumps(candidate) + "\n")

    problems = replay(
        corpus,
        methodologies.Registry(
            [
                {
                    "id": "M-1",
                    "runnable": True,
                },
                {
                    "id": "M-3",
                    "runnable": True,
                },
            ],
            "test",
        ),
        solutions.load_fixture(),
    )

    assert any("must record primary methodology M-1 for S-1" in problem for problem in problems)


def test_active_corpus_replays_against_both_registries() -> None:
    problems = replay(
        Path("corpus/autosuggest-sft-v3.jsonl"),
        methodologies.load_fixture(),
        solutions.load_fixture(),
    )

    assert problems == []
