"""Methodology registry transport and corpus replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_autosuggest_corpus import methodologies
from agent_autosuggest_corpus.agent import replay


def test_workspace_query_uses_the_lake_envelope() -> None:
    sent: list[tuple[str, str, dict[str, Any]]] = []

    def post(url: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        sent.append((url, token, body))
        return {"ok": True, "value": {"methodologies": []}}

    query = methodologies.make_query("https://lake.example/", "turn-token", "customer-7", post)
    assert query("list_methodologies", {}) == {"methodologies": []}
    assert sent == [
        (
            "https://lake.example/v1/workspace_query",
            "turn-token",
            {
                "customer_id": "customer-7",
                "capability": "list_methodologies",
                "input": {},
            },
        )
    ]


def test_active_corpus_replays_against_the_standalone_fixture() -> None:
    registry = methodologies.load_fixture()
    assert replay(Path("corpus/autosuggest-sft-v2.jsonl"), registry) == []


def test_replay_rejects_a_non_runnable_suggestion(tmp_path: Path) -> None:
    fixture = tmp_path / "methodologies.json"
    fixture.write_text(
        json.dumps(
            {
                "methodologies": [
                    {
                        "id": "M-3",
                        "name": "Segmentation / clustering",
                        "runnable": False,
                        "issue": "zetta#1042",
                    }
                ]
            }
        )
    )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "profile_id": "bad-example",
                "persona": {"constraints": ["only 10 results"]},
                "project_profile": {
                    "suggested": [
                        {
                            "intent": "segment customers",
                            "solution": "S-9",
                            "methodology": "M-3",
                            "runnable": False,
                            "why_this_data_supports_it": "customer_id and purchase_count",
                            "pipeline_shape": {},
                            "fits_the_constraint": "returns 10 segments",
                        }
                    ],
                    "declined": [
                        {
                            "intent": "forecast sales",
                            "methodology": "M-3",
                            "runnable": False,
                            "reason": "needs M-3 (not supported yet, zetta#1042)",
                            "answer": "not supported yet",
                        }
                    ],
                },
            }
        )
        + "\n"
    )

    problems = replay(corpus, methodologies.load_fixture(fixture))
    assert any("uses non-runnable M-3" in problem for problem in problems)
