"""The corpus generator as a lake library agent (lake#263): one checked example per request."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent_autosuggest_corpus import service
from agent_autosuggest_corpus.guard import EvalGuard

EXAMPLE = json.loads(Path("corpus/autosuggest-sft-v1.jsonl").read_text().splitlines()[0])
EVALS = Path(__file__).parents[2] / "agent-eval-generator" / "evals" / "profiles"


class Scripted:
    def __init__(self, *contents: str) -> None:
        self.contents, self.sent = list(contents), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        message = SimpleNamespace(content=self.contents.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_a_request_steers_one_checked_example():
    client = Scripted(json.dumps(EXAMPLE))
    answer = service.run_turn({"request": "a contract bakery"}, client, EvalGuard(EVALS), "m")
    assert answer["ok"], answer
    assert "a contract bakery" in client.sent[0]["messages"][1]["content"]
    assert answer["result"]["example"]["profile_id"] == EXAMPLE["profile_id"]
    assert "not added to the corpus" in answer["reply"]


def test_an_example_from_the_eval_set_is_never_answered():
    evaluated = json.loads(next(EVALS.glob("*/profile.json")).read_text())
    leaked = {**EXAMPLE, "profile_id": evaluated["profile_id"], "industry": evaluated["industry"]}
    answer = service.run_turn({"request": "x"}, Scripted(json.dumps(leaked), json.dumps(leaked)),
                              EvalGuard(EVALS), "m")
    assert not answer["ok"] and "CONTAMINATED" in answer["error"]


def test_without_the_eval_set_it_writes_nothing(tmp_path):
    answer = service.run_turn({"request": "x"}, Scripted(), EvalGuard(tmp_path), "m")
    assert not answer["ok"] and "eval set" in answer["error"]
