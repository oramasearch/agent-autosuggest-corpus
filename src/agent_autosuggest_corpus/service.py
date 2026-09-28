"""The Auto-Suggest Corpus Generator as a lake library agent (lake#263): one example per turn.

A person asks for a training example for Auto-Suggest ("a quality manager at a contract bakery").
One turn is one `generate_one`, steered by the request and checked exactly as the batch CLI
checks it: the contamination guard against the eval set (the service refuses to answer without
it, as the CLI refuses to start) and the example's defects. It answers the example; it writes
nothing to the corpus, which stays the CLI's to grow.

Turns arrive on lake's agent bus (lake_bus.py); `POST /v1/turn` answers the same for a direct call.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent_autosuggest_corpus import lake_bus
from agent_autosuggest_corpus import methodologies
from agent_autosuggest_corpus import solutions
from agent_autosuggest_corpus.agent import PROMPT_VERSION, generate_one
from agent_autosuggest_corpus.guard import EvalGuard

ATTEMPTS = 2


def run_turn(
    turn: dict[str, Any],
    client: Any = None,
    guard: EvalGuard | None = None,
    model: str | None = None,
    methodology_query: methodologies.Query | None = None,
    methodology_fixture: Path = methodologies.FIXTURE,
    solution_query: methodologies.Query | None = None,
    solution_fixture: Path = solutions.FIXTURE,
) -> dict[str, Any]:
    request = (turn.get("request") or turn.get("message") or "").strip()
    guard = guard or EvalGuard(Path(os.environ.get("EVAL_PROFILES", "/app/evals/profiles")))
    if not guard.ids:
        return {"ok": False, "error": "the eval set is not in this image, so contamination "
                                      "cannot be checked; no example is written without it"}
    try:
        registry = methodologies.for_turn(
            turn,
            query=methodology_query,
            fixture=methodology_fixture,
        )
    except methodologies.CapabilityError as error:
        return {
            "ok": False,
            "error": f"cannot read methodology registry ({error.code}): {error}",
        }
    try:
        solution_registry = solutions.for_turn(
            turn,
            query=solution_query or methodology_query,
            fixture=solution_fixture,
        )
    except methodologies.CapabilityError as error:
        return {
            "ok": False,
            "error": f"cannot read solution registry ({error.code}): {error}",
        }
    model = model or os.environ.get("AUTOSUGGEST_MODEL", os.environ.get("AGENT_MODEL", "gpt-4o"))
    client = client or OpenAI()
    reasons: list[str] = []
    for _ in range(ATTEMPTS):
        example, reason = generate_one(
            client,
            model,
            guard,
            request=request,
            registry=registry,
            solution_registry=solution_registry,
        )
        if example is not None:
            project = example.get("project_profile") or {}
            suggested = project.get("suggested") or []
            declined = project.get("declined") or []
            reply = (f"I wrote a training example for Auto-Suggest, **{example['profile_id']}**: "
                     f"{example.get('industry') or 'an industry'}. It suggests "
                     + (", ".join(f"*{item.get('intent', '?')}* ({item.get('solution', '?')})"
                                  for item in suggested) or "nothing")
                     + f", and declines {len(declined)}. It is kept apart from the eval set "
                       "and was not added to the corpus.")
            if registry.old_engine:
                reply += " This older workspace engine can run only M-1."
            if solution_registry.unavailable:
                reply += " This workspace engine does not deploy the solution registry yet."
            return {"ok": True, "reply": reply,
                    "result": {"kind": "autosuggest_example", "example": example,
                               "prompt_version": PROMPT_VERSION}}
        reasons.append(reason)
    return {"ok": False, "error": "no usable example: " + "; ".join(reasons)}


def serve() -> int:
    token = os.environ.get("AGENT_SERVICE_TOKEN", "")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802 (the stdlib's name)
            if self.path == "/healthz":
                guard = EvalGuard(Path(os.environ.get("EVAL_PROFILES", "/app/evals/profiles")))
                self._send(200 if guard.ids else 503,
                           {"ok": bool(guard.ids), "agent": "agent-autosuggest-corpus",
                            "eval_profiles": len(guard.ids)})
                return
            self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/turn":
                self._send(404, {"ok": False, "error": "not found"})
                return
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self._send(401, {"ok": False, "error": "unauthorized"})
                return
            try:
                turn = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            except (json.JSONDecodeError, ValueError):
                self._send(400, {"ok": False, "error": "the body is not JSON"})
                return
            try:
                answer = run_turn(turn if isinstance(turn, dict) else {})
            except Exception as error:  # a failed turn is an answer, not a dead worker
                answer = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            self._send(200 if answer.get("ok") else 422, answer)

        def log_message(self, format: str, *args: Any) -> None:
            print(json.dumps({"agent": "agent-autosuggest-corpus", "request": format % args}),
                  flush=True)

    port = int(os.environ.get("PORT", "8080"))
    print(json.dumps({"agent": "agent-autosuggest-corpus", "listening": port}), flush=True)
    lake_bus.start_from_env(run_turn, "agent-autosuggest-corpus")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
    return 0
