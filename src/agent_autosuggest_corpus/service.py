"""The Auto-Suggest Corpus Generator as a lake library agent (lake#263): one example per turn.

A person asks for a training example for Auto-Suggest ("a quality manager at a contract bakery").
One turn is one `generate_one`, steered by the request and checked exactly as the batch CLI
checks it: the contamination guard against the eval set (the service refuses to answer without
it, as the CLI refuses to start) and the example's defects. It answers the example; it writes
nothing to the corpus, which stays the CLI's to grow.

Turns arrive on lake's agent bus (lake_bus.py); `POST /v1/turn` answers the same for a direct call.
"""

from __future__ import annotations

import hmac
import json
import os
from importlib.metadata import PackageNotFoundError, version as package_version
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agent_autosuggest_corpus import lake_bus
from agent_autosuggest_corpus import methodologies
from agent_autosuggest_corpus import solutions
from agent_autosuggest_corpus.agent import PROMPT_VERSION, generate_one
from agent_autosuggest_corpus.guard import EvalGuard
from agent_autosuggest_corpus.model import gateway_client

ATTEMPTS = 2


def _version() -> str:
    try:
        return package_version("agent-autosuggest-corpus")
    except PackageNotFoundError:
        return "unknown"


def health() -> tuple[int, dict[str, Any]]:
    """`GET /healthz`: 503 without the eval set, since no turn can be answered without it."""
    guard = EvalGuard(Path(os.environ.get("EVAL_PROFILES", "/app/evals/profiles")))
    return (200 if guard.ids else 503,
            {"ok": bool(guard.ids), "agent": "agent-autosuggest-corpus", "version": _version(),
             "eval_profiles": len(guard.ids)})


def lakes_url() -> str:
    """Lake's address from this service's environment (`LAKE_URL`), the only place a turn's
    credential is ever sent; empty when unset or not an absolute http(s) URL without user info."""
    url = os.environ.get("LAKE_URL", "").strip().rstrip("/")
    try:
        parts = urlsplit(url)
    except ValueError:
        return ""
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname or "@" in parts.netloc:
        return ""
    return url


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
    if client is None:
        # The model is reached through lake with the turn's own key, which counts its tokens in
        # the turn's workspace (lake#248); this agent holds no provider key. The address is this
        # service's LAKE_URL; the turn's own `lake_url` is never used, so the key cannot be sent
        # to a host the caller names (agent-autosuggest-corpus#13).
        model_url = lakes_url()
        if not model_url:
            return {"ok": False, "error": "LAKE_URL is not set: this agent sends a turn's "
                                          "credential only to Lake's configured address"}
        model_key = turn.get("lake_token") or ""
        if not model_key:
            return {"ok": False, "error": "a turn needs its lake_token to reach the model "
                                          "through lake"}
        # A turn claimed by reference names its task on every model call (lake#871).
        client = gateway_client(model_url, model_key, headers=lake_bus.task_headers(turn))
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


def turn_refusal(authorization: str | None, token: str) -> tuple[int, str] | None:
    """Why `POST /v1/turn` is refused, as `(status, error)`, or None when the caller may proceed.

    The route fails closed: with `AGENT_SERVICE_TOKEN` unset or empty every turn is refused, so a
    service started without it is never open. The bearer is compared in constant time
    (`hmac.compare_digest` over bytes), so the comparison's timing says nothing about the token.
    """
    if not token:
        return 503, "AGENT_SERVICE_TOKEN is not set; this service refuses every turn"
    expected = f"Bearer {token}".encode()
    supplied = (authorization or "").encode("utf-8", "surrogateescape")
    if not hmac.compare_digest(supplied, expected):
        return 401, "unauthorized"
    return None


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
                self._send(*health())
                return
            self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/turn":
                self._send(404, {"ok": False, "error": "not found"})
                return
            refusal = turn_refusal(self.headers.get("Authorization"), token)
            if refusal is not None:
                status, error = refusal
                self._send(status, {"ok": False, "error": error})
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
