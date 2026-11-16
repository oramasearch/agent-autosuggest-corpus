# agent-autosuggest-corpus

Generates the **SFT training corpus for the Auto-Suggest model**
([zetta#986](https://github.com/oramasearch/zetta/issues/986)).

Auto-Suggest reads a **profile** — a situation plus its data — and proposes what is worth
building. This repo produces the examples that teach it to do that.

## The rule that matters most

**Never read from `OramaSearch/agent-eval-profiles`.**

Those profiles are the **test set** for this exact model. Training on them makes every eval score
meaningless — the model would be scored on situations it had memorised, and we would have no way
to tell a model that reads data from one that recalls an answer key. This repo generates its own
situations, independently.

The rule is enforced, not merely written down: `guard.py` fails the run if a generated situation
matches an eval profile, and nothing publishes without passing it.

## What one training example looks like

**Input:** a profile — persona, connections with capture state, tables with schemas and null
rates.

**Output:** a **project profile** — the suggested approach for *this* data.

That output shape is deliberate: **it is the same shape the project-creation agent consumes
later**. Auto-Suggest proposes the approach; the project agent turns it into a Pipeline and chats
about it. One shape means the two models compose instead of needing a translation layer that can
drift.

```jsonc
{
  "suggested": [{
    "intent": "predict which accounts are likely to leave in the next 90 days",
    "solution": "S-1",                    // the catalogue module
    "level": 1,                            // a registry capability level
    "solution_runnable": true,             // copied from list_solutions/get_solution
    "methodology": "M-1",                // the primary engine methodology
    "methodologies": ["M-2", "M-1", "M-3"], // registry-derived composition order
    "runnable": true,                     // copied from list_methodologies
    "why_this_data_supports_it": "accounts.churn_date gives a label; ...",
    "required_first": ["entity resolution across crm and billing"],
    "pipeline_shape": {                   // the correctness profile it must satisfy
      "connect": true, "clean": true, "point_in_time": true, "leakage_free": true,
      "model": true, "calibrated": true, "holdout": true, "explain": "shap", "serve": true
    },
    "business_action": "route to the retention team's weekly call list",
    "fits_the_constraint": "40 accounts a month, so the output is a ranked top-40"
  }],
  "declined": [{
    "intent": "predict account churn from product usage",
    "solution": "S-1",
    "level": 1,
    "methodology": "M-1",
    "runnable": true,
    "reason": "product.telemetry is discovered but not captured",
    "answer": "capture it first"          // NOT 'cannot be done'
  }]
}
```

## What makes an example worth training on

- **The approach must be specific to the input data.** "Build a churn model" is not a suggestion,
  it is a category. A suggestion names the label column, the cutoff, and what makes this data
  support it.
- **Declining is a first-class output.** A model that always suggests something will suggest
  something worthless. And *discovered-but-not-captured* has a different answer from *cannot be
  done* — "capture it first".
- **Ordering is part of the answer.** Entity resolution before anything account-level; a churn
  score as an input to next-best-action rather than a competitor to it.
- **The persona's constraints change the suggestion.** Forty actionable items a month means a
  ranked top-40, not a score for every account.

## Status

Current release: [v0.2.0](docs/releases/v0.2.0.md) (release notes for every version are in
[`docs/releases/`](docs/releases/README.md)).

The active v3 corpus contains one replay-validated example. The prompt and corpus moved from v2 to
v3 because solution level, runnability, and ordered methodology composition are now part of the
training contract. Consumers that copy a versioned corpus file must copy
`autosuggest-sft-v3.jsonl`. The previous v1 rows were removed because their free-form methodology
labels could not prove engine support; v2 was superseded because it did not record solution
registry facts.

## Solutions the agent teaches

Every opportunity names a solution, selects one of its capability levels, and records the ordered
methodologies composed at that level. In lake, the service reads `list_solutions` and hydrates every
listed entry with `get_solution` through `workspace_query` at the start of every turn. The live
registry is authoritative: runnable entries can be suggested, while non-runnable entries move to
`declined` with `not supported yet` and a `needs S-n` reason. A runnable solution whose input needs
reshaping is declined as `transform first` with the concrete transform.

An older workspace that answers `unknown_capability` has no available solutions. The generator
does not substitute a built-in list. Standalone generation and replay use the checked-in solution
fixture described in [`docs/solutions/README.md`](docs/solutions/README.md).

## Methodologies the agent teaches

Every suggested and declined opportunity records an `M-n` methodology id. In lake, the service
reads `list_methodologies` through `workspace_query` at the start of every turn and treats that
workspace registry as authoritative. Runnable entries may be suggested. Non-runnable entries are
moved to `declined` with `not supported yet`, a `needs M-n` reason, and the registry's tracking
issue. A runnable methodology whose input needs reshaping is declined as `transform first`.

Standalone generation has no workspace, so it reads the checked-in methodology fixture. The
fixture is deliberately conservative and currently permits only M-1. If an older workspace answers
`unknown_capability`, the documented compatibility rule also permits only M-1 and the service says
that it used the older-engine behavior. Other registry failures stop generation instead of falling
back silently.

Replay the active corpus against the fixture with:

```sh
uv run agent-autosuggest-corpus --replay corpus/autosuggest-sft-v3.jsonl
```

## In lake (lake#263)

A library agent of lake: a person asks it for one training example ("a quality manager at a
contract bakery"). One turn is one `generate_one` (the loop body of the CLI), steered by the request
and checked exactly as the CLI checks it: the contamination guard against the eval set, then the
example's defects, with one retry. It answers `{reply, result: {kind: autosuggest_example,
example}}` and **writes nothing to the corpus**; growing the corpus stays the CLI's.

**The model is reached through lake** (lake#248, lake `docs/api/model-gateway.md`): in lake every
call goes to `<LAKE_URL>/openai/v1` with the turn's own `lake_token` as the API key (`model.py`), so
lake counts its tokens in the turn's workspace and enforces the account's model-token limit. The
service holds no `OPENAI_API_KEY`, and a turn without a lake credential is refused. Only the batch
CLI, which runs for no workspace, calls OpenAI directly with your own key.

**A turn's credential goes only to Lake at this service's own `LAKE_URL`** (#13). The model address
and the registry reads (`<LAKE_URL>/v1/workspace_query`) both come from `LAKE_URL`; a turn's
`lake_url` is ignored. With `LAKE_URL` unset, or not an absolute `http(s)` URL without user info,
every turn that would call the model is refused before any request is made.

- `model.py`: the OpenAI client pointed at lake's model gateway with the turn's credential.
- `service.py`: `run_turn`, and `agent-autosuggest-corpus-serve` (`GET /healthz` reports the
  version and is 503 without the eval set; `POST /v1/turn`).
  `POST /v1/turn` requires `Authorization: Bearer <AGENT_SERVICE_TOKEN>`, compared in constant time (`401` otherwise), and fails closed: with `AGENT_SERVICE_TOKEN` unset or empty it refuses every turn with `503`. The operator always sets it; set it yourself for a hand-run service. Turns from lake's bus do not use it.
- `methodologies.py`: reads the live registry, applies the M-1 old-engine fallback, filters
  generated suggestions, and validates replayed rows.
- `solutions.py`: reads and hydrates the live catalogue, treats an absent capability as an empty
  catalogue, filters generated suggestions, and validates levels and composition during replay.
- `lake_bus.py`: takes turns from lake's agent bus with its own `AGENT_BUS_TOKEN`.
- `Dockerfile`: private image `oramasearch/agent-autosuggest-corpus`; the eval set is the named
  build context `evals` (`--local evals=../agent-eval-generator/evals/profiles --opt
  context:evals=local:evals`).
- Tests: `uv run --group dev pytest`; `tests/test_release.py` requires release notes for the
  version in `pyproject.toml`, which `GET /healthz` also reports as `version`.

## License

MIT, see [LICENSE](LICENSE). Third-party components and their licenses are in
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES). Contributions are welcome under the
[Contributor License Agreement](CLA.md); see [CONTRIBUTING.md](CONTRIBUTING.md).
