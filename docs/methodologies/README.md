# Methodology-aware corpus generation

The corpus must teach only pipeline methodologies the target workspace can execute. Availability
belongs to Zetta's methodology registry and changes independently of this repository.

## Registry sources

Lake service turns call `workspace_query` with the turn's customer and credential:

```json
{"customer_id":"...","capability":"list_methodologies","input":{}}
```

The response is read once at the start of each turn. The generator does not cache or translate the
list. When `list_methodologies` returns `unknown_capability`, the workspace predates the registry;
only M-1 is treated as runnable, and the service response states that limitation. Any other live
registry error stops the turn.

The bus passes the claimed task's `customer_id` into the turn body alongside its one-use
`lake_token`. Those values scope `workspace_query` to the workspace and caller that asked the turn.

The batch CLI has no workspace. It therefore uses
`src/agent_autosuggest_corpus/fixtures/methodologies.json`, an explicit conservative fixture. The
fixture is data rather than a Python list so a fixture update is reviewable on its own.

## Generation contract

Prompt v2 receives the selected registry as context and requires an `M-n` id on every suggested
and declined opportunity. After model output is parsed, the generator copies each entry's current
`runnable` value from the registry. It then:

1. keeps runnable suggestions;
2. moves non-runnable suggestions to `declined`;
3. writes `not supported yet` with a `needs M-n` reason and the registry issue; and
4. reserves `transform first` for runnable methods whose data needs a named reshaping step.

Unknown methodology ids fail validation. They are not guessed or mapped from labels such as
`gbdt` or `forecast`.

## Replay contract

The same validation runs when stored JSONL is replayed. It rejects a suggested method when the
selected registry does not contain it or does not mark it runnable. It also checks that unsupported
declines name both the M-n id and its tracking issue.

```sh
uv run agent-autosuggest-corpus --replay corpus/autosuggest-sft-v2.jsonl
```

The old v1 corpus was removed from the active corpus because its free-form labels could not meet
this contract. It remains recoverable from git history.
