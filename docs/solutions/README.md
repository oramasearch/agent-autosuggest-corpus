# Solution-aware corpus generation

The corpus maps a profile's data roles to the catalogue solution that can use them. Availability,
capability levels, and methodology composition belong to Zetta's solution registry and can change
independently of this repository.

## Registry reads

Lake service turns call `workspace_query` using the turn's customer and credential. Each turn first
lists the catalogue:

```json
{"customer_id":"...","capability":"list_solutions","input":{}}
```

It then calls `get_solution {"solution":"S-n"}` for every returned id. The hydrated entries, not a
Python list, become the prompt and validation boundary for that turn. This ensures the mapping uses
the current data roles, capability levels, runnability, tracking issue, composition, and pipeline
order.

If `list_solutions` returns `unknown_capability`, the workspace has no available solutions. The
service supplies an empty registry to the prompt and does not fall back to remembered catalogue
ids. Any other registry error stops the turn.

Standalone generation has no workspace, so it reads
`src/agent_autosuggest_corpus/fixtures/solutions.json`. The fixture contains a runnable S-1 and a
non-runnable S-2 so offline generation and replay exercise both recommendation and refusal paths.

## Generation contract

Prompt v3 receives the hydrated registry and asks the model to map table columns to a solution's
required data roles. The model chooses a registry id and level. After parsing, the generator copies
the solution's current `runnable` value and derives the selected methodologies:

1. select composition entries whose `minimum_level` is at or below the chosen level;
2. order those ids using the solution's `pipeline_order`;
3. keep a suggestion only when the solution is runnable;
4. move a non-runnable suggestion to `declined` with `not supported yet`, `needs S-n`, and its
   registry issue; and
5. reserve `transform first` for a runnable solution whose existing data can support it after a
   named reshaping step.

Band-B sketches have no capability levels and are never suggested. Their declined entry records
the sketch's methodology order and repeats the registry promotion condition.

The prompt also forbids ROI figures, pricing, and success odds. A registry measurement design is an
experiment contract, not a promised result.

## Replay contract

Replay validates every stored solution id, level, runnability value, and ordered methodology list
against the selected fixture. It also checks that the item's singular methodology is the
composition entry marked `primary`. Unknown ids, invented levels, non-runnable suggestions, stale
composition, a mismatched primary methodology, and incomplete unsupported reasons fail replay.

```sh
uv run agent-autosuggest-corpus --replay corpus/autosuggest-sft-v3.jsonl
```

The corpus filename moved from v2 to v3 because the output contract gained solution level,
runnability, and composition. A consuming image that copies a versioned corpus must update its copy
path in the same release.
