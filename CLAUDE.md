# agent-autosuggest-corpus — Claude rules

Generates the **SFT training corpus for the Auto-Suggest model**
([zetta#986](https://github.com/oramasearch/zetta/issues/986)). Read
[`zetta-bridge/docs/agent-scaffold.md`](../../docs/agent-scaffold.md) first.

## The rule that outranks everything else here

**Never read from `OramaSearch/agent-eval-profiles`.** Those profiles are the **test set** for the
model this corpus trains. Overlap makes every eval score a measurement of memory rather than
reading, and nothing afterwards can tell the difference.

`guard.py` enforces it — the run **refuses to start** if it cannot load the eval set, because
generating blind risks silent contamination. Near-misses count: a candidate sharing >60% of its
industry and table names with an eval profile is rejected.

## The output shape is a contract

An example is `(profile → project profile)`. The **project profile is the same shape the
project-creation agent consumes**, so Auto-Suggest and the project agent compose without a
translation layer that can drift.

## Quality checks, and why each exists

Every one was added because something weak got through:

- **`solution` must be a catalogue id** (`S-1`, `UC-13`), not prose. The id is what lets the
  project agent look up the correctness profile; *"Predict high-value purchase likelihood"* is the
  intent restated and tells it nothing.
- **Constraints must narrow the answer.** *"Must provide actionable insights"* is decoration —
  every suggestion claims it. A real constraint carries a number, a prohibition, or an ordering.
- **`fits_the_constraint` must not restate the constraint.** It says what about the **output**
  satisfies it: *"a ranked top-40, not a score per account"*.
- **A declined item's answer must match its reason.** *"Capture it first"* is only right for a
  discovered-but-uncaptured source. Data that lacks a label is not fixed by capturing it, and
  saying so teaches the wrong remedy.
- **Forecast must not be calibrated** — there is no probability to calibrate.
- **Supervised suggestions need a holdout**, or they cannot quote an honest figure.
- **Something must be declined.** A model that always suggests will suggest something worthless.

When a check rejects everything, **teach the prompt — do not loosen the check.** Tightening the
constraint rule rejected 6 of 6; explaining what a real constraint looks like brought it to 50%
with the remaining rejections all genuine.

## Running

```sh
uv sync
export EVAL_PROFILES=../agent-eval-generator/evals/profiles   # required; run refuses without it
export OPENAI_API_KEY=...                                     # or: source ~/.zshrc
uv run agent-autosuggest-corpus --examples 20
```

uv, never pip. Prompts are versioned — a prompt change is a model change.
