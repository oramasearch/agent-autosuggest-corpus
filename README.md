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
    "methodology": "gbdt",
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
    "intent": "forecast usage per feature",
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

Nothing generated yet.

## In lake (lake#263)

A library agent of lake: a person asks it for one training example ("a quality manager at a
contract bakery"). One turn is one `generate_one` (the loop body of the CLI), steered by the request
and checked exactly as the CLI checks it: the contamination guard against the eval set, then the
example's defects, with one retry. It answers `{reply, result: {kind: autosuggest_example,
example}}` and **writes nothing to the corpus**; growing the corpus stays the CLI's.

- `service.py`: `run_turn`, and `agent-autosuggest-corpus-serve` (`GET /healthz` is 503 without
  the eval set; `POST /v1/turn`).
- `lake_bus.py`: takes turns from lake's agent bus with its own `AGENT_BUS_TOKEN`.
- `Dockerfile`: private image `oramasearch/agent-autosuggest-corpus`; the eval set is the named
  build context `evals` (`--local evals=../agent-eval-generator/evals/profiles --opt
  context:evals=local:evals`).
- Tests: `uv run --group dev pytest`.
