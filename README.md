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
