# SPDX-License-Identifier: MIT

"""Keep the training corpus out of the eval set.

The eval profiles (agent-eval-generator's `evals/profiles/`, passed as `EVAL_PROFILES`) are the TEST SET for the model this corpus trains. Training on it
makes every eval score meaningless: the model would be scored on situations it had memorised, and
nobody could tell a model that reads data from one that recalls an answer key.

Writing that rule down is not enough — the publish rule was written in two places and still missed
three checkpoints in a row. So it is checked, and nothing publishes without passing.

The check is deliberately conservative. A near-miss is treated as contamination, because the cost
of a false positive is regenerating one situation and the cost of a false negative is an eval
suite that silently measures nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _normalize(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z]{4,}", text.lower())}


class EvalGuard:
    """Knows what is in the eval set, so training never touches it."""

    def __init__(self, eval_profiles: Path | None = None) -> None:
        self.ids: set[str] = set()
        self.industries: set[str] = set()
        self.fingerprints: list[set[str]] = []
        if eval_profiles and eval_profiles.exists():
            for profile_json in eval_profiles.glob("*/profile.json"):
                try:
                    profile = json.loads(profile_json.read_text())
                except json.JSONDecodeError:
                    continue
                self.ids.add(str(profile.get("profile_id", "")).lower())
                self.industries.add(str(profile.get("industry", "")).lower())
                tables = " ".join(
                    f"{t.get('database')}.{t.get('name')}" for t in profile.get("tables", []))
                self.fingerprints.append(_normalize(f"{profile.get('industry','')} {tables}"))

    def contamination(self, candidate: dict) -> str | None:
        """Return why this candidate overlaps the eval set, or None if it is clean."""
        if not self.ids:
            return "eval set not loaded — refusing to generate blind rather than risk overlap"

        profile_id = str(candidate.get("profile_id", "")).lower()
        industry = str(candidate.get("industry", "")).lower()
        if profile_id in self.ids:
            return f"profile_id `{profile_id}` is in the eval set"
        if industry and industry in self.industries:
            return f"industry `{industry}` is in the eval set"

        tables = " ".join(
            f"{t.get('database')}.{t.get('name')}" for t in candidate.get("tables", []))
        fingerprint = _normalize(f"{industry} {tables}")
        if not fingerprint:
            return None
        for other in self.fingerprints:
            overlap = len(fingerprint & other) / max(1, len(fingerprint | other))
            if overlap > 0.6:
                return (f"{overlap:.0%} of its industry and table names match an eval profile — "
                        f"too close to train on")
        return None
