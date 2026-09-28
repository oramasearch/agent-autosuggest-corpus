"""Generate the Auto-Suggest SFT corpus.

One example is (profile -> project profile). The profile is invented here; the project profile is
the suggested approach for that data, in the shape the project-creation agent consumes later.

Nothing is written without passing the contamination guard. The eval profiles are the test set for
this model, and a corpus that overlaps them turns every eval score into a measurement of memory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent_autosuggest_corpus.guard import EvalGuard
from agent_autosuggest_corpus.methodologies import FIXTURE, CapabilityError, Registry
from agent_autosuggest_corpus.methodologies import defects as methodology_defects
from agent_autosuggest_corpus.methodologies import enforce as enforce_methodologies
from agent_autosuggest_corpus.methodologies import load_fixture

PROMPT_VERSION = "v2"
PROMPTS = Path(__file__).parent / "prompts"


def _parse(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    body = fence.group(1) if fence else text[text.find("{"): text.rfind("}") + 1]
    try:
        example = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    return example if example.get("profile_id") and example.get("project_profile") else None


def _defects(example: dict[str, Any], registry: Registry) -> list[str]:
    """Cheap checks that a training example is worth learning from."""
    problems = methodology_defects(example, registry)
    # A constraint that narrows nothing makes the whole example untestable: the suggestion cannot
    # be judged against it, and the model learns that constraints are decoration.
    for constraint in (example.get("persona") or {}).get("constraints", []):
        text = str(constraint).lower()
        narrows = any(char.isdigit() for char in text) or any(
            word in text for word in ("cannot", "must not", "only", "never", "no more than",
                                       "before", "without", "requires", "every"))
        if not narrows:
            problems.append(f"constraint `{str(constraint)[:50]}` does not narrow what a good "
                            f"answer is")
    project = example.get("project_profile") or {}
    suggested = project.get("suggested") or []
    if not suggested:
        problems.append("no suggestion at all")
    if not project.get("declined"):
        # A model that never declines will suggest something worthless when asked.
        problems.append("nothing declined — declining is a first-class output")
    for item in suggested:
        if not item.get("why_this_data_supports_it"):
            problems.append(f"`{item.get('intent','?')[:40]}` does not say what makes the data support it")
        shape = item.get("pipeline_shape") or {}
        methodology = registry.entry(item.get("methodology")) or {}
        family = str(methodology.get("family") or "").lower()
        model_kinds = {str(kind).lower() for kind in methodology.get("model_kinds") or []}
        # Forecast has no probability to calibrate; calibrating one is the churn recipe misapplied.
        if "forecast" in family and shape.get("calibrated"):
            problems.append("a forecast is calibrated — there is no probability to calibrate")
        supervised = "classification" in family or bool(
            model_kinds & {"gbdt", "classifier", "classification"}
        )
        if supervised and not shape.get("holdout"):
            problems.append("a supervised suggestion with no holdout cannot quote an honest figure")
        # The catalogue id, not a restatement of the intent. "Predict high-value purchase
        # likelihood" is the intent again; S-2 is a module with a known correctness profile, and
        # only the id lets the project agent look up what the pipeline must satisfy.
        if not re.fullmatch(r"(S|M|UC)-\d+", str(item.get("solution", "")).strip()):
            problems.append(f"solution `{item.get('solution')}` is not a catalogue id (S-n/UC-n)")
        # Restating the constraint is not showing it was honoured. "Provides actionable insights"
        # against a constraint of "must provide actionable insights" says nothing about the output.
        fits = str(item.get("fits_the_constraint", "")).lower()
        for constraint in (example.get("persona") or {}).get("constraints", []):
            if fits and fits.strip(" .") in str(constraint).lower():
                problems.append("`fits_the_constraint` restates the constraint instead of saying "
                                "what about the OUTPUT satisfies it")

    for item in project.get("declined") or []:
        reason = str(item.get("reason", "")).lower()
        answer = str(item.get("answer", "")).lower()
        uncaptured = "captur" in reason or "discovered" in reason
        if uncaptured and "cannot be done" in answer:
            problems.append("an uncaptured source was answered 'cannot be done' rather than "
                            "'capture it first'")
        # And the inverse, which is the one that just slipped through: data that genuinely lacks
        # the shape is NOT fixed by capturing it, and saying so teaches the wrong remedy.
        if not uncaptured and "capture it first" in answer:
            problems.append(f"`{item.get('intent','?')[:40]}` was answered 'capture it first' but "
                            f"the reason is not about capture — capturing will not fix it")
    return problems


def generate_one(
    client: Any,
    model: str,
    guard: EvalGuard,
    request: str = "",
    seen: list[str] | None = None,
    registry: Registry | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """One training example, or None and why it was not kept (unparseable, contaminated by the
    eval set, or defective). A person's `request` (lake#263) steers what it is about."""
    registry = registry or load_fixture()
    system = (PROMPTS / f"{PROMPT_VERSION}.md").read_text() + registry.context()
    # Tell it what is off-limits BEFORE it spends a call. 12 of 40 generations were rejected
    # for landing on an industry already in the eval set — the guard caught every one, but
    # rejecting after the fact wastes the call and teaches the model nothing.
    taken = sorted(i for i in guard.industries if i)[:60]
    avoid = ("\n\nThese industries belong to the EVALUATION set and are off-limits — a training "
             "example in any of them would make the eval meaningless:\n" + ", ".join(taken))
    if seen:
        avoid += ("\n\nAlready generated here, pick something further away:\n"
                  + "\n".join(f"- {s}" for s in seen))
    ask = "Produce one training example." + (f" The person asking wants: {request}" if request else "")
    response = client.chat.completions.create(model=model, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": ask + avoid}])
    example = _parse(response.choices[0].message.content)
    if example is None:
        return None, "unparseable — skipped"
    overlap = guard.contamination(example)
    if overlap:
        return None, f"{example['profile_id']} CONTAMINATED · {overlap}"
    example = enforce_methodologies(example, registry)
    defects = _defects(example, registry)
    if defects:
        return None, f"{example['profile_id']} REJECTED · {defects[0]}"
    example["prompt_version"] = PROMPT_VERSION
    return example, "kept"


def replay(path: Path, registry: Registry) -> list[str]:
    """Validate stored examples against the same registry contract as generation."""
    problems: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        return [f"cannot read {path}: {error}"]
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            example = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"{path}:{number}: invalid JSON: {error.msg}")
            continue
        for defect in _defects(example, registry):
            problems.append(f"{path}:{number}: {defect}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Auto-Suggest SFT examples.")
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--model", default=os.environ.get("AUTOSUGGEST_MODEL", "gpt-4o"))
    parser.add_argument("--out", type=Path, default=Path("corpus"))
    parser.add_argument("--eval-profiles", type=Path,
                        default=Path(os.environ.get("EVAL_PROFILES", "")))
    parser.add_argument("--methodologies-fixture", type=Path, default=FIXTURE)
    parser.add_argument("--replay", type=Path,
                        help="validate a JSONL corpus against the methodologies fixture")
    args = parser.parse_args()

    try:
        registry = load_fixture(args.methodologies_fixture)
    except CapabilityError as error:
        print(f"refusing to run: {error}")
        return 2

    if args.replay:
        problems = replay(args.replay, registry)
        for problem in problems:
            print(problem)
        if problems:
            print(f"replay failed: {len(problems)} defect(s)")
            return 1
        print(f"replay passed: {args.replay}")
        return 0

    guard = EvalGuard(args.eval_profiles)
    if not guard.ids:
        print("refusing to run: the eval set could not be read, so contamination cannot be "
              "checked. Set EVAL_PROFILES to the eval profiles directory.")
        return 2

    client = OpenAI()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"autosuggest-sft-{PROMPT_VERSION}.jsonl"

    seen: list[str] = []
    kept = rejected = 0
    for index in range(args.examples):
        example, reason = generate_one(client, args.model, guard, seen=seen, registry=registry)
        if example is None:
            print(f"[{index+1}/{args.examples}] {reason}")
            rejected += 1
            continue
        with path.open("a") as handle:
            handle.write(json.dumps(example) + "\n")
        seen.append(f"{example['profile_id']} ({example.get('industry')})")
        kept += 1
        n = len(example["project_profile"].get("suggested", []))
        print(f"[{index+1}/{args.examples}] {example['profile_id']} · {example.get('industry')} "
              f"· {n} suggested · kept")

    print(f"kept={kept} rejected={rejected} -> {path}")
    return 0
