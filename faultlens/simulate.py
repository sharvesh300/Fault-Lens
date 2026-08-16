"""Synthetic trace generator -- the ground-truth oracle.

We sample from exactly the generative story the model assumes, so every true
value (which step was broken, which broken step actually killed the episode)
is known. Inference only ever sees the detectors and the outcome bit.
"""

from __future__ import annotations

import random

from faultlens.episode import Episode, Step
from faultlens.params import DEFAULT_PARAMS, ModelParams

# A few plausible agent shapes. Each is a linear chain of component types.
TOPOLOGIES: list[list[str]] = [
    ["planner", "retriever", "sql_writer", "db_tool", "summarizer", "formatter"],
    ["planner", "retriever", "summarizer", "formatter"],
    ["retriever", "sql_writer", "db_tool", "formatter"],
    ["planner", "retriever", "retriever", "summarizer", "formatter"],
]


def sample_episode(
    episode_id: str,
    rng: random.Random,
    params: ModelParams = DEFAULT_PARAMS,
    topology: list[str] | None = None,
) -> Episode:
    """Ancestral sampling: R ~ prior, E ~ detector channel, Y ~ leaky noisy-OR."""
    component_types = topology or rng.choice(TOPOLOGIES)

    steps: list[Step] = []
    truth: dict[str, str] = {}
    killers: list[str] = []

    for i, ctype in enumerate(component_types, start=1):
        step_id = f"s{i}"
        broken = rng.random() < params.prior_of(ctype)
        truth[step_id] = "fail" if broken else "ok"

        detectors = {}
        for detector in params.detectors:
            rate = params.alpha[detector] if broken else params.beta[detector]
            detectors[detector] = int(rng.random() < rate)

        # noisy-OR: each broken step independently gets a chance to kill the run
        if broken and rng.random() < params.criticality_of(ctype):
            killers.append(step_id)

        steps.append(
            Step(
                step_id=step_id,
                component_type=ctype,
                parent_ids=[f"s{i - 1}"] if i > 1 else [],
                detectors=detectors,
            )
        )

    leak_fired = rng.random() < params.leak
    failed = bool(killers) or leak_fired
    # The culprit is the first cause that fired; "leak" means the failure came
    # from outside the model.
    culprit = killers[0] if killers else ("leak" if leak_fired else None)

    return Episode(
        episode_id=episode_id,
        steps=steps,
        outcome="fail" if failed else "ok",
        true_reliability=truth,
        true_culprit=culprit,
    )


def sample_corpus(
    n: int, seed: int = 0, params: ModelParams = DEFAULT_PARAMS
) -> list[Episode]:
    rng = random.Random(seed)
    return [sample_episode(f"ep{i:05d}", rng, params) for i in range(n)]
