"""Canonical episode record: what the adapters produce and the model consumes."""

from __future__ import annotations

from dataclasses import dataclass, field

from faultlens.params import DETECTORS


@dataclass
class Step:
    """One step instance inside one episode."""

    step_id: str
    component_type: str
    parent_ids: list[str] = field(default_factory=list)
    # detector name -> 1 if it fired, 0 otherwise
    detectors: dict[str, int] = field(default_factory=dict)

    def fired(self, detector: str) -> int:
        return int(self.detectors.get(detector, 0))


@dataclass
class Episode:
    """A normalised trace plus the single bit of supervision we ever get."""

    episode_id: str
    steps: list[Step]
    outcome: str  # "ok" | "fail"
    # only present in synthetic data; never used by inference
    true_reliability: dict[str, str] = field(default_factory=dict)
    true_culprit: str | None = None

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        raise KeyError(step_id)

    @property
    def step_ids(self) -> list[str]:
        return [s.step_id for s in self.steps]


def chain_episode(
    episode_id: str,
    component_types: list[str],
    outcome: str = "fail",
    fired: dict[str, list[str]] | None = None,
) -> Episode:
    """Build a linear-chain episode: s1 -> s2 -> ... -> sn.

    `fired` maps step_id -> list of detector names that fired on that step.
    """
    fired = fired or {}
    steps: list[Step] = []
    for i, ctype in enumerate(component_types, start=1):
        step_id = f"s{i}"
        steps.append(
            Step(
                step_id=step_id,
                component_type=ctype,
                parent_ids=[f"s{i - 1}"] if i > 1 else [],
                detectors={d: int(d in fired.get(step_id, [])) for d in DETECTORS},
            )
        )
    return Episode(episode_id=episode_id, steps=steps, outcome=outcome)


# The worked example from the design doc: the retriever fetched an outdated
# churn definition, every later step was correct given its input, and the only
# label anyone provides is `outcome = fail`.
EPISODE_1471 = chain_episode(
    episode_id="1471",
    component_types=[
        "planner",
        "retriever",
        "sql_writer",
        "db_tool",
        "summarizer",
        "formatter",
    ],
    outcome="fail",
    fired={"s2": ["low_retrieval"], "s5": ["length_outlier"]},
)
