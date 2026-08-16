"""Inference: blame posteriors for one episode, by variable elimination."""

from __future__ import annotations

from dataclasses import dataclass

from pgmpy.inference import VariableElimination

from faultlens.episode import Episode
from faultlens.network import build_network, evidence_of, r_node
from faultlens.params import DEFAULT_PARAMS, ModelParams


@dataclass
class Attribution:
    """Result of explaining one episode."""

    episode_id: str
    # P(R_i = fail | detectors, Y) per step
    blame: dict[str, float]
    # P(every step was fine | detectors, Y) -- the episode failed for a cause
    # the model does not represent
    unmodelled: float

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.blame.items(), key=lambda kv: kv[1], reverse=True)

    def argmax(self) -> str:
        return self.ranked()[0][0]


def explain(episode: Episode, params: ModelParams = DEFAULT_PARAMS) -> Attribution:
    """Posterior probability that each step was broken, given what we saw."""
    model = build_network(episode, params)
    engine = VariableElimination(model)
    evidence = evidence_of(episode, params)

    blame = {}
    for step in episode.steps:
        node = r_node(step.step_id)
        marginal = engine.query([node], evidence=evidence, show_progress=False)
        blame[step.step_id] = float(marginal.get_value(**{node: "fail"}))

    nodes = [r_node(s.step_id) for s in episode.steps]
    joint = engine.query(nodes, evidence=evidence, show_progress=False)
    unmodelled = float(joint.get_value(**{n: "ok" for n in nodes}))

    return Attribution(
        episode_id=episode.episode_id, blame=blame, unmodelled=unmodelled
    )


def format_attribution(episode: Episode, attribution: Attribution) -> str:
    """The `faultlens explain` view of one episode."""
    types = {s.step_id: s.component_type for s in episode.steps}
    fired = {
        s.step_id: [d for d, v in s.detectors.items() if v] for s in episode.steps
    }
    lines = [f"episode {attribution.episode_id}  ·  Y = {episode.outcome}", ""]
    for step_id, p in attribution.ranked():
        flags = ", ".join(fired[step_id]) or "-"
        lines.append(f"  P({types[step_id]:<11} broken) = {p:.3f}   [{flags}]")
    lines.append(f"  P(unmodelled cause)      = {attribution.unmodelled:.3f}")
    return "\n".join(lines)
