"""Compile one episode into one discrete Bayesian network (pgmpy).

Variables, for an episode with n steps and m detectors:

    R_i   latent   step reliability          states: ok | fail
    E_ij  observed detector firing           states: no | yes
    Y     observed terminal outcome          states: ok | fail

Edges:  R_i -> E_ij   (one per detector)      R_i -> Y  (all steps)

Y's CPD is a leaky noisy-OR written out as a full table:

    P(Y = ok | R) = (1 - leak) * prod_{i : R_i = fail} (1 - q_i)

The execution DAG of the trace determines *which steps exist* and supplies the
features/criticality of each; it is deliberately not used as edges between the
R_i.  Modelling downstream corruption as an explicit propagation variable on
each edge is the documented future extension.
"""

from __future__ import annotations

import itertools

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from faultlens.episode import Episode
from faultlens.params import ModelParams

R_STATES = ["ok", "fail"]
E_STATES = ["no", "yes"]
Y_STATES = ["ok", "fail"]


def r_node(step_id: str) -> str:
    return f"R_{step_id}"


def e_node(step_id: str, detector: str) -> str:
    return f"E_{step_id}_{detector}"


Y_NODE = "Y"


def build_network(episode: Episode, params: ModelParams) -> DiscreteBayesianNetwork:
    """Return the Bayesian network for a single episode."""
    edges: list[tuple[str, str]] = []
    for step in episode.steps:
        for detector in params.detectors:
            edges.append((r_node(step.step_id), e_node(step.step_id, detector)))
        edges.append((r_node(step.step_id), Y_NODE))

    model = DiscreteBayesianNetwork(edges)
    cpds: list[TabularCPD] = []

    for step in episode.steps:
        r = r_node(step.step_id)
        p_fail = params.prior_of(step.component_type)
        cpds.append(
            TabularCPD(
                variable=r,
                variable_card=2,
                values=[[1.0 - p_fail], [p_fail]],
                state_names={r: R_STATES},
            )
        )
        # Detector channel: the Dawid-Skene noisy-annotator CPD.
        for detector in params.detectors:
            a = params.alpha[detector]  # P(fire | fail)
            b = params.beta[detector]  # P(fire | ok)
            e = e_node(step.step_id, detector)
            cpds.append(
                TabularCPD(
                    variable=e,
                    variable_card=2,
                    # columns follow R = (ok, fail)
                    values=[[1.0 - b, 1.0 - a], [b, a]],
                    evidence=[r],
                    evidence_card=[2],
                    state_names={e: E_STATES, r: R_STATES},
                )
            )

    cpds.append(_noisy_or_cpd(episode, params))
    model.add_cpds(*cpds)
    model.check_model()
    return model


def _noisy_or_cpd(episode: Episode, params: ModelParams) -> TabularCPD:
    """Full 2^n table for the leaky noisy-OR outcome CPD."""
    parents = [r_node(s.step_id) for s in episode.steps]
    q = [params.criticality_of(s.component_type) for s in episode.steps]

    p_ok: list[float] = []
    # pgmpy column order: last evidence variable varies fastest, which is what
    # itertools.product gives us.
    for assignment in itertools.product([0, 1], repeat=len(parents)):
        prob_ok = 1.0 - params.leak
        for broken, q_i in zip(assignment, q):
            if broken:
                prob_ok *= 1.0 - q_i
        p_ok.append(prob_ok)

    values = [p_ok, [1.0 - p for p in p_ok]]
    state_names = {Y_NODE: Y_STATES}
    state_names.update({p: R_STATES for p in parents})
    return TabularCPD(
        variable=Y_NODE,
        variable_card=2,
        values=values,
        evidence=parents,
        evidence_card=[2] * len(parents),
        state_names=state_names,
    )


def evidence_of(episode: Episode, params: ModelParams) -> dict[str, str]:
    """Everything we actually observe: detector firings plus the outcome bit."""
    evidence: dict[str, str] = {}
    for step in episode.steps:
        for detector in params.detectors:
            evidence[e_node(step.step_id, detector)] = E_STATES[step.fired(detector)]
    evidence[Y_NODE] = episode.outcome
    return evidence
