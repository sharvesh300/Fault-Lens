"""The enumeration test and friends.

The highest-value test in the project: whatever pgmpy's variable elimination
returns must equal brute-force enumeration over all 2^n assignments of R,
computed straight from the mathematical definition of the model.
"""

from __future__ import annotations

import itertools
import random

import pytest

from faultlens.episode import EPISODE_1471, Episode, chain_episode
from faultlens.infer import explain
from faultlens.network import build_network
from faultlens.params import DEFAULT_PARAMS, ModelParams
from faultlens.simulate import sample_corpus

TOL = 1e-10


def brute_force(episode: Episode, params: ModelParams) -> tuple[dict[str, float], float]:
    """P(R_i = fail | E, Y) and P(all R = ok | E, Y) by explicit enumeration."""
    steps = episode.steps
    total = 0.0
    per_step = {s.step_id: 0.0 for s in steps}
    all_ok = 0.0

    for assignment in itertools.product([0, 1], repeat=len(steps)):
        weight = 1.0
        p_ok = 1.0 - params.leak
        for step, broken in zip(steps, assignment):
            prior = params.prior_of(step.component_type)
            weight *= prior if broken else 1.0 - prior
            for detector in params.detectors:
                rate = params.alpha[detector] if broken else params.beta[detector]
                weight *= rate if step.fired(detector) else 1.0 - rate
            if broken:
                p_ok *= 1.0 - params.criticality_of(step.component_type)

        weight *= p_ok if episode.outcome == "ok" else 1.0 - p_ok
        total += weight
        for step, broken in zip(steps, assignment):
            if broken:
                per_step[step.step_id] += weight
        if not any(assignment):
            all_ok += weight

    return {k: v / total for k, v in per_step.items()}, all_ok / total


def test_network_is_valid():
    model = build_network(EPISODE_1471, DEFAULT_PARAMS)
    assert model.check_model()
    # n step priors + n*m detector CPDs + the outcome CPD
    n, m = len(EPISODE_1471.steps), len(DEFAULT_PARAMS.detectors)
    assert len(model.get_cpds()) == n + n * m + 1


def test_exact_inference_matches_enumeration_worked_example():
    got = explain(EPISODE_1471, DEFAULT_PARAMS)
    want_blame, want_unmodelled = brute_force(EPISODE_1471, DEFAULT_PARAMS)
    for step_id, value in want_blame.items():
        assert got.blame[step_id] == pytest.approx(value, abs=TOL)
    assert got.unmodelled == pytest.approx(want_unmodelled, abs=TOL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_inference_matches_enumeration_on_sampled_episodes(seed):
    for episode in sample_corpus(6, seed=seed):
        got = explain(episode, DEFAULT_PARAMS)
        want_blame, want_unmodelled = brute_force(episode, DEFAULT_PARAMS)
        for step_id, value in want_blame.items():
            assert got.blame[step_id] == pytest.approx(value, abs=TOL)
        assert got.unmodelled == pytest.approx(want_unmodelled, abs=TOL)


def test_random_parameters_still_match_enumeration():
    """Guards against accidentally testing only symmetric/default numbers."""
    rng = random.Random(3)
    params = ModelParams(
        prior_fail={c: rng.uniform(0.02, 0.4) for c in ("planner", "retriever", "formatter")},
        criticality={c: rng.uniform(0.3, 0.99) for c in ("planner", "retriever", "formatter")},
        alpha={d: rng.uniform(0.3, 0.9) for d in DEFAULT_PARAMS.detectors},
        beta={d: rng.uniform(0.01, 0.2) for d in DEFAULT_PARAMS.detectors},
        leak=0.13,
    )
    episode = chain_episode(
        "rand",
        ["planner", "retriever", "formatter"],
        outcome="fail",
        fired={"s2": ["low_retrieval", "retry_occurred"], "s3": ["schema_violation"]},
    )
    got = explain(episode, params)
    want_blame, want_unmodelled = brute_force(episode, params)
    for step_id, value in want_blame.items():
        assert got.blame[step_id] == pytest.approx(value, abs=TOL)
    assert got.unmodelled == pytest.approx(want_unmodelled, abs=TOL)


def test_explaining_away():
    """Upstream evidence must pull blame off the weak downstream flag."""
    with_upstream = explain(EPISODE_1471, DEFAULT_PARAMS)
    without = chain_episode(
        "1471-no-d2",
        [s.component_type for s in EPISODE_1471.steps],
        outcome="fail",
        fired={"s5": ["length_outlier"]},
    )
    without_upstream = explain(without, DEFAULT_PARAMS)
    assert with_upstream.blame["s5"] < without_upstream.blame["s5"]
    assert with_upstream.argmax() == "s2"


def test_successful_episode_lowers_blame():
    ok_episode = chain_episode(
        "ok-1",
        [s.component_type for s in EPISODE_1471.steps],
        outcome="ok",
        fired={"s2": ["low_retrieval"]},
    )
    fail_episode = EPISODE_1471
    assert explain(ok_episode).blame["s2"] < explain(fail_episode).blame["s2"]


def test_simulator_respects_its_own_parameters():
    """A crude check that sampling matches the priors it was given."""
    corpus = sample_corpus(4000, seed=5)
    retriever_steps = [
        s for ep in corpus for s in ep.steps if s.component_type == "retriever"
    ]
    broken = [
        1
        for ep in corpus
        for s in ep.steps
        if s.component_type == "retriever" and ep.true_reliability[s.step_id] == "fail"
    ]
    rate = len(broken) / len(retriever_steps)
    assert rate == pytest.approx(DEFAULT_PARAMS.prior_fail["retriever"], abs=0.03)
