"""The O(n) formula must agree with pgmpy's variable elimination exactly."""

from __future__ import annotations

import pytest

from faultlens.closed_form import explain_closed_form
from faultlens.episode import EPISODE_1471, chain_episode
from faultlens.infer import explain
from faultlens.params import DEFAULT_PARAMS
from faultlens.simulate import sample_corpus

TOL = 1e-10


def _assert_same(episode):
    exact = explain(episode, DEFAULT_PARAMS)
    fast = explain_closed_form(episode, DEFAULT_PARAMS)
    for step_id, value in exact.blame.items():
        assert fast.blame[step_id] == pytest.approx(value, abs=TOL)
    assert fast.unmodelled == pytest.approx(exact.unmodelled, abs=TOL)


def test_worked_example():
    _assert_same(EPISODE_1471)


def test_successful_episode():
    _assert_same(
        chain_episode(
            "ok-1",
            ["planner", "retriever", "summarizer", "formatter"],
            outcome="ok",
            fired={"s2": ["low_retrieval"]},
        )
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_sampled_episodes(seed):
    for episode in sample_corpus(10, seed=seed):
        _assert_same(episode)
