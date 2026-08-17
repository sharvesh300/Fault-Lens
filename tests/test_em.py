"""EM must recover parameters from data we generated ourselves.

If these fail, nothing downstream is trustworthy -- more data would only buy
more confident nonsense.
"""

from __future__ import annotations

import pytest

from faultlens.em import fit, mean_absolute_error, parameter_errors
from faultlens.params import DEFAULT_PARAMS
from faultlens.simulate import sample_corpus


@pytest.fixture(scope="module")
def fitted():
    return fit(sample_corpus(3000, seed=42), n_restarts=3, seed=1)


def test_log_likelihood_never_decreases():
    """The EM guarantee. A violation means the M-step is wrong."""
    result = fit(sample_corpus(800, seed=3), n_restarts=1, seed=0)
    assert result.monotone
    assert result.history[-1] > result.history[0]


def test_converges_before_the_iteration_cap(fitted):
    assert fitted.converged


def test_restarts_agree(fitted):
    """No local-optimum problem: every restart should find the same optimum."""
    spread = max(fitted.restarts) - min(fitted.restarts)
    assert spread < 1e-3 * abs(fitted.log_likelihood)


def test_recovers_component_priors(fitted):
    """The P1 exit gate: MAE on r_c below 0.05."""
    errors = parameter_errors(fitted.params, DEFAULT_PARAMS)
    assert mean_absolute_error(errors["prior_fail"]) < 0.05


def test_recovers_detector_parameters(fitted):
    errors = parameter_errors(fitted.params, DEFAULT_PARAMS)
    assert mean_absolute_error(errors["alpha"]) < 0.10
    assert mean_absolute_error(errors["beta"]) < 0.05


def test_recovers_leak(fitted):
    assert fitted.params.leak == pytest.approx(DEFAULT_PARAMS.leak, abs=0.05)


def test_no_label_flip(fitted):
    """alpha > beta must survive fitting, or 'fail' and 'ok' have swapped."""
    for detector in DEFAULT_PARAMS.detectors:
        assert fitted.params.alpha[detector] > fitted.params.beta[detector]


def test_recovers_the_fix_list_ranking(fitted):
    """What the product actually ships: components ranked by expected damage."""

    def ranking(params):
        return [
            c
            for c, _ in sorted(
                (
                    (c, params.prior_fail[c] * params.criticality_of(c))
                    for c in params.prior_fail
                ),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ]

    assert ranking(fitted.params) == ranking(DEFAULT_PARAMS)


def test_learned_parameters_drive_inference(fitted):
    """End to end: fit, then explain an episode with the learned parameters."""
    from faultlens.episode import EPISODE_1471
    from faultlens.infer import explain

    attribution = explain(EPISODE_1471, fitted.params)
    assert attribution.argmax() == "s2"
