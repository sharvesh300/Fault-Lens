"""The O(n) closed form of the same posterior pgmpy computes by elimination.

Because the leaky noisy-OR factorises over steps, the normalisers are products
rather than sums over 2^n states, so every blame posterior is a difference of
two closed-form products. pgmpy is the readable reference implementation; this
is the one that scales to long episodes and makes the E-step of EM free.

Kept in sync by `tests/test_closed_form.py`, which asserts the two agree.
"""

from __future__ import annotations

from faultlens.episode import Episode
from faultlens.infer import Attribution
from faultlens.params import ModelParams


def _weights(episode: Episode, params: ModelParams) -> list[tuple[float, float]]:
    """a_i = P(R_i=ok)·P(E_i|ok),  b_i = P(R_i=fail)·P(E_i|fail)."""
    out = []
    for step in episode.steps:
        prior = params.prior_of(step.component_type)
        a, b = 1.0 - prior, prior
        for detector in params.detectors:
            alpha, beta = params.alpha[detector], params.beta[detector]
            if step.fired(detector):
                a *= beta
                b *= alpha
            else:
                a *= 1.0 - beta
                b *= 1.0 - alpha
        out.append((a, b))
    return out


def explain_closed_form(episode: Episode, params: ModelParams) -> Attribution:
    weights = _weights(episode, params)
    q = [params.criticality_of(s.component_type) for s in episode.steps]

    z_any = 1.0
    z_ok = 1.0 - params.leak
    for (a, b), q_i in zip(weights, q):
        z_any *= a + b
        z_ok *= a + b * (1.0 - q_i)

    # P(E, Y) marginalising over all 2^n assignments of R
    z_evidence = z_ok if episode.outcome == "ok" else z_any - z_ok

    blame = {}
    for step, (a, b), q_i in zip(episode.steps, weights, q):
        joint_ok = b * (1.0 - q_i) / (a + b * (1.0 - q_i)) * z_ok
        joint_any = b / (a + b) * z_any
        joint = joint_ok if episode.outcome == "ok" else joint_any - joint_ok
        blame[step.step_id] = joint / z_evidence

    prod_a = 1.0
    for a, _ in weights:
        prod_a *= a
    all_ok = prod_a * (1.0 - params.leak if episode.outcome == "ok" else params.leak)

    return Attribution(
        episode_id=episode.episode_id,
        blame=blame,
        unmodelled=all_ok / z_evidence,
    )
