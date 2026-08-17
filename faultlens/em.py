"""MAP-EM: learn the model parameters from outcome-only supervision.

Only phi (detector firings) and Y (the terminal pass/fail bit) are observed.
The per-step reliabilities R are latent, so we maximise the observed-data
likelihood by EM.

E-step
    For every episode compute gamma_i = P(R_i = fail | E, Y) exactly, using the
    O(n) formulation in `closed_form.py`. No sampling, no variational bound.

M-step
    Every update is closed form, including the noisy-OR ones. The trick is to
    augment the outcome with its causal-independence indicators: let Z_i = 1
    mean "step i was broken AND its fault actually killed the episode", and
    Z_0 = 1 mean "an unmodelled cause killed it". Given R, these are
    independent Bernoulli(q_i) and Bernoulli(lambda), and Y = fail iff any
    fired. With Z filled in, the outcome likelihood decomposes into per-step
    Bernoulli terms and q_c and lambda become weighted counts like everything
    else.

    The required expectation has a closed form as well. If Z_i fires the
    episode dies regardless of every other variable, so

        P(R_i = fail, Z_i = 1, E, Y = fail) = b_i * q_i * prod_{k != i} (a_k + b_k)
                                            = b_i * q_i * Z_any / (a_i + b_i)

    and for a successful episode no Z fired at all, so the expectation is zero.

The E-step produces probabilities, never hard labels. The M-step treats them as
fractional counts. That is exactly why the system can learn from a terminal
pass/fail bit without pretending to know the true step label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from faultlens.episode import Episode
from faultlens.params import DETECTORS, ModelParams

EPS = 1e-6


@dataclass
class BetaPrior:
    """Beta(a, b) pseudo-counts. Beta(1, 1) is uniform, i.e. plain ML."""

    a: float = 1.0
    b: float = 1.0

    def map_estimate(self, successes: float, total: float) -> float:
        num = successes + self.a - 1.0
        den = total + self.a + self.b - 2.0
        if den <= 0.0:
            return 0.5
        return float(np.clip(num / den, EPS, 1.0 - EPS))


@dataclass
class Priors:
    prior_fail: BetaPrior = field(default_factory=BetaPrior)
    criticality: BetaPrior = field(default_factory=BetaPrior)
    alpha: BetaPrior = field(default_factory=BetaPrior)
    beta: BetaPrior = field(default_factory=BetaPrior)
    leak: BetaPrior = field(default_factory=BetaPrior)


@dataclass
class EMResult:
    params: ModelParams
    log_likelihood: float
    history: list[float]
    n_iter: int
    converged: bool
    restarts: list[float]

    @property
    def monotone(self) -> bool:
        """EM guarantees the objective never decreases; check we got that."""
        return all(
            b >= a - 1e-9 for a, b in zip(self.history, self.history[1:])
        )


class _Corpus:
    """Episodes flattened into arrays so an EM iteration is pure numpy."""

    def __init__(self, episodes: list[Episode], detectors: tuple[str, ...]):
        self.detectors = detectors
        self.component_types = sorted(
            {s.component_type for ep in episodes for s in ep.steps}
        )
        index = {c: i for i, c in enumerate(self.component_types)}

        comp_idx: list[int] = []
        fired: list[list[int]] = []
        starts: list[int] = []
        failed: list[bool] = []

        for ep in episodes:
            starts.append(len(comp_idx))
            failed.append(ep.outcome == "fail")
            for step in ep.steps:
                comp_idx.append(index[step.component_type])
                fired.append([step.fired(d) for d in detectors])

        self.comp_idx = np.asarray(comp_idx, dtype=np.int64)
        self.fired = np.asarray(fired, dtype=bool)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.failed = np.asarray(failed, dtype=bool)
        self.n_steps = len(comp_idx)
        self.n_episodes = len(episodes)
        # per-step lookup of its episode, for broadcasting episode-level terms
        self.step_episode = np.repeat(np.arange(self.n_episodes), np.diff(
            np.append(self.starts, self.n_steps)
        ))

    def as_params(self, r, q, alpha, beta, leak) -> ModelParams:
        return ModelParams(
            prior_fail={c: float(r[i]) for i, c in enumerate(self.component_types)},
            criticality={c: float(q[i]) for i, c in enumerate(self.component_types)},
            alpha={d: float(alpha[j]) for j, d in enumerate(self.detectors)},
            beta={d: float(beta[j]) for j, d in enumerate(self.detectors)},
            leak=float(leak),
            detectors=self.detectors,
        )


def _e_step(corpus: _Corpus, r, q, alpha, beta, leak):
    """Return (gamma, zeta, leak_expectation, log_likelihood).

    gamma[s] = P(R_s = fail | E, Y)
    zeta[s]  = P(R_s = fail and step s killed the episode | E, Y)
    """
    # detector likelihoods per step under each hypothesis
    p_ok = np.prod(np.where(corpus.fired, beta, 1.0 - beta), axis=1)
    p_fail = np.prod(np.where(corpus.fired, alpha, 1.0 - alpha), axis=1)

    r_step = r[corpus.comp_idx]
    q_step = q[corpus.comp_idx]
    a = (1.0 - r_step) * p_ok
    b = r_step * p_fail

    any_term = a + b
    ok_term = a + b * (1.0 - q_step)

    # per-episode products
    z_any = np.multiply.reduceat(any_term, corpus.starts)
    z_ok = (1.0 - leak) * np.multiply.reduceat(ok_term, corpus.starts)

    p_episode = np.where(corpus.failed, z_any - z_ok, z_ok)
    p_episode = np.maximum(p_episode, 1e-300)
    log_likelihood = float(np.sum(np.log(p_episode)))

    # broadcast episode-level quantities back onto steps
    z_any_s = z_any[corpus.step_episode]
    z_ok_s = z_ok[corpus.step_episode]
    p_s = p_episode[corpus.step_episode]
    failed_s = corpus.failed[corpus.step_episode]

    t_any = b / any_term
    t_ok = b * (1.0 - q_step) / ok_term

    gamma = np.where(failed_s, (t_any * z_any_s - t_ok * z_ok_s) / p_s, t_ok)
    gamma = np.clip(gamma, 0.0, 1.0)

    # P(R_i = fail, this step fired the outcome | evidence)
    zeta = np.where(failed_s, (b * q_step / any_term) * z_any_s / p_s, 0.0)
    zeta = np.clip(zeta, 0.0, gamma)

    leak_expectation = np.where(corpus.failed, leak * z_any / p_episode, 0.0)
    leak_expectation = np.clip(leak_expectation, 0.0, 1.0)

    return gamma, zeta, leak_expectation, log_likelihood


def _m_step(corpus: _Corpus, gamma, zeta, leak_expectation, priors: Priors):
    n_types = len(corpus.component_types)
    n_det = len(corpus.detectors)

    soft_fail = np.bincount(corpus.comp_idx, weights=gamma, minlength=n_types)
    soft_kill = np.bincount(corpus.comp_idx, weights=zeta, minlength=n_types)
    counts = np.bincount(corpus.comp_idx, minlength=n_types).astype(float)

    r = np.array(
        [priors.prior_fail.map_estimate(soft_fail[i], counts[i]) for i in range(n_types)]
    )
    # criticality: of the mass that was broken, how much actually killed the run
    q = np.array(
        [
            priors.criticality.map_estimate(soft_kill[i], soft_fail[i])
            for i in range(n_types)
        ]
    )

    total_fail = float(gamma.sum())
    total_ok = float((1.0 - gamma).sum())
    alpha = np.empty(n_det)
    beta = np.empty(n_det)
    for j in range(n_det):
        fired_j = corpus.fired[:, j].astype(float)
        alpha[j] = priors.alpha.map_estimate(float(gamma @ fired_j), total_fail)
        beta[j] = priors.beta.map_estimate(float((1.0 - gamma) @ fired_j), total_ok)

    leak = priors.leak.map_estimate(float(leak_expectation.sum()), corpus.n_episodes)
    return r, q, alpha, beta, leak


def _initialise(corpus: _Corpus, rng: np.random.Generator, smart: bool):
    n_types = len(corpus.component_types)
    n_det = len(corpus.detectors)

    if smart:
        # Seed alpha/beta from the measured fire rates rather than randomly --
        # this matters more than it sounds.
        rates = corpus.fired.mean(axis=0)
        alpha = np.clip(rates * 3.0, 0.15, 0.90)
        beta = np.clip(rates * 0.5, 0.01, 0.30)
        r = np.full(n_types, 0.10)
        q = np.full(n_types, 0.80)
        leak = 0.10
    else:
        beta = rng.uniform(0.01, 0.20, n_det)
        # alpha > beta breaks the label-flip symmetry at initialisation
        alpha = np.clip(beta + rng.uniform(0.10, 0.60, n_det), 0.0, 0.95)
        r = rng.uniform(0.03, 0.30, n_types)
        q = rng.uniform(0.40, 0.95, n_types)
        leak = float(rng.uniform(0.02, 0.20))
    return r, q, alpha, beta, leak


def fit(
    episodes: list[Episode],
    detectors: tuple[str, ...] = DETECTORS,
    n_restarts: int = 5,
    max_iter: int = 500,
    tol: float = 1e-9,
    seed: int = 0,
    priors: Priors | None = None,
) -> EMResult:
    """Fit the model to episodes labelled only with their terminal outcome."""
    priors = priors or Priors()
    corpus = _Corpus(episodes, detectors)
    rng = np.random.default_rng(seed)

    best: EMResult | None = None
    restart_scores: list[float] = []

    for restart in range(n_restarts):
        r, q, alpha, beta, leak = _initialise(corpus, rng, smart=(restart == 0))
        history: list[float] = []
        converged = False

        for _ in range(max_iter):
            gamma, zeta, leak_exp, ll = _e_step(corpus, r, q, alpha, beta, leak)
            history.append(ll)
            # Relative tolerance: an absolute one silently stops converging as
            # the corpus grows, because the log-likelihood scales with it.
            threshold = tol * max(1.0, abs(ll))
            if len(history) > 1 and abs(history[-1] - history[-2]) < threshold:
                converged = True
                break
            r, q, alpha, beta, leak = _m_step(corpus, gamma, zeta, leak_exp, priors)

        restart_scores.append(history[-1])
        if best is None or history[-1] > best.log_likelihood:
            best = EMResult(
                params=corpus.as_params(r, q, alpha, beta, leak),
                log_likelihood=history[-1],
                history=history,
                n_iter=len(history),
                converged=converged,
                restarts=[],
            )

    assert best is not None
    best.restarts = restart_scores
    return best


def parameter_errors(
    learned: ModelParams, true: ModelParams
) -> dict[str, dict[str, float]]:
    """Absolute error per parameter, for the recovery check."""
    out: dict[str, dict[str, float]] = {"prior_fail": {}, "criticality": {}}
    for c, value in true.prior_fail.items():
        if c in learned.prior_fail:
            out["prior_fail"][c] = abs(learned.prior_fail[c] - value)
    for c, value in true.criticality.items():
        if c in learned.criticality:
            out["criticality"][c] = abs(learned.criticality[c] - value)
    out["alpha"] = {j: abs(learned.alpha[j] - v) for j, v in true.alpha.items()}
    out["beta"] = {j: abs(learned.beta[j] - v) for j, v in true.beta.items()}
    out["leak"] = {"leak": abs(learned.leak - true.leak)}
    return out


def mean_absolute_error(errors: dict[str, float]) -> float:
    values = list(errors.values())
    return math.fsum(values) / len(values) if values else 0.0
