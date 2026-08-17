# FaultLens

Probabilistic fault attribution for LLM agents. Given a trace and **one bit of
supervision** — did the task succeed? — infer which component was responsible,
as a calibrated posterior. No step-level labels, no LLM calls at inference time.

This repository currently holds the **engine**: the Bayesian network, exact
inference, MAP-EM, and a synthetic study showing the parameters are recoverable
from outcome-only supervision. It has not yet met a real trace.

## Quickstart

```bash
uv run main.py
```

```
episode 1471  ·  Y = fail

  P(retriever   broken) = 0.859   [low_retrieval]
  P(sql_writer  broken) = 0.013   [-]
  P(summarizer  broken) = 0.012   [length_outlier]
  P(db_tool     broken) = 0.005   [-]
  P(planner     broken) = 0.004   [-]
  P(formatter   broken) = 0.001   [-]
  P(unmodelled cause)   = 0.118
```

The summarizer's `length_outlier` flag is *explained away* once the retriever
accounts for the failure — the Bayes-net move a per-step scorer cannot make.

```python
from faultlens import explain, format_attribution
from faultlens.episode import EPISODE_1471

print(format_attribution(EPISODE_1471, explain(EPISODE_1471)))
```

## The model in one screen

Per episode, with `n` steps and `m = 5` cheap detectors:

| Variable | Status | Meaning |
| --- | --- | --- |
| `R_i` | **latent** | step `i` did its job correctly |
| `E_ij` | observed | detector `j` fired on step `i` |
| `Y` | observed | terminal outcome — the only supervision |

```
P(R_i = fail)              = r_c(i)                        component prior, tied by type
P(E_ij = 1 | R_i = fail)   = α_j                           detector sensitivity   (Dawid–Skene)
P(E_ij = 1 | R_i = ok)     = β_j                           false-positive rate
P(Y = ok | R)              = (1 − λ) · ∏_{i: R_i=fail} (1 − q_i)    leaky noisy-OR
```

Two documents, depending on what you want:

- [`docs/how-it-works.md`](docs/how-it-works.md) — how the network is built from
  a trace, worked end to end on one episode with the real arithmetic, plus the
  intuition behind each equation
- [`docs/model-spec.md`](docs/model-spec.md) — the compact formal reference: CPDs,
  the exact O(n) posterior, the EM derivation, and the assumptions

## Layout

```
faultlens/params.py       fleet parameters (r_c, q_c, α, β, λ)
faultlens/episode.py      canonical episode record + the worked example
faultlens/network.py      episode -> pgmpy DiscreteBayesianNetwork
faultlens/infer.py        blame posteriors by variable elimination
faultlens/closed_form.py  the same posteriors in O(n), no 2^n table
faultlens/simulate.py     generative sampler = the ground-truth oracle
faultlens/em.py           MAP-EM, closed-form throughout
experiments/              E1 parameter recovery
tests/                    enumeration test, closed-form agreement, recovery
docs/how-it-works.md      walkthrough: trace -> DAG -> network -> posterior
docs/model-spec.md        mathematical specification
```

## Does the learning actually work?

```bash
uv run experiments/e1_parameter_recovery.py
```

Fits on synthetic corpora where every true parameter is known, showing the model
nothing but one pass/fail bit per episode:

| n | MAE on `r_c` | fix-list ranking |
| --- | --- | --- |
| 500 | 0.0059 | |
| 2000 | 0.0042 | |
| 5000 | **0.0022** | recovered exactly |

Gate was `MAE < 0.05` at 5,000 episodes. Converges in 44 iterations and 0.3 s,
log-likelihood monotone, all restarts reaching the same optimum.

Criticality `q_c` is the weakly identified parameter — it is estimable only in
proportion to how often a component actually breaks, so the formatter (broken
~25 times in 5,000 episodes) is recovered poorly. That is a sample-size limit
rather than a modelling error, and it is harmless for the fix-list: expected
damage is `r_c · q_c`, so the components with uncertain `q_c` are exactly the
ones contributing negligible damage.

## Tests

```bash
uv run pytest -q
```

The three that matter:

- **enumeration test** — pgmpy's answer equals brute force over all `2^n`
  assignments of `R`, to `1e-10`
- **closed-form test** — the `O(n)` formula equals pgmpy's answer, to `1e-10`
- **recovery test** — EM recovers the generating parameters, and the
  log-likelihood never decreases

## What this does not do yet

No real-trace adapters (Who&When, Langfuse), no detector implementations that
read an actual trace, no feature-conditioned prior, no calibration study, no
Langfuse write-back, no CLI.

The important caveat: **every number here comes from data the model itself
generated.** That makes it evidence the mathematics and the code are right, not
evidence the model describes real agent failures. Misspecification — fitting
data the model *cannot* represent — is the next experiment.

MIT licensed.
