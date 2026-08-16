# FaultLens

Probabilistic fault attribution for LLM agents. Given a trace and **one bit of
supervision** — did the task succeed? — infer which component was responsible,
as a calibrated posterior. No step-level labels, no LLM calls at inference time.

This repository currently holds the **basic engine**: the Bayesian network,
exact inference, and a synthetic simulation. Parameters are hand-set; learning
them by MAP-EM is the next phase.

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

Full derivation, including the exact O(n) closed form for the posterior:
[`docs/model-spec.md`](docs/model-spec.md).

## Layout

```
faultlens/params.py       fleet parameters (r_c, q_c, α, β, λ)
faultlens/episode.py      canonical episode record + the worked example
faultlens/network.py      episode -> pgmpy DiscreteBayesianNetwork
faultlens/infer.py        blame posteriors by variable elimination
faultlens/closed_form.py  the same posteriors in O(n), no 2^n table
faultlens/simulate.py     generative sampler = the ground-truth oracle
tests/                    enumeration test + closed-form agreement
docs/model-spec.md        mathematical specification
```

## Tests

```bash
uv run pytest -q
```

The two that matter:

- **enumeration test** — pgmpy's answer equals brute force over all `2^n`
  assignments of `R`, to `1e-10`
- **closed-form test** — the `O(n)` formula equals pgmpy's answer, to `1e-10`

## What this does not do yet

No parameter learning (MAP-EM), no real-trace adapters, no calibration study,
no Langfuse write-back, no CLI. Parameters are hand-set from the design doc, so
the numbers demonstrate the machinery, not measured reliability of any real
agent.

MIT licensed.
