# FaultLens — model specification

Mathematical formulation of the basic engine: the Bayesian network compiled per
episode, its conditional distributions, exact inference, and the derivation of
the closed form that inference is checked against.

Everything in this document corresponds to code:

| Section | Code |
| --- | --- |
| §1–§3 variables and CPDs | `faultlens/network.py`, `faultlens/params.py` |
| §4 inference | `faultlens/infer.py` (pgmpy), `faultlens/closed_form.py` (O(n)) |
| §5 generative model | `faultlens/simulate.py` |
| §6 learning | not yet implemented — next phase |

---

## 1. Setting

An episode is a run of an LLM agent: an ordered set of $n$ steps
$s_1,\dots,s_n$, each executed by a component of type $c(i)$ (planner,
retriever, sql_writer, db_tool, summarizer, formatter). After the run we observe
exactly one bit of supervision: did the task succeed?

The quantity we want — *which step was broken* — is never observed. It is a
latent variable, and this is the textbook setting for a latent-variable
graphical model.

## 2. Variables

For each step $i \in \{1,\dots,n\}$ and detector $j \in \{1,\dots,m\}$
($m = 5$):

| Variable | Domain | Status | Meaning |
| --- | --- | --- | --- |
| $R_i$ | $\{\text{ok}, \text{fail}\}$ | **latent** | step $i$ did its job correctly given its input |
| $E_{ij}$ | $\{0, 1\}$ | observed | detector $j$ fired on step $i$ |
| $Y$ | $\{\text{ok}, \text{fail}\}$ | observed | terminal task outcome |

Encode $R_i = 1$ for *fail* and $E_{ij} = 1$ for *fired*.

**Structure.** $R_i \to E_{ij}$ for every $j$, and $R_i \to Y$ for every $i$:

```
   R_1        R_2        ...        R_n          (latent)
  / | \      / | \                 / | \
E_11 ... E_1m   E_21 ... E_2m    E_n1 ... E_nm   (observed)
    \        |         |        /
     \       |         |       /
      ----------->  Y  <-------                  (observed, 1 bit)
```

The execution DAG recovered from the trace determines *which steps exist*, their
component types, and their depth/criticality features. It is deliberately **not**
used as edges between the $R_i$: the current model does not claim that a
downstream step fails *because* an upstream one did. Explicit propagation
variables on DAG edges are a documented future extension.

**Joint factorisation.** Given the structure,

$$
P(R, E, Y) \;=\; \prod_{i=1}^{n} P(R_i)\,\prod_{j=1}^{m} P(E_{ij} \mid R_i)\;\cdot\; P(Y \mid R_1,\dots,R_n).
$$

## 3. Conditional distributions

### 3.1 Prior on step reliability

$$
R_i \sim \mathrm{Bernoulli}(r_{c(i)}), \qquad r_c = P(R_i = \text{fail}).
$$

Parameters are **tied by component type**: every `retriever` instance in every
episode shares one $r_c$. Without tying there is one free parameter per step
instance and nothing is identifiable.

> The full model conditions this prior on step features,
> $P(R_i = \text{fail} \mid \varphi_i) = \sigma(w_{c(i)}^\top \varphi_i)$, with
> $\varphi_i$ = token counts, retry count, retrieval score, DAG depth. The basic
> engine uses the intercept-only special case $\sigma(w_c) = r_c$; nothing else
> in the derivation changes.

### 3.2 Detector channel (Dawid–Skene)

$$
P(E_{ij} = 1 \mid R_i = \text{fail}) = \alpha_j, \qquad
P(E_{ij} = 1 \mid R_i = \text{ok}) = \beta_j .
$$

$\alpha_j$ is the detector's sensitivity, $\beta_j$ its false-positive rate.
Detectors are conditionally independent given $R_i$ — the classical model for
recovering a hidden truth from several individually untrustworthy annotators,
here with cheap trace heuristics playing the part of the annotators.

A detector with $\alpha_j \approx \beta_j$ carries no information; the
likelihood ratio it contributes is $\approx 1$. `length_outlier`
($\alpha = 0.30, \beta = 0.17$) is deliberately near-useless, and the model is
expected to discover that rather than be told.

### 3.3 Outcome: leaky noisy-OR

$$
P(Y = \text{ok} \mid R) \;=\; (1-\lambda)\prod_{i \,:\, R_i = \text{fail}} (1 - q_i),
\qquad q_i \equiv q_{c(i)} .
$$

- $q_c \in [0,1]$ — **criticality**: if a component of type $c$ is broken, how
  likely is it to sink the episode on its own. A broken formatter almost always
  does; a broken optional enrichment step rarely does.
- $\lambda \in [0,1]$ — **leak**: the episode fails for a cause the model does
  not represent. It is the honesty valve — if $\lambda$ is fitted large, the
  model reports that it does not understand these failures instead of
  confidently blaming a component.

Semantics: each broken step independently gets a chance to kill the run, and so
does an unmodelled cause. The special case $R = (\text{ok},\dots,\text{ok})$
gives $P(Y = \text{fail} \mid \text{all ok}) = \lambda$.

In code this is written out as the full $2 \times 2^n$ table
(`_noisy_or_cpd`) so pgmpy can consume it — correct, and fine for episode sizes
where $2^n$ is tractable. §4.2 shows how to avoid the table entirely.

## 4. Inference

### 4.1 The query

Observing detector firings $E = e$ and the outcome $Y = y$, the blame posterior
for step $i$ is

$$
P(R_i = \text{fail} \mid E = e, Y = y)
= \frac{\sum_{R : R_i = \text{fail}} P(R, e, y)}{\sum_{R} P(R, e, y)} ,
$$

and the probability that the failure had no modelled cause is

$$
P(\text{unmodelled}) = P\big(R = (\text{ok},\dots,\text{ok}) \mid E = e, Y = \text{fail}\big).
$$

`faultlens/infer.py` computes both with pgmpy's `VariableElimination` on the
compiled network. This is the reference implementation: readable, and directly
the object the course is about.

### 4.2 Closed form, exact and O(n)

Because the noisy-OR factorises over steps, the sums over $2^n$ states collapse
into products. Define, per step, the two unnormalised weights that fold in the
prior and all of that step's detector evidence:

$$
a_i = P(R_i = \text{ok})\prod_{j} P(e_{ij} \mid R_i = \text{ok}), \qquad
b_i = P(R_i = \text{fail})\prod_{j} P(e_{ij} \mid R_i = \text{fail}).
$$

Then

$$
Z_{\text{any}} \;=\; \sum_{R} \prod_i \big(\dots\big) \;=\; \prod_{i=1}^{n}\,(a_i + b_i),
$$

$$
Z_{\text{ok}} \;=\; \sum_{R} P(R, e)\,P(Y = \text{ok} \mid R)
\;=\; (1-\lambda)\prod_{i=1}^{n}\big(a_i + b_i(1 - q_i)\big),
$$

where $Z_{\text{ok}}$ follows because $P(Y=\text{ok}\mid R)$ contributes an
independent factor $(1-q_i)$ for exactly those steps with $R_i = \text{fail}$,
so the sum over each $R_i$ factorises term by term. Hence

$$
P(e, Y = \text{fail}) = Z_{\text{any}} - Z_{\text{ok}} ,
$$

$$
P(R_i = \text{fail},\, e,\, Y = \text{fail})
= \frac{b_i}{a_i + b_i}\,Z_{\text{any}}
- \frac{b_i(1 - q_i)}{a_i + b_i(1-q_i)}\,Z_{\text{ok}} ,
$$

$$
P(R = \mathbf{ok},\, e,\, Y=\text{fail}) = \lambda \prod_i a_i .
$$

Dividing by $P(e, Y=\text{fail})$ gives the posteriors. Every blame score is a
**difference of two closed-form products: exact, $O(nm)$ per episode, no
sampling and no variational bound.** A 30-step episode resolves in microseconds,
which is what makes the E-step of EM essentially free and keeps the whole engine
CPU-only.

### 4.3 What is verified

`tests/test_model.py` asserts pgmpy's answer equals **brute-force enumeration
over all $2^n$ assignments of $R$**, computed straight from §3, to $10^{-10}$ —
on the worked example, on sampled episodes, and under randomised parameters.
`tests/test_closed_form.py` asserts the O(n) formula equals pgmpy's answer to
the same tolerance. Together these rule out an entire class of silent numerical
bugs.

### 4.4 Explaining away

The worked example (`faultlens/episode.py:EPISODE_1471`) is a six-step chain in
which `low_retrieval` fired on the retriever, `length_outlier` fired on the
summarizer, and $Y = \text{fail}$:

```
P(retriever  broken) = 0.859   [low_retrieval]
P(sql_writer broken) = 0.013
P(summarizer broken) = 0.012   [length_outlier]
P(unmodelled cause)  = 0.118
```

The two latent variables $R_2$ and $R_5$ are marginally independent but become
dependent once $Y$ is observed — the common-effect (v-structure) pattern. Once
the retriever accounts for the failure, the summarizer's weak flag is *explained
away*, and its posterior drops **below** what the same flag alone would imply.
This is the move a per-step independent scorer structurally cannot make, and it
is why a graphical model is load-bearing here rather than decorative.

Removing the retriever's evidence redistributes mass to the unmodelled cause
(0.62) rather than to another component: with no informative signal, the model
declines to accuse.

## 5. The generative model (simulation)

`faultlens/simulate.py` samples ancestrally from exactly §3, so every truth is
known:

1. draw a topology (component-type sequence),
2. $R_i \sim \mathrm{Bernoulli}(r_{c(i)})$,
3. $E_{ij} \sim \mathrm{Bernoulli}(\alpha_j)$ if $R_i = \text{fail}$, else $\mathrm{Bernoulli}(\beta_j)$,
4. each broken step kills the episode w.p. $q_{c(i)}$; an unmodelled cause kills it w.p. $\lambda$; $Y = \text{fail}$ if any killer fired.

The recorded `true_culprit` is the first cause that fired (or `"leak"`), which
gives ground truth for evaluation. Inference never sees `true_reliability` or
`true_culprit` — only $E$ and $Y$.

This is why the simulator is written before anything else: on synthetic data a
wrong answer is unambiguously a bug, not task difficulty or a parser error.

## 6. Learning (next phase, not implemented here)

The basic engine uses hand-set parameters. Fitting
$\theta = \{r_c, q_c, \alpha_j, \beta_j, \lambda\}$ from a corpus of episodes
labelled only with $Y$ is MAP-EM on the latent $R$:

- **E-step** — for every episode compute $\gamma_i = P(R_i = \text{fail} \mid e, y)$ exactly, by §4.2.
- **M-step** — treat $\gamma_i$ as fractional counts:
  $\hat{\alpha_j} \propto \sum \gamma_i e_{ij}$, $\hat{\beta_j} \propto \sum (1-\gamma_i) e_{ij}$ (closed form, with Beta priors);
  $\hat{r_c}$ from the soft counts per component type (a logistic regression on soft labels once features are switched on);
  $q_c$ and $\lambda$ by numeric maximisation of the noisy-OR outcome likelihood.

The E-step produces probabilities, never hard labels — that is precisely why the
system can learn from a terminal pass/fail bit without pretending to know the
true step label.

Three failure modes to expect and diagnose: the leak absorbing everything
($\lambda \to 1$), EM collapsing to the prior, and detectors learned
uninformative ($\alpha_j \approx \beta_j$).

## 7. Assumptions and limitations

| Assumption | Risk |
| --- | --- |
| Binary reliability | Crude for an LLM component; an ordinal state space is the natural relaxation. |
| Detectors conditionally independent given $R_i$ | Two detectors keying off the same symptom double-count evidence. |
| Noisy-OR outcome | Assumes single causes suffice; failures needing two components to fail jointly are not represented. Testable by fitting both and comparing with an information criterion. |
| No fault propagation along DAG edges | The model attributes to a step, but does not model corruption flowing downstream. |
| Parameters tied by component type | Breaks silently if the adapter does not canonicalise component names. |
