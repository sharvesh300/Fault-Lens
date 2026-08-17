# How the Bayesian network is built — a worked walkthrough

This document follows one real episode from a raw trace to a blame posterior,
explaining the mathematics as it appears and giving an intuition for each piece.

[`model-spec.md`](model-spec.md) is the formal reference: it states the model
compactly and completely. This document is the slow version — the same content,
motivated. Every number below is computed by the code in this repository, not
illustrative.

**Contents**

1. [The trace we start with](#1-the-trace-we-start-with)
2. [Trace → execution DAG](#2-trace--execution-dag)
3. [DAG → Bayesian network](#3-dag--bayesian-network-the-key-move)
4. [Mathematical background](#4-mathematical-background)
5. [The three CPDs](#5-the-three-cpds)
6. [The joint distribution for this episode](#6-the-joint-distribution-for-this-episode)
7. [Inference, with the real arithmetic](#7-inference-with-the-real-arithmetic)
8. [Explaining away](#8-explaining-away)
9. [Why the 2^n sum collapses](#9-why-the-2n-sum-collapses)
10. [What EM adds](#10-what-em-adds)

---

## 1. The trace we start with

An analytics agent is asked *"How many enterprise customers churned in Q2, and
what was the top reason?"* It runs six steps and answers **"12 customers, top
reason: pricing."** The true answer is 47.

```json
{
  "episode_id": "1471",
  "outcome": "fail",
  "steps": [
    {"step_id": "s1", "component_type": "planner",    "parent_id": null},
    {"step_id": "s2", "component_type": "retriever",  "parent_id": "s1", "top1_score": 0.31},
    {"step_id": "s3", "component_type": "sql_writer", "parent_id": "s2"},
    {"step_id": "s4", "component_type": "db_tool",    "parent_id": "s3"},
    {"step_id": "s5", "component_type": "summarizer", "parent_id": "s4"},
    {"step_id": "s6", "component_type": "formatter",  "parent_id": "s5"}
  ]
}
```

The retriever fetched an outdated churn definition. Every later step is
*correct given the input it was handed* — the SQL is valid, the database
answered honestly, the summary faithfully reports what it received.

**The only supervision anyone provides is `outcome: fail`.** Nobody labels `s2`.
That single bit is the entire training signal, and recovering per-step
responsibility from it is the problem.

Two cheap checks fire while parsing:

| Step | Detector | Why |
| --- | --- | --- |
| `s2` retriever | `low_retrieval` | top-1 similarity 0.31, below threshold |
| `s5` summarizer | `length_outlier` | output longer than p99 for summarizers |

`length_outlier` fires on roughly one innocent step in six. It is nearly
worthless on its own. Keep that in mind — it becomes the point.

## 2. Trace → execution DAG

Nodes are step instances; edges are data dependencies, recovered from span
parentage plus a data-flow check (does step *i*'s output appear in step *j*'s
input?).

```
s1 planner → s2 retriever → s3 sql_writer → s4 db_tool → s5 summarizer → s6 formatter
```

This DAG is **read off the execution record, not learned from the failure
labels.** That distinction is what separates a principled Bayesian network from
an arbitrary one — we are not fitting structure to outcomes and then claiming
the structure explains them.

## 3. DAG → Bayesian network: the key move

Here is the step that is easiest to get wrong.

> **The execution DAG is not the Bayesian network.**

They are different graphs answering different questions:

| Graph | Nodes | Edges mean |
| --- | --- | --- |
| Execution DAG | step instances | "this step's output fed that step's input" |
| Bayesian network | random variables | "this variable's distribution depends on that one" |

The DAG tells us **which random variables exist** — one reliability variable per
step, with its component type and features. It does not itself become the edges.

For each step $i$ we introduce one **latent** variable and its observed
consequences:

| Variable | Domain | Observed? | Reads as |
| --- | --- | --- | --- |
| $R_i$ | ok / fail | **no** | did step $i$ do its job correctly, *given its input* |
| $E_{ij}$ | 0 / 1 | yes | detector $j$ fired on step $i$ |
| $Y$ | ok / fail | yes | did the whole task succeed |

Note the phrase *given its input*. `s3` wrote correct SQL for the definition it
received, so $R_3 = \text{ok}$ even though the final answer is wrong. Blame
belongs to whoever introduced the error, not whoever carried it.

The resulting network, for our six-step episode:

```
      R_1        R_2         R_3        R_4        R_5        R_6      latent
     / | \      / | \       / | \      / | \      / | \      / | \
   E_11...E_15  E_21...E_25    ...        ...        ...        ...   observed
       \          \        \      |      /         /          /
        \          \        \     |     /         /          /
         ------------------→   Y   ←------------------------          observed
```

Two edge families, and that is the whole structure:

- $R_i \to E_{ij}$ — a broken step tends to trip its detectors
- $R_i \to Y$ — a broken step tends to sink the episode

**Why are there no edges between the $R_i$?** Because the current model does not
claim a downstream step fails *because* an upstream one did. It says each step
is independently more or less reliable, and any of them can sink the run. Adding
propagation means adding explicit variables on the DAG edges — a real extension,
deliberately not smuggled in. What the DAG *does* contribute is the step set,
component identity, and depth.

## 4. Mathematical background

### 4.1 What a Bayesian network is

A Bayesian network is a directed acyclic graph plus one conditional distribution
per node given its parents. Its defining property is that the joint distribution
factorises along the graph:

$$
P(X_1, \dots, X_N) = \prod_{k=1}^{N} P\big(X_k \mid \mathrm{parents}(X_k)\big).
$$

This is the entire payoff. A joint over our episode's 37 binary variables
(6 $R$, 30 $E$, 1 $Y$) has $2^{37}$ entries — roughly 140 billion numbers.
Factorised, it is specified by a handful of parameters, because most variables
depend on almost nothing.

**Intuition.** The graph is a claim about what is *irrelevant*. An edge you leave
out is a statement that, once you know the parents, that variable tells you
nothing more.

### 4.2 The three connection patterns

Every Bayesian network is built from three patterns, and knowing which ones
appear tells you how information flows:

| Pattern | Shape | Behaviour |
| --- | --- | --- |
| Chain | $A \to B \to C$ | $A$ and $C$ dependent; independent once $B$ is known |
| Fork | $A \leftarrow B \to C$ | same — a common cause explains the correlation |
| **Collider** | $A \to B \leftarrow C$ | $A$ and $C$ **independent**, but **dependent once $B$ is known** |

Our network contains both a fork and a collider:

- **Fork:** $E_{ij} \leftarrow R_i \to E_{ik}$. Two detectors on the same step are correlated only because both react to that step's health. Given $R_i$, they are independent.
- **Collider:** $R_i \to Y \leftarrow R_j$. **This is the important one.** Two steps' reliabilities are marginally independent — the retriever breaking tells you nothing about the summarizer. But *conditioning on the failure* makes them dependent, because now they compete to explain it.

That collider is the entire reason a graphical model is the right tool here, and
§8 shows it operating.

### 4.3 Latent variables

$R$ is never observed, so what we can actually compute is a marginal — sum the
joint over every possible configuration of the hidden variables:

$$
P(E, Y) = \sum_{R \in \{\text{ok},\text{fail}\}^n} P(R, E, Y).
$$

That sum has $2^n$ terms. §9 shows why we never have to compute it that way.

## 5. The three CPDs

### 5.1 Prior on step reliability

$$
R_i \sim \mathrm{Bernoulli}(r_{c(i)})
$$

Each component type $c$ has a base failure rate: retriever 0.29, sql_writer
0.07, formatter 0.005. Parameters are **tied by component type** — every
retriever instance in every episode shares one number.

**Intuition.** This is the model's opinion before it looks at anything specific
about this run: a reputation. Tying is what makes learning possible at all —
untied, there is one free parameter per step instance, each observed once, and
nothing can be estimated.

> **Analogy.** A hospital knows which of its machines break most often. That is
> a prior. It is useful, and it is not a diagnosis of today's fault.

### 5.2 The detector channel

$$
P(E_{ij} = 1 \mid R_i = \text{fail}) = \alpha_j, \qquad
P(E_{ij} = 1 \mid R_i = \text{ok}) = \beta_j
$$

$\alpha_j$ is how often the detector catches a genuinely broken step;
$\beta_j$ is how often it cries wolf. What matters is the **ratio**, because
that is what Bayes' rule multiplies by:

| Detector | $\alpha$ | $\beta$ | LR if it fires | LR if silent |
| --- | --- | --- | --- | --- |
| `tool_error` | 0.55 | 0.02 | **27.5** | 0.46 |
| `schema_violation` | 0.45 | 0.03 | 15.0 | 0.57 |
| `low_retrieval` | 0.64 | 0.09 | 7.1 | 0.40 |
| `retry_occurred` | 0.40 | 0.06 | 6.7 | 0.64 |
| `length_outlier` | 0.30 | 0.17 | **1.8** | 0.84 |

A firing `tool_error` multiplies the odds of that step being broken by 27.5. A
firing `length_outlier` multiplies them by 1.8 — almost nothing. **A detector
with $\alpha_j = \beta_j$ has a likelihood ratio of exactly 1 and is
informationally dead**, no matter how often it fires.

Note the right-hand column: *silence is evidence too*. Every detector that does
not fire multiplies the odds by less than 1, nudging that step toward innocent.

**Intuition.** Each detector is a witness of known unreliability. You do not
discard a bad witness; you learn how much to discount them, and their testimony
still moves the needle by exactly the right amount.

> **Analogy.** This is [Dawid & Skene (1979)](https://www.jstor.org/stable/2346806),
> the crowdsourcing model: several annotators of unknown quality label the same
> items, no gold answers exist, and you recover both the true labels *and* each
> annotator's reliability from their patterns of agreement. Here the annotators
> are cheap trace heuristics.

### 5.3 The outcome — a leaky noisy-OR

$$
P(Y = \text{ok} \mid R) = (1-\lambda) \prod_{i \,:\, R_i = \text{fail}} (1 - q_i)
$$

Read it as a story. Each broken step independently gets a chance to sink the
episode — step $i$ does so with probability $q_i$, its **criticality**.
Separately, an unmodelled cause sinks it with probability $\lambda$, the
**leak**. The episode survives only if nothing sank it.

Healthy steps contribute nothing to the product, which is why only failing steps
appear in it. If every step is fine, $P(Y = \text{fail}) = \lambda$.

**Why this shape and not a free table?** A general CPD over $n$ binary parents
needs $2^n$ numbers — 64 for this episode, a billion for a 30-step trace, none
of them interpretable. The noisy-OR needs $n + 1$, and each one means something
you can say out loud. This is **causal independence**: several causes act
separately, each with its own chance of producing the effect.

> **Analogy.** A relay team loses if any runner drops the baton — but not
> always: sometimes a runner fumbles and the team still wins. $q_i$ is how
> fatal that particular runner's mistake tends to be. And sometimes the team
> loses for reasons no runner caused, like a judging error. That is $\lambda$.

$\lambda$ has a second job: it is the **honesty valve**. When failures do not
match anything the model represents, mass flows to the leak instead of being
forced onto an innocent component. (One caveat, measured in `model-spec.md` §6.5:
the leak does not catch *every* kind of misspecification.)

## 6. The joint distribution for this episode

Assembling §5 by the factorisation rule of §4.1:

$$
P(R, E, Y) \;=\; \underbrace{\prod_{i=1}^{6} P(R_i)}_{\text{reputations}}
\;\cdot\; \underbrace{\prod_{i=1}^{6}\prod_{j=1}^{5} P(E_{ij} \mid R_i)}_{\text{witness testimony}}
\;\cdot\; \underbrace{P(Y \mid R_1,\dots,R_6)}_{\text{noisy-OR}}
$$

37 variables, fully specified by 6 priors + 10 detector parameters + 6
criticalities + 1 leak. That compression is what the graph bought us.

## 7. Inference, with the real arithmetic

We observe all 30 detector values and $Y = \text{fail}$. We want
$P(R_i = \text{fail} \mid E, Y)$ — Bayes' rule, with the hidden variables summed
out.

**Step 1 — fold each step's prior and its own detectors into two numbers.**

$$
a_i = P(R_i = \text{ok}) \prod_j P(e_{ij} \mid \text{ok}),
\qquad
b_i = P(R_i = \text{fail}) \prod_j P(e_{ij} \mid \text{fail})
$$

$a_i$ is the weight of "this step is fine *and* produced these detector
readings"; $b_i$ the same for "broken". Computed for episode 1471:

| Step | $r_c$ | $a_i$ | $b_i$ | $b_i/(a_i{+}b_i)$ | $q_i$ |
| --- | --- | --- | --- | --- | --- |
| s1 planner | 0.020 | 6.614e-01 | 7.484e-04 | 0.0011 | 0.94 |
| s2 retriever | 0.290 | 4.739e-02 | 1.929e-02 | **0.2893** | 0.88 |
| s3 sql_writer | 0.070 | 6.277e-01 | 2.620e-03 | 0.0042 | 0.91 |
| s4 db_tool | 0.030 | 6.547e-01 | 1.123e-03 | 0.0017 | 0.90 |
| s5 summarizer | 0.040 | 1.327e-01 | 6.415e-04 | 0.0048 | 0.62 |
| s6 formatter | 0.005 | 6.715e-01 | 1.871e-04 | 0.0003 | 0.99 |

The fourth column is each step's guilt **on its own evidence alone, ignoring the
outcome**. Look at how little has happened so far: the retriever sits at 0.289
against a prior of 0.290. Its `low_retrieval` flag (LR 7.1) was almost exactly
cancelled by four silent detectors pulling the other way.

Even more striking, the summarizer has gone *down*: prior 0.040 → **0.0048**,
despite `length_outlier` firing. A flag worth LR 1.8 loses to four silences.
**On detector evidence alone, the model thinks every step is probably fine.**

**Step 2 — bring in the outcome.** Two normalisers (derivation in §9):

$$
Z_{\text{any}} = \prod_i (a_i + b_i) = 1.6348 \times 10^{-3}
$$
$$
Z_{\text{ok}} = (1-\lambda)\prod_i \big(a_i + b_i(1-q_i)\big) = 1.1465 \times 10^{-3}
$$

$Z_{\text{any}}$ is the total weight of the detector readings over every possible
$R$; $Z_{\text{ok}}$ is the part of that weight compatible with the episode
having *survived*. The difference is the part compatible with failure:

$$
P(e, Y = \text{fail}) = Z_{\text{any}} - Z_{\text{ok}} = 4.883 \times 10^{-4}
\quad\Longrightarrow\quad
P(Y = \text{fail} \mid e) = 0.2987
$$

So before being told the outcome, the model gave this run a 30% chance of
failing. Then it failed — and that is genuine information, because failure was
the less likely branch.

**Step 3 — the posterior.**

$$
P(R_i = \text{fail} \mid e, Y=\text{fail}) =
\frac{\dfrac{b_i}{a_i + b_i} Z_{\text{any}} - \dfrac{b_i(1-q_i)}{a_i + b_i(1-q_i)} Z_{\text{ok}}}{Z_{\text{any}} - Z_{\text{ok}}}
$$

| Step | on its own evidence | **after conditioning on the failure** |
| --- | --- | --- |
| s2 retriever | 0.2893 | **0.8593** |
| s3 sql_writer | 0.0042 | 0.0130 |
| s5 summarizer | 0.0048 | 0.0118 |
| s4 db_tool | 0.0017 | 0.0053 |
| s1 planner | 0.0011 | 0.0036 |
| s6 formatter | 0.0003 | 0.0009 |
| *unmodelled cause* | — | 0.1175 |

**Intuition.** Learning the episode failed means *something* must account for it.
That obligation gets distributed in proportion to how plausible each explanation
was to begin with — and the retriever, at 0.289, was sixty times more plausible
than any other step. It absorbs almost all of it. The leftover 0.1175
is the model saying it might be none of them.

## 8. Explaining away

Now the collider from §4.2 does its work. Watch the summarizer across four
information states:

| What we know | $P(R_5 = \text{fail})$ |
| --- | --- |
| nothing (prior) | 0.0400 |
| its own detectors (`length_outlier` fired, four silent) | 0.0048 |
| **+ the episode failed**, retriever's flag *absent* | 0.0403 |
| **+ the episode failed**, retriever's flag present | **0.0118** |

Read the last two rows against each other. Same summarizer, same
`length_outlier` firing, same failed outcome. The only difference is evidence
about a *different step* — and the summarizer's guilt drops by a factor of 3.4.

That is **explaining away**. $R_2$ and $R_5$ are marginally independent; nothing
connects them. Conditioning on $Y$ opens the collider and makes them compete. A
strong explanation for the failure appearing upstream *reduces* the need for a
downstream one.

> **Analogy.** Your car won't start. Battery or fuel — two independent causes.
> Discovering the battery is dead makes an empty tank *less* likely, not because
> the battery affects the fuel, but because the symptom is already accounted
> for.

This is why the problem needs a graphical model rather than a per-step scorer.
A scorer evaluating each span independently sees five ordinary-looking steps and
one odd one, and blames the odd one. It has no representation in which upstream
evidence can discount a downstream anomaly. It structurally cannot make this
move.

And it matches how these failures actually behave: subtle, originating early,
hidden behind locally-correct outputs. The failure *looks* like it happened at
step 5; it happened at step 2; everything in between did the right thing with
the wrong input.

## 9. Why the 2^n sum collapses

$P(e, Y)$ is a sum over $2^n$ assignments of $R$ — 64 here, a billion at 30
steps. It never has to be computed that way.

Look again at what a summand contains: each step contributes its own factor
($a_i$ or $b_i$), and the noisy-OR contributes $(1-q_i)$ for each *failing*
step. Nothing couples step $i$ to step $k$. When a sum of products has no
cross-terms, it factors into a product of sums — the distributive law, at scale:

$$
\sum_{R} \prod_i f_i(R_i) = \prod_i \sum_{R_i} f_i(R_i).
$$

Applying it to the two cases:

$$
Z_{\text{any}} = \prod_i (a_i + b_i),
\qquad
Z_{\text{ok}} = (1-\lambda)\prod_i \big(a_i + b_i(1-q_i)\big)
$$

Each factor is just "this step is ok, **or** broken" — with the broken branch
discounted by $(1-q_i)$ when we additionally require the episode to have
survived.

> **Analogy.** To find the chance that at least one of 30 independent alarms
> fires, nobody enumerates $2^{30}$ patterns. You compute
> $1 - \prod(1-p_i)$. Same trick, same reason: independence turns enumeration
> into multiplication.

Every blame posterior is therefore a **difference of two closed-form products**:
exact, $O(nm)$ per episode, no sampling and no variational approximation. A
30-step episode resolves in microseconds — which is what makes learning
affordable, since EM runs this once per episode per iteration.

The repository keeps two implementations and asserts they agree to $10^{-10}$:
`infer.py` (pgmpy variable elimination, readable) and `closed_form.py` (the
formula above, fast). Both are checked against brute-force enumeration.

## 10. What EM adds

Everything so far assumed $r_c, q_c, \alpha_j, \beta_j, \lambda$ were known. On
a real system nobody knows them. But we have thousands of episodes, each with
one outcome bit — and that turns out to be enough.

The obstacle is circular:

- if we knew which steps were broken, measuring each detector's $\alpha_j$ and $\beta_j$ would be simple counting;
- if we knew $\alpha_j$ and $\beta_j$, inferring which steps were broken would be §7.

We know neither. **EM breaks the circle by alternating, using soft guesses.**

- **E-step** — with current parameters, compute $\gamma_i = P(R_i = \text{fail} \mid e, y)$ for every step of every episode. Not labels: probabilities.
- **M-step** — re-estimate every parameter treating those probabilities as fractional counts. A step that is 70% likely broken contributes 0.7 to the "broken" tally and 0.3 to "ok".

Repeat. Each round is guaranteed not to decrease the likelihood.

> **Analogy.** Several teaching assistants grade the same essays. You have no
> answer key and no idea which TA is harsh or lenient. You can still recover
> both — bootstrap from where they agree, use provisional grades to estimate
> each TA's bias, use the biases to sharpen the grades, iterate. EM is that
> procedure, made precise.

The critical detail is that the E-step never commits to a hard label. Committing
would be pretending to know the step-level truth we said we could not observe;
carrying the uncertainty forward as fractional counts is what makes learning
from a single terminal bit legitimate rather than a fudge.

One structural asset makes this better behaved than textbook Dawid–Skene: the
noisy-OR link to $Y$. Unsupervised annotator models suffer a label-permutation
symmetry — nothing says which hidden state means "broken", so EM can converge to
a perfectly mirrored solution. The outcome variable anchors it: one hidden state
demonstrably causes episodes to fail, and that breaks the tie.

**Measured result.** On synthetic corpora where the truth is known and only the
pass/fail bit is shown, EM recovers the generating parameters to a mean absolute
error of 0.0022 on component priors at 5,000 episodes — against a gate of 0.05 —
and recovers the fix-list ranking exactly. Derivation and full results in
[`model-spec.md`](model-spec.md) §6.

---

## Glossary

| Symbol | Name | Reads as |
| --- | --- | --- |
| $R_i$ | reliability | did step $i$ do its job, given its input |
| $E_{ij}$ | detector | cheap heuristic $j$ fired on step $i$ |
| $Y$ | outcome | did the task succeed — the only supervision |
| $r_c$ | prior | how often component type $c$ breaks |
| $q_c$ | criticality | if $c$ breaks, how often the episode dies |
| $\alpha_j$ | sensitivity | how often detector $j$ catches a real fault |
| $\beta_j$ | false-positive rate | how often it cries wolf |
| $\lambda$ | leak | failures from causes outside the model |
| $a_i, b_i$ | step weights | prior × detector evidence, per hypothesis |
| $\gamma_i$ | soft label | posterior that step $i$ was broken (EM's E-step) |

**Where to go next.** [`model-spec.md`](model-spec.md) for the formal statement,
the EM derivation, and the assumptions this model makes. `main.py` runs the
example above; `experiments/e1_parameter_recovery.py` runs the learning study.
