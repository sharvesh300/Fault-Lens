"""E1 -- can EM recover the parameters from data we generated ourselves?

This is the P1 exit gate and the project's kill criterion. If EM cannot recover
known parameters from synthetic traces, it will never work on real ones, and no
amount of extra data will fix that -- it will only produce more confident
nonsense.

    uv run experiments/e1_parameter_recovery.py [n_episodes]

The only supervision handed to `fit` is each episode's terminal pass/fail bit.
The true reliabilities are generated, recorded, and never shown to the model.
"""

from __future__ import annotations

import sys
import time

from faultlens.em import fit, mean_absolute_error, parameter_errors
from faultlens.params import DEFAULT_PARAMS
from faultlens.simulate import sample_corpus

SIZES = (500, 2000, 5000)
GATE = 0.05  # MAE on the component priors r_c


def table(learned, true, corpus) -> str:
    lines = ["  parameter                 true   learned    error"]
    lines.append("  " + "-" * 46)
    for c in sorted(true.prior_fail):
        if c in learned.prior_fail:
            t, got = true.prior_fail[c], learned.prior_fail[c]
            lines.append(f"  r  {c:<20} {t:.3f}     {got:.3f}    {abs(got - t):.3f}")
    lines.append("")
    for c in sorted(true.criticality):
        if c in learned.criticality:
            t, got = true.criticality[c], learned.criticality[c]
            lines.append(f"  q  {c:<20} {t:.3f}     {got:.3f}    {abs(got - t):.3f}")
    lines.append("")
    for j in learned.alpha:
        t, got = true.alpha[j], learned.alpha[j]
        lines.append(f"  a  {j:<20} {t:.3f}     {got:.3f}    {abs(got - t):.3f}")
    lines.append("")
    for j in learned.beta:
        t, got = true.beta[j], learned.beta[j]
        lines.append(f"  b  {j:<20} {t:.3f}     {got:.3f}    {abs(got - t):.3f}")
    lines.append("")
    lines.append(
        f"  l  {'leak':<20} {true.leak:.3f}     {learned.leak:.3f}"
        f"    {abs(learned.leak - true.leak):.3f}"
    )
    return "\n".join(lines)


def damage_ranking(params) -> list[tuple[str, float]]:
    """Expected damage = reliability x criticality. This is the fix-list."""
    return sorted(
        (
            (c, params.prior_fail[c] * params.criticality_of(c))
            for c in params.prior_fail
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )


def damage_table(learned, true) -> str:
    got, want = damage_ranking(learned), damage_ranking(true)
    lines = ["  rank  true fix-list          learned fix-list"]
    lines.append("  " + "-" * 46)
    for i, ((tc, tv), (lc, lv)) in enumerate(zip(want, got), start=1):
        mark = " " if tc == lc else " <- differs"
        lines.append(f"  {i}     {tc:<12} {tv:.4f}   {lc:<12} {lv:.4f}{mark}")
    order_match = [c for c, _ in want] == [c for c, _ in got]
    lines.append("")
    lines.append(f"  ranking recovered exactly: {order_match}")
    return "\n".join(lines)


def run(n: int, verbose: bool = True) -> dict[str, float]:
    corpus = sample_corpus(n, seed=42)
    failures = sum(ep.outcome == "fail" for ep in corpus)

    started = time.time()
    result = fit(corpus, n_restarts=5, seed=1)
    elapsed = time.time() - started

    errors = parameter_errors(result.params, DEFAULT_PARAMS)
    maes = {group: mean_absolute_error(values) for group, values in errors.items()}

    if verbose:
        print(f"  episodes            : {n}  ({failures} failed)")
        print(f"  supervision         : {n} pass/fail bits, 0 step labels")
        print(
            f"  EM                  : {result.n_iter} iters, "
            f"converged={result.converged}, {elapsed:.1f}s"
        )
        print(f"  log-likelihood      : {result.log_likelihood:.1f}")
        print(f"  monotone increase   : {result.monotone}")
        print(
            "  restart spread      : "
            + ", ".join(f"{s:.1f}" for s in sorted(result.restarts, reverse=True))
        )
        print()
        print(table(result.params, DEFAULT_PARAMS, corpus))
        print()
        print("  MAE by group:")
        for group, value in maes.items():
            print(f"    {group:<12} {value:.4f}")
        print()
        print(damage_table(result.params, DEFAULT_PARAMS))
        print()

    return maes


def main() -> None:
    if len(sys.argv) > 1:
        sizes = (int(sys.argv[1]),)
    else:
        sizes = SIZES

    summary = {}
    for n in sizes:
        print("=" * 62)
        print(f"E1 · parameter recovery at n = {n}")
        print("=" * 62)
        summary[n] = run(n)

    print("=" * 62)
    print("Corpus-size curve -- MAE on the component priors r_c")
    print("=" * 62)
    for n, maes in summary.items():
        verdict = "PASS" if maes["prior_fail"] < GATE else "FAIL"
        print(f"  n = {n:<6} MAE(r) = {maes['prior_fail']:.4f}   [{verdict}]")
    print()
    print(f"  P1 exit gate: MAE(r) < {GATE} at 5,000 mixed-topology episodes")


if __name__ == "__main__":
    main()
