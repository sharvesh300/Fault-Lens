"""Demo: the worked example, then a small simulation study.

    uv run main.py [n_episodes]
"""

from __future__ import annotations

import random
import sys

from faultlens.episode import EPISODE_1471
from faultlens.infer import explain, format_attribution
from faultlens.params import DEFAULT_PARAMS
from faultlens.simulate import sample_corpus


def worked_example() -> None:
    print("=" * 62)
    print("1 · Worked example -- one failed episode, one bit of supervision")
    print("=" * 62)
    attribution = explain(EPISODE_1471, DEFAULT_PARAMS)
    print(format_attribution(EPISODE_1471, attribution))
    print()
    print("  The summarizer's length_outlier flag is explained away once the")
    print("  retriever accounts for the failure -- the Bayes-net move a")
    print("  per-step scorer cannot make.")
    print()


def explaining_away_check() -> None:
    """Same episode, but without the retriever's detector firing."""
    from copy import deepcopy

    print("=" * 62)
    print("2 · Explaining away -- drop the retriever's evidence")
    print("=" * 62)
    episode = deepcopy(EPISODE_1471)
    episode.episode_id = "1471-no-d2"
    episode.step("s2").detectors["low_retrieval"] = 0
    print(format_attribution(episode, explain(episode, DEFAULT_PARAMS)))
    print()
    print("  With no informative evidence, no step explains the failure well:")
    print("  most mass goes to the unmodelled cause, and what is left follows")
    print("  the fleet priors. The model says 'I don't know' instead of")
    print("  confidently blaming someone.")
    print()


def simulation_study(n: int) -> None:
    print("=" * 62)
    print(f"3 · Simulation -- {n} episodes from the generative model")
    print("=" * 62)
    corpus = sample_corpus(n, seed=7)
    failures = [
        ep for ep in corpus if ep.outcome == "fail" and ep.true_culprit != "leak"
    ]
    print(f"  episodes            : {len(corpus)}")
    print(f"  failures            : {sum(ep.outcome == 'fail' for ep in corpus)}")
    print(f"  attributable ones   : {len(failures)}")
    print("  step-level labels   : 0        LLM calls: 0")
    print()

    rng = random.Random(11)
    hits = last_step = random_step = 0
    for ep in failures:
        if explain(ep, DEFAULT_PARAMS).argmax() == ep.true_culprit:
            hits += 1
        if ep.step_ids[-1] == ep.true_culprit:
            last_step += 1
        if rng.choice(ep.step_ids) == ep.true_culprit:
            random_step += 1

    n_fail = max(len(failures), 1)
    print("  top-1 accuracy at naming the responsible step")
    print(f"    faultlens posterior : {hits / n_fail:.3f}")
    print(f"    last-step baseline  : {last_step / n_fail:.3f}")
    print(f"    random-step baseline: {random_step / n_fail:.3f}")
    print()


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    worked_example()
    explaining_away_check()
    simulation_study(n)


if __name__ == "__main__":
    main()
