"""FaultLens -- probabilistic fault attribution for LLM agents (basic engine)."""

from faultlens.closed_form import explain_closed_form
from faultlens.em import EMResult, Priors, fit
from faultlens.episode import Episode, Step, chain_episode
from faultlens.infer import Attribution, explain, format_attribution
from faultlens.network import build_network, evidence_of
from faultlens.params import DEFAULT_PARAMS, DETECTORS, ModelParams
from faultlens.simulate import sample_corpus, sample_episode

__all__ = [
    "DEFAULT_PARAMS",
    "DETECTORS",
    "Attribution",
    "EMResult",
    "Episode",
    "ModelParams",
    "Priors",
    "Step",
    "build_network",
    "chain_episode",
    "evidence_of",
    "explain",
    "explain_closed_form",
    "fit",
    "format_attribution",
    "sample_corpus",
    "sample_episode",
]
