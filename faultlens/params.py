"""Fleet-level parameters of the FaultLens model.

Everything here is hand-set for the basic version. Phase 3 replaces these
numbers with values learned by MAP-EM from outcome-only supervision; the model
structure does not change when that happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The five cheap, label-free detectors. Each is individually unreliable.
DETECTORS: tuple[str, ...] = (
    "tool_error",
    "low_retrieval",
    "schema_violation",
    "retry_occurred",
    "length_outlier",
)


@dataclass(frozen=True)
class ModelParams:
    """Parameters shared across every episode, tied by component type.

    prior_fail  r_c  = P(R_i = fail) for a step of component type c
    criticality q_c  = P(episode dies | this step is broken)
    alpha       a_j  = P(E_ij = 1 | R_i = fail)   detector sensitivity
    beta        b_j  = P(E_ij = 1 | R_i = ok)     detector false-positive rate
    leak        l    = P(episode dies | every step is fine)
    """

    prior_fail: dict[str, float]
    criticality: dict[str, float]
    alpha: dict[str, float]
    beta: dict[str, float]
    leak: float = 0.05
    default_prior_fail: float = 0.05
    default_criticality: float = 0.80

    detectors: tuple[str, ...] = field(default=DETECTORS)

    def prior_of(self, component_type: str) -> float:
        return self.prior_fail.get(component_type, self.default_prior_fail)

    def criticality_of(self, component_type: str) -> float:
        return self.criticality.get(component_type, self.default_criticality)


# Reliabilities mirror the fleet report in the design doc:
# retriever 0.71, sql_writer 0.93, summarizer 0.96, planner 0.98, formatter 0.995.
DEFAULT_PARAMS = ModelParams(
    prior_fail={
        "planner": 0.02,
        "retriever": 0.29,
        "sql_writer": 0.07,
        "db_tool": 0.03,
        "summarizer": 0.04,
        "formatter": 0.005,
    },
    criticality={
        "planner": 0.94,
        "retriever": 0.88,
        "sql_writer": 0.91,
        "db_tool": 0.90,
        "summarizer": 0.62,
        "formatter": 0.99,
    },
    # alpha > beta means the detector carries signal; length_outlier barely does.
    alpha={
        "tool_error": 0.55,
        "low_retrieval": 0.64,
        "schema_violation": 0.45,
        "retry_occurred": 0.40,
        "length_outlier": 0.30,
    },
    beta={
        "tool_error": 0.02,
        "low_retrieval": 0.09,
        "schema_violation": 0.03,
        "retry_occurred": 0.06,
        "length_outlier": 0.17,
    },
    leak=0.05,
)
