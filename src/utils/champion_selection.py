from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional


@dataclass
class ModelMetrics:
    family: str                      # "AutoTFT", "AutoNHITS", "AutoDLinear", "BaselineLinear", "graph_stgcnn", ...
    mape: float                      # e.g. 0.015
    smape: float                     # e.g. 0.016
    directional_accuracy: float      # 0..1
    swase: Optional[float] = None    # optional, not used in score here but you can add it
    guardrail_flags: Set[str] = field(default_factory=set)


@dataclass
class ChampionSelection:
    champion: ModelMetrics
    reason: str
    scores: Dict[str, float]         # family -> composite score (lower is better)
    disqualified: Dict[str, Set[str]]  # family -> guardrail flags that disqualified it


CRITICAL_GUARDRAIL_FLAGS: Set[str] = {
    "data_drift_suspected",
    "deep_vs_baseline_critical",
    "pipeline_failure_recently",
}


def _composite_score(m: ModelMetrics) -> float:
    """
    Lower is better.
    You can tweak weights; this version uses MAPE, SMAPE, and (1 - DA).
    """
    return (
        0.4 * m.mape +
        0.3 * m.smape +
        0.3 * (1.0 - m.directional_accuracy)
    )


def select_champion_model(
    models: List[ModelMetrics],
    required_improvement_over_linear: float = 0.03,   # 3% better than AutoDLinear
    required_improvement_for_graph: float = 0.05,     # 5% better than AutoDLinear if graph_stgcnn
    critical_flags: Set[str] = CRITICAL_GUARDRAIL_FLAGS,
) -> ChampionSelection:
    """
    Deterministically select a champion model family for a symbol/horizon.

    Rules:
    - Compute a composite score (lower is better).
    - Disqualify any model with critical guardrail flags.
    - AutoDLinear is the main linear baseline; BaselineLinear is last-resort.
    - Only promote a more complex model if it beats the linear baseline by
      `required_improvement_over_linear` (or `required_improvement_for_graph` for graph_stgcnn).
    - If everything is disqualified, fall back to BaselineLinear (if present),
      otherwise the best-scoring model ignoring guardrails.
    """

    if not models:
        raise ValueError("select_champion_model() called with empty model list")

    # Index by family for convenience
    by_family: Dict[str, ModelMetrics] = {m.family: m for m in models}

    # Compute scores
    scores: Dict[str, float] = {m.family: _composite_score(m) for m in models}

    # Track disqualifications
    disqualified: Dict[str, Set[str]] = {}

    # Step 1: disqualify models with critical guardrail flags
    eligible: List[ModelMetrics] = []
    for m in models:
        bad_flags = m.guardrail_flags & critical_flags
        if bad_flags:
            disqualified[m.family] = bad_flags
        else:
            eligible.append(m)

    # Identify baselines
    auto_dlinear = by_family.get("AutoDLinear")
    baseline_linear = by_family.get("BaselineLinear")

    # Helper: pick best by score from a list of families
    def _best_of(candidates: List[ModelMetrics]) -> ModelMetrics:
        return min(candidates, key=lambda m: scores[m.family])

    # If nothing eligible, fall back to BaselineLinear or best overall
    if not eligible:
        if baseline_linear is not None:
            return ChampionSelection(
                champion=baseline_linear,
                reason="All models disqualified by critical guardrails; falling back to BaselineLinear.",
                scores=scores,
                disqualified=disqualified,
            )
        # really bad situation – choose best ignoring guardrails
        fallback = _best_of(models)
        return ChampionSelection(
            champion=fallback,
            reason="All models disqualified, no BaselineLinear available; using best composite score ignoring guardrails.",
            scores=scores,
            disqualified=disqualified,
        )

    # Determine baseline score to compare against
    baseline_for_comparison: Optional[ModelMetrics] = auto_dlinear or baseline_linear
    baseline_score: Optional[float] = scores[baseline_for_comparison.family] if baseline_for_comparison else None

    # Prefer non-BaselineLinear candidates when possible
    non_baseline_candidates = [m for m in eligible if m.family != "BaselineLinear"]
    if not non_baseline_candidates:
        # Only BaselineLinear survived → use it
        champ = baseline_linear or _best_of(eligible)
        return ChampionSelection(
            champion=champ,
            reason="Only BaselineLinear eligible after guardrails; using BaselineLinear as champion.",
            scores=scores,
            disqualified=disqualified,
        )

    # Best non-baseline candidate by composite score
    raw_best = _best_of(non_baseline_candidates)

    # If we have a linear baseline, enforce improvement margin
    if baseline_for_comparison and baseline_score is not None:
        best_score = scores[raw_best.family]
        required_margin = (
            required_improvement_for_graph
            if raw_best.family == "graph_stgcnn"
            else required_improvement_over_linear
        )

        # improvement required: baseline_score - best_score >= required_margin * baseline_score
        improvement = baseline_score - best_score
        needed = required_margin * baseline_score

        if improvement < needed:
            # Deep/graph model not clearly better than linear baseline → keep baseline
            return ChampionSelection(
                champion=baseline_for_comparison,
                reason=(
                    f"{raw_best.family} did not beat {baseline_for_comparison.family} "
                    f"by required margin ({improvement:.4f} < {needed:.4f}); keeping linear baseline."
                ),
                scores=scores,
                disqualified=disqualified,
            )

    # Otherwise, promote the best candidate
    return ChampionSelection(
        champion=raw_best,
        reason=f"Selected {raw_best.family} as champion based on lowest composite score.",
        scores=scores,
        disqualified=disqualified,
    )
