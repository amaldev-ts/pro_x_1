"""
evaluation.py

Reliability and quality evaluation suite for model explanations.
Implements Roadmap Phase 5 criteria:
  A. Perturbation-based Fidelity: Score reduction upon masking top-K sensors vs random vs raw deviation.
  B. Explanation Stability: Consistency under repeated input perturbations / background subsampling.
  C. SHAP-LIME Agreement: Top-K Jaccard overlap, Spearman rank correlation, and sign consistency.
  D. Ground-Truth Attribution: Precision@K against known benchmark attack target components.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from scipy.stats import spearmanr

from src.explainability.base import ExplanationResult

logger = logging.getLogger(__name__)


def evaluate_perturbation_fidelity(
    model_predict_fn: Callable[[np.ndarray], float],
    instance: np.ndarray,
    top_k_features: List[str],
    feature_names: List[str],
    reference_baseline: Optional[np.ndarray] = None,
    num_random_trials: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Measure score reduction when masking top-K influential sensors vs random K vs raw deviation.

    Masking strategy replaces selected feature values with their reference baseline (0.0 in standardized space).

    Args:
        model_predict_fn: Function mapping (1, F) feature array to scalar score or probability.
        instance: Original unmasked instance of shape (F,) or (1, F).
        top_k_features: Top-K sensor names identified by the explanation method.
        feature_names: Complete ordered list of sensor feature names.
        reference_baseline: Baseline vector to substitute into masked positions (defaults to zeros).
        num_random_trials: Number of trials for random masking baseline.
        random_state: Random seed for reproducible random trials.

    Returns:
        Dictionary with original score, masked score, score drop, and comparison baselines.
    """
    inst = np.asarray(instance, dtype=np.float32).copy().reshape(1, -1)
    F = inst.shape[1]
    K = len(top_k_features)

    if reference_baseline is None:
        baseline = np.zeros_like(inst)
    else:
        baseline = np.asarray(reference_baseline, dtype=np.float32).reshape(1, -1)

    # 1. Original score
    orig_score = float(model_predict_fn(inst))

    # 2. Mask explained top-K sensors
    inst_masked_exp = inst.copy()
    top_k_indices = [feature_names.index(feat) for feat in top_k_features if feat in feature_names]
    for idx in top_k_indices:
        inst_masked_exp[0, idx] = baseline[0, idx]

    score_masked_exp = float(model_predict_fn(inst_masked_exp))
    drop_exp = orig_score - score_masked_exp

    # 3. Mask top-K sensors ranked by raw deviation from baseline
    raw_deviations = np.abs(inst[0] - baseline[0])
    raw_top_k_idx = np.argsort(raw_deviations)[::-1][:K]
    inst_masked_raw = inst.copy()
    for idx in raw_top_k_idx:
        inst_masked_raw[0, idx] = baseline[0, idx]

    score_masked_raw = float(model_predict_fn(inst_masked_raw))
    drop_raw = orig_score - score_masked_raw

    # 4. Mask random K sensors across repeated trials
    rng = np.random.RandomState(random_state)
    random_drops = []
    for _ in range(num_random_trials):
        rand_idx = rng.choice(F, size=min(K, F), replace=False)
        inst_rand = inst.copy()
        for idx in rand_idx:
            inst_rand[0, idx] = baseline[0, idx]
        s_rand = float(model_predict_fn(inst_rand))
        random_drops.append(orig_score - s_rand)

    mean_random_drop = float(np.mean(random_drops))
    std_random_drop = float(np.std(random_drops))

    # Relative fidelity ratio: does explanation masking reduce score more than random masking?
    fidelity_advantage = drop_exp - mean_random_drop

    return {
        "original_score": orig_score,
        "k_masked": K,
        "explained_top_k_score": score_masked_exp,
        "explained_score_drop": drop_exp,
        "raw_deviation_score_drop": drop_raw,
        "random_k_mean_score_drop": mean_random_drop,
        "random_k_std_score_drop": std_random_drop,
        "fidelity_advantage_over_random": fidelity_advantage,
        "evaluation_description": "Perturbation-based fidelity test measuring prediction change under feature masking.",
    }


def evaluate_explanation_stability(
    explain_fn: Callable[[np.ndarray], ExplanationResult],
    instance: np.ndarray,
    noise_std: float = 0.05,
    num_perturbations: int = 5,
    top_k: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluate stability of explanations under small input feature perturbations.

    Args:
        explain_fn: Callable taking input vector and returning an ExplanationResult.
        instance: Input feature vector.
        noise_std: Standard deviation of Gaussian perturbation noise added to input.
        num_perturbations: Number of noisy variants evaluated.
        top_k: Size of top-K feature set compared.
        random_state: Random seed.

    Returns:
        Dictionary of mean Jaccard overlap, mean rank correlation, and stability assessment.
    """
    base_exp = explain_fn(instance)
    base_top_k = set(base_exp.top_k_features[:top_k])
    base_attrs = [base_exp.attributions.get(k, 0.0) for k in base_exp.top_k_features[:top_k]]

    rng = np.random.RandomState(random_state)
    inst_arr = np.asarray(instance, dtype=np.float32)

    jaccard_scores = []
    rank_correlations = []

    for _ in range(num_perturbations):
        perturbed_inst = inst_arr + rng.normal(0.0, noise_std, size=inst_arr.shape).astype(np.float32)
        pert_exp = explain_fn(perturbed_inst)
        pert_top_k = set(pert_exp.top_k_features[:top_k])

        # Jaccard index
        intersection = len(base_top_k.intersection(pert_top_k))
        union = len(base_top_k.union(pert_top_k))
        jaccard = intersection / union if union > 0 else 1.0
        jaccard_scores.append(jaccard)

        # Spearman correlation on base top_k features
        pert_attrs = [pert_exp.attributions.get(k, 0.0) for k in base_exp.top_k_features[:top_k]]
        if len(base_attrs) >= 3:
            corr, _ = spearmanr(base_attrs, pert_attrs)
            if not np.isnan(corr):
                rank_correlations.append(corr)

    mean_jaccard = float(np.mean(jaccard_scores))
    mean_corr = float(np.mean(rank_correlations)) if rank_correlations else 1.0

    return {
        "num_perturbations": num_perturbations,
        "noise_std": noise_std,
        "mean_top_k_jaccard_stability": mean_jaccard,
        "mean_rank_correlation_stability": mean_corr,
        "stability_category": "High" if mean_jaccard >= 0.7 else ("Moderate" if mean_jaccard >= 0.4 else "Low"),
    }


def compute_shap_lime_agreement(
    shap_result: ExplanationResult,
    lime_result: ExplanationResult,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Compute consensus and agreement statistics between genuine SHAP and LIME explanations.

    Args:
        shap_result: ExplanationResult produced by SHAP.
        lime_result: ExplanationResult produced by LIME.
        top_k: Number of leading features to evaluate for agreement.

    Returns:
        Dictionary with Jaccard overlap, common features, Spearman rank correlation, and sign consistency.
    """
    shap_top = shap_result.top_k_features[:top_k]
    lime_top = lime_result.top_k_features[:top_k]

    set_shap = set(shap_top)
    set_lime = set(lime_top)

    common_features = list(set_shap.intersection(set_lime))
    overlap_count = len(common_features)
    union_count = len(set_shap.union(set_lime))
    jaccard_similarity = overlap_count / union_count if union_count > 0 else 1.0

    # Sign agreement on overlapping features
    sign_matches = 0
    for feat in common_features:
        s_val = shap_result.attributions.get(feat, 0.0)
        l_val = lime_result.attributions.get(feat, 0.0)
        if (s_val >= 0 and l_val >= 0) or (s_val < 0 and l_val < 0):
            sign_matches += 1
    sign_agreement_rate = sign_matches / overlap_count if overlap_count > 0 else 1.0

    # Spearman rank correlation across all features
    all_features = sorted(list(set(shap_result.attributions.keys()).union(lime_result.attributions.keys())))
    s_vec = [shap_result.attributions.get(f, 0.0) for f in all_features]
    l_vec = [lime_result.attributions.get(f, 0.0) for f in all_features]

    spearman_corr, p_val = spearmanr(s_vec, l_vec)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0

    if overlap_count >= 4:
        level = "High"
    elif overlap_count >= 2:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "top_k": top_k,
        "overlap_count": overlap_count,
        "jaccard_similarity": float(jaccard_similarity),
        "common_features": common_features,
        "sign_agreement_rate": float(sign_agreement_rate),
        "spearman_rank_correlation": float(spearman_corr),
        "spearman_p_value": float(p_val) if not np.isnan(p_val) else 1.0,
        "agreement_level": level,
    }


def evaluate_ground_truth_attribution(
    top_k_features: List[str],
    attacked_component: str,
    component_sensor_map: Dict[str, List[str]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Check whether top-K influential sensors match sensors associated with the known attacked component.

    Distinct from full root-cause physical localization.

    Args:
        top_k_features: Ordered list of top-K explained sensors.
        attacked_component: Ground truth target component name (e.g. 'PU1', 'T1', 'V2', 'J280').
        component_sensor_map: Mapping from component ID to list of related sensor names.
        top_k: Evaluated cut-off.

    Returns:
        Dictionary reporting hit count, precision@K, and matched sensors.
    """
    clean_comp = attacked_component.strip()
    related_sensors: Set[str] = set()

    # Look up direct component mapping
    if clean_comp in component_sensor_map:
        related_sensors.update(component_sensor_map[clean_comp])

    # Also match sensor naming heuristics (e.g. PU1 -> F_PU1, S_PU1; T1 -> L_T1; J280 -> P_J280)
    for s_name in ["F_" + clean_comp, "S_" + clean_comp, "L_" + clean_comp, "P_" + clean_comp]:
        related_sensors.add(s_name)

    evaluated_top_k = top_k_features[:top_k]
    matched_hits = [s for s in evaluated_top_k if s in related_sensors or clean_comp in s]

    hit = len(matched_hits) > 0
    precision_at_k = len(matched_hits) / len(evaluated_top_k) if evaluated_top_k else 0.0

    return {
        "attacked_component": clean_comp,
        "top_k_evaluated": evaluated_top_k,
        "matched_sensors": matched_hits,
        "hit": hit,
        "precision_at_k": float(precision_at_k),
    }
