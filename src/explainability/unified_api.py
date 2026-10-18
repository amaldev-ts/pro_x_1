"""
unified_api.py

Unified, model-agnostic explanation interface for X-Resilience-WDS.
Provides a clean, uniform API for generating genuine SHAP and LIME explanations
across Random Forest, XGBoost, and SpatioTemporalGNN detectors.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from src.explainability.base import ExplanationResult
from src.explainability.shap_explainer import TreeSHAPExplainer, GNNKernelSHAPExplainer
from src.explainability.lime_explainer import TabularLIMEExplainer, GNNLIMEExplainer

logger = logging.getLogger(__name__)


def explain_alert(
    model_id: str,
    model: Any,
    instance: np.ndarray,
    feature_names: List[str],
    background_data: Optional[np.ndarray] = None,
    method: str = "SHAP",
    top_k: int = 5,
    categorical_features: Optional[List[int]] = None,
    gnn_score_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    random_state: int = 42,
    **kwargs: Any,
) -> Union[ExplanationResult, Dict[str, ExplanationResult]]:
    """Generate genuine model explanations using SHAP, LIME, or both.

    Args:
        model_id: Model identifier ('RandomForest', 'XGBoost', or 'SpatioTemporalGNN').
        model: Trained detector object (or None if using gnn_score_fn).
        instance: Input vector (scaled features for tabular, or sensor deltas for GNN).
        feature_names: Exact ordered list of sensor feature names.
        background_data: Representative background array (fitted only on normal training data).
        method: Explainer method ('SHAP', 'LIME', or 'BOTH').
        top_k: Number of highest-magnitude features to rank.
        categorical_features: Indices of binary categorical features (for LIME).
        gnn_score_fn: Scoring function mapping (B, 43) deltas to (B,) scores (for GNN).
        random_state: Random seed for reproducibility.

    Returns:
        ExplanationResult (if method is 'SHAP' or 'LIME') or
        Dict[str, ExplanationResult] (if method is 'BOTH').
    """
    method_upper = method.upper()

    if method_upper not in {"SHAP", "LIME", "BOTH"}:
        raise ValueError(f"Unknown explanation method '{method}'. Choose 'SHAP', 'LIME', or 'BOTH'.")

    results: Dict[str, ExplanationResult] = {}

    # 1. SHAP Branch
    if method_upper in {"SHAP", "BOTH"}:
        if model_id in {"RandomForest", "XGBoost"}:
            explainer_shap = TreeSHAPExplainer(
                model=model,
                model_id=model_id,
                feature_names=feature_names,
                background_data=background_data,
            )
            res_shap = explainer_shap.explain_instance(instance=instance, top_k=top_k)
            results["SHAP"] = res_shap
        elif model_id == "SpatioTemporalGNN":
            if gnn_score_fn is None:
                raise ValueError("gnn_score_fn must be provided when explaining SpatioTemporalGNN.")
            explainer_shap = GNNKernelSHAPExplainer(
                gnn_score_fn=gnn_score_fn,
                feature_names=feature_names,
                nsamples=kwargs.get("nsamples", 44),
            )
            res_shap = explainer_shap.explain_instance(sensor_delta=instance, top_k=top_k)
            results["SHAP"] = res_shap
        else:
            raise ValueError(f"Unsupported model_id '{model_id}' for SHAP.")

    # 2. LIME Branch
    if method_upper in {"LIME", "BOTH"}:
        if model_id in {"RandomForest", "XGBoost"}:
            if background_data is None:
                raise ValueError("background_data must be provided for TabularLIMEExplainer.")
            explainer_lime = TabularLIMEExplainer(
                training_data=background_data,
                feature_names=feature_names,
                model=model,
                model_id=model_id,
                categorical_features=categorical_features,
                random_state=random_state,
            )
            res_lime = explainer_lime.explain_instance(
                instance=instance,
                top_k=top_k,
                num_samples=kwargs.get("num_samples", 500),
            )
            results["LIME"] = res_lime
        elif model_id == "SpatioTemporalGNN":
            if gnn_score_fn is None:
                raise ValueError("gnn_score_fn must be provided when explaining SpatioTemporalGNN.")
            explainer_lime = GNNLIMEExplainer(
                gnn_score_fn=gnn_score_fn,
                feature_names=feature_names,
                random_state=random_state,
            )
            res_lime = explainer_lime.explain_instance(
                sensor_delta=instance,
                top_k=top_k,
                num_samples=kwargs.get("num_samples", 300),
            )
            results["LIME"] = res_lime
        else:
            raise ValueError(f"Unsupported model_id '{model_id}' for LIME.")

    if method_upper == "BOTH":
        return results
    return results[method_upper]
