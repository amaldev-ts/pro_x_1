"""
base.py

Core data structures and interfaces for genuine SHAP and LIME explainability.
Provides standardized explanation result representations, attribution normalization,
and explainer base classes across tabular and temporal WDS detection models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ExplanationResult:
    """Standardized explanation result container.

    Attributes:
        method: Explainer method name ('SHAP' or 'LIME').
        model_id: Identifier of explained model ('RandomForest', 'XGBoost', or 'SpatioTemporalGNN').
        prediction: Scalar model prediction (probability of attack or system anomaly score).
        prediction_label: Human-readable classification ('Attack', 'Normal', or 'Statistical Anomaly').
        top_k_features: Ordered list of top-K most influential sensor names.
        attributions: Mapping from feature name to signed attribution value.
        abs_attributions: Mapping from feature name to absolute attribution magnitude.
        normalized_attributions: Mapping from feature name to percentage contribution (summing to 100%).
        runtime_seconds: Execution time for generating the explanation.
        metadata: Configuration, background sample counts, random seeds, and explainer-specific details.
    """
    method: str
    model_id: str
    prediction: float
    prediction_label: str
    top_k_features: List[str]
    attributions: Dict[str, float]
    abs_attributions: Dict[str, float]
    normalized_attributions: Dict[str, float]
    runtime_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to serializable dictionary."""
        return {
            "method": self.method,
            "model_id": self.model_id,
            "prediction": float(self.prediction),
            "prediction_label": self.prediction_label,
            "top_k_features": self.top_k_features,
            "attributions": {k: float(v) for k, v in self.attributions.items()},
            "abs_attributions": {k: float(v) for k, v in self.abs_attributions.items()},
            "normalized_attributions": {k: float(v) for k, v in self.normalized_attributions.items()},
            "runtime_seconds": float(self.runtime_seconds),
            "metadata": self.metadata,
        }


def compute_normalized_attributions(
    attributions: Dict[str, float],
    top_k: int = 5,
) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    """Sort and normalize feature attributions by absolute magnitude.

    Args:
        attributions: Mapping of feature names to signed values.
        top_k: Number of leading features to return in top_k_features.

    Returns:
        Tuple of (top_k_features, abs_attributions, normalized_attributions)
    """
    sorted_items = sorted(attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_k_keys = [k for k, _ in sorted_items[:top_k]]

    abs_map = {k: abs(v) for k, v in sorted_items}
    total_abs = sum(abs_map.values())

    if total_abs > 1e-12:
        norm_map = {k: (abs_val / total_abs) * 100.0 for k, abs_val in abs_map.items()}
    else:
        norm_map = {k: 0.0 for k in abs_map}

    return top_k_keys, abs_map, norm_map
