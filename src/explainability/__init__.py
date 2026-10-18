"""
src/explainability package

Genuine SHAP and LIME Explainability Module for X-Resilience-WDS.
Provides model-agnostic and tree-based local and global explanations,
common explanation result containers, and quality evaluation suites.
"""

from src.explainability.base import (
    ExplanationResult,
    compute_normalized_attributions,
)
from src.explainability.shap_explainer import (
    TreeSHAPExplainer,
    GNNKernelSHAPExplainer,
)
from src.explainability.lime_explainer import (
    TabularLIMEExplainer,
    GNNLIMEExplainer,
)
from src.explainability.unified_api import (
    explain_alert,
)
from src.explainability.evaluation import (
    evaluate_perturbation_fidelity,
    evaluate_explanation_stability,
    compute_shap_lime_agreement,
    evaluate_ground_truth_attribution,
)

__all__ = [
    "ExplanationResult",
    "compute_normalized_attributions",
    "TreeSHAPExplainer",
    "GNNKernelSHAPExplainer",
    "TabularLIMEExplainer",
    "GNNLIMEExplainer",
    "explain_alert",
    "evaluate_perturbation_fidelity",
    "evaluate_explanation_stability",
    "compute_shap_lime_agreement",
    "evaluate_ground_truth_attribution",
]
