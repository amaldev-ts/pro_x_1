"""
lime_explainer.py

Genuine LIME explainability implementations using the official `lime` library.
Provides:
  1. TabularLIMEExplainer: Wraps lime.lime_tabular.LimeTabularExplainer for RF and XGBoost.
  2. GNNLIMEExplainer: Model-agnostic LimeTabularExplainer adapter for SpatioTemporalGNN.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from src.explainability.base import ExplanationResult, compute_normalized_attributions

logger = logging.getLogger(__name__)


class TabularLIMEExplainer:
    """Genuine LIME explainer for tabular tree detectors (Random Forest and XGBoost).

    Uses `lime.lime_tabular.LimeTabularExplainer` fitted strictly on representative
    training data with deterministic random seed and explicit binary categorical declarations.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        model: Any,
        model_id: str,
        categorical_features: Optional[List[int]] = None,
        random_state: int = 42,
        kernel_width: Optional[float] = None,
    ):
        """Initialize TabularLIMEExplainer.

        Args:
            training_data: 2D array of representative training samples (fitted only on normal train data).
            feature_names: Exact ordered list of sensor feature names (43 sensors).
            model: Trained classifier model providing `predict_proba`.
            model_id: Model identifier string ('RandomForest' or 'XGBoost').
            categorical_features: Column indices of binary status channels (e.g. S_PU*, S_V*).
            random_state: Fixed seed for deterministic perturbation sampling.
            kernel_width: Kernel width for exponential distance weighting.
        """
        self.model = model
        self.model_id = model_id
        self.feature_names = list(feature_names)
        self.num_features = len(self.feature_names)
        self.random_state = random_state

        # Auto-detect binary status channels if not explicitly passed
        if categorical_features is None:
            categorical_features = [
                i for i, name in enumerate(self.feature_names) if name.startswith("S_")
            ]
        self.categorical_features = categorical_features

        # Initialize official LimeTabularExplainer
        self.explainer = LimeTabularExplainer(
            training_data=np.asarray(training_data, dtype=np.float64),
            feature_names=self.feature_names,
            class_names=["Normal", "Attack"],
            categorical_features=self.categorical_features,
            mode="classification",
            random_state=self.random_state,
            kernel_width=kernel_width,
            verbose=False,
        )

    def explain_instance(
        self,
        instance: np.ndarray,
        top_k: int = 5,
        num_features_to_sample: int = 15,
        target_class: int = 1,
        num_samples: int = 500,
    ) -> ExplanationResult:
        """Compute local LIME linear surrogate explanation for a single sensor feature vector.

        Args:
            instance: 1D array of shape (F,) or 2D array of shape (1, F).
            top_k: Number of highest-magnitude features to rank.
            num_features_to_sample: Number of terms in the sparse linear surrogate explanation.
            target_class: Class to explain (default 1 = attack positive class).
            num_samples: Number of perturbed samples to draw in local neighborhood.

        Returns:
            ExplanationResult containing signed surrogate weights, normalized contributions, and top-K ranking.
        """
        t0 = time.perf_counter()
        inst_1d = np.asarray(instance, dtype=np.float64).flatten()

        if len(inst_1d) != self.num_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.num_features} features, got {len(inst_1d)}"
            )

        # Prediction function wrapper
        predict_fn = getattr(self.model, "predict_proba", None)
        if predict_fn is None:
            raise AttributeError(f"Model {self.model_id} does not expose predict_proba.")

        # Execute genuine LIME local perturbation and weighted ridge regression
        exp = self.explainer.explain_instance(
            data_row=inst_1d,
            predict_fn=predict_fn,
            labels=(target_class,),
            num_features=min(num_features_to_sample, self.num_features),
            num_samples=num_samples,
        )

        # Extract LIME explanation weights directly via exact integer feature indexing
        # avoiding any string parsing or substring token collision
        attributions: Dict[str, float] = {name: 0.0 for name in self.feature_names}
        local_key = target_class if target_class in exp.local_exp else list(exp.local_exp.keys())[0]
        for feat_idx, weight in exp.local_exp.get(local_key, []):
            if 0 <= feat_idx < self.num_features:
                attributions[self.feature_names[feat_idx]] = float(weight)

        top_k_keys, abs_map, norm_map = compute_normalized_attributions(attributions, top_k=top_k)

        # Get actual model probability prediction
        probs = predict_fn(inst_1d.reshape(1, -1))
        pred_prob = float(probs[0, target_class]) if probs.shape[1] > target_class else float(probs[0, -1])
        label = "Attack" if pred_prob >= 0.5 else "Normal"
        runtime = time.perf_counter() - t0

        return ExplanationResult(
            method="LIME",
            model_id=self.model_id,
            prediction=pred_prob,
            prediction_label=label,
            top_k_features=top_k_keys,
            attributions=attributions,
            abs_attributions=abs_map,
            normalized_attributions=norm_map,
            runtime_seconds=runtime,
            metadata={
                "explainer_class": "lime.lime_tabular.LimeTabularExplainer",
                "random_state": self.random_state,
                "num_samples": num_samples,
                "target_class": target_class,
                "categorical_features_count": len(self.categorical_features),
            },
        )


class GNNLIMEExplainer:
    """Model-agnostic LIME surrogate adapter for SpatioTemporalGNN anomaly scoring.

    Explains how each sensor's perturbation drives the system anomaly score
    using `lime.lime_tabular.LimeTabularExplainer` in regression mode.
    """

    def __init__(
        self,
        gnn_score_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: List[str],
        reference_deltas: Optional[np.ndarray] = None,
        categorical_features: Optional[List[int]] = None,
        random_state: int = 42,
    ):
        """Initialize GNNLIMEExplainer.

        Args:
            gnn_score_fn: Callable taking (B, 43) sensor delta vectors and returning (B,) scalar anomaly scores.
            feature_names: Ordered list of sensor feature names (43 sensors).
            reference_deltas: Background delta matrix (defaults to normal zero delta with small noise).
            categorical_features: Column indices of discrete/binary categorical features if applicable.
            random_state: Fixed random seed for reproducibility.
        """
        self.gnn_score_fn = gnn_score_fn
        self.feature_names = list(feature_names)
        self.num_features = len(self.feature_names)
        self.random_state = random_state
        self.categorical_features = categorical_features

        if reference_deltas is None:
            # Generate small reference background centered around 0 (normal baseline)
            rng = np.random.RandomState(random_state)
            self.reference_deltas = rng.normal(0.0, 0.2, size=(100, self.num_features)).astype(np.float64)
        else:
            self.reference_deltas = np.asarray(reference_deltas, dtype=np.float64)

        self.explainer = LimeTabularExplainer(
            training_data=self.reference_deltas,
            feature_names=self.feature_names,
            categorical_features=self.categorical_features,
            mode="regression",
            random_state=self.random_state,
            verbose=False,
        )

    def explain_instance(
        self,
        sensor_delta: np.ndarray,
        baseline_score: float = 1.0,
        top_k: int = 5,
        num_samples: int = 300,
    ) -> ExplanationResult:
        """Compute local linear surrogate weights for sensor contributions to GNN anomaly score.

        Args:
            sensor_delta: 1D array of shape (43,) or 2D array of shape (1, 43).
            baseline_score: Normal baseline anomaly score.
            top_k: Number of highest-magnitude features to rank.
            num_samples: Number of perturbed samples for LIME regression.

        Returns:
            ExplanationResult containing signed surrogate coefficients and top-K ranked sensors.
        """
        t0 = time.perf_counter()
        delta_1d = np.asarray(sensor_delta, dtype=np.float64).flatten()

        current_score = float(self.gnn_score_fn(delta_1d.reshape(1, -1))[0])

        def score_wrapper(X_mat: np.ndarray) -> np.ndarray:
            return self.gnn_score_fn(X_mat)

        exp = self.explainer.explain_instance(
            data_row=delta_1d,
            predict_fn=score_wrapper,
            num_features=self.num_features,
            num_samples=num_samples,
        )

        # Extract weights via direct integer feature indexing, avoiding substring matching bugs
        attributions: Dict[str, float] = {name: 0.0 for name in self.feature_names}
        local_key = 1 if 1 in exp.local_exp else list(exp.local_exp.keys())[0]
        for feat_idx, weight in exp.local_exp.get(local_key, []):
            if 0 <= feat_idx < self.num_features:
                attributions[self.feature_names[feat_idx]] = float(weight)

        top_k_keys, abs_map, norm_map = compute_normalized_attributions(attributions, top_k=top_k)

        label = "Statistical Anomaly" if current_score >= 3.013382 else "Normal"
        runtime = time.perf_counter() - t0

        return ExplanationResult(
            method="LIME",
            model_id="SpatioTemporalGNN",
            prediction=current_score,
            prediction_label=label,
            top_k_features=top_k_keys,
            attributions=attributions,
            abs_attributions=abs_map,
            normalized_attributions=norm_map,
            runtime_seconds=runtime,
            metadata={
                "explainer_class": "lime.lime_tabular.LimeTabularExplainer (regression)",
                "random_state": self.random_state,
                "num_samples": num_samples,
                "baseline_score": float(baseline_score),
                "surrogate_r2": float(exp.score) if hasattr(exp, "score") and exp.score is not None else 0.0,
            },
        )
