"""
shap_explainer.py

Genuine SHAP explainability implementations using the official `shap` library.
Provides:
  1. TreeSHAPExplainer: Wraps shap.TreeExplainer for Random Forest and XGBoost detectors.
  2. GNNKernelSHAPExplainer: Model-agnostic shap.KernelExplainer adapter for SpatioTemporalGNN.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import shap

from src.explainability.base import ExplanationResult, compute_normalized_attributions

logger = logging.getLogger(__name__)


class TreeSHAPExplainer:
    """Genuine TreeSHAP explainer for Random Forest and XGBoost tree ensemble detectors.

    Uses `shap.TreeExplainer` with exact polynomial-time Shapley computation.
    """

    def __init__(
        self,
        model: Any,
        model_id: str,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
        max_background_samples: int = 100,
    ):
        """Initialize TreeSHAPExplainer.

        Args:
            model: Trained tree ensemble model (RandomForestClassifier or XGBClassifier).
            model_id: Model identifier string ('RandomForest' or 'XGBoost').
            feature_names: Exact ordered list of sensor feature names.
            background_data: Optional reference background data (fitted only on normal training data).
            max_background_samples: Maximum background rows to sample for TreeExplainer background.
        """
        self.model = model
        self.model_id = model_id
        self.feature_names = list(feature_names)
        self.num_features = len(self.feature_names)

        # Subsample background if provided to keep evaluation fast
        bg_sample = None
        if background_data is not None:
            if len(background_data) > max_background_samples:
                rng = np.random.RandomState(42)
                indices = rng.choice(len(background_data), size=max_background_samples, replace=False)
                bg_sample = background_data[indices]
            else:
                bg_sample = background_data

        self.background_data = bg_sample

        try:
            # Note: For TreeExplainer, background data activates interventional feature perturbation
            if bg_sample is not None:
                self.explainer = shap.TreeExplainer(self.model, data=bg_sample)
            else:
                self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning("TreeExplainer initialization with background failed (%s). Falling back to tree-path explainer.", e)
            self.explainer = shap.TreeExplainer(self.model)

    def explain_instance(
        self,
        instance: np.ndarray,
        top_k: int = 5,
        target_class: int = 1,
    ) -> ExplanationResult:
        """Compute local SHAP attribution values for a single input vector.

        Args:
            instance: 1D array of shape (F,) or 2D array of shape (1, F).
            top_k: Number of highest-magnitude features to rank.
            target_class: Target class to explain (default 1 = attack positive class).

        Returns:
            ExplanationResult containing signed attributions, normalized contributions, and top-K ranking.
        """
        t0 = time.perf_counter()
        inst_2d = np.asarray(instance, dtype=np.float32)
        if inst_2d.ndim == 1:
            inst_2d = inst_2d.reshape(1, -1)

        if inst_2d.shape[1] != self.num_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.num_features} features, got {inst_2d.shape[1]}"
            )

        # Compute raw SHAP values
        raw_shap = self.explainer.shap_values(inst_2d)

        # Handle different SHAP output formats across sklearn and xgboost versions:
        # 1. 3D ndarray (1, F, num_classes) -> extract target_class slice
        # 2. list of length num_classes, each element (1, F) -> extract list[target_class]
        # 3. 2D ndarray (1, F) -> already single target class (common in XGBoost binary)
        if isinstance(raw_shap, list):
            class_idx = min(target_class, len(raw_shap) - 1)
            shap_vec = np.asarray(raw_shap[class_idx])[0]
        elif isinstance(raw_shap, np.ndarray):
            if raw_shap.ndim == 3:
                class_idx = min(target_class, raw_shap.shape[2] - 1)
                shap_vec = raw_shap[0, :, class_idx]
            elif raw_shap.ndim == 2:
                shap_vec = raw_shap[0]
            elif raw_shap.ndim == 1:
                shap_vec = raw_shap
            else:
                raise ValueError(f"Unexpected SHAP array dimensions: {raw_shap.shape}")
        else:
            shap_vec = np.asarray(raw_shap).flatten()

        attributions = {name: float(shap_vec[i]) for i, name in enumerate(self.feature_names)}
        top_k_keys, abs_map, norm_map = compute_normalized_attributions(attributions, top_k=top_k)

        # Get model probability prediction
        pred_prob = 0.0
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(inst_2d)
                pred_prob = float(probs[0, target_class]) if probs.shape[1] > target_class else float(probs[0, -1])
            except Exception:
                pred_prob = float(self.model.predict(inst_2d)[0])
        elif hasattr(self.model, "predict"):
            pred_prob = float(self.model.predict(inst_2d)[0])

        label = "Attack" if pred_prob >= 0.5 else "Normal"
        runtime = time.perf_counter() - t0

        return ExplanationResult(
            method="SHAP",
            model_id=self.model_id,
            prediction=pred_prob,
            prediction_label=label,
            top_k_features=top_k_keys,
            attributions=attributions,
            abs_attributions=abs_map,
            normalized_attributions=norm_map,
            runtime_seconds=runtime,
            metadata={
                "explainer_class": "shap.TreeExplainer",
                "target_class": target_class,
                "background_samples": len(self.background_data) if self.background_data is not None else 0,
            },
        )

    def compute_global_importance(
        self,
        dataset: np.ndarray,
        top_k: int = 10,
        target_class: int = 1,
    ) -> Dict[str, float]:
        """Compute global feature importance as mean absolute SHAP value across samples.

        Args:
            dataset: 2D array of shape (N, F).
            top_k: Number of top features to return.
            target_class: Target class index.

        Returns:
            Dictionary of {feature_name: mean_abs_shap} sorted descending.
        """
        data_arr = np.asarray(dataset, dtype=np.float32)
        raw_shap = self.explainer.shap_values(data_arr)

        if isinstance(raw_shap, list):
            class_idx = min(target_class, len(raw_shap) - 1)
            shap_mat = np.asarray(raw_shap[class_idx])
        elif isinstance(raw_shap, np.ndarray):
            if raw_shap.ndim == 3:
                class_idx = min(target_class, raw_shap.shape[2] - 1)
                shap_mat = raw_shap[:, :, class_idx]
            else:
                shap_mat = raw_shap
        else:
            shap_mat = np.asarray(raw_shap)

        mean_abs = np.mean(np.abs(shap_mat), axis=0)
        sorted_indices = np.argsort(mean_abs)[::-1][:top_k]

        return {self.feature_names[i]: float(mean_abs[i]) for i in sorted_indices}


class GNNKernelSHAPExplainer:
    """Model-agnostic KernelSHAP explainer for SpatioTemporalGNN.

    Explains the scalar system anomaly score S(X) by mapping sensor-level perturbations
    back to the 3D multivariate time-series window (Batch, Window=12, Features=43).
    """

    def __init__(
        self,
        gnn_score_fn: Any,
        feature_names: List[str],
        background_deltas: Optional[np.ndarray] = None,
        nsamples: int = 44,
    ):
        """Initialize GNNKernelSHAPExplainer.

        Args:
            gnn_score_fn: Callable taking (B, 43) sensor delta vectors and returning (B,) scalar anomaly scores.
            feature_names: Ordered list of sensor feature names (43 sensors).
            background_deltas: Reference delta array of shape (N_bg, 43). Defaults to all-zeros reference.
            nsamples: Number of Monte Carlo coalition evaluations for shap.KernelExplainer.
        """
        self.gnn_score_fn = gnn_score_fn
        self.feature_names = list(feature_names)
        self.num_features = len(self.feature_names)
        self.nsamples = nsamples

        if background_deltas is None:
            self.background_deltas = np.zeros((1, self.num_features), dtype=np.float32)
        else:
            self.background_deltas = np.asarray(background_deltas, dtype=np.float32)

        self.explainer = shap.KernelExplainer(self.gnn_score_fn, self.background_deltas)

    def explain_instance(
        self,
        sensor_delta: np.ndarray,
        baseline_score: float = 1.0,
        top_k: int = 5,
    ) -> ExplanationResult:
        """Compute Shapley attribution values for each sensor's perturbation in the GNN input.

        Args:
            sensor_delta: 1D array of shape (43,) or 2D array of shape (1, 43).
            baseline_score: Normal baseline anomaly score for reference.
            top_k: Number of highest-magnitude features to rank.

        Returns:
            ExplanationResult containing signed Shapley values and top-K contributing sensors.
        """
        t0 = time.perf_counter()
        delta_2d = np.asarray(sensor_delta, dtype=np.float32)
        if delta_2d.ndim == 1:
            delta_2d = delta_2d.reshape(1, -1)

        current_score = float(self.gnn_score_fn(delta_2d)[0])
        raw_shap = self.explainer.shap_values(delta_2d, nsamples=self.nsamples, silent=True)

        if isinstance(raw_shap, list):
            shap_vec = np.asarray(raw_shap[0]).flatten()
        elif isinstance(raw_shap, np.ndarray):
            shap_vec = raw_shap.flatten()
        else:
            shap_vec = np.asarray(raw_shap).flatten()

        attributions = {name: float(shap_vec[i]) for i, name in enumerate(self.feature_names)}
        top_k_keys, abs_map, norm_map = compute_normalized_attributions(attributions, top_k=top_k)

        label = "Statistical Anomaly" if current_score >= 3.013382 else "Normal"
        runtime = time.perf_counter() - t0

        return ExplanationResult(
            method="SHAP",
            model_id="SpatioTemporalGNN",
            prediction=current_score,
            prediction_label=label,
            top_k_features=top_k_keys,
            attributions=attributions,
            abs_attributions=abs_map,
            normalized_attributions=norm_map,
            runtime_seconds=runtime,
            metadata={
                "explainer_class": "shap.KernelExplainer",
                "nsamples": self.nsamples,
                "baseline_score": float(baseline_score),
            },
        )
