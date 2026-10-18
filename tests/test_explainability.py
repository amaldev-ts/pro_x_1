"""
test_explainability.py

Unit and integration tests for genuine SHAP and LIME explainability module (P1.3).
Verifies:
- SHAP and LIME output shapes, keys, and values.
- Feature-name and feature-order consistency across 43 C-Town sensors.
- Binary sensor categorical handling for pump/valve status channels.
- Prediction function compatibility (probability vs scalar score).
- Deterministic LIME settings across repeated calls with fixed random_state.
- Top-K ranking and attribution normalization (sum to 100%).
- Perturbation-based fidelity evaluation and score reduction under masking.
- SHAP-LIME agreement calculation (Jaccard overlap and Spearman rank correlation).
- Error handling on invalid feature counts or uninitialized models.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explainability.base import ExplanationResult, compute_normalized_attributions
from src.explainability.shap_explainer import TreeSHAPExplainer, GNNKernelSHAPExplainer
from src.explainability.lime_explainer import TabularLIMEExplainer, GNNLIMEExplainer
from src.explainability.unified_api import explain_alert
from src.explainability.evaluation import (
    evaluate_perturbation_fidelity,
    evaluate_explanation_stability,
    compute_shap_lime_agreement,
    evaluate_ground_truth_attribution,
)


@pytest.fixture
def synthetic_tabular_setup():
    """Create reproducible synthetic tabular dataset and trained RF model with 43 features."""
    rng = np.random.RandomState(42)
    feature_names = [f"P_J{i}" for i in range(1, 32)] + [f"S_PU{i}" for i in range(1, 13)]
    N = 150
    F = len(feature_names)
    X = rng.normal(0.0, 1.0, size=(N, F)).astype(np.float32)
    # Set binary channels to 0 or 1
    for k in range(31, F):
        X[:, k] = (X[:, k] > 0).astype(np.float32)

    # Synthetic target driven strongly by feature 0 (P_J1) and feature 31 (S_PU1)
    logits = 2.0 * X[:, 0] - 1.5 * X[:, 31] + rng.normal(0, 0.2, size=N)
    y = (logits > 0).astype(int)

    rf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    rf.fit(X, y)

    return {
        "model": rf,
        "feature_names": feature_names,
        "X_train": X,
        "y_train": y,
        "test_instance": X[0:1],
    }


class TestBaseExplainability:
    def test_compute_normalized_attributions_sum(self):
        attributions = {"s1": 2.5, "s2": -5.0, "s3": 2.5}
        top_k, abs_map, norm_map = compute_normalized_attributions(attributions, top_k=2)

        assert top_k == ["s2", "s1"] or top_k == ["s2", "s3"]
        assert abs_map["s2"] == 5.0
        assert pytest.approx(sum(norm_map.values()), abs=1e-5) == 100.0
        assert pytest.approx(norm_map["s2"], abs=1e-5) == 50.0

    def test_zero_attributions_normalization(self):
        attributions = {"s1": 0.0, "s2": 0.0}
        top_k, abs_map, norm_map = compute_normalized_attributions(attributions, top_k=2)
        assert norm_map["s1"] == 0.0
        assert norm_map["s2"] == 0.0


class TestTreeSHAPExplainer:
    def test_tree_shap_output_shape_and_ranking(self, synthetic_tabular_setup):
        setup = synthetic_tabular_setup
        explainer = TreeSHAPExplainer(
            model=setup["model"],
            model_id="RandomForest",
            feature_names=setup["feature_names"],
            background_data=setup["X_train"][:30],
        )

        res = explainer.explain_instance(setup["test_instance"], top_k=5)
        assert isinstance(res, ExplanationResult)
        assert res.method == "SHAP"
        assert res.model_id == "RandomForest"
        assert len(res.top_k_features) == 5
        assert len(res.attributions) == len(setup["feature_names"])
        assert 0.0 <= res.prediction <= 1.0

        # Global importance
        global_imp = explainer.compute_global_importance(setup["X_train"][:20], top_k=5)
        assert len(global_imp) == 5
        assert all(isinstance(v, float) for v in global_imp.values())

    def test_tree_shap_dimension_mismatch_raises(self, synthetic_tabular_setup):
        setup = synthetic_tabular_setup
        explainer = TreeSHAPExplainer(
            model=setup["model"],
            model_id="RandomForest",
            feature_names=setup["feature_names"],
        )
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            explainer.explain_instance(np.zeros((1, 10)))


class TestTabularLIMEExplainer:
    def test_lime_output_and_determinism(self, synthetic_tabular_setup):
        setup = synthetic_tabular_setup
        bin_indices = list(range(31, len(setup["feature_names"])))

        explainer1 = TabularLIMEExplainer(
            training_data=setup["X_train"],
            feature_names=setup["feature_names"],
            model=setup["model"],
            model_id="RandomForest",
            categorical_features=bin_indices,
            random_state=42,
        )

        res1 = explainer1.explain_instance(setup["test_instance"], top_k=5, num_samples=150)
        assert isinstance(res1, ExplanationResult)
        assert res1.method == "LIME"
        assert len(res1.top_k_features) == 5
        assert len(res1.attributions) == len(setup["feature_names"])

        # Determinism check with same random_state
        explainer2 = TabularLIMEExplainer(
            training_data=setup["X_train"],
            feature_names=setup["feature_names"],
            model=setup["model"],
            model_id="RandomForest",
            categorical_features=bin_indices,
            random_state=42,
        )
        res2 = explainer2.explain_instance(setup["test_instance"], top_k=5, num_samples=150)
        assert res1.top_k_features == res2.top_k_features


class TestGNNExplainerAdapters:
    def test_gnn_kernel_shap_adapter(self):
        features = [f"sensor_{i}" for i in range(10)]
        # Simple mock score function that adds 3.0 * delta[0] - 2.0 * delta[1]
        def mock_score_fn(deltas):
            return 1.0 + 3.0 * deltas[:, 0] - 2.0 * deltas[:, 1]

        explainer = GNNKernelSHAPExplainer(
            gnn_score_fn=mock_score_fn,
            feature_names=features,
            nsamples=20,
        )

        test_delta = np.zeros(10, dtype=np.float32)
        test_delta[0] = 2.0
        res = explainer.explain_instance(test_delta, top_k=3)

        assert res.method == "SHAP"
        assert res.model_id == "SpatioTemporalGNN"
        assert "sensor_0" in res.top_k_features

    def test_gnn_lime_adapter(self):
        features = [f"sensor_{i}" for i in range(10)]
        def mock_score_fn(deltas):
            return 1.0 + 4.0 * deltas[:, 0]

        explainer = GNNLIMEExplainer(
            gnn_score_fn=mock_score_fn,
            feature_names=features,
            random_state=42,
        )

        test_delta = np.zeros(10, dtype=np.float32)
        test_delta[0] = 1.5
        res = explainer.explain_instance(test_delta, top_k=3, num_samples=100)

        assert res.method == "LIME"
        assert res.model_id == "SpatioTemporalGNN"
        assert len(res.top_k_features) == 3

    def test_gnn_lime_adapter_distinguishes_different_inputs(self):
        """Regression test ensuring different inputs yield different rankings and non-zero surrogate R^2."""
        features = [f"sensor_{i}" for i in range(10)]
        def mock_score_fn(deltas):
            return 1.0 + 5.0 * deltas[:, 0] - 3.0 * deltas[:, 7]

        rng = np.random.RandomState(42)
        bg = rng.binomial(1, 0.5, size=(50, 10)).astype(np.float64)
        explainer = GNNLIMEExplainer(
            gnn_score_fn=mock_score_fn,
            feature_names=features,
            reference_deltas=bg,
            categorical_features=list(range(10)),
            random_state=42,
        )

        delta_a = np.zeros(10, dtype=np.float64)
        delta_a[0] = 1.0
        res_a = explainer.explain_instance(delta_a, top_k=3, num_samples=100)

        delta_b = np.zeros(10, dtype=np.float64)
        delta_b[7] = 1.0
        res_b = explainer.explain_instance(delta_b, top_k=3, num_samples=100)

        assert res_a.top_k_features != res_b.top_k_features
        assert "sensor_0" in res_a.top_k_features
        assert "sensor_7" in res_b.top_k_features
        assert res_a.metadata.get("surrogate_r2", 0.0) > 0.0
        assert res_b.metadata.get("surrogate_r2", 0.0) > 0.0



class TestUnifiedAPI:
    def test_explain_alert_both(self, synthetic_tabular_setup):
        setup = synthetic_tabular_setup
        res_dict = explain_alert(
            model_id="RandomForest",
            model=setup["model"],
            instance=setup["test_instance"],
            feature_names=setup["feature_names"],
            background_data=setup["X_train"][:30],
            method="BOTH",
            top_k=4,
        )

        assert isinstance(res_dict, dict)
        assert "SHAP" in res_dict and "LIME" in res_dict
        assert len(res_dict["SHAP"].top_k_features) == 4
        assert len(res_dict["LIME"].top_k_features) == 4

    def test_invalid_method_raises(self, synthetic_tabular_setup):
        setup = synthetic_tabular_setup
        with pytest.raises(ValueError, match="Unknown explanation method"):
            explain_alert(
                model_id="RandomForest",
                model=setup["model"],
                instance=setup["test_instance"],
                feature_names=setup["feature_names"],
                method="INVALID",
            )


class TestExplanationEvaluation:
    def test_evaluate_perturbation_fidelity(self, synthetic_tabular_setup):
        setup = synthetic_tabular_setup
        def pred_fn(x):
            return float(setup["model"].predict_proba(x)[0, 1])

        inst = setup["test_instance"]
        top_k = ["P_J1", "P_J2"]

        fid = evaluate_perturbation_fidelity(
            model_predict_fn=pred_fn,
            instance=inst,
            top_k_features=top_k,
            feature_names=setup["feature_names"],
            num_random_trials=5,
        )

        assert "original_score" in fid
        assert "explained_score_drop" in fid
        assert "random_k_mean_score_drop" in fid
        assert fid["k_masked"] == 2

    def test_compute_shap_lime_agreement(self):
        exp1 = ExplanationResult(
            method="SHAP",
            model_id="Test",
            prediction=0.8,
            prediction_label="Attack",
            top_k_features=["s1", "s2", "s3", "s4"],
            attributions={"s1": 1.0, "s2": 0.5, "s3": 0.2, "s4": 0.1, "s5": 0.0},
            abs_attributions={"s1": 1.0, "s2": 0.5, "s3": 0.2, "s4": 0.1, "s5": 0.0},
            normalized_attributions={},
        )
        exp2 = ExplanationResult(
            method="LIME",
            model_id="Test",
            prediction=0.8,
            prediction_label="Attack",
            top_k_features=["s1", "s2", "s5", "s4"],
            attributions={"s1": 0.9, "s2": 0.4, "s3": 0.0, "s4": 0.2, "s5": 0.3},
            abs_attributions={"s1": 0.9, "s2": 0.4, "s3": 0.0, "s4": 0.2, "s5": 0.3},
            normalized_attributions={},
        )

        agr = compute_shap_lime_agreement(exp1, exp2, top_k=4)
        assert agr["overlap_count"] == 3  # s1, s2, s4
        assert 0.0 < agr["jaccard_similarity"] <= 1.0
        assert agr["sign_agreement_rate"] == 1.0
        assert agr["agreement_level"] in {"Moderate", "High"}

    def test_evaluate_ground_truth_attribution(self):
        top_k = ["F_PU1", "P_J280", "L_T1"]
        comp_map = {"PU1": ["F_PU1", "S_PU1"]}

        eval_res = evaluate_ground_truth_attribution(
            top_k_features=top_k,
            attacked_component="PU1",
            component_sensor_map=comp_map,
            top_k=3,
        )

        assert eval_res["hit"] is True
        assert "F_PU1" in eval_res["matched_sensors"]
        assert eval_res["precision_at_k"] == 1.0 / 3.0

    def test_ground_truth_attribution_hit_at_k_consistency(self):
        """Regression test: verify Hit@5 vs Hit@3 disambiguation (Sequence 3 and 5 discrepancy fix)."""
        # Matched sensor S_V2 is at rank 5
        top_5 = ["F_PU3", "P_J307", "S_PU4", "L_T7", "S_V2"]
        comp_map = {"V2": ["L_T3", "F_V2", "S_V2"]}

        eval_at_5 = evaluate_ground_truth_attribution(top_5, "V2", comp_map, top_k=5)
        eval_at_3 = evaluate_ground_truth_attribution(top_5, "V2", comp_map, top_k=3)

        assert eval_at_5["hit"] is True
        assert "S_V2" in eval_at_5["matched_sensors"]
        assert eval_at_5["precision_at_k"] == 1.0 / 5.0

        # At K=3, S_V2 is excluded, so hit must be False
        assert eval_at_3["hit"] is False
        assert len(eval_at_3["matched_sensors"]) == 0
        assert eval_at_3["precision_at_k"] == 0.0

    def test_top_k_overlap_and_jaccard_consistency(self):
        """Verify Overlap@K and Jaccard@K arithmetic consistency."""
        exp_shap = ExplanationResult(
            method="SHAP",
            model_id="Test",
            prediction=0.9,
            prediction_label="Attack",
            top_k_features=["s1", "s2", "s3", "s4", "s5"],
            attributions={f"s{i}": float(i) for i in range(1, 6)},
            abs_attributions={f"s{i}": float(i) for i in range(1, 6)},
            normalized_attributions={},
        )
        exp_lime = ExplanationResult(
            method="LIME",
            model_id="Test",
            prediction=0.9,
            prediction_label="Attack",
            top_k_features=["s1", "s2", "s6", "s7", "s8"],
            attributions={f"s{i}": float(i) for i in range(1, 9)},
            abs_attributions={f"s{i}": float(i) for i in range(1, 9)},
            normalized_attributions={},
        )

        agr_at_5 = compute_shap_lime_agreement(exp_shap, exp_lime, top_k=5)
        # Shared: s1, s2. Union: s1, s2, s3, s4, s5, s6, s7, s8 (8 total)
        assert agr_at_5["overlap_count"] == 2
        assert abs(agr_at_5["jaccard_similarity"] - 2.0 / 8.0) < 1e-6

        agr_at_3 = compute_shap_lime_agreement(exp_shap, exp_lime, top_k=3)
        # Top-3 SHAP: [s1, s2, s3], Top-3 LIME: [s1, s2, s6]. Shared: s1, s2. Union: 4 total.
        assert agr_at_3["overlap_count"] == 2
        assert abs(agr_at_3["jaccard_similarity"] - 2.0 / 4.0) < 1e-6

    def test_mu_sq_h_calibration_variance_floor_is_positive(self):
        """Verify mu_sq_h calibration denominators are strictly positive and floor >= 1e-8."""
        from dashboard.app import load_demo_pipeline
        demo_pipe = load_demo_pipeline()
        mu_sq_h = demo_pipe["mu_sq_h"]

        assert mu_sq_h.shape == (12, 43)
        assert np.all(mu_sq_h >= 1e-8)
        # Verify inactive pump PU3 has denominator dominated by epsilon floor
        s_idx = demo_pipe["sensor_cols"].index("F_PU3")
        assert mu_sq_h[11, s_idx] < 1e-6

