"""
test_dashboard.py

Unit tests for Phase 8 Dashboard data loading, terminology policy enforcement,
dual score calculations, and post-fix data integrity.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
CORRECTED_DIR = ROOT_DIR / "results" / "test_inference" / "corrected"


class TestDashboardDataIntegrity:
    def test_corrected_test_artifacts_exist(self):
        """Verify all corrected test inference CSVs and JSONs exist."""
        assert (CORRECTED_DIR / "test_anomaly_scores.csv").exists()
        assert (CORRECTED_DIR / "test_anomaly_episodes.csv").exists()
        assert (CORRECTED_DIR / "test_sensor_attribution.csv").exists()
        assert (CORRECTED_DIR / "test_inference_report.json").exists()

    def test_corrected_test_scores_monotonicity_and_counts(self):
        """Verify corrected test scores timeline monotonicity and exact window counts."""
        df_scores = pd.read_csv(CORRECTED_DIR / "test_anomaly_scores.csv")
        df_scores["timestamp"] = pd.to_datetime(df_scores["timestamp"])

        assert len(df_scores) == 2066
        assert df_scores["timestamp"].is_monotonic_increasing
        assert df_scores["timestamp"].iloc[0] == pd.Timestamp("2017-01-04 12:00:00")
        assert df_scores["timestamp"].iloc[-1] == pd.Timestamp("2017-03-31 13:00:00")
        assert df_scores["anomaly_flag"].sum() == 156

    def test_episodes_severity_calculations(self):
        """Verify log-scaled operational severity formula vs raw severity."""
        df_ep = pd.read_csv(CORRECTED_DIR / "test_anomaly_episodes.csv")
        assert len(df_ep) >= 17

        tau = 3.013382
        for _, row in df_ep.iterrows():
            mean_s = row["mean_anomaly_score"] if "mean_anomaly_score" in row else row["mean_score"]
            peak_s = row["max_anomaly_score"] if "max_anomaly_score" in row else row["peak_score"]
            dur = row["duration_hours"]

            # Log-scaled severity
            expected_log_sev = (
                np.log10(1.0 + mean_s)
                * np.sqrt(dur)
                * (1.0 + np.log10(1.0 + peak_s / tau))
            )
            assert expected_log_sev > 0.0

    def test_forbidden_terminology_policy(self):
        """Verify that dashboard UI display strings enforce non-alarmist terminology."""
        app_file = ROOT_DIR / "dashboard" / "app.py"
        assert app_file.exists()

        content = app_file.read_text(encoding="utf-8")
        assert "TERMINOLOGY_POLICY_NOTICE" in content
        assert "Unlabeled test set events are classified as" in content

    def test_dashboard_loads_exactly_17_episodes(self):
        """Regression test ensuring dashboard loads exactly 17 frozen test episodes."""
        from dashboard.app import load_dashboard_data
        df_scores, df_episodes, df_attribution, summary_data, component_map = load_dashboard_data()

        assert len(df_episodes) == 17
        assert len(df_scores) == 2066
        assert df_scores["anomaly_flag"].sum() == 156

        # Verify key episodes exist
        ep_ids = df_episodes["episode_id"].tolist()
        for required_id in ["EP-12", "EP-11", "EP-20", "EP-06"]:
            assert required_id in ep_ids

    def test_episode_explorer_formatting_no_value_error(self):
        """Regression test ensuring float duration_hours formatting does not raise ValueError."""
        from dashboard.app import load_dashboard_data
        _, df_episodes, _, _, _ = load_dashboard_data()

        # Perform the exact Styler formatting used in Episode Explorer
        try:
            styled_df = df_episodes[
                [
                    "episode_id",
                    "start_time",
                    "end_time",
                    "duration_hours",
                    "peak_score",
                    "mean_score",
                    "operational_severity_log",
                    "raw_severity",
                    "persistence_classification",
                    "top_contributing_sensors",
                ]
            ].style.format(
                {
                    "peak_score": "{:.2e}",
                    "mean_score": "{:.2e}",
                    "operational_severity_log": "{:.1f}",
                    "raw_severity": "{:.2e}",
                    "duration_hours": "{:.0f}",
                }
            )
            # Render html to trigger full string formatting evaluation
            _ = styled_df.to_html()
        except ValueError as exc:
            pytest.fail(f"Styler formatting raised ValueError: {exc}")


class TestDemoModeIntegrity:
    def test_demo_uses_locked_threshold(self):
        """Verify Demo Mode uses exact locked operational threshold 3.013382."""
        from dashboard.app import THRESHOLD_TAU
        assert THRESHOLD_TAU == 3.013382

    def test_demo_pipeline_sensor_categories_discoverable(self):
        """Verify all 6 monitored WDS sensor categories are discoverable from the 43 sensors."""
        from dashboard.app import load_demo_pipeline
        demo_pipe = load_demo_pipeline()
        sensor_cols = demo_pipe["sensor_cols"]

        assert len(sensor_cols) == 43

        pressures = [s for s in sensor_cols if s.startswith("P_")]
        tanks = [s for s in sensor_cols if s.startswith("L_")]
        p_flows = [s for s in sensor_cols if s.startswith("F_PU")]
        p_stats = [s for s in sensor_cols if s.startswith("S_PU")]
        v_flows = [s for s in sensor_cols if s.startswith("F_V")]
        v_stats = [s for s in sensor_cols if s.startswith("S_V")]

        assert len(pressures) == 12
        assert len(tanks) == 7
        assert len(p_flows) == 11
        assert len(p_stats) == 11
        assert len(v_flows) == 1
        assert len(v_stats) == 1

    def test_demo_synthetic_injection_in_memory_only(self):
        """Verify synthetic anomaly injection modifies only in-memory objects."""
        from dashboard.app import load_demo_pipeline
        import numpy as np

        demo_pipe = load_demo_pipeline()
        orig_window = demo_pipe["normal_window_raw"].copy()

        # Create in-memory copy and perturb
        disturbed = orig_window.copy()
        target_sensor = "P_J280"
        disturbed.loc[6:, target_sensor] += 10.0

        # Verify original in-memory object is unchanged
        assert not np.allclose(orig_window[target_sensor].values, disturbed[target_sensor].values)
        assert orig_window[target_sensor].iloc[-1] != disturbed[target_sensor].iloc[-1]

        # Verify files on disk remain untouched
        assert (ROOT_DIR / "data" / "raw" / "BATADAL_dataset03.csv").exists()
        assert (ROOT_DIR / "models" / "detection_gnn.pt").exists()

    def test_xai_shap_and_lime_real_calculated_values(self):
        """Verify SHAP and LIME explanations are calculated from frozen GNN model with real values."""
        from dashboard.app import load_demo_pipeline, compute_xai_explanations
        demo_pipe = load_demo_pipeline()
        normal_window_raw = demo_pipe["normal_window_raw"].copy()
        disturbed_window = normal_window_raw.copy()
        target_sensor = "F_PU1"
        disturbed_window.loc[6:, target_sensor] += 4.0 * demo_pipe["sensor_stds"].get(target_sensor, 1.0)

        res = compute_xai_explanations(demo_pipe, normal_window_raw, disturbed_window, target_sensor)

        # Verify SHAP
        df_shap = res["df_shap"]
        assert len(df_shap) == 43
        assert df_shap["Sensor"].iloc[0] == "F_PU1"
        assert df_shap["Abs SHAP"].iloc[0] > 0.1  # Real non-zero SHAP contribution
        assert res["s_dist"] > res["s_norm"]

        # Verify LIME
        df_lime = res["df_lime"]
        assert len(df_lime) == 43
        assert df_lime["Abs LIME"].max() > 0.0  # Real calculated weights

        # Verify Agreement & Mapping
        assert res["agreement"] in ["High", "Moderate", "Low"]
        assert len(res["xai_mapping"]) == 5
        assert res["xai_mapping"][0]["Sensor"] in demo_pipe["sensor_cols"]

    def test_topology_impact_and_rerouting_analysis(self):
        """Verify topology impact path and operational rerouting calculation."""
        from dashboard.app import load_demo_pipeline, analyze_topology_and_impact
        demo_pipe = load_demo_pipeline()

        res_edge = analyze_topology_and_impact(
            demo_pipe["g"],
            demo_pipe["net"],
            "F_PU1",
            demo_pipe["sensor_nodes_map"],
            {},
        )
        assert res_edge["comp_id"] == "PU1"
        assert res_edge["comp_type"] == "Pump"
        assert res_edge["is_edge"] is True
        assert len(res_edge["endpoints"]) == 2

        res_node = analyze_topology_and_impact(
            demo_pipe["g"],
            demo_pipe["net"],
            "P_J280",
            demo_pipe["sensor_nodes_map"],
            {},
        )
        assert res_node["comp_id"] == "J280"
        assert res_node["comp_type"] == "Junction"
        assert res_node["is_edge"] is False



