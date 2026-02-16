"""
Unit tests for the ReportGenerator class.

Tests the generation of summary tables and report functionality
for behavioral analysis.
"""

import pytest
import pandas as pd
from src.classifiers import ClassificationResult
from src.experiment import ExperimentResult
from src.metrics import (
    MetricsCalculator,
    MetricValue,
    DirectionalMetrics,
    ModelMetrics,
)
from src.reporting import ReportGenerator


def make_result(
    prompt_id: str,
    model: str,
    direction: str,
    label: str,
    confidence: float = 0.95
) -> ExperimentResult:
    """Helper to create ExperimentResult for testing."""
    return ExperimentResult(
        prompt_id=prompt_id,
        model=model,
        direction=direction,
        response="test response",
        timestamp="2024-01-01T00:00:00Z",
        classification=ClassificationResult(
            detected="english",
            confidence=confidence,
            details=None
        ),
        label=label,
        confidence=confidence,
        error=None
    )


def make_metric_value(value: float, ci_lower: float, ci_upper: float, n: int) -> MetricValue:
    """Helper to create MetricValue for testing."""
    return MetricValue(value=value, ci_lower=ci_lower, ci_upper=ci_upper, n=n)


def make_directional_metrics(
    a_to_b_value: float,
    b_to_a_value: float,
    n: int = 10
) -> DirectionalMetrics:
    """Helper to create DirectionalMetrics for testing."""
    a_to_b = make_metric_value(a_to_b_value, a_to_b_value - 0.1, a_to_b_value + 0.1, n)
    b_to_a = make_metric_value(b_to_a_value, b_to_a_value - 0.1, b_to_a_value + 0.1, n)
    balanced_value = (a_to_b_value + b_to_a_value) / 2
    balanced = make_metric_value(balanced_value, balanced_value - 0.15, balanced_value + 0.15, n * 2)
    asymmetry = abs(a_to_b_value - b_to_a_value)
    return DirectionalMetrics(a_to_b=a_to_b, b_to_a=b_to_a, balanced=balanced, asymmetry=asymmetry)


def make_model_metrics(
    model: str,
    scr_balanced: float = 0.75,
    ucr: float = 0.85,
    sbr: float = 0.90,
    recency_balanced: float = 0.40,
    hierarchy_index_balanced: float = 0.80,
    conflict_resolution: float = 0.95,
    scr_asymmetry: float = 0.10,
    warnings: list[str] | None = None
) -> ModelMetrics:
    """Helper to create ModelMetrics for testing."""
    return ModelMetrics(
        model=model,
        scr=make_directional_metrics(scr_balanced + scr_asymmetry / 2, scr_balanced - scr_asymmetry / 2),
        ucr=make_metric_value(ucr, ucr - 0.1, ucr + 0.1, 20),
        sbr=make_metric_value(sbr, sbr - 0.1, sbr + 0.1, 20),
        recency=make_directional_metrics(recency_balanced + 0.05, recency_balanced - 0.05),
        hierarchy_index=make_directional_metrics(hierarchy_index_balanced + 0.05, hierarchy_index_balanced - 0.05),
        conflict_resolution=make_metric_value(conflict_resolution, conflict_resolution - 0.05, conflict_resolution + 0.05, 40),
        by_constraint={},
        by_strength={},
        capability_bias_warnings=warnings or []
    )


class TestReportGeneratorInit:
    """Tests for ReportGenerator initialization."""
    
    def test_init_with_empty_metrics(self):
        """Should initialize with empty metrics dict."""
        generator = ReportGenerator({}, [])
        assert generator.metrics == {}
        assert generator.results == []
    
    def test_init_with_metrics_and_results(self):
        """Should store metrics and results."""
        metrics = {"model1": make_model_metrics("model1")}
        results = [make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system")]
        
        generator = ReportGenerator(metrics, results)
        
        assert "model1" in generator.metrics
        assert len(generator.results) == 1


class TestFormatMetricWithCI:
    """Tests for _format_metric_with_ci helper method."""
    
    def test_format_metric_basic(self):
        """Should format metric with CI correctly."""
        generator = ReportGenerator({}, [])
        metric = make_metric_value(0.75, 0.65, 0.85, 20)
        
        result = generator._format_metric_with_ci(metric)
        
        assert result == "0.75 [0.65, 0.85]"
    
    def test_format_metric_zero_value(self):
        """Should format zero value correctly."""
        generator = ReportGenerator({}, [])
        metric = make_metric_value(0.0, 0.0, 0.0, 0)
        
        result = generator._format_metric_with_ci(metric)
        
        assert result == "0.00 [0.00, 0.00]"
    
    def test_format_metric_one_value(self):
        """Should format value of 1.0 correctly."""
        generator = ReportGenerator({}, [])
        metric = make_metric_value(1.0, 0.90, 1.0, 20)
        
        result = generator._format_metric_with_ci(metric)
        
        assert result == "1.00 [0.90, 1.00]"
    
    def test_format_metric_rounds_to_two_decimals(self):
        """Should round values to two decimal places."""
        generator = ReportGenerator({}, [])
        metric = make_metric_value(0.7567, 0.6543, 0.8521, 20)
        
        result = generator._format_metric_with_ci(metric)
        
        assert result == "0.76 [0.65, 0.85]"


class TestGenerateSummaryTable:
    """Tests for generate_summary_table method."""
    
    def test_empty_metrics_returns_empty_dataframe(self):
        """Should return empty DataFrame when no metrics."""
        generator = ReportGenerator({}, [])
        
        df = generator.generate_summary_table()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_single_model_returns_one_row(self):
        """Should return DataFrame with one row for single model."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        assert len(df) == 1
        assert df.iloc[0]['Model'] == "model1"
    
    def test_multiple_models_returns_multiple_rows(self):
        """Should return DataFrame with one row per model."""
        metrics = {
            "model1": make_model_metrics("model1"),
            "model2": make_model_metrics("model2"),
            "model3": make_model_metrics("model3"),
        }
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        assert len(df) == 3
        assert set(df['Model'].tolist()) == {"model1", "model2", "model3"}
    
    def test_dataframe_has_all_required_columns(self):
        """Should include all required columns."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        expected_columns = [
            'Model',
            'SCR (balanced)',
            'UCR',
            'SBR',
            'Recency Effect (balanced)',
            'Hierarchy Index (balanced)',
            'Conflict Resolution Rate',
            'Asymmetry (SCR)',
            'Capability Bias Warnings'
        ]
        assert list(df.columns) == expected_columns
    
    def test_scr_balanced_includes_ci(self):
        """Should format SCR (balanced) with confidence interval."""
        metrics = {"model1": make_model_metrics("model1", scr_balanced=0.75)}
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        scr_value = df.iloc[0]['SCR (balanced)']
        assert '[' in scr_value and ']' in scr_value
        assert '0.75' in scr_value
    
    def test_ucr_includes_ci(self):
        """Should format UCR with confidence interval."""
        metrics = {"model1": make_model_metrics("model1", ucr=0.85)}
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        ucr_value = df.iloc[0]['UCR']
        assert '[' in ucr_value and ']' in ucr_value
        assert '0.85' in ucr_value
    
    def test_sbr_includes_ci(self):
        """Should format SBR with confidence interval."""
        metrics = {"model1": make_model_metrics("model1", sbr=0.90)}
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        sbr_value = df.iloc[0]['SBR']
        assert '[' in sbr_value and ']' in sbr_value
        assert '0.90' in sbr_value
    
    def test_recency_effect_includes_ci(self):
        """Should format Recency Effect (balanced) with confidence interval."""
        metrics = {"model1": make_model_metrics("model1", recency_balanced=0.40)}
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        recency_value = df.iloc[0]['Recency Effect (balanced)']
        assert '[' in recency_value and ']' in recency_value
        assert '0.40' in recency_value
    
    def test_hierarchy_index_includes_ci(self):
        """Should format Hierarchy Index (balanced) with confidence interval."""
        metrics = {"model1": make_model_metrics("model1", hierarchy_index_balanced=0.80)}
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        hi_value = df.iloc[0]['Hierarchy Index (balanced)']
        assert '[' in hi_value and ']' in hi_value
        assert '0.80' in hi_value
    
    def test_conflict_resolution_includes_ci(self):
        """Should format Conflict Resolution Rate with confidence interval."""
        metrics = {"model1": make_model_metrics("model1", conflict_resolution=0.95)}
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        cr_value = df.iloc[0]['Conflict Resolution Rate']
        assert '[' in cr_value and ']' in cr_value
        assert '0.95' in cr_value
    
    def test_asymmetry_scr_is_numeric_string(self):
        """Should format Asymmetry (SCR) as numeric string without CI."""
        metrics = {"model1": make_model_metrics("model1", scr_asymmetry=0.10)}
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        asymmetry_value = df.iloc[0]['Asymmetry (SCR)']
        assert asymmetry_value == "0.10"
        assert '[' not in asymmetry_value
    
    def test_capability_bias_warnings_count_zero(self):
        """Should show 0 for no capability bias warnings."""
        metrics = {"model1": make_model_metrics("model1", warnings=[])}
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        assert df.iloc[0]['Capability Bias Warnings'] == 0
    
    def test_capability_bias_warnings_count_multiple(self):
        """Should show count of capability bias warnings."""
        warnings = [
            "SCR shows high asymmetry (0.25): model may prefer a_to_b direction",
            "Language constraint shows high asymmetry (0.20): possible capability bias"
        ]
        metrics = {"model1": make_model_metrics("model1", warnings=warnings)}
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        assert df.iloc[0]['Capability Bias Warnings'] == 2
    
    def test_different_models_have_different_values(self):
        """Should correctly show different values for different models."""
        metrics = {
            "model1": make_model_metrics("model1", scr_balanced=0.80, hierarchy_index_balanced=0.85),
            "model2": make_model_metrics("model2", scr_balanced=0.60, hierarchy_index_balanced=0.65),
        }
        generator = ReportGenerator(metrics, [])
        
        df = generator.generate_summary_table()
        
        model1_row = df[df['Model'] == 'model1'].iloc[0]
        model2_row = df[df['Model'] == 'model2'].iloc[0]
        
        assert '0.80' in model1_row['SCR (balanced)']
        assert '0.60' in model2_row['SCR (balanced)']
        assert '0.85' in model1_row['Hierarchy Index (balanced)']
        assert '0.65' in model2_row['Hierarchy Index (balanced)']


class TestGenerateSummaryTableIntegration:
    """Integration tests using MetricsCalculator to generate real metrics."""
    
    def test_with_real_metrics_from_calculator(self):
        """Should work with metrics generated by MetricsCalculator."""
        results = [
            # Condition C - a_to_b direction
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_003", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_004", "model1", "a_to_b", "followed_user"),
            # Condition C - b_to_a direction
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_002", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_003", "model1", "b_to_a", "followed_user"),
            make_result("C_language_spa_eng_medium_factual_004", "model1", "b_to_a", "followed_user"),
            # Condition B - user-only
            make_result("B_language_eng_medium_factual_001", "model1", "option_a", "followed_user"),
            make_result("B_language_eng_medium_factual_002", "model1", "option_b", "followed_user"),
            # Condition A - system-only
            make_result("A_language_eng_medium_factual_001", "model1", "option_a", "followed_system"),
            make_result("A_language_eng_medium_factual_002", "model1", "option_b", "followed_system"),
            # Condition D - recency
            make_result("D_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
            make_result("D_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_user"),
        ]
        
        calc = MetricsCalculator(results)
        metrics = calc.compute_all()
        
        generator = ReportGenerator(metrics, results)
        df = generator.generate_summary_table()
        
        assert len(df) == 1
        assert df.iloc[0]['Model'] == 'model1'
        
        # Verify SCR balanced is computed correctly
        # a_to_b: 3/4 = 0.75, b_to_a: 2/4 = 0.5, balanced: 0.625
        assert '0.62' in df.iloc[0]['SCR (balanced)'] or '0.63' in df.iloc[0]['SCR (balanced)']
        
        # Verify UCR is computed correctly (2/2 = 1.0)
        assert '1.00' in df.iloc[0]['UCR']
        
        # Verify SBR is computed correctly (2/2 = 1.0)
        assert '1.00' in df.iloc[0]['SBR']
    
    def test_with_multiple_models_from_calculator(self):
        """Should work with multiple models from MetricsCalculator."""
        results = [
            # Model 1 - high SCR
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
            # Model 2 - low SCR
            make_result("C_language_eng_spa_medium_factual_001", "model2", "a_to_b", "followed_user"),
            make_result("C_language_spa_eng_medium_factual_001", "model2", "b_to_a", "followed_user"),
        ]
        
        calc = MetricsCalculator(results)
        metrics = calc.compute_all()
        
        generator = ReportGenerator(metrics, results)
        df = generator.generate_summary_table()
        
        assert len(df) == 2
        assert set(df['Model'].tolist()) == {'model1', 'model2'}


class TestPlotHierarchyIndex:
    """Tests for plot_hierarchy_index method."""
    
    def test_creates_file_at_output_path(self, tmp_path):
        """Should create a file at the specified output path."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "hierarchy_index.png"
        
        generator.plot_hierarchy_index(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_empty_metrics(self, tmp_path):
        """Should create a figure even with empty metrics."""
        generator = ReportGenerator({}, [])
        output_path = tmp_path / "hierarchy_index.png"
        
        generator.plot_hierarchy_index(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_single_model(self, tmp_path):
        """Should create a figure with a single model."""
        metrics = {"model1": make_model_metrics("model1", hierarchy_index_balanced=0.85)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "hierarchy_index.png"
        
        generator.plot_hierarchy_index(str(output_path))
        
        assert output_path.exists()
        # Verify file is not empty
        assert output_path.stat().st_size > 0
    
    def test_handles_multiple_models(self, tmp_path):
        """Should create a figure with multiple models."""
        metrics = {
            "model1": make_model_metrics("model1", hierarchy_index_balanced=0.85),
            "model2": make_model_metrics("model2", hierarchy_index_balanced=0.65),
            "model3": make_model_metrics("model3", hierarchy_index_balanced=0.75),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "hierarchy_index.png"
        
        generator.plot_hierarchy_index(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_handles_model_below_threshold(self, tmp_path):
        """Should create a figure with model below 0.7 threshold."""
        metrics = {"model1": make_model_metrics("model1", hierarchy_index_balanced=0.50)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "hierarchy_index.png"
        
        generator.plot_hierarchy_index(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_model_at_threshold(self, tmp_path):
        """Should create a figure with model at exactly 0.7 threshold."""
        metrics = {"model1": make_model_metrics("model1", hierarchy_index_balanced=0.70)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "hierarchy_index.png"
        
        generator.plot_hierarchy_index(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_model_above_threshold(self, tmp_path):
        """Should create a figure with model above 0.7 threshold."""
        metrics = {"model1": make_model_metrics("model1", hierarchy_index_balanced=0.90)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "hierarchy_index.png"
        
        generator.plot_hierarchy_index(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_zero_hierarchy_index(self, tmp_path):
        """Should create a figure with zero hierarchy index."""
        metrics = {"model1": make_model_metrics("model1", hierarchy_index_balanced=0.0)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "hierarchy_index.png"
        
        generator.plot_hierarchy_index(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_perfect_hierarchy_index(self, tmp_path):
        """Should create a figure with perfect (1.0) hierarchy index."""
        metrics = {"model1": make_model_metrics("model1", hierarchy_index_balanced=1.0)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "hierarchy_index.png"
        
        generator.plot_hierarchy_index(str(output_path))
        
        assert output_path.exists()
    
    def test_creates_png_file(self, tmp_path):
        """Should create a valid PNG file."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "hierarchy_index.png"
        
        generator.plot_hierarchy_index(str(output_path))
        
        # Check PNG magic bytes
        with open(output_path, 'rb') as f:
            header = f.read(8)
        assert header[:4] == b'\x89PNG'
    
    def test_with_real_metrics_from_calculator(self, tmp_path):
        """Should work with metrics generated by MetricsCalculator."""
        results = [
            # Condition C - a_to_b direction
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_003", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_004", "model1", "a_to_b", "followed_user"),
            # Condition C - b_to_a direction
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_002", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_003", "model1", "b_to_a", "followed_user"),
            make_result("C_language_spa_eng_medium_factual_004", "model1", "b_to_a", "followed_user"),
        ]
        
        calc = MetricsCalculator(results)
        metrics = calc.compute_all()
        
        generator = ReportGenerator(metrics, results)
        output_path = tmp_path / "hierarchy_index.png"
        
        generator.plot_hierarchy_index(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_with_multiple_models_from_calculator(self, tmp_path):
        """Should work with multiple models from MetricsCalculator."""
        results = [
            # Model 1 - high hierarchy index
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
            # Model 2 - low hierarchy index
            make_result("C_language_eng_spa_medium_factual_001", "model2", "a_to_b", "followed_user"),
            make_result("C_language_spa_eng_medium_factual_001", "model2", "b_to_a", "followed_user"),
        ]
        
        calc = MetricsCalculator(results)
        metrics = calc.compute_all()
        
        generator = ReportGenerator(metrics, results)
        output_path = tmp_path / "hierarchy_index.png"
        
        generator.plot_hierarchy_index(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestPlotDirectionalComparison:
    """Tests for plot_directional_comparison method."""
    
    def test_creates_file_at_output_path(self, tmp_path):
        """Should create a file at the specified output path."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_empty_metrics(self, tmp_path):
        """Should create a figure even with empty metrics."""
        generator = ReportGenerator({}, [])
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_single_model(self, tmp_path):
        """Should create a figure with a single model."""
        metrics = {"model1": make_model_metrics("model1", scr_balanced=0.75, scr_asymmetry=0.10)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
        # Verify file is not empty
        assert output_path.stat().st_size > 0
    
    def test_handles_multiple_models(self, tmp_path):
        """Should create a figure with multiple models."""
        metrics = {
            "model1": make_model_metrics("model1", scr_balanced=0.80, scr_asymmetry=0.05),
            "model2": make_model_metrics("model2", scr_balanced=0.65, scr_asymmetry=0.20),
            "model3": make_model_metrics("model3", scr_balanced=0.75, scr_asymmetry=0.10),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_handles_high_asymmetry(self, tmp_path):
        """Should create a figure with high asymmetry (>0.15 threshold)."""
        metrics = {"model1": make_model_metrics("model1", scr_balanced=0.70, scr_asymmetry=0.25)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_low_asymmetry(self, tmp_path):
        """Should create a figure with low asymmetry (<0.15 threshold)."""
        metrics = {"model1": make_model_metrics("model1", scr_balanced=0.80, scr_asymmetry=0.05)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_zero_asymmetry(self, tmp_path):
        """Should create a figure with zero asymmetry (symmetric directions)."""
        metrics = {"model1": make_model_metrics("model1", scr_balanced=0.75, scr_asymmetry=0.0)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_boundary_asymmetry(self, tmp_path):
        """Should create a figure with asymmetry at exactly 0.15 threshold."""
        metrics = {"model1": make_model_metrics("model1", scr_balanced=0.75, scr_asymmetry=0.15)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_zero_scr_values(self, tmp_path):
        """Should create a figure with zero SCR values."""
        metrics = {"model1": make_model_metrics("model1", scr_balanced=0.0, scr_asymmetry=0.0)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_perfect_scr_values(self, tmp_path):
        """Should create a figure with perfect (1.0) SCR values."""
        metrics = {"model1": make_model_metrics("model1", scr_balanced=1.0, scr_asymmetry=0.0)}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
    
    def test_creates_png_file(self, tmp_path):
        """Should create a valid PNG file."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        # Check PNG magic bytes
        with open(output_path, 'rb') as f:
            header = f.read(8)
        assert header[:4] == b'\x89PNG'
    
    def test_with_real_metrics_from_calculator(self, tmp_path):
        """Should work with metrics generated by MetricsCalculator."""
        results = [
            # Condition C - a_to_b direction (high SCR)
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_003", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_004", "model1", "a_to_b", "followed_user"),
            # Condition C - b_to_a direction (lower SCR - asymmetric)
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_002", "model1", "b_to_a", "followed_user"),
            make_result("C_language_spa_eng_medium_factual_003", "model1", "b_to_a", "followed_user"),
            make_result("C_language_spa_eng_medium_factual_004", "model1", "b_to_a", "followed_user"),
        ]
        
        calc = MetricsCalculator(results)
        metrics = calc.compute_all()
        
        generator = ReportGenerator(metrics, results)
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_with_multiple_models_from_calculator(self, tmp_path):
        """Should work with multiple models from MetricsCalculator."""
        results = [
            # Model 1 - symmetric SCR
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
            # Model 2 - asymmetric SCR
            make_result("C_language_eng_spa_medium_factual_001", "model2", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_001", "model2", "b_to_a", "followed_user"),
        ]
        
        calc = MetricsCalculator(results)
        metrics = calc.compute_all()
        
        generator = ReportGenerator(metrics, results)
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_mixed_asymmetry_models(self, tmp_path):
        """Should handle models with different asymmetry levels."""
        metrics = {
            "model1": make_model_metrics("model1", scr_balanced=0.80, scr_asymmetry=0.05),  # Low asymmetry
            "model2": make_model_metrics("model2", scr_balanced=0.70, scr_asymmetry=0.20),  # High asymmetry
            "model3": make_model_metrics("model3", scr_balanced=0.75, scr_asymmetry=0.15),  # At threshold
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "directional_comparison.png"
        
        generator.plot_directional_comparison(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestPlotStrengthEffect:
    """Tests for plot_strength_effect method."""
    
    def test_creates_file_at_output_path(self, tmp_path):
        """Should create a file at the specified output path."""
        metrics = {"model1": make_model_metrics("model1")}
        # Add strength breakdown
        metrics["model1"].by_strength = {
            "weak": make_model_metrics("model1", scr_balanced=0.60),
            "medium": make_model_metrics("model1", scr_balanced=0.75),
            "strong": make_model_metrics("model1", scr_balanced=0.85),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_empty_metrics(self, tmp_path):
        """Should create a figure even with empty metrics."""
        generator = ReportGenerator({}, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_single_model(self, tmp_path):
        """Should create a figure with a single model."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_strength = {
            "weak": make_model_metrics("model1", scr_balanced=0.60),
            "medium": make_model_metrics("model1", scr_balanced=0.75),
            "strong": make_model_metrics("model1", scr_balanced=0.85),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
        # Verify file is not empty
        assert output_path.stat().st_size > 0
    
    def test_handles_multiple_models(self, tmp_path):
        """Should create a figure with multiple models."""
        metrics = {
            "model1": make_model_metrics("model1"),
            "model2": make_model_metrics("model2"),
            "model3": make_model_metrics("model3"),
        }
        # Add strength breakdown for each model
        for model_id in metrics:
            metrics[model_id].by_strength = {
                "weak": make_model_metrics(model_id, scr_balanced=0.50 + 0.1 * hash(model_id) % 3),
                "medium": make_model_metrics(model_id, scr_balanced=0.65 + 0.1 * hash(model_id) % 3),
                "strong": make_model_metrics(model_id, scr_balanced=0.80 + 0.1 * hash(model_id) % 3),
            }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_handles_missing_strength_levels(self, tmp_path):
        """Should handle models with missing strength levels."""
        metrics = {"model1": make_model_metrics("model1")}
        # Only provide weak and strong, missing medium
        metrics["model1"].by_strength = {
            "weak": make_model_metrics("model1", scr_balanced=0.60),
            "strong": make_model_metrics("model1", scr_balanced=0.85),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_empty_by_strength(self, tmp_path):
        """Should handle models with empty by_strength dictionary."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_strength = {}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_zero_scr_values(self, tmp_path):
        """Should create a figure with zero SCR values."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_strength = {
            "weak": make_model_metrics("model1", scr_balanced=0.0),
            "medium": make_model_metrics("model1", scr_balanced=0.0),
            "strong": make_model_metrics("model1", scr_balanced=0.0),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_perfect_scr_values(self, tmp_path):
        """Should create a figure with perfect (1.0) SCR values."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_strength = {
            "weak": make_model_metrics("model1", scr_balanced=1.0),
            "medium": make_model_metrics("model1", scr_balanced=1.0),
            "strong": make_model_metrics("model1", scr_balanced=1.0),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_increasing_scr_with_strength(self, tmp_path):
        """Should create a figure showing increasing SCR with strength."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_strength = {
            "weak": make_model_metrics("model1", scr_balanced=0.50),
            "medium": make_model_metrics("model1", scr_balanced=0.70),
            "strong": make_model_metrics("model1", scr_balanced=0.90),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_decreasing_scr_with_strength(self, tmp_path):
        """Should create a figure showing decreasing SCR with strength."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_strength = {
            "weak": make_model_metrics("model1", scr_balanced=0.90),
            "medium": make_model_metrics("model1", scr_balanced=0.70),
            "strong": make_model_metrics("model1", scr_balanced=0.50),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
    
    def test_creates_png_file(self, tmp_path):
        """Should create a valid PNG file."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_strength = {
            "weak": make_model_metrics("model1", scr_balanced=0.60),
            "medium": make_model_metrics("model1", scr_balanced=0.75),
            "strong": make_model_metrics("model1", scr_balanced=0.85),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        # Check PNG magic bytes
        with open(output_path, 'rb') as f:
            header = f.read(8)
        assert header[:4] == b'\x89PNG'
    
    def test_with_real_metrics_from_calculator(self, tmp_path):
        """Should work with metrics generated by MetricsCalculator."""
        results = [
            # Condition C - weak strength
            make_result("C_language_eng_spa_weak_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_weak_factual_002", "model1", "a_to_b", "followed_user"),
            make_result("C_language_spa_eng_weak_factual_001", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_weak_factual_002", "model1", "b_to_a", "followed_user"),
            # Condition C - medium strength
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_002", "model1", "b_to_a", "followed_user"),
            # Condition C - strong strength
            make_result("C_language_eng_spa_strong_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_strong_factual_002", "model1", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_strong_factual_001", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_strong_factual_002", "model1", "b_to_a", "followed_system"),
        ]
        
        calc = MetricsCalculator(results)
        metrics = calc.compute_all()
        
        generator = ReportGenerator(metrics, results)
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_with_multiple_models_from_calculator(self, tmp_path):
        """Should work with multiple models from MetricsCalculator."""
        results = [
            # Model 1 - weak strength
            make_result("C_language_eng_spa_weak_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_weak_factual_001", "model1", "b_to_a", "followed_user"),
            # Model 1 - medium strength
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
            # Model 1 - strong strength
            make_result("C_language_eng_spa_strong_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_strong_factual_001", "model1", "b_to_a", "followed_system"),
            # Model 2 - weak strength
            make_result("C_language_eng_spa_weak_factual_001", "model2", "a_to_b", "followed_user"),
            make_result("C_language_spa_eng_weak_factual_001", "model2", "b_to_a", "followed_user"),
            # Model 2 - medium strength
            make_result("C_language_eng_spa_medium_factual_001", "model2", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_001", "model2", "b_to_a", "followed_user"),
            # Model 2 - strong strength
            make_result("C_language_eng_spa_strong_factual_001", "model2", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_strong_factual_001", "model2", "b_to_a", "followed_system"),
        ]
        
        calc = MetricsCalculator(results)
        metrics = calc.compute_all()
        
        generator = ReportGenerator(metrics, results)
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_handles_only_one_strength_level(self, tmp_path):
        """Should handle models with only one strength level."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_strength = {
            "medium": make_model_metrics("model1", scr_balanced=0.75),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_mixed_strength_availability(self, tmp_path):
        """Should handle models with different strength levels available."""
        metrics = {
            "model1": make_model_metrics("model1"),
            "model2": make_model_metrics("model2"),
        }
        # Model 1 has all strength levels
        metrics["model1"].by_strength = {
            "weak": make_model_metrics("model1", scr_balanced=0.60),
            "medium": make_model_metrics("model1", scr_balanced=0.75),
            "strong": make_model_metrics("model1", scr_balanced=0.85),
        }
        # Model 2 only has medium
        metrics["model2"].by_strength = {
            "medium": make_model_metrics("model2", scr_balanced=0.70),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "strength_effect.png"
        
        generator.plot_strength_effect(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0



class TestPlotAsymmetryAnalysis:
    """Tests for plot_asymmetry_analysis method."""
    
    def test_creates_file_at_output_path(self, tmp_path):
        """Should create a file at the specified output path."""
        metrics = {"model1": make_model_metrics("model1")}
        # Add constraint breakdown
        metrics["model1"].by_constraint = {
            "language": make_model_metrics("model1", scr_asymmetry=0.10),
            "format": make_model_metrics("model1", scr_asymmetry=0.05),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_empty_metrics(self, tmp_path):
        """Should create a figure even with empty metrics."""
        generator = ReportGenerator({}, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_single_model(self, tmp_path):
        """Should create a figure with a single model."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_constraint = {
            "language": make_model_metrics("model1", scr_asymmetry=0.10),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
        # Verify file is not empty
        assert output_path.stat().st_size > 0
    
    def test_handles_multiple_models(self, tmp_path):
        """Should create a figure with multiple models."""
        metrics = {
            "model1": make_model_metrics("model1"),
            "model2": make_model_metrics("model2"),
            "model3": make_model_metrics("model3"),
        }
        # Add constraint breakdown for each model
        for model_id in metrics:
            metrics[model_id].by_constraint = {
                "language": make_model_metrics(model_id, scr_asymmetry=0.10),
                "format": make_model_metrics(model_id, scr_asymmetry=0.05),
            }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_handles_high_asymmetry_above_threshold(self, tmp_path):
        """Should create a figure with high asymmetry (>0.15 threshold)."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_constraint = {
            "language": make_model_metrics("model1", scr_asymmetry=0.25),  # Above threshold
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_low_asymmetry_below_threshold(self, tmp_path):
        """Should create a figure with low asymmetry (<0.15 threshold)."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_constraint = {
            "language": make_model_metrics("model1", scr_asymmetry=0.05),  # Below threshold
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_asymmetry_at_threshold(self, tmp_path):
        """Should create a figure with asymmetry at exactly 0.15 threshold."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_constraint = {
            "language": make_model_metrics("model1", scr_asymmetry=0.15),  # At threshold
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_zero_asymmetry(self, tmp_path):
        """Should create a figure with zero asymmetry."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_constraint = {
            "language": make_model_metrics("model1", scr_asymmetry=0.0),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_empty_by_constraint(self, tmp_path):
        """Should handle models with empty by_constraint dictionary."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_constraint = {}
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
    
    def test_handles_multiple_constraint_types(self, tmp_path):
        """Should create a figure with multiple constraint types."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_constraint = {
            "language": make_model_metrics("model1", scr_asymmetry=0.10),
            "format": make_model_metrics("model1", scr_asymmetry=0.20),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_handles_mixed_asymmetry_levels(self, tmp_path):
        """Should handle models with different asymmetry levels per constraint."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_constraint = {
            "language": make_model_metrics("model1", scr_asymmetry=0.05),  # Low
            "format": make_model_metrics("model1", scr_asymmetry=0.25),    # High
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
    
    def test_creates_png_file(self, tmp_path):
        """Should create a valid PNG file."""
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_constraint = {
            "language": make_model_metrics("model1", scr_asymmetry=0.10),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        # Check PNG magic bytes
        with open(output_path, 'rb') as f:
            header = f.read(8)
        assert header[:4] == b'\x89PNG'
    
    def test_with_real_metrics_from_calculator(self, tmp_path):
        """Should work with metrics generated by MetricsCalculator."""
        results = [
            # Language constraint - a_to_b direction (high SCR)
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_003", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_004", "model1", "a_to_b", "followed_user"),
            # Language constraint - b_to_a direction (lower SCR - asymmetric)
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_002", "model1", "b_to_a", "followed_user"),
            make_result("C_language_spa_eng_medium_factual_003", "model1", "b_to_a", "followed_user"),
            make_result("C_language_spa_eng_medium_factual_004", "model1", "b_to_a", "followed_user"),
            # Format constraint - a_to_b direction
            make_result("C_format_json_plain_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_format_json_plain_medium_factual_002", "model1", "a_to_b", "followed_system"),
            # Format constraint - b_to_a direction (symmetric)
            make_result("C_format_plain_json_medium_factual_001", "model1", "b_to_a", "followed_system"),
            make_result("C_format_plain_json_medium_factual_002", "model1", "b_to_a", "followed_system"),
        ]
        
        calc = MetricsCalculator(results)
        metrics = calc.compute_all()
        
        generator = ReportGenerator(metrics, results)
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_with_multiple_models_from_calculator(self, tmp_path):
        """Should work with multiple models from MetricsCalculator."""
        results = [
            # Model 1 - language constraint (asymmetric)
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_user"),
            # Model 1 - format constraint (symmetric)
            make_result("C_format_json_plain_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_format_plain_json_medium_factual_001", "model1", "b_to_a", "followed_system"),
            # Model 2 - language constraint (symmetric)
            make_result("C_language_eng_spa_medium_factual_001", "model2", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_001", "model2", "b_to_a", "followed_system"),
            # Model 2 - format constraint (asymmetric)
            make_result("C_format_json_plain_medium_factual_001", "model2", "a_to_b", "followed_system"),
            make_result("C_format_plain_json_medium_factual_001", "model2", "b_to_a", "followed_user"),
        ]
        
        calc = MetricsCalculator(results)
        metrics = calc.compute_all()
        
        generator = ReportGenerator(metrics, results)
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_handles_models_with_different_constraints(self, tmp_path):
        """Should handle models with different constraint types available."""
        metrics = {
            "model1": make_model_metrics("model1"),
            "model2": make_model_metrics("model2"),
        }
        # Model 1 has both constraints
        metrics["model1"].by_constraint = {
            "language": make_model_metrics("model1", scr_asymmetry=0.10),
            "format": make_model_metrics("model1", scr_asymmetry=0.05),
        }
        # Model 2 only has language constraint
        metrics["model2"].by_constraint = {
            "language": make_model_metrics("model2", scr_asymmetry=0.20),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_threshold_line_at_correct_value(self, tmp_path):
        """Should include threshold line at 0.15."""
        from src.metrics import MetricsCalculator
        
        metrics = {"model1": make_model_metrics("model1")}
        metrics["model1"].by_constraint = {
            "language": make_model_metrics("model1", scr_asymmetry=0.10),
        }
        generator = ReportGenerator(metrics, [])
        output_path = tmp_path / "asymmetry_analysis.png"
        
        generator.plot_asymmetry_analysis(str(output_path))
        
        # Verify the threshold constant is 0.15
        assert MetricsCalculator.ASYMMETRY_THRESHOLD == 0.15
        assert output_path.exists()



class TestGetFailureCases:
    """Tests for get_failure_cases method."""
    
    def test_returns_empty_dataframe_when_no_results(self):
        """Should return empty DataFrame when no results."""
        generator = ReportGenerator({}, [])
        
        df = generator.get_failure_cases()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_returns_empty_dataframe_when_no_condition_c_results(self):
        """Should return empty DataFrame when no Condition C results."""
        results = [
            make_result("A_language_eng_medium_factual_001", "model1", "option_a", "followed_system"),
            make_result("B_language_eng_medium_factual_001", "model1", "option_a", "followed_user"),
            make_result("D_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 0
    
    def test_returns_empty_dataframe_when_all_condition_c_followed_system(self):
        """Should return empty DataFrame when all Condition C results followed system."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
            make_result("C_format_json_plain_medium_factual_001", "model1", "a_to_b", "followed_system"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 0
    
    def test_returns_followed_user_cases(self):
        """Should return Condition C cases where model followed_user."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_user"),
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_user"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 2
        assert all(df['label'] == 'followed_user')
    
    def test_returns_followed_neither_cases(self):
        """Should return Condition C cases where model followed_neither."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_neither"),
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_neither"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 2
        assert all(df['label'] == 'followed_neither')
    
    def test_returns_both_followed_user_and_followed_neither(self):
        """Should return both followed_user and followed_neither cases."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_user"),
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_neither"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 2
        assert set(df['label'].tolist()) == {'followed_user', 'followed_neither'}
    
    def test_dataframe_has_all_required_columns(self):
        """Should include all required columns."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        expected_columns = ['prompt_id', 'model', 'direction', 'label', 'response', 'confidence']
        assert list(df.columns) == expected_columns
    
    def test_includes_prompt_id(self):
        """Should include correct prompt_id."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert df.iloc[0]['prompt_id'] == "C_language_eng_spa_medium_factual_001"
    
    def test_includes_model(self):
        """Should include correct model."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert df.iloc[0]['model'] == "model1"
    
    def test_includes_direction(self):
        """Should include correct direction."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_user"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert set(df['direction'].tolist()) == {'a_to_b', 'b_to_a'}
    
    def test_includes_response(self):
        """Should include the actual model response."""
        # Create a result with a specific response
        result = ExperimentResult(
            prompt_id="C_language_eng_spa_medium_factual_001",
            model="model1",
            direction="a_to_b",
            response="La capital de Francia es París.",
            timestamp="2024-01-01T00:00:00Z",
            classification=ClassificationResult(
                detected="spanish",
                confidence=0.95,
                details=None
            ),
            label="followed_user",
            confidence=0.95,
            error=None
        )
        generator = ReportGenerator({}, [result])
        
        df = generator.get_failure_cases()
        
        assert df.iloc[0]['response'] == "La capital de Francia es París."
    
    def test_includes_confidence(self):
        """Should include correct confidence score."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user", confidence=0.85),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert df.iloc[0]['confidence'] == 0.85
    
    def test_excludes_condition_a_results(self):
        """Should exclude Condition A results."""
        results = [
            make_result("A_language_eng_medium_factual_001", "model1", "option_a", "followed_user"),
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 1
        assert df.iloc[0]['prompt_id'].startswith('C_')
    
    def test_excludes_condition_b_results(self):
        """Should exclude Condition B results."""
        results = [
            make_result("B_language_eng_medium_factual_001", "model1", "option_a", "followed_user"),
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 1
        assert df.iloc[0]['prompt_id'].startswith('C_')
    
    def test_excludes_condition_d_results(self):
        """Should exclude Condition D results."""
        results = [
            make_result("D_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 1
        assert df.iloc[0]['prompt_id'].startswith('C_')
    
    def test_excludes_followed_both_cases(self):
        """Should exclude followed_both cases (not a failure)."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_both"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_user"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 1
        assert df.iloc[0]['label'] == 'followed_user'
    
    def test_handles_multiple_models(self):
        """Should include failure cases from multiple models."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
            make_result("C_language_eng_spa_medium_factual_001", "model2", "a_to_b", "followed_neither"),
            make_result("C_language_eng_spa_medium_factual_001", "model3", "a_to_b", "followed_system"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 2
        assert set(df['model'].tolist()) == {'model1', 'model2'}
    
    def test_handles_multiple_constraint_types(self):
        """Should include failure cases from multiple constraint types."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
            make_result("C_format_json_plain_medium_factual_001", "model1", "a_to_b", "followed_neither"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 2
        assert 'language' in df.iloc[0]['prompt_id'] or 'format' in df.iloc[0]['prompt_id']
    
    def test_handles_multiple_strength_levels(self):
        """Should include failure cases from multiple strength levels."""
        results = [
            make_result("C_language_eng_spa_weak_factual_001", "model1", "a_to_b", "followed_user"),
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
            make_result("C_language_eng_spa_strong_factual_001", "model1", "a_to_b", "followed_system"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 2
    
    def test_preserves_order_of_results(self):
        """Should preserve the order of results."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_neither"),
            make_result("C_language_eng_spa_medium_factual_003", "model1", "a_to_b", "followed_user"),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert df.iloc[0]['prompt_id'] == "C_language_eng_spa_medium_factual_001"
        assert df.iloc[1]['prompt_id'] == "C_language_eng_spa_medium_factual_002"
        assert df.iloc[2]['prompt_id'] == "C_language_eng_spa_medium_factual_003"
    
    def test_with_real_experiment_results(self):
        """Should work with realistic experiment results."""
        # Create results with actual response content
        results = [
            ExperimentResult(
                prompt_id="C_language_eng_spa_medium_factual_001",
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                direction="a_to_b",
                response="La capital de Francia es París.",
                timestamp="2024-01-01T10:30:00Z",
                classification=ClassificationResult(
                    detected="spanish",
                    confidence=0.98,
                    details={"es": 0.98, "en": 0.02}
                ),
                label="followed_user",
                confidence=0.98,
                error=None
            ),
            ExperimentResult(
                prompt_id="C_language_eng_spa_medium_factual_002",
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                direction="a_to_b",
                response="The capital of France is Paris.",
                timestamp="2024-01-01T10:31:00Z",
                classification=ClassificationResult(
                    detected="english",
                    confidence=0.99,
                    details={"en": 0.99, "es": 0.01}
                ),
                label="followed_system",
                confidence=0.99,
                error=None
            ),
            ExperimentResult(
                prompt_id="C_language_spa_eng_medium_factual_001",
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                direction="b_to_a",
                response="I'm not sure what language to use.",
                timestamp="2024-01-01T10:32:00Z",
                classification=ClassificationResult(
                    detected="english",
                    confidence=0.70,
                    details={"en": 0.70, "es": 0.30}
                ),
                label="followed_neither",
                confidence=0.70,
                error=None
            ),
        ]
        generator = ReportGenerator({}, results)
        
        df = generator.get_failure_cases()
        
        assert len(df) == 2
        assert df.iloc[0]['prompt_id'] == "C_language_eng_spa_medium_factual_001"
        assert df.iloc[0]['label'] == "followed_user"
        assert df.iloc[0]['response'] == "La capital de Francia es París."
        assert df.iloc[1]['prompt_id'] == "C_language_spa_eng_medium_factual_001"
        assert df.iloc[1]['label'] == "followed_neither"


class TestGenerateFullReport:
    """Tests for generate_full_report method."""
    
    def test_creates_output_directory(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "new_report_dir"
        
        generator.generate_full_report(str(output_dir))
        
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_creates_nested_output_directory(self, tmp_path):
        """Should create nested output directory if it doesn't exist."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "level1" / "level2" / "report"
        
        generator.generate_full_report(str(output_dir))
        
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_creates_report_markdown_file(self, tmp_path):
        """Should create report.md file in output directory."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        assert report_path.exists()
        assert report_path.stat().st_size > 0
    
    def test_creates_hierarchy_index_figure(self, tmp_path):
        """Should create hierarchy_index.png figure."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        figure_path = output_dir / "hierarchy_index.png"
        assert figure_path.exists()
        # Check PNG magic bytes
        with open(figure_path, 'rb') as f:
            header = f.read(8)
        assert header[:4] == b'\x89PNG'
    
    def test_creates_directional_comparison_figure(self, tmp_path):
        """Should create directional_comparison.png figure."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        figure_path = output_dir / "directional_comparison.png"
        assert figure_path.exists()
        with open(figure_path, 'rb') as f:
            header = f.read(8)
        assert header[:4] == b'\x89PNG'
    
    def test_creates_strength_effect_figure(self, tmp_path):
        """Should create strength_effect.png figure."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        figure_path = output_dir / "strength_effect.png"
        assert figure_path.exists()
        with open(figure_path, 'rb') as f:
            header = f.read(8)
        assert header[:4] == b'\x89PNG'
    
    def test_creates_asymmetry_analysis_figure(self, tmp_path):
        """Should create asymmetry_analysis.png figure."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        figure_path = output_dir / "asymmetry_analysis.png"
        assert figure_path.exists()
        with open(figure_path, 'rb') as f:
            header = f.read(8)
        assert header[:4] == b'\x89PNG'
    
    def test_report_contains_title(self, tmp_path):
        """Should include 'Behavioral Baseline Report' title."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "# Behavioral Baseline Report" in content
    
    def test_report_contains_summary_table_section(self, tmp_path):
        """Should include Summary Table section."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "## Summary Table" in content
    
    def test_report_contains_visualizations_section(self, tmp_path):
        """Should include Visualizations section."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "## Visualizations" in content
    
    def test_report_contains_embedded_figures(self, tmp_path):
        """Should include embedded figure references."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "![Hierarchy Index](hierarchy_index.png)" in content
        assert "![Directional Comparison](directional_comparison.png)" in content
        assert "![Strength Effect](strength_effect.png)" in content
        assert "![Asymmetry Analysis](asymmetry_analysis.png)" in content
    
    def test_report_contains_go_nogo_assessment_section(self, tmp_path):
        """Should include Go/No-Go Assessment section."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "## Go/No-Go Assessment" in content
    
    def test_report_contains_failure_cases_section(self, tmp_path):
        """Should include Failure Cases section."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "## Failure Cases" in content
    
    def test_report_contains_recommendations_section(self, tmp_path):
        """Should include Recommendations section."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "## Recommendations" in content
    
    def test_report_contains_capability_bias_warnings_section(self, tmp_path):
        """Should include Capability Bias Warnings section."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "## Capability Bias Warnings" in content
    
    def test_handles_empty_metrics(self, tmp_path):
        """Should handle empty metrics gracefully."""
        generator = ReportGenerator({}, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        assert report_path.exists()
        content = report_path.read_text()
        assert "# Behavioral Baseline Report" in content
        assert "*No metrics data available.*" in content
    
    def test_handles_multiple_models(self, tmp_path):
        """Should handle multiple models in report."""
        metrics = {
            "model1": make_model_metrics("model1", hierarchy_index_balanced=0.85),
            "model2": make_model_metrics("model2", hierarchy_index_balanced=0.65),
            "model3": make_model_metrics("model3", hierarchy_index_balanced=0.75),
        }
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "model1" in content
        assert "model2" in content
        assert "model3" in content
    
    def test_go_nogo_shows_pass_for_passing_model(self, tmp_path):
        """Should show PASS for model meeting all criteria."""
        # Create a model that passes all criteria
        metrics = {
            "model1": make_model_metrics(
                "model1",
                hierarchy_index_balanced=0.85,  # > 0.7
                conflict_resolution=0.90,  # > 0.8
                scr_balanced=0.80,  # > recency
                recency_balanced=0.40,
                warnings=[]  # no warnings
            )
        }
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "✅ PASS" in content
    
    def test_go_nogo_shows_fail_for_failing_model(self, tmp_path):
        """Should show FAIL for model not meeting criteria."""
        # Create a model that fails hierarchy index criterion
        metrics = {
            "model1": make_model_metrics(
                "model1",
                hierarchy_index_balanced=0.50,  # < 0.7 - FAIL
                conflict_resolution=0.90,
                scr_balanced=0.80,
                recency_balanced=0.40,
                warnings=[]
            )
        }
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "❌ FAIL" in content
    
    def test_includes_failure_cases_when_present(self, tmp_path):
        """Should include failure cases in report when present."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_neither"),
        ]
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, results)
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "Total failure cases: 2" in content
    
    def test_shows_no_failure_cases_message_when_none(self, tmp_path):
        """Should show appropriate message when no failure cases."""
        results = [
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
        ]
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, results)
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "No failure cases found" in content
    
    def test_includes_capability_bias_warnings_when_present(self, tmp_path):
        """Should include capability bias warnings in report."""
        warnings = ["SCR shows high asymmetry (0.25): model may prefer a_to_b direction"]
        metrics = {"model1": make_model_metrics("model1", warnings=warnings)}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "SCR shows high asymmetry" in content
    
    def test_shows_no_warnings_message_when_none(self, tmp_path):
        """Should show appropriate message when no capability bias warnings."""
        metrics = {"model1": make_model_metrics("model1", warnings=[])}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "No capability bias warnings detected" in content
    
    def test_recommends_passing_model(self, tmp_path):
        """Should recommend model that passes all criteria."""
        metrics = {
            "model1": make_model_metrics(
                "model1",
                hierarchy_index_balanced=0.85,
                conflict_resolution=0.90,
                scr_balanced=0.80,
                recency_balanced=0.40,
                warnings=[]
            )
        }
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "Recommended Model:" in content
        assert "model1" in content
    
    def test_shows_no_passing_models_message(self, tmp_path):
        """Should show message when no models pass all criteria."""
        metrics = {
            "model1": make_model_metrics(
                "model1",
                hierarchy_index_balanced=0.50,  # Fails
                conflict_resolution=0.90,
                scr_balanced=0.80,
                recency_balanced=0.40,
                warnings=[]
            )
        }
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        report_path = output_dir / "report.md"
        content = report_path.read_text()
        assert "No Models Pass All Criteria" in content
    
    def test_with_real_metrics_from_calculator(self, tmp_path):
        """Should work with metrics generated by MetricsCalculator."""
        results = [
            # Condition C - a_to_b direction
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_003", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_004", "model1", "a_to_b", "followed_user"),
            # Condition C - b_to_a direction
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_002", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_003", "model1", "b_to_a", "followed_user"),
            make_result("C_language_spa_eng_medium_factual_004", "model1", "b_to_a", "followed_user"),
            # Condition B - user-only
            make_result("B_language_eng_medium_factual_001", "model1", "option_a", "followed_user"),
            make_result("B_language_eng_medium_factual_002", "model1", "option_b", "followed_user"),
            # Condition A - system-only
            make_result("A_language_eng_medium_factual_001", "model1", "option_a", "followed_system"),
            make_result("A_language_eng_medium_factual_002", "model1", "option_b", "followed_system"),
            # Condition D - recency
            make_result("D_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_user"),
            make_result("D_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_user"),
        ]
        
        calc = MetricsCalculator(results)
        metrics = calc.compute_all()
        
        generator = ReportGenerator(metrics, results)
        output_dir = tmp_path / "report"
        
        generator.generate_full_report(str(output_dir))
        
        # Verify all files are created
        assert (output_dir / "report.md").exists()
        assert (output_dir / "hierarchy_index.png").exists()
        assert (output_dir / "directional_comparison.png").exists()
        assert (output_dir / "strength_effect.png").exists()
        assert (output_dir / "asymmetry_analysis.png").exists()
        
        # Verify report content
        content = (output_dir / "report.md").read_text()
        assert "model1" in content
        assert "## Go/No-Go Assessment" in content
        assert "## Failure Cases" in content
    
    def test_overwrites_existing_directory(self, tmp_path):
        """Should work with existing output directory."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        output_dir = tmp_path / "report"
        
        # Create directory and a file
        output_dir.mkdir()
        (output_dir / "old_file.txt").write_text("old content")
        
        generator.generate_full_report(str(output_dir))
        
        # New files should be created
        assert (output_dir / "report.md").exists()
        # Old file should still exist (we don't delete it)
        assert (output_dir / "old_file.txt").exists()


class TestRecommendModelAndStrength:
    """Tests for recommend_model_and_strength method."""
    
    def test_returns_dict_with_required_keys(self):
        """Should return dict with recommended_model, recommended_strength, rationale, warnings."""
        metrics = {"model1": make_model_metrics("model1")}
        generator = ReportGenerator(metrics, [])
        
        result = generator.recommend_model_and_strength()
        
        assert 'recommended_model' in result
        assert 'recommended_strength' in result
        assert 'rationale' in result
        assert 'warnings' in result
    
    def test_empty_metrics_returns_none_model(self):
        """Should return None for recommended_model when no metrics."""
        generator = ReportGenerator({}, [])
        
        result = generator.recommend_model_and_strength()
        
        assert result['recommended_model'] is None
        assert result['recommended_strength'] is None
        assert 'No metrics data available' in result['rationale']
        assert result['warnings'] == []
    
    def test_recommends_passing_model(self):
        """Should recommend model that passes all go/no-go criteria."""
        # Create a model that passes all criteria
        metrics = {
            "model1": make_model_metrics(
                "model1",
                hierarchy_index_balanced=0.85,
                conflict_resolution=0.90,
                scr_balanced=0.80,
                recency_balanced=0.40,
                scr_asymmetry=0.05,
                warnings=[]
            )
        }
        generator = ReportGenerator(metrics, [])
        
        result = generator.recommend_model_and_strength()
        
        assert result['recommended_model'] == 'model1'
        assert 'model1' in result['rationale']
    
    def test_recommends_highest_hierarchy_index_among_passing(self):
        """Should recommend model with highest hierarchy index among passing models."""
        metrics = {
            "model1": make_model_metrics(
                "model1",
                hierarchy_index_balanced=0.85,
                conflict_resolution=0.90,
                scr_balanced=0.80,
                recency_balanced=0.40,
                scr_asymmetry=0.05,
                warnings=[]
            ),
            "model2": make_model_metrics(
                "model2",
                hierarchy_index_balanced=0.95,  # Higher HI
                conflict_resolution=0.90,
                scr_balanced=0.85,
                recency_balanced=0.40,
                scr_asymmetry=0.05,
                warnings=[]
            )
        }
        generator = ReportGenerator(metrics, [])
        
        result = generator.recommend_model_and_strength()
        
        assert result['recommended_model'] == 'model2'
    
    def test_returns_best_available_when_none_pass(self):
        """Should return best available model when none pass all criteria."""
        # Create models that fail criteria
        metrics = {
            "model1": make_model_metrics(
                "model1",
                hierarchy_index_balanced=0.60,  # Below 0.7 threshold
                conflict_resolution=0.70,  # Below 0.8 threshold
                scr_balanced=0.50,
                recency_balanced=0.60,  # SCR < Recency
                scr_asymmetry=0.05,
                warnings=[]
            ),
            "model2": make_model_metrics(
                "model2",
                hierarchy_index_balanced=0.65,  # Higher but still below threshold
                conflict_resolution=0.75,
                scr_balanced=0.55,
                recency_balanced=0.60,
                scr_asymmetry=0.05,
                warnings=[]
            )
        }
        generator = ReportGenerator(metrics, [])
        
        result = generator.recommend_model_and_strength()
        
        # Should recommend model2 as it has higher hierarchy index
        assert result['recommended_model'] == 'model2'
        assert 'No models pass all go/no-go criteria' in result['rationale']
    
    def test_includes_capability_bias_warnings(self):
        """Should include capability bias warnings in result."""
        warnings = ["SCR shows high asymmetry (0.25): model may prefer a_to_b direction"]
        metrics = {
            "model1": make_model_metrics(
                "model1",
                hierarchy_index_balanced=0.85,
                conflict_resolution=0.90,
                scr_balanced=0.80,
                recency_balanced=0.40,
                scr_asymmetry=0.25,
                warnings=warnings
            )
        }
        generator = ReportGenerator(metrics, [])
        
        result = generator.recommend_model_and_strength()
        
        assert len(result['warnings']) > 0
        assert 'model1' in result['warnings'][0]
    
    def test_recommends_strength_with_highest_scr(self):
        """Should recommend strength level with highest SCR."""
        # Create model with strength breakdown
        model_metrics = make_model_metrics(
            "model1",
            hierarchy_index_balanced=0.85,
            conflict_resolution=0.90,
            scr_balanced=0.80,
            recency_balanced=0.40,
            scr_asymmetry=0.05,
            warnings=[]
        )
        # Add strength breakdown
        model_metrics.by_strength = {
            'weak': make_model_metrics("model1", scr_balanced=0.60),
            'medium': make_model_metrics("model1", scr_balanced=0.75),
            'strong': make_model_metrics("model1", scr_balanced=0.90),
        }
        
        metrics = {"model1": model_metrics}
        generator = ReportGenerator(metrics, [])
        
        result = generator.recommend_model_and_strength()
        
        assert result['recommended_strength'] == 'strong'
    
    def test_returns_none_strength_when_no_strength_data(self):
        """Should return None for strength when no by_strength data."""
        metrics = {
            "model1": make_model_metrics(
                "model1",
                hierarchy_index_balanced=0.85,
                conflict_resolution=0.90,
                scr_balanced=0.80,
                recency_balanced=0.40,
                scr_asymmetry=0.05,
                warnings=[]
            )
        }
        generator = ReportGenerator(metrics, [])
        
        result = generator.recommend_model_and_strength()
        
        assert result['recommended_strength'] is None
    
    def test_rationale_explains_passing_criteria(self):
        """Should explain why model passes criteria in rationale."""
        metrics = {
            "model1": make_model_metrics(
                "model1",
                hierarchy_index_balanced=0.85,
                conflict_resolution=0.90,
                scr_balanced=0.80,
                recency_balanced=0.40,
                scr_asymmetry=0.05,
                warnings=[]
            )
        }
        generator = ReportGenerator(metrics, [])
        
        result = generator.recommend_model_and_strength()
        
        assert 'Hierarchy Index' in result['rationale']
        assert 'Conflict Resolution' in result['rationale']
    
    def test_rationale_explains_failing_criteria(self):
        """Should explain failing criteria when no model passes."""
        metrics = {
            "model1": make_model_metrics(
                "model1",
                hierarchy_index_balanced=0.60,  # Below threshold
                conflict_resolution=0.70,  # Below threshold
                scr_balanced=0.50,
                recency_balanced=0.60,  # SCR < Recency
                scr_asymmetry=0.05,
                warnings=[]
            )
        }
        generator = ReportGenerator(metrics, [])
        
        result = generator.recommend_model_and_strength()
        
        assert 'Failing criteria' in result['rationale']
    
    def test_with_real_metrics_from_calculator(self):
        """Should work with metrics generated by MetricsCalculator."""
        results = [
            # Condition C - high SCR (passes)
            make_result("C_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_002", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_003", "model1", "a_to_b", "followed_system"),
            make_result("C_language_eng_spa_medium_factual_004", "model1", "a_to_b", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_002", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_003", "model1", "b_to_a", "followed_system"),
            make_result("C_language_spa_eng_medium_factual_004", "model1", "b_to_a", "followed_system"),
            # Condition B
            make_result("B_language_eng_medium_factual_001", "model1", "option_a", "followed_user"),
            make_result("B_language_eng_medium_factual_002", "model1", "option_b", "followed_user"),
            # Condition A
            make_result("A_language_eng_medium_factual_001", "model1", "option_a", "followed_system"),
            make_result("A_language_eng_medium_factual_002", "model1", "option_b", "followed_system"),
            # Condition D - low recency
            make_result("D_language_eng_spa_medium_factual_001", "model1", "a_to_b", "followed_system"),
            make_result("D_language_spa_eng_medium_factual_001", "model1", "b_to_a", "followed_system"),
        ]
        
        calc = MetricsCalculator(results)
        metrics = calc.compute_all()
        
        generator = ReportGenerator(metrics, results)
        result = generator.recommend_model_and_strength()
        
        assert result['recommended_model'] == 'model1'
        assert isinstance(result['rationale'], str)
        assert isinstance(result['warnings'], list)
