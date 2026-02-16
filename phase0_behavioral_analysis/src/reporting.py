"""
Report generation for Phase 0 Behavioral Analysis.

Generates visualizations and reports for behavioral baseline analysis,
including summary tables, hierarchy index charts, directional comparisons,
and go/no-go assessments.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from .metrics import ModelMetrics
from .experiment import ExperimentResult


class ReportGenerator:
    """
    Generates visualizations and reports for behavioral analysis.
    
    Creates summary tables, charts, and reports to support go/no-go
    decisions and model selection for Phase 1.
    """
    
    def __init__(
        self,
        metrics: dict[str, ModelMetrics],
        results: list[ExperimentResult]
    ):
        """
        Initialize the report generator.
        
        Args:
            metrics: Dictionary mapping model ID to ModelMetrics
            results: List of experiment results for detailed analysis
        """
        self.metrics = metrics
        self.results = results
    
    def _format_metric_with_ci(self, metric) -> str:
        """
        Format a metric value with its confidence interval.
        
        Args:
            metric: A MetricValue object with value, ci_lower, and ci_upper
            
        Returns:
            Formatted string like "0.75 [0.65, 0.85]"
        """
        return f"{metric.value:.2f} [{metric.ci_lower:.2f}, {metric.ci_upper:.2f}]"
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate summary table of all metrics (balanced values).
        
        Creates a pandas DataFrame with one row per model containing all
        key metrics with confidence intervals where applicable.
        
        Returns:
            DataFrame with columns:
            - Model: Model name/ID
            - SCR (balanced): System Compliance Rate with CI
            - UCR: User Compliance Rate with CI
            - SBR: System Baseline Rate with CI
            - Recency Effect (balanced): Recency effect with CI
            - Hierarchy Index (balanced): Hierarchy index with CI
            - Conflict Resolution Rate: Conflict resolution rate with CI
            - Asymmetry (SCR): SCR asymmetry value
            - Capability Bias Warnings: Count of capability bias warnings
        """
        rows = []
        
        for model_id, model_metrics in self.metrics.items():
            row = {
                'Model': model_id,
                'SCR (balanced)': self._format_metric_with_ci(model_metrics.scr.balanced),
                'UCR': self._format_metric_with_ci(model_metrics.ucr),
                'SBR': self._format_metric_with_ci(model_metrics.sbr),
                'Recency Effect (balanced)': self._format_metric_with_ci(model_metrics.recency.balanced),
                'Hierarchy Index (balanced)': self._format_metric_with_ci(model_metrics.hierarchy_index.balanced),
                'Conflict Resolution Rate': self._format_metric_with_ci(model_metrics.conflict_resolution),
                'Asymmetry (SCR)': f"{model_metrics.scr.asymmetry:.2f}",
                'Capability Bias Warnings': len(model_metrics.capability_bias_warnings)
            }
            rows.append(row)
        
        # Create DataFrame with specified column order
        columns = [
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
        
        df = pd.DataFrame(rows, columns=columns)
        
        return df
    
    def plot_hierarchy_index(self, output_path: str) -> None:
        """
        Bar chart of Hierarchy Index by model with CIs (balanced).
        
        Creates a bar chart showing the balanced Hierarchy Index for each model
        with error bars representing the confidence intervals. Includes a
        horizontal line at 0.7 indicating the go/no-go threshold.
        
        Args:
            output_path: Path where the figure will be saved (e.g., 'reports/hierarchy_index.png')
        """
        if not self.metrics:
            # Create empty figure if no metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Hierarchy Index by Model')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return
        
        # Extract data for plotting
        models = list(self.metrics.keys())
        hierarchy_values = []
        ci_lower_errors = []
        ci_upper_errors = []
        
        for model_id in models:
            model_metrics = self.metrics[model_id]
            hi_balanced = model_metrics.hierarchy_index.balanced
            hierarchy_values.append(hi_balanced.value)
            # Error bars are relative to the value (asymmetric)
            ci_lower_errors.append(hi_balanced.value - hi_balanced.ci_lower)
            ci_upper_errors.append(hi_balanced.ci_upper - hi_balanced.value)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar positions
        x_positions = range(len(models))
        
        # Plot bars with error bars
        bars = ax.bar(
            x_positions,
            hierarchy_values,
            yerr=[ci_lower_errors, ci_upper_errors],
            capsize=5,
            color='steelblue',
            edgecolor='black',
            alpha=0.8,
            error_kw={'elinewidth': 2, 'capthick': 2}
        )
        
        # Add horizontal threshold line at 0.7
        ax.axhline(
            y=0.7,
            color='red',
            linestyle='--',
            linewidth=2,
            label='Go/No-Go Threshold (0.7)'
        )
        
        # Customize the plot
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Hierarchy Index (balanced)', fontsize=12)
        ax.set_title('Hierarchy Index by Model with 95% Confidence Intervals', fontsize=14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)  # Hierarchy index is between 0 and 1
        ax.legend(loc='lower right')
        
        # Add grid for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def plot_directional_comparison(self, output_path: str) -> None:
        """
        Side-by-side comparison of both directions per constraint.
        
        Creates a grouped bar chart showing SCR for both directions (a_to_b and b_to_a)
        for each model. Highlights asymmetry to visualize capability bias.
        
        Args:
            output_path: Path where the figure will be saved (e.g., 'reports/directional_comparison.png')
        """
        import numpy as np
        
        if not self.metrics:
            # Create empty figure if no metrics
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Directional Comparison of System Compliance Rate')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return
        
        # Extract data for plotting
        models = list(self.metrics.keys())
        n_models = len(models)
        
        # Data arrays for each direction
        a_to_b_values = []
        a_to_b_ci_lower = []
        a_to_b_ci_upper = []
        b_to_a_values = []
        b_to_a_ci_lower = []
        b_to_a_ci_upper = []
        asymmetries = []
        
        for model_id in models:
            model_metrics = self.metrics[model_id]
            scr = model_metrics.scr
            
            # a_to_b direction
            a_to_b_values.append(scr.a_to_b.value)
            a_to_b_ci_lower.append(scr.a_to_b.value - scr.a_to_b.ci_lower)
            a_to_b_ci_upper.append(scr.a_to_b.ci_upper - scr.a_to_b.value)
            
            # b_to_a direction
            b_to_a_values.append(scr.b_to_a.value)
            b_to_a_ci_lower.append(scr.b_to_a.value - scr.b_to_a.ci_lower)
            b_to_a_ci_upper.append(scr.b_to_a.ci_upper - scr.b_to_a.value)
            
            # Asymmetry
            asymmetries.append(scr.asymmetry)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Bar positions
        x = np.arange(n_models)
        bar_width = 0.35
        
        # Plot bars for each direction with different colors
        bars_a_to_b = ax.bar(
            x - bar_width / 2,
            a_to_b_values,
            bar_width,
            yerr=[a_to_b_ci_lower, a_to_b_ci_upper],
            capsize=4,
            color='steelblue',
            edgecolor='black',
            alpha=0.8,
            label='Direction A→B',
            error_kw={'elinewidth': 1.5, 'capthick': 1.5}
        )
        
        bars_b_to_a = ax.bar(
            x + bar_width / 2,
            b_to_a_values,
            bar_width,
            yerr=[b_to_a_ci_lower, b_to_a_ci_upper],
            capsize=4,
            color='coral',
            edgecolor='black',
            alpha=0.8,
            label='Direction B→A',
            error_kw={'elinewidth': 1.5, 'capthick': 1.5}
        )
        
        # Annotate asymmetry above each pair of bars
        for i, (a_val, b_val, asym) in enumerate(zip(a_to_b_values, b_to_a_values, asymmetries)):
            # Position annotation above the higher bar
            max_val = max(a_val, b_val)
            # Get the upper CI for the higher bar to position annotation above error bars
            if a_val >= b_val:
                annotation_y = a_val + a_to_b_ci_upper[i] + 0.03
            else:
                annotation_y = b_val + b_to_a_ci_upper[i] + 0.03
            
            # Highlight high asymmetry (> 0.15 threshold) in red
            if asym > 0.15:
                color = 'red'
                fontweight = 'bold'
            else:
                color = 'black'
                fontweight = 'normal'
            
            ax.annotate(
                f'Δ={asym:.2f}',
                xy=(i, annotation_y),
                ha='center',
                va='bottom',
                fontsize=9,
                color=color,
                fontweight=fontweight
            )
        
        # Customize the plot
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('System Compliance Rate (SCR)', fontsize=12)
        ax.set_title('Directional Comparison of SCR with Asymmetry Annotations', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1.15)  # Leave room for annotations
        ax.legend(loc='upper right')
        
        # Add grid for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add note about asymmetry threshold
        ax.text(
            0.02, 0.98,
            'Red Δ indicates high asymmetry (>0.15)',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            color='red',
            style='italic'
        )
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def plot_asymmetry_analysis(self, output_path: str) -> None:
        """
        Bar chart showing asymmetry per constraint type per model.
        
        Creates a grouped bar chart showing asymmetry values for each constraint
        type per model. Includes a horizontal red line at 0.15 (the ASYMMETRY_THRESHOLD)
        to flag problematic cases. Bars exceeding the threshold are highlighted
        with a different color.
        
        Args:
            output_path: Path where the figure will be saved (e.g., 'reports/asymmetry_analysis.png')
        """
        import numpy as np
        from .metrics import MetricsCalculator
        
        if not self.metrics:
            # Create empty figure if no metrics
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Asymmetry Analysis by Constraint Type')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return
        
        # Collect all constraint types across all models
        all_constraint_types = set()
        for model_metrics in self.metrics.values():
            all_constraint_types.update(model_metrics.by_constraint.keys())
        
        if not all_constraint_types:
            # No constraint breakdown data available
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No constraint breakdown data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Asymmetry Analysis by Constraint Type')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return
        
        # Sort constraint types for consistent ordering
        constraint_types = sorted(all_constraint_types)
        models = list(self.metrics.keys())
        n_models = len(models)
        n_constraints = len(constraint_types)
        
        # Define colors for constraint types
        constraint_colors = {
            'language': 'steelblue',
            'format': 'coral',
        }
        # Default colors for additional constraint types
        default_colors = ['seagreen', 'orchid', 'goldenrod', 'slategray', 'crimson', 'teal']
        for i, ct in enumerate(constraint_types):
            if ct not in constraint_colors:
                constraint_colors[ct] = default_colors[i % len(default_colors)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Bar positioning
        bar_width = 0.8 / n_constraints if n_constraints > 0 else 0.8
        x = np.arange(n_models)
        
        # Plot bars for each constraint type
        for i, constraint_type in enumerate(constraint_types):
            asymmetry_values = []
            bar_colors = []
            
            for model_id in models:
                model_metrics = self.metrics[model_id]
                if constraint_type in model_metrics.by_constraint:
                    constraint_metrics = model_metrics.by_constraint[constraint_type]
                    asymmetry = constraint_metrics.scr.asymmetry
                    asymmetry_values.append(asymmetry)
                    # Highlight bars exceeding threshold
                    if asymmetry > MetricsCalculator.ASYMMETRY_THRESHOLD:
                        bar_colors.append('red')
                    else:
                        bar_colors.append(constraint_colors[constraint_type])
                else:
                    asymmetry_values.append(0.0)
                    bar_colors.append(constraint_colors[constraint_type])
            
            # Calculate bar positions
            offset = (i - (n_constraints - 1) / 2) * bar_width
            positions = x + offset
            
            # Plot bars
            bars = ax.bar(
                positions,
                asymmetry_values,
                bar_width,
                color=bar_colors,
                edgecolor='black',
                alpha=0.8,
                label=constraint_type.capitalize()
            )
            
            # Add value labels on top of bars
            for j, (pos, val) in enumerate(zip(positions, asymmetry_values)):
                if val > 0:
                    color = 'red' if val > MetricsCalculator.ASYMMETRY_THRESHOLD else 'black'
                    fontweight = 'bold' if val > MetricsCalculator.ASYMMETRY_THRESHOLD else 'normal'
                    ax.annotate(
                        f'{val:.2f}',
                        xy=(pos, val),
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        color=color,
                        fontweight=fontweight
                    )
        
        # Add horizontal threshold line at 0.15
        ax.axhline(
            y=MetricsCalculator.ASYMMETRY_THRESHOLD,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Threshold ({MetricsCalculator.ASYMMETRY_THRESHOLD})'
        )
        
        # Customize the plot
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Asymmetry (|SCR_a→b - SCR_b→a|)', fontsize=12)
        ax.set_title('Asymmetry Analysis by Constraint Type per Model', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, max(0.5, ax.get_ylim()[1] * 1.1))  # Ensure y-axis shows at least up to 0.5
        ax.legend(loc='upper right')
        
        # Add grid for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Add note about threshold
        ax.text(
            0.02, 0.98,
            'Red bars indicate high asymmetry (capability bias)',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            color='red',
            style='italic'
        )
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def get_failure_cases(self) -> pd.DataFrame:
        """
        List prompts where model followed_user or followed_neither in Condition C.
        
        These are "failure cases" because in Condition C (hierarchy conflict),
        we expect the model to follow the system instruction. Cases where the
        model followed the user instruction or neither instruction are failures
        from a hierarchy compliance perspective.
        
        Returns:
            DataFrame with columns:
            - prompt_id: Unique identifier of the prompt
            - model: Model ID
            - direction: Counterbalancing direction ('a_to_b' or 'b_to_a')
            - label: Compliance label ('followed_user' or 'followed_neither')
            - response: The actual model response text
            - confidence: Classification confidence score
        """
        failure_rows = []
        
        for result in self.results:
            # Check if this is a Condition C result (prompt_id starts with 'C_')
            if not result.prompt_id.startswith('C_'):
                continue
            
            # Check if this is a failure case (not followed_system)
            if result.label in ('followed_user', 'followed_neither'):
                failure_rows.append({
                    'prompt_id': result.prompt_id,
                    'model': result.model,
                    'direction': result.direction,
                    'label': result.label,
                    'response': result.response,
                    'confidence': result.confidence
                })
        
        # Create DataFrame with specified column order
        columns = ['prompt_id', 'model', 'direction', 'label', 'response', 'confidence']
        df = pd.DataFrame(failure_rows, columns=columns)
        
        return df

    def plot_strength_effect(self, output_path: str) -> None:
        """
        Line plot of SCR vs strength level per model (balanced).
        
        Creates a line plot showing how System Compliance Rate (balanced) changes
        across strength levels (weak, medium, strong) for each model. Each model
        is represented by a different colored line with markers. Error bars or
        shaded regions show confidence intervals.
        
        Args:
            output_path: Path where the figure will be saved (e.g., 'reports/strength_effect.png')
        """
        import numpy as np
        
        if not self.metrics:
            # Create empty figure if no metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Effect of Prompt Strength on System Compliance Rate')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return
        
        # Define strength levels in order
        strength_levels = ['weak', 'medium', 'strong']
        x_positions = np.arange(len(strength_levels))
        
        # Define colors and markers for different models
        colors = ['steelblue', 'coral', 'seagreen', 'orchid', 'goldenrod', 'slategray']
        markers = ['o', 's', '^', 'D', 'v', 'p']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Track if any model has strength data
        has_data = False
        
        # Plot line for each model
        for idx, (model_id, model_metrics) in enumerate(self.metrics.items()):
            # Get color and marker for this model
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            
            # Extract SCR values for each strength level
            scr_values = []
            ci_lower_errors = []
            ci_upper_errors = []
            valid_positions = []
            
            for i, strength in enumerate(strength_levels):
                if strength in model_metrics.by_strength:
                    strength_metrics = model_metrics.by_strength[strength]
                    scr_balanced = strength_metrics.scr.balanced
                    scr_values.append(scr_balanced.value)
                    ci_lower_errors.append(scr_balanced.value - scr_balanced.ci_lower)
                    ci_upper_errors.append(scr_balanced.ci_upper - scr_balanced.value)
                    valid_positions.append(i)
                    has_data = True
            
            if scr_values:
                # Convert to numpy arrays for plotting
                valid_positions = np.array(valid_positions)
                scr_values = np.array(scr_values)
                ci_lower_errors = np.array(ci_lower_errors)
                ci_upper_errors = np.array(ci_upper_errors)
                
                # Plot line with markers
                ax.plot(
                    valid_positions,
                    scr_values,
                    color=color,
                    marker=marker,
                    markersize=8,
                    linewidth=2,
                    label=model_id,
                    alpha=0.8
                )
                
                # Add error bars for confidence intervals
                ax.errorbar(
                    valid_positions,
                    scr_values,
                    yerr=[ci_lower_errors, ci_upper_errors],
                    fmt='none',
                    color=color,
                    capsize=5,
                    capthick=2,
                    elinewidth=2,
                    alpha=0.6
                )
        
        if not has_data:
            # No strength data available
            ax.text(
                0.5, 0.5,
                'No strength breakdown data available',
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=12
            )
        
        # Customize the plot
        ax.set_xlabel('Prompt Strength Level', fontsize=12)
        ax.set_ylabel('System Compliance Rate (balanced)', fontsize=12)
        ax.set_title('Effect of Prompt Strength on SCR with 95% Confidence Intervals', fontsize=14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(strength_levels)
        ax.set_ylim(0, 1.05)  # SCR is between 0 and 1
        
        # Add legend if there's data
        if has_data:
            ax.legend(loc='best', fontsize=10)
        
        # Add grid for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def generate_full_report(self, output_dir: str) -> None:
        """
        Generate complete Behavioral Baseline Report as markdown + figures.
        
        Creates the output directory if it doesn't exist, generates all figures
        by calling the existing plot methods, and creates a markdown report file
        that includes the summary table, embedded figures with descriptions,
        go/no-go assessment results, failure cases summary, and recommendations.
        
        Args:
            output_dir: Directory where the report and figures will be saved.
                       Will be created if it doesn't exist.
        """
        import os
        from .metrics import MetricsCalculator
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all figures
        self.plot_hierarchy_index(os.path.join(output_dir, 'hierarchy_index.png'))
        self.plot_directional_comparison(os.path.join(output_dir, 'directional_comparison.png'))
        self.plot_strength_effect(os.path.join(output_dir, 'strength_effect.png'))
        self.plot_asymmetry_analysis(os.path.join(output_dir, 'asymmetry_analysis.png'))
        
        # Build markdown report content
        report_lines = []
        
        # Title
        report_lines.append("# Behavioral Baseline Report")
        report_lines.append("")
        report_lines.append("This report summarizes the behavioral analysis results for instruction hierarchy evaluation.")
        report_lines.append("")
        
        # Summary Table Section
        report_lines.append("## Summary Table")
        report_lines.append("")
        summary_df = self.generate_summary_table()
        if len(summary_df) > 0:
            report_lines.append(summary_df.to_markdown(index=False))
        else:
            report_lines.append("*No metrics data available.*")
        report_lines.append("")
        
        # Figures Section
        report_lines.append("## Visualizations")
        report_lines.append("")
        
        # Hierarchy Index Figure
        report_lines.append("### Hierarchy Index by Model")
        report_lines.append("")
        report_lines.append("The Hierarchy Index measures how strongly each model prefers system instructions over user instructions when they conflict. A value above 0.7 is required to pass the go/no-go threshold.")
        report_lines.append("")
        report_lines.append("![Hierarchy Index](hierarchy_index.png)")
        report_lines.append("")
        
        # Directional Comparison Figure
        report_lines.append("### Directional Comparison")
        report_lines.append("")
        report_lines.append("This chart compares System Compliance Rate (SCR) for both directions (A→B and B→A) per model. High asymmetry (Δ > 0.15) indicates potential capability bias where the model may prefer one option regardless of instruction source.")
        report_lines.append("")
        report_lines.append("![Directional Comparison](directional_comparison.png)")
        report_lines.append("")
        
        # Strength Effect Figure
        report_lines.append("### Effect of Prompt Strength")
        report_lines.append("")
        report_lines.append("This chart shows how System Compliance Rate changes across different prompt strength levels (weak, medium, strong). Higher strength prompts are expected to produce higher compliance rates.")
        report_lines.append("")
        report_lines.append("![Strength Effect](strength_effect.png)")
        report_lines.append("")
        
        # Asymmetry Analysis Figure
        report_lines.append("### Asymmetry Analysis")
        report_lines.append("")
        report_lines.append("This chart shows asymmetry values per constraint type per model. Bars exceeding the 0.15 threshold (red line) indicate capability bias that may confound hierarchy measurements.")
        report_lines.append("")
        report_lines.append("![Asymmetry Analysis](asymmetry_analysis.png)")
        report_lines.append("")
        
        # Go/No-Go Assessment Section
        report_lines.append("## Go/No-Go Assessment")
        report_lines.append("")
        report_lines.append("The following criteria are evaluated for each model:")
        report_lines.append("")
        report_lines.append("- **Hierarchy Index (balanced) > 0.7**: Model shows clear preference for system instructions")
        report_lines.append("- **Conflict Resolution Rate > 0.8**: Model resolves conflicts rather than ignoring both instructions")
        report_lines.append("- **Low Asymmetry**: No capability bias warnings (asymmetry ≤ 0.15)")
        report_lines.append("")
        
        if self.metrics:
            report_lines.append("| Model | Hierarchy Index | Conflict Resolution | Low Asymmetry | **Overall** |")
            report_lines.append("|-------|-----------------|---------------------|---------------|-------------|")
            
            for model_id, model_metrics in self.metrics.items():
                # Use MetricsCalculator to get go/no-go assessment
                calc = MetricsCalculator(self.results)
                assessment = calc.go_nogo_assessment(model_metrics)
                
                # Format pass/fail indicators
                hi_status = "✅ PASS" if assessment['hierarchy_index_pass'] else "❌ FAIL"
                cr_status = "✅ PASS" if assessment['conflict_resolution_pass'] else "❌ FAIL"
                asymmetry_status = "✅ PASS" if assessment['low_asymmetry'] else "❌ FAIL"
                overall_status = "✅ **PASS**" if assessment['overall_pass'] else "❌ **FAIL**"
                
                report_lines.append(f"| {model_id} | {hi_status} | {cr_status} | {asymmetry_status} | {overall_status} |")
            
            report_lines.append("")
            
            # Add detailed values for reference (still includes recency for informational purposes)
            report_lines.append("### Detailed Values")
            report_lines.append("")
            report_lines.append("| Model | HI (balanced) | CR Rate | SCR (balanced) | Recency (balanced) |")
            report_lines.append("|-------|---------------|---------|----------------|-------------------|")
            
            for model_id, model_metrics in self.metrics.items():
                hi_val = f"{model_metrics.hierarchy_index.balanced.value:.3f}"
                cr_val = f"{model_metrics.conflict_resolution.value:.3f}"
                scr_val = f"{model_metrics.scr.balanced.value:.3f}"
                recency_val = f"{model_metrics.recency.balanced.value:.3f}"
                report_lines.append(f"| {model_id} | {hi_val} | {cr_val} | {scr_val} | {recency_val} |")
            
            report_lines.append("")
            report_lines.append("*Note: Recency Effect is shown for informational purposes. It measures behavior in user-user conflicts (Condition D), which is independent of system-user hierarchy.*")
            report_lines.append("")
        else:
            report_lines.append("*No metrics data available for assessment.*")
            report_lines.append("")
        
        # Failure Cases Section
        report_lines.append("## Failure Cases")
        report_lines.append("")
        report_lines.append("Failure cases are prompts in Condition C (hierarchy conflict) where the model followed the user instruction or neither instruction instead of the system instruction.")
        report_lines.append("")
        
        failure_df = self.get_failure_cases()
        if len(failure_df) > 0:
            report_lines.append(f"**Total failure cases: {len(failure_df)}**")
            report_lines.append("")
            
            # Group by model and label for summary
            if 'model' in failure_df.columns and 'label' in failure_df.columns:
                summary = failure_df.groupby(['model', 'label']).size().reset_index(name='count')
                report_lines.append("### Summary by Model and Label")
                report_lines.append("")
                report_lines.append(summary.to_markdown(index=False))
                report_lines.append("")
            
            # Show sample failure cases (limit to first 10 for readability)
            report_lines.append("### Sample Failure Cases (first 10)")
            report_lines.append("")
            sample_df = failure_df.head(10)[['prompt_id', 'model', 'direction', 'label', 'confidence']]
            report_lines.append(sample_df.to_markdown(index=False))
            report_lines.append("")
        else:
            report_lines.append("*No failure cases found. All Condition C prompts followed system instructions.*")
            report_lines.append("")
        
        # Capability Bias Warnings Section
        report_lines.append("## Capability Bias Warnings")
        report_lines.append("")
        
        all_warnings = []
        for model_id, model_metrics in self.metrics.items():
            for warning in model_metrics.capability_bias_warnings:
                all_warnings.append(f"- **{model_id}**: {warning}")
        
        if all_warnings:
            report_lines.append("The following capability bias warnings were detected:")
            report_lines.append("")
            report_lines.extend(all_warnings)
            report_lines.append("")
        else:
            report_lines.append("*No capability bias warnings detected.*")
            report_lines.append("")
        
        # Recommendations Section
        report_lines.append("## Recommendations")
        report_lines.append("")
        
        if self.metrics:
            # Find models that pass all criteria
            passing_models = []
            for model_id, model_metrics in self.metrics.items():
                calc = MetricsCalculator(self.results)
                assessment = calc.go_nogo_assessment(model_metrics)
                if assessment['overall_pass']:
                    passing_models.append((model_id, model_metrics))
            
            if passing_models:
                # Recommend the model with highest hierarchy index among passing models
                best_model = max(passing_models, key=lambda x: x[1].hierarchy_index.balanced.value)
                best_model_id, best_metrics = best_model
                
                report_lines.append(f"### Recommended Model: **{best_model_id}**")
                report_lines.append("")
                report_lines.append("This model is recommended for Phase 1 based on:")
                report_lines.append("")
                report_lines.append(f"- Highest Hierarchy Index among passing models: {best_metrics.hierarchy_index.balanced.value:.3f}")
                report_lines.append(f"- Conflict Resolution Rate: {best_metrics.conflict_resolution.value:.3f}")
                report_lines.append(f"- Low asymmetry (no capability bias warnings)")
                report_lines.append("")
                
                # Recommend strength level based on by_strength data
                if best_metrics.by_strength:
                    # Find strength with highest SCR
                    best_strength = None
                    best_scr = -1
                    for strength, strength_metrics in best_metrics.by_strength.items():
                        if strength_metrics.scr.balanced.value > best_scr:
                            best_scr = strength_metrics.scr.balanced.value
                            best_strength = strength
                    
                    if best_strength:
                        report_lines.append(f"### Recommended Prompt Strength: **{best_strength}**")
                        report_lines.append("")
                        report_lines.append(f"This strength level achieved the highest SCR ({best_scr:.3f}) for the recommended model.")
                        report_lines.append("")
            else:
                report_lines.append("### No Models Pass All Criteria")
                report_lines.append("")
                report_lines.append("None of the evaluated models pass all go/no-go criteria. Consider:")
                report_lines.append("")
                report_lines.append("1. Evaluating additional models")
                report_lines.append("2. Adjusting prompt strength levels")
                report_lines.append("3. Investigating capability bias issues")
                report_lines.append("")
                
                # Still provide best available option
                if self.metrics:
                    best_model_id = max(
                        self.metrics.keys(),
                        key=lambda m: self.metrics[m].hierarchy_index.balanced.value
                    )
                    best_metrics = self.metrics[best_model_id]
                    report_lines.append(f"**Best available option**: {best_model_id} (Hierarchy Index: {best_metrics.hierarchy_index.balanced.value:.3f})")
                    report_lines.append("")
        else:
            report_lines.append("*No metrics data available for recommendations.*")
            report_lines.append("")
        
        # Write report to file
        report_path = os.path.join(output_dir, 'report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

    def recommend_model_and_strength(self) -> dict:
        """
        Recommend best model and strength for Phase 1.
        
        Considers:
        - Hierarchy Index (balanced)
        - Low asymmetry (no capability bias)
        - Conflict resolution rate
        
        Returns:
            Dictionary with:
            - recommended_model: str or None if no suitable model
            - recommended_strength: str or None if no strength data
            - rationale: str explaining the recommendation
            - warnings: list[str] of any capability bias concerns
        """
        from .metrics import MetricsCalculator
        
        if not self.metrics:
            return {
                'recommended_model': None,
                'recommended_strength': None,
                'rationale': 'No metrics data available for recommendation.',
                'warnings': []
            }
        
        # Collect all warnings across models
        all_warnings = []
        for model_id, model_metrics in self.metrics.items():
            for warning in model_metrics.capability_bias_warnings:
                all_warnings.append(f"{model_id}: {warning}")
        
        # Find models that pass all go/no-go criteria
        passing_models = []
        calc = MetricsCalculator(self.results)
        
        for model_id, model_metrics in self.metrics.items():
            assessment = calc.go_nogo_assessment(model_metrics)
            if assessment['overall_pass']:
                passing_models.append((model_id, model_metrics))
        
        if passing_models:
            # Select the model with highest hierarchy index among passing models
            best_model_id, best_metrics = max(
                passing_models,
                key=lambda x: x[1].hierarchy_index.balanced.value
            )
            
            # Determine best strength level
            recommended_strength = None
            if best_metrics.by_strength:
                # Find strength with highest SCR
                best_strength = None
                best_scr = -1.0
                for strength, strength_metrics in best_metrics.by_strength.items():
                    if strength_metrics.scr.balanced.value > best_scr:
                        best_scr = strength_metrics.scr.balanced.value
                        best_strength = strength
                recommended_strength = best_strength
            
            # Build rationale
            rationale_parts = [
                f"Model '{best_model_id}' is recommended for Phase 1.",
                f"Hierarchy Index (balanced): {best_metrics.hierarchy_index.balanced.value:.3f} (passes >0.7 threshold).",
                f"Conflict Resolution Rate: {best_metrics.conflict_resolution.value:.3f} (passes >0.8 threshold).",
                "No capability bias warnings detected."
            ]
            
            if recommended_strength:
                rationale_parts.append(
                    f"Recommended strength '{recommended_strength}' achieved highest SCR ({best_scr:.3f})."
                )
            
            return {
                'recommended_model': best_model_id,
                'recommended_strength': recommended_strength,
                'rationale': ' '.join(rationale_parts),
                'warnings': all_warnings
            }
        else:
            # No models pass all criteria - recommend best available with warnings
            best_model_id = max(
                self.metrics.keys(),
                key=lambda m: self.metrics[m].hierarchy_index.balanced.value
            )
            best_metrics = self.metrics[best_model_id]
            
            # Determine best strength level
            recommended_strength = None
            best_scr = -1.0
            if best_metrics.by_strength:
                for strength, strength_metrics in best_metrics.by_strength.items():
                    if strength_metrics.scr.balanced.value > best_scr:
                        best_scr = strength_metrics.scr.balanced.value
                        recommended_strength = strength
            
            # Build rationale explaining why no model fully passes
            calc = MetricsCalculator(self.results)
            assessment = calc.go_nogo_assessment(best_metrics)
            
            failing_criteria = []
            if not assessment['hierarchy_index_pass']:
                failing_criteria.append(
                    f"Hierarchy Index ({best_metrics.hierarchy_index.balanced.value:.3f}) below 0.7 threshold"
                )
            if not assessment['conflict_resolution_pass']:
                failing_criteria.append(
                    f"Conflict Resolution Rate ({best_metrics.conflict_resolution.value:.3f}) below 0.8 threshold"
                )
            if not assessment['low_asymmetry']:
                failing_criteria.append("High asymmetry detected (capability bias)")
            
            rationale_parts = [
                "No models pass all go/no-go criteria.",
                f"Best available option: '{best_model_id}' with Hierarchy Index {best_metrics.hierarchy_index.balanced.value:.3f}.",
                f"Failing criteria: {'; '.join(failing_criteria)}.",
                "Consider evaluating additional models or adjusting experimental parameters."
            ]
            
            if recommended_strength:
                rationale_parts.append(
                    f"If proceeding, strength '{recommended_strength}' achieved highest SCR ({best_scr:.3f})."
                )
            
            return {
                'recommended_model': best_model_id,
                'recommended_strength': recommended_strength,
                'rationale': ' '.join(rationale_parts),
                'warnings': all_warnings
            }
