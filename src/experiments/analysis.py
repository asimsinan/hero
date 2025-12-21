"""Analysis utilities for experiment results.

This module provides functions for:
- Loading and aggregating results
- Statistical tests (Wilcoxon, Friedman)
- Visualization (convergence, Pareto fronts, box plots)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOTTING = True
except ImportError:
    plt = None
    sns = None
    _HAS_PLOTTING = False

try:
    from scipy import stats
    _HAS_SCIPY = True
except ImportError:
    stats = None
    _HAS_SCIPY = False


@dataclass
class AggregatedResults:
    """Aggregated results across multiple runs."""
    instance_name: str
    n_runs: int
    
    # Cost statistics
    mean_initial_cost: float
    mean_final_cost: float
    std_final_cost: float
    mean_improvement: float
    
    # Fairness statistics
    mean_driver_cv: float
    std_driver_cv: float
    mean_customer_jain: float
    
    # Performance statistics
    mean_runtime: float
    std_runtime: float
    mean_iterations: int


def load_results(results_dir: str | Path) -> List[Dict[str, Any]]:
    """Load all JSON result files from a directory.
    
    Args:
        results_dir: Directory containing result JSON files
        
    Returns:
        List of result dictionaries
    """
    results_dir = Path(results_dir)
    results = []
    
    for json_file in results_dir.glob("*_results.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            data['_file'] = str(json_file)
            results.append(data)
    
    return results


def aggregate_by_instance(results: List[Dict]) -> Dict[str, AggregatedResults]:
    """Aggregate results by instance name.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dict mapping instance name to aggregated results
    """
    by_instance = {}
    
    for r in results:
        name = r['instance_name']
        if name not in by_instance:
            by_instance[name] = []
        by_instance[name].append(r)
    
    aggregated = {}
    for name, runs in by_instance.items():
        final_costs = [r['final_cost'] for r in runs]
        driver_cvs = [r.get('driver_cv', 0) for r in runs]
        runtimes = [r['runtime'] for r in runs]
        
        aggregated[name] = AggregatedResults(
            instance_name=name,
            n_runs=len(runs),
            mean_initial_cost=np.mean([r['initial_cost'] for r in runs]),
            mean_final_cost=np.mean(final_costs),
            std_final_cost=np.std(final_costs),
            mean_improvement=np.mean([r['improvement'] for r in runs]),
            mean_driver_cv=np.mean(driver_cvs),
            std_driver_cv=np.std(driver_cvs),
            mean_customer_jain=np.mean([r.get('customer_jain', 1.0) for r in runs]),
            mean_runtime=np.mean(runtimes),
            std_runtime=np.std(runtimes),
            mean_iterations=int(np.mean([r['iterations'] for r in runs])),
        )
    
    return aggregated


def statistical_tests(
    results_a: List[float],
    results_b: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Perform statistical significance tests.
    
    Args:
        results_a: Results from algorithm A
        results_b: Results from algorithm B
        alpha: Significance level
        
    Returns:
        Dict with test results
    """
    if not _HAS_SCIPY:
        return {"error": "scipy not installed"}
    
    results = {}
    
    # Wilcoxon signed-rank test (paired)
    if len(results_a) == len(results_b) and len(results_a) >= 5:
        try:
            stat, p_value = stats.wilcoxon(results_a, results_b)
            results['wilcoxon'] = {
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'better': 'A' if np.mean(results_a) < np.mean(results_b) else 'B',
            }
        except Exception as e:
            results['wilcoxon'] = {'error': str(e)}
    
    # Mann-Whitney U test (unpaired)
    try:
        stat, p_value = stats.mannwhitneyu(results_a, results_b)
        results['mann_whitney'] = {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < alpha,
        }
    except Exception as e:
        results['mann_whitney'] = {'error': str(e)}
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(results_a)**2 + np.std(results_b)**2) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(results_a) - np.mean(results_b)) / pooled_std
        results['effect_size'] = {
            'cohens_d': cohens_d,
            'interpretation': (
                'negligible' if abs(cohens_d) < 0.2 else
                'small' if abs(cohens_d) < 0.5 else
                'medium' if abs(cohens_d) < 0.8 else
                'large'
            ),
        }
    
    return results


def plot_convergence(
    cost_history: List[float],
    title: str = "ALNS Convergence",
    save_path: Optional[str] = None,
) -> None:
    """Plot cost convergence over iterations.
    
    Args:
        cost_history: List of best costs at each recording point
        title: Plot title
        save_path: Path to save figure (None = show)
    """
    if not _HAS_PLOTTING:
        print("matplotlib/seaborn not installed for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(cost_history, linewidth=2, color='#2ecc71')
    ax.set_xlabel('Iteration (segments)')
    ax.set_ylabel('Best Cost')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_pareto_front(
    costs: List[float],
    fairness: List[float],
    labels: Optional[List[str]] = None,
    title: str = "Cost-Fairness Pareto Front",
    save_path: Optional[str] = None,
) -> None:
    """Plot Pareto front of cost vs fairness.
    
    Args:
        costs: Total route costs
        fairness: Fairness metric values (e.g., CV - lower is better)
        labels: Optional labels for each point
        title: Plot title
        save_path: Path to save figure
    """
    if not _HAS_PLOTTING:
        print("matplotlib/seaborn not installed for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points
    scatter = ax.scatter(costs, fairness, s=100, alpha=0.7, c='#3498db')
    
    # Identify and highlight Pareto front
    pareto_mask = _get_pareto_mask(costs, fairness)
    pareto_costs = [c for c, m in zip(costs, pareto_mask) if m]
    pareto_fair = [f for f, m in zip(fairness, pareto_mask) if m]
    
    # Sort for line plot
    sorted_pairs = sorted(zip(pareto_costs, pareto_fair))
    if sorted_pairs:
        pc, pf = zip(*sorted_pairs)
        ax.plot(pc, pf, 'r-', linewidth=2, label='Pareto Front')
        ax.scatter(pc, pf, s=150, c='#e74c3c', marker='*', label='Pareto Optimal')
    
    # Labels
    if labels:
        for i, (c, f, l) in enumerate(zip(costs, fairness, labels)):
            ax.annotate(l, (c, f), textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel('Total Cost')
    ax.set_ylabel('Driver CV (lower = more fair)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def _get_pareto_mask(costs: List[float], fairness: List[float]) -> List[bool]:
    """Identify Pareto-optimal points (minimizing both objectives)."""
    n = len(costs)
    mask = [True] * n
    
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            # j dominates i if j is better in both objectives
            if costs[j] <= costs[i] and fairness[j] <= fairness[i]:
                if costs[j] < costs[i] or fairness[j] < fairness[i]:
                    mask[i] = False
                    break
    
    return mask


def plot_operator_usage(
    destroy_usage: Dict[str, int],
    repair_usage: Dict[str, int],
    title: str = "Operator Usage",
    save_path: Optional[str] = None,
) -> None:
    """Plot operator usage statistics.
    
    Args:
        destroy_usage: Dict of destroy operator -> count
        repair_usage: Dict of repair operator -> count
        title: Plot title
        save_path: Path to save figure
    """
    if not _HAS_PLOTTING:
        print("matplotlib/seaborn not installed for plotting")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Destroy operators
    if destroy_usage:
        names = list(destroy_usage.keys())
        counts = list(destroy_usage.values())
        axes[0].barh(names, counts, color='#e74c3c')
        axes[0].set_xlabel('Usage Count')
        axes[0].set_title('Destroy Operators')
    
    # Repair operators
    if repair_usage:
        names = list(repair_usage.keys())
        counts = list(repair_usage.values())
        axes[1].barh(names, counts, color='#2ecc71')
        axes[1].set_xlabel('Usage Count')
        axes[1].set_title('Repair Operators')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def generate_report(
    results: List[Dict],
    output_path: str = "report.md",
) -> str:
    """Generate a markdown report from results.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save report
        
    Returns:
        Report as markdown string
    """
    aggregated = aggregate_by_instance(results)
    
    lines = [
        "# HNSW-FairVRP Experiment Report",
        "",
        f"**Total Runs:** {len(results)}",
        f"**Instances:** {len(aggregated)}",
        "",
        "## Summary by Instance",
        "",
        "| Instance | Runs | Initial | Final | Improvement | CV | Runtime |",
        "|----------|------|---------|-------|-------------|-----|---------|",
    ]
    
    for name, agg in aggregated.items():
        lines.append(
            f"| {name} | {agg.n_runs} | {agg.mean_initial_cost:.2f} | "
            f"{agg.mean_final_cost:.2f} ± {agg.std_final_cost:.2f} | "
            f"{agg.mean_improvement:.2%} | {agg.mean_driver_cv:.4f} | "
            f"{agg.mean_runtime:.2f}s |"
        )
    
    lines.extend([
        "",
        "## Overall Statistics",
        "",
    ])
    
    all_improvements = [r['improvement'] for r in results]
    all_runtimes = [r['runtime'] for r in results]
    
    lines.extend([
        f"- **Mean Improvement:** {np.mean(all_improvements):.2%} ± {np.std(all_improvements):.2%}",
        f"- **Mean Runtime:** {np.mean(all_runtimes):.2f}s ± {np.std(all_runtimes):.2f}s",
        "",
    ])
    
    report = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    return report

