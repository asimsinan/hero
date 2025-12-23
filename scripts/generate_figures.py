#!/usr/bin/env python3
"""
Generate figures for HERO paper results.

This script generates all essential figures for the IEEE T-ITS paper:
- Speedup vs Instance Size
- Scalability Analysis (log-log)
- Fairness Improvement Bar Chart
- Cost-Fairness Trade-off (Pareto Fronts)
- Ablation Study Results
- Convergence Curves

Usage:
    python scripts/generate_figures.py --results-dir results/benchmark --output-dir figures/
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import pandas as pd
from scipy import stats

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure matplotlib for IEEE paper style
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON result files from directory."""
    results = []
    results_dir = Path(results_dir)
    
    # Find all JSON files
    json_files = list(results_dir.glob("**/*.json"))
    if not json_files:
        # Try CSV as fallback
        csv_files = list(results_dir.glob("**/*.csv"))
        if csv_files:
            df = pd.read_csv(csv_files[0])
            return df.to_dict('records')
        raise ValueError(f"No JSON or CSV files found in {results_dir}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results


def aggregate_by_method(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by method."""
    grouped = defaultdict(list)
    for r in results:
        method = r.get('method', 'unknown')
        grouped[method].append(r)
    return dict(grouped)


def figure_speedup_vs_size(results: List[Dict[str, Any]], output_path: Path):
    """Generate speedup vs instance size figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Group by instance size and method
    by_size = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        n_customers = r.get('n_customers', 0)
        method = r.get('method', '')
        time_seconds = r.get('time_seconds', 0)
        
        if n_customers > 0 and time_seconds > 0:
            by_size[n_customers][method].append(time_seconds)
    
    # Calculate speedup relative to ALNS
    sizes = sorted([s for s in by_size.keys() if 'alns' in by_size[s]])
    if not sizes:
        print("Warning: No ALNS baseline found for speedup calculation")
        return
    
    # Get ALNS baseline times
    alns_times = {}
    for size in sizes:
        if 'alns' in by_size[size]:
            alns_times[size] = np.mean(by_size[size]['alns'])
    
    # Plot speedup for HNSW and HERO
    methods_to_plot = [
        ('alns_hnsw', 'ALNS-HNSW', '--', 'blue'),
        ('hero', 'HERO', '-', 'red'),
    ]
    
    for method_key, method_label, linestyle, color in methods_to_plot:
        x_vals = []
        y_vals = []
        y_err_lower = []
        y_err_upper = []
        
        for size in sizes:
            if method_key in by_size[size] and size in alns_times:
                method_times = by_size[size][method_key]
                if method_times:
                    speedup = alns_times[size] / np.mean(method_times)
                    std_speedup = speedup * (np.std(method_times) / np.mean(method_times))
                    
                    x_vals.append(size)
                    y_vals.append(speedup)
                    y_err_lower.append(max(0, speedup - std_speedup))
                    y_err_upper.append(speedup + std_speedup)
        
        if x_vals:
            ax.errorbar(x_vals, y_vals, 
                       yerr=[np.array(y_vals) - np.array(y_err_lower),
                             np.array(y_err_upper) - np.array(y_vals)],
                       label=method_label, linestyle=linestyle, 
                       color=color, marker='o', markersize=6,
                       capsize=3, capthick=1.5, linewidth=2)
    
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Instance Size (Number of Customers)', fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontweight='bold')
    ax.set_title('Speedup of HERO over Classical ALNS', fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log', base=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'speedup_vs_size.pdf', format='pdf')
    plt.savefig(output_path / 'speedup_vs_size.png', format='png')
    print(f"Generated: {output_path / 'speedup_vs_size.pdf'}")
    plt.close()


def figure_scalability(results: List[Dict[str, Any]], output_path: Path):
    """Generate scalability analysis (log-log plot)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Group by instance size and method
    by_size = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        n_customers = r.get('n_customers', 0)
        method = r.get('method', '')
        time_seconds = r.get('time_seconds', 0)
        
        if n_customers > 0 and time_seconds > 0:
            by_size[n_customers][method].append(time_seconds)
    
    sizes = sorted(by_size.keys())
    
    methods_to_plot = [
        ('alns', 'ALNS', 'o', 'black'),
        ('alns_hnsw', 'ALNS-HNSW', 's', 'blue'),
        ('hero', 'HERO', '^', 'red'),
    ]
    
    for method_key, method_label, marker, color in methods_to_plot:
        x_vals = []
        y_vals_mean = []
        y_vals_std = []
        
        for size in sizes:
            if method_key in by_size[size]:
                times = by_size[size][method_key]
                if times:
                    x_vals.append(size)
                    y_vals_mean.append(np.mean(times))
                    y_vals_std.append(np.std(times))
        
        if x_vals:
            ax.errorbar(x_vals, y_vals_mean, yerr=y_vals_std,
                       label=method_label, marker=marker, color=color,
                       markersize=6, capsize=3, capthick=1.5,
                       linewidth=2, linestyle='-', alpha=0.8)
    
    ax.set_xlabel('Instance Size (Number of Customers)', fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontweight='bold')
    ax.set_title('Scalability Analysis: Runtime vs Instance Size', fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'scalability.pdf', format='pdf')
    plt.savefig(output_path / 'scalability.png', format='png')
    print(f"Generated: {output_path / 'scalability.pdf'}")
    plt.close()


def figure_fairness_improvement(results: List[Dict[str, Any]], output_path: Path):
    """Generate fairness improvement bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Group by instance
    by_instance = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        instance = r.get('instance_name', '')
        method = r.get('method', '')
        cv = r.get('cv', 0)
        
        if instance and cv > 0:
            by_instance[instance][method].append(cv)
    
    # Calculate improvements
    instances = []
    improvements = []
    errors = []
    methods_compare = [('alns_hnsw', 'ALNS-HNSW'), ('alns', 'ALNS')]
    
    for instance in sorted(by_instance.keys()):
        if 'hero' in by_instance[instance]:
            hero_cv = np.mean(by_instance[instance]['hero'])
            
            for method_key, method_label in methods_compare:
                if method_key in by_instance[instance]:
                    baseline_cv = np.mean(by_instance[instance][method_key])
                    if baseline_cv > 0:
                        improvement = ((baseline_cv - hero_cv) / baseline_cv) * 100
                        std_improvement = (np.std(by_instance[instance][method_key]) / baseline_cv) * 100
                        
                        instances.append(f"{instance}\nvs {method_label}")
                        improvements.append(improvement)
                        errors.append(std_improvement)
    
    if not instances:
        print("Warning: No fairness data found")
        return
    
    # Create bar chart
    x_pos = np.arange(len(instances))
    colors = ['steelblue' if 'HNSW' in inst else 'darkorange' for inst in instances]
    
    bars = ax.bar(x_pos, improvements, yerr=errors, 
                  color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                  capsize=5, capthick=2, error_kw={'elinewidth': 2})
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Instance and Baseline Comparison', fontweight='bold')
    ax.set_ylabel('CV Improvement (%)', fontweight='bold')
    ax.set_title('Fairness Improvement: HERO vs Baselines', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(instances, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add legend
    hns_patch = mpatches.Patch(color='steelblue', label='vs HNSW')
    alns_patch = mpatches.Patch(color='darkorange', label='vs ALNS')
    ax.legend(handles=[hns_patch, alns_patch], loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path / 'fairness_improvement.pdf', format='pdf')
    plt.savefig(output_path / 'fairness_improvement.png', format='png')
    print(f"Generated: {output_path / 'fairness_improvement.pdf'}")
    plt.close()


def figure_pareto_fronts(results: List[Dict[str, Any]], output_path: Path):
    """Generate cost-fairness trade-off (Pareto fronts)."""
    # Select representative instances
    instances_to_plot = ['C101', 'R101', 'RC101', 'R201']
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for idx, instance_name in enumerate(instances_to_plot):
        ax = axes[idx]
        
        # Filter results for this instance
        instance_results = [r for r in results if r.get('instance_name', '').startswith(instance_name)]
        
        if not instance_results:
            ax.text(0.5, 0.5, f'No data for {instance_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(instance_name, fontweight='bold')
            continue
        
        # Group by method
        by_method = defaultdict(list)
        for r in instance_results:
            method = r.get('method', '')
            cost = r.get('cost', 0)
            cv = r.get('cv', 0)
            if cost > 0 and cv >= 0:
                by_method[method].append((cost, cv))
        
        # Plot each method
        method_styles = {
            'alns': ('ALNS', 'o', 'black', 0.6),
            'alns_hnsw': ('ALNS-HNSW', 's', 'blue', 0.6),
            'hero': ('HERO', '^', 'red', 0.8),
            'ortools': ('OR-Tools', 'D', 'green', 0.6),
            'pyvrp': ('PyVRP', 'p', 'purple', 0.6),
        }
        
        for method_key, (method_label, marker, color, alpha) in method_styles.items():
            if method_key in by_method:
                points = by_method[method_key]
                costs = [p[0] for p in points]
                cvs = [p[1] for p in points]
                
                # Normalize costs to [0, 1] for comparison
                if costs:
                    min_cost = min(costs)
                    max_cost = max(costs)
                    if max_cost > min_cost:
                        costs_norm = [(c - min_cost) / (max_cost - min_cost) for c in costs]
                    else:
                        costs_norm = [0.5] * len(costs)
                    
                    ax.scatter(costs_norm, cvs, label=method_label, 
                             marker=marker, color=color, s=80, alpha=alpha,
                             edgecolors='black', linewidths=1)
        
        ax.set_xlabel('Normalized Cost', fontweight='bold')
        ax.set_ylabel('CV (Coefficient of Variation)', fontweight='bold')
        ax.set_title(instance_name, fontweight='bold')
        ax.legend(loc='best', fontsize=8, frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.invert_xaxis()  # Lower cost is better (left side)
    
    plt.suptitle('Cost-Fairness Trade-off: Pareto Front Analysis', 
                fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path / 'pareto_fronts.pdf', format='pdf')
    plt.savefig(output_path / 'pareto_fronts.png', format='png')
    print(f"Generated: {output_path / 'pareto_fronts.pdf'}")
    plt.close()


def figure_ablation_study(results: List[Dict[str, Any]], output_path: Path):
    """Generate ablation study results."""
    # This requires specific ablation study results
    # For now, we'll create a placeholder structure
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Component contributions (example structure)
    components = ['HNSW\nAcceleration', 'Fairness\nFeatures', 'Subsequence\nFeatures', 'Adaptive\nk']
    
    # Speedup contribution
    speedup_contrib = [0.4, 0.0, 0.1, 0.1]  # Placeholder - should come from ablation results
    ax1.bar(components, speedup_contrib, color='steelblue', alpha=0.7, 
           edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Speedup Contribution (×)', fontweight='bold')
    ax1.set_title('Component Contribution to Speedup', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.set_ylim(0, max(speedup_contrib) * 1.2)
    
    # Fairness improvement contribution
    fairness_contrib = [0.0, 0.15, 0.05, 0.02]  # Placeholder
    ax2.bar(components, fairness_contrib, color='darkgreen', alpha=0.7,
           edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('CV Improvement', fontweight='bold')
    ax2.set_title('Component Contribution to Fairness', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_ylim(0, max(fairness_contrib) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path / 'ablation_study.pdf', format='pdf')
    plt.savefig(output_path / 'ablation_study.png', format='png')
    print(f"Generated: {output_path / 'ablation_study.pdf'}")
    print("Note: Ablation study uses placeholder data. Update with actual ablation results.")
    plt.close()


def figure_convergence_curves(results: List[Dict[str, Any]], output_path: Path):
    """Generate convergence curves over iterations."""
    # This requires iteration-by-iteration data
    # For now, we'll create a structure that can be populated
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    iterations = np.arange(0, 201, 10)
    
    methods_curves = {
        'ALNS': (iterations, 1.0 - 0.3 * (1 - np.exp(-iterations/50)), 'black', '-'),
        'HNSW': (iterations, 1.0 - 0.28 * (1 - np.exp(-iterations/45)), 'blue', '--'),
        'HERO': (iterations, 1.0 - 0.25 * (1 - np.exp(-iterations/40)), 'red', '-'),
    }
    
    for method_label, (iters, costs, color, linestyle) in methods_curves.items():
        ax.plot(iters, costs, label=method_label, color=color, 
               linestyle=linestyle, linewidth=2, marker='o', markersize=4, alpha=0.8)
    
    ax.set_xlabel('ALNS Iterations', fontweight='bold')
    ax.set_ylabel('Normalized Cost', fontweight='bold')
    ax.set_title('Convergence Curves: Solution Quality over Iterations', fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 200)
    
    plt.tight_layout()
    plt.savefig(output_path / 'convergence_curves.pdf', format='pdf')
    plt.savefig(output_path / 'convergence_curves.png', format='png')
    print(f"Generated: {output_path / 'convergence_curves.pdf'}")
    print("Note: Convergence curves use example data. Update with actual iteration history.")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate figures for HERO paper")
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing experimental results (JSON/CSV)')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for generated figures')
    parser.add_argument('--figures', type=str, nargs='+',
                       choices=['all', 'speedup', 'scalability', 'fairness', 
                               'pareto', 'ablation', 'convergence'],
                       default=['all'],
                       help='Which figures to generate')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {results_dir}...")
    try:
        results = load_results(results_dir)
        print(f"Loaded {len(results)} result entries")
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    if not results:
        print("No results found. Please check the results directory.")
        return
    
    # Generate figures
    figures_to_generate = args.figures
    if 'all' in figures_to_generate:
        figures_to_generate = ['speedup', 'scalability', 'fairness', 'pareto', 'ablation', 'convergence']
    
    print(f"\nGenerating figures: {', '.join(figures_to_generate)}")
    
    if 'speedup' in figures_to_generate:
        figure_speedup_vs_size(results, output_dir)
    
    if 'scalability' in figures_to_generate:
        figure_scalability(results, output_dir)
    
    if 'fairness' in figures_to_generate:
        figure_fairness_improvement(results, output_dir)
    
    if 'pareto' in figures_to_generate:
        figure_pareto_fronts(results, output_dir)
    
    if 'ablation' in figures_to_generate:
        figure_ablation_study(results, output_dir)
    
    if 'convergence' in figures_to_generate:
        figure_convergence_curves(results, output_dir)
    
    print(f"\nAll figures generated in {output_dir}/")


if __name__ == '__main__':
    main()

