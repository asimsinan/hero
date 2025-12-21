#!/usr/bin/env python3
"""
Generate Figure 4: Pareto Front Analysis showing cost-fairness trade-off.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

def load_results(results_file: Path):
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def plot_single_instance(ax, instance_results, instance_name):
    """Plot Pareto front for a single instance."""
    # Group by method
    by_method = defaultdict(list)
    for r in instance_results:
        method = r.get('method', '')
        cost = r.get('cost', 0)
        cv = r.get('cv', 0)
        if cost > 0 and cost != float('inf') and cv >= 0:
            by_method[method].append((cost, cv))
    
    # Method styles
    method_styles = {
        'alns': ('ALNS', 'o', 'black', 0.7, 80),
        'alns_hnsw': ('ALNS-HNSW', 's', 'blue', 0.7, 80),
        'hero': ('HERO', '^', 'red', 0.8, 100),
        'ortools': ('OR-Tools', 'D', 'green', 0.7, 80),
        'pyvrp': ('PyVRP', 'p', 'purple', 0.7, 80),
    }
    
    # Plot each method
    for method_key, (method_label, marker, color, alpha, size) in method_styles.items():
        if method_key in by_method:
            points = by_method[method_key]
            costs = [p[0] for p in points]
            cvs = [p[1] for p in points]
            
            if costs and cvs:
                # Calculate mean and std for error bars
                if len(costs) > 1:
                    mean_cost = np.mean(costs)
                    std_cost = np.std(costs)
                    mean_cv = np.mean(cvs)
                    std_cv = np.std(cvs)
                    
                    ax.errorbar(mean_cost, mean_cv, 
                               xerr=std_cost, yerr=std_cv,
                               label=method_label, marker=marker, 
                               color=color, markersize=size//10, 
                               alpha=alpha, capsize=3, capthick=1.5,
                               linewidth=2, linestyle='None')
                else:
                    # Single point
                    ax.scatter(costs[0], cvs[0], label=method_label,
                             marker=marker, color=color, s=size, 
                             alpha=alpha, edgecolors='black', linewidths=1)
    
    ax.set_xlabel('Cost', fontweight='bold')
    ax.set_ylabel('CV (Coefficient of Variation)', fontweight='bold')
    ax.set_title(instance_name, fontweight='bold')
    ax.legend(loc='best', fontsize=8, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

def generate_pareto_figure(results, output_path: Path):
    """Generate Pareto front figure with two subfigures."""
    # Find available instances
    instances = set()
    for r in results:
        inst = r.get('instance_name', '')
        if inst:
            instances.add(inst)
    
    # Select instances: C101 and best alternative
    instance1 = 'C101'
    instance2 = None
    
    # Try to find R201, otherwise use another C-type
    if any('R201' in inst for inst in instances):
        instance2 = 'R201'
    else:
        # Use another C-type (C102 is a good representative)
        c_instances = [inst for inst in instances if inst.startswith('C') and len(inst) <= 5 and inst != 'C101']
        if c_instances:
            instance2 = sorted(c_instances)[0]  # C102
        else:
            instance2 = sorted(instances)[1] if len(instances) > 1 else instance1
    
    print(f"Generating Pareto front for instances: {instance1} and {instance2}")
    
    # Create combined figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    instances_to_plot = [instance1, instance2]
    
    for idx, instance_name in enumerate(instances_to_plot):
        ax = axes[idx]
        
        # Filter results for this instance
        instance_results = [r for r in results 
                          if r.get('instance_name', '').startswith(instance_name)]
        
        if not instance_results:
            ax.text(0.5, 0.5, f'No data for {instance_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(instance_name, fontweight='bold')
            continue
        
        plot_single_instance(ax, instance_results, instance_name)
    
    plt.suptitle('Cost-Fairness Trade-off: Pareto Front Analysis', 
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save as both PDF and PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path.with_suffix('.pdf'), format='pdf')
    plt.savefig(output_path.with_suffix('.png'), format='png')
    print(f"Generated: {output_path.with_suffix('.pdf')}")
    print(f"Generated: {output_path.with_suffix('.png')}")
    plt.close()

if __name__ == '__main__':
    # Load results
    results_file = Path("results/main_benchmark_10seeds/results_20251221_193921.json")
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        sys.exit(1)
    
    results = load_results(results_file)
    print(f"Loaded {len(results)} results")
    
    # Generate figure
    script_dir = Path(__file__).parent.parent.parent
    output_path = script_dir / "IEEE-Transactions-paper" / "figures" / "fig4"
    generate_pareto_figure(results, output_path)
    print("Figure 4 generated successfully!")

