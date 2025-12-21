#!/usr/bin/env python3
"""
Generate Figure 3: Speedup of HERO over Classical ALNS across instance sizes.
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

def generate_speedup_figure(results, output_path: Path):
    """Generate speedup vs instance size figure."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Group by instance size and method
    by_size = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        n_customers = r.get('n_customers', 0)
        method = r.get('method', '')
        time_seconds = r.get('time_seconds', 0)
        
        # Filter valid results
        if n_customers > 0 and time_seconds > 0 and time_seconds != float('inf'):
            by_size[n_customers][method].append(time_seconds)
    
    # Calculate speedup relative to ALNS
    sizes = sorted([s for s in by_size.keys() if 'alns' in by_size[s] and by_size[s]['alns']])
    
    if not sizes:
        print("Warning: No ALNS baseline found for speedup calculation")
        return
    
    # Get ALNS baseline times
    alns_times = {}
    for size in sizes:
        if 'alns' in by_size[size] and by_size[size]['alns']:
            alns_times[size] = np.mean(by_size[size]['alns'])
    
    # Plot speedup for HERO only (as per paper caption)
    x_vals = []
    y_vals = []
    y_err_lower = []
    y_err_upper = []
    
    for size in sizes:
        if 'hero' in by_size[size] and size in alns_times:
            hero_times = by_size[size]['hero']
            if hero_times:
                hero_mean = np.mean(hero_times)
                hero_std = np.std(hero_times)
                speedup = alns_times[size] / hero_mean if hero_mean > 0 else 0
                
                # Calculate error bars using propagation of uncertainty
                if hero_mean > 0:
                    speedup_std = speedup * (hero_std / hero_mean)
                else:
                    speedup_std = 0
                
                # Cap error bars to prevent visual artifacts (max 2x the speedup value)
                speedup_std = min(speedup_std, speedup * 2.0)
                
                x_vals.append(size)
                y_vals.append(speedup)
                y_err_lower.append(max(0, speedup - speedup_std))
                y_err_upper.append(min(speedup + speedup_std, speedup * 3.0))  # Cap upper bound
    
    if x_vals:
        ax.errorbar(x_vals, y_vals, 
                   yerr=[np.array(y_vals) - np.array(y_err_lower),
                         np.array(y_err_upper) - np.array(y_vals)],
                   label='HERO', linestyle='-', 
                   color='red', marker='o', markersize=6,
                   capsize=3, capthick=1.5, linewidth=2, alpha=0.8)
    
    # Add reference line at 1.0x - make it more visible with contrasting color
    ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=2.5, alpha=0.8, label='Baseline (1.0×)', zorder=1)
    
    ax.set_xlabel('Instance Size (Number of Customers)', fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontweight='bold')
    ax.set_title('Speedup of HERO over Classical ALNS', fontweight='bold')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Use log scale for x-axis to better show the range
    ax.set_xscale('log', base=10)
    
    # Set y-axis limits to prevent visual artifacts
    # Cap at reasonable maximum (3x speedup) unless data exceeds it
    max_y = max(y_vals) if y_vals else 3.0
    ax.set_ylim(bottom=0, top=max(max_y * 1.2, 3.5))  # Add 20% padding, but cap at 3.5
    
    plt.tight_layout()
    
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
    
    # Generate figure directly in figures directory
    # Get the script's directory and navigate to paper directory
    script_dir = Path(__file__).parent.parent.parent  # Go up from scripts/ to project root
    output_path = script_dir / "IEEE-Transactions-paper" / "figures" / "fig3"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_speedup_figure(results, output_path)
    print(f"Figure 3 generated at: {output_path}")

