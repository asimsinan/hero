#!/usr/bin/env python3
"""Visualization utilities for VRP solutions.

Provides functions to visualize:
- Routes on a 2D map
- Convergence curves
- Fairness distributions
- Comparison charts
"""
from __future__ import annotations

from typing import List, Optional, Dict
from pathlib import Path
import json

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba
    _HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    _HAS_MATPLOTLIB = False


def check_matplotlib():
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")


def plot_routes(
    depot_coords: tuple,
    customer_coords: List[tuple],
    routes: List[List[int]],
    title: str = "VRP Solution",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10),
) -> None:
    """Plot VRP routes on a 2D map.
    
    Args:
        depot_coords: (x, y) of depot
        customer_coords: List of (x, y) for each customer (0-indexed)
        routes: List of routes, each route is a list of customer indices (0-indexed)
        title: Plot title
        save_path: Path to save figure (None = show)
        figsize: Figure size
    """
    check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    colors = plt.cm.tab10.colors
    
    # Plot depot
    ax.scatter(depot_coords[0], depot_coords[1], c='red', s=200, marker='s', 
               label='Depot', zorder=5, edgecolors='black', linewidths=2)
    
    # Plot customers and routes
    for route_idx, route in enumerate(routes):
        if not route:
            continue
        
        color = colors[route_idx % len(colors)]
        
        # Build path: depot -> customers -> depot
        path_x = [depot_coords[0]]
        path_y = [depot_coords[1]]
        
        for cust_idx in route:
            if cust_idx < len(customer_coords):
                x, y = customer_coords[cust_idx]
                path_x.append(x)
                path_y.append(y)
        
        path_x.append(depot_coords[0])
        path_y.append(depot_coords[1])
        
        # Plot route line
        ax.plot(path_x, path_y, c=color, linewidth=2, alpha=0.7)
        
        # Plot customers in this route
        for i, cust_idx in enumerate(route):
            if cust_idx < len(customer_coords):
                x, y = customer_coords[cust_idx]
                ax.scatter(x, y, c=[color], s=100, zorder=4, edgecolors='black', linewidths=1)
                ax.annotate(str(cust_idx + 1), (x, y), textcoords="offset points",
                           xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Legend
    handles = [mpatches.Patch(color=colors[i], label=f'Route {i+1}') 
               for i in range(min(len(routes), len(colors)))]
    handles.insert(0, mpatches.Patch(color='red', label='Depot'))
    ax.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved route visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_route_loads(
    route_loads: List[float],
    capacity: float,
    title: str = "Route Load Distribution",
    save_path: Optional[str] = None,
) -> None:
    """Plot route load distribution.
    
    Args:
        route_loads: Load for each route
        capacity: Vehicle capacity
        title: Plot title
        save_path: Path to save figure
    """
    check_matplotlib()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_routes = len(route_loads)
    x = np.arange(n_routes)
    
    # Color based on utilization
    utilizations = [load / capacity for load in route_loads]
    colors = ['#2ecc71' if u < 0.8 else '#f39c12' if u < 0.95 else '#e74c3c' for u in utilizations]
    
    bars = ax.bar(x, route_loads, color=colors, edgecolor='black')
    ax.axhline(y=capacity, color='red', linestyle='--', linewidth=2, label=f'Capacity ({capacity})')
    
    ax.set_xlabel('Route')
    ax.set_ylabel('Load')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'R{i+1}' for i in range(n_routes)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add utilization labels
    for bar, util in zip(bars, utilizations):
        height = bar.get_height()
        ax.annotate(f'{util:.0%}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_fairness_comparison(
    results: List[Dict],
    metric: str = 'driver_cv',
    title: str = "Fairness Comparison",
    save_path: Optional[str] = None,
) -> None:
    """Plot fairness metric comparison across experiments.
    
    Args:
        results: List of experiment result dicts
        metric: Fairness metric to plot ('driver_cv', 'customer_jain')
        title: Plot title
        save_path: Path to save figure
    """
    check_matplotlib()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    names = [r.get('instance_name', f'Run {i}') for i, r in enumerate(results)]
    values = [r.get(metric, 0) for r in results]
    
    x = np.arange(len(names))
    
    color = '#3498db' if metric == 'driver_cv' else '#2ecc71'
    bars = ax.bar(x, values, color=color, edgecolor='black')
    
    ax.set_xlabel('Instance')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_comparison_bars(
    methods: List[str],
    costs: List[float],
    cvs: List[float],
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
) -> None:
    """Plot side-by-side comparison of methods.
    
    Args:
        methods: Method names
        costs: Costs for each method
        cvs: CV values for each method
        title: Plot title
        save_path: Path to save figure
    """
    check_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(methods))
    width = 0.6
    
    # Cost comparison
    axes[0].bar(x, costs, width, color='#3498db', edgecolor='black')
    axes[0].set_ylabel('Total Cost')
    axes[0].set_title('Cost Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # CV comparison
    axes[1].bar(x, cvs, width, color='#e74c3c', edgecolor='black')
    axes[1].set_ylabel('Driver CV')
    axes[1].set_title('Fairness Comparison (lower = better)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_from_solution_file(solution_path: str, instance_path: Optional[str] = None) -> None:
    """Visualize solution from a saved JSON file.
    
    Args:
        solution_path: Path to solution JSON file
        instance_path: Optional path to instance file for customer coordinates
    """
    with open(solution_path, 'r') as f:
        solution_data = json.load(f)
    
    routes = [r['customers'] for r in solution_data.get('routes', [])]
    
    # If no instance provided, create dummy coordinates
    if instance_path:
        # Load instance coordinates
        # This would need proper parsing based on file format
        pass
    
    # For now, just print route info
    print(f"Solution from {solution_path}")
    print(f"Total cost: {solution_data.get('total_cost', 'N/A')}")
    print(f"Number of routes: {len(routes)}")
    for i, route in enumerate(routes):
        if route:
            print(f"  Route {i+1}: {len(route)} customers")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        visualize_from_solution_file(sys.argv[1])
    else:
        print("Usage: python visualize_solution.py <solution.json>")

