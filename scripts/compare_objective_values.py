#!/usr/bin/env python3
"""Compare objective function values across methods.

This script analyzes experimental results to compare:
- Cost (total route cost)
- CV (coefficient of variation - driver fairness)
- Combined objective (α×cost + β×CV - γ×Jain)

Usage:
    python scripts/compare_objective_values.py --results-dir results/benchmark --output results/objective_comparison.json
"""
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fairness import coefficient_of_variation, jains_fairness_index


def compute_objective_components(solution, instance, alpha: float = 1.0, beta: float = 0.2, gamma: float = 0.0) -> Dict:
    """Compute objective function components for a solution.
    
    Args:
        solution: VRP solution
        instance: VRP instance
        alpha: Cost weight
        beta: CV weight
        gamma: Jain's index weight
        
    Returns:
        Dictionary with cost, CV, jain, and combined objective
    """
    # Cost component - per-customer average
    cost_per_customer = solution.total_cost / max(instance.n_customers, 1)
    
    # Driver fairness (CV of route costs)
    route_costs = np.array([r.cost for r in solution.routes if not r.is_empty()])
    if len(route_costs) > 1:
        cv = coefficient_of_variation(route_costs)
        jain = jains_fairness_index(route_costs)
    else:
        cv = 0.0
        jain = 1.0
    
    # Combined objective (matching ALNS._compute_objective)
    # Note: ALNS uses cost_per_customer, not total_cost
    combined = alpha * cost_per_customer + beta * cv - gamma * jain
    
    return {
        "total_cost": solution.total_cost,
        "cost_per_customer": cost_per_customer,
        "cv": cv,
        "jain": jain,
        "combined_objective": combined,
        "n_routes": len([r for r in solution.routes if not r.is_empty()]),
    }


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load results from JSON files."""
    results_dir = Path(results_dir)
    
    all_results = []
    
    # Find all JSON result files
    json_files = list(results_dir.glob("**/*results*.json"))
    if not json_files:
        json_files = list(results_dir.glob("**/*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
                elif isinstance(data, dict):
                    if 'results' in data:
                        all_results.extend(data['results'])
                    else:
                        all_results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    if not all_results:
        raise ValueError(f"No results found in {results_dir}")
    
    return pd.DataFrame(all_results)


def analyze_objective_values(df: pd.DataFrame, alpha: float = 1.0, beta: float = 0.2, gamma: float = 0.0) -> Dict:
    """Analyze objective values across methods.
    
    Args:
        df: DataFrame with experimental results
        alpha: Cost weight
        beta: CV weight  
        gamma: Jain's index weight
        
    Returns:
        Dictionary with analysis results
    """
    # Filter valid results
    valid = df[
        (df['cost'] > 0) & 
        (df['cost'] != float('inf')) & 
        (df['cv'] >= 0)
    ].copy()
    
    if len(valid) == 0:
        return {"error": "No valid results found"}
    
    # Group by method
    methods = valid['method'].unique()
    
    analysis = {
        "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "methods": {},
        "comparisons": {},
    }
    
    # Compute statistics for each method
    for method in methods:
        method_data = valid[valid['method'] == method]
        
        # Compute combined objective for each result
        # Note: We need to compute this from cost and CV
        # For now, use cost_per_customer approximation
        method_data = method_data.copy()
        method_data['cost_per_customer'] = method_data['cost'] / method_data['n_customers'].clip(lower=1)
        method_data['combined_objective'] = (
            alpha * method_data['cost_per_customer'] + 
            beta * method_data['cv'] - 
            gamma * 0.8  # Approximate jain as 0.8 (will be computed properly if solution available)
        )
        
        analysis["methods"][method] = {
            "count": len(method_data),
            "cost": {
                "mean": float(method_data['cost'].mean()),
                "std": float(method_data['cost'].std()),
                "min": float(method_data['cost'].min()),
                "max": float(method_data['cost'].max()),
            },
            "cv": {
                "mean": float(method_data['cv'].mean()),
                "std": float(method_data['cv'].std()),
                "min": float(method_data['cv'].min()),
                "max": float(method_data['cv'].max()),
            },
            "combined_objective": {
                "mean": float(method_data['combined_objective'].mean()),
                "std": float(method_data['combined_objective'].std()),
                "min": float(method_data['combined_objective'].min()),
                "max": float(method_data['combined_objective'].max()),
            },
        }
    
    # Pairwise comparisons
    method_list = list(methods)
    for i, method1 in enumerate(method_list):
        for method2 in method_list[i+1:]:
            data1 = valid[valid['method'] == method1]
            data2 = valid[valid['method'] == method2]
            
            if len(data1) == 0 or len(data2) == 0:
                continue
            
            # Match by instance and seed for fair comparison
            comparison_key = f"{method1}_vs_{method2}"
            
            # Merge on instance_name and seed
            merged = pd.merge(
                data1[['instance_name', 'seed', 'cost', 'cv', 'n_customers']],
                data2[['instance_name', 'seed', 'cost', 'cv', 'n_customers']],
                on=['instance_name', 'seed'],
                suffixes=('_1', '_2')
            )
            
            if len(merged) == 0:
                continue
            
            # Compute combined objectives
            merged['cost_per_customer_1'] = merged['cost_1'] / merged['n_customers_1'].clip(lower=1)
            merged['cost_per_customer_2'] = merged['cost_2'] / merged['n_customers_2'].clip(lower=1)
            merged['combined_1'] = alpha * merged['cost_per_customer_1'] + beta * merged['cv_1'] - gamma * 0.8
            merged['combined_2'] = alpha * merged['cost_per_customer_2'] + beta * merged['cv_2'] - gamma * 0.8
            
            # Cost comparison
            cost_diff = merged['cost_2'] - merged['cost_1']
            cost_improvement_pct = (cost_diff / merged['cost_1'] * 100).mean()
            
            # CV comparison
            cv_diff = merged['cv_2'] - merged['cv_1']
            cv_improvement_pct = (cv_diff / merged['cv_1'] * 100).mean() if merged['cv_1'].mean() > 0 else 0
            
            # Combined objective comparison
            obj_diff = merged['combined_2'] - merged['combined_1']
            obj_improvement_pct = (obj_diff / merged['combined_1'] * 100).mean()
            
            # Statistical tests
            cost_stat, cost_p = stats.wilcoxon(merged['cost_1'], merged['cost_2'], alternative='two-sided')
            cv_stat, cv_p = stats.wilcoxon(merged['cv_1'], merged['cv_2'], alternative='two-sided')
            obj_stat, obj_p = stats.wilcoxon(merged['combined_1'], merged['combined_2'], alternative='two-sided')
            
            analysis["comparisons"][comparison_key] = {
                "n_pairs": len(merged),
                "cost": {
                    "improvement_pct": float(cost_improvement_pct),
                    "mean_diff": float(cost_diff.mean()),
                    "wilcoxon_stat": float(cost_stat),
                    "wilcoxon_p": float(cost_p),
                    "significant": cost_p < 0.05,
                },
                "cv": {
                    "improvement_pct": float(cv_improvement_pct),
                    "mean_diff": float(cv_diff.mean()),
                    "wilcoxon_stat": float(cv_stat),
                    "wilcoxon_p": float(cv_p),
                    "significant": cv_p < 0.05,
                },
                "combined_objective": {
                    "improvement_pct": float(obj_improvement_pct),
                    "mean_diff": float(obj_diff.mean()),
                    "wilcoxon_stat": float(obj_stat),
                    "wilcoxon_p": float(obj_p),
                    "significant": obj_p < 0.05,
                },
            }
    
    return analysis


def print_analysis(analysis: Dict):
    """Print analysis results in a readable format."""
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print("="*80)
    print("OBJECTIVE VALUE COMPARISON")
    print("="*80)
    print(f"\nWeights: α={analysis['weights']['alpha']}, β={analysis['weights']['beta']}, γ={analysis['weights']['gamma']}")
    print()
    
    # Method statistics
    print("METHOD STATISTICS:")
    print("-"*80)
    for method, stats_dict in analysis['methods'].items():
        print(f"\n{method.upper()}:")
        print(f"  Count: {stats_dict['count']}")
        print(f"  Cost: {stats_dict['cost']['mean']:.2f} ± {stats_dict['cost']['std']:.2f} "
              f"(range: {stats_dict['cost']['min']:.2f} - {stats_dict['cost']['max']:.2f})")
        print(f"  CV: {stats_dict['cv']['mean']:.4f} ± {stats_dict['cv']['std']:.4f} "
              f"(range: {stats_dict['cv']['min']:.4f} - {stats_dict['cv']['max']:.4f})")
        print(f"  Combined Objective: {stats_dict['combined_objective']['mean']:.4f} ± {stats_dict['combined_objective']['std']:.4f}")
    
    # Comparisons
    print("\n" + "="*80)
    print("PAIRWISE COMPARISONS:")
    print("="*80)
    
    for comparison_key, comp in analysis['comparisons'].items():
        method1, method2 = comparison_key.split('_vs_')
        print(f"\n{method1.upper()} vs {method2.upper()} (n={comp['n_pairs']} pairs):")
        print("-"*80)
        
        # Cost
        cost_sig = "***" if comp['cost']['significant'] else ""
        print(f"  Cost: {comp['cost']['improvement_pct']:+.2f}% "
              f"(p={comp['cost']['wilcoxon_p']:.4f}){cost_sig}")
        
        # CV
        cv_sig = "***" if comp['cv']['significant'] else ""
        print(f"  CV: {comp['cv']['improvement_pct']:+.2f}% "
              f"(p={comp['cv']['wilcoxon_p']:.4f}){cv_sig}")
        
        # Combined objective
        obj_sig = "***" if comp['combined_objective']['significant'] else ""
        print(f"  Combined Objective: {comp['combined_objective']['improvement_pct']:+.2f}% "
              f"(p={comp['combined_objective']['wilcoxon_p']:.4f}){obj_sig}")
        
        print("  (*** = p < 0.05)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Compare objective values across methods"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing experimental results (JSON files)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for analysis results"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Cost weight (default: 1.0)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.2,
        help="CV weight (default: 0.2)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.0,
        help="Jain's index weight (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    # Load results
    try:
        df = load_results(Path(args.results_dir))
        print(f"Loaded {len(df)} results from {args.results_dir}")
    except Exception as e:
        print(f"Error loading results: {e}")
        return 1
    
    # Analyze
    analysis = analyze_objective_values(
        df, 
        alpha=args.alpha, 
        beta=args.beta, 
        gamma=args.gamma
    )
    
    # Print results
    print_analysis(analysis)
    
    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

