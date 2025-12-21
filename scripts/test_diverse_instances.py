#!/usr/bin/env python3
"""Test on diverse instance sizes to analyze scalability and behavior.

This script runs experiments on instances of different sizes to:
- Analyze scalability (time vs instance size)
- Compare behavior across size categories
- Identify size-dependent performance differences

Usage:
    python scripts/test_diverse_instances.py \
        --size-categories small medium large \
        --methods alns alns_hnsw hero \
        --seeds 42 123 456 \
        --output-dir results/diverse_instances
"""
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import traceback

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.comprehensive_benchmark import (
    run_single_experiment,
    ExperimentConfig,
    ExperimentResult,
    discover_instances,
)


@dataclass
class SizeCategory:
    """Size category definition."""
    name: str
    min_customers: int
    max_customers: int
    description: str


SIZE_CATEGORIES = {
    "small": SizeCategory("small", 50, 100, "Small instances (50-100 customers)"),
    "medium": SizeCategory("medium", 200, 300, "Medium instances (200-300 customers)"),
    "large": SizeCategory("large", 500, 1000, "Large instances (500-1000 customers)"),
    "xlarge": SizeCategory("xlarge", 1000, 2000, "Extra large instances (1000-2000 customers)"),
}


def filter_instances_by_size(
    instances: Dict[str, List[Path]],
    size_category: SizeCategory,
) -> Dict[str, List[Path]]:
    """Filter instances by size category.
    
    Args:
        instances: Dictionary of instance type -> list of paths
        size_category: Size category to filter by
        
    Returns:
        Filtered instances dictionary
    """
    from src.models.parsers import parse_instance
    
    filtered = {}
    
    for instance_type, instance_paths in instances.items():
        filtered_paths = []
        
        for instance_path in instance_paths:
            try:
                instance = parse_instance(instance_path)
                n_customers = instance.n_customers
                
                if size_category.min_customers <= n_customers <= size_category.max_customers:
                    filtered_paths.append(instance_path)
            except Exception as e:
                print(f"Warning: Could not parse {instance_path}: {e}")
                continue
        
        if filtered_paths:
            filtered[instance_type] = filtered_paths
    
    return filtered


def run_diverse_instance_experiments(
    size_categories: List[str],
    methods: List[str],
    seeds: List[int],
    max_iterations: int,
    max_instances_per_category: int,
    output_dir: Path,
) -> Dict:
    """Run experiments on diverse instance sizes.
    
    Args:
        size_categories: List of size category names to test
        methods: List of methods to test
        seeds: List of random seeds
        max_iterations: Maximum ALNS iterations
        max_instances_per_category: Max instances per category
        output_dir: Output directory for results
        
    Returns:
        Dictionary with all results and analysis
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("DIVERSE INSTANCE SIZE ANALYSIS")
    print("="*80)
    print(f"Size categories: {', '.join(size_categories)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Seeds: {seeds}")
    print(f"Max iterations: {max_iterations}")
    print()
    
    # Discover all instances
    config = ExperimentConfig(
        seeds=seeds,
        max_iterations=max_iterations,
        max_solomon_instances=100,
        max_homberger_instances=100,
        max_cvrp_instances=100,
        max_euro_instances=100,
        max_customers=10000,  # High limit to discover all
        methods=methods,
        output_dir=None,
    )
    
    all_instances = discover_instances(config)
    
    all_results = []
    category_results = {}
    
    # Process each size category
    for category_name in size_categories:
        if category_name not in SIZE_CATEGORIES:
            print(f"Warning: Unknown size category '{category_name}', skipping")
            continue
        
        category = SIZE_CATEGORIES[category_name]
        print(f"\n{'─'*80}")
        print(f"SIZE CATEGORY: {category.name.upper()} ({category.description})")
        print(f"{'─'*80}")
        
        # Filter instances by size
        category_instances = filter_instances_by_size(all_instances, category)
        
        if not category_instances:
            print(f"  No instances found in size range {category.min_customers}-{category.max_customers}")
            continue
        
        # Count total instances
        total_category_instances = sum(len(v) for v in category_instances.values())
        print(f"  Found {total_category_instances} instances")
        
        # Limit instances per category
        category_instances_limited = {}
        for instance_type, instance_paths in category_instances.items():
            category_instances_limited[instance_type] = instance_paths[:max_instances_per_category]
        
        total_to_test = sum(len(v) for v in category_instances_limited.values())
        print(f"  Testing {total_to_test} instances (limited to {max_instances_per_category} per type)")
        
        category_results[category_name] = []
        
        # Run experiments
        experiment_count = 0
        total_experiments = total_to_test * len(methods) * len(seeds)
        
        for instance_type, instance_paths in category_instances_limited.items():
            for instance_path in instance_paths:
                for method in methods:
                    for seed in seeds:
                        experiment_count += 1
                        print(f"  [{experiment_count}/{total_experiments}] {instance_path.stem}/{method}/seed{seed} ... ", end="", flush=True)
                        
                        try:
                            result = run_single_experiment(
                                instance_path=instance_path,
                                instance_type=instance_type,
                                method=method,
                                seed=seed,
                                config=config,
                            )
                            
                            # Add size category info
                            result_dict = asdict(result)
                            result_dict['size_category'] = category_name
                            result_dict['size_min'] = category.min_customers
                            result_dict['size_max'] = category.max_customers
                            
                            all_results.append(result_dict)
                            category_results[category_name].append(result_dict)
                            
                            print(f"✓ (cost={result.cost:.2f}, CV={result.cv:.4f}, time={result.time_seconds:.2f}s)")
                        except Exception as e:
                            print(f"✗ Error: {e}")
                            # Log error but continue
                            import traceback
                            logger.error(f"Error in {instance_path.stem}/{method}/seed{seed}: {e}")
                            logger.debug(traceback.format_exc())
        
        print(f"\n  Category '{category_name}': {len(category_results[category_name])} results")
        
        # Incremental save after each category
        if category_results[category_name]:
            try:
                category_file = output_dir / f"category_{category_name}_results.json"
                with open(category_file, 'w') as f:
                    json.dump(category_results[category_name], f, indent=2)
                logger.info(f"Incremental save: {len(category_results[category_name])} results for category {category_name}")
            except Exception as e:
                logger.error(f"Failed to save incremental results for {category_name}: {e}")
    
    # Save all results (final save)
    if all_results:
        try:
            results_file = output_dir / "diverse_instances_results.json"
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Also save as CSV
            csv_file = output_dir / "diverse_instances_results.csv"
            with open(csv_file, 'w') as f:
                f.write("instance_name,instance_type,n_customers,method,seed,cost,n_routes,cv,time_seconds,size_category,size_min,size_max\n")
                for r in all_results:
                    f.write(f"{r['instance_name']},{r['instance_type']},{r['n_customers']},{r['method']},"
                           f"{r['seed']},{r['cost']:.2f},{r['n_routes']},{r['cv']:.4f},{r['time_seconds']:.3f},"
                           f"{r.get('size_category', '')},{r.get('size_min', '')},{r.get('size_max', '')}\n")
            
            print(f"\n{'='*80}")
            print(f"Results saved to {results_file} and {csv_file}")
            print(f"{'='*80}\n")
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
            # Try to save at least something
            try:
                backup_file = output_dir / f"diverse_instances_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(backup_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                logger.info(f"Saved backup to {backup_file}")
            except Exception:
                logger.error("Failed to save backup results!")
    else:
        logger.warning("No results to save!")
    
    # Analyze results
    analysis = analyze_diverse_instances(all_results, category_results)
    
    # Save analysis
    analysis_file = output_dir / "diverse_instances_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print_analysis(analysis)
    
    return {
        "results": all_results,
        "analysis": analysis,
    }


def analyze_diverse_instances(
    all_results: List[Dict],
    category_results: Dict[str, List[Dict]],
) -> Dict:
    """Analyze results across different instance sizes.
    
    Args:
        all_results: All experimental results
        category_results: Results grouped by size category
        
    Returns:
        Dictionary with analysis
    """
    import pandas as pd
    import numpy as np
    
    if not all_results:
        return {"error": "No results to analyze"}
    
    df = pd.DataFrame(all_results)
    
    analysis = {
        "by_category": {},
        "scalability": {},
        "size_dependent_behavior": {},
    }
    
    # Analyze by size category
    for category_name, category_data in category_results.items():
        if not category_data:
            continue
        
        cat_df = pd.DataFrame(category_data)
        
        analysis["by_category"][category_name] = {
            "n_instances": len(cat_df['instance_name'].unique()),
            "n_experiments": len(cat_df),
            "avg_customers": float(cat_df['n_customers'].mean()),
            "by_method": {},
        }
        
        # Analyze by method
        for method in cat_df['method'].unique():
            method_data = cat_df[cat_df['method'] == method]
            
            analysis["by_category"][category_name]["by_method"][method] = {
                "avg_cost": float(method_data['cost'].mean()),
                "avg_cv": float(method_data['cv'].mean()),
                "avg_time": float(method_data['time_seconds'].mean()),
                "avg_routes": float(method_data['n_routes'].mean()),
            }
    
    # Scalability analysis: time vs instance size
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        # Group by instance size (rounded to nearest 50)
        method_data = method_data.copy()
        method_data['size_bin'] = (method_data['n_customers'] // 50) * 50
        
        scalability = method_data.groupby('size_bin').agg({
            'time_seconds': ['mean', 'std'],
            'cost': ['mean', 'std'],
            'cv': ['mean', 'std'],
        }).to_dict()
        
        analysis["scalability"][method] = {
            "size_bins": sorted(method_data['size_bin'].unique()),
            "time_by_size": {
                str(int(size)): {
                    "mean": float(method_data[method_data['size_bin'] == size]['time_seconds'].mean()),
                    "std": float(method_data[method_data['size_bin'] == size]['time_seconds'].std()),
                }
                for size in method_data['size_bin'].unique()
            },
        }
    
    # Size-dependent behavior: compare methods across sizes
    for method1 in df['method'].unique():
        for method2 in df['method'].unique():
            if method1 >= method2:
                continue
            
            comparison_key = f"{method1}_vs_{method2}"
            analysis["size_dependent_behavior"][comparison_key] = {}
            
            for category_name in category_results.keys():
                cat_df = pd.DataFrame(category_results[category_name])
                
                method1_data = cat_df[cat_df['method'] == method1]
                method2_data = cat_df[cat_df['method'] == method2]
                
                if len(method1_data) == 0 or len(method2_data) == 0:
                    continue
                
                # Compare average performance
                cost_diff = method2_data['cost'].mean() - method1_data['cost'].mean()
                cv_diff = method2_data['cv'].mean() - method1_data['cv'].mean()
                time_ratio = method2_data['time_seconds'].mean() / method1_data['time_seconds'].mean() if method1_data['time_seconds'].mean() > 0 else 0
                
                analysis["size_dependent_behavior"][comparison_key][category_name] = {
                    "cost_diff": float(cost_diff),
                    "cv_diff": float(cv_diff),
                    "time_ratio": float(time_ratio),
                }
    
    return analysis


def print_analysis(analysis: Dict):
    """Print analysis results."""
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print("\n" + "="*80)
    print("DIVERSE INSTANCE SIZE ANALYSIS RESULTS")
    print("="*80)
    
    # By category
    print("\nBY SIZE CATEGORY:")
    print("-"*80)
    
    for category_name, cat_data in analysis["by_category"].items():
        print(f"\n{category_name.upper()} (avg {cat_data['avg_customers']:.0f} customers, {cat_data['n_instances']} instances):")
        
        for method, method_data in cat_data["by_method"].items():
            print(f"  {method:12s}: Cost={method_data['avg_cost']:8.2f}, "
                  f"CV={method_data['avg_cv']:.4f}, "
                  f"Time={method_data['avg_time']:6.2f}s, "
                  f"Routes={method_data['avg_routes']:.1f}")
    
    # Scalability
    print("\n" + "="*80)
    print("SCALABILITY (Time vs Instance Size):")
    print("="*80)
    
    for method, scal_data in analysis["scalability"].items():
        print(f"\n{method.upper()}:")
        for size_bin in scal_data["size_bins"]:
            size_str = str(int(size_bin))
            time_mean = scal_data["time_by_size"][size_str]["mean"]
            time_std = scal_data["time_by_size"][size_str]["std"]
            print(f"  Size {size_str:4s}: {time_mean:6.2f}±{time_std:.2f}s")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test on diverse instance sizes"
    )
    parser.add_argument(
        "--size-categories", type=str, nargs="+",
        default=["small", "medium", "large"],
        choices=list(SIZE_CATEGORIES.keys()),
        help="Size categories to test (default: small medium large)"
    )
    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=["alns", "alns_hnsw", "hero"],
        help="Methods to test (default: alns alns_hnsw hero)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Random seeds (default: 42 123 456)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=200,
        help="Maximum ALNS iterations (default: 200)"
    )
    parser.add_argument(
        "--max-instances-per-category", type=int, default=5,
        help="Maximum instances per category (default: 5)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/diverse_instances",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Run experiments
    results = run_diverse_instance_experiments(
        size_categories=args.size_categories,
        methods=args.methods,
        seeds=args.seeds,
        max_iterations=args.max_iterations,
        max_instances_per_category=args.max_instances_per_category,
        output_dir=Path(args.output_dir),
    )
    
    print(f"\n✓ Diverse instance analysis complete!")
    print(f"  Results: {args.output_dir}/diverse_instances_results.json")
    print(f"  Analysis: {args.output_dir}/diverse_instances_analysis.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

