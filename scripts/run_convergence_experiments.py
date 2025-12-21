#!/usr/bin/env python3
"""Run convergence experiments with different iteration counts.

This script runs experiments with increasing iteration counts to analyze
convergence behavior and see if methods diverge over time.

Usage:
    python scripts/run_convergence_experiments.py \
        --instance data/benchmarks/solomon_original/C101.TXT \
        --iterations 100 200 500 1000 \
        --seeds 42 123 456 \
        --output-dir results/convergence
"""
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
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
)


@dataclass
class ConvergenceResult:
    """Result from a convergence experiment."""
    instance_name: str
    method: str
    seed: int
    iterations: int
    final_cost: float
    final_cv: float
    final_objective: float
    time_seconds: float
    n_routes: int
    initial_cost: float
    improvement_pct: float


def run_convergence_experiment(
    instance_path: Path,
    method: str,
    seed: int,
    iterations: int,
) -> ConvergenceResult:
    """Run a single convergence experiment.
    
    Args:
        instance_path: Path to instance file
        method: Method name (alns, alns_hnsw, hero)
        seed: Random seed
        iterations: Number of ALNS iterations
        
    Returns:
        ConvergenceResult with metrics
    """
    config = ExperimentConfig(
        seeds=[seed],
        max_iterations=iterations,
        segment_size=10,
        max_solomon_instances=1,
        max_homberger_instances=0,
        max_cvrp_instances=0,
        max_euro_instances=0,
        max_customers=1000,
        methods=[method],
        output_dir=None,
    )
    
    result = run_single_experiment(
        instance_path=instance_path,
        instance_type="solomon",
        method=method,
        seed=seed,
        config=config,
    )
    
    # Calculate improvement
    # Note: initial_cost is not directly available, we'll need to compute it
    # For now, we'll use the result's cost as final_cost
    improvement_pct = 0.0  # Would need initial cost to compute
    
    return ConvergenceResult(
        instance_name=result.instance_name,
        method=method,
        seed=seed,
        iterations=iterations,
        final_cost=result.cost,
        final_cv=result.cv,
        final_objective=0.0,  # Would need to compute from solution
        time_seconds=result.time_seconds,
        n_routes=result.n_routes,
        initial_cost=0.0,  # Not available from result
        improvement_pct=improvement_pct,
    )


def run_convergence_analysis(
    instance_path: Path,
    methods: List[str],
    seeds: List[int],
    iteration_counts: List[int],
    output_dir: Path,
) -> Dict:
    """Run convergence analysis across different iteration counts.
    
    Args:
        instance_path: Path to instance file
        methods: List of methods to test
        seeds: List of random seeds
        iteration_counts: List of iteration counts to test
        output_dir: Output directory for results
        
    Returns:
        Dictionary with all results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    print(f"Instance: {instance_path.name}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Seeds: {seeds}")
    print(f"Iteration counts: {iteration_counts}")
    print()
    
    all_results = []
    
    total_experiments = len(methods) * len(seeds) * len(iteration_counts)
    completed = 0
    
    for method in methods:
        print(f"\n{'─'*80}")
        print(f"METHOD: {method.upper()}")
        print(f"{'─'*80}")
        
        for seed in seeds:
            print(f"\n  Seed: {seed}")
            
            for iterations in iteration_counts:
                completed += 1
                print(f"    Iterations: {iterations} ({completed}/{total_experiments})", end=" ... ", flush=True)
                
                try:
                    result = run_convergence_experiment(
                        instance_path=instance_path,
                        method=method,
                        seed=seed,
                        iterations=iterations,
                    )
                    all_results.append(asdict(result))
                    print(f"✓ (cost={result.final_cost:.2f}, CV={result.final_cv:.4f}, time={result.time_seconds:.2f}s)")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    import traceback
                    logger.error(f"Error in {instance_path.stem}/{method}/seed{seed}/iter{iterations}: {e}")
                    logger.debug(traceback.format_exc())
    
    # Save results (always save, even if some failed)
    if all_results:
        try:
            results_file = output_dir / "convergence_results.json"
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Also save as CSV
            csv_file = output_dir / "convergence_results.csv"
            with open(csv_file, 'w') as f:
                f.write("instance_name,method,seed,iterations,final_cost,final_cv,final_objective,time_seconds,n_routes,initial_cost,improvement_pct\n")
                for r in all_results:
                    f.write(f"{r['instance_name']},{r['method']},{r['seed']},{r['iterations']},"
                           f"{r['final_cost']:.2f},{r['final_cv']:.4f},{r['final_objective']:.4f},"
                           f"{r['time_seconds']:.3f},{r['n_routes']},{r['initial_cost']:.2f},"
                           f"{r['improvement_pct']:.2f}\n")
            
            print(f"\n{'='*80}")
            print(f"Results saved to {results_file} and {csv_file}")
            print(f"{'='*80}\n")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            # Try backup save
            try:
                backup_file = output_dir / f"convergence_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(backup_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                logger.info(f"Saved backup to {backup_file}")
            except Exception:
                logger.error("Failed to save backup results!")
    else:
        logger.warning("No results to save!")
    
    # Analyze convergence
    analysis = analyze_convergence(all_results)
    
    # Save analysis
    analysis_file = output_dir / "convergence_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print_analysis(analysis)
    
    return {
        "results": all_results,
        "analysis": analysis,
    }


def analyze_convergence(results: List[Dict]) -> Dict:
    """Analyze convergence behavior from results.
    
    Args:
        results: List of ConvergenceResult dictionaries
        
    Returns:
        Dictionary with convergence analysis
    """
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame(results)
    
    analysis = {
        "by_method": {},
        "convergence_trends": {},
    }
    
    # Analyze by method
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        # Group by iteration count
        by_iterations = method_data.groupby('iterations').agg({
            'final_cost': ['mean', 'std', 'min', 'max'],
            'final_cv': ['mean', 'std', 'min', 'max'],
            'time_seconds': ['mean', 'std'],
        }).to_dict()
        
        analysis["by_method"][method] = {
            "iteration_counts": sorted(method_data['iterations'].unique()),
            "cost_by_iterations": {
                str(iterations): {
                    "mean": float(method_data[method_data['iterations'] == iterations]['final_cost'].mean()),
                    "std": float(method_data[method_data['iterations'] == iterations]['final_cost'].std()),
                }
                for iterations in method_data['iterations'].unique()
            },
            "cv_by_iterations": {
                str(iterations): {
                    "mean": float(method_data[method_data['iterations'] == iterations]['final_cv'].mean()),
                    "std": float(method_data[method_data['iterations'] == iterations]['final_cv'].std()),
                }
                for iterations in method_data['iterations'].unique()
            },
        }
        
        # Convergence trend: does cost improve with more iterations?
        iterations_sorted = sorted(method_data['iterations'].unique())
        if len(iterations_sorted) > 1:
            first_cost = method_data[method_data['iterations'] == iterations_sorted[0]]['final_cost'].mean()
            last_cost = method_data[method_data['iterations'] == iterations_sorted[-1]]['final_cost'].mean()
            improvement = (first_cost - last_cost) / first_cost * 100 if first_cost > 0 else 0
            
            analysis["convergence_trends"][method] = {
                "cost_improvement_pct": float(improvement),
                "converged": abs(improvement) < 5.0,  # Less than 5% improvement = converged
            }
    
    return analysis


def print_analysis(analysis: Dict):
    """Print convergence analysis."""
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS RESULTS")
    print("="*80)
    
    for method, data in analysis["by_method"].items():
        print(f"\n{method.upper()}:")
        print("-"*80)
        
        for iterations in data["iteration_counts"]:
            iter_str = str(iterations)
            cost_mean = data["cost_by_iterations"][iter_str]["mean"]
            cost_std = data["cost_by_iterations"][iter_str]["std"]
            cv_mean = data["cv_by_iterations"][iter_str]["mean"]
            cv_std = data["cv_by_iterations"][iter_str]["std"]
            
            print(f"  {iterations:4d} iter: Cost={cost_mean:8.2f}±{cost_std:6.2f}, "
                  f"CV={cv_mean:.4f}±{cv_std:.4f}")
    
    print("\n" + "="*80)
    print("CONVERGENCE TRENDS:")
    print("="*80)
    
    for method, trend in analysis["convergence_trends"].items():
        status = "✓ Converged" if trend["converged"] else "→ Still improving"
        print(f"{method.upper()}: {trend['cost_improvement_pct']:+.2f}% improvement {status}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run convergence experiments with different iteration counts"
    )
    parser.add_argument(
        "--instance", type=str, required=True,
        help="Path to instance file"
    )
    parser.add_argument(
        "--iterations", type=int, nargs="+", default=[100, 200, 500, 1000],
        help="Iteration counts to test (default: 100 200 500 1000)"
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=["alns", "alns_hnsw", "hero"],
        help="Methods to test (default: alns alns_hnsw hero)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456],
        help="Random seeds (default: 42 123 456)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/convergence",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    instance_path = Path(args.instance)
    if not instance_path.exists():
        print(f"Error: Instance file not found: {instance_path}")
        return 1
    
    # Run convergence analysis
    results = run_convergence_analysis(
        instance_path=instance_path,
        methods=args.methods,
        seeds=args.seeds,
        iteration_counts=args.iterations,
        output_dir=Path(args.output_dir),
    )
    
    print(f"\n✓ Convergence analysis complete!")
    print(f"  Results: {args.output_dir}/convergence_results.json")
    print(f"  Analysis: {args.output_dir}/convergence_analysis.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

