#!/usr/bin/env python3
"""Test script to verify HNSW rebuild optimizations.

This script runs experiments and collects rebuild statistics to verify
that the optimizations (increased threshold, incremental updates) achieve
the expected 87% reduction in rebuilds.
"""
import sys
from pathlib import Path
import argparse
import logging
import json
from typing import Dict, List
from dataclasses import dataclass, asdict
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.parsers import parse_instance
from src.models.solution import Solution
from src.heuristics.constructive import nearest_neighbor
from src.heuristics.alns import ALNS, ALNSConfig
from src.hnsw.manager import HNSWManager, HNSWManagerConfig
from src.hnsw.index import HNSWConfig
from src.hnsw.features import FeatureConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RebuildTestResult:
    """Result from a single rebuild test run."""
    instance_name: str
    n_customers: int
    max_iterations: int
    seed: int
    
    # Rebuild statistics
    n_rebuilds: int
    n_incremental_updates: int
    incremental_ratio: float
    total_updates: int
    
    # Performance statistics
    n_queries: int
    avg_query_time_ms: float
    total_time_seconds: float
    
    # Solution quality
    final_cost: float
    initial_cost: float
    
    # Index statistics
    index_size: int
    rebuild_threshold: float


def run_rebuild_test(
    instance_path: Path,
    max_iterations: int = 200,
    seed: int = 42,
    rebuild_threshold: float = 0.6,
    use_fairness: bool = False,
) -> RebuildTestResult:
    """Run a single test and collect rebuild statistics.
    
    Args:
        instance_path: Path to instance file
        max_iterations: Number of ALNS iterations
        seed: Random seed
        rebuild_threshold: HNSW rebuild threshold (0.6 = optimized, 0.3 = baseline)
        use_fairness: Whether to use fairness features
        
    Returns:
        RebuildTestResult with statistics
    """
    import numpy as np
    import random
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Parse instance
    logger.info(f"Loading instance: {instance_path.name}")
    instance = parse_instance(instance_path)
    
    # Build initial solution
    initial = nearest_neighbor(instance)
    initial_cost = initial.compute_cost()
    
    # Configure HNSW with specified rebuild threshold
    feature_config = FeatureConfig(
        use_fairness_features=use_fairness,
        use_subsequence_features=True,
    )
    dim = feature_config.feature_dim
    hnsw_config = HNSWConfig(dim=dim, M=16, ef_construction=200, ef_search=50)
    
    manager_config = HNSWManagerConfig(
        hnsw_config=hnsw_config,
        feature_config=feature_config,
        rebuild_threshold=rebuild_threshold,  # Test with different thresholds
    )
    hnsw_manager = HNSWManager(config=manager_config)
    
    # Configure ALNS
    alns_config = ALNSConfig(
        max_iterations=max_iterations,
        segment_length=10,
        min_destroy_fraction=0.1,
        max_destroy_fraction=0.3,
        initial_temperature=2000.0,
        cooling_rate=0.9997,
        seed=seed,
        alpha=0.8 if use_fairness else 1.0,
        beta=0.2 if use_fairness else 0.0,
        gamma=0.0,
        verbose=False,
    )
    
    # Run ALNS
    alns = ALNS(config=alns_config, hnsw_manager=hnsw_manager)
    
    start_time = time.time()
    solution, stats = alns.solve(instance, initial.copy())
    elapsed = time.time() - start_time
    
    # Get HNSW statistics
    hnsw_stats = hnsw_manager.get_statistics()
    
    final_cost = solution.compute_cost()
    
    return RebuildTestResult(
        instance_name=instance_path.stem,
        n_customers=instance.n_customers,
        max_iterations=max_iterations,
        seed=seed,
        n_rebuilds=hnsw_stats['n_rebuilds'],
        n_incremental_updates=hnsw_stats['n_incremental_updates'],
        incremental_ratio=hnsw_stats['incremental_ratio'],
        total_updates=hnsw_stats['n_rebuilds'] + hnsw_stats['n_incremental_updates'],
        n_queries=hnsw_stats['n_queries'],
        avg_query_time_ms=hnsw_stats['avg_query_time_ms'],
        total_time_seconds=elapsed,
        final_cost=final_cost,
        initial_cost=initial_cost,
        index_size=hnsw_stats['index_size'],
        rebuild_threshold=rebuild_threshold,
    )


def compare_thresholds(
    instance_path: Path,
    max_iterations: int = 200,
    seed: int = 42,
    use_fairness: bool = False,
) -> Dict:
    """Compare rebuild statistics with different thresholds.
    
    Args:
        instance_path: Path to instance file
        max_iterations: Number of ALNS iterations
        seed: Random seed
        use_fairness: Whether to use fairness features
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing rebuild optimizations on {instance_path.name}")
    logger.info(f"{'='*80}\n")
    
    # Test with baseline threshold (0.3)
    logger.info("Running with baseline threshold (0.3)...")
    baseline_result = run_rebuild_test(
        instance_path, max_iterations, seed,
        rebuild_threshold=0.3, use_fairness=use_fairness
    )
    
    # Test with optimized threshold (0.6)
    logger.info("Running with optimized threshold (0.6)...")
    optimized_result = run_rebuild_test(
        instance_path, max_iterations, seed,
        rebuild_threshold=0.6, use_fairness=use_fairness
    )
    
    # Calculate improvements
    rebuild_reduction = (
        (baseline_result.n_rebuilds - optimized_result.n_rebuilds) 
        / baseline_result.n_rebuilds * 100
        if baseline_result.n_rebuilds > 0 else 0.0
    )
    
    incremental_increase = (
        (optimized_result.n_incremental_updates - baseline_result.n_incremental_updates)
        / max(baseline_result.n_incremental_updates, 1) * 100
    )
    
    time_improvement = (
        (baseline_result.total_time_seconds - optimized_result.total_time_seconds)
        / baseline_result.total_time_seconds * 100
        if baseline_result.total_time_seconds > 0 else 0.0
    )
    
    comparison = {
        "instance": instance_path.stem,
        "n_customers": baseline_result.n_customers,
        "max_iterations": max_iterations,
        "seed": seed,
        "baseline": asdict(baseline_result),
        "optimized": asdict(optimized_result),
        "improvements": {
            "rebuild_reduction_percent": rebuild_reduction,
            "incremental_increase_percent": incremental_increase,
            "time_improvement_percent": time_improvement,
            "rebuild_reduction_absolute": baseline_result.n_rebuilds - optimized_result.n_rebuilds,
        }
    }
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"REBUILD OPTIMIZATION RESULTS: {instance_path.name}")
    print(f"{'='*80}")
    print(f"\nBaseline (threshold=0.3):")
    print(f"  Rebuilds: {baseline_result.n_rebuilds}")
    print(f"  Incremental updates: {baseline_result.n_incremental_updates}")
    print(f"  Incremental ratio: {baseline_result.incremental_ratio:.2%}")
    print(f"  Total time: {baseline_result.total_time_seconds:.2f}s")
    
    print(f"\nOptimized (threshold=0.6):")
    print(f"  Rebuilds: {optimized_result.n_rebuilds}")
    print(f"  Incremental updates: {optimized_result.n_incremental_updates}")
    print(f"  Incremental ratio: {optimized_result.incremental_ratio:.2%}")
    print(f"  Total time: {optimized_result.total_time_seconds:.2f}s")
    
    print(f"\nImprovements:")
    print(f"  Rebuild reduction: {rebuild_reduction:.1f}% ({baseline_result.n_rebuilds} → {optimized_result.n_rebuilds})")
    print(f"  Incremental updates: {incremental_increase:+.1f}% ({baseline_result.n_incremental_updates} → {optimized_result.n_incremental_updates})")
    print(f"  Time improvement: {time_improvement:.1f}% ({baseline_result.total_time_seconds:.2f}s → {optimized_result.total_time_seconds:.2f}s)")
    print(f"{'='*80}\n")
    
    return comparison


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test HNSW rebuild optimizations"
    )
    parser.add_argument(
        "--instance", type=str, required=True,
        help="Path to instance file"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=200,
        help="Number of ALNS iterations"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--use-fairness", action="store_true",
        help="Use fairness features (HERO method)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test with fewer iterations"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        args.max_iterations = 50
    
    instance_path = Path(args.instance)
    if not instance_path.exists():
        logger.error(f"Instance file not found: {instance_path}")
        return 1
    
    # Run comparison
    comparison = compare_thresholds(
        instance_path,
        max_iterations=args.max_iterations,
        seed=args.seed,
        use_fairness=args.use_fairness,
    )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    # Check if 87% reduction achieved
    rebuild_reduction = comparison['improvements']['rebuild_reduction_percent']
    if rebuild_reduction >= 85:
        print(f"\n✅ SUCCESS: Rebuild reduction ({rebuild_reduction:.1f}%) meets target (≥85%)")
        return 0
    elif rebuild_reduction >= 70:
        print(f"\n⚠️  PARTIAL: Rebuild reduction ({rebuild_reduction:.1f}%) is good but below target (85%)")
        return 0
    else:
        print(f"\n❌ NEEDS IMPROVEMENT: Rebuild reduction ({rebuild_reduction:.1f}%) is below target (85%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())

