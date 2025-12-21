#!/usr/bin/env python3
"""Quick test script to verify OR-Tools and PyVRP baselines are working."""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.comprehensive_benchmark import run_ortools_experiment, run_pyvrp_experiment
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    max_iterations: int = 200
    max_customers: int = 1000
    seeds: list = None
    max_solomon_instances: int = 20
    max_homberger_instances: int = 5
    max_cvrp_instances: int = 10
    max_euro_instances: int = 10
    output_dir: Path = Path('results/test')


def test_baselines():
    """Test both OR-Tools and PyVRP on sample instances."""
    config = ExperimentConfig()
    
    # Test instances (small ones for quick testing)
    test_instances = [
        ('data/benchmarks/solomon_original/C101.TXT', 'solomon'),
        ('data/benchmarks/solomon_original/C201.TXT', 'solomon'),
        ('data/benchmarks/solomon_original/R101.TXT', 'solomon'),
    ]
    
    print("="*80)
    print("BASELINE SOLVER TEST")
    print("="*80)
    print()
    
    results = {'ortools': {'success': 0, 'failed': 0, 'total_time': 0},
               'pyvrp': {'success': 0, 'failed': 0, 'total_time': 0}}
    
    for instance_path, instance_type in test_instances:
        instance_path = Path(instance_path)
        if not instance_path.exists():
            print(f"⚠ Skipping {instance_path.name} (file not found)")
            continue
        
        print(f"Testing {instance_path.name} ({instance_type})...")
        print("-" * 80)
        
        # Test OR-Tools
        print("  OR-Tools:", end=" ", flush=True)
        try:
            start = time.time()
            result = run_ortools_experiment(instance_path, instance_type, 42, config)
            elapsed = time.time() - start
            
            if result.cost != float('inf') and result.n_routes > 0:
                print(f"✓ SUCCESS - Cost: {result.cost:.2f}, Routes: {result.n_routes}, "
                      f"CV: {result.cv:.4f}, Time: {result.time_seconds:.2f}s")
                results['ortools']['success'] += 1
                results['ortools']['total_time'] += result.time_seconds
            else:
                print(f"✗ FAILED - Cost: {result.cost}, Routes: {result.n_routes}")
                results['ortools']['failed'] += 1
        except Exception as e:
            print(f"✗ ERROR - {e}")
            results['ortools']['failed'] += 1
        
        # Test PyVRP (with shorter time limit for testing)
        print("  PyVRP:   ", end=" ", flush=True)
        try:
            # Use shorter time limit for testing
            original_max_iter = config.max_iterations
            config.max_iterations = 50  # Shorter for testing
            
            start = time.time()
            result = run_pyvrp_experiment(instance_path, instance_type, 42, config)
            elapsed = time.time() - start
            
            config.max_iterations = original_max_iter
            
            if result.cost != float('inf') and result.n_routes > 0:
                print(f"✓ SUCCESS - Cost: {result.cost:.2f}, Routes: {result.n_routes}, "
                      f"CV: {result.cv:.4f}, Time: {result.time_seconds:.2f}s")
                results['pyvrp']['success'] += 1
                results['pyvrp']['total_time'] += result.time_seconds
            else:
                print(f"✗ FAILED - Cost: {result.cost}, Routes: {result.n_routes}")
                results['pyvrp']['failed'] += 1
        except Exception as e:
            print(f"✗ ERROR - {e}")
            import traceback
            traceback.print_exc()
            results['pyvrp']['failed'] += 1
        
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    for solver_name, stats in results.items():
        total = stats['success'] + stats['failed']
        success_rate = (stats['success'] / total * 100) if total > 0 else 0
        avg_time = stats['total_time'] / stats['success'] if stats['success'] > 0 else 0
        
        print(f"{solver_name.upper()}:")
        print(f"  Success: {stats['success']}/{total} ({success_rate:.1f}%)")
        print(f"  Failed:  {stats['failed']}/{total}")
        if stats['success'] > 0:
            print(f"  Avg Time: {avg_time:.2f}s")
        print()
    
    # Overall status
    ortools_ok = results['ortools']['success'] > 0
    pyvrp_ok = results['pyvrp']['success'] > 0
    
    if ortools_ok and pyvrp_ok:
        print("✓ Both baselines are working correctly!")
    elif ortools_ok:
        print("⚠ OR-Tools is working, but PyVRP needs attention")
    elif pyvrp_ok:
        print("⚠ PyVRP is working, but OR-Tools needs attention")
    else:
        print("✗ Both baselines are failing - needs investigation")
    
    print()


if __name__ == '__main__':
    test_baselines()

