#!/usr/bin/env python3
"""
Debug script to investigate why methods produce identical results.

Checks:
1. Random seed handling
2. HNSW usage
3. Fairness features
4. Method configuration differences
"""

import sys
from pathlib import Path
import numpy as np
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.parsers import parse_instance
from src.models.solution import Solution
from src.heuristics.constructive import nearest_neighbor
from src.heuristics.alns import ALNS, ALNSConfig
from src.hnsw.manager import HNSWManager, HNSWManagerConfig
from src.hnsw.index import HNSWConfig
from src.hnsw.features import FeatureEncoder, FeatureConfig

def test_method_configuration(instance_path: Path, seed: int = 42):
    """Test configuration differences between methods."""
    
    print("="*80)
    print("METHOD CONFIGURATION DIAGNOSTICS")
    print("="*80)
    print(f"Instance: {instance_path.name}")
    print(f"Seed: {seed}")
    print()
    
    # Parse instance
    instance = parse_instance(instance_path)
    print(f"Instance: {instance.n_customers} customers, {instance.n_vehicles} vehicles")
    print()
    
    # Test each method
    methods = ["alns", "alns_hnsw", "hero"]
    
    for method in methods:
        print(f"\n{'─'*80}")
        print(f"METHOD: {method.upper()}")
        print(f"{'─'*80}")
        
        # Reset random state
        np.random.seed(seed)
        random.seed(seed)
        
        # Build initial solution
        initial = nearest_neighbor(instance)
        initial_cost = initial.compute_cost()
        print(f"Initial solution cost: {initial_cost:.2f}")
        print(f"Initial solution routes: {len([r for r in initial.routes if r.customers])}")
        
        # Configure ALNS
        use_fairness = (method == "hero")
        print(f"Use fairness: {use_fairness}")
        
        alns_config = ALNSConfig(
            max_iterations=10,  # Small for testing
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
        
        print(f"ALNS Config:")
        print(f"  alpha (cost weight): {alns_config.alpha}")
        print(f"  beta (fairness weight): {alns_config.beta}")
        print(f"  seed: {alns_config.seed}")
        
        # Configure HNSW
        hnsw_manager = None
        if method in ["alns_hnsw", "hero"]:
            feature_config = FeatureConfig(
                use_fairness_features=use_fairness,
                use_subsequence_features=True,
            )
            
            dim = feature_config.feature_dim
            print(f"HNSW Config:")
            print(f"  Feature dimension: {dim}D")
            print(f"  Use fairness features: {feature_config.use_fairness_features}")
            print(f"  Use subsequence features: {feature_config.use_subsequence_features}")
            
            hnsw_config = HNSWConfig(dim=dim, M=16, ef_construction=200, ef_search=50)
            manager_config = HNSWManagerConfig(
                hnsw_config=hnsw_config,
                feature_config=feature_config,
            )
            hnsw_manager = HNSWManager(config=manager_config)
            print(f"  HNSW Manager created: {hnsw_manager is not None}")
        else:
            print(f"HNSW Config: None (not using HNSW)")
        
        # Create ALNS
        alns = ALNS(config=alns_config, hnsw_manager=hnsw_manager)
        
        # Check random state
        print(f"Random state check:")
        print(f"  ALNS._rng state: {alns._rng.getstate()[1][:3]}...")  # First few values
        print(f"  np.random state: {np.random.get_state()[1][:3]}...")
        
        # Check repair operators
        print(f"Repair operators:")
        for i, op in enumerate(alns.repair_operators):
            has_hnsw = hasattr(op, 'hnsw_manager') and op.hnsw_manager is not None
            print(f"  {i+1}. {op.name}: HNSW={has_hnsw}")
            if has_hnsw:
                print(f"      k_candidates: {getattr(op, 'k_candidates', 'N/A')}")
        
        # Initialize HNSW if available
        if hnsw_manager is not None:
            print(f"Initializing HNSW index...")
            hnsw_manager.initialize(instance, initial)
            index_size = "N/A"
            if hasattr(hnsw_manager, '_index') and hnsw_manager._index is not None:
                # Try to get index size
                if hasattr(hnsw_manager._index, 'get_current_count'):
                    index_size = hnsw_manager._index.get_current_count()
                elif hasattr(hnsw_manager._index, '_max_elements'):
                    index_size = f"max={hnsw_manager._index._max_elements}"
            print(f"  HNSW index: {index_size}")
            
            # Test a query
            if instance.n_customers > 0:
                test_customer = 1
                try:
                    candidates = hnsw_manager.find_insertion_candidates(
                        test_customer, initial, k=5
                    )
                    print(f"  Test query (customer {test_customer}): {len(candidates)} candidates found")
                    if candidates:
                        print(f"    First candidate: route={candidates[0].route_id}, pos={candidates[0].position}")
                except Exception as e:
                    print(f"  Test query failed: {e}")
        
        # Run a few iterations
        print(f"\nRunning 3 iterations...")
        solution, stats = alns.solve(instance, initial.copy())
        final_cost = solution.compute_cost()
        print(f"Final solution cost: {final_cost:.2f}")
        print(f"Cost change: {final_cost - initial_cost:+.2f}")
        
        # Check if HNSW was actually used
        if hnsw_manager is not None:
            # Check if index was updated
            print(f"HNSW index after solve: {len(hnsw_manager._index._data) if hasattr(hnsw_manager, '_index') else 'N/A'} entries")
        
        print()

def test_random_seed_isolation():
    """Test if random seeds are properly isolated."""
    print("="*80)
    print("RANDOM SEED ISOLATION TEST")
    print("="*80)
    
    seeds = [42, 123, 456]
    
    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        values = [np.random.random() for _ in range(5)]
        print(f"Seed {seed}: {[f'{v:.4f}' for v in values]}")
    
    print("\nIf seeds are isolated, each should produce different sequences.")
    print()

def test_initial_solution_consistency():
    """Test if initial solutions are the same across methods."""
    print("="*80)
    print("INITIAL SOLUTION CONSISTENCY TEST")
    print("="*80)
    
    # Use a test instance
    test_instances = [
        Path("data/benchmarks/solomon/C101.txt"),
        Path("data/benchmarks/solomon/C102.txt"),
    ]
    
    for instance_path in test_instances:
        if not instance_path.exists():
            print(f"Skipping {instance_path} (not found)")
            continue
        
        instance = parse_instance(instance_path)
        seed = 42
        
        # Generate initial solution multiple times
        costs = []
        for i in range(3):
            np.random.seed(seed)
            random.seed(seed)
            initial = nearest_neighbor(instance)
            cost = initial.compute_cost()
            costs.append(cost)
            print(f"  Run {i+1}: cost={cost:.2f}")
        
        if len(set(costs)) == 1:
            print(f"  ⚠️  All runs produce identical cost: {costs[0]:.2f}")
        else:
            print(f"  ✓ Runs produce different costs: {costs}")
        print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug method differences")
    parser.add_argument("--instance", type=str, help="Instance file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test-seeds", action="store_true", help="Test random seed isolation")
    parser.add_argument("--test-initial", action="store_true", help="Test initial solution consistency")
    
    args = parser.parse_args()
    
    if args.test_seeds:
        test_random_seed_isolation()
    
    if args.test_initial:
        test_initial_solution_consistency()
    
    if args.instance:
        instance_path = Path(args.instance)
        if instance_path.exists():
            test_method_configuration(instance_path, args.seed)
        else:
            print(f"Error: Instance file not found: {args.instance}")
    else:
        # Default: test with C101 if available
        test_instance = Path("data/benchmarks/solomon/C101.txt")
        if test_instance.exists():
            test_method_configuration(test_instance, args.seed)
        else:
            print("Error: No instance specified and C101.txt not found.")
            print("Usage: python debug_method_differences.py --instance <path>")
            print("Or: python debug_method_differences.py --test-seeds --test-initial")

