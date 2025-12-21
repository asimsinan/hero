#!/usr/bin/env python3
"""
Track Method Divergence

This script runs all three methods (ALNS, ALNS-HNSW, HERO) on the same instance
and tracks when and how they diverge during optimization.

Tracks:
- Solution states at key iterations (0, 10, 50, 100, 200, etc.)
- Cost differences between methods
- Route differences (customer assignments)
- CV differences
- When methods first diverge
- Divergence metrics over time
"""

import sys
from pathlib import Path
import numpy as np
import random
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.parsers import parse_instance
from src.models.solution import Solution
from src.heuristics.constructive import nearest_neighbor
from src.heuristics.alns import ALNS, ALNSConfig
from src.hnsw.manager import HNSWManager, HNSWManagerConfig
from src.hnsw.index import HNSWConfig
from src.hnsw.features import FeatureEncoder, FeatureConfig
from src.models.fairness import coefficient_of_variation


@dataclass
class SolutionSnapshot:
    """Snapshot of solution state at a specific iteration."""
    iteration: int
    cost: float
    n_routes: int
    cv: float
    objective: float
    customer_assignments: Dict[int, Tuple[int, int]]  # customer_id -> (route_id, position)
    route_customers: Dict[int, List[int]]  # route_id -> list of customer_ids
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'iteration': self.iteration,
            'cost': self.cost,
            'n_routes': self.n_routes,
            'cv': self.cv,
            'objective': self.objective,
            'customer_assignments': {str(k): v for k, v in self.customer_assignments.items()},
            'route_customers': {str(k): v for k, v in self.route_customers.items()},
        }


@dataclass
class DivergenceMetrics:
    """Metrics comparing two solutions."""
    cost_diff: float
    cost_diff_pct: float
    cv_diff: float
    route_count_diff: int
    customer_assignment_diff: int  # Number of customers in different routes
    route_structure_diff: float  # Jaccard distance of route structures
    is_identical: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


class DivergenceTracker:
    """Tracks solution divergence between methods."""
    
    def __init__(self, checkpoints: List[int] = None):
        """Initialize tracker.
        
        Args:
            checkpoints: Iterations at which to capture snapshots
        """
        if checkpoints is None:
            checkpoints = [0, 10, 25, 50, 100, 200, 500, 1000]
        self.checkpoints = sorted(checkpoints)
        self.snapshots: Dict[str, List[SolutionSnapshot]] = defaultdict(list)
        self.divergence_log: List[Dict] = []
    
    def capture_snapshot(self, method: str, iteration: int, solution: Solution, 
                         instance, objective: float) -> Optional[SolutionSnapshot]:
        """Capture solution snapshot if at checkpoint.
        
        Args:
            method: Method name
            iteration: Current iteration
            solution: Current solution
            instance: VRP instance
            objective: Current objective value
            
        Returns:
            Snapshot if at checkpoint, None otherwise
        """
        if iteration not in self.checkpoints:
            return None
        
        # Compute CV
        route_costs = np.array([r.cost for r in solution.routes if r.customers])
        cv = coefficient_of_variation(route_costs) if len(route_costs) > 1 else 0.0
        
        # Extract customer assignments
        customer_assignments = {}
        route_customers = {}
        
        for route in solution.routes:
            if not route.customers:
                continue
            
            route_id = route.vehicle_id
            route_customers[route_id] = route.customers.copy()
            
            for pos, customer_id in enumerate(route.customers):
                customer_assignments[customer_id] = (route_id, pos)
        
        snapshot = SolutionSnapshot(
            iteration=iteration,
            cost=solution.total_cost,
            n_routes=len([r for r in solution.routes if r.customers]),
            cv=cv,
            objective=objective,
            customer_assignments=customer_assignments,
            route_customers=route_customers,
        )
        
        self.snapshots[method].append(snapshot)
        return snapshot
    
    def compare_snapshots(self, snapshot1: SolutionSnapshot, 
                         snapshot2: SolutionSnapshot) -> DivergenceMetrics:
        """Compare two snapshots and compute divergence metrics.
        
        Args:
            snapshot1: First snapshot
            snapshot2: Second snapshot
            
        Returns:
            DivergenceMetrics object
        """
        # Cost difference
        cost_diff = abs(snapshot1.cost - snapshot2.cost)
        cost_diff_pct = (cost_diff / max(snapshot1.cost, snapshot2.cost, 1.0)) * 100
        
        # CV difference
        cv_diff = abs(snapshot1.cv - snapshot2.cv)
        
        # Route count difference
        route_count_diff = abs(snapshot1.n_routes - snapshot2.n_routes)
        
        # Customer assignment difference
        # Count how many customers are in different routes
        assignment_diff = 0
        all_customers = set(snapshot1.customer_assignments.keys()) | set(snapshot2.customer_assignments.keys())
        
        for customer_id in all_customers:
            route1 = snapshot1.customer_assignments.get(customer_id, (None, None))[0]
            route2 = snapshot2.customer_assignments.get(customer_id, (None, None))[0]
            if route1 != route2:
                assignment_diff += 1
        
        # Route structure difference (Jaccard distance)
        routes1 = set(frozenset(customers) for customers in snapshot1.route_customers.values())
        routes2 = set(frozenset(customers) for customers in snapshot2.route_customers.values())
        
        intersection = len(routes1 & routes2)
        union = len(routes1 | routes2)
        route_structure_diff = 1.0 - (intersection / union) if union > 0 else 0.0
        
        # Check if identical
        is_identical = (
            cost_diff < 1e-6 and
            cv_diff < 1e-6 and
            route_count_diff == 0 and
            assignment_diff == 0
        )
        
        return DivergenceMetrics(
            cost_diff=cost_diff,
            cost_diff_pct=cost_diff_pct,
            cv_diff=cv_diff,
            route_count_diff=route_count_diff,
            customer_assignment_diff=assignment_diff,
            route_structure_diff=route_structure_diff,
            is_identical=is_identical,
        )
    
    def analyze_divergence(self) -> Dict:
        """Analyze divergence across all methods and checkpoints.
        
        Returns:
            Dictionary with divergence analysis
        """
        methods = list(self.snapshots.keys())
        if len(methods) < 2:
            return {"error": "Need at least 2 methods to compare"}
        
        analysis = {
            'methods': methods,
            'checkpoints': self.checkpoints,
            'divergence_by_iteration': {},
            'first_divergence': {},
            'divergence_evolution': [],
        }
        
        # Compare each pair of methods at each checkpoint
        for iteration in self.checkpoints:
            iteration_snapshots = {}
            for method in methods:
                # Find snapshot at this iteration
                snapshot = next(
                    (s for s in self.snapshots[method] if s.iteration == iteration),
                    None
                )
                if snapshot:
                    iteration_snapshots[method] = snapshot
            
            if len(iteration_snapshots) < 2:
                continue
            
            # Compare all pairs
            comparisons = {}
            method_list = list(iteration_snapshots.keys())
            for i, method1 in enumerate(method_list):
                for method2 in method_list[i+1:]:
                    metrics = self.compare_snapshots(
                        iteration_snapshots[method1],
                        iteration_snapshots[method2]
                    )
                    key = f"{method1}_vs_{method2}"
                    comparisons[key] = metrics.to_dict()
                    
                    # Track first divergence
                    if not metrics.is_identical:
                        if key not in analysis['first_divergence']:
                            analysis['first_divergence'][key] = {
                                'iteration': iteration,
                                'metrics': metrics.to_dict(),
                            }
            
            analysis['divergence_by_iteration'][iteration] = comparisons
            
            # Track evolution
            if comparisons:
                evolution_entry = {
                    'iteration': iteration,
                    'comparisons': comparisons,
                }
                analysis['divergence_evolution'].append(evolution_entry)
        
        return analysis
    
    def save_results(self, output_path: Path):
        """Save divergence analysis to JSON file."""
        analysis = self.analyze_divergence()
        
        # Also save raw snapshots
        snapshots_dict = {}
        for method, snapshots in self.snapshots.items():
            snapshots_dict[method] = [s.to_dict() for s in snapshots]
        
        output = {
            'analysis': analysis,
            'snapshots': snapshots_dict,
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved divergence analysis to {output_path}")


class ALNSWithTracking(ALNS):
    """ALNS solver with divergence tracking."""
    
    def __init__(self, *args, tracker: Optional[DivergenceTracker] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker
    
    def solve(self, instance, initial_solution=None):
        """Solve with divergence tracking."""
        # Call parent solve but intercept at checkpoints
        # We'll need to modify the solve loop to call tracker
        solution, stats = super().solve(instance, initial_solution)
        
        # If tracker is provided, we need to hook into iterations
        # For now, we'll track at the end
        if self.tracker:
            objective = self._compute_objective(solution, instance)
            self.tracker.capture_snapshot(
                method=getattr(self, '_method_name', 'unknown'),
                iteration=self.config.max_iterations,
                solution=solution,
                instance=instance,
                objective=objective,
            )
        
        return solution, stats


def run_method_with_tracking(
    instance_path: Path,
    method: str,
    seed: int,
    max_iterations: int = 200,
    tracker: Optional[DivergenceTracker] = None,
) -> Tuple[Solution, Dict]:
    """Run a method with divergence tracking.
    
    Args:
        instance_path: Path to instance file
        method: Method name ('alns', 'alns_hnsw', 'hero')
        seed: Random seed
        max_iterations: Maximum iterations
        tracker: Divergence tracker
        
    Returns:
        Tuple of (solution, statistics)
    """
    from src.models.parsers import parse_instance
    from src.heuristics.constructive import nearest_neighbor
    from src.heuristics.alns import ALNS, ALNSConfig
    from src.hnsw.manager import HNSWManager, HNSWManagerConfig
    from src.hnsw.index import HNSWConfig
    from src.hnsw.features import FeatureConfig
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Parse instance
    instance = parse_instance(instance_path)
    
    # Build initial solution
    initial = nearest_neighbor(instance)
    
    # Configure ALNS
    use_fairness = (method == "hero")
    
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
    
    # Configure HNSW
    hnsw_manager = None
    if method in ["alns_hnsw", "hero"]:
        feature_config = FeatureConfig(
            use_fairness_features=use_fairness,
            use_subsequence_features=True,
        )
        
        dim = feature_config.feature_dim
        hnsw_config = HNSWConfig(dim=dim, M=16, ef_construction=200, ef_search=50)
        manager_config = HNSWManagerConfig(
            hnsw_config=hnsw_config,
            feature_config=feature_config,
        )
        hnsw_manager = HNSWManager(config=manager_config)
    
    # Create ALNS
    alns = ALNS(config=alns_config, hnsw_manager=hnsw_manager)
    
    # Create callback for tracking
    callback = None
    if tracker:
        callback = _create_callback(tracker, method, instance)
    
    # Solve with callback
    solution, stats = alns.solve(instance, initial, iteration_callback=callback)
    
    return solution, stats


def _create_callback(tracker: Optional[DivergenceTracker], method: str, instance):
    """Create iteration callback for tracking."""
    if tracker is None:
        return None
    
    def callback(iteration: int, solution: Solution, objective: float):
        """Callback function called at each iteration."""
        tracker.capture_snapshot(method, iteration, solution, instance, objective)
    return callback


def track_divergence(
    instance_path: Path,
    seed: int = 42,
    max_iterations: int = 200,
    checkpoints: List[int] = None,
    output_path: Optional[Path] = None,
) -> DivergenceTracker:
    """Track divergence between all methods on an instance.
    
    Args:
        instance_path: Path to instance file
        seed: Random seed
        max_iterations: Maximum iterations
        checkpoints: Iterations at which to capture snapshots
        output_path: Path to save results (optional)
        
    Returns:
        DivergenceTracker with results
    """
    print("="*80)
    print("METHOD DIVERGENCE TRACKING")
    print("="*80)
    print(f"Instance: {instance_path.name}")
    print(f"Seed: {seed}")
    print(f"Max iterations: {max_iterations}")
    print()
    
    # Create tracker
    tracker = DivergenceTracker(checkpoints=checkpoints)
    
    # Run each method
    methods = ["alns", "alns_hnsw", "hero"]
    
    for method in methods:
        print(f"Running {method}...")
        solution, stats = run_method_with_tracking(
            instance_path=instance_path,
            method=method,
            seed=seed,
            max_iterations=max_iterations,
            tracker=tracker,
        )
        print(f"  {method}: cost={solution.total_cost:.2f}, routes={len([r for r in solution.routes if r.customers])}")
    
    # Analyze divergence
    print("\n" + "="*80)
    print("DIVERGENCE ANALYSIS")
    print("="*80)
    
    analysis = tracker.analyze_divergence()
    
    # Print first divergence
    if 'first_divergence' in analysis and analysis['first_divergence']:
        print("\nFirst Divergence:")
        for key, info in analysis['first_divergence'].items():
            print(f"  {key}:")
            print(f"    Iteration: {info['iteration']}")
            print(f"    Cost diff: {info['metrics']['cost_diff']:.2f} ({info['metrics']['cost_diff_pct']:.2f}%)")
            print(f"    CV diff: {info['metrics']['cv_diff']:.4f}")
            print(f"    Route diff: {info['metrics']['route_count_diff']}")
            print(f"    Assignment diff: {info['metrics']['customer_assignment_diff']} customers")
    else:
        print("\n⚠️  No divergence detected - all methods produced identical results!")
    
    # Print divergence by iteration
    print("\nDivergence by Iteration:")
    print("-"*80)
    for iteration in sorted(analysis.get('divergence_by_iteration', {}).keys()):
        comparisons = analysis['divergence_by_iteration'][iteration]
        print(f"\nIteration {iteration}:")
        for key, metrics in comparisons.items():
            identical = "✅ IDENTICAL" if metrics['is_identical'] else "❌ DIFFERENT"
            print(f"  {key}: {identical}")
            if not metrics['is_identical']:
                print(f"    Cost: {metrics['cost_diff']:.2f} ({metrics['cost_diff_pct']:.2f}%)")
                print(f"    CV: {metrics['cv_diff']:.4f}")
                print(f"    Routes: {metrics['route_count_diff']}")
                print(f"    Assignments: {metrics['customer_assignment_diff']} customers")
    
    # Save results
    if output_path:
        tracker.save_results(output_path)
    else:
        # Default output path
        output_path = Path("results/divergence") / f"{instance_path.stem}_seed{seed}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tracker.save_results(output_path)
    
    return tracker


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Track method divergence")
    parser.add_argument("--instance", type=str, required=True, help="Instance file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iterations", type=int, default=200, help="Maximum iterations")
    parser.add_argument("--checkpoints", type=str, default="0,10,25,50,100,200", 
                       help="Comma-separated list of checkpoint iterations")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    
    args = parser.parse_args()
    
    instance_path = Path(args.instance)
    if not instance_path.exists():
        print(f"Error: Instance file not found: {args.instance}")
        sys.exit(1)
    
    checkpoints = [int(x.strip()) for x in args.checkpoints.split(',')]
    output_path = Path(args.output) if args.output else None
    
    tracker = track_divergence(
        instance_path=instance_path,
        seed=args.seed,
        max_iterations=args.max_iterations,
        checkpoints=checkpoints,
        output_path=output_path,
    )

