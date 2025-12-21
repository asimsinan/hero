"""Experiment runner for HNSW-FairVRP benchmarking.

This module provides the main experiment runner that:
- Loads benchmark instances
- Runs ALNS with different configurations
- Collects metrics (cost, fairness, runtime)
- Saves results for analysis

Can be configured via YAML files or command-line arguments.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import time
import logging
import argparse
from datetime import datetime

import numpy as np

# Local imports
from ..models.problem import VRPInstance
from ..models.parsers import parse_instance
from ..models.fairness import coefficient_of_variation, jains_fairness_index
from ..heuristics.constructive import create_initial_solution
from ..heuristics.alns import ALNS, ALNSConfig, create_alns
from ..hnsw.manager import create_hnsw_manager

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.
    
    Attributes:
        name: Experiment name
        instance_path: Path to benchmark instance file
        seed: Random seed
        
        # ALNS parameters
        max_iterations: Maximum ALNS iterations
        max_time: Maximum runtime (seconds)
        initial_temperature: SA starting temperature
        cooling_rate: SA cooling rate
        
        # Multi-objective weights
        alpha: Cost weight
        beta: CV (driver fairness) weight  
        gamma: Jain's (customer fairness) weight
        
        # HNSW parameters
        use_hnsw: Enable HNSW acceleration
        hnsw_k: Number of HNSW candidates
        hnsw_M: HNSW connectivity parameter
        
        # Output
        output_dir: Directory for results
        save_solution: Save final solution to file
    """
    name: str = "experiment"
    instance_path: Optional[str] = None
    seed: int = 42
    
    max_iterations: int = 10000
    max_time: Optional[float] = None
    initial_temperature: float = 2000.0  # Higher for better exploration
    cooling_rate: float = 0.9997
    
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 0.2
    
    use_hnsw: bool = True  # HNSW-guided route selection
    hnsw_k: int = 10
    hnsw_M: int = 16
    
    output_dir: str = "results"
    save_solution: bool = True


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    
    # Instance info
    instance_name: str = ""
    n_customers: int = 0
    n_vehicles: int = 0
    
    # Solution quality
    initial_cost: float = 0.0
    final_cost: float = 0.0
    improvement: float = 0.0
    
    # Fairness metrics
    driver_cv: float = 0.0  # Coefficient of variation
    customer_jain: float = 0.0  # Jain's index
    
    # Performance
    runtime: float = 0.0
    iterations: int = 0
    
    # Solution details
    n_routes_used: int = 0
    customers_served: int = 0
    
    # Operator statistics
    destroy_usage: Dict[str, int] = field(default_factory=dict)
    repair_usage: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'instance_name': self.instance_name,
            'n_customers': self.n_customers,
            'n_vehicles': self.n_vehicles,
            'initial_cost': self.initial_cost,
            'final_cost': self.final_cost,
            'improvement': self.improvement,
            'driver_cv': self.driver_cv,
            'customer_jain': self.customer_jain,
            'runtime': self.runtime,
            'iterations': self.iterations,
            'n_routes_used': self.n_routes_used,
            'customers_served': self.customers_served,
            'destroy_usage': self.destroy_usage,
            'repair_usage': self.repair_usage,
            'config': asdict(self.config),
        }
        return result


class ExperimentRunner:
    """Runs experiments and collects results."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[ExperimentResult] = []
    
    def run(self, instance: Optional[VRPInstance] = None) -> ExperimentResult:
        """Run a single experiment.
        
        Args:
            instance: VRP instance (loaded from config.instance_path if None)
            
        Returns:
            ExperimentResult with all metrics
        """
        logger.info(f"Starting experiment: {self.config.name}")
        
        # Load instance
        if instance is None:
            if self.config.instance_path is None:
                raise ValueError("Either instance or instance_path must be provided")
            instance = parse_instance(Path(self.config.instance_path))
        
        # Create initial solution
        initial_solution = create_initial_solution(instance, method="clarke_wright")
        initial_solution.compute_cost()
        
        # Setup HNSW manager
        hnsw_manager = None
        if self.config.use_hnsw:
            hnsw_manager = create_hnsw_manager(
                k_candidates=self.config.hnsw_k,
                M=self.config.hnsw_M,
            )
        
        # Create ALNS solver
        alns = create_alns(
            max_iterations=self.config.max_iterations,
            max_time=self.config.max_time,
            initial_temperature=self.config.initial_temperature,
            cooling_rate=self.config.cooling_rate,
            alpha=self.config.alpha,
            beta=self.config.beta,
            gamma=self.config.gamma,
            hnsw_manager=hnsw_manager,
            seed=self.config.seed,
            verbose=True,
        )
        
        # Run ALNS
        logger.info(f"Running ALNS on {instance.name} ({instance.n_customers} customers)")
        best_solution, stats = alns.solve(instance, initial_solution=initial_solution)
        
        # Compute fairness metrics
        route_costs = np.array([r.cost for r in best_solution.routes if not r.is_empty()])
        driver_cv = coefficient_of_variation(route_costs) if len(route_costs) > 1 else 0.0
        
        # Customer fairness (from waiting times if available)
        customer_jain = 1.0  # Default perfect fairness
        
        # Build result
        result = ExperimentResult(
            config=self.config,
            instance_name=instance.name,
            n_customers=instance.n_customers,
            n_vehicles=instance.n_vehicles,
            initial_cost=stats.initial_cost,
            final_cost=stats.best_cost,
            improvement=stats.improvement,
            driver_cv=driver_cv,
            customer_jain=customer_jain,
            runtime=stats.runtime,
            iterations=stats.iterations,
            n_routes_used=best_solution.n_routes_used(),
            customers_served=best_solution.n_customers_served(),
            destroy_usage=stats.destroy_usage,
            repair_usage=stats.repair_usage,
        )
        
        self.results.append(result)
        
        # Save results
        if self.config.output_dir:
            self._save_results(result, best_solution)
        
        return result
    
    def run_benchmark_suite(
        self,
        instance_paths: List[str],
        seeds: List[int] = [42, 123, 456],
    ) -> List[ExperimentResult]:
        """Run experiments on multiple instances with multiple seeds.
        
        Args:
            instance_paths: List of paths to benchmark instances
            seeds: List of random seeds for each run
            
        Returns:
            List of all results
        """
        all_results = []
        
        for path in instance_paths:
            for seed in seeds:
                self.config.instance_path = path
                self.config.seed = seed
                self.config.name = f"{Path(path).stem}_seed{seed}"
                
                try:
                    result = self.run()
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error running {path} with seed {seed}: {e}")
        
        return all_results
    
    def _save_results(self, result: ExperimentResult, solution) -> None:
        """Save experiment results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.config.name}_{timestamp}"
        
        # Save JSON results
        json_path = output_dir / f"{base_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {json_path}")
        
        # Save solution if requested
        if self.config.save_solution:
            sol_path = output_dir / f"{base_name}_solution.json"
            solution_data = {
                'routes': [
                    {
                        'vehicle_id': r.vehicle_id,
                        'customers': r.customers,
                        'cost': r.cost,
                        'load': r.load,
                    }
                    for r in solution.routes
                ],
                'total_cost': solution.total_cost,
            }
            with open(sol_path, 'w') as f:
                json.dump(solution_data, f, indent=2)


def run_quick_benchmark(
    n_customers: int = 50,
    n_vehicles: int = 10,
    iterations: int = 1000,
    seed: int = 42,
) -> ExperimentResult:
    """Run a quick benchmark on a random instance.
    
    Args:
        n_customers: Number of customers
        n_vehicles: Number of vehicles
        iterations: ALNS iterations
        seed: Random seed
        
    Returns:
        ExperimentResult
    """
    # Create random instance
    instance = VRPInstance.create_random(
        n_customers=n_customers,
        n_vehicles=n_vehicles,
        seed=seed,
    )
    
    # Configure experiment
    config = ExperimentConfig(
        name=f"quick_benchmark_n{n_customers}",
        max_iterations=iterations,
        seed=seed,
        use_hnsw=True,
        output_dir="results/quick",
    )
    
    # Run
    runner = ExperimentRunner(config)
    return runner.run(instance)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run HNSW-FairVRP experiments")
    parser.add_argument("--instance", type=str, help="Path to instance file")
    parser.add_argument("--iterations", type=int, default=10000, help="Max iterations")
    parser.add_argument("--time-limit", type=float, default=None, help="Time limit (seconds)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", type=float, default=1.0, help="Cost weight")
    parser.add_argument("--beta", type=float, default=0.3, help="CV weight")
    parser.add_argument("--gamma", type=float, default=0.2, help="Jain's weight")
    parser.add_argument("--no-hnsw", action="store_true", help="Disable HNSW")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.quick:
        result = run_quick_benchmark(
            n_customers=50,
            n_vehicles=10,
            iterations=args.iterations,
            seed=args.seed,
        )
        print(f"\nQuick Benchmark Results:")
        print(f"  Initial cost: {result.initial_cost:.2f}")
        print(f"  Final cost: {result.final_cost:.2f}")
        print(f"  Improvement: {result.improvement:.2%}")
        print(f"  Runtime: {result.runtime:.2f}s")
        return
    
    if args.instance:
        config = ExperimentConfig(
            name=Path(args.instance).stem,
            instance_path=args.instance,
            max_iterations=args.iterations,
            max_time=args.time_limit,
            seed=args.seed,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            use_hnsw=not args.no_hnsw,
            output_dir=args.output,
        )
        
        runner = ExperimentRunner(config)
        result = runner.run()
        
        print(f"\nExperiment Results:")
        print(f"  Instance: {result.instance_name}")
        print(f"  Initial cost: {result.initial_cost:.2f}")
        print(f"  Final cost: {result.final_cost:.2f}")
        print(f"  Improvement: {result.improvement:.2%}")
        print(f"  Driver CV: {result.driver_cv:.4f}")
        print(f"  Runtime: {result.runtime:.2f}s")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

