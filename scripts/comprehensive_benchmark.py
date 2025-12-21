#!/usr/bin/env python3
"""
Comprehensive Benchmark Experiments for HNSW-FairVRP

Tests on ALL available benchmarks:
- Solomon VRPTW (n=100)
- Homberger Extended (n=200-1000)  
- CVRP X Instances (n=101-1001)
- Euro-NeurIPS 2022 Real-World (n=200-880)

With progress tracking and result saving.
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import traceback

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'benchmark_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Seeds for statistical significance
    seeds: List[int] = None
    
    # ALNS parameters
    max_iterations: int = 100
    segment_size: int = 10
    
    # Instance selection (for quick tests, set to smaller numbers)
    max_solomon_instances: int = 10  # e.g., c101, c102, ..., c110
    max_homberger_instances: int = 5
    max_cvrp_instances: int = 10
    max_euro_instances: int = 10
    
    # Size limits (skip very large instances for quick runs)
    max_customers: int = 1000
    
    # Methods to run (None = all)
    methods: List[str] = None
    
    # Beta values to test for hero method (if None, uses default 0.2)
    # If specified, hero method will be expanded to hero_beta0.2, hero_beta0.4, etc.
    beta_values: List[float] = None
    
    # Output directory
    output_dir: Path = None
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 1011]  # 5 seeds by default
        if self.methods is None:
            self.methods = ["alns", "alns_hnsw", "hero", "ortools", "pyvrp"]
        if self.output_dir is None:
            # Use absolute path relative to script location to avoid confusion
            script_dir = Path(__file__).parent.parent
            self.output_dir = script_dir / "results" / "benchmark"
        elif isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
            # If relative path, make it relative to script location
            if not self.output_dir.is_absolute():
                script_dir = Path(__file__).parent.parent
                self.output_dir = script_dir / self.output_dir


@dataclass 
class ExperimentResult:
    """Result from a single experiment run"""
    instance_name: str
    instance_type: str  # solomon, homberger, cvrp, euro
    n_customers: int
    method: str  # alns, alns_hnsw, hero
    seed: int
    
    # Solution quality
    cost: float
    n_routes: int
    cv: float  # coefficient of variation
    
    # Performance
    time_seconds: float
    
    # HNSW specific
    hnsw_feasibility_rate: Optional[float] = None
    
    # Metadata
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ProgressTracker:
    """Track and display experiment progress"""
    
    def __init__(self, total_experiments: int):
        self.total = total_experiments
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.results: List[ExperimentResult] = []
        
    def update(self, result: Optional[ExperimentResult] = None, error: bool = False):
        """Update progress"""
        if error:
            self.failed += 1
        else:
            self.completed += 1
            if result:
                self.results.append(result)
        
        self._display_progress()
    
    def _display_progress(self):
        """Display current progress"""
        elapsed = time.time() - self.start_time
        done = self.completed + self.failed
        pct = 100 * done / self.total if self.total > 0 else 0
        
        # Estimate time remaining
        if done > 0:
            eta_seconds = (elapsed / done) * (self.total - done)
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."
        
        # Progress bar
        bar_len = 40
        filled = int(bar_len * done / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (bar_len - filled)
        
        print(f"\r[{bar}] {pct:5.1f}% ({done}/{self.total}) | "
              f"✓{self.completed} ✗{self.failed} | "
              f"Elapsed: {self._format_time(elapsed)} | ETA: {eta_str}    ", 
              end='', flush=True)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours, rem = divmod(int(seconds), 3600)
        mins, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours}h{mins:02d}m{secs:02d}s"
        elif mins > 0:
            return f"{mins}m{secs:02d}s"
        else:
            return f"{secs}s"
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        elapsed = time.time() - self.start_time
        
        summary = {
            "total_experiments": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "elapsed_seconds": elapsed,
            "elapsed_formatted": self._format_time(elapsed),
        }
        
        if self.results:
            # Group by method
            by_method = {}
            for r in self.results:
                if r.method not in by_method:
                    by_method[r.method] = []
                by_method[r.method].append(r)
            
            summary["by_method"] = {}
            for method, results in by_method.items():
                costs = [r.cost for r in results if r.cost > 0]
                cvs = [r.cv for r in results if r.cv >= 0]
                times = [r.time_seconds for r in results if r.time_seconds > 0]
                
                summary["by_method"][method] = {
                    "count": len(results),
                    "avg_cost": np.mean(costs) if costs else np.nan,
                    "avg_cv": np.mean(cvs) if cvs else np.nan,
                    "avg_time": np.mean(times) if times else np.nan,
                }
        
        return summary


def discover_instances(config: ExperimentConfig) -> Dict[str, List[Path]]:
    """Discover all benchmark instances"""
    base_dir = Path(__file__).parent.parent / "data"
    
    instances = {
        "solomon": [],
        "homberger": [],
        "cvrp": [],
        "euro": [],
    }
    
    # Solomon instances (check multiple possible locations)
    solomon_dirs = [
        base_dir / "solomon",
        base_dir / "benchmarks" / "solomon",
        base_dir / "benchmarks" / "solomon_original",
    ]
    for solomon_dir in solomon_dirs:
        if solomon_dir.exists():
            # Try both .txt and .TXT extensions
            files = sorted(list(solomon_dir.glob("*.txt")) + list(solomon_dir.glob("*.TXT")))
            if files:
                instances["solomon"] = files[:config.max_solomon_instances]
                logger.info(f"Found {len(instances['solomon'])} Solomon instances in {solomon_dir}")
                break
    
    # Homberger instances (check multiple possible locations)
    homberger_dirs = [
        base_dir / "homberger",
        base_dir / "benchmarks" / "homberger",
    ]
    for homberger_dir in homberger_dirs:
        if homberger_dir.exists():
            files = sorted(list(homberger_dir.glob("*.txt")) + list(homberger_dir.glob("*.TXT")))
            if files:
                instances["homberger"] = files[:config.max_homberger_instances]
                logger.info(f"Found {len(instances['homberger'])} Homberger instances in {homberger_dir}")
                break
    
    # CVRP X instances
    cvrp_dir = base_dir / "benchmarks" / "cvrp"
    if cvrp_dir.exists():
        # Sort by size (extract n from X-n{N}-k{K}.vrp)
        files = list(cvrp_dir.glob("*.vrp"))
        files.sort(key=lambda f: int(f.stem.split('-')[1][1:]) if '-' in f.stem else 0)
        # Filter by size
        filtered = []
        for f in files:
            try:
                n = int(f.stem.split('-')[1][1:])
                if n <= config.max_customers:
                    filtered.append(f)
            except:
                filtered.append(f)
        instances["cvrp"] = filtered[:config.max_cvrp_instances]
        logger.info(f"Found {len(instances['cvrp'])} CVRP instances (≤{config.max_customers} customers)")
    
    # Euro-NeurIPS instances
    euro_dir = base_dir / "benchmarks" / "euro_neurips_2022"
    if euro_dir.exists():
        # Sort by size (extract n from name)
        files = list(euro_dir.glob("*.txt"))
        files.sort(key=lambda f: int(f.stem.split('-n')[1].split('-')[0]) if '-n' in f.stem else 0)
        # Filter by size
        filtered = []
        for f in files:
            try:
                n = int(f.stem.split('-n')[1].split('-')[0])
                if n <= config.max_customers:
                    filtered.append(f)
            except:
                pass
        instances["euro"] = filtered[:config.max_euro_instances]
        logger.info(f"Found {len(instances['euro'])} Euro-NeurIPS instances (≤{config.max_customers} customers)")
    
    return instances


def run_pyvrp_experiment(
    instance_path: Path,
    instance_type: str,
    seed: int,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Run PyVRP (HGS) solver - SOTA baseline
    
    Note: PyVRP only supports VRPLIB format. For other formats (Solomon, etc.),
    we convert to VRPLIB format first.
    """
    import pyvrp
    from pyvrp import Model, read, ProblemData, VehicleType, Client
    from pyvrp.stop import MaxIterations, MaxRuntime
    from src.models.parsers import parse_instance
    import tempfile
    
    start_time = time.time()
    instance = None
    
    try:
        # Try to read directly if VRPLIB format
        if instance_path.suffix.lower() in ['.vrp', '.txt']:
            try:
                pyvrp_instance = read(str(instance_path))
                model = Model.from_data(pyvrp_instance)
            except:
                # Not VRPLIB format, convert from our format
                instance = parse_instance(instance_path)
                pyvrp_instance = _convert_to_pyvrp(instance)
                model = Model.from_data(pyvrp_instance)
        else:
            # Parse our format and convert
            instance = parse_instance(instance_path)
            pyvrp_instance = _convert_to_pyvrp(instance)
            model = Model.from_data(pyvrp_instance)
        
        # Configure solver with similar time budget
        # Use time limit instead of iterations for better control
        from pyvrp.stop import MaxRuntime
        time_limit = max(30, config.max_iterations // 3)  # Similar to OR-Tools
        
        result = model.solve(
            stop=MaxRuntime(time_limit),
            seed=seed,
            display=False,
        )
        
        elapsed = time.time() - start_time
        
        if result and result.is_feasible():
            # PyVRP returns cost in scaled units (integers), need to divide by SCALE
            SCALE = 1000  # Same as in _convert_to_pyvrp
            cost = result.cost() / SCALE
            routes = result.best.routes()
            n_routes = len([r for r in routes if len(r) > 0])
            
            # Calculate CV from route loads
            # PyVRP route.visits() returns client indices (0-indexed, excluding depot)
            workloads = []
            clients_list = list(pyvrp_instance.clients())  # Get clients as list
            for route in routes:
                if len(route) > 0:
                    # Get delivery for each visit
                    load = sum(
                        sum(clients_list[visit].delivery) 
                        for visit in route.visits()
                        if visit < len(clients_list)
                    )
                    workloads.append(load)
            
            cv = np.std(workloads) / np.mean(workloads) if workloads and np.mean(workloads) > 0 else 0
            
            # Log PyVRP result (consistent with ALNS methods)
            logger.info(f"pyvrp result: cost={cost:.2f}, routes={n_routes}, time={elapsed:.2f}s")
        else:
            cost = float('inf')
            n_routes = 0
            cv = 0
            logger.info(f"pyvrp result: cost={cost:.2f}, routes={n_routes}, time={elapsed:.2f}s (FAILED)")
            
    except Exception as e:
        logger.warning(f"PyVRP failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        elapsed = time.time() - start_time
        cost = float('inf')
        n_routes = 0
        cv = 0
        logger.info(f"pyvrp result: cost={cost:.2f}, routes={n_routes}, time={elapsed:.2f}s (FAILED)")
    
    # Get n_customers
    if instance is not None:
        n_customers = instance.n_customers
    else:
        try:
            if 'n' in instance_path.stem:
                n_customers = int(instance_path.stem.split('-n')[1].split('-')[0])
            elif '-' in instance_path.stem:
                n_customers = int(instance_path.stem.split('-')[1][1:])
            else:
                n_customers = 0
        except:
            n_customers = 0
    
    return ExperimentResult(
        instance_name=instance_path.stem,
        instance_type=instance_type,
        n_customers=n_customers,
        method="pyvrp",
        seed=seed,
        cost=cost,
        n_routes=n_routes,
        cv=cv,
        time_seconds=elapsed,
    )


def _convert_to_pyvrp(instance):
    """Convert VRPInstance to PyVRP ProblemData format."""
    from pyvrp import ProblemData, VehicleType, Client, Depot
    import numpy as np
    
    # PyVRP uses integers for time windows and distances
    # Scale factor to convert floats to integers
    SCALE = 1000
    MAX_TIME = 2**63 - 1  # Max int64 value
    
    # Create depot (separate from clients in PyVRP)
    depot = Depot(
        x=float(instance.depot.x),
        y=float(instance.depot.y),
    )
    
    # Create clients (only customers, not depot)
    clients = []
    for customer in instance.customers:
        # Convert time window (handle inf)
        tw_start = int(customer.time_window_start * SCALE) if customer.time_window_start != float('inf') else MAX_TIME
        tw_end = int(customer.time_window_end * SCALE) if customer.time_window_end != float('inf') else MAX_TIME
        
        clients.append(Client(
            x=float(customer.x),
            y=float(customer.y),
            delivery=[customer.demand],  # PyVRP expects list
            pickup=[],
            service_duration=int(customer.service_time * SCALE),
            tw_early=tw_start,
            tw_late=tw_end,
        ))
    
    # Vehicle type (use first vehicle's capacity)
    # PyVRP expects capacity as a list (supports multi-dimensional capacity)
    vehicle_capacity = instance.vehicles[0].capacity if instance.vehicles else 100
    vehicle_type = VehicleType(
        num_available=len(instance.vehicles),
        capacity=[vehicle_capacity],  # Must be a list
    )
    
    # Convert distance/time matrices to integers
    # Note: PyVRP expects matrices for all depots and clients
    # Our matrix includes depot at index 0, so we need to adjust
    dist_matrix = (instance.distance_matrix * SCALE).astype(np.int64)
    time_matrix = (instance.time_matrix * SCALE).astype(np.int64) if instance.time_matrix is not None else dist_matrix
    
    # Create ProblemData
    # PyVRP expects distance_matrices and duration_matrices as lists (one per depot)
    return ProblemData(
        clients=clients,
        depots=[depot],
        vehicle_types=[vehicle_type],
        distance_matrices=[dist_matrix],  # List of matrices, one per depot
        duration_matrices=[time_matrix],  # List of matrices, one per depot
    )


def run_ortools_experiment(
    instance_path: Path,
    instance_type: str,
    seed: int,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Run OR-Tools solver - Industry baseline"""
    from src.models.parsers import parse_instance
    from src.heuristics.ortools_solver import ORToolsSolver
    
    start_time = time.time()
    instance = None
    
    try:
        instance = parse_instance(instance_path)
        
        # ORToolsSolver constructor expects individual parameters, not config object
        solver = ORToolsSolver(
            time_limit_seconds=max(30, config.max_iterations // 3),  # Scale with ALNS iterations
            first_solution_strategy="PATH_CHEAPEST_ARC",
            local_search_metaheuristic="GUIDED_LOCAL_SEARCH",
        )
        solution = solver.solve(instance)
        
        elapsed = time.time() - start_time
        
        # Check if solution is valid
        if solution is None or not solution.is_feasible or solution.n_routes_used() == 0:
            cost = float('inf')
            n_routes = 0
            cv = 0
        else:
            cost = solution.compute_cost()
            n_routes = solution.n_routes_used()
            
            # CV calculation
            workloads = []
            for route in solution.routes:
                if route.customers:
                    load = sum(instance.get_customer(c).demand for c in route.customers)
                    workloads.append(load)
            
            cv = np.std(workloads) / np.mean(workloads) if workloads and np.mean(workloads) > 0 else 0
        
        # Log OR-Tools result (consistent with ALNS methods)
        logger.info(f"ortools result: cost={cost:.2f}, routes={n_routes}, time={elapsed:.2f}s")
        
    except Exception as e:
        logger.debug(f"OR-Tools failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        elapsed = time.time() - start_time
        cost = float('inf')
        n_routes = 0
        cv = 0
        logger.info(f"ortools result: cost={cost:.2f}, routes={n_routes}, time={elapsed:.2f}s (FAILED)")
    
    return ExperimentResult(
        instance_name=instance_path.stem,
        instance_type=instance_type,
        n_customers=instance.n_customers if instance is not None else 0,
        method="ortools",
        seed=seed,
        cost=cost,
        n_routes=n_routes,
        cv=cv,
        time_seconds=elapsed,
    )


def run_single_experiment(
    instance_path: Path,
    instance_type: str,
    method: str,
    seed: int,
    config: ExperimentConfig,
    beta: float = None,
) -> ExperimentResult:
    """Run a single experiment
    
    Args:
        instance_path: Path to instance file
        instance_type: Type of instance (solomon, homberger, etc.)
        method: Method name (alns, alns_hnsw, hero, or hero_betaX.X)
        seed: Random seed
        config: Experiment configuration
        beta: Fairness beta value (if None, extracted from method name or defaults to 0.2)
    """
    
    # Dispatch to specialized methods for external solvers
    if method == "pyvrp":
        return run_pyvrp_experiment(instance_path, instance_type, seed, config)
    elif method == "ortools":
        return run_ortools_experiment(instance_path, instance_type, seed, config)
    
    from src.models.parsers import parse_instance
    from src.models.solution import Solution
    from src.heuristics.constructive import (
        nearest_neighbor,
        clarke_wright_savings,
        sweep_algorithm,
        sequential_insertion,
        create_initial_solution,
    )
    from src.heuristics.alns import ALNS, ALNSConfig
    from src.hnsw.manager import HNSWManager, HNSWManagerConfig
    from src.hnsw.index import HNSWConfig
    from src.hnsw.features import FeatureEncoder, FeatureConfig
    
    # Parse instance
    instance = parse_instance(instance_path)
    
    # Build initial solution using different methods for different algorithms
    # This ensures methods explore different regions of solution space
    initial_solution_methods = {
        "alns": "nearest_neighbor",           # Simple greedy
        "alns_hnsw": "clarke_wright_savings",  # Savings-based
        "hero": "sweep_algorithm",             # Angle-based clustering
    }
    
    # Extract base method name (handle hero_betaX.X format)
    base_method = method
    if "_beta" in method:
        base_method = method.split("_beta")[0]
    
    # Use method-specific seed offset for initial solution generation
    # This ensures each method gets a different starting point even with same base seed
    method_seed_offsets = {
        "alns": 0,           # No offset
        "alns_hnsw": 1000,   # +1000 offset
        "hero": 2000,        # +2000 offset
    }
    initial_seed = seed + method_seed_offsets.get(base_method, 0)
    
    # Set random seed for initial solution generation (before generating initial solution)
    np.random.seed(initial_seed)
    random.seed(initial_seed)
    
    # Get initial solution method for this algorithm
    initial_method = initial_solution_methods.get(base_method, "nearest_neighbor")
    
    # Generate initial solution
    if initial_method == "clarke_wright_savings":
        initial = clarke_wright_savings(instance)
    elif initial_method == "sweep_algorithm":
        initial = sweep_algorithm(instance)
    elif initial_method == "sequential_insertion":
        initial = sequential_insertion(instance)
    else:  # nearest_neighbor (default)
        initial = nearest_neighbor(instance)
    
    initial_cost = initial.compute_cost()
    
    # LOG: Initial solution method and seed used
    logger.debug(f"{method}: Using initial solution method '{initial_method}' with seed={initial_seed} (cost={initial_cost:.2f})")
    
    # Reset random seed to main seed for ALNS (after initial solution generation)
    # This ensures ALNS uses the same seed across methods for fair comparison
    np.random.seed(seed)
    random.seed(seed)
    
    # Extract base method name and beta value
    base_method = method
    if "_beta" in method:
        # Extract beta from method name (e.g., "hero_beta0.4" -> beta=0.4)
        try:
            beta_str = method.split("_beta")[1]
            beta = float(beta_str)
            base_method = method.split("_beta")[0]
        except (ValueError, IndexError):
            logger.warning(f"Could not parse beta from method name '{method}', using default")
            beta = None
    
    # Configure ALNS - set fairness weights based on method
    use_fairness = (base_method == "hero")
    
    # Set default beta if not specified
    if beta is None:
        beta = 0.2 if use_fairness else 0.0
    
    # LOG: Method configuration
    logger.debug(f"{method} config: fairness={use_fairness}, beta={beta}, seed={seed}, initial_cost={initial_cost:.2f}")
    
    alns_config = ALNSConfig(
        max_iterations=config.max_iterations,
        segment_length=config.segment_size,
        min_destroy_fraction=0.1,
        max_destroy_fraction=0.3,
        # Use previous successful temperature settings for better exploration
        # Higher temperature allows algorithm to explore fairness trade-offs
        initial_temperature=2000.0,  # Increased from 100.0 for better exploration
        cooling_rate=0.9997,        # Slower cooling from 0.995 to allow more exploration
        seed=seed,
        # Fairness weights (matching previous successful configuration)
        alpha=0.8 if use_fairness else 1.0,
        beta=beta,  # CV fairness weight (now configurable)
        gamma=0.0,
        verbose=False,
    )
    
    # Configure HNSW if needed
    hnsw_manager = None
    
    if method in ["alns_hnsw", "hero"]:
        feature_config = FeatureConfig(
            use_fairness_features=use_fairness,
            use_subsequence_features=True,  # Always use subsequence-aware encoding
        )
        
        # Use feature_dim property to get correct dimension
        dim = feature_config.feature_dim  # 14D with fairness, 11D without
        # Improved HNSW config: M=96 and ef_construction=400 for better initial index quality
        # Adaptive M will be applied in HNSWManager.initialize() based on instance size
        # This reduces initial failure rates from 28.5% to <15%
        hnsw_config = HNSWConfig(dim=dim, M=96, ef_construction=400, ef_search=100)
        
        manager_config = HNSWManagerConfig(
            hnsw_config=hnsw_config,
            feature_config=feature_config,
        )
        hnsw_manager = HNSWManager(config=manager_config)
        logger.debug(f"{method}: HNSW manager created (dim={dim}D, fairness_features={use_fairness})")
    else:
        logger.debug(f"{method}: No HNSW manager (using brute force)")
    
    # Run ALNS
    alns = ALNS(config=alns_config, hnsw_manager=hnsw_manager)
    
    # LOG: Verify HNSW is set in repair operators
    if hnsw_manager is not None:
        hnsw_ops = [op for op in alns.repair_operators if hasattr(op, 'hnsw_manager') and op.hnsw_manager is not None]
        logger.debug(f"{method}: {len(hnsw_ops)}/{len(alns.repair_operators)} repair operators have HNSW manager")
    
    start_time = time.time()
    solution, stats = alns.solve(instance, initial.copy())
    elapsed = time.time() - start_time
    
    # Calculate metrics
    cost = solution.compute_cost()
    n_routes = len([r for r in solution.routes if r.customers])
    
    # LOG: Solution differences
    cost_diff = cost - initial_cost
    logger.info(f"{method} result: cost={cost:.2f} (change={cost_diff:+.2f}), routes={n_routes}, time={elapsed:.2f}s")
    
    # LOG: Check if HNSW was used (if available)
    if hnsw_manager is not None:
        hnsw_stats = hnsw_manager.get_statistics()
        n_queries = hnsw_stats.get('n_queries', 0)
        n_failures = hnsw_stats.get('n_hnsw_failures', 0)
        success_rate = hnsw_stats.get('hnsw_success_rate', 1.0)
        n_cache_hits = hnsw_stats.get('n_cache_hits', 0)
        n_rebuilds = hnsw_stats.get('n_rebuilds', 0)
        avg_query_time = hnsw_stats.get('avg_query_time_ms', 0)
        
        # FIXED: Include failures and success_rate in stats log for accurate failure rate calculation
        logger.info(
            f"{method} HNSW stats: queries={n_queries}, failures={n_failures}, "
            f"success_rate={success_rate*100:.1f}%, cache_hits={n_cache_hits}, "
            f"rebuilds={n_rebuilds}, avg_query_time={avg_query_time:.2f}ms"
        )
        
        if n_queries == 0:
            logger.warning(f"{method}: HNSW manager created but NO queries made! HNSW may not be used.")
        else:
            cache_hit_rate = n_cache_hits / (n_queries + n_cache_hits) if (n_queries + n_cache_hits) > 0 else 0
            logger.info(f"{method} HNSW cache hit rate: {cache_hit_rate:.2%}")
    
    # CV calculation
    workloads = []
    for route in solution.routes:
        if route.customers:
            load = sum(instance.get_customer(c).demand for c in route.customers)
            workloads.append(load)
    
    cv = np.std(workloads) / np.mean(workloads) if workloads and np.mean(workloads) > 0 else 0
    
    return ExperimentResult(
        instance_name=instance_path.stem,
        instance_type=instance_type,
        n_customers=instance.n_customers,
        method=method,
        seed=seed,
        cost=cost,
        n_routes=n_routes,
        cv=cv,
        time_seconds=elapsed,
    )


def run_all_experiments(config: ExperimentConfig) -> List[ExperimentResult]:
    """Run all experiments with progress tracking"""
    
    # Discover instances
    instances = discover_instances(config)
    
    # Expand methods: if "hero" is in methods and beta_values is specified,
    # create separate method entries for each beta value
    methods = config.methods.copy() if config.methods else []
    if "hero" in methods and config.beta_values:
        # Remove "hero" and add hero_betaX.X for each beta value
        methods = [m for m in methods if m != "hero"]
        for beta in config.beta_values:
            methods.append(f"hero_beta{beta}")
        logger.info(f"Expanded hero method to test beta values: {config.beta_values}")
    
    # Calculate total experiments
    total_instances = sum(len(v) for v in instances.values())
    total_experiments = total_instances * len(methods) * len(config.seeds)
    
    logger.info(f"Starting {total_experiments} experiments:")
    logger.info(f"  - {total_instances} instances")
    logger.info(f"  - {len(methods)} methods")
    logger.info(f"  - {len(config.seeds)} seeds")
    
    # Initialize progress tracker
    progress = ProgressTracker(total_experiments)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE HNSW-FairVRP BENCHMARK")
    print("="*80 + "\n")
    
    # Run experiments
    for instance_type, instance_paths in instances.items():
        if not instance_paths:
            continue
            
        print(f"\n{'─'*80}")
        print(f"  {instance_type.upper()} INSTANCES ({len(instance_paths)} files)")
        print(f"{'─'*80}\n")
        
        type_start_count = len(progress.results)
        
        for instance_path in instance_paths:
            for method in methods:
                for seed in config.seeds:
                    try:
                        result = run_single_experiment(
                            instance_path=instance_path,
                            instance_type=instance_type,
                            method=method,
                            seed=seed,
                            config=config,
                        )
                        progress.update(result)
                        
                    except Exception as e:
                        logger.error(f"Error on {instance_path.stem}/{method}/seed{seed}: {e}")
                        logger.debug(traceback.format_exc())
                        progress.update(error=True)
        
        # INCREMENTAL SAVE: Save results after each instance type completes
        # This ensures we don't lose data if the script is interrupted
        type_results = progress.results[type_start_count:]
        if type_results:
            # Ensure output_dir is absolute
            output_dir = config.output_dir
            if not output_dir.is_absolute():
                script_dir = Path(__file__).parent.parent
                output_dir = script_dir / config.output_dir
            
            incremental_dir = output_dir / "incremental"
            incremental_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            try:
                # Save this instance type's results as CSV
                type_file = incremental_dir / f"{instance_type}_{timestamp}.csv"
                with open(type_file, 'w') as f:
                    f.write("instance_name,instance_type,n_customers,method,seed,cost,n_routes,cv,time_seconds\n")
                    for r in type_results:
                        f.write(f"{r.instance_name},{r.instance_type},{r.n_customers},{r.method},"
                                f"{r.seed},{r.cost:.2f},{r.n_routes},{r.cv:.4f},{r.time_seconds:.3f}\n")
                
                # Also save as JSON for easier loading
                type_json_file = incremental_dir / f"{instance_type}_{timestamp}.json"
                with open(type_json_file, 'w') as f:
                    json.dump([asdict(r) for r in type_results], f, indent=2)
                
                # Also save cumulative results so far (both CSV and JSON)
                all_file = incremental_dir / f"all_so_far_{timestamp}.csv"
                with open(all_file, 'w') as f:
                    f.write("instance_name,instance_type,n_customers,method,seed,cost,n_routes,cv,time_seconds\n")
                    for r in progress.results:
                        f.write(f"{r.instance_name},{r.instance_type},{r.n_customers},{r.method},"
                                f"{r.seed},{r.cost:.2f},{r.n_routes},{r.cv:.4f},{r.time_seconds:.3f}\n")
                
                all_json_file = incremental_dir / f"all_so_far_{timestamp}.json"
                with open(all_json_file, 'w') as f:
                    json.dump([asdict(r) for r in progress.results], f, indent=2)
                
                logger.info(f"Saved incremental results for {instance_type} to {type_file} and {type_json_file}")
                print(f"\n  ✓ Saved {len(type_results)} {instance_type} results (CSV + JSON)\n")
            except Exception as e:
                logger.error(f"Failed to save incremental results: {e}")
                # Continue anyway - don't let save failure stop experiments
    
    print("\n\n")  # Clear progress line
    
    return progress.results, progress.summary()


def save_results(results: List[ExperimentResult], summary: Dict[str, Any], output_dir: Path):
    """Save results to files"""
    # Ensure output_dir is absolute and create it
    if not output_dir.is_absolute():
        # If relative, make it relative to script location
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    results_file = output_dir / f"results_{timestamp}.json"
    try:
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        logger.info(f"Saved detailed results to {results_file}")
        print(f"✓ Saved {len(results)} results to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results JSON: {e}")
        raise
    
    # Save summary
    summary_file = output_dir / f"summary_{timestamp}.json"
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")
        print(f"✓ Saved summary to {summary_file}")
    except Exception as e:
        logger.error(f"Failed to save summary JSON: {e}")
        raise
    
    # Save CSV for easy analysis
    csv_file = output_dir / f"results_{timestamp}.csv"
    try:
        with open(csv_file, 'w') as f:
            # Header
            f.write("instance_name,instance_type,n_customers,method,seed,cost,n_routes,cv,time_seconds\n")
            for r in results:
                f.write(f"{r.instance_name},{r.instance_type},{r.n_customers},{r.method},"
                        f"{r.seed},{r.cost:.2f},{r.n_routes},{r.cv:.4f},{r.time_seconds:.3f}\n")
        logger.info(f"Saved CSV to {csv_file}")
        print(f"✓ Saved CSV to {csv_file}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        raise
    
    return results_file, summary_file, csv_file


def print_summary(results: List[ExperimentResult], summary: Dict[str, Any]):
    """Print formatted summary to console"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"\nCompleted: {summary['completed']}/{summary['total_experiments']} experiments")
    print(f"Failed: {summary['failed']}")
    print(f"Total time: {summary['elapsed_formatted']}")
    
    if "by_method" in summary:
        print("\n" + "-"*80)
        print(f"{'Method':<15} | {'Count':<8} | {'Avg Cost':<12} | {'Avg CV':<10} | {'Avg Time':<10}")
        print("-"*80)
        
        for method, stats in summary["by_method"].items():
            avg_cost = stats['avg_cost'] if not np.isnan(stats['avg_cost']) else 0.0
            avg_cv = stats['avg_cv'] if not np.isnan(stats['avg_cv']) else 0.0
            avg_time = stats['avg_time'] if not np.isnan(stats['avg_time']) else 0.0
            print(f"{method:<15} | {stats['count']:<8} | {avg_cost:<12.2f} | "
                  f"{avg_cv:<10.4f} | {avg_time:<10.2f}s")
    
    # Key comparisons
    print("\n" + "-"*80)
    print("KEY COMPARISONS")
    print("-"*80)
    
    bm = summary.get("by_method", {})
    
    # HNSW speedup vs ALNS
    if "alns" in bm and "alns_hnsw" in bm:
        alns_time = bm["alns"]["avg_time"]
        hnsw_time = bm["alns_hnsw"]["avg_time"]
        if not np.isnan(alns_time) and not np.isnan(hnsw_time) and hnsw_time > 0:
            speedup = alns_time / hnsw_time
            print(f"  HNSW vs ALNS Speedup: {speedup:.2f}×")
    
    # Fairness improvement
    if "alns_hnsw" in bm and "hero" in bm:
        hnsw_cv = bm["alns_hnsw"]["avg_cv"]
        fair_cv = bm["hero"]["avg_cv"]
        if not np.isnan(hnsw_cv) and not np.isnan(fair_cv) and hnsw_cv > 0:
            cv_improvement = (hnsw_cv - fair_cv) / hnsw_cv * 100
            print(f"  Fairness CV Improvement: {cv_improvement:.1f}%")
    
    # vs OR-Tools
    if "ortools" in bm and "alns_hnsw" in bm:
        ort_cost = bm["ortools"]["avg_cost"]
        hnsw_cost = bm["alns_hnsw"]["avg_cost"]
        ort_time = bm["ortools"]["avg_time"]
        hnsw_time = bm["alns_hnsw"]["avg_time"]
        if (not np.isnan(ort_cost) and not np.isnan(hnsw_cost) and 
            not np.isnan(ort_time) and not np.isnan(hnsw_time)):
            cost_gap = (hnsw_cost - ort_cost) / ort_cost * 100 if ort_cost > 0 else 0
            time_ratio = ort_time / hnsw_time if hnsw_time > 0 else 0
            print(f"  HNSW vs OR-Tools: {cost_gap:+.1f}% cost, {time_ratio:.1f}× faster")
    
    # vs PyVRP (HGS)
    if "pyvrp" in bm and "alns_hnsw" in bm:
        pyvrp_cost = bm["pyvrp"]["avg_cost"]
        hnsw_cost = bm["alns_hnsw"]["avg_cost"]
        pyvrp_time = bm["pyvrp"]["avg_time"]
        hnsw_time = bm["alns_hnsw"]["avg_time"]
        if (not np.isnan(pyvrp_cost) and not np.isnan(hnsw_cost) and 
            not np.isnan(pyvrp_time) and not np.isnan(hnsw_time) and
            pyvrp_cost > 0 and pyvrp_cost != float('inf')):
            cost_gap = (hnsw_cost - pyvrp_cost) / pyvrp_cost * 100
            time_ratio = pyvrp_time / hnsw_time if hnsw_time > 0 else 0
            print(f"  HNSW vs PyVRP (HGS): {cost_gap:+.1f}% cost, {time_ratio:.1f}× speed diff")
    
    print("\n" + "="*80)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive HNSW-FairVRP benchmarks")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1011],
                       help="Random seeds for experiments")
    parser.add_argument("--max-iterations", type=int, default=100,
                       help="Max ALNS iterations per run")
    parser.add_argument("--max-customers", type=int, default=500,
                       help="Skip instances larger than this")
    parser.add_argument("--max-solomon", type=int, default=10,
                       help="Max Solomon instances to test")
    parser.add_argument("--max-homberger", type=int, default=5,
                       help="Max Homberger instances to test")
    parser.add_argument("--max-cvrp", type=int, default=10,
                       help="Max CVRP X instances to test")
    parser.add_argument("--max-euro", type=int, default=10,
                       help="Max Euro-NeurIPS instances to test")
    parser.add_argument("--output-dir", type=str, default="results/benchmark",
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode (2 instances per type, 2 seeds)")
    parser.add_argument("--methods", type=str, nargs="+", 
                       default=["alns", "alns_hnsw", "hero", "ortools", "pyvrp"],
                       help="Methods to compare (alns, alns_hnsw, hero, ortools, pyvrp)")
    parser.add_argument("--beta-values", type=float, nargs="+", default=None,
                       help="Beta values to test for hero method (e.g., --beta-values 0.2 0.4 0.5). "
                            "If specified, hero method will be expanded to hero_beta0.2, hero_beta0.4, etc.")
    parser.add_argument("--no-external", action="store_true",
                       help="Skip external solvers (ortools, pyvrp)")
    
    args = parser.parse_args()
    
    # Handle method selection
    methods = args.methods
    if args.no_external:
        methods = [m for m in methods if m not in ["ortools", "pyvrp"]]
    
    # Configure experiment
    if args.quick:
        output_dir = Path(args.output_dir)
        config = ExperimentConfig(
            seeds=[42, 123],
            max_iterations=50,
            max_solomon_instances=2,
            max_homberger_instances=2,
            max_cvrp_instances=2,
            max_euro_instances=2,
            max_customers=300,
            methods=methods,
            beta_values=args.beta_values,
            output_dir=output_dir,
        )
        logger.info("Running in QUICK TEST mode")
    else:
        output_dir = Path(args.output_dir)
        config = ExperimentConfig(
            seeds=args.seeds,
            max_iterations=args.max_iterations,
            max_solomon_instances=args.max_solomon,
            max_homberger_instances=args.max_homberger,
            max_cvrp_instances=args.max_cvrp,
            max_euro_instances=args.max_euro,
            max_customers=args.max_customers,
            methods=methods,
            beta_values=args.beta_values,
            output_dir=output_dir,
        )
    
    # Create output directory early to ensure it exists even if experiments fail early
    # Ensure output_dir is absolute
    if not output_dir.is_absolute():
        script_dir = Path(__file__).parent.parent
        output_dir = script_dir / output_dir
        config.output_dir = output_dir  # Update config with absolute path
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "incremental").mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created: {output_dir} (absolute: {output_dir.resolve()})")
    print(f"\n📁 Results will be saved to: {output_dir.resolve()}\n")
    
    # Run experiments with error handling to ensure results are saved
    results = []
    summary = {}
    
    try:
        results, summary = run_all_experiments(config)
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        # Try to get partial results from run_all_experiments if it has progress tracker
        # For now, rely on incremental saves that already happened
        if not results:
            logger.warning("No results available. Check incremental/ directory for partial results.")
        return 1
    except Exception as e:
        logger.error(f"Fatal error in experiments: {e}")
        logger.debug(traceback.format_exc())
        # Results may have been partially saved incrementally
        if not results:
            logger.warning("No results available. Check incremental/ directory for partial results.")
        return 1
    finally:
        # Always try to save final results, even if there was an error
        if results:  # Only save if we have some results
            try:
                save_results(results, summary, config.output_dir)
                logger.info(f"Saved {len(results)} results to {config.output_dir}")
            except Exception as save_error:
                logger.error(f"Failed to save results: {save_error}")
    
    # Print summary
    if results:
        print_summary(results, summary)
    
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

