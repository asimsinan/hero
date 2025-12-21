"""Adaptive Large Neighborhood Search (ALNS) for VRP.

This module implements the ALNS metaheuristic with:
- Simulated Annealing acceptance criterion
- Adaptive operator weight adjustment
- HNSW-accelerated repair operators
- Multi-objective optimization (cost + fairness)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Callable, Dict, Tuple
import numpy as np
import random
import time
import logging
from copy import deepcopy

from .destroy import DestroyOperator, DestroyResult, get_destroy_operators
from .repair import RepairOperator, RepairResult, get_repair_operators
from .constructive import create_initial_solution

if TYPE_CHECKING:
    from ..models.problem import VRPInstance
    from ..models.solution import Solution
    from ..hnsw.manager import HNSWManager

logger = logging.getLogger(__name__)


@dataclass
class ALNSConfig:
    """Configuration for ALNS algorithm.
    
    Attributes:
        max_iterations: Maximum number of iterations
        max_time: Maximum runtime in seconds (None = no limit)
        
        # Destroy parameters
        min_destroy_fraction: Minimum fraction of customers to remove
        max_destroy_fraction: Maximum fraction of customers to remove
        
        # Simulated Annealing parameters
        initial_temperature: Starting temperature
        cooling_rate: Temperature decay per iteration (0-1)
        final_temperature: Stop cooling at this temperature
        
        # Adaptive weight parameters
        segment_length: Iterations per segment for weight update
        sigma_1: Score for new best solution
        sigma_2: Score for improving solution
        sigma_3: Score for accepted non-improving solution
        reaction_factor: How quickly weights adapt (0-1)
        
        # Multi-objective
        alpha: Weight for cost
        beta: Weight for CV (driver fairness)
        gamma: Weight for Jain's index (customer fairness)
        
        # Other
        seed: Random seed
        verbose: Print progress
    """
    max_iterations: int = 10000
    max_time: Optional[float] = None
    
    min_destroy_fraction: float = 0.15
    max_destroy_fraction: float = 0.45
    
    initial_temperature: float = 2000.0  # Higher for better exploration
    cooling_rate: float = 0.9997
    final_temperature: float = 0.01
    
    segment_length: int = 100
    sigma_1: float = 33.0  # New best
    sigma_2: float = 9.0   # Better than current
    sigma_3: float = 13.0  # Accepted non-improving
    reaction_factor: float = 0.1
    
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 0.2
    
    seed: Optional[int] = None
    verbose: bool = True


@dataclass
class ALNSStatistics:
    """Statistics from ALNS run."""
    iterations: int = 0
    runtime: float = 0.0
    initial_cost: float = 0.0
    best_cost: float = 0.0
    improvement: float = 0.0
    
    # Per-operator statistics
    destroy_usage: Dict[str, int] = field(default_factory=dict)
    destroy_scores: Dict[str, float] = field(default_factory=dict)
    repair_usage: Dict[str, int] = field(default_factory=dict)
    repair_scores: Dict[str, float] = field(default_factory=dict)
    
    # Convergence history
    cost_history: List[float] = field(default_factory=list)
    temperature_history: List[float] = field(default_factory=list)
    
    def summary(self) -> str:
        return (
            f"ALNS Statistics:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Runtime: {self.runtime:.2f}s\n"
            f"  Initial cost: {self.initial_cost:.2f}\n"
            f"  Best cost: {self.best_cost:.2f}\n"
            f"  Improvement: {self.improvement:.2%}\n"
        )


@dataclass
class ALNS:
    """Adaptive Large Neighborhood Search algorithm.
    
    The main HNSW-FairVRP solver that combines:
    - Destroy operators to remove customers
    - HNSW-accelerated repair operators to reinsert
    - Simulated Annealing for acceptance
    - Adaptive weight adjustment for operator selection
    """
    config: ALNSConfig = field(default_factory=ALNSConfig)
    
    # Operators
    destroy_operators: List[DestroyOperator] = field(default_factory=list)
    repair_operators: List[RepairOperator] = field(default_factory=list)
    
    # HNSW manager (optional)
    hnsw_manager: Optional['HNSWManager'] = None
    
    # Internal state
    _rng: random.Random = field(init=False, repr=False)
    _destroy_weights: np.ndarray = field(init=False, repr=False)
    _repair_weights: np.ndarray = field(init=False, repr=False)
    _destroy_scores: np.ndarray = field(init=False, repr=False)
    _repair_scores: np.ndarray = field(init=False, repr=False)
    _destroy_counts: np.ndarray = field(init=False, repr=False)
    _repair_counts: np.ndarray = field(init=False, repr=False)
    # Cumulative counts for statistics (not reset)
    _destroy_total: np.ndarray = field(init=False, repr=False)
    _repair_total: np.ndarray = field(init=False, repr=False)
    _destroy_score_total: np.ndarray = field(init=False, repr=False)
    _repair_score_total: np.ndarray = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize random generator and operators."""
        self._rng = random.Random(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Initialize operators if not provided
        if not self.destroy_operators:
            self.destroy_operators = get_destroy_operators(seed=self.config.seed)
        
        if not self.repair_operators:
            self.repair_operators = get_repair_operators(
                hnsw_manager=self.hnsw_manager
            )
        
        # Initialize weights (uniform)
        n_destroy = len(self.destroy_operators)
        n_repair = len(self.repair_operators)
        
        self._destroy_weights = np.ones(n_destroy) / n_destroy
        self._repair_weights = np.ones(n_repair) / n_repair
        
        self._destroy_scores = np.zeros(n_destroy)
        self._repair_scores = np.zeros(n_repair)
        self._destroy_counts = np.zeros(n_destroy)
        self._repair_counts = np.zeros(n_repair)
        
        # Cumulative totals for statistics
        self._destroy_total = np.zeros(n_destroy)
        self._repair_total = np.zeros(n_repair)
        self._destroy_score_total = np.zeros(n_destroy)
        self._repair_score_total = np.zeros(n_repair)
    
    def solve(
        self,
        instance: 'VRPInstance',
        initial_solution: Optional['Solution'] = None,
        iteration_callback: Optional[Callable[[int, 'Solution', float], None]] = None,
    ) -> Tuple['Solution', ALNSStatistics]:
        """Run ALNS to solve the VRP instance.
        
        Args:
            instance: VRP problem instance
            initial_solution: Starting solution (created if None)
            
        Returns:
            Tuple of (best_solution, statistics)
        """
        start_time = time.perf_counter()
        stats = ALNSStatistics()
        
        # Create initial solution
        if initial_solution is None:
            current = create_initial_solution(instance, method="clarke_wright")
        else:
            current = initial_solution.copy()
        
        current.compute_cost()
        
        best = current.copy()
        best_objective = self._compute_objective(best, instance)
        current_objective = best_objective
        
        stats.initial_cost = current.total_cost
        stats.cost_history.append(current.total_cost)
        
        # Initialize HNSW if available
        if self.hnsw_manager is not None:
            self.hnsw_manager.initialize(instance, current)
            # Set manager for HNSW repair operators
            for op in self.repair_operators:
                if hasattr(op, 'set_hnsw_manager'):
                    op.set_hnsw_manager(self.hnsw_manager)
        
        # Initialize SA temperature
        temperature = self.config.initial_temperature
        
        # Call callback at iteration 0 (initial state)
        if iteration_callback:
            iteration_callback(0, current, current_objective)
        
        # Main ALNS loop
        iteration = 0
        while self._should_continue(iteration, start_time):
            iteration += 1
            
            # Select operators
            destroy_idx = self._select_operator(self._destroy_weights)
            repair_idx = self._select_operator(self._repair_weights)
            
            destroy_op = self.destroy_operators[destroy_idx]
            repair_op = self.repair_operators[repair_idx]
            
            # Log operator selection (every 100 iterations to avoid spam)
            if iteration % 100 == 0:
                logger.debug(f"Iter {iteration}: Selected destroy={destroy_op.name}, repair={repair_op.name} (HNSW={hasattr(repair_op, 'hnsw_manager') and repair_op.hnsw_manager is not None})")
            
            # Determine number of customers to remove
            n_customers = current.n_customers_served()
            n_remove = self._get_destroy_size(n_customers)
            
            # Create candidate solution
            candidate = current.copy()
            
            # Destroy
            destroy_result = destroy_op.destroy(candidate, instance, n_remove)
            
            # Repair
            repair_result = repair_op.repair(
                candidate, instance, destroy_result.removed_customers
            )
            
            # If some customers couldn't be inserted, solution is infeasible
            # Penalize heavily but still consider (allows escape from local optima)
            candidate.compute_cost()
            candidate_objective = self._compute_objective(candidate, instance)
            
            # Add penalty for unservable customers
            if repair_result.n_unservable > 0:
                # Heavy penalty to discourage but not eliminate
                candidate_objective += repair_result.n_unservable * 10000
            
            # Acceptance decision
            accept, score = self._accept(
                candidate_objective, current_objective, best_objective, temperature
            )
            
            # Update operator scores (for adaptive weights)
            self._destroy_scores[destroy_idx] += score
            self._repair_scores[repair_idx] += score
            self._destroy_counts[destroy_idx] += 1
            self._repair_counts[repair_idx] += 1
            
            # Update cumulative totals (for statistics)
            self._destroy_total[destroy_idx] += 1
            self._repair_total[repair_idx] += 1
            self._destroy_score_total[destroy_idx] += score
            self._repair_score_total[repair_idx] += score
            
            if accept:
                current = candidate
                current_objective = candidate_objective
                
                # Update HNSW index after accepted move
                # (positions change after destroy/repair, so index becomes stale)
                if self.hnsw_manager is not None:
                    self.hnsw_manager.update_index(current, iteration=iteration)
                
                # Check for new best
                if candidate_objective < best_objective:
                    best = candidate.copy()
                    best_objective = candidate_objective
                    
                    if self.config.verbose and iteration % 100 == 0:
                        from ..models.fairness import coefficient_of_variation
                        route_costs = np.array([r.cost for r in best.routes if not r.is_empty()])
                        best_cv = coefficient_of_variation(route_costs) if len(route_costs) > 1 else 0.0
                        logger.info(
                            f"Iter {iteration}: New best = {best.total_cost:.2f} "
                            f"(obj: {best_objective:.4f}, CV: {best_cv:.4f})"
                        )
            
            # Update temperature
            temperature = max(
                temperature * self.config.cooling_rate,
                self.config.final_temperature
            )
            
            # Adaptive weight update
            if iteration % self.config.segment_length == 0:
                self._update_weights()
                stats.cost_history.append(best.total_cost)
                stats.temperature_history.append(temperature)
            
            # Call iteration callback if provided (for divergence tracking)
            if iteration_callback:
                iteration_callback(iteration, current, current_objective)
            
            stats.iterations = iteration
        
        # Finalize statistics
        stats.runtime = time.perf_counter() - start_time
        stats.best_cost = best.total_cost
        stats.improvement = (stats.initial_cost - stats.best_cost) / stats.initial_cost
        
        # Operator statistics (use cumulative totals)
        for i, op in enumerate(self.destroy_operators):
            stats.destroy_usage[op.name] = int(self._destroy_total[i])
            stats.destroy_scores[op.name] = float(self._destroy_score_total[i])
        
        for i, op in enumerate(self.repair_operators):
            stats.repair_usage[op.name] = int(self._repair_total[i])
            stats.repair_scores[op.name] = float(self._repair_score_total[i])
        
        if self.config.verbose:
            logger.info(stats.summary())
        
        return best, stats
    
    def _should_continue(self, iteration: int, start_time: float) -> bool:
        """Check if search should continue."""
        if iteration >= self.config.max_iterations:
            return False
        
        if self.config.max_time is not None:
            elapsed = time.perf_counter() - start_time
            if elapsed >= self.config.max_time:
                return False
        
        return True
    
    def _select_operator(self, weights: np.ndarray) -> int:
        """Select operator using roulette wheel selection."""
        cumsum = np.cumsum(weights)
        r = self._rng.random() * cumsum[-1]
        return int(np.searchsorted(cumsum, r))
    
    def _get_destroy_size(self, n_customers: int) -> int:
        """Determine number of customers to remove."""
        min_remove = max(1, int(n_customers * self.config.min_destroy_fraction))
        max_remove = max(min_remove, int(n_customers * self.config.max_destroy_fraction))
        return self._rng.randint(min_remove, max_remove)
    
    def _compute_objective(
        self,
        solution: 'Solution',
        instance: 'VRPInstance',
    ) -> float:
        """Compute multi-objective value.
        
        Uses scalarization: f = α×cost + β×CV - γ×Jain
        
        Terms are scaled so they have similar magnitudes:
        - Cost: per-customer average (typically ~10)
        - CV: coefficient of variation (typically 0.2-0.5)
        - Jain: fairness index (typically 0.8-1.0)
        
        With default α=1, β=0.3, γ=0.2, the CV and Jain terms
        contribute meaningfully to the objective.
        """
        from ..models.fairness import coefficient_of_variation, jains_fairness_index
        
        # Cost component - per-customer average
        cost_per_customer = solution.total_cost / max(instance.n_customers, 1)
        
        # Driver fairness (CV of route costs)
        # CV = std/mean, typically in [0.2, 0.6] for VRP solutions
        route_costs = np.array([r.cost for r in solution.routes if not r.is_empty()])
        if len(route_costs) > 1:
            cv = coefficient_of_variation(route_costs)
        else:
            cv = 0.0
        
        # Customer fairness (Jain's index of route workloads)
        # Jain's index is in [1/n, 1], higher = fairer
        if len(route_costs) > 1:
            jain = jains_fairness_index(route_costs)
        else:
            jain = 1.0
        
        # Multi-objective scalarization
        # 
        # To make fairness terms competitive with cost:
        # - CV term: multiply by cost scale factor
        # - Jain term: also scale by cost
        #
        # Objective = α×(cost/n) + β×(cost/n)×CV - γ×(cost/n)×Jain
        #           = (cost/n) × [α + β×CV - γ×Jain]
        #
        # This ensures fairness weights are meaningful
        
        objective = cost_per_customer * (
            self.config.alpha +
            self.config.beta * cv -
            self.config.gamma * jain
        )
        
        # Penalty for unserved customers
        n_unserved = instance.n_customers - solution.n_customers_served()
        if n_unserved > 0:
            objective += n_unserved * 1000  # Heavy penalty
        
        return objective
    
    def _accept(
        self,
        candidate_obj: float,
        current_obj: float,
        best_obj: float,
        temperature: float,
    ) -> Tuple[bool, float]:
        """Simulated Annealing acceptance criterion.
        
        Returns:
            Tuple of (accept, score) where score is for operator update
        """
        # New best
        if candidate_obj < best_obj:
            return True, self.config.sigma_1
        
        # Better than current
        if candidate_obj < current_obj:
            return True, self.config.sigma_2
        
        # SA acceptance for worse solutions
        delta = candidate_obj - current_obj
        if temperature > 0:
            prob = np.exp(-delta / temperature)
            if self._rng.random() < prob:
                return True, self.config.sigma_3
        
        return False, 0.0
    
    def _update_weights(self) -> None:
        """Update operator weights based on performance."""
        rho = self.config.reaction_factor
        
        # Update destroy weights
        for i in range(len(self.destroy_operators)):
            if self._destroy_counts[i] > 0:
                score = self._destroy_scores[i] / self._destroy_counts[i]
                self._destroy_weights[i] = (
                    self._destroy_weights[i] * (1 - rho) + rho * score
                )
        
        # Normalize
        total = self._destroy_weights.sum()
        if total > 0:
            self._destroy_weights /= total
        else:
            self._destroy_weights[:] = 1.0 / len(self.destroy_operators)
        
        # Update repair weights
        for i in range(len(self.repair_operators)):
            if self._repair_counts[i] > 0:
                score = self._repair_scores[i] / self._repair_counts[i]
                self._repair_weights[i] = (
                    self._repair_weights[i] * (1 - rho) + rho * score
                )
        
        # Normalize
        total = self._repair_weights.sum()
        if total > 0:
            self._repair_weights /= total
        else:
            self._repair_weights[:] = 1.0 / len(self.repair_operators)
        
        # Reset scores and counts
        self._destroy_scores[:] = 0
        self._repair_scores[:] = 0
        self._destroy_counts[:] = 0
        self._repair_counts[:] = 0


def create_alns(
    max_iterations: int = 10000,
    max_time: Optional[float] = None,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.9999,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.2,
    hnsw_manager: Optional['HNSWManager'] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> ALNS:
    """Factory function to create ALNS solver.
    
    Args:
        max_iterations: Maximum iterations
        max_time: Maximum runtime (seconds)
        initial_temperature: SA starting temperature
        cooling_rate: SA cooling rate
        alpha: Cost weight
        beta: CV (driver fairness) weight
        gamma: Jain's (customer fairness) weight
        hnsw_manager: HNSW manager for accelerated repair
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Configured ALNS solver
    """
    config = ALNSConfig(
        max_iterations=max_iterations,
        max_time=max_time,
        initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        seed=seed,
        verbose=verbose,
    )
    
    return ALNS(config=config, hnsw_manager=hnsw_manager)

