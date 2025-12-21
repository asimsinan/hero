"""Destroy operators for ALNS.

This module implements various destroy (removal) operators used in
Adaptive Large Neighborhood Search. Each operator removes a subset
of customers from the current solution.

Operators:
- RandomRemoval: Remove random customers
- WorstRemoval: Remove customers with highest removal cost savings
- RelatedRemoval (Shaw): Remove customers similar to each other
- RouteRemoval: Remove entire routes
- FairnessRemoval: Remove customers that hurt fairness metrics
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple, Optional, Set
import numpy as np
import random
import logging

if TYPE_CHECKING:
    from ..models.problem import VRPInstance
    from ..models.solution import Solution, Route

logger = logging.getLogger(__name__)


@dataclass
class DestroyResult:
    """Result of a destroy operation.
    
    Attributes:
        removed_customers: List of removed customer IDs
        modified_routes: Set of route IDs that were modified
        removal_costs: Dict mapping customer_id to cost savings from removal
    """
    removed_customers: List[int]
    modified_routes: Set[int] = field(default_factory=set)
    removal_costs: dict = field(default_factory=dict)
    
    @property
    def n_removed(self) -> int:
        return len(self.removed_customers)


class DestroyOperator(ABC):
    """Abstract base class for destroy operators."""
    
    name: str = "BaseDestroy"
    
    @abstractmethod
    def destroy(
        self,
        solution: 'Solution',
        instance: 'VRPInstance',
        n_remove: int,
    ) -> DestroyResult:
        """Remove customers from solution.
        
        Args:
            solution: Current solution (will be modified in-place)
            instance: VRP problem instance
            n_remove: Number of customers to remove
            
        Returns:
            DestroyResult containing removed customers and metadata
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomRemoval(DestroyOperator):
    """Remove random customers from the solution.
    
    Simple baseline operator. Fast but not informed.
    Time complexity: O(n_remove)
    """
    
    name = "RandomRemoval"
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def destroy(
        self,
        solution: 'Solution',
        instance: 'VRPInstance',
        n_remove: int,
    ) -> DestroyResult:
        # Collect all served customers
        served = []
        customer_to_route = {}
        
        for route in solution.routes:
            for cid in route.customers:
                served.append(cid)
                customer_to_route[cid] = route.vehicle_id
        
        if not served:
            return DestroyResult(removed_customers=[])
        
        # Select random customers to remove
        n_remove = min(n_remove, len(served))
        to_remove = self.rng.sample(served, n_remove)
        
        # Remove from routes
        modified_routes = set()
        for cid in to_remove:
            route_id = customer_to_route[cid]
            for route in solution.routes:
                if route.vehicle_id == route_id and cid in route.customers:
                    route.remove(cid)
                    modified_routes.add(route_id)
                    break
        
        return DestroyResult(
            removed_customers=to_remove,
            modified_routes=modified_routes,
        )


class WorstRemoval(DestroyOperator):
    """Remove customers with highest removal cost savings.
    
    Removes customers whose removal saves the most distance/cost.
    Uses randomized selection with determinism parameter.
    
    Time complexity: O(n log n) for sorting
    """
    
    name = "WorstRemoval"
    
    def __init__(
        self,
        determinism: float = 3.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            determinism: Higher = more greedy, lower = more random.
                        Customer i selected with prob ~ (1/rank_i)^determinism
        """
        self.determinism = determinism
        self.rng = random.Random(seed)
    
    def destroy(
        self,
        solution: 'Solution',
        instance: 'VRPInstance',
        n_remove: int,
    ) -> DestroyResult:
        # Calculate removal cost for each customer
        removal_costs = {}
        customer_to_route = {}
        
        for route in solution.routes:
            for i, cid in enumerate(route.customers):
                cost = self._calculate_removal_cost(route, i, instance)
                removal_costs[cid] = cost
                customer_to_route[cid] = route.vehicle_id
        
        if not removal_costs:
            return DestroyResult(removed_customers=[])
        
        # Sort by removal cost (highest savings first)
        sorted_customers = sorted(
            removal_costs.keys(),
            key=lambda c: removal_costs[c],
            reverse=True,
        )
        
        # Select using randomized worst removal
        removed = []
        modified_routes = set()
        remaining = sorted_customers.copy()
        
        while len(removed) < n_remove and remaining:
            # Compute selection probabilities
            n = len(remaining)
            probs = np.array([(1.0 / (i + 1)) ** self.determinism for i in range(n)])
            probs /= probs.sum()
            
            # Select customer
            idx = self.rng.choices(range(n), weights=probs)[0]
            cid = remaining[idx]
            
            # Remove from solution
            route_id = customer_to_route[cid]
            for route in solution.routes:
                if route.vehicle_id == route_id and cid in route.customers:
                    route.remove(cid)
                    modified_routes.add(route_id)
                    break
            
            removed.append(cid)
            remaining.remove(cid)
        
        return DestroyResult(
            removed_customers=removed,
            modified_routes=modified_routes,
            removal_costs=removal_costs,
        )
    
    def _calculate_removal_cost(
        self,
        route: 'Route',
        position: int,
        instance: 'VRPInstance',
    ) -> float:
        """Calculate cost saved by removing customer at position."""
        customers = route.customers
        cid = customers[position]
        
        # Get predecessor and successor
        if position == 0:
            pred = 0  # Depot
        else:
            pred = customers[position - 1]
        
        if position == len(customers) - 1:
            succ = 0  # Depot
        else:
            succ = customers[position + 1]
        
        # Current cost (pred -> cid -> succ)
        current_cost = (
            instance.get_distance(pred, cid) +
            instance.get_distance(cid, succ)
        )
        
        # New cost (pred -> succ)
        new_cost = instance.get_distance(pred, succ)
        
        # Savings
        return current_cost - new_cost


class RelatedRemoval(DestroyOperator):
    """Remove related (similar) customers using Shaw removal.
    
    Removes customers that are similar to each other based on:
    - Distance
    - Time windows
    - Demand
    
    Time complexity: O(n_remove * n)
    """
    
    name = "RelatedRemoval"
    
    def __init__(
        self,
        distance_weight: float = 9.0,
        time_weight: float = 3.0,
        demand_weight: float = 2.0,
        determinism: float = 6.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            distance_weight: Weight for distance similarity
            time_weight: Weight for time window similarity
            demand_weight: Weight for demand similarity
            determinism: Selection randomness parameter
        """
        self.distance_weight = distance_weight
        self.time_weight = time_weight
        self.demand_weight = demand_weight
        self.determinism = determinism
        self.rng = random.Random(seed)
        
        # Normalization factors (will be computed)
        self._max_distance = 1.0
        self._max_time_diff = 1.0
        self._max_demand_diff = 1.0
    
    def destroy(
        self,
        solution: 'Solution',
        instance: 'VRPInstance',
        n_remove: int,
    ) -> DestroyResult:
        # Collect served customers and their routes
        served = []
        customer_to_route = {}
        
        for route in solution.routes:
            for cid in route.customers:
                served.append(cid)
                customer_to_route[cid] = route.vehicle_id
        
        if not served:
            return DestroyResult(removed_customers=[])
        
        # Update normalization factors
        self._update_normalization(instance)
        
        # Start with a random customer
        seed_customer = self.rng.choice(served)
        removed = [seed_customer]
        remaining = [c for c in served if c != seed_customer]
        
        # Remove seed customer
        route_id = customer_to_route[seed_customer]
        modified_routes = {route_id}
        for route in solution.routes:
            if route.vehicle_id == route_id and seed_customer in route.customers:
                route.remove(seed_customer)
                break
        
        # Iteratively add related customers
        while len(removed) < n_remove and remaining:
            # Calculate relatedness to already removed customers
            relatedness = {}
            for cid in remaining:
                # Average relatedness to removed customers
                total_rel = 0.0
                for removed_cid in removed:
                    total_rel += self._calculate_relatedness(
                        cid, removed_cid, instance
                    )
                relatedness[cid] = total_rel / len(removed)
            
            # Sort by relatedness (most related first = lowest value)
            sorted_remaining = sorted(remaining, key=lambda c: relatedness[c])
            
            # Select using randomized selection
            n = len(sorted_remaining)
            probs = np.array([(1.0 / (i + 1)) ** self.determinism for i in range(n)])
            probs /= probs.sum()
            
            idx = self.rng.choices(range(n), weights=probs)[0]
            selected = sorted_remaining[idx]
            
            # Remove selected customer
            route_id = customer_to_route[selected]
            for route in solution.routes:
                if route.vehicle_id == route_id and selected in route.customers:
                    route.remove(selected)
                    modified_routes.add(route_id)
                    break
            
            removed.append(selected)
            remaining.remove(selected)
        
        return DestroyResult(
            removed_customers=removed,
            modified_routes=modified_routes,
        )
    
    def _calculate_relatedness(
        self,
        cid1: int,
        cid2: int,
        instance: 'VRPInstance',
    ) -> float:
        """Calculate relatedness score (lower = more related)."""
        c1 = instance.get_customer(cid1)
        c2 = instance.get_customer(cid2)
        
        # Distance component
        distance = instance.get_distance(cid1, cid2)
        dist_score = (distance / self._max_distance) * self.distance_weight
        
        # Time window component
        time_diff = abs(c1.time_window_start - c2.time_window_start)
        time_score = (time_diff / self._max_time_diff) * self.time_weight
        
        # Demand component
        demand_diff = abs(c1.demand - c2.demand)
        demand_score = (demand_diff / self._max_demand_diff) * self.demand_weight
        
        return dist_score + time_score + demand_score
    
    def _update_normalization(self, instance: 'VRPInstance') -> None:
        """Update normalization factors from instance."""
        customers = instance.customers
        
        if len(customers) < 2:
            return
        
        # Max distance
        max_dist = 0.0
        for i, c1 in enumerate(customers):
            for c2 in customers[i+1:]:
                d = instance.get_distance(c1.id, c2.id)
                max_dist = max(max_dist, d)
        self._max_distance = max(max_dist, 1.0)
        
        # Max time difference
        tw_starts = [c.time_window_start for c in customers]
        self._max_time_diff = max(max(tw_starts) - min(tw_starts), 1.0)
        
        # Max demand difference
        demands = [c.demand for c in customers]
        self._max_demand_diff = max(max(demands) - min(demands), 1.0)


class RouteRemoval(DestroyOperator):
    """Remove all customers from entire routes.
    
    Selects routes based on cost efficiency (cost per customer).
    Good for restructuring solutions with poor routes.
    
    Time complexity: O(m) where m = number of routes
    """
    
    name = "RouteRemoval"
    
    def __init__(
        self,
        max_routes: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Args:
            max_routes: Maximum number of routes to remove
        """
        self.max_routes = max_routes
        self.rng = random.Random(seed)
    
    def destroy(
        self,
        solution: 'Solution',
        instance: 'VRPInstance',
        n_remove: int,
    ) -> DestroyResult:
        # Find non-empty routes
        non_empty_routes = [r for r in solution.routes if not r.is_empty()]
        
        if not non_empty_routes:
            return DestroyResult(removed_customers=[])
        
        # Calculate cost efficiency for each route
        route_scores = []
        for route in non_empty_routes:
            n_customers = len(route.customers)
            if n_customers > 0:
                efficiency = route.cost / n_customers  # Cost per customer
                route_scores.append((route, efficiency))
        
        # Sort by efficiency (worst first = highest cost per customer)
        route_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select routes to remove
        removed = []
        modified_routes = set()
        n_routes_to_remove = min(
            self.max_routes,
            len(route_scores),
            max(1, n_remove // 5),  # Roughly estimate routes needed
        )
        
        for route, _ in route_scores[:n_routes_to_remove]:
            # Check if we've removed enough
            if len(removed) >= n_remove:
                break
            
            # Remove all customers from this route
            customers_to_remove = route.customers.copy()
            for cid in customers_to_remove:
                route.remove(cid)
                removed.append(cid)
            
            modified_routes.add(route.vehicle_id)
        
        return DestroyResult(
            removed_customers=removed,
            modified_routes=modified_routes,
        )


class FairnessRemoval(DestroyOperator):
    """Remove customers that negatively impact fairness.
    
    Removes customers from routes that have:
    - Highest workload (to reduce CV)
    - Highest customer waiting times (to improve Jain's index)
    
    Novel operator for fairness-aware ALNS.
    """
    
    name = "FairnessRemoval"
    
    def __init__(
        self,
        workload_focus: float = 0.5,  # Balance between workload and waiting
        determinism: float = 4.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            workload_focus: Weight for workload vs waiting time (0-1)
            determinism: Selection randomness parameter
        """
        self.workload_focus = workload_focus
        self.determinism = determinism
        self.rng = random.Random(seed)
    
    def destroy(
        self,
        solution: 'Solution',
        instance: 'VRPInstance',
        n_remove: int,
    ) -> DestroyResult:
        # Calculate fairness scores for each customer
        fairness_scores = {}
        customer_to_route = {}
        
        # Get route statistics
        route_costs = {r.vehicle_id: r.cost for r in solution.routes}
        mean_cost = np.mean(list(route_costs.values())) if route_costs else 1.0
        
        for route in solution.routes:
            # Route workload deviation from mean
            workload_deviation = (route.cost - mean_cost) / (mean_cost + 1e-8)
            
            for i, cid in enumerate(route.customers):
                customer = instance.get_customer(cid)
                
                # Waiting time estimate
                waiting_score = self._estimate_waiting(route, i, instance)
                
                # Combined fairness impact score
                # Higher score = worse for fairness
                score = (
                    self.workload_focus * max(0, workload_deviation) +
                    (1 - self.workload_focus) * waiting_score
                )
                
                fairness_scores[cid] = score
                customer_to_route[cid] = route.vehicle_id
        
        if not fairness_scores:
            return DestroyResult(removed_customers=[])
        
        # Sort by fairness impact (worst first)
        sorted_customers = sorted(
            fairness_scores.keys(),
            key=lambda c: fairness_scores[c],
            reverse=True,
        )
        
        # Select using randomized selection
        removed = []
        modified_routes = set()
        remaining = sorted_customers.copy()
        
        while len(removed) < n_remove and remaining:
            n = len(remaining)
            probs = np.array([(1.0 / (i + 1)) ** self.determinism for i in range(n)])
            probs /= probs.sum()
            
            idx = self.rng.choices(range(n), weights=probs)[0]
            cid = remaining[idx]
            
            # Remove from solution
            route_id = customer_to_route[cid]
            for route in solution.routes:
                if route.vehicle_id == route_id and cid in route.customers:
                    route.remove(cid)
                    modified_routes.add(route_id)
                    break
            
            removed.append(cid)
            remaining.remove(cid)
        
        return DestroyResult(
            removed_customers=removed,
            modified_routes=modified_routes,
        )
    
    def _estimate_waiting(
        self,
        route: 'Route',
        position: int,
        instance: 'VRPInstance',
    ) -> float:
        """Estimate waiting time for customer at position."""
        cid = route.customers[position]
        customer = instance.get_customer(cid)
        
        # Simple estimate: compare time window to position in route
        # Early position + late time window = likely waiting
        route_length = len(route.customers)
        position_ratio = position / max(route_length, 1)
        
        # Normalize time window start
        tw_ratio = customer.time_window_start / 1000.0  # Assume max 1000
        
        # Mismatch between position and time window suggests waiting
        mismatch = abs(position_ratio - tw_ratio)
        
        return mismatch


def get_destroy_operators(seed: Optional[int] = None) -> List[DestroyOperator]:
    """Get all destroy operators with default parameters.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        List of destroy operators
    """
    return [
        RandomRemoval(seed=seed),
        WorstRemoval(determinism=3.0, seed=seed),
        RelatedRemoval(determinism=6.0, seed=seed),
        RouteRemoval(max_routes=2, seed=seed),
        FairnessRemoval(workload_focus=0.5, seed=seed),
    ]

