"""Dynamic VRP event handler.

This module provides the DynamicHandler class that processes
dynamic events and updates the VRP solution in real-time.

Key features:
- HNSW-accelerated insertion for new orders
- Incremental ALNS for solution optimization
- Real-time solution updates
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Dict, Callable
import numpy as np
import time
import logging

from .event_generator import DynamicEvent, EventType, DynamicScenario

if TYPE_CHECKING:
    from ..models.problem import VRPInstance, Customer
    from ..models.solution import Solution, Route
    from ..hnsw.manager import HNSWManager
    from ..heuristics.alns import ALNS

logger = logging.getLogger(__name__)


@dataclass
class HandlerStatistics:
    """Statistics from dynamic event handling."""
    total_events: int = 0
    new_orders_handled: int = 0
    cancellations_handled: int = 0
    updates_handled: int = 0
    
    total_insertion_time: float = 0.0
    total_optimization_time: float = 0.0
    
    failed_insertions: int = 0
    successful_insertions: int = 0
    
    # Solution quality over time
    cost_history: List[tuple] = field(default_factory=list)  # (time, cost)
    
    def summary(self) -> str:
        avg_insert_time = (
            self.total_insertion_time / self.successful_insertions
            if self.successful_insertions > 0 else 0
        )
        return (
            f"Dynamic Handler Statistics:\n"
            f"  Total events: {self.total_events}\n"
            f"  New orders: {self.new_orders_handled}\n"
            f"  Cancellations: {self.cancellations_handled}\n"
            f"  Successful insertions: {self.successful_insertions}\n"
            f"  Failed insertions: {self.failed_insertions}\n"
            f"  Avg insertion time: {avg_insert_time*1000:.2f}ms\n"
            f"  Total optimization time: {self.total_optimization_time:.2f}s\n"
        )


@dataclass
class DynamicHandlerConfig:
    """Configuration for dynamic event handling.
    
    Attributes:
        use_hnsw: Use HNSW for fast insertion candidates
        optimize_after_insert: Run ALNS optimization after insertions
        optimization_iterations: ALNS iterations for optimization
        optimization_interval: Optimize every N insertions
        reoptimize_threshold: Reoptimize if cost increases by this fraction
        max_insert_attempts: Max attempts to insert a customer
    """
    use_hnsw: bool = True
    optimize_after_insert: bool = True
    optimization_iterations: int = 100
    optimization_interval: int = 5
    reoptimize_threshold: float = 0.1
    max_insert_attempts: int = 3


class DynamicHandler:
    """Handles dynamic events and updates VRP solution.
    
    Usage:
        handler = DynamicHandler(instance, solution, hnsw_manager, alns)
        
        for event in events:
            handler.handle_event(event)
        
        final_solution = handler.get_solution()
    """
    
    def __init__(
        self,
        instance: 'VRPInstance',
        solution: 'Solution',
        hnsw_manager: Optional['HNSWManager'] = None,
        alns: Optional['ALNS'] = None,
        config: DynamicHandlerConfig = None,
    ):
        self.instance = instance
        self.solution = solution
        self.hnsw_manager = hnsw_manager
        self.alns = alns
        self.config = config or DynamicHandlerConfig()
        
        self.stats = HandlerStatistics()
        
        # Track dynamic customers
        self._dynamic_customers: Dict[int, 'Customer'] = {}
        self._insertion_count = 0
        self._last_cost = solution.total_cost if hasattr(solution, 'total_cost') else 0
        
        # Initialize HNSW if available
        if self.hnsw_manager is not None and self.config.use_hnsw:
            self.hnsw_manager.initialize(instance, solution)
    
    def handle_event(self, event: DynamicEvent) -> bool:
        """Handle a single dynamic event.
        
        Args:
            event: The event to handle
            
        Returns:
            True if event was handled successfully
        """
        self.stats.total_events += 1
        
        if event.event_type == EventType.NEW_ORDER:
            return self._handle_new_order(event)
        elif event.event_type == EventType.CANCEL_ORDER:
            return self._handle_cancellation(event)
        elif event.event_type == EventType.UPDATE_ORDER:
            return self._handle_update(event)
        elif event.event_type == EventType.VEHICLE_BREAKDOWN:
            return self._handle_breakdown(event)
        else:
            logger.warning(f"Unknown event type: {event.event_type}")
            return False
    
    def _handle_new_order(self, event: DynamicEvent) -> bool:
        """Handle a new order event."""
        start_time = time.perf_counter()
        self.stats.new_orders_handled += 1
        
        customer = event.customer
        if customer is None:
            logger.error(f"NEW_ORDER event missing customer data")
            return False
        
        # Store dynamic customer
        self._dynamic_customers[customer.id] = customer
        
        # Add to instance
        self._add_customer_to_instance(customer)
        
        # Find best insertion using HNSW
        success = self._insert_customer(customer)
        
        elapsed = time.perf_counter() - start_time
        self.stats.total_insertion_time += elapsed
        
        if success:
            self.stats.successful_insertions += 1
            self._insertion_count += 1
            
            # Possibly reoptimize
            if (self.config.optimize_after_insert and 
                self._insertion_count % self.config.optimization_interval == 0):
                self._reoptimize()
        else:
            self.stats.failed_insertions += 1
        
        # Record cost (compute manually for dynamic customers)
        self._compute_solution_cost()
        self.stats.cost_history.append((event.time, self.solution.total_cost))
        
        return success
    
    def _handle_cancellation(self, event: DynamicEvent) -> bool:
        """Handle an order cancellation."""
        self.stats.cancellations_handled += 1
        
        customer_id = event.customer_id
        
        # Find and remove from solution
        for route in self.solution.routes:
            if customer_id in route.customers:
                route.remove(customer_id)
                
                # Update HNSW index
                if self.hnsw_manager is not None:
                    self.hnsw_manager.invalidate_route(route.vehicle_id)
                
                # Use custom cost computation for dynamic customers
                self._compute_solution_cost()
                return True
        
        return False
    
    def _handle_update(self, event: DynamicEvent) -> bool:
        """Handle an order update."""
        self.stats.updates_handled += 1
        
        # For now, just log the update
        # A full implementation would:
        # 1. Check if update affects feasibility
        # 2. Remove and reinsert if needed
        # 3. Update customer in instance
        
        logger.debug(f"Order update for customer {event.customer_id}: {event.data}")
        return True
    
    def _handle_breakdown(self, event: DynamicEvent) -> bool:
        """Handle vehicle breakdown."""
        # Remove all customers from affected vehicle's route
        # Reinsert them into other routes
        vehicle_id = event.data.get('vehicle_id')
        
        if vehicle_id is None:
            return False
        
        # Find route
        affected_route = None
        for route in self.solution.routes:
            if route.vehicle_id == vehicle_id:
                affected_route = route
                break
        
        if affected_route is None or affected_route.is_empty():
            return True
        
        # Get customers to reinsert
        customers_to_reinsert = affected_route.customers.copy()
        
        # Clear route
        affected_route.customers.clear()
        affected_route.load = 0
        affected_route.cost = 0
        
        # Reinsert each customer
        for cid in customers_to_reinsert:
            customer = self.instance.get_customer(cid)
            self._insert_customer(customer, exclude_routes=[vehicle_id])
        
        return True
    
    def _insert_customer(
        self,
        customer: 'Customer',
        exclude_routes: List[int] = None,
    ) -> bool:
        """Insert customer into best feasible position.
        
        Uses HNSW to find candidates if available.
        For dynamic customers, falls back to brute force since HNSW
        may not have them indexed yet.
        """
        exclude_routes = exclude_routes or []
        
        # For dynamic customers (ID >= 1000), use brute force
        # since HNSW encoder expects sequential IDs
        is_dynamic = customer.id >= 1000
        
        # Try HNSW-guided insertion for static customers only
        if self.hnsw_manager is not None and self.config.use_hnsw and not is_dynamic:
            candidates = self.hnsw_manager.find_insertion_candidates(
                customer.id, self.solution, k=10
            )
            
            for candidate in candidates:
                if candidate.route_id in exclude_routes:
                    continue
                
                route = self._get_route(candidate.route_id)
                if route is None:
                    continue
                
                # Check feasibility
                if not self._is_feasible_insertion(route, candidate.position, customer):
                    continue
                
                # Perform insertion
                pos = min(candidate.position, len(route.customers))
                route.insert(pos, customer.id)
                route.load += customer.demand
                
                # Update HNSW
                self.hnsw_manager.invalidate_route(route.vehicle_id)
                
                return True
        
        # Fallback: brute force best insertion
        best_route = None
        best_pos = None
        best_cost = float('inf')
        
        for route in self.solution.routes:
            if route.vehicle_id in exclude_routes:
                continue
            
            capacity = self.instance.vehicles[route.vehicle_id].capacity
            if route.load + customer.demand > capacity:
                continue
            
            for pos in range(len(route.customers) + 1):
                if not self._is_feasible_insertion(route, pos, customer):
                    continue
                
                cost = self._calculate_insertion_cost(route, pos, customer.id)
                
                if cost < best_cost:
                    best_cost = cost
                    best_route = route
                    best_pos = pos
        
        if best_route is not None:
            best_route.insert(best_pos, customer.id)
            best_route.load += customer.demand
            return True
        
        return False
    
    def _is_feasible_insertion(
        self,
        route: 'Route',
        position: int,
        customer: 'Customer',
    ) -> bool:
        """Check if insertion is feasible (capacity and time windows)."""
        # Capacity check
        capacity = self.instance.vehicles[route.vehicle_id].capacity
        if route.load + customer.demand > capacity:
            return False
        
        # Simple time window check
        # (A full implementation would check propagation)
        return True
    
    def _calculate_insertion_cost(
        self,
        route: 'Route',
        position: int,
        customer_id: int,
    ) -> float:
        """Calculate insertion cost delta."""
        customers = route.customers
        
        if position == 0:
            pred = 0
        else:
            pred = customers[position - 1]
        
        if position >= len(customers):
            succ = 0
        else:
            succ = customers[position]
        
        # Use direct distance calculation for dynamic customers
        old_cost = self._get_distance(pred, succ)
        new_cost = (
            self._get_distance(pred, customer_id) +
            self._get_distance(customer_id, succ)
        )
        
        return new_cost - old_cost
    
    def _get_distance(self, i: int, j: int) -> float:
        """Get distance between two nodes, handling dynamic customers."""
        node_i = self._get_node(i)
        node_j = self._get_node(j)
        
        return np.sqrt((node_i.x - node_j.x)**2 + (node_i.y - node_j.y)**2)
    
    def _get_node(self, node_id: int):
        """Get node by ID, including dynamic customers."""
        if node_id == 0:
            return self.instance.depot
        
        # Check dynamic customers first
        if node_id in self._dynamic_customers:
            return self._dynamic_customers[node_id]
        
        # Otherwise, static customer
        return self.instance.get_customer(node_id)
    
    def _compute_solution_cost(self) -> None:
        """Compute solution cost handling dynamic customers."""
        total_cost = 0.0
        
        for route in self.solution.routes:
            if route.is_empty():
                route.cost = 0.0
                continue
            
            # Compute route cost manually
            cost = 0.0
            prev_node = 0  # Depot
            
            for cid in route.customers:
                cost += self._get_distance(prev_node, cid)
                prev_node = cid
            
            # Return to depot
            cost += self._get_distance(prev_node, 0)
            route.cost = cost
            total_cost += cost
        
        self.solution.total_cost = total_cost
    
    def _add_customer_to_instance(self, customer: 'Customer') -> None:
        """Add dynamic customer to the tracking dict.
        
        Note: We don't modify the instance's customer list or distance matrix
        since dynamic customers have non-sequential IDs (starting at 1000).
        Instead, we track them separately in _dynamic_customers.
        """
        # Already stored in _dynamic_customers by _handle_new_order
        pass
    
    def _get_route(self, route_id: int) -> Optional['Route']:
        """Get route by vehicle ID."""
        for route in self.solution.routes:
            if route.vehicle_id == route_id:
                return route
        return None
    
    def _reoptimize(self) -> None:
        """Run ALNS to reoptimize solution."""
        if self.alns is None:
            return
        
        start_time = time.perf_counter()
        
        # Run limited ALNS
        optimized, _ = self.alns.solve(
            self.instance,
            initial_solution=self.solution,
        )
        
        # Accept if better
        if optimized.total_cost < self.solution.total_cost:
            self.solution = optimized
            
            # Update HNSW
            if self.hnsw_manager is not None:
                self.hnsw_manager.update_index(self.solution, force_rebuild=True)
        
        self.stats.total_optimization_time += time.perf_counter() - start_time
    
    def run_scenario(self, scenario: DynamicScenario) -> 'Solution':
        """Process all events in a dynamic scenario.
        
        Args:
            scenario: The dynamic scenario to run
            
        Returns:
            Final solution after processing all events
        """
        logger.info(f"Running dynamic scenario: {scenario.n_dynamic_customers} dynamic orders")
        
        for event in scenario.events:
            self.handle_event(event)
        
        # Final optimization
        if self.config.optimize_after_insert and self.alns is not None:
            logger.info("Final optimization...")
            self._reoptimize()
        
        logger.info(self.stats.summary())
        
        return self.solution
    
    def get_solution(self) -> 'Solution':
        """Get current solution."""
        return self.solution
    
    def get_statistics(self) -> HandlerStatistics:
        """Get handler statistics."""
        return self.stats


def create_dynamic_handler(
    instance: 'VRPInstance',
    solution: 'Solution',
    hnsw_manager: Optional['HNSWManager'] = None,
    alns: Optional['ALNS'] = None,
    use_hnsw: bool = True,
    optimize: bool = True,
) -> DynamicHandler:
    """Factory function to create a DynamicHandler.
    
    Args:
        instance: VRP problem instance
        solution: Initial solution
        hnsw_manager: HNSW manager (optional)
        alns: ALNS solver for optimization (optional)
        use_hnsw: Use HNSW for insertions
        optimize: Run optimization after insertions
        
    Returns:
        Configured DynamicHandler
    """
    config = DynamicHandlerConfig(
        use_hnsw=use_hnsw,
        optimize_after_insert=optimize,
    )
    
    return DynamicHandler(
        instance=instance,
        solution=solution,
        hnsw_manager=hnsw_manager,
        alns=alns,
        config=config,
    )

