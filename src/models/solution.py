"""Solution representation for VRP.

This module defines the Route and Solution classes for representing
VRP solutions with support for cost tracking and schedule computation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator
from copy import deepcopy
import numpy as np

if TYPE_CHECKING:
    from .problem import VRPInstance, Customer


@dataclass
class Route:
    """Single vehicle route.
    
    A route is a sequence of customer IDs visited by a single vehicle,
    starting and ending at the depot (implicitly).
    
    Attributes:
        vehicle_id: ID of the vehicle assigned to this route
        customers: Ordered list of customer IDs to visit
        arrival_times: Arrival time at each customer (computed)
        departure_times: Departure time from each customer (computed)
        load: Current cumulative load along the route
        cost: Total route cost (distance or time)
    """
    vehicle_id: int
    customers: list[int] = field(default_factory=list)
    arrival_times: list[float] = field(default_factory=list)
    departure_times: list[float] = field(default_factory=list)
    load: int = 0
    cost: float = 0.0
    
    def __len__(self) -> int:
        """Number of customers in route."""
        return len(self.customers)
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over customer IDs."""
        return iter(self.customers)
    
    def __contains__(self, customer_id: int) -> bool:
        """Check if customer is in route."""
        return customer_id in self.customers
    
    def __getitem__(self, index: int) -> int:
        """Get customer ID at index."""
        return self.customers[index]
    
    def is_empty(self) -> bool:
        """Check if route has no customers."""
        return len(self.customers) == 0
    
    def insert(self, position: int, customer_id: int) -> None:
        """Insert customer at position.
        
        Args:
            position: Index where to insert (0 = after depot)
            customer_id: Customer ID to insert
        """
        self.customers.insert(position, customer_id)
    
    def remove(self, customer_id: int) -> int:
        """Remove customer from route.
        
        Args:
            customer_id: Customer ID to remove
            
        Returns:
            Position where customer was removed from
            
        Raises:
            ValueError: If customer not in route
        """
        pos = self.customers.index(customer_id)
        self.customers.pop(pos)
        return pos
    
    def remove_at(self, position: int) -> int:
        """Remove customer at position.
        
        Args:
            position: Index of customer to remove
            
        Returns:
            Customer ID that was removed
        """
        return self.customers.pop(position)
    
    def index_of(self, customer_id: int) -> int:
        """Get position of customer in route.
        
        Args:
            customer_id: Customer ID to find
            
        Returns:
            Position in route (0-indexed)
            
        Raises:
            ValueError: If customer not in route
        """
        return self.customers.index(customer_id)
    
    def predecessor(self, position: int) -> int:
        """Get predecessor node ID (0 = depot if position is 0)."""
        if position <= 0:
            return 0
        return self.customers[position - 1]
    
    def successor(self, position: int) -> int:
        """Get successor node ID (0 = depot if position is last)."""
        if position >= len(self.customers) - 1:
            return 0
        return self.customers[position + 1]
    
    def copy(self) -> Route:
        """Create a deep copy of this route."""
        return Route(
            vehicle_id=self.vehicle_id,
            customers=list(self.customers),
            arrival_times=list(self.arrival_times),
            departure_times=list(self.departure_times),
            load=self.load,
            cost=self.cost,
        )
    
    def clear(self) -> None:
        """Remove all customers from route."""
        self.customers.clear()
        self.arrival_times.clear()
        self.departure_times.clear()
        self.load = 0
        self.cost = 0.0
    
    def __repr__(self) -> str:
        if self.is_empty():
            return f"Route(v{self.vehicle_id}: empty)"
        customers_str = " -> ".join(map(str, self.customers[:5]))
        if len(self.customers) > 5:
            customers_str += f" ... ({len(self.customers)} total)"
        return f"Route(v{self.vehicle_id}: 0 -> {customers_str} -> 0, cost={self.cost:.1f})"


@dataclass
class Solution:
    """Complete VRP solution.
    
    A solution consists of multiple routes, one per vehicle (some may be empty).
    
    Attributes:
        routes: List of routes, one per vehicle
        instance: Reference to the problem instance
        total_cost: Sum of all route costs
        is_feasible: Whether solution satisfies all constraints
    """
    routes: list[Route]
    instance: VRPInstance
    total_cost: float = 0.0
    is_feasible: bool = True
    
    def __post_init__(self):
        """Initialize routes if needed."""
        # Ensure we have one route per vehicle
        while len(self.routes) < self.instance.n_vehicles:
            self.routes.append(Route(vehicle_id=len(self.routes)))
    
    def __len__(self) -> int:
        """Number of routes."""
        return len(self.routes)
    
    def __iter__(self) -> Iterator[Route]:
        """Iterate over routes."""
        return iter(self.routes)
    
    def __getitem__(self, index: int) -> Route:
        """Get route by index."""
        return self.routes[index]
    
    @classmethod
    def empty(cls, instance: VRPInstance) -> Solution:
        """Create an empty solution with no assigned customers."""
        routes = [Route(vehicle_id=k) for k in range(instance.n_vehicles)]
        return cls(routes=routes, instance=instance)
    
    def copy(self) -> Solution:
        """Create a deep copy of this solution."""
        return Solution(
            routes=[r.copy() for r in self.routes],
            instance=self.instance,
            total_cost=self.total_cost,
            is_feasible=self.is_feasible,
        )
    
    def get_route_of(self, customer_id: int) -> tuple[int, int] | None:
        """Find which route contains a customer.
        
        Args:
            customer_id: Customer ID to find
            
        Returns:
            Tuple of (route_index, position_in_route) or None if not found
        """
        for route_idx, route in enumerate(self.routes):
            if customer_id in route:
                return (route_idx, route.index_of(customer_id))
        return None
    
    def remove_customer(self, customer_id: int) -> tuple[int, int] | None:
        """Remove customer from solution.
        
        Args:
            customer_id: Customer ID to remove
            
        Returns:
            Tuple of (route_index, position) where customer was, or None
        """
        for route_idx, route in enumerate(self.routes):
            if customer_id in route:
                pos = route.remove(customer_id)
                return (route_idx, pos)
        return None
    
    def get_route_costs(self) -> np.ndarray:
        """Get array of route costs."""
        return np.array([r.cost for r in self.routes], dtype=np.float64)
    
    def get_route_loads(self) -> np.ndarray:
        """Get array of route loads."""
        return np.array([r.load for r in self.routes], dtype=np.int32)
    
    def get_route_lengths(self) -> np.ndarray:
        """Get array of route lengths (number of customers)."""
        return np.array([len(r) for r in self.routes], dtype=np.int32)
    
    def get_non_empty_routes(self) -> list[Route]:
        """Get list of routes with at least one customer."""
        return [r for r in self.routes if not r.is_empty()]
    
    def n_routes_used(self) -> int:
        """Count number of non-empty routes."""
        return sum(1 for r in self.routes if not r.is_empty())
    
    def n_customers_served(self) -> int:
        """Count total customers served across all routes."""
        return sum(len(r) for r in self.routes)
    
    def all_customers_served(self) -> bool:
        """Check if all customers are served exactly once."""
        served = set()
        for route in self.routes:
            for cust_id in route:
                if cust_id in served:
                    return False  # Duplicate
                served.add(cust_id)
        expected = set(range(1, self.instance.n_customers + 1))
        return served == expected
    
    def get_unserved_customers(self) -> set[int]:
        """Get set of customer IDs not in any route."""
        served = set()
        for route in self.routes:
            served.update(route.customers)
        all_customers = set(range(1, self.instance.n_customers + 1))
        return all_customers - served
    
    def get_customer_waiting_times(self) -> np.ndarray:
        """Get waiting times for all served customers.
        
        Waiting time = max(0, arrival_time - time_window_start)
        
        Returns:
            Array of waiting times (one per served customer)
        """
        waiting = []
        for route in self.routes:
            for i, cust_id in enumerate(route.customers):
                customer = self.instance.get_customer(cust_id)
                if route.arrival_times and i < len(route.arrival_times):
                    arrival = route.arrival_times[i]
                else:
                    arrival = 0.0
                wait = max(0.0, arrival - customer.time_window_start)
                waiting.append(wait)
        return np.array(waiting, dtype=np.float64)
    
    def compute_schedule(self) -> None:
        """Compute arrival and departure times for all routes."""
        for route in self.routes:
            self._compute_route_schedule(route)
    
    def _compute_route_schedule(self, route: Route) -> None:
        """Compute schedule for a single route."""
        if route.is_empty():
            route.arrival_times = []
            route.departure_times = []
            route.load = 0
            route.cost = 0.0
            return
        
        arrival_times = []
        departure_times = []
        
        current_time = 0.0
        current_load = 0
        prev_node = 0  # Depot
        
        for cust_id in route.customers:
            customer = self.instance.get_customer(cust_id)
            
            # Travel to customer
            travel_time = self.instance.get_travel_time(prev_node, cust_id)
            arrival = current_time + travel_time
            
            # Wait if arriving early
            if arrival < customer.time_window_start:
                arrival = customer.time_window_start
            
            arrival_times.append(arrival)
            
            # Service
            departure = arrival + customer.service_time
            departure_times.append(departure)
            
            # Update state
            current_time = departure
            current_load += customer.demand
            prev_node = cust_id
        
        route.arrival_times = arrival_times
        route.departure_times = departure_times
        route.load = current_load
    
    def compute_cost(self) -> float:
        """Compute and update total cost of solution."""
        total = 0.0
        for route in self.routes:
            route.cost = self._compute_route_cost(route)
            total += route.cost
        self.total_cost = total
        return total
    
    def _compute_route_cost(self, route: Route) -> float:
        """Compute cost (total distance) of a single route."""
        if route.is_empty():
            return 0.0
        
        cost = 0.0
        prev_node = 0  # Depot
        
        for cust_id in route.customers:
            cost += self.instance.get_distance(prev_node, cust_id)
            prev_node = cust_id
        
        # Return to depot
        cost += self.instance.get_distance(prev_node, 0)
        
        return cost
    
    def validate(self) -> tuple[bool, list[str]]:
        """Check if solution is feasible.
        
        Returns:
            Tuple of (is_feasible, list of violation messages)
        """
        violations = []
        
        # Check all customers served exactly once
        served_count = {}
        for route in self.routes:
            for cust_id in route.customers:
                served_count[cust_id] = served_count.get(cust_id, 0) + 1
        
        for cust_id in range(1, self.instance.n_customers + 1):
            count = served_count.get(cust_id, 0)
            if count == 0:
                violations.append(f"Customer {cust_id} not served")
            elif count > 1:
                violations.append(f"Customer {cust_id} served {count} times")
        
        # Check capacity constraints
        for route in self.routes:
            if route.is_empty():
                continue
            
            load = sum(
                self.instance.get_customer(cid).demand 
                for cid in route.customers
            )
            vehicle = self.instance.vehicles[route.vehicle_id]
            
            if load > vehicle.capacity:
                violations.append(
                    f"Route {route.vehicle_id} exceeds capacity: {load} > {vehicle.capacity}"
                )
        
        # Check time windows (if schedule computed)
        for route in self.routes:
            if not route.arrival_times:
                continue
            
            for i, cust_id in enumerate(route.customers):
                customer = self.instance.get_customer(cust_id)
                arrival = route.arrival_times[i]
                
                if arrival > customer.time_window_end:
                    violations.append(
                        f"Customer {cust_id} time window violated: "
                        f"arrival {arrival:.1f} > end {customer.time_window_end:.1f}"
                    )
        
        is_feasible = len(violations) == 0
        self.is_feasible = is_feasible
        return is_feasible, violations
    
    def summary(self) -> str:
        """Return summary string of solution."""
        return (
            f"Solution: {self.n_routes_used()} routes, "
            f"{self.n_customers_served()}/{self.instance.n_customers} customers, "
            f"cost={self.total_cost:.2f}, feasible={self.is_feasible}"
        )
    
    def to_dict(self) -> dict:
        """Convert solution to dictionary for serialization."""
        return {
            "instance_name": self.instance.name,
            "total_cost": self.total_cost,
            "is_feasible": self.is_feasible,
            "n_routes": self.n_routes_used(),
            "routes": [
                {
                    "vehicle_id": r.vehicle_id,
                    "customers": r.customers,
                    "cost": r.cost,
                    "load": r.load,
                }
                for r in self.routes
                if not r.is_empty()
            ],
        }
    
    def __repr__(self) -> str:
        return self.summary()

