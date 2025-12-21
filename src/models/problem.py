"""Core data models for VRP problem representation.

This module defines the fundamental data structures for representing
Vehicle Routing Problems with Time Windows (VRPTW).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np


@dataclass(frozen=True, slots=True)
class Customer:
    """Immutable customer representation.
    
    Attributes:
        id: Unique customer identifier (1-indexed, 0 is depot)
        x: X-coordinate of customer location
        y: Y-coordinate of customer location
        demand: Quantity of goods to deliver
        service_time: Time required to service customer
        time_window_start: Earliest allowed arrival time
        time_window_end: Latest allowed arrival time
        release_time: Time when customer becomes available (for dynamic VRP)
    """
    id: int
    x: float
    y: float
    demand: int
    service_time: float = 0.0
    time_window_start: float = 0.0
    time_window_end: float = float('inf')
    release_time: float = 0.0
    
    def distance_to(self, other: Customer | Depot) -> float:
        """Calculate Euclidean distance to another node."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def is_time_window_feasible(self, arrival_time: float) -> bool:
        """Check if arrival time respects time window."""
        return self.time_window_start <= arrival_time <= self.time_window_end
    
    def waiting_time(self, arrival_time: float) -> float:
        """Calculate waiting time if arriving before time window opens."""
        return max(0.0, self.time_window_start - arrival_time)
    
    def __repr__(self) -> str:
        return (
            f"Customer(id={self.id}, pos=({self.x:.1f}, {self.y:.1f}), "
            f"demand={self.demand}, tw=[{self.time_window_start:.0f}, {self.time_window_end:.0f}])"
        )


@dataclass(frozen=True, slots=True)
class Depot:
    """Depot location where all routes start and end.
    
    Attributes:
        id: Always 0 for the depot
        x: X-coordinate of depot location
        y: Y-coordinate of depot location
        time_window_start: Depot opening time
        time_window_end: Depot closing time (vehicles must return by this time)
    """
    id: int = 0
    x: float = 0.0
    y: float = 0.0
    time_window_start: float = 0.0
    time_window_end: float = float('inf')
    
    def distance_to(self, other: Customer | Depot) -> float:
        """Calculate Euclidean distance to another node."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def __repr__(self) -> str:
        return f"Depot(pos=({self.x:.1f}, {self.y:.1f}))"


@dataclass(slots=True)
class Vehicle:
    """Vehicle with capacity and current state.
    
    Attributes:
        id: Unique vehicle identifier
        capacity: Maximum load capacity
        current_load: Current load (for tracking during simulation)
        current_time: Current time (for tracking during simulation)
        speed: Travel speed (default 1.0 = distance equals time)
    """
    id: int
    capacity: int
    current_load: int = 0
    current_time: float = 0.0
    speed: float = 1.0
    
    def can_serve(self, customer: Customer) -> bool:
        """Check if vehicle can serve customer without exceeding capacity."""
        return self.current_load + customer.demand <= self.capacity
    
    def remaining_capacity(self) -> int:
        """Get remaining capacity."""
        return self.capacity - self.current_load
    
    def reset(self) -> None:
        """Reset vehicle state for new route."""
        self.current_load = 0
        self.current_time = 0.0
    
    def copy(self) -> Vehicle:
        """Create a copy of this vehicle."""
        return Vehicle(
            id=self.id,
            capacity=self.capacity,
            current_load=self.current_load,
            current_time=self.current_time,
            speed=self.speed,
        )
    
    def __repr__(self) -> str:
        return f"Vehicle(id={self.id}, cap={self.capacity}, load={self.current_load})"


@dataclass
class VRPInstance:
    """Complete VRP problem instance.
    
    Attributes:
        name: Instance name/identifier
        depot: Depot location
        customers: List of customers to serve
        vehicles: List of available vehicles
        distance_matrix: Precomputed distance matrix (optional, computed if None)
        time_matrix: Travel time matrix (optional, defaults to distance_matrix)
    """
    name: str
    depot: Depot
    customers: list[Customer]
    vehicles: list[Vehicle]
    distance_matrix: np.ndarray = field(repr=False, default=None)
    time_matrix: Optional[np.ndarray] = field(repr=False, default=None)
    
    def __post_init__(self):
        """Compute distance matrix if not provided."""
        if self.distance_matrix is None:
            self.distance_matrix = self._compute_distance_matrix()
        if self.time_matrix is None:
            # Default: travel time equals distance (unit speed)
            self.time_matrix = self.distance_matrix.copy()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute Euclidean distance matrix between all nodes."""
        nodes = self.all_nodes
        n = len(nodes)
        distances = np.zeros((n, n), dtype=np.float64)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(
                    (nodes[i].x - nodes[j].x) ** 2 + 
                    (nodes[i].y - nodes[j].y) ** 2
                )
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    @property
    def n_customers(self) -> int:
        """Number of customers."""
        return len(self.customers)
    
    @property
    def n_vehicles(self) -> int:
        """Number of vehicles."""
        return len(self.vehicles)
    
    @property
    def n_nodes(self) -> int:
        """Total number of nodes (depot + customers)."""
        return 1 + len(self.customers)
    
    @property
    def all_nodes(self) -> list[Depot | Customer]:
        """List of all nodes: depot at index 0, then customers."""
        return [self.depot] + self.customers
    
    @property
    def total_demand(self) -> int:
        """Total demand across all customers."""
        return sum(c.demand for c in self.customers)
    
    @property
    def total_capacity(self) -> int:
        """Total capacity across all vehicles."""
        return sum(v.capacity for v in self.vehicles)
    
    def get_node(self, node_id: int) -> Depot | Customer:
        """Get node by ID (0 = depot, 1+ = customers)."""
        if node_id == 0:
            return self.depot
        return self.customers[node_id - 1]
    
    def get_customer(self, customer_id: int) -> Customer:
        """Get customer by ID (1-indexed)."""
        return self.customers[customer_id - 1]
    
    def get_distance(self, i: int, j: int) -> float:
        """Get distance between nodes i and j."""
        return float(self.distance_matrix[i, j])
    
    def get_travel_time(self, i: int, j: int) -> float:
        """Get travel time between nodes i and j."""
        return float(self.time_matrix[i, j])
    
    def customer_ids(self) -> list[int]:
        """Get list of all customer IDs."""
        return [c.id for c in self.customers]
    
    def validate(self) -> list[str]:
        """Validate instance consistency, return list of issues."""
        issues = []
        
        # Check depot
        if self.depot.id != 0:
            issues.append(f"Depot ID should be 0, got {self.depot.id}")
        
        # Check customer IDs are 1-indexed and consecutive
        expected_ids = set(range(1, self.n_customers + 1))
        actual_ids = set(c.id for c in self.customers)
        if expected_ids != actual_ids:
            issues.append(f"Customer IDs should be 1..{self.n_customers}, got {sorted(actual_ids)}")
        
        # Check demands are non-negative
        for c in self.customers:
            if c.demand < 0:
                issues.append(f"Customer {c.id} has negative demand: {c.demand}")
        
        # Check time windows
        for c in self.customers:
            if c.time_window_start > c.time_window_end:
                issues.append(f"Customer {c.id} has invalid time window: [{c.time_window_start}, {c.time_window_end}]")
        
        # Check capacity vs demand
        if self.total_demand > self.total_capacity:
            issues.append(f"Total demand ({self.total_demand}) exceeds total capacity ({self.total_capacity})")
        
        # Check distance matrix shape
        expected_shape = (self.n_nodes, self.n_nodes)
        if self.distance_matrix.shape != expected_shape:
            issues.append(f"Distance matrix shape {self.distance_matrix.shape} != expected {expected_shape}")
        
        return issues
    
    def summary(self) -> str:
        """Return a summary string of the instance."""
        return (
            f"VRPInstance '{self.name}': "
            f"{self.n_customers} customers, {self.n_vehicles} vehicles, "
            f"total demand={self.total_demand}, total capacity={self.total_capacity}"
        )
    
    def copy(self) -> VRPInstance:
        """Create a deep copy of this instance."""
        return VRPInstance(
            name=self.name,
            depot=self.depot,
            customers=list(self.customers),
            vehicles=[v.copy() for v in self.vehicles],
            distance_matrix=self.distance_matrix.copy(),
            time_matrix=self.time_matrix.copy() if self.time_matrix is not None else None,
        )
    
    @classmethod
    def create_random(
        cls,
        n_customers: int,
        n_vehicles: int,
        capacity: int = 100,
        area_size: float = 100.0,
        demand_range: tuple[int, int] = (1, 20),
        seed: int | None = None,
    ) -> VRPInstance:
        """Create a random VRP instance for testing.
        
        Args:
            n_customers: Number of customers
            n_vehicles: Number of vehicles
            capacity: Vehicle capacity
            area_size: Size of the square area
            demand_range: (min, max) demand per customer
            seed: Random seed for reproducibility
            
        Returns:
            Randomly generated VRPInstance
        """
        rng = np.random.default_rng(seed)
        
        # Depot at center
        depot = Depot(id=0, x=area_size / 2, y=area_size / 2)
        
        # Random customers
        customers = []
        for i in range(1, n_customers + 1):
            x = rng.uniform(0, area_size)
            y = rng.uniform(0, area_size)
            demand = rng.integers(demand_range[0], demand_range[1] + 1)
            
            # Random time windows
            tw_start = rng.uniform(0, 500)
            tw_end = tw_start + rng.uniform(50, 200)
            service_time = rng.uniform(5, 20)
            
            customers.append(Customer(
                id=i,
                x=x,
                y=y,
                demand=int(demand),
                service_time=service_time,
                time_window_start=tw_start,
                time_window_end=tw_end,
            ))
        
        # Vehicles
        vehicles = [Vehicle(id=k, capacity=capacity) for k in range(n_vehicles)]
        
        return cls(
            name=f"random_{n_customers}c_{n_vehicles}v",
            depot=depot,
            customers=customers,
            vehicles=vehicles,
        )
    
    def __repr__(self) -> str:
        return f"VRPInstance(name='{self.name}', customers={self.n_customers}, vehicles={self.n_vehicles})"

