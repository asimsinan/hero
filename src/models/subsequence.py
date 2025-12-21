"""Subsequence data structures for O(1) feasibility checking.

This module implements the concatenation trick from Vidal (2012) for
efficient time window feasibility checking in VRP. Instead of simulating
the full route for each insertion position, we pre-compute subsequence
data that can be merged in O(1) time.

Reference:
    Vidal, T. (2012). "A hybrid genetic algorithm with adaptive diversity
    management for a large class of vehicle routing problems with time windows"
    Computers & Operations Research, 39(11), 2609-2627.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional
import numpy as np

if TYPE_CHECKING:
    from .problem import VRPInstance, Customer


@dataclass
class SubsequenceData:
    """Data structure for O(1) concatenation-based feasibility checking.
    
    Stores the essential information about a subsequence of customers
    that allows checking feasibility of insertions in constant time.
    
    Attributes:
        duration: Total time to traverse the subsequence (travel + service)
        earliest_start: Earliest feasible departure from first node
        latest_start: Latest feasible departure from first node
        load: Total demand in the subsequence
        first_node: First customer ID in subsequence (0 for depot)
        last_node: Last customer ID in subsequence (0 for depot)
        time_warp: Total time window violation (for soft constraints)
    """
    duration: float = 0.0
    earliest_start: float = 0.0
    latest_start: float = float('inf')
    load: int = 0
    first_node: int = 0
    last_node: int = 0
    time_warp: float = 0.0
    
    @classmethod
    def from_single_customer(
        cls,
        customer_id: int,
        instance: 'VRPInstance',
    ) -> 'SubsequenceData':
        """Create subsequence data for a single customer.
        
        Args:
            customer_id: ID of the customer
            instance: VRP instance
            
        Returns:
            SubsequenceData for this single customer
        """
        customer = instance.get_customer(customer_id)
        return cls(
            duration=customer.service_time,
            earliest_start=customer.time_window_start,
            latest_start=customer.time_window_end,
            load=customer.demand,
            first_node=customer_id,
            last_node=customer_id,
            time_warp=0.0,
        )
    
    @classmethod
    def from_depot(cls, instance: 'VRPInstance') -> 'SubsequenceData':
        """Create subsequence data for the depot.
        
        Args:
            instance: VRP instance
            
        Returns:
            SubsequenceData for the depot
        """
        depot = instance.depot
        return cls(
            duration=0.0,
            earliest_start=depot.time_window_start,
            latest_start=depot.time_window_end,
            load=0,
            first_node=0,
            last_node=0,
            time_warp=0.0,
        )
    
    def is_feasible(self, capacity: int = float('inf')) -> bool:
        """Check if this subsequence is feasible.
        
        Args:
            capacity: Vehicle capacity limit
            
        Returns:
            True if feasible (earliest <= latest and load <= capacity)
        """
        return self.earliest_start <= self.latest_start and self.load <= capacity
    
    def copy(self) -> 'SubsequenceData':
        """Create a copy of this subsequence data."""
        return SubsequenceData(
            duration=self.duration,
            earliest_start=self.earliest_start,
            latest_start=self.latest_start,
            load=self.load,
            first_node=self.first_node,
            last_node=self.last_node,
            time_warp=self.time_warp,
        )


def concatenate(
    seq_a: SubsequenceData,
    seq_b: SubsequenceData,
    travel_time: float,
) -> SubsequenceData:
    """Concatenate two subsequences in O(1) time.
    
    This is the core of the concatenation trick. Given two subsequences A and B,
    and the travel time from A's last node to B's first node, we compute the
    combined subsequence A⊕B in constant time.
    
    The key insight is that we can propagate time window constraints without
    simulating the full route:
    - earliest_AB = max(earliest_A, earliest_B - duration_A - travel)
    - latest_AB = min(latest_A, latest_B - duration_A - travel)
    
    Args:
        seq_a: First subsequence
        seq_b: Second subsequence
        travel_time: Travel time from seq_a.last_node to seq_b.first_node
        
    Returns:
        Combined subsequence A⊕B
    """
    # Total duration of combined sequence
    combined_duration = seq_a.duration + travel_time + seq_b.duration
    
    # Time shift: how much earlier/later can we start seq_b?
    # If we start seq_a at time t, we arrive at seq_b.first_node at:
    # t + seq_a.duration + travel_time
    shift = seq_a.duration + travel_time
    
    # Earliest start: we need to satisfy both A's and B's constraints
    # For B to be feasible, we need: t + shift >= seq_b.earliest_start
    # So: t >= seq_b.earliest_start - shift
    earliest_for_b = seq_b.earliest_start - shift
    combined_earliest = max(seq_a.earliest_start, earliest_for_b)
    
    # Latest start: we need to satisfy both A's and B's constraints
    # For B to be feasible, we need: t + shift <= seq_b.latest_start
    # So: t <= seq_b.latest_start - shift
    latest_for_b = seq_b.latest_start - shift
    combined_latest = min(seq_a.latest_start, latest_for_b)
    
    # Compute time warp if infeasible
    time_warp = seq_a.time_warp + seq_b.time_warp
    if combined_earliest > combined_latest:
        time_warp += combined_earliest - combined_latest
    
    return SubsequenceData(
        duration=combined_duration,
        earliest_start=combined_earliest,
        latest_start=combined_latest,
        load=seq_a.load + seq_b.load,
        first_node=seq_a.first_node,
        last_node=seq_b.last_node,
        time_warp=time_warp,
    )


@dataclass
class RouteSubsequenceData:
    """Pre-computed subsequence data for a route enabling O(1) feasibility checks.
    
    Stores prefix and suffix subsequences for each position in the route,
    allowing any insertion feasibility to be checked in O(1) time.
    
    For a route [depot, c1, c2, ..., cn, depot]:
    - prefix[i] = subsequence from depot to customer i (inclusive)
    - suffix[i] = subsequence from customer i to depot (inclusive)
    
    To check insertion of customer X at position k:
    - Combined = prefix[k-1] ⊕ X ⊕ suffix[k]
    - Check: combined.is_feasible(capacity)
    """
    prefixes: List[SubsequenceData] = field(default_factory=list)
    suffixes: List[SubsequenceData] = field(default_factory=list)
    route_length: int = 0
    _instance: Optional['VRPInstance'] = field(default=None, repr=False)
    _customer_ids: List[int] = field(default_factory=list, repr=False)
    
    def build(
        self,
        customer_ids: List[int],
        instance: 'VRPInstance',
    ) -> None:
        """Build prefix and suffix subsequence data for a route.
        
        Args:
            customer_ids: List of customer IDs in route order
            instance: VRP instance
        """
        self._instance = instance
        self._customer_ids = list(customer_ids)
        self.route_length = len(customer_ids)
        
        if not customer_ids:
            self.prefixes = [SubsequenceData.from_depot(instance)]
            self.suffixes = [SubsequenceData.from_depot(instance)]
            return
        
        # Build prefix subsequences: depot → c1 → ... → ci
        self.prefixes = []
        depot_seq = SubsequenceData.from_depot(instance)
        
        # prefix[0] = depot only (for insertion at position 0)
        self.prefixes.append(depot_seq.copy())
        
        current = depot_seq
        prev_node = 0  # depot
        
        for i, cust_id in enumerate(customer_ids):
            customer_seq = SubsequenceData.from_single_customer(cust_id, instance)
            travel = instance.get_travel_time(prev_node, cust_id)
            current = concatenate(current, customer_seq, travel)
            self.prefixes.append(current.copy())
            prev_node = cust_id
        
        # Build suffix subsequences: ci → ... → cn → depot
        self.suffixes = [None] * (len(customer_ids) + 1)  # type: ignore
        
        depot_end_seq = SubsequenceData.from_depot(instance)
        
        # suffix[n] = depot only (for insertion at last position)
        self.suffixes[-1] = depot_end_seq.copy()
        
        current = depot_end_seq
        next_node = 0  # depot
        
        for i in range(len(customer_ids) - 1, -1, -1):
            cust_id = customer_ids[i]
            customer_seq = SubsequenceData.from_single_customer(cust_id, instance)
            travel = instance.get_travel_time(cust_id, next_node)
            # Concatenate in reverse: customer ⊕ current_suffix
            current = concatenate(customer_seq, current, travel)
            self.suffixes[i] = current.copy()
            next_node = cust_id
    
    def check_insertion_feasibility(
        self,
        position: int,
        customer_id: int,
        capacity: int,
    ) -> bool:
        """Check if inserting a customer at position is feasible in O(1) time.
        
        Args:
            position: Position to insert (0 = after depot, before first customer)
            customer_id: Customer to insert
            capacity: Vehicle capacity
            
        Returns:
            True if insertion is feasible
        """
        if self._instance is None:
            raise ValueError("RouteSubsequenceData not initialized. Call build() first.")
        
        # Get customer subsequence
        customer_seq = SubsequenceData.from_single_customer(customer_id, self._instance)
        
        # Get prefix (everything before position)
        # prefix[position] is the subsequence from depot to customer at position-1
        prefix = self.prefixes[position]
        
        # Get suffix (everything from position onwards)
        suffix = self.suffixes[position]
        
        # Compute travel times
        if position == 0:
            prev_node = 0  # depot
        else:
            prev_node = self._customer_ids[position - 1]
        
        if position >= self.route_length:
            next_node = 0  # depot
        else:
            next_node = self._customer_ids[position]
        
        travel_to_customer = self._instance.get_travel_time(prev_node, customer_id)
        travel_from_customer = self._instance.get_travel_time(customer_id, next_node)
        
        # Concatenate: prefix ⊕ customer ⊕ suffix
        temp = concatenate(prefix, customer_seq, travel_to_customer)
        combined = concatenate(temp, suffix, travel_from_customer)
        
        return combined.is_feasible(capacity)
    
    def compute_insertion_cost_delta(
        self,
        position: int,
        customer_id: int,
    ) -> float:
        """Compute the cost (distance) increase from inserting a customer.
        
        This is O(1) as it only needs predecessor and successor distances.
        
        Args:
            position: Position to insert
            customer_id: Customer to insert
            
        Returns:
            Cost increase (delta) from insertion
        """
        if self._instance is None:
            raise ValueError("RouteSubsequenceData not initialized. Call build() first.")
        
        if position == 0:
            prev_node = 0
        else:
            prev_node = self._customer_ids[position - 1]
        
        if position >= self.route_length:
            next_node = 0
        else:
            next_node = self._customer_ids[position]
        
        # Current distance: prev → next
        current_dist = self._instance.get_distance(prev_node, next_node)
        
        # New distance: prev → customer → next
        new_dist = (
            self._instance.get_distance(prev_node, customer_id) +
            self._instance.get_distance(customer_id, next_node)
        )
        
        return new_dist - current_dist
    
    def find_best_insertion(
        self,
        customer_id: int,
        capacity: int,
    ) -> tuple[int, float] | None:
        """Find the best feasible insertion position for a customer.
        
        Uses O(1) feasibility check per position, total O(route_length).
        
        Args:
            customer_id: Customer to insert
            capacity: Vehicle capacity
            
        Returns:
            Tuple of (position, cost_delta) or None if no feasible position
        """
        best_pos = None
        best_delta = float('inf')
        
        for pos in range(self.route_length + 1):
            if self.check_insertion_feasibility(pos, customer_id, capacity):
                delta = self.compute_insertion_cost_delta(pos, customer_id)
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos
        
        if best_pos is not None:
            return (best_pos, best_delta)
        return None
    
    def update_after_insertion(
        self,
        position: int,
        customer_id: int,
    ) -> None:
        """Update subsequence data after inserting a customer.
        
        This is O(n) but only called when route actually changes.
        For ALNS repair, we rebuild once after all insertions.
        
        Args:
            position: Position where customer was inserted
            customer_id: Inserted customer ID
        """
        # Insert into customer list
        self._customer_ids.insert(position, customer_id)
        # Rebuild (could be optimized but O(n) is acceptable for now)
        self.build(self._customer_ids, self._instance)
    
    def __len__(self) -> int:
        """Return route length."""
        return self.route_length


# Convenience function for repair operators
def check_insertion_feasible_fast(
    route_data: RouteSubsequenceData,
    position: int,
    customer_id: int,
    capacity: int,
) -> bool:
    """Fast O(1) feasibility check using pre-computed subsequence data.
    
    Args:
        route_data: Pre-computed subsequence data for the route
        position: Position to check
        customer_id: Customer to insert
        capacity: Vehicle capacity
        
    Returns:
        True if feasible
    """
    return route_data.check_insertion_feasibility(position, customer_id, capacity)

