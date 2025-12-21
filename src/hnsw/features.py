"""Feature encoder for VRP insertion positions with fairness gradients.

This module encodes insertion positions into feature vectors for HNSW indexing,
including EQUITY-AWARE features that guide the search toward fairer solutions.

**Key Design Decision**: 
Both queries (customers) and index entries (positions) must use the SAME
feature space for HNSW to work correctly. We encode positions by their
spatial/temporal context AND fairness impact.

Feature Vector (14 dimensions with subsequence-aware encoding):
1-2. Predecessor location (x, y normalized)
3-4. Successor location (x, y normalized)  
5. Time slack at position (normalized)
6. Capacity remaining (normalized)
7. Neighbor time window center (normalized)
8. Route cost ratio (normalized)
--- SUBSEQUENCE FEATURES (for temporal feasibility) ---
9. Prefix time slack (normalized) - flexibility before this position
10. Suffix time slack (normalized) - flexibility after this position
11. Estimated combined slack (normalized) - predicted feasibility window
--- FAIRNESS FEATURES ---
12. Delta CV: Impact on coefficient of variation if inserted here
13. Delta Jain: Impact on Jain's fairness index if inserted here
14. Route imbalance: How far this route's cost is from mean (equity signal)

The subsequence features enable HNSW to pre-filter positions based on
temporal compatibility, not just spatial proximity!
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple, Optional
import numpy as np

if TYPE_CHECKING:
    from ..models.problem import VRPInstance, Customer
    from ..models.solution import Solution, Route

import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature encoding.
    
    Attributes:
        time_horizon: Maximum time horizon for normalization
        coord_range: Coordinate range for normalization (min, max)
        use_fairness_features: Whether to include fairness gradient features
        use_subsequence_features: Whether to include subsequence-aware temporal features
    """
    time_horizon: float = 1500.0
    coord_range: Tuple[float, float] = (0.0, 100.0)
    use_fairness_features: bool = True  # Enable fairness features by default
    use_subsequence_features: bool = True  # Enable subsequence-aware features
    
    @property
    def feature_dim(self) -> int:
        """Return the dimension of feature vectors."""
        base_dim = 8  # Spatial + temporal + capacity + cost
        subseq_dim = 3 if self.use_subsequence_features else 0  # prefix/suffix/combined slack
        fairness_dim = 3 if self.use_fairness_features else 0
        return base_dim + subseq_dim + fairness_dim  # 14D with all features


@dataclass
class FeatureEncoder:
    """Encodes VRP insertion positions into feature vectors for HNSW.
    
    The encoder creates 11-dimensional feature vectors that capture:
    - Position context (predecessor and successor locations)
    - Temporal compatibility (time slack, neighbor time windows)
    - Capacity compatibility (remaining capacity)
    - Route cost context
    - FAIRNESS GRADIENTS (delta CV, delta Jain, route imbalance)
    
    The fairness features enable HNSW to prioritize insertions that improve
    equity among drivers (CV) and customers (Jain's index).
    """
    config: FeatureConfig = field(default_factory=FeatureConfig)
    _instance: Optional['VRPInstance'] = field(default=None, init=False)
    _mean_cost: float = field(default=100.0, init=False)
    _std_cost: float = field(default=50.0, init=False)
    _route_costs: List[float] = field(default_factory=list, init=False)
    
    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim
    
    def fit(self, instance: 'VRPInstance', solution: 'Solution') -> None:
        """Fit encoder to instance (compute normalization stats)."""
        self._instance = instance
        
        # Compute route cost statistics for fairness features
        costs = [r.cost for r in solution.routes if r.customers]
        self._route_costs = costs
        self._mean_cost = np.mean(costs) if costs else 100.0
        self._std_cost = np.std(costs) if len(costs) > 1 else 50.0
        
        # Update time horizon based on instance
        max_tw = max(
            c.time_window_end 
            for c in [instance.get_customer(i) for i in range(1, instance.n_customers + 1)]
        )
        self.config.time_horizon = max(max_tw, self.config.time_horizon)
        
        # Update coord range based on instance
        all_coords = [(c.x, c.y) for c in [instance.depot] + 
                      [instance.get_customer(i) for i in range(1, instance.n_customers + 1)]]
        min_coord = min(min(x, y) for x, y in all_coords)
        max_coord = max(max(x, y) for x, y in all_coords)
        self.config.coord_range = (min_coord, max_coord)
    
    def update_solution_stats(self, solution: 'Solution') -> None:
        """Update statistics based on current solution (for incremental updates)."""
        costs = [r.cost for r in solution.routes if r.customers]
        self._route_costs = costs
        self._mean_cost = np.mean(costs) if costs else 100.0
        self._std_cost = np.std(costs) if len(costs) > 1 else 50.0
    
    def _normalize_coord(self, coord: float) -> float:
        """Normalize coordinate to [0, 1]."""
        cmin, cmax = self.config.coord_range
        return (coord - cmin) / (cmax - cmin + 1e-8)
    
    def _normalize_time(self, time: float) -> float:
        """Normalize time to [0, 1]."""
        return time / (self.config.time_horizon + 1e-8)
    
    def _compute_cv(self, costs: List[float]) -> float:
        """Compute coefficient of variation."""
        if len(costs) < 2:
            return 0.0
        mean = np.mean(costs)
        if mean < 1e-8:
            return 0.0
        return np.std(costs) / mean
    
    def _compute_jain(self, costs: List[float]) -> float:
        """Compute Jain's fairness index."""
        if len(costs) < 1:
            return 1.0
        costs = np.array(costs)
        sum_c = np.sum(costs)
        sum_sq = np.sum(costs ** 2)
        if sum_sq < 1e-8:
            return 1.0
        return (sum_c ** 2) / (len(costs) * sum_sq)
    
    def _compute_fairness_features(
        self,
        route: 'Route',
        position: int,
        instance: 'VRPInstance',
        customer_demand: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Compute fairness gradient features for a position.
        
        Returns:
            Tuple of (delta_cv_norm, delta_jain_norm, route_imbalance_norm)
            
        These features guide HNSW toward positions that improve fairness:
        - Lower delta_cv = better (less increase in CV)
        - Higher delta_jain = better (more increase in Jain's index)
        - Lower route_imbalance = prefer underloaded routes
        """
        if not self._route_costs or len(self._route_costs) < 2:
            return 0.5, 0.5, 0.5
        
        # Current fairness metrics
        current_cv = self._compute_cv(self._route_costs)
        current_jain = self._compute_jain(self._route_costs)
        
        # Estimate cost increase from insertion
        # Simple estimate: average distance to neighbors
        if position == 0:
            pred = instance.depot
        else:
            pred = instance.get_customer(route.customers[position - 1])
        
        if position >= len(route.customers):
            succ = instance.depot
        else:
            succ = instance.get_customer(route.customers[position])
        
        # Estimate insertion cost
        # If customer_demand is 0, we're encoding position (not customer query)
        # Use a typical insertion cost estimate
        if customer_demand > 0:
            # This is a customer query - estimate impact
            estimated_cost_increase = 20.0  # Typical insertion cost
        else:
            # This is a position encoding - use route context
            dist_pred_succ = np.sqrt((pred.x - succ.x)**2 + (pred.y - succ.y)**2)
            estimated_cost_increase = dist_pred_succ * 0.5  # Rough estimate
        
        # Simulate new costs
        route_idx = None
        for i, c in enumerate(self._route_costs):
            if abs(c - route.cost) < 1e-6:
                route_idx = i
                break
        
        if route_idx is None:
            route_idx = 0
        
        simulated_costs = self._route_costs.copy()
        if route_idx < len(simulated_costs):
            simulated_costs[route_idx] += estimated_cost_increase
        
        # Compute fairness after simulated insertion
        new_cv = self._compute_cv(simulated_costs)
        new_jain = self._compute_jain(simulated_costs)
        
        # Delta values (positive = worse, negative = better)
        delta_cv = new_cv - current_cv
        delta_jain = new_jain - current_jain  # Positive = better for Jain
        
        # Route imbalance: how far this route is from mean
        # Positive = overloaded, Negative = underloaded
        route_imbalance = (route.cost - self._mean_cost) / (self._std_cost + 1e-8)
        
        # Normalize to [0, 1] range for HNSW
        # delta_cv: typically in [-0.2, 0.2], map to [0, 1]
        delta_cv_norm = np.clip((delta_cv + 0.2) / 0.4, 0, 1)
        
        # delta_jain: typically in [-0.1, 0.1], map to [0, 1]
        # We want higher delta_jain (improvement) to have LOWER distance
        # So we invert: 1 - normalized value
        delta_jain_norm = 1.0 - np.clip((delta_jain + 0.1) / 0.2, 0, 1)
        
        # route_imbalance: typically in [-2, 2], map to [0, 1]
        # We want to prefer underloaded routes (negative imbalance)
        route_imbalance_norm = np.clip((route_imbalance + 2) / 4, 0, 1)
        
        return delta_cv_norm, delta_jain_norm, route_imbalance_norm
    
    def encode_customer(
        self,
        customer_id: int,
        instance: 'VRPInstance',
    ) -> np.ndarray:
        """Encode a customer as a query vector.
        
        The customer's location becomes the "ideal" predecessor/successor
        location, and their time window defines the temporal requirements.
        Fairness features express preference for fair insertions.
        
        Args:
            customer_id: ID of the customer
            instance: VRP problem instance
            
        Returns:
            Feature vector of shape (11,) or (8,) depending on config
        """
        customer = instance.get_customer(customer_id)
        
        # Spatial features (want position near customer)
        x_norm = self._normalize_coord(customer.x)
        y_norm = self._normalize_coord(customer.y)
        
        # Time window info
        tw_center = (customer.time_window_start + customer.time_window_end) / 2
        tw_center_norm = self._normalize_time(tw_center)
        
        # Capacity need
        capacity = instance.vehicles[0].capacity if instance.vehicles else 200
        demand_ratio = customer.demand / capacity
        capacity_need = 1.0 - demand_ratio
        
        base_features = [
            x_norm,           # pred_x: want position near customer
            y_norm,           # pred_y
            x_norm,           # succ_x: want position near customer
            y_norm,           # succ_y
            0.5,              # time_slack: moderate slack preferred
            capacity_need,    # capacity: need this much remaining
            tw_center_norm,   # neighbor_tw: compatible time windows
            0.5,              # cost_ratio: moderate cost routes
        ]
        
        features = base_features
        
        # Subsequence-aware temporal requirements
        if self.config.use_subsequence_features:
            # Customer's time window flexibility determines how much slack they need
            tw_width = customer.time_window_end - customer.time_window_start
            tw_flexibility = min(1.0, tw_width / (self.config.time_horizon / 2 + 1e-8))
            
            # Service time determines minimum slack needed
            service_need = customer.service_time / (self.config.time_horizon / 10 + 1e-8)
            service_need = min(1.0, service_need)
            
            # Customers prefer positions with at least their flexibility level
            # Lower flexibility customers (tight windows) need positions with MORE slack
            # Higher flexibility customers can tolerate tighter positions
            min_slack_required = 1.0 - tw_flexibility  # Tight window = need more slack
            
            subseq_features = [
                0.5,  # prefix_slack: prefer moderate prefix flexibility
                0.5,  # suffix_slack: prefer moderate suffix flexibility  
                max(0.2, min_slack_required),  # combined_slack: need at least this much
            ]
            features = features + subseq_features
        
        # Fairness preferences for customer queries
        if self.config.use_fairness_features:
            # - Prefer low delta_cv (insertions that don't increase CV)
            # - Prefer high delta_jain (insertions that improve Jain)
            # - Prefer underloaded routes (negative imbalance)
            fairness_features = [
                0.3,  # delta_cv_norm: prefer positions with low CV impact
                0.3,  # delta_jain_norm: prefer positions that improve Jain
                0.3,  # route_imbalance: prefer underloaded routes
            ]
            features = features + fairness_features
        
        return np.array(features, dtype=np.float32)
    
    def encode_insertion_position(
        self,
        customer_id: int,  # Can be used for customer-specific encoding
        route: 'Route',
        position: int,
        instance: 'VRPInstance',
        solution: 'Solution',
    ) -> np.ndarray:
        """Encode an insertion position by its context and fairness impact.
        
        Args:
            customer_id: Customer (can inform fairness gradient)
            route: Route where position is located
            position: Position in route (0 = before first customer)
            instance: VRP problem instance
            solution: Current solution
            
        Returns:
            Feature vector of shape (11,) or (8,)
        """
        customer = instance.get_customer(customer_id) if customer_id > 0 else None
        customer_demand = customer.demand if customer else 0
        
        return self._encode_position_impl(
            route, position, instance, customer_demand
        )
    
    def encode_position_context(
        self,
        route: 'Route',
        position: int,
        instance: 'VRPInstance',
    ) -> np.ndarray:
        """Encode a position without needing a customer ID.
        
        Used when building the index (we don't know which customer yet).
        
        Args:
            route: Route where position is located
            position: Position in route
            instance: VRP problem instance
            
        Returns:
            Feature vector of shape (11,) or (8,)
        """
        return self._encode_position_impl(route, position, instance, 0.0)
    
    def _encode_position_impl(
        self,
        route: 'Route',
        position: int,
        instance: 'VRPInstance',
        customer_demand: float,
    ) -> np.ndarray:
        """Internal implementation for position encoding."""
        # Get predecessor and successor
        if position == 0:
            pred = instance.depot
        else:
            pred = instance.get_customer(route.customers[position - 1])
        
        if position >= len(route.customers):
            succ = instance.depot
        else:
            succ = instance.get_customer(route.customers[position])
        
        # Spatial context
        pred_x = self._normalize_coord(pred.x)
        pred_y = self._normalize_coord(pred.y)
        succ_x = self._normalize_coord(succ.x)
        succ_y = self._normalize_coord(succ.y)
        
        # Time slack
        time_slack = self._compute_time_slack(route, position, instance)
        time_slack_norm = self._normalize_time(time_slack)
        
        # Capacity remaining
        vehicle = instance.vehicles[route.vehicle_id]
        capacity_remaining = (vehicle.capacity - route.load) / vehicle.capacity
        
        # Neighbor time window average
        pred_tw = getattr(pred, 'time_window_end', 0)
        succ_tw = getattr(succ, 'time_window_start', self.config.time_horizon)
        neighbor_tw = (pred_tw + succ_tw) / 2
        neighbor_tw_norm = self._normalize_time(neighbor_tw)
        
        # Route cost ratio
        cost_ratio = route.cost / (self._mean_cost + 1e-8)
        cost_ratio = np.clip(cost_ratio, 0, 2)
        
        base_features = [
            pred_x,
            pred_y,
            succ_x,
            succ_y,
            time_slack_norm,
            capacity_remaining,
            neighbor_tw_norm,
            cost_ratio / 2,
        ]
        
        features = base_features
        
        # Add subsequence-aware temporal features
        if self.config.use_subsequence_features:
            prefix_slack, suffix_slack, combined_slack = self._compute_subsequence_features(
                route, position, instance
            )
            subseq_features = [prefix_slack, suffix_slack, combined_slack]
            features = features + subseq_features
        
        # Add fairness gradient features
        if self.config.use_fairness_features:
            delta_cv, delta_jain, route_imbalance = self._compute_fairness_features(
                route, position, instance, customer_demand
            )
            fairness_features = [delta_cv, delta_jain, route_imbalance]
            features = features + fairness_features
        
        return np.array(features, dtype=np.float32)
    
    def _compute_time_slack(
        self,
        route: 'Route',
        position: int,
        instance: 'VRPInstance',
    ) -> float:
        """Compute time slack available at a position."""
        if position == 0:
            earliest_arrival = 0.0
        else:
            pred_id = route.customers[position - 1]
            pred = instance.get_customer(pred_id)
            earliest_arrival = pred.time_window_start + pred.service_time
        
        if position >= len(route.customers):
            latest_departure = instance.depot.time_window_end
        else:
            succ_id = route.customers[position]
            succ = instance.get_customer(succ_id)
            latest_departure = succ.time_window_end
        
        slack = max(0, latest_departure - earliest_arrival)
        return slack
    
    def _compute_subsequence_features(
        self,
        route: 'Route',
        position: int,
        instance: 'VRPInstance',
    ) -> Tuple[float, float, float]:
        """Compute subsequence-aware temporal features for a position.
        
        These features encode the time window flexibility BEFORE and AFTER
        the insertion point, enabling HNSW to filter by temporal compatibility.
        
        Returns:
            Tuple of (prefix_slack_norm, suffix_slack_norm, combined_slack_norm)
            
        The key insight: positions with high combined_slack are more likely
        to be feasible for insertions!
        """
        # Compute prefix slack: flexibility from depot to this position
        # This is the difference between latest and earliest departure from depot
        # that still allows reaching position on time
        
        prefix_earliest = 0.0  # Can always start from depot at time 0
        prefix_latest = float('inf')
        prefix_duration = 0.0
        
        current_node = 0  # depot
        current_time_earliest = instance.depot.time_window_start
        current_time_latest = instance.depot.time_window_end
        
        # Simulate prefix: depot -> customer[0] -> ... -> customer[position-1]
        for i in range(position):
            if i >= len(route.customers):
                break
            cust_id = route.customers[i]
            customer = instance.get_customer(cust_id)
            travel = instance.get_travel_time(current_node, cust_id)
            
            # Update earliest arrival
            earliest_arrival = current_time_earliest + travel
            earliest_arrival = max(earliest_arrival, customer.time_window_start)
            
            # Update latest arrival  
            latest_arrival = current_time_latest + travel
            latest_arrival = min(latest_arrival, customer.time_window_end)
            
            prefix_duration += travel + customer.service_time
            current_time_earliest = earliest_arrival + customer.service_time
            current_time_latest = latest_arrival + customer.service_time
            current_node = cust_id
        
        # Prefix slack = latest - earliest at the insertion point
        prefix_slack = max(0, current_time_latest - current_time_earliest)
        
        # Compute suffix slack: flexibility from this position to depot
        suffix_earliest = float('inf')
        suffix_latest = 0.0
        
        if position < len(route.customers):
            # There are customers after this position
            first_succ_id = route.customers[position]
            first_succ = instance.get_customer(first_succ_id)
            suffix_earliest = first_succ.time_window_start
            suffix_latest = first_succ.time_window_end
        else:
            # Position is at the end, suffix is just return to depot
            suffix_earliest = 0.0
            suffix_latest = instance.depot.time_window_end
        
        suffix_slack = max(0, suffix_latest - suffix_earliest)
        
        # Combined slack estimate: how much room for a new customer?
        # This is approximate but captures the key constraint
        # A customer with service time S needs at least S slack in the combined window
        
        # Simple heuristic: combined slack is min of prefix and suffix slack
        # minus the gap needed for travel
        if position == 0:
            pred_node = 0
        else:
            pred_node = route.customers[position - 1]
            
        if position >= len(route.customers):
            succ_node = 0
        else:
            succ_node = route.customers[position]
        
        # Estimate travel time gap
        travel_gap = instance.get_travel_time(pred_node, succ_node)
        
        # Combined slack = overlap of feasible windows
        combined_slack = min(prefix_slack, suffix_slack) - travel_gap
        combined_slack = max(0, combined_slack)
        
        # Normalize all to [0, 1]
        prefix_slack_norm = min(1.0, prefix_slack / (self.config.time_horizon + 1e-8))
        suffix_slack_norm = min(1.0, suffix_slack / (self.config.time_horizon + 1e-8))
        combined_slack_norm = min(1.0, combined_slack / (self.config.time_horizon / 2 + 1e-8))
        
        return prefix_slack_norm, suffix_slack_norm, combined_slack_norm
    
    def encode_all_positions(
        self,
        instance: 'VRPInstance',
        solution: 'Solution',
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Encode all valid insertion positions in a solution.
        
        Args:
            instance: VRP problem instance
            solution: Current solution
            
        Returns:
            Tuple of (features array, list of (route_id, position) tuples)
        """
        # Update solution stats for fairness features
        self.update_solution_stats(solution)
        
        features_list = []
        labels = []
        
        for route in solution.routes:
            # Skip routes with no capacity
            vehicle = instance.vehicles[route.vehicle_id]
            if route.load >= vehicle.capacity:
                continue
            
            # Encode each position in the route
            for pos in range(len(route.customers) + 1):
                feat = self.encode_position_context(route, pos, instance)
                features_list.append(feat)
                labels.append((route.vehicle_id, pos))
        
        if not features_list:
            return np.array([]).reshape(0, self.feature_dim), []
        
        features = np.vstack(features_list)
        return features, labels


def create_feature_encoder(
    time_horizon: float = 1500.0,
    coord_range: Tuple[float, float] = (0.0, 100.0),
    use_fairness_features: bool = True,
    use_subsequence_features: bool = True,
) -> FeatureEncoder:
    """Factory function to create a FeatureEncoder.
    
    Args:
        time_horizon: Maximum time for normalization
        coord_range: (min, max) coordinate range
        use_fairness_features: Whether to include fairness gradient features
        use_subsequence_features: Whether to include subsequence-aware temporal features
        
    Returns:
        Configured FeatureEncoder
    """
    config = FeatureConfig(
        time_horizon=time_horizon,
        coord_range=coord_range,
        use_fairness_features=use_fairness_features,
        use_subsequence_features=use_subsequence_features,
    )
    return FeatureEncoder(config=config)
