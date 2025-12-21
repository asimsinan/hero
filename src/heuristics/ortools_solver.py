"""OR-Tools VRP solver wrapper.

Provides a high-level interface to Google OR-Tools for solving VRPTW
as a baseline comparison for HNSW-FairVRP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import logging

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from ..models.problem import VRPInstance
from ..models.solution import Solution, Route

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ORToolsConfig:
    """Configuration for OR-Tools solver.
    
    Attributes:
        time_limit_seconds: Maximum solving time
        first_solution_strategy: Strategy for initial solution
        local_search_metaheuristic: Metaheuristic for local search
        solution_limit: Maximum number of solutions (0 = no limit)
        log_search: Enable OR-Tools search logging
    """
    time_limit_seconds: int = 30
    first_solution_strategy: str = "PATH_CHEAPEST_ARC"
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH"
    solution_limit: int = 0
    log_search: bool = False


class ORToolsSolver:
    """Wrapper for OR-Tools VRPTW solver.
    
    Supports:
    - Capacity constraints
    - Time windows
    - Service times
    - Multiple vehicles
    
    Example:
        >>> solver = ORToolsSolver(time_limit_seconds=60)
        >>> solution = solver.solve(instance)
        >>> print(f"Cost: {solution.total_cost}")
    """
    
    # Strategy mappings
    FIRST_SOLUTION_STRATEGIES = {
        "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        "PATH_MOST_CONSTRAINED_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
        "SAVINGS": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        "SWEEP": routing_enums_pb2.FirstSolutionStrategy.SWEEP,
        "CHRISTOFIDES": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        "PARALLEL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
        "LOCAL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
        "GLOBAL_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
        "LOCAL_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC,
        "FIRST_UNBOUND_MIN_VALUE": routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE,
    }
    
    METAHEURISTICS = {
        "GUIDED_LOCAL_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        "SIMULATED_ANNEALING": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        "TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        "GENERIC_TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH,
        "GREEDY_DESCENT": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
        "AUTOMATIC": routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
    }
    
    # Scale factor for integer conversion (OR-Tools uses integers)
    SCALE_FACTOR = 100
    
    def __init__(
        self,
        time_limit_seconds: int = 30,
        first_solution_strategy: str = "PATH_CHEAPEST_ARC",
        local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH",
        log_search: bool = False,
    ):
        """Initialize OR-Tools solver.
        
        Args:
            time_limit_seconds: Maximum solving time
            first_solution_strategy: Strategy for initial solution
            local_search_metaheuristic: Metaheuristic for improvement
            log_search: Enable search logging
        """
        self.config = ORToolsConfig(
            time_limit_seconds=time_limit_seconds,
            first_solution_strategy=first_solution_strategy,
            local_search_metaheuristic=local_search_metaheuristic,
            log_search=log_search,
        )
    
    def solve(self, instance: VRPInstance) -> Solution:
        """Solve VRP instance using OR-Tools.
        
        Args:
            instance: VRP problem instance
            
        Returns:
            Solution object with routes
            
        Raises:
            ValueError: If no solution found
        """
        logger.info(f"Solving {instance.name} with OR-Tools...")
        
        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(
            instance.n_nodes,      # Number of locations (depot + customers)
            instance.n_vehicles,   # Number of vehicles
            0                      # Depot index
        )
        
        # Create routing model
        routing = pywrapcp.RoutingModel(manager)
        
        # Register callbacks
        self._add_distance_callback(routing, manager, instance)
        self._add_capacity_constraints(routing, manager, instance)
        self._add_time_window_constraints(routing, manager, instance)
        
        # Set search parameters
        search_params = self._get_search_parameters()
        
        # Solve
        logger.debug("Starting OR-Tools search...")
        assignment = routing.SolveWithParameters(search_params)
        
        if assignment is None:
            logger.warning("OR-Tools found no solution")
            # Return empty solution
            return Solution.empty(instance)
        
        # Extract solution
        solution = self._extract_solution(routing, manager, assignment, instance)
        
        logger.info(f"OR-Tools solution: cost={solution.total_cost:.2f}, "
                   f"routes={solution.n_routes_used()}")
        
        return solution
    
    def _add_distance_callback(
        self,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        instance: VRPInstance,
    ) -> int:
        """Add distance callback to routing model.
        
        Returns:
            Transit callback index
        """
        def distance_callback(from_index: int, to_index: int) -> int:
            """Return scaled distance between nodes."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            distance = instance.get_distance(from_node, to_node)
            return int(distance * self.SCALE_FACTOR)
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        return transit_callback_index
    
    def _add_capacity_constraints(
        self,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        instance: VRPInstance,
    ) -> None:
        """Add capacity constraints to routing model."""
        def demand_callback(index: int) -> int:
            """Return demand at node."""
            node = manager.IndexToNode(index)
            if node == 0:
                return 0  # Depot has no demand
            return instance.get_customer(node).demand
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        # Get capacities per vehicle
        capacities = [v.capacity for v in instance.vehicles]
        
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,                    # Null capacity slack
            capacities,           # Vehicle maximum capacities
            True,                 # Start cumul to zero
            "Capacity"
        )
    
    def _add_time_window_constraints(
        self,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        instance: VRPInstance,
    ) -> None:
        """Add time window constraints to routing model."""
        def time_callback(from_index: int, to_index: int) -> int:
            """Return travel time + service time."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            
            travel_time = instance.get_travel_time(from_node, to_node)
            
            # Add service time at from_node (except depot)
            if from_node > 0:
                service_time = instance.get_customer(from_node).service_time
            else:
                service_time = 0
            
            return int((travel_time + service_time) * self.SCALE_FACTOR)
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        # Find maximum time window end for horizon
        max_time = max(
            instance.depot.time_window_end,
            max(c.time_window_end for c in instance.customers)
        )
        horizon = int(max_time * self.SCALE_FACTOR) + 1
        
        routing.AddDimension(
            time_callback_index,
            horizon,              # Allow waiting time
            horizon,              # Maximum time per vehicle
            False,                # Don't force start cumul to zero
            "Time"
        )
        
        time_dimension = routing.GetDimensionOrDie("Time")
        
        # Add time window constraints for depot
        depot_idx = manager.NodeToIndex(0)
        time_dimension.CumulVar(depot_idx).SetRange(
            int(instance.depot.time_window_start * self.SCALE_FACTOR),
            int(instance.depot.time_window_end * self.SCALE_FACTOR)
        )
        
        # Add time window constraints for each customer
        for customer in instance.customers:
            index = manager.NodeToIndex(customer.id)
            time_dimension.CumulVar(index).SetRange(
                int(customer.time_window_start * self.SCALE_FACTOR),
                int(customer.time_window_end * self.SCALE_FACTOR)
            )
        
        # Minimize time slack (optional)
        for vehicle_id in range(instance.n_vehicles):
            start_idx = routing.Start(vehicle_id)
            time_dimension.CumulVar(start_idx).SetRange(0, horizon)
    
    def _get_search_parameters(self) -> pywrapcp.DefaultRoutingSearchParameters:
        """Create and configure search parameters."""
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        
        # First solution strategy
        strategy_name = self.config.first_solution_strategy
        if strategy_name in self.FIRST_SOLUTION_STRATEGIES:
            search_params.first_solution_strategy = (
                self.FIRST_SOLUTION_STRATEGIES[strategy_name]
            )
        else:
            logger.warning(f"Unknown strategy {strategy_name}, using PATH_CHEAPEST_ARC")
            search_params.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
        
        # Local search metaheuristic
        meta_name = self.config.local_search_metaheuristic
        if meta_name in self.METAHEURISTICS:
            search_params.local_search_metaheuristic = self.METAHEURISTICS[meta_name]
        else:
            logger.warning(f"Unknown metaheuristic {meta_name}, using GUIDED_LOCAL_SEARCH")
            search_params.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
        
        # Time limit
        search_params.time_limit.seconds = self.config.time_limit_seconds
        
        # Solution limit
        if self.config.solution_limit > 0:
            search_params.solution_limit = self.config.solution_limit
        
        # Logging
        search_params.log_search = self.config.log_search
        
        return search_params
    
    def _extract_solution(
        self,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        assignment: pywrapcp.Assignment,
        instance: VRPInstance,
    ) -> Solution:
        """Extract solution from OR-Tools assignment.
        
        Args:
            routing: OR-Tools routing model
            manager: Index manager
            assignment: Solved assignment
            instance: Original problem instance
            
        Returns:
            Converted Solution object
        """
        routes = []
        total_cost = 0.0
        
        time_dimension = routing.GetDimensionOrDie("Time")
        
        for vehicle_id in range(instance.n_vehicles):
            route = Route(vehicle_id=vehicle_id)
            index = routing.Start(vehicle_id)
            
            route_distance = 0.0
            arrival_times = []
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                
                if node > 0:  # Skip depot
                    route.customers.append(node)
                    
                    # Get arrival time
                    time_var = time_dimension.CumulVar(index)
                    arrival = assignment.Value(time_var) / self.SCALE_FACTOR
                    arrival_times.append(arrival)
                
                # Get next index
                next_index = assignment.Value(routing.NextVar(index))
                
                # Accumulate distance
                route_distance += routing.GetArcCostForVehicle(
                    index, next_index, vehicle_id
                ) / self.SCALE_FACTOR
                
                index = next_index
            
            route.arrival_times = arrival_times
            route.cost = route_distance
            total_cost += route_distance
            
            # Compute load
            if route.customers:
                route.load = sum(
                    instance.get_customer(cid).demand
                    for cid in route.customers
                )
            
            routes.append(route)
        
        solution = Solution(
            routes=routes,
            instance=instance,
            total_cost=total_cost,
        )
        
        # Validate
        is_feasible, _ = solution.validate()
        solution.is_feasible = is_feasible
        
        return solution
    
    def solve_with_initial(
        self,
        instance: VRPInstance,
        initial_solution: Solution,
    ) -> Solution:
        """Solve with a provided initial solution.
        
        Note: OR-Tools has limited support for warm starts.
        This attempts to lock in the initial routes.
        
        Args:
            instance: VRP problem instance
            initial_solution: Starting solution
            
        Returns:
            Improved solution
        """
        # For now, just solve normally
        # Full warm-start implementation would require setting
        # initial route hints via routing.AddSoftSameVehicleConstraint
        logger.warning("Warm-start not fully implemented, solving from scratch")
        return self.solve(instance)


def solve_vrp_ortools(
    instance: VRPInstance,
    time_limit: int = 30,
    strategy: str = "PATH_CHEAPEST_ARC",
    metaheuristic: str = "GUIDED_LOCAL_SEARCH",
) -> Solution:
    """Convenience function to solve VRP with OR-Tools.
    
    Args:
        instance: VRP problem instance
        time_limit: Maximum solving time in seconds
        strategy: First solution strategy
        metaheuristic: Local search metaheuristic
        
    Returns:
        Solution object
    """
    solver = ORToolsSolver(
        time_limit_seconds=time_limit,
        first_solution_strategy=strategy,
        local_search_metaheuristic=metaheuristic,
    )
    return solver.solve(instance)

