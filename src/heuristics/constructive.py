"""Constructive heuristics for VRP initial solution generation.

Provides fast heuristics to generate feasible initial solutions:
- Nearest Neighbor: Greedy sequential insertion
- Clarke-Wright Savings: Merge routes based on savings
- Sweep Algorithm: Polar angle-based clustering
- Sequential Insertion: Best insertion heuristic
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import logging

from ..models.solution import Solution, Route

if TYPE_CHECKING:
    from ..models.problem import VRPInstance, Customer

logger = logging.getLogger(__name__)


def nearest_neighbor(instance: VRPInstance) -> Solution:
    """Nearest Neighbor heuristic for VRP.
    
    Builds routes sequentially by always visiting the nearest
    unvisited customer that doesn't violate constraints.
    
    Time Complexity: O(n² * m) where n = customers, m = vehicles
    
    Args:
        instance: VRP problem instance
        
    Returns:
        Feasible solution (may not serve all customers if infeasible)
    """
    unvisited = set(range(1, instance.n_customers + 1))
    routes = []
    
    for vehicle_idx, vehicle in enumerate(instance.vehicles):
        if not unvisited:
            # No more customers, add empty route
            routes.append(Route(vehicle_id=vehicle_idx))
            continue
        
        route = Route(vehicle_id=vehicle_idx)
        current_node = 0  # Start at depot
        current_load = 0
        current_time = 0.0
        
        while unvisited:
            # Find nearest feasible customer
            best_customer = None
            best_distance = float('inf')
            
            for cust_id in unvisited:
                customer = instance.get_customer(cust_id)
                
                # Check capacity
                if current_load + customer.demand > vehicle.capacity:
                    continue
                
                # Check time window (can we arrive before it closes?)
                travel_time = instance.get_travel_time(current_node, cust_id)
                arrival_time = current_time + travel_time
                
                # Wait if arriving early
                if arrival_time < customer.time_window_start:
                    arrival_time = customer.time_window_start
                
                # Check if we can arrive before window closes
                if arrival_time > customer.time_window_end:
                    continue
                
                # Check if we can return to depot in time
                departure_time = arrival_time + customer.service_time
                return_time = departure_time + instance.get_travel_time(cust_id, 0)
                if return_time > instance.depot.time_window_end:
                    continue
                
                # Feasible - check distance
                distance = instance.get_distance(current_node, cust_id)
                if distance < best_distance:
                    best_distance = distance
                    best_customer = cust_id
            
            if best_customer is None:
                # No feasible customer found, close this route
                break
            
            # Add customer to route
            customer = instance.get_customer(best_customer)
            route.customers.append(best_customer)
            unvisited.remove(best_customer)
            
            # Update state
            travel_time = instance.get_travel_time(current_node, best_customer)
            arrival_time = current_time + travel_time
            if arrival_time < customer.time_window_start:
                arrival_time = customer.time_window_start
            
            route.arrival_times.append(arrival_time)
            current_time = arrival_time + customer.service_time
            current_load += customer.demand
            current_node = best_customer
        
        route.load = current_load
        routes.append(route)
    
    # Create solution and compute costs
    solution = Solution(routes=routes, instance=instance)
    solution.compute_cost()
    
    if unvisited:
        logger.warning(f"Nearest neighbor: {len(unvisited)} customers unserved")
        solution.is_feasible = False
    
    return solution


def clarke_wright_savings(instance: VRPInstance, parallel: bool = True) -> Solution:
    """Clarke-Wright Savings Algorithm.
    
    Merges routes based on savings s(i,j) = d(0,i) + d(0,j) - d(i,j).
    
    Args:
        instance: VRP problem instance
        parallel: If True, use parallel version (merge any compatible pair).
                  If False, use sequential version (only merge at route ends).
        
    Returns:
        Feasible solution
    """
    n = instance.n_customers
    
    # Compute savings for all pairs
    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = (instance.get_distance(0, i) + 
                 instance.get_distance(0, j) - 
                 instance.get_distance(i, j))
            if s > 0:  # Only positive savings
                savings.append((s, i, j))
    
    # Sort by savings (descending)
    savings.sort(key=lambda x: x[0], reverse=True)
    
    # Initialize: one route per customer
    route_of = {}  # customer -> route_id
    routes_dict = {}  # route_id -> list of customers
    route_load = {}  # route_id -> total load
    
    for i in range(1, n + 1):
        route_of[i] = i
        routes_dict[i] = [i]
        route_load[i] = instance.get_customer(i).demand
    
    capacity = instance.vehicles[0].capacity
    
    # Process savings
    for s, i, j in savings:
        ri = route_of[i]
        rj = route_of[j]
        
        if ri == rj:
            continue  # Already same route
        
        # Check if merge is feasible
        if route_load[ri] + route_load[rj] > capacity:
            continue  # Capacity violated
        
        # Check positions (parallel version allows any merge)
        route_i = routes_dict[ri]
        route_j = routes_dict[rj]
        
        # Find valid merge configuration
        merged = None
        
        # Case 1: i at end of ri, j at start of rj
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        # Case 2: j at end of rj, i at start of ri
        elif route_j[-1] == j and route_i[0] == i:
            merged = route_j + route_i
        # Case 3: i at end, j at end (reverse rj)
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + list(reversed(route_j))
        # Case 4: i at start, j at start (reverse ri)
        elif route_i[0] == i and route_j[0] == j:
            merged = list(reversed(route_i)) + route_j
        
        if merged is None:
            if not parallel:
                continue  # Sequential version: strict endpoint matching
            # Parallel version: try anyway
            merged = route_i + route_j
        
        # Check time window feasibility of merged route
        if not _is_route_time_feasible(merged, instance):
            continue
        
        # Perform merge
        new_route_id = ri
        routes_dict[new_route_id] = merged
        route_load[new_route_id] = route_load[ri] + route_load[rj]
        
        # Update customer assignments
        for c in route_j:
            route_of[c] = new_route_id
        
        # Remove old route
        if rj in routes_dict:
            del routes_dict[rj]
            del route_load[rj]
    
    # Convert to Solution format
    routes = []
    route_list = list(routes_dict.values())
    
    for vehicle_idx in range(instance.n_vehicles):
        if vehicle_idx < len(route_list):
            customers = route_list[vehicle_idx]
            route = Route(vehicle_id=vehicle_idx, customers=customers)
            route.load = sum(instance.get_customer(c).demand for c in customers)
        else:
            route = Route(vehicle_id=vehicle_idx)
        routes.append(route)
    
    # Handle excess routes (more routes than vehicles)
    if len(route_list) > instance.n_vehicles:
        logger.warning(f"Clarke-Wright: {len(route_list)} routes > {instance.n_vehicles} vehicles")
        # Append excess customers to last route (may violate capacity)
        for extra_route in route_list[instance.n_vehicles:]:
            routes[-1].customers.extend(extra_route)
    
    solution = Solution(routes=routes, instance=instance)
    solution.compute_cost()
    solution.compute_schedule()
    
    return solution


def _is_route_time_feasible(customers: list[int], instance: VRPInstance) -> bool:
    """Check if a route sequence is time-window feasible."""
    current_time = 0.0
    current_node = 0
    
    for cust_id in customers:
        customer = instance.get_customer(cust_id)
        
        travel_time = instance.get_travel_time(current_node, cust_id)
        arrival_time = current_time + travel_time
        
        # Wait if early
        if arrival_time < customer.time_window_start:
            arrival_time = customer.time_window_start
        
        # Check if too late
        if arrival_time > customer.time_window_end:
            return False
        
        current_time = arrival_time + customer.service_time
        current_node = cust_id
    
    # Check return to depot
    return_time = current_time + instance.get_travel_time(current_node, 0)
    return return_time <= instance.depot.time_window_end


def sweep_algorithm(instance: VRPInstance) -> Solution:
    """Sweep Algorithm for VRP.
    
    Clusters customers by polar angle from depot,
    then assigns to vehicles in sweep order.
    
    Best for instances with depot near center.
    
    Args:
        instance: VRP problem instance
        
    Returns:
        Feasible solution
    """
    # Compute polar angles from depot
    depot = instance.depot
    angles = []
    
    for customer in instance.customers:
        dx = customer.x - depot.x
        dy = customer.y - depot.y
        angle = np.arctan2(dy, dx)
        angles.append((angle, customer.id))
    
    # Sort by angle
    angles.sort(key=lambda x: x[0])
    sorted_customers = [cid for _, cid in angles]
    
    # Assign to routes in sweep order
    routes = []
    current_route = []
    current_load = 0
    capacity = instance.vehicles[0].capacity
    
    for cust_id in sorted_customers:
        customer = instance.get_customer(cust_id)
        
        if current_load + customer.demand <= capacity:
            current_route.append(cust_id)
            current_load += customer.demand
        else:
            # Start new route
            if current_route:
                routes.append(current_route)
            current_route = [cust_id]
            current_load = customer.demand
    
    # Don't forget last route
    if current_route:
        routes.append(current_route)
    
    # Convert to Solution
    solution_routes = []
    for vehicle_idx in range(instance.n_vehicles):
        if vehicle_idx < len(routes):
            customers = routes[vehicle_idx]
            route = Route(vehicle_id=vehicle_idx, customers=customers)
            route.load = sum(instance.get_customer(c).demand for c in customers)
        else:
            route = Route(vehicle_id=vehicle_idx)
        solution_routes.append(route)
    
    # Handle excess
    if len(routes) > instance.n_vehicles:
        for extra_route in routes[instance.n_vehicles:]:
            solution_routes[-1].customers.extend(extra_route)
    
    solution = Solution(routes=solution_routes, instance=instance)
    solution.compute_cost()
    
    return solution


def sequential_insertion(instance: VRPInstance) -> Solution:
    """Sequential Best Insertion Heuristic.
    
    For each unassigned customer, find the best insertion position
    across all routes.
    
    Args:
        instance: VRP problem instance
        
    Returns:
        Feasible solution
    """
    # Initialize empty routes
    routes = [Route(vehicle_id=k) for k in range(instance.n_vehicles)]
    unassigned = list(range(1, instance.n_customers + 1))
    
    # Seed each route with a customer (optional)
    # For now, start with all empty and insert one by one
    
    while unassigned:
        best_insertion = None
        best_cost = float('inf')
        best_customer = None
        best_route_idx = None
        best_position = None
        
        for cust_id in unassigned:
            customer = instance.get_customer(cust_id)
            
            for route_idx, route in enumerate(routes):
                # Check capacity
                if route.load + customer.demand > instance.vehicles[route_idx].capacity:
                    continue
                
                # Try each position in route
                for pos in range(len(route.customers) + 1):
                    # Calculate insertion cost
                    cost = _insertion_cost(instance, route, pos, cust_id)
                    
                    # Check time window feasibility
                    test_route = route.customers[:pos] + [cust_id] + route.customers[pos:]
                    if not _is_route_time_feasible(test_route, instance):
                        continue
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_customer = cust_id
                        best_route_idx = route_idx
                        best_position = pos
        
        if best_customer is None:
            # No feasible insertion found
            logger.warning(f"Sequential insertion: {len(unassigned)} customers unassigned")
            break
        
        # Perform insertion
        routes[best_route_idx].insert(best_position, best_customer)
        routes[best_route_idx].load += instance.get_customer(best_customer).demand
        unassigned.remove(best_customer)
    
    solution = Solution(routes=routes, instance=instance)
    solution.compute_cost()
    solution.compute_schedule()
    
    return solution


def _insertion_cost(
    instance: VRPInstance,
    route: Route,
    position: int,
    customer_id: int,
) -> float:
    """Calculate cost of inserting customer at position in route."""
    customers = route.customers
    
    if position == 0:
        pred = 0  # Depot
    else:
        pred = customers[position - 1]
    
    if position >= len(customers):
        succ = 0  # Depot
    else:
        succ = customers[position]
    
    # Current arc cost
    old_cost = instance.get_distance(pred, succ)
    
    # New arc costs
    new_cost = (instance.get_distance(pred, customer_id) + 
                instance.get_distance(customer_id, succ))
    
    return new_cost - old_cost


def regret_insertion(instance: VRPInstance, regret_k: int = 2) -> Solution:
    """Regret-k Insertion Heuristic.
    
    Prioritizes customers with high regret (difference between
    best and k-th best insertion cost).
    
    Args:
        instance: VRP problem instance
        regret_k: Number of alternatives to consider (default 2)
        
    Returns:
        Feasible solution
    """
    routes = [Route(vehicle_id=k) for k in range(instance.n_vehicles)]
    unassigned = list(range(1, instance.n_customers + 1))
    
    while unassigned:
        best_regret = -float('inf')
        best_customer = None
        best_route_idx = None
        best_position = None
        
        for cust_id in unassigned:
            customer = instance.get_customer(cust_id)
            
            # Collect all feasible insertions
            insertions = []
            
            for route_idx, route in enumerate(routes):
                if route.load + customer.demand > instance.vehicles[route_idx].capacity:
                    continue
                
                for pos in range(len(route.customers) + 1):
                    test_route = route.customers[:pos] + [cust_id] + route.customers[pos:]
                    if not _is_route_time_feasible(test_route, instance):
                        continue
                    
                    cost = _insertion_cost(instance, route, pos, cust_id)
                    insertions.append((cost, route_idx, pos))
            
            if not insertions:
                continue
            
            # Sort by cost
            insertions.sort(key=lambda x: x[0])
            
            # Calculate regret
            if len(insertions) >= regret_k:
                regret = sum(insertions[i][0] for i in range(1, regret_k)) - (regret_k - 1) * insertions[0][0]
            else:
                regret = 0  # Not enough alternatives
            
            if regret > best_regret or (regret == best_regret and insertions[0][0] < best_regret):
                best_regret = regret
                best_customer = cust_id
                best_route_idx = insertions[0][1]
                best_position = insertions[0][2]
        
        if best_customer is None:
            logger.warning(f"Regret insertion: {len(unassigned)} customers unassigned")
            break
        
        # Perform insertion
        routes[best_route_idx].insert(best_position, best_customer)
        routes[best_route_idx].load += instance.get_customer(best_customer).demand
        unassigned.remove(best_customer)
    
    solution = Solution(routes=routes, instance=instance)
    solution.compute_cost()
    solution.compute_schedule()
    
    return solution


def create_initial_solution(
    instance: VRPInstance,
    method: str = "clarke_wright",
) -> Solution:
    """Create initial solution using specified method.
    
    Args:
        instance: VRP problem instance
        method: One of 'nearest_neighbor', 'clarke_wright', 'sweep', 
                'sequential', 'regret'
                
    Returns:
        Initial feasible solution
    """
    methods = {
        "nearest_neighbor": nearest_neighbor,
        "nn": nearest_neighbor,
        "clarke_wright": clarke_wright_savings,
        "cw": clarke_wright_savings,
        "savings": clarke_wright_savings,
        "sweep": sweep_algorithm,
        "sequential": sequential_insertion,
        "insertion": sequential_insertion,
        "regret": regret_insertion,
    }
    
    method_lower = method.lower()
    if method_lower not in methods:
        logger.warning(f"Unknown method '{method}', using clarke_wright")
        method_lower = "clarke_wright"
    
    return methods[method_lower](instance)

