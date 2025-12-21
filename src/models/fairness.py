"""Fairness metric implementations for VRP.

This module provides fairness metrics for evaluating VRP solutions:
- Coefficient of Variation (CV): Driver workload fairness
- Jain's Fairness Index: Customer waiting time fairness
- Combined multi-objective function

References:
    - Jain, R., Chiu, D. M., & Hawe, W. R. (1984). 
      A quantitative measure of fairness and discrimination.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .solution import Solution


def coefficient_of_variation(values: np.ndarray) -> float:
    """Compute Coefficient of Variation (CV) for driver workload fairness.
    
    CV = σ / μ, where σ is standard deviation and μ is mean.
    Lower values indicate more equitable distribution.
    CV = 0 means all values are equal (perfectly fair).
    
    Args:
        values: Array of values (e.g., route costs or durations)
        
    Returns:
        CV value (0.0 if all values equal or empty/zero mean)
    """
    if len(values) == 0:
        return 0.0
    
    mean = np.mean(values)
    if mean == 0:
        return 0.0
    
    std = np.std(values)
    return float(std / mean)


def jains_fairness_index(values: np.ndarray) -> float:
    """Compute Jain's Fairness Index for customer waiting time fairness.
    
    J(x₁, ..., xₙ) = (Σxᵢ)² / (n · Σxᵢ²)
    
    Properties:
    - Range: [1/n, 1]
    - J = 1 means all values are equal (perfectly fair)
    - J = 1/n means maximum unfairness (one has all, others have none)
    - Higher is better
    
    Note: For waiting times, we typically want LOW waiting times but
    EQUAL distribution. So J measures equality of waiting distribution.
    
    Args:
        values: Array of values (e.g., customer waiting times)
        
    Returns:
        Jain's index value (1.0 if empty or all zeros)
    """
    if len(values) == 0:
        return 1.0
    
    total_sum = np.sum(values)
    if total_sum == 0:
        return 1.0  # All zeros = perfectly equal
    
    n = len(values)
    sum_squared = total_sum ** 2
    squared_sum = n * np.sum(values ** 2)
    
    if squared_sum == 0:
        return 1.0
    
    return float(sum_squared / squared_sum)


def gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient for inequality measurement.
    
    Properties:
    - Range: [0, 1]
    - G = 0 means perfect equality
    - G = 1 means maximum inequality
    - Lower is better (more fair)
    
    Args:
        values: Array of non-negative values
        
    Returns:
        Gini coefficient value
    """
    if len(values) == 0:
        return 0.0
    
    values = np.array(values, dtype=np.float64)
    if np.sum(values) == 0:
        return 0.0
    
    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Compute Gini using formula: G = (2 * Σᵢ(i * xᵢ)) / (n * Σxᵢ) - (n+1)/n
    index = np.arange(1, n + 1)
    return float(
        (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
    )


def max_min_ratio(values: np.ndarray) -> float:
    """Compute max/min ratio for fairness.
    
    Properties:
    - Range: [1, ∞)
    - Ratio = 1 means perfect equality
    - Higher means more unfair
    
    Args:
        values: Array of positive values
        
    Returns:
        Max/min ratio (inf if min is 0, 1.0 if empty)
    """
    if len(values) == 0:
        return 1.0
    
    min_val = np.min(values)
    max_val = np.max(values)
    
    if min_val == 0:
        if max_val == 0:
            return 1.0
        return float('inf')
    
    return float(max_val / min_val)


def range_ratio(values: np.ndarray) -> float:
    """Compute range as ratio of mean.
    
    (max - min) / mean
    
    Properties:
    - 0 means all equal
    - Higher means more spread
    
    Args:
        values: Array of values
        
    Returns:
        Range ratio value
    """
    if len(values) == 0:
        return 0.0
    
    mean = np.mean(values)
    if mean == 0:
        return 0.0
    
    return float((np.max(values) - np.min(values)) / mean)


def compute_driver_fairness(solution: Solution) -> dict[str, float]:
    """Compute driver fairness metrics from solution.
    
    Args:
        solution: VRP solution
        
    Returns:
        Dictionary with fairness metrics for driver workload
    """
    route_costs = solution.get_route_costs()
    # Only consider non-empty routes
    non_zero = route_costs[route_costs > 0]
    
    if len(non_zero) == 0:
        return {
            "cv": 0.0,
            "jain": 1.0,
            "gini": 0.0,
            "max_min_ratio": 1.0,
            "range_ratio": 0.0,
        }
    
    return {
        "cv": coefficient_of_variation(non_zero),
        "jain": jains_fairness_index(non_zero),
        "gini": gini_coefficient(non_zero),
        "max_min_ratio": max_min_ratio(non_zero),
        "range_ratio": range_ratio(non_zero),
    }


def compute_customer_fairness(solution: Solution) -> dict[str, float]:
    """Compute customer fairness metrics from solution.
    
    Based on customer waiting times.
    
    Args:
        solution: VRP solution with computed schedules
        
    Returns:
        Dictionary with fairness metrics for customer waiting
    """
    waiting_times = solution.get_customer_waiting_times()
    
    if len(waiting_times) == 0:
        return {
            "jain": 1.0,
            "cv": 0.0,
            "mean_waiting": 0.0,
            "max_waiting": 0.0,
            "total_waiting": 0.0,
        }
    
    return {
        "jain": jains_fairness_index(waiting_times),
        "cv": coefficient_of_variation(waiting_times),
        "mean_waiting": float(np.mean(waiting_times)),
        "max_waiting": float(np.max(waiting_times)),
        "total_waiting": float(np.sum(waiting_times)),
    }


def compute_fairness_objective(
    solution: Solution,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> float:
    """Compute multi-objective fairness-aware cost function.
    
    f(R; α, β, γ) = α·cost + β·CV - γ·J
    
    Where:
    - cost: Total route cost (to minimize)
    - CV: Coefficient of variation of route costs (to minimize)
    - J: Jain's fairness index of customer waiting (to maximize, hence negative)
    
    Args:
        solution: VRP solution
        alpha: Weight for total cost (default 1.0)
        beta: Weight for driver fairness CV penalty (default 0.3)
        gamma: Weight for customer fairness Jain bonus (default 0.2)
        
    Returns:
        Combined objective value (lower is better)
    """
    # Total cost
    total_cost = solution.total_cost
    
    # Driver fairness: CV of route costs
    route_costs = solution.get_route_costs()
    non_zero_costs = route_costs[route_costs > 0]
    cv = coefficient_of_variation(non_zero_costs) if len(non_zero_costs) > 0 else 0.0
    
    # Customer fairness: Jain's index of waiting times
    waiting_times = solution.get_customer_waiting_times()
    jain = jains_fairness_index(waiting_times) if len(waiting_times) > 0 else 1.0
    
    # Combined objective: minimize cost and CV, maximize Jain's
    objective = alpha * total_cost + beta * cv - gamma * jain
    
    return float(objective)


def compute_all_metrics(solution: Solution) -> dict[str, float]:
    """Compute all fairness and quality metrics for a solution.
    
    Args:
        solution: VRP solution
        
    Returns:
        Dictionary with all metrics
    """
    driver_metrics = compute_driver_fairness(solution)
    customer_metrics = compute_customer_fairness(solution)
    
    return {
        "total_cost": solution.total_cost,
        "n_routes": solution.n_routes_used(),
        "n_customers": solution.n_customers_served(),
        "is_feasible": solution.is_feasible,
        # Driver fairness
        "driver_cv": driver_metrics["cv"],
        "driver_jain": driver_metrics["jain"],
        "driver_gini": driver_metrics["gini"],
        "driver_max_min_ratio": driver_metrics["max_min_ratio"],
        # Customer fairness
        "customer_jain": customer_metrics["jain"],
        "customer_cv": customer_metrics["cv"],
        "mean_waiting": customer_metrics["mean_waiting"],
        "max_waiting": customer_metrics["max_waiting"],
        "total_waiting": customer_metrics["total_waiting"],
    }


def normalize_objective(
    cost: float,
    cv: float,
    jain: float,
    cost_bounds: tuple[float, float] | None = None,
) -> tuple[float, float, float]:
    """Normalize objective components to [0, 1] range.
    
    Useful for Pareto analysis and visualization.
    
    Args:
        cost: Total cost
        cv: Coefficient of variation
        jain: Jain's fairness index
        cost_bounds: Optional (min, max) for cost normalization
        
    Returns:
        Tuple of (normalized_cost, normalized_cv, normalized_jain)
    """
    # Normalize cost
    if cost_bounds:
        min_cost, max_cost = cost_bounds
        if max_cost > min_cost:
            norm_cost = (cost - min_cost) / (max_cost - min_cost)
        else:
            norm_cost = 0.0
    else:
        norm_cost = cost  # No normalization without bounds
    
    # CV is already somewhat normalized (typically 0-2 range)
    norm_cv = min(cv / 2.0, 1.0)
    
    # Jain's is already in [1/n, 1], map to [0, 1]
    # Assuming at least n=2, Jain's minimum is 0.5
    norm_jain = max(0.0, min(1.0, (jain - 0.5) / 0.5)) if jain < 1.0 else 1.0
    
    return (float(norm_cost), float(norm_cv), float(norm_jain))

