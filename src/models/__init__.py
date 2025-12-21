"""Data models for VRP problem representation."""

from .problem import Customer, Depot, Vehicle, VRPInstance
from .solution import Route, Solution
from .subsequence import (
    SubsequenceData,
    RouteSubsequenceData,
    concatenate,
    check_insertion_feasible_fast,
)
from .fairness import (
    coefficient_of_variation,
    jains_fairness_index,
    gini_coefficient,
    compute_fairness_objective,
    compute_driver_fairness,
    compute_customer_fairness,
    compute_all_metrics,
)
from .parsers import (
    parse_solomon,
    parse_vrplib,
    parse_cvrp,
    parse_csv,
    parse_instance,
    write_solomon,
)

__all__ = [
    # Problem classes
    "Customer",
    "Depot", 
    "Vehicle",
    "VRPInstance",
    # Solution classes
    "Route",
    "Solution",
    # Subsequence data (concatenation trick)
    "SubsequenceData",
    "RouteSubsequenceData",
    "concatenate",
    "check_insertion_feasible_fast",
    # Fairness metrics
    "coefficient_of_variation",
    "jains_fairness_index",
    "gini_coefficient",
    "compute_fairness_objective",
    "compute_driver_fairness",
    "compute_customer_fairness",
    "compute_all_metrics",
    # Parsers
    "parse_solomon",
    "parse_vrplib",
    "parse_cvrp",
    "parse_csv",
    "parse_instance",
    "write_solomon",
]

