"""Heuristic algorithms for VRP solving."""

from .constructive import (
    nearest_neighbor,
    clarke_wright_savings,
    sweep_algorithm,
    sequential_insertion,
    regret_insertion,
    create_initial_solution,
)

# OR-Tools is optional dependency
try:
    from .ortools_solver import ORToolsSolver, ORToolsConfig, solve_vrp_ortools
    _HAS_ORTOOLS = True
except ImportError:
    ORToolsSolver = None
    ORToolsConfig = None
    solve_vrp_ortools = None
    _HAS_ORTOOLS = False

# Destroy operators
from .destroy import (
    DestroyOperator,
    DestroyResult,
    RandomRemoval,
    WorstRemoval,
    RelatedRemoval,
    RouteRemoval,
    FairnessRemoval,
    get_destroy_operators,
)

# Repair operators
from .repair import (
    RepairOperator,
    RepairResult,
    HNSWGreedyRepair,
    HNSWRegretRepair,
    HNSWFastRepair,  # New O(1) concatenation-based repair
    GreedyRepair,
    RegretRepair,
    get_repair_operators,
)

# ALNS
from .alns import ALNS, ALNSConfig, ALNSStatistics, create_alns

__all__ = [
    # Constructive heuristics
    "nearest_neighbor",
    "clarke_wright_savings",
    "sweep_algorithm",
    "sequential_insertion",
    "regret_insertion",
    "create_initial_solution",
    # OR-Tools solver (optional)
    "ORToolsSolver",
    "ORToolsConfig",
    "solve_vrp_ortools",
    "_HAS_ORTOOLS",
    # Destroy operators
    "DestroyOperator",
    "DestroyResult",
    "RandomRemoval",
    "WorstRemoval",
    "RelatedRemoval",
    "RouteRemoval",
    "FairnessRemoval",
    "get_destroy_operators",
    # Repair operators
    "RepairOperator",
    "RepairResult",
    "HNSWGreedyRepair",
    "HNSWRegretRepair",
    "HNSWFastRepair",
    "GreedyRepair",
    "RegretRepair",
    "get_repair_operators",
    # ALNS
    "ALNS",
    "ALNSConfig",
    "ALNSStatistics",
    "create_alns",
]

