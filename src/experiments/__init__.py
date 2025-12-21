"""Experiment orchestration and analysis."""

from .run_experiment import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    run_quick_benchmark,
    main as run_experiment_main,
)
from .analysis import (
    load_results,
    aggregate_by_instance,
    statistical_tests,
    plot_convergence,
    plot_pareto_front,
    plot_operator_usage,
    generate_report,
    AggregatedResults,
)

__all__ = [
    # Experiment runner
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "run_quick_benchmark",
    "run_experiment_main",
    # Analysis
    "load_results",
    "aggregate_by_instance",
    "statistical_tests",
    "plot_convergence",
    "plot_pareto_front",
    "plot_operator_usage",
    "generate_report",
    "AggregatedResults",
]
