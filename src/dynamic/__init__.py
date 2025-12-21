"""Dynamic VRP extensions for real-time order arrivals."""

from .event_generator import (
    EventGenerator,
    EventGeneratorConfig,
    EventType,
    DynamicEvent,
    DynamicScenario,
    create_dynamic_scenario,
    _HAS_SIMPY,
)
from .handler import (
    DynamicHandler,
    DynamicHandlerConfig,
    HandlerStatistics,
    create_dynamic_handler,
)

__all__ = [
    # Event generator
    "EventGenerator",
    "EventGeneratorConfig",
    "EventType",
    "DynamicEvent",
    "DynamicScenario",
    "create_dynamic_scenario",
    "_HAS_SIMPY",
    # Handler
    "DynamicHandler",
    "DynamicHandlerConfig",
    "HandlerStatistics",
    "create_dynamic_handler",
]
