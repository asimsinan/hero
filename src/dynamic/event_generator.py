"""Dynamic event generation for VRP simulation.

This module provides event generators for simulating dynamic VRP scenarios
where new orders arrive over time. Uses SimPy for discrete-event simulation.

Event Types:
- NEW_ORDER: A new customer order arrives
- CANCEL_ORDER: An existing order is cancelled
- UPDATE_ORDER: Order details change (time window, demand)
- VEHICLE_BREAKDOWN: A vehicle becomes unavailable
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Iterator, Callable
from enum import Enum, auto
import random
import numpy as np

try:
    import simpy
    _HAS_SIMPY = True
except ImportError:
    simpy = None
    _HAS_SIMPY = False

if TYPE_CHECKING:
    from ..models.problem import Customer, VRPInstance

import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of dynamic events."""
    NEW_ORDER = auto()
    CANCEL_ORDER = auto()
    UPDATE_ORDER = auto()
    VEHICLE_BREAKDOWN = auto()


@dataclass
class DynamicEvent:
    """Represents a dynamic event in the VRP simulation.
    
    Attributes:
        time: Time when event occurs
        event_type: Type of event
        customer_id: Customer involved (for order events)
        customer: New customer object (for NEW_ORDER)
        data: Additional event data
    """
    time: float
    event_type: EventType
    customer_id: Optional[int] = None
    customer: Optional['Customer'] = None
    data: dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"DynamicEvent(t={self.time:.2f}, type={self.event_type.name}, cid={self.customer_id})"


@dataclass
class EventGeneratorConfig:
    """Configuration for event generation.
    
    Attributes:
        simulation_horizon: Total simulation time
        mean_arrival_rate: Average orders per time unit (Poisson)
        order_cancel_prob: Probability an order gets cancelled
        order_update_prob: Probability an order gets updated
        vehicle_breakdown_prob: Probability of vehicle breakdown
        
        # Customer generation parameters
        coord_range: (min, max) for customer coordinates
        demand_range: (min, max) for customer demand
        service_time_range: (min, max) for service time
        time_window_width: Average time window width
        
        seed: Random seed
    """
    simulation_horizon: float = 480.0  # 8 hours
    mean_arrival_rate: float = 0.5  # orders per time unit
    order_cancel_prob: float = 0.05
    order_update_prob: float = 0.02
    vehicle_breakdown_prob: float = 0.01
    
    coord_range: tuple = (0.0, 100.0)
    demand_range: tuple = (1, 20)
    service_time_range: tuple = (5.0, 15.0)
    time_window_width: float = 60.0
    
    seed: Optional[int] = None


class EventGenerator:
    """Generates dynamic events for VRP simulation.
    
    Can operate in two modes:
    1. SimPy mode: Real-time discrete-event simulation
    2. Batch mode: Pre-generate all events (no SimPy required)
    """
    
    def __init__(self, config: EventGeneratorConfig = None):
        self.config = config or EventGeneratorConfig()
        self.rng = random.Random(self.config.seed)
        np.random.seed(self.config.seed)
        
        self._next_customer_id = 1000  # Start dynamic customers at 1000
        self._events: List[DynamicEvent] = []
    
    def generate_events(self) -> List[DynamicEvent]:
        """Generate all events in batch mode.
        
        Returns:
            List of DynamicEvent objects sorted by time
        """
        events = []
        current_time = 0.0
        
        while current_time < self.config.simulation_horizon:
            # Generate inter-arrival time (exponential distribution)
            if self.config.mean_arrival_rate > 0:
                inter_arrival = self.rng.expovariate(self.config.mean_arrival_rate)
            else:
                inter_arrival = self.config.simulation_horizon  # No arrivals
            
            current_time += inter_arrival
            
            if current_time >= self.config.simulation_horizon:
                break
            
            # Generate new order event
            customer = self._generate_customer(current_time)
            event = DynamicEvent(
                time=current_time,
                event_type=EventType.NEW_ORDER,
                customer_id=customer.id,
                customer=customer,
            )
            events.append(event)
            
            # Possibly generate cancel event
            if self.rng.random() < self.config.order_cancel_prob:
                cancel_time = current_time + self.rng.uniform(10, 60)
                if cancel_time < self.config.simulation_horizon:
                    cancel_event = DynamicEvent(
                        time=cancel_time,
                        event_type=EventType.CANCEL_ORDER,
                        customer_id=customer.id,
                    )
                    events.append(cancel_event)
            
            # Possibly generate update event
            if self.rng.random() < self.config.order_update_prob:
                update_time = current_time + self.rng.uniform(5, 30)
                if update_time < self.config.simulation_horizon:
                    update_event = DynamicEvent(
                        time=update_time,
                        event_type=EventType.UPDATE_ORDER,
                        customer_id=customer.id,
                        data={
                            'new_demand': self.rng.randint(*self.config.demand_range),
                            'new_tw_end': customer.time_window_end + self.rng.uniform(-30, 30),
                        }
                    )
                    events.append(update_event)
        
        # Sort by time
        events.sort(key=lambda e: e.time)
        self._events = events
        
        return events
    
    def _generate_customer(self, arrival_time: float) -> 'Customer':
        """Generate a new dynamic customer."""
        from ..models.problem import Customer
        
        customer_id = self._next_customer_id
        self._next_customer_id += 1
        
        x = self.rng.uniform(*self.config.coord_range)
        y = self.rng.uniform(*self.config.coord_range)
        demand = self.rng.randint(*self.config.demand_range)
        service_time = self.rng.uniform(*self.config.service_time_range)
        
        # Time window starts after arrival
        tw_start = arrival_time + self.rng.uniform(30, 120)
        tw_width = self.rng.uniform(
            self.config.time_window_width * 0.5,
            self.config.time_window_width * 1.5,
        )
        tw_end = tw_start + tw_width
        
        return Customer(
            id=customer_id,
            x=x,
            y=y,
            demand=demand,
            service_time=service_time,
            time_window_start=tw_start,
            time_window_end=tw_end,
            release_time=arrival_time,
        )
    
    def run_simpy(
        self,
        env: 'simpy.Environment',
        callback: Callable[[DynamicEvent], None],
    ) -> None:
        """Run event generation as a SimPy process.
        
        Args:
            env: SimPy environment
            callback: Function to call for each event
        """
        if not _HAS_SIMPY:
            raise ImportError("SimPy is required for real-time simulation. Install with: pip install simpy")
        
        def event_process():
            while env.now < self.config.simulation_horizon:
                # Generate inter-arrival time
                if self.config.mean_arrival_rate > 0:
                    inter_arrival = self.rng.expovariate(self.config.mean_arrival_rate)
                else:
                    inter_arrival = self.config.simulation_horizon - env.now
                
                yield env.timeout(inter_arrival)
                
                if env.now >= self.config.simulation_horizon:
                    break
                
                # Generate and dispatch event
                customer = self._generate_customer(env.now)
                event = DynamicEvent(
                    time=env.now,
                    event_type=EventType.NEW_ORDER,
                    customer_id=customer.id,
                    customer=customer,
                )
                
                callback(event)
                self._events.append(event)
        
        env.process(event_process())
    
    def iterate_events(
        self,
        time_step: float = 1.0,
    ) -> Iterator[List[DynamicEvent]]:
        """Iterate through events in time windows.
        
        Useful for simulating without SimPy.
        
        Args:
            time_step: Time window size
            
        Yields:
            List of events in each time window
        """
        if not self._events:
            self.generate_events()
        
        current_time = 0.0
        event_idx = 0
        
        while current_time < self.config.simulation_horizon:
            window_end = current_time + time_step
            
            # Collect events in this window
            window_events = []
            while event_idx < len(self._events) and self._events[event_idx].time < window_end:
                window_events.append(self._events[event_idx])
                event_idx += 1
            
            yield window_events
            current_time = window_end
    
    def get_statistics(self) -> dict:
        """Get statistics about generated events."""
        if not self._events:
            return {}
        
        event_counts = {}
        for event in self._events:
            event_counts[event.event_type.name] = event_counts.get(event.event_type.name, 0) + 1
        
        arrival_times = [e.time for e in self._events if e.event_type == EventType.NEW_ORDER]
        
        return {
            'total_events': len(self._events),
            'event_counts': event_counts,
            'n_new_orders': event_counts.get('NEW_ORDER', 0),
            'avg_inter_arrival': np.mean(np.diff(arrival_times)) if len(arrival_times) > 1 else 0,
            'simulation_horizon': self.config.simulation_horizon,
        }


@dataclass
class DynamicScenario:
    """A complete dynamic VRP scenario.
    
    Combines static customers (known at start) with
    dynamic events (arriving over time).
    """
    base_instance: 'VRPInstance'
    events: List[DynamicEvent]
    config: EventGeneratorConfig
    
    @property
    def n_static_customers(self) -> int:
        return self.base_instance.n_customers
    
    @property
    def n_dynamic_customers(self) -> int:
        return sum(1 for e in self.events if e.event_type == EventType.NEW_ORDER)
    
    @property
    def total_customers(self) -> int:
        return self.n_static_customers + self.n_dynamic_customers
    
    def get_events_until(self, time: float) -> List[DynamicEvent]:
        """Get all events that occur before given time."""
        return [e for e in self.events if e.time <= time]
    
    def summary(self) -> str:
        return (
            f"DynamicScenario:\n"
            f"  Static customers: {self.n_static_customers}\n"
            f"  Dynamic customers: {self.n_dynamic_customers}\n"
            f"  Total events: {len(self.events)}\n"
            f"  Horizon: {self.config.simulation_horizon}"
        )


def create_dynamic_scenario(
    base_instance: 'VRPInstance',
    simulation_horizon: float = 480.0,
    arrival_rate: float = 0.5,
    seed: Optional[int] = None,
) -> DynamicScenario:
    """Create a dynamic VRP scenario.
    
    Args:
        base_instance: Static VRP instance (initial customers)
        simulation_horizon: Total simulation time
        arrival_rate: Mean arrivals per time unit
        seed: Random seed
        
    Returns:
        DynamicScenario with events
    """
    config = EventGeneratorConfig(
        simulation_horizon=simulation_horizon,
        mean_arrival_rate=arrival_rate,
        coord_range=(0.0, 100.0),
        seed=seed,
    )
    
    generator = EventGenerator(config)
    events = generator.generate_events()
    
    return DynamicScenario(
        base_instance=base_instance,
        events=events,
        config=config,
    )

