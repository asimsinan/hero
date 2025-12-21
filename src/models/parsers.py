"""Parsers for standard VRP benchmark formats.

Supports:
- Solomon VRPTW format
- VRPLib/TSPLIB format
- CVRP format
- Custom CSV format
"""
from __future__ import annotations

from pathlib import Path
from typing import TextIO
import numpy as np

from .problem import Customer, Depot, Vehicle, VRPInstance


def parse_solomon(filepath: str | Path) -> VRPInstance:
    """Parse Solomon VRPTW benchmark format.
    
    Format:
    - Line 1: Instance name
    - Line 5: Number of vehicles, vehicle capacity
    - Line 10+: Node data (id, x, y, demand, ready_time, due_date, service_time)
    
    Args:
        filepath: Path to Solomon format file
        
    Returns:
        Parsed VRPInstance
        
    Example file:
        C101
        
        VEHICLE
        NUMBER     CAPACITY
          25         200
        
        CUSTOMER
        CUST NO.   XCOORD.   YCOORD.    DEMAND   READY TIME   DUE DATE   SERVICE TIME
            0      40        50           0          0       1236          0
            1      45        68          10        912        967         90
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse instance name
    name = lines[0].strip()
    if not name:
        name = filepath.stem
    
    # Find vehicle info line (contains two numbers)
    n_vehicles = 25  # Default
    capacity = 200   # Default
    
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 2:
            try:
                n_vehicles = int(parts[0])
                capacity = int(parts[1])
                break
            except ValueError:
                continue
    
    # Parse node data (find first line with 7 numbers)
    customers = []
    depot = None
    
    for line in lines:
        parts = line.split()
        if len(parts) >= 7:
            try:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = int(parts[3])
                ready_time = float(parts[4])
                due_date = float(parts[5])
                service_time = float(parts[6])
                
                if node_id == 0:
                    depot = Depot(
                        id=0,
                        x=x,
                        y=y,
                        time_window_start=ready_time,
                        time_window_end=due_date,
                    )
                else:
                    customers.append(Customer(
                        id=node_id,
                        x=x,
                        y=y,
                        demand=demand,
                        service_time=service_time,
                        time_window_start=ready_time,
                        time_window_end=due_date,
                    ))
            except ValueError:
                continue
    
    if depot is None:
        raise ValueError(f"No depot (node 0) found in {filepath}")
    
    if not customers:
        raise ValueError(f"No customers found in {filepath}")
    
    # Ensure customer IDs are consecutive starting from 1
    customers.sort(key=lambda c: c.id)
    
    # Create vehicles
    vehicles = [Vehicle(id=k, capacity=capacity) for k in range(n_vehicles)]
    
    return VRPInstance(
        name=name,
        depot=depot,
        customers=customers,
        vehicles=vehicles,
    )


def parse_vrplib(filepath: str | Path) -> VRPInstance:
    """Parse VRPLib/TSPLIB format.
    
    Format:
        NAME : <instance name>
        COMMENT : <comment>
        TYPE : CVRP
        DIMENSION : <n>
        EDGE_WEIGHT_TYPE : EUC_2D
        CAPACITY : <capacity>
        NODE_COORD_SECTION
        1 x1 y1
        2 x2 y2
        ...
        DEMAND_SECTION
        1 d1
        2 d2
        ...
        DEPOT_SECTION
        1
        -1
        EOF
    
    Args:
        filepath: Path to VRPLib format file
        
    Returns:
        Parsed VRPInstance
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    
    # Parse header
    name = filepath.stem
    dimension = 0
    capacity = 100
    edge_weight_type = "EUC_2D"
    
    coords = {}
    demands = {}
    depot_ids = []
    
    section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse header fields
        if ':' in line and section is None:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()
            
            if key == "NAME":
                name = value
            elif key == "DIMENSION":
                dimension = int(value)
            elif key == "CAPACITY":
                capacity = int(value)
            elif key == "EDGE_WEIGHT_TYPE":
                edge_weight_type = value
            continue
        
        # Detect sections
        if "NODE_COORD_SECTION" in line:
            section = "coords"
            continue
        elif "DEMAND_SECTION" in line:
            section = "demand"
            continue
        elif "DEPOT_SECTION" in line:
            section = "depot"
            continue
        elif line == "EOF":
            break
        
        # Parse section data
        parts = line.split()
        
        if section == "coords" and len(parts) >= 3:
            try:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords[node_id] = (x, y)
            except ValueError:
                continue
        
        elif section == "demand" and len(parts) >= 2:
            try:
                node_id = int(parts[0])
                demand = int(parts[1])
                demands[node_id] = demand
            except ValueError:
                continue
        
        elif section == "depot" and len(parts) >= 1:
            try:
                depot_id = int(parts[0])
                if depot_id > 0:
                    depot_ids.append(depot_id)
            except ValueError:
                continue
    
    if not coords:
        raise ValueError(f"No coordinates found in {filepath}")
    
    # Determine depot (first in depot section, or node 1)
    depot_id = depot_ids[0] if depot_ids else 1
    depot_x, depot_y = coords.get(depot_id, (0, 0))
    depot = Depot(id=0, x=depot_x, y=depot_y)
    
    # Create customers
    customers = []
    for node_id, (x, y) in coords.items():
        if node_id == depot_id:
            continue
        
        demand = demands.get(node_id, 0)
        
        # Remap customer ID to be 1-indexed excluding depot
        customers.append(Customer(
            id=len(customers) + 1,
            x=x,
            y=y,
            demand=demand,
        ))
    
    # Estimate number of vehicles
    total_demand = sum(c.demand for c in customers)
    n_vehicles = max(1, -(-total_demand // capacity))  # Ceiling division
    n_vehicles = min(n_vehicles * 2, len(customers))  # Add buffer
    
    vehicles = [Vehicle(id=k, capacity=capacity) for k in range(n_vehicles)]
    
    return VRPInstance(
        name=name,
        depot=depot,
        customers=customers,
        vehicles=vehicles,
    )


def parse_cvrp(filepath: str | Path) -> VRPInstance:
    """Parse simple CVRP format (no time windows).
    
    Alias for parse_vrplib.
    """
    return parse_vrplib(filepath)


def parse_csv(filepath: str | Path, has_header: bool = True) -> VRPInstance:
    """Parse custom CSV format.
    
    Expected columns:
    - id: Node ID (0 = depot)
    - x: X coordinate
    - y: Y coordinate
    - demand: Demand (0 for depot)
    - tw_start: Time window start (optional)
    - tw_end: Time window end (optional)
    - service_time: Service time (optional)
    
    Args:
        filepath: Path to CSV file
        has_header: Whether file has header row
        
    Returns:
        Parsed VRPInstance
    """
    import csv
    
    filepath = Path(filepath)
    
    customers = []
    depot = None
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        
        if has_header:
            header = next(reader)
            # Find column indices
            col_map = {name.lower().strip(): idx for idx, name in enumerate(header)}
        else:
            # Assume standard order: id, x, y, demand, [tw_start, tw_end, service_time]
            col_map = {"id": 0, "x": 1, "y": 2, "demand": 3, 
                       "tw_start": 4, "tw_end": 5, "service_time": 6}
        
        for row in reader:
            if not row:
                continue
            
            try:
                node_id = int(row[col_map.get("id", 0)])
                x = float(row[col_map.get("x", 1)])
                y = float(row[col_map.get("y", 2)])
                demand = int(row[col_map.get("demand", 3)])
                
                # Optional time window fields
                tw_start = 0.0
                tw_end = float('inf')
                service_time = 0.0
                
                if "tw_start" in col_map and col_map["tw_start"] < len(row):
                    tw_start = float(row[col_map["tw_start"]])
                if "tw_end" in col_map and col_map["tw_end"] < len(row):
                    tw_end = float(row[col_map["tw_end"]])
                if "service_time" in col_map and col_map["service_time"] < len(row):
                    service_time = float(row[col_map["service_time"]])
                
                if node_id == 0:
                    depot = Depot(id=0, x=x, y=y, 
                                  time_window_start=tw_start, time_window_end=tw_end)
                else:
                    customers.append(Customer(
                        id=node_id,
                        x=x,
                        y=y,
                        demand=demand,
                        service_time=service_time,
                        time_window_start=tw_start,
                        time_window_end=tw_end,
                    ))
            except (ValueError, IndexError):
                continue
    
    if depot is None:
        # Default depot at origin
        depot = Depot(id=0, x=0.0, y=0.0)
    
    if not customers:
        raise ValueError(f"No customers found in {filepath}")
    
    # Sort customers by ID
    customers.sort(key=lambda c: c.id)
    
    # Renumber if needed
    customers = [
        Customer(
            id=i + 1,
            x=c.x, y=c.y,
            demand=c.demand,
            service_time=c.service_time,
            time_window_start=c.time_window_start,
            time_window_end=c.time_window_end,
        )
        for i, c in enumerate(customers)
    ]
    
    # Estimate vehicles
    total_demand = sum(c.demand for c in customers)
    capacity = 100
    n_vehicles = max(1, -(-total_demand // capacity) * 2)
    
    vehicles = [Vehicle(id=k, capacity=capacity) for k in range(n_vehicles)]
    
    return VRPInstance(
        name=filepath.stem,
        depot=depot,
        customers=customers,
        vehicles=vehicles,
    )


def parse_instance(filepath: str | Path) -> VRPInstance:
    """Auto-detect format and parse instance.
    
    Detection based on file extension and content.
    
    Args:
        filepath: Path to instance file
        
    Returns:
        Parsed VRPInstance
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == ".csv":
        return parse_csv(filepath)
    
    # Try to detect format from content
    with open(filepath, 'r') as f:
        first_lines = [f.readline() for _ in range(10)]
    
    content = '\n'.join(first_lines).upper()
    
    if "DIMENSION" in content or "NODE_COORD_SECTION" in content:
        return parse_vrplib(filepath)
    elif "VEHICLE" in content or "CUSTOMER" in content:
        return parse_solomon(filepath)
    else:
        # Default to Solomon format
        return parse_solomon(filepath)


def write_solomon(instance: VRPInstance, filepath: str | Path) -> None:
    """Write instance in Solomon format.
    
    Args:
        instance: VRPInstance to write
        filepath: Output file path
    """
    filepath = Path(filepath)
    
    with open(filepath, 'w') as f:
        f.write(f"{instance.name}\n")
        f.write("\n")
        f.write("VEHICLE\n")
        f.write("NUMBER     CAPACITY\n")
        f.write(f"  {instance.n_vehicles:3d}         {instance.vehicles[0].capacity}\n")
        f.write("\n")
        f.write("CUSTOMER\n")
        f.write("CUST NO.   XCOORD.   YCOORD.    DEMAND   READY TIME   DUE DATE   SERVICE TIME\n")
        f.write("\n")
        
        # Depot
        d = instance.depot
        f.write(f"    {d.id:3d}      {d.x:5.0f}     {d.y:5.0f}          0          "
                f"{d.time_window_start:4.0f}       {d.time_window_end:4.0f}          0\n")
        
        # Customers
        for c in instance.customers:
            f.write(f"    {c.id:3d}      {c.x:5.0f}     {c.y:5.0f}         {c.demand:2d}          "
                    f"{c.time_window_start:4.0f}       {c.time_window_end:4.0f}         {c.service_time:2.0f}\n")

