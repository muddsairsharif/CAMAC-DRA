"""EV Environment module for CAMAC-DRA.

This module defines the main EVEnvironment class that integrates electric vehicles,
charging stations, and grid simulation into a cohesive simulation environment.
"""

from typing import List, Dict, Tuple, Optional
from src.environment.ev import EV
from src.environment.charging_station import ChargingStation
from src.environment.grid_simulator import GridSimulator


class EVEnvironment:
    """Integrates EV, charging stations, and grid simulation."""

    def __init__(self, num_stations=1, grid_capacity_kw=1000.0, time_step_minutes=15):
        """Initialize the EV environment.

        Args:
            num_stations: Number of charging stations to create
            grid_capacity_kw: Total grid capacity in kW
            time_step_minutes: Time step duration in minutes
        """
        self.num_stations = num_stations
        self.time_step_minutes = time_step_minutes
        self.time_step_hours = time_step_minutes / 60.0
        self.current_time = 0
        
        # Initialize components
        self.grid = GridSimulator(grid_capacity_kw, renewable_capacity_kw=grid_capacity_kw * 0.3)
        self.stations: Dict[int, ChargingStation] = {}
        self.evs: Dict[int, EV] = {}
        
        # Statistics
        self.total_energy_charged_kwh = 0.0
        self.total_cost = 0.0
        self.total_emissions_kg_co2 = 0.0

    def add_charging_station(self, station_id: int, num_chargers: int, max_power_kw: float) -> bool:
        """Add a charging station to the environment.

        Args:
            station_id: Unique ID for the station
            num_chargers: Number of chargers at the station
            max_power_kw: Maximum power capacity in kW

        Returns:
            True if station was added successfully, False otherwise
        """
        if station_id in self.stations:
            return False
        self.stations[station_id] = ChargingStation(station_id, num_chargers, max_power_kw)
        return True

    def add_ev(self, ev_id: int, battery_capacity_kwh: float, initial_soc: float = 0.0) -> bool:
        """Add an electric vehicle to the environment.

        Args:
            ev_id: Unique ID for the EV
            battery_capacity_kwh: Battery capacity in kWh
            initial_soc: Initial state of charge (0.0 to 1.0)

        Returns:
            True if EV was added successfully, False otherwise
        """
        if ev_id in self.evs:
            return False
        self.evs[ev_id] = EV(ev_id, battery_capacity_kwh, initial_soc)
        return True

    def get_ev(self, ev_id: int) -> Optional[EV]:
        """Get an EV by ID.

        Args:
            ev_id: ID of the EV

        Returns:
            EV object if found, None otherwise
        """
        return self.evs.get(ev_id)

    def get_station(self, station_id: int) -> Optional[ChargingStation]:
        """Get a charging station by ID.

        Args:
            station_id: ID of the station

        Returns:
            ChargingStation object if found, None otherwise
        """
        return self.stations.get(station_id)

    def step(self):
        """Execute one time step of the simulation."""
        # Update EV charging
        total_charging_power = 0.0
        for station in self.stations.values():
            total_charging_power += station.get_total_power_usage()
            
            # Update SOC for all charging EVs
            for ev in station.chargers.values():
                if ev is not None and ev.is_charging:
                    energy_charged = ev.current_charging_power * self.time_step_hours
                    ev.update_soc(ev.current_charging_power, self.time_step_hours)
                    self.total_energy_charged_kwh += energy_charged
                    self.total_cost += energy_charged * self.grid.price_per_kwh
                    emissions = self.grid.calculate_carbon_emissions(energy_charged)
                    self.total_emissions_kg_co2 += emissions
        
        # Update grid demand
        self.grid.update_demand(total_charging_power)
        self.current_time += self.time_step_minutes

    def get_status(self) -> Dict:
        """Get a status dictionary of the environment.

        Returns:
            Dictionary containing current status information
        """
        total_evs_charging = sum(
            1 for station in self.stations.values()
            for ev in station.chargers.values()
            if ev is not None and ev.is_charging
        )
        
        return {
            'current_time': self.current_time,
            'grid_load': self.grid.get_grid_load(),
            'available_power_kw': self.grid.get_available_power(),
            'total_evs': len(self.evs),
            'charging_evs': total_evs_charging,
            'total_energy_charged_kwh': self.total_energy_charged_kwh,
            'total_cost_usd': self.total_cost,
            'total_emissions_kg_co2': self.total_emissions_kg_co2,
        }

    def __repr__(self):
        """String representation of the environment."""
        return f"EVEnvironment(stations={len(self.stations)}, evs={len(self.evs)}, time={self.current_time}min)"
