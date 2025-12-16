"""Grid Simulator module for CAMAC-DRA.

This module defines the GridSimulator class for simulating electrical grid
behavior and managing power supply dynamics.
"""

from typing import List, Dict


class GridSimulator:
    """Simulates the electrical grid and power supply dynamics."""

    def __init__(self, total_capacity_kw, renewable_capacity_kw=0.0):
        """Initialize the grid simulator.

        Args:
            total_capacity_kw: Total grid capacity in kW
            renewable_capacity_kw: Renewable energy capacity in kW
        """
        self.total_capacity_kw = total_capacity_kw
        self.renewable_capacity_kw = renewable_capacity_kw
        self.current_demand_kw = 0.0
        self.current_renewable_production_kw = 0.0
        self.carbon_emission_rate = 0.5  # kg CO2/kWh (can be tuned)
        self.price_per_kwh = 0.12  # USD/kWh (can be dynamic)

    def update_demand(self, demand_kw: float) -> bool:
        """Update the current demand on the grid.

        Args:
            demand_kw: Current demand in kW

        Returns:
            True if demand is within capacity, False otherwise
        """
        if demand_kw > self.total_capacity_kw:
            return False
        self.current_demand_kw = demand_kw
        return True

    def update_renewable_production(self, production_kw: float) -> bool:
        """Update the current renewable energy production.

        Args:
            production_kw: Current renewable production in kW

        Returns:
            True if production is within renewable capacity, False otherwise
        """
        if production_kw > self.renewable_capacity_kw:
            return False
        self.current_renewable_production_kw = production_kw
        return True

    def get_grid_load(self) -> float:
        """Get the current grid load as a percentage.

        Returns:
            Grid load percentage (0.0 to 1.0)
        """
        return min(1.0, self.current_demand_kw / self.total_capacity_kw)

    def get_available_power(self) -> float:
        """Get the available power for charging.

        Returns:
            Available power in kW
        """
        return max(0.0, self.total_capacity_kw - self.current_demand_kw)

    def get_green_power_ratio(self) -> float:
        """Get the ratio of renewable energy in current production.

        Returns:
            Ratio of green power (0.0 to 1.0)
        """
        if self.current_demand_kw == 0:
            return 0.0
        return min(1.0, self.current_renewable_production_kw / self.current_demand_kw)

    def calculate_carbon_emissions(self, energy_kwh: float) -> float:
        """Calculate carbon emissions for energy consumption.

        Args:
            energy_kwh: Energy in kWh

        Returns:
            Carbon emissions in kg CO2
        """
        green_ratio = self.get_green_power_ratio()
        non_green_energy = energy_kwh * (1.0 - green_ratio)
        return non_green_energy * self.carbon_emission_rate

    def set_price_per_kwh(self, price: float):
        """Set the electricity price.

        Args:
            price: Price per kWh in USD
        """
        self.price_per_kwh = max(0.0, price)

    def __repr__(self):
        """String representation of the grid simulator."""
        return f"GridSimulator(capacity={self.total_capacity_kw}kW, demand={self.current_demand_kw:.1f}kW, load={self.get_grid_load():.1%})"
