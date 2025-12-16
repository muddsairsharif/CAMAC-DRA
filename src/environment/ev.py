"""Electric Vehicle (EV) module for CAMAC-DRA.

This module defines the EV class and related functionality for representing
electric vehicles in the charging and grid simulation environment.
"""


class EV:
    """Represents an electric vehicle in the charging network."""

    def __init__(self, ev_id, battery_capacity, initial_soc=0.0):
        """Initialize an electric vehicle.

        Args:
            ev_id: Unique identifier for the vehicle
            battery_capacity: Maximum battery capacity in kWh
            initial_soc: Initial state of charge (0.0 to 1.0)
        """
        self.ev_id = ev_id
        self.battery_capacity = battery_capacity
        self.soc = initial_soc
        self.is_charging = False
        self.current_charging_power = 0.0

    def update_soc(self, power_kw, time_duration_hours):
        """Update the state of charge based on charging/discharging.

        Args:
            power_kw: Power in kilowatts (positive for charging, negative for discharging)
            time_duration_hours: Duration in hours
        """
        energy_kwh = power_kw * time_duration_hours
        energy_change = energy_kwh / self.battery_capacity
        self.soc = max(0.0, min(1.0, self.soc + energy_change))

    def get_remaining_capacity(self):
        """Get the remaining energy capacity that can be charged.

        Returns:
            Remaining capacity in kWh
        """
        return self.battery_capacity * (1.0 - self.soc)

    def __repr__(self):
        """String representation of the EV."""
        return f"EV(id={self.ev_id}, capacity={self.battery_capacity}kWh, soc={self.soc:.2%})"
