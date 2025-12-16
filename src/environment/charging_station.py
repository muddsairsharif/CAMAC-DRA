"""Charging Station module for CAMAC-DRA.

This module defines the ChargingStation class and related functionality for managing
charging operations and power distribution.
"""

from typing import Optional, List


class ChargingStation:
    """Represents a charging station with one or more charging points."""

    def __init__(self, station_id, num_chargers, max_power_kw):
        """Initialize a charging station.

        Args:
            station_id: Unique identifier for the station
            num_chargers: Number of charging points at the station
            max_power_kw: Maximum power capacity of the station in kW
        """
        self.station_id = station_id
        self.num_chargers = num_chargers
        self.max_power_kw = max_power_kw
        self.chargers = {i: None for i in range(num_chargers)}
        self.current_power_usage = 0.0

    def add_ev(self, charger_id, ev) -> bool:
        """Add an EV to a charger.

        Args:
            charger_id: ID of the charger
            ev: EV object to add

        Returns:
            True if EV was added successfully, False otherwise
        """
        if charger_id < 0 or charger_id >= self.num_chargers:
            return False
        if self.chargers[charger_id] is not None:
            return False
        self.chargers[charger_id] = ev
        ev.is_charging = True
        return True

    def remove_ev(self, charger_id) -> Optional[object]:
        """Remove an EV from a charger.

        Args:
            charger_id: ID of the charger

        Returns:
            The removed EV object if successful, None otherwise
        """
        if charger_id < 0 or charger_id >= self.num_chargers:
            return None
        ev = self.chargers[charger_id]
        if ev is not None:
            ev.is_charging = False
            self.chargers[charger_id] = None
        return ev

    def set_charger_power(self, charger_id, power_kw) -> bool:
        """Set the charging power for a specific charger.

        Args:
            charger_id: ID of the charger
            power_kw: Power to set in kW

        Returns:
            True if power was set successfully, False otherwise
        """
        if charger_id < 0 or charger_id >= self.num_chargers:
            return False
        ev = self.chargers[charger_id]
        if ev is None:
            return False
        ev.current_charging_power = power_kw
        return True

    def get_available_chargers(self) -> int:
        """Get the number of available (unoccupied) chargers.

        Returns:
            Number of available chargers
        """
        return sum(1 for ev in self.chargers.values() if ev is None)

    def get_total_power_usage(self) -> float:
        """Get the total power usage of the station.

        Returns:
            Total power usage in kW
        """
        return sum(ev.current_charging_power for ev in self.chargers.values() if ev is not None)

    def __repr__(self):
        """String representation of the charging station."""
        occupied = self.num_chargers - self.get_available_chargers()
        return f"ChargingStation(id={self.station_id}, chargers={occupied}/{self.num_chargers}, max_power={self.max_power_kw}kW)"
