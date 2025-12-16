"""
EV (Electric Vehicle) class module.

This module contains the EV class which represents an electric vehicle
with properties like battery capacity, charge level, and charging behavior.
"""


class EV:
    """
    Represents an Electric Vehicle with battery and charging capabilities.
    
    Attributes:
        ev_id: Unique identifier for the vehicle
        battery_capacity: Maximum battery capacity in kWh
        current_charge: Current battery charge level in kWh
        charging_power: Power rating for charging in kW
    """
    
    def __init__(self, ev_id, battery_capacity, charging_power=11.0):
        """
        Initialize an EV instance.
        
        Args:
            ev_id: Unique identifier for the vehicle
            battery_capacity: Maximum battery capacity in kWh
            charging_power: Power rating for charging in kW (default: 11.0)
        """
        self.ev_id = ev_id
        self.battery_capacity = battery_capacity
        self.current_charge = 0.0
        self.charging_power = charging_power
    
    def get_soc(self):
        """
        Get the State of Charge (SOC) as a percentage.
        
        Returns:
            float: SOC percentage (0-100)
        """
        return (self.current_charge / self.battery_capacity) * 100
    
    def charge(self, amount):
        """
        Charge the vehicle battery.
        
        Args:
            amount: Amount of energy to charge in kWh
        """
        self.current_charge = min(self.current_charge + amount, self.battery_capacity)
    
    def discharge(self, amount):
        """
        Discharge the vehicle battery.
        
        Args:
            amount: Amount of energy to discharge in kWh
        """
        self.current_charge = max(self.current_charge - amount, 0.0)
