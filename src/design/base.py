"""
Base abstract class for design code implementations.

This module provides the Strategy Pattern interface for multi-standard support.
All design codes (TCVN, ACI, Eurocode) inherit from DesignCode.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class DesignCode(ABC):
    """Abstract base class for design code checkers."""
    
    @property
    @abstractmethod
    def code_name(self) -> str:
        """Return code identifier (e.g., 'ACI 318-25', 'TCVN 5574:2018')."""
        pass
    
    @property
    @abstractmethod
    def code_units(self) -> str:
        """Return primary unit system ('SI' or 'Imperial')."""
        pass
    
    @abstractmethod
    def check_beam_flexure(self, **kwargs) -> Dict[str, Any]:
        """
        Check beam flexural capacity.
        
        Returns:
            dict with keys: 'UR' (utilization ratio), 'status' ('PASS'/'FAIL'),
            'Mn' (nominal capacity), 'details' (calculation narrative)
        """
        pass
    
    @abstractmethod
    def check_beam_shear(self, **kwargs) -> Dict[str, Any]:
        """
        Check beam shear capacity.
        
        Returns:
            dict with keys: 'UR', 'status', 'Vn', 'details'
        """
        pass
    
    @abstractmethod
    def check_column(self, **kwargs) -> Dict[str, Any]:
        """
        Check column axial-moment interaction.
        
        Returns:
            dict with keys: 'UR', 'status', 'figure' (Plotly), 'details'
        """
        pass
    
    @abstractmethod
    def check_serviceability(self, **kwargs) -> Dict[str, Any]:
        """
        Check serviceability (deflection and crack width).
        
        Returns:
            dict with keys: 'deflection' (dict), 'crack_width' (dict), 'status'
        """
        pass
