"""
Design code checking modules for PyLinPro
Supports TCVN 5574:2018, ACI 318-25, and Eurocode 2
"""

from .tcvn_setup import (
    ConcreteGrade,
    SteelGrade,
    DesignContext,
    MaterialLoader,
    create_design_context_from_streamlit
)
from .base import DesignCode
from .aci318 import ACI318Code, ConcreteACI, SteelACI, MaterialLoaderACI
from .eurocode2 import Eurocode2Code, ConcreteEC2, SteelEC2, MaterialLoaderEC2

# ============================================================================
# CODE REGISTRY (Factory Pattern)
# ============================================================================

CODE_REGISTRY = {
    "ACI 318-25 (USA)": ACI318Code(),
    "Eurocode 2 EN 1992-1-1:2004 (Europe)": Eurocode2Code(),
}


def get_design_code(code_name: str) -> DesignCode:
    """
    Factory method to get design code checker.
    
    Args:
        code_name: Design code identifier (e.g., "ACI 318-25 (USA)")
    
    Returns:
        DesignCode implementation instance
    
    Raises:
        ValueError: If code_name not found in registry
    """
    if code_name not in CODE_REGISTRY:
        available = ", ".join(CODE_REGISTRY.keys())
        raise ValueError(f"Unknown code: {code_name}. Available: {available}")
    return CODE_REGISTRY[code_name]


__all__ = [
    'ConcreteGrade',
    'SteelGrade',
    'DesignContext',
    'MaterialLoader',
    'create_design_context_from_streamlit',
    'DesignCode',
    'ACI318Code',
    'ConcreteACI',
    'SteelACI',
    'MaterialLoaderACI',
    'Eurocode2Code',
    'ConcreteEC2',
    'SteelEC2',
    'MaterialLoaderEC2',
    'CODE_REGISTRY',
    'get_design_code',
]
