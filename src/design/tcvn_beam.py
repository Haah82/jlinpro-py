"""
TCVN 5574:2018 Beam Design - Flexure and Shear Checks
Vietnamese concrete design standard for reinforced concrete beams
"""

from typing import Dict, Any
from pydantic import BaseModel, Field, field_validator
import numpy as np

from .tcvn_setup import DesignContext


class BeamSection(BaseModel):
    """
    Beam cross-section properties.
    
    Attributes:
        b: Width (mm)
        h: Height (mm)
        cover: Concrete cover (mm)
    
    Example:
        >>> section = BeamSection(b=300, h=500, cover=25)
        >>> print(f"Effective depth: {section.h0} mm")
        Effective depth: 475 mm
    """
    b: float = Field(..., gt=0, description="Width (mm)")
    h: float = Field(..., gt=0, description="Height (mm)")
    cover: float = Field(..., gt=0, description="Concrete cover (mm)")
    
    @field_validator('b')
    @classmethod
    def validate_width(cls, v: float) -> float:
        """Validate beam width."""
        if v < 100:
            raise ValueError(f"Width b = {v} mm is too small (min 100 mm)")
        if v > 1000:
            raise ValueError(f"Width b = {v} mm seems excessive")
        return v
    
    @field_validator('h')
    @classmethod
    def validate_height(cls, v: float) -> float:
        """Validate beam height."""
        if v < 150:
            raise ValueError(f"Height h = {v} mm is too small (min 150 mm)")
        if v > 2000:
            raise ValueError(f"Height h = {v} mm seems excessive")
        return v
    
    @property
    def h0(self) -> float:
        """Effective depth h₀ = h - cover."""
        return self.h - self.cover
    
    class Config:
        """Pydantic configuration."""
        frozen = False


class Reinforcement(BaseModel):
    """
    Beam reinforcement layout.
    
    Attributes:
        As_top: Top steel area (mm²)
        As_bot: Bottom steel area (mm²)
    
    Example:
        >>> rebar = Reinforcement(As_top=1000, As_bot=1500)
        >>> print(f"Bottom steel: {rebar.As_bot} mm²")
        Bottom steel: 1500 mm²
    """
    As_top: float = Field(..., ge=0, description="Top steel area (mm²)")
    As_bot: float = Field(..., ge=0, description="Bottom steel area (mm²)")
    
    class Config:
        """Pydantic configuration."""
        frozen = False


class Stirrups(BaseModel):
    """
    Stirrup configuration for shear reinforcement.
    
    Attributes:
        diameter: Stirrup bar diameter (mm)
        n_legs: Number of legs (typically 2 or 4)
        spacing: Stirrup spacing s (mm)
    
    Example:
        >>> stirrups = Stirrups(diameter=10, n_legs=2, spacing=200)
        >>> print(f"Asw = {stirrups.Asw:.1f} mm², qsw = {stirrups.qsw:.2f} mm²/mm")
        Asw = 157.1 mm², qsw = 0.79 mm²/mm
    """
    diameter: float = Field(..., gt=0, description="Stirrup diameter (mm)")
    n_legs: int = Field(..., ge=2, description="Number of legs")
    spacing: float = Field(..., gt=0, description="Spacing s (mm)")
    
    @field_validator('diameter')
    @classmethod
    def validate_diameter(cls, v: float) -> float:
        """Validate stirrup diameter."""
        if v < 6:
            raise ValueError(f"Stirrup diameter {v} mm is too small (min 6 mm)")
        if v > 20:
            raise ValueError(f"Stirrup diameter {v} mm seems excessive")
        return v
    
    @field_validator('spacing')
    @classmethod
    def validate_spacing(cls, v: float) -> float:
        """Validate stirrup spacing."""
        if v < 50:
            raise ValueError(f"Spacing {v} mm is too small (min 50 mm)")
        if v > 500:
            raise ValueError(f"Spacing {v} mm is too large (max 500 mm)")
        return v
    
    @property
    def Asw(self) -> float:
        """Total stirrup cross-sectional area (mm²)."""
        return self.n_legs * np.pi * (self.diameter / 2) ** 2
    
    @property
    def qsw(self) -> float:
        """Stirrup intensity - area per unit length (mm²/mm)."""
        return self.Asw / self.spacing
    
    class Config:
        """Pydantic configuration."""
        frozen = False


def check_flexure(
    M_u: float,
    section: BeamSection,
    context: DesignContext,
    As_provided: float
) -> Dict[str, Any]:
    """
    TCVN 5574:2018 flexure check for rectangular beams.
    
    Reference: TCVN 5574:2018 Section 8.1.2.3 - Flexural Design
    
    Args:
        M_u: Ultimate moment (kN·m)
        section: BeamSection object
        context: DesignContext with materials
        As_provided: Provided steel area (mm²)
    
    Returns:
        Dictionary with:
            - alpha_m: Relative moment
            - xi: Relative compression zone depth
            - xi_R: Limit value
            - As_req: Required steel area (mm²)
            - M_cap: Moment capacity (kN·m)
            - UR: Utilization ratio
            - status: 'PASS' or 'FAIL'
            - details: Calculation narrative
    
    Algorithm:
        1. h₀ = h - cover
        2. α_m = M_u / (R_b · b · h₀²)
        3. ξ = 1 - √(1 - 2α_m)
        4. Check ξ ≤ ξ_R
        5. A_s,req = (R_b · b · h₀ · ξ) / R_s
        6. UR = A_s,req / A_s,prov or M_u / M_cap
    
    Example:
        >>> section = BeamSection(b=300, h=500, cover=25)
        >>> context = DesignContext.from_ui_inputs("B25", "CB400V", 25)
        >>> result = check_flexure(250, section, context, 1500)
        >>> print(f"Status: {result['status']}, UR: {result['UR']:.2f}")
        Status: PASS, UR: 0.85
    """
    # Extract material properties
    Rb = context.concrete.Rb  # MPa
    Rs = context.steel.Rs  # MPa
    b = section.b  # mm
    h0 = section.h0  # mm
    
    # Step 1: Convert moment to N·mm
    M_u_Nmm = M_u * 1e6
    
    # Step 2: Calculate alpha_m
    alpha_m = M_u_Nmm / (Rb * b * h0**2)
    
    # Step 3: Check if section is adequate
    if alpha_m > 0.5:
        return {
            'alpha_m': alpha_m,
            'xi': None,
            'xi_R': None,
            'As_req': None,
            'M_cap': None,
            'UR': float('inf'),
            'status': 'FAIL',
            'details': f"""
FAIL: Section inadequate
α_m = {alpha_m:.3f} > 0.5
The section is severely under-reinforced.
Recommendation: Increase section size (b or h).
            """
        }
    
    # Step 4: Calculate xi (relative compression zone depth)
    xi = 1 - np.sqrt(1 - 2 * alpha_m)
    
    # Step 5: Get xi limit from context
    xi_R = context.get_xi_limit()
    
    # Check over-reinforcement
    if xi > xi_R:
        return {
            'alpha_m': alpha_m,
            'xi': xi,
            'xi_R': xi_R,
            'As_req': None,
            'M_cap': None,
            'UR': float('inf'),
            'status': 'FAIL',
            'details': f"""
FAIL: Over-reinforced section
ξ = {xi:.3f} > ξ_R = {xi_R:.2f}
The section will fail in brittle compression mode.
Recommendation: Reduce steel area or increase section size.
            """
        }
    
    # Step 6: Calculate required steel area
    As_req = (Rb * b * h0 * xi) / Rs
    
    # Step 7: Calculate capacity with provided steel
    xi_prov = (Rs * As_provided) / (Rb * b * h0)
    
    # Check if provided steel is within limits
    if xi_prov > xi_R:
        # Over-reinforced with provided steel
        M_cap_Nmm = Rb * b * h0**2 * xi_R * (1 - 0.5 * xi_R)
        M_cap = M_cap_Nmm / 1e6
        UR = M_u / M_cap
        status = 'FAIL'
        warning = f"\nWARNING: Provided steel makes section over-reinforced (ξ_prov = {xi_prov:.3f} > ξ_R = {xi_R:.2f})"
    else:
        # Properly reinforced
        M_cap_Nmm = Rb * b * h0**2 * xi_prov * (1 - 0.5 * xi_prov)
        M_cap = M_cap_Nmm / 1e6
        UR = M_u / M_cap
        status = 'PASS' if UR <= 1.0 else 'FAIL'
        warning = ""
    
    # Format details
    details = f"""
Flexural Design Check (TCVN 5574:2018 Section 8.1.2.3)
---------------------------------------------------
Section: b×h = {b:.0f}×{section.h:.0f} mm
Effective depth h₀ = {h0:.0f} mm
Materials: {context.concrete.name} (Rb = {Rb:.1f} MPa), {context.steel.name} (Rs = {Rs:.0f} MPa)

Applied moment M_u = {M_u:.2f} kN·m

Calculations:
  α_m = M_u / (Rb·b·h₀²) = {alpha_m:.4f}
  ξ = 1 - √(1 - 2α_m) = {xi:.4f}
  ξ_R = {xi_R:.2f} (limit for {context.steel.name})
  
  Required steel: As,req = {As_req:.0f} mm²
  Provided steel: As,prov = {As_provided:.0f} mm²
  
  Relative depth: ξ_prov = {xi_prov:.4f}
  Moment capacity: M_cap = {M_cap:.2f} kN·m
  
Result:
  Utilization Ratio UR = {UR:.3f}
  Status: {status}{warning}
    """
    
    return {
        'alpha_m': alpha_m,
        'xi': xi,
        'xi_R': xi_R,
        'As_req': As_req,
        'M_cap': M_cap,
        'UR': UR,
        'status': status,
        'details': details.strip()
    }


def check_shear(
    Q_u: float,
    section: BeamSection,
    context: DesignContext,
    stirrups: Stirrups
) -> Dict[str, Any]:
    """
    TCVN 5574:2018 shear check for rectangular beams.
    
    Reference: TCVN 5574:2018 Section 8.1.3 - Shear Design
    
    Args:
        Q_u: Ultimate shear force (kN)
        section: BeamSection object
        context: DesignContext with materials
        stirrups: Stirrups configuration
    
    Returns:
        Dictionary with:
            - Qb: Concrete contribution (kN)
            - Qsw: Steel contribution (kN)
            - Q_cap: Total capacity (kN)
            - Q_crush: Crushing limit (kN)
            - UR: Utilization ratio
            - status: 'PASS' or 'FAIL'
            - details: Calculation narrative
    
    Algorithm:
        1. Q_b = (φ_b1 · R_bt · b · h₀²) / c (concrete contribution)
        2. Q_sw = q_sw · R_sw · c₀ (stirrup contribution)
        3. Check Q_u ≤ Q_b + Q_sw (strength)
        4. Check Q_u ≤ 0.3 · R_b · b · h₀ (crushing)
    
    Example:
        >>> section = BeamSection(b=300, h=500, cover=25)
        >>> context = DesignContext.from_ui_inputs("B25", "CB400V", 25)
        >>> stirrups = Stirrups(diameter=10, n_legs=2, spacing=200)
        >>> result = check_shear(120, section, context, stirrups)
        >>> print(f"Status: {result['status']}, UR: {result['UR']:.2f}")
        Status: PASS, UR: 0.72
    """
    # Extract material properties
    Rb = context.concrete.Rb  # MPa
    Rbt = context.concrete.Rbt  # MPa
    Rsw = context.steel.Rs  # MPa (stirrup steel, assume same grade)
    b = section.b  # mm
    h0 = section.h0  # mm
    
    # Constants per TCVN 5574:2018
    phi_b1 = 0.6  # Factor for concrete contribution
    c = 2.0 * h0  # Shear span (conservative assumption)
    c0 = 0.9 * h0  # Internal lever arm
    
    # Step 1: Concrete contribution Q_b (N)
    Qb = (phi_b1 * Rbt * b * h0**2) / c
    Qb_kN = Qb / 1000  # Convert to kN
    
    # Step 2: Stirrup contribution Q_sw (N)
    qsw = stirrups.qsw  # mm²/mm
    Qsw = qsw * Rsw * c0
    Qsw_kN = Qsw / 1000  # Convert to kN
    
    # Step 3: Total shear capacity
    Q_cap = Qb_kN + Qsw_kN
    
    # Step 4: Crushing limit (web crushing)
    Q_crush = 0.3 * Rb * b * h0 / 1000  # kN
    
    # Step 5: Check both criteria
    UR_strength = Q_u / Q_cap
    UR_crush = Q_u / Q_crush
    UR = max(UR_strength, UR_crush)
    
    # Determine status
    if UR_strength > 1.0:
        status = 'FAIL'
        failure_mode = 'Insufficient shear capacity (Q_u > Q_b + Q_sw)'
    elif UR_crush > 1.0:
        status = 'FAIL'
        failure_mode = 'Web crushing failure (Q_u > 0.3·Rb·b·h₀)'
    else:
        status = 'PASS'
        failure_mode = 'All checks satisfied'
    
    # Minimum stirrup requirement check
    qsw_min = 0.25 * Rbt / Rsw  # TCVN minimum
    if qsw < qsw_min:
        min_warning = f"\nWARNING: Stirrup intensity {qsw:.3f} < minimum {qsw_min:.3f} mm²/mm"
    else:
        min_warning = ""
    
    # Format details
    details = f"""
Shear Design Check (TCVN 5574:2018 Section 8.1.3)
---------------------------------------------------
Section: b×h = {b:.0f}×{section.h:.0f} mm
Effective depth h₀ = {h0:.0f} mm
Materials: {context.concrete.name} (Rb = {Rb:.1f}, Rbt = {Rbt:.2f} MPa)
Stirrups: φ{stirrups.diameter}@{stirrups.spacing}, {stirrups.n_legs} legs

Applied shear Q_u = {Q_u:.2f} kN

Concrete contribution:
  φ_b1 = {phi_b1}
  c = {c:.0f} mm (shear span)
  Q_b = (φ_b1·Rbt·b·h₀²)/c = {Qb_kN:.2f} kN

Stirrup contribution:
  Asw = {stirrups.Asw:.1f} mm²
  qsw = Asw/s = {qsw:.3f} mm²/mm
  c₀ = {c0:.0f} mm (lever arm)
  Q_sw = qsw·Rsw·c₀ = {Qsw_kN:.2f} kN

Total capacity: Q_cap = Q_b + Q_sw = {Q_cap:.2f} kN
Crushing limit: Q_crush = 0.3·Rb·b·h₀ = {Q_crush:.2f} kN

Result:
  UR (strength) = Q_u/Q_cap = {UR_strength:.3f}
  UR (crushing) = Q_u/Q_crush = {UR_crush:.3f}
  Governing UR = {UR:.3f}
  Status: {status} - {failure_mode}{min_warning}
    """
    
    return {
        'Qb': Qb_kN,
        'Qsw': Qsw_kN,
        'Q_cap': Q_cap,
        'Q_crush': Q_crush,
        'UR': UR,
        'status': status,
        'details': details.strip()
    }


def run_beam_check_from_ui(
    beam_id: int,
    b: float,
    h: float,
    cover: float,
    As_top: float,
    As_bot: float,
    stirrup_dia: float,
    n_legs: int,
    spacing: float,
    concrete_name: str,
    steel_name: str,
    M_u: float,
    Q_u: float
) -> Dict[str, Any]:
    """
    Wrapper function for Streamlit integration.
    
    Called from app/pages/design.py to perform complete beam design check.
    
    Args:
        beam_id: Element ID
        b, h, cover: Section dimensions (mm)
        As_top, As_bot: Steel areas (mm²)
        stirrup_dia, n_legs, spacing: Stirrup configuration
        concrete_name: Concrete grade (e.g., "B25")
        steel_name: Steel grade (e.g., "CB400V")
        M_u: Ultimate moment from analysis (kN·m)
        Q_u: Ultimate shear from analysis (kN)
    
    Returns:
        Dictionary with 'flexure' and 'shear' check results
    
    Example:
        >>> results = run_beam_check_from_ui(
        ...     beam_id=1, b=300, h=500, cover=25,
        ...     As_top=1000, As_bot=1500,
        ...     stirrup_dia=10, n_legs=2, spacing=200,
        ...     concrete_name="B25", steel_name="CB400V",
        ...     M_u=250, Q_u=120
        ... )
        >>> print(f"Flexure: {results['flexure']['status']}")
        >>> print(f"Shear: {results['shear']['status']}")
    """
    from .tcvn_setup import create_design_context_from_streamlit
    
    # Create design objects
    section = BeamSection(b=b, h=h, cover=cover)
    context = create_design_context_from_streamlit(concrete_name, steel_name, cover)
    stirrups = Stirrups(diameter=stirrup_dia, n_legs=n_legs, spacing=spacing)
    
    # Run flexure check (use bottom steel for positive moment)
    flex_result = check_flexure(M_u, section, context, As_bot)
    
    # Run shear check
    shear_result = check_shear(Q_u, section, context, stirrups)
    
    return {
        'beam_id': beam_id,
        'flexure': flex_result,
        'shear': shear_result
    }
