"""
TCVN 5574:2018 Serviceability Limit State (SLS) Module

Implements:
- Crack width check (Section 8.2.1)
- Deflection check (Section 8.2.2)

Reference: TCVN 5574:2018 Section 8.2
"""

import numpy as np
from typing import Dict, Any
from pydantic import BaseModel, field_validator

from .tcvn_beam import BeamSection
from .tcvn_setup import DesignContext, create_design_context_from_streamlit


def check_crack_width(
    M_ser: float,  # Service moment (unfactored, kN·m)
    section: BeamSection,
    context: DesignContext,
    As: float,  # Tension steel area (mm²)
    bar_diameter: float = 12,  # Assumed bar diameter (mm)
    environment: str = 'normal'  # 'normal' or 'aggressive'
) -> Dict[str, Any]:
    """
    TCVN 5574:2018 Section 8.2.1 crack width check.
    
    Formula (TCVN 8.2.1):
    a_cr = φ₁ · φ₂ · φ₃ · (σs / Es) · 20 · (3.5 - 100μ) · √(d / d_eq)
    
    Where:
        φ₁ = Long-term load factor (1.0 for short-term, 1.4 for long-term)
        φ₂ = Profile factor (1.0 for periodic profile, 0.5 for smooth)
        φ₃ = Load pattern factor (1.0 typically)
        σs = Steel stress at service load (MPa)
        Es = Steel modulus of elasticity (MPa)
        μ = Reinforcement ratio = As / (b·h₀)
        d = Bar diameter (mm)
        d_eq = Equivalent bar diameter (mm)
    
    Args:
        M_ser: Service moment (unfactored, kN·m)
        section: BeamSection instance
        context: DesignContext with material properties
        As: Tension reinforcement area (mm²)
        bar_diameter: Actual bar diameter (mm), default 12
        environment: 'normal' or 'aggressive' (affects allowable width)
    
    Returns:
        {
            'a_cr': float,  # Calculated crack width (mm)
            'a_limit': float,  # Allowable crack width (mm)
            'sigma_s': float,  # Steel stress (MPa)
            'mu': float,  # Reinforcement ratio
            'UR': float,  # Utilization ratio
            'status': str,  # 'PASS' or 'FAIL'
            'details': str  # Calculation narrative
        }
    """
    Rs = context.steel.Rs
    Es = context.steel.Es
    b = section.b
    h0 = section.h0
    
    # Calculate stress in steel at service load
    # Assuming internal lever arm z ≈ 0.9·h₀
    M_ser_Nmm = M_ser * 1e6  # Convert kN·m to N·mm
    z = 0.9 * h0  # Approximate lever arm (mm)
    sigma_s = M_ser_Nmm / (As * z)  # MPa
    
    # Factors per TCVN 5574:2018 Table 8.3
    phi_1 = 1.0  # Long-term load factor (use 1.4 for sustained loads)
    phi_2 = 1.0  # Profile factor (1.0 for periodic/ribbed bars, 0.5 for smooth)
    phi_3 = 1.0  # Load pattern factor (1.0 for typical loading)
    
    # Reinforcement ratio
    mu = As / (b * h0)
    
    # Effective bar diameter
    d_eq = bar_diameter  # For single bar size
    d = bar_diameter  # Actual bar diameter
    
    # Calculate crack width per TCVN formula
    # a_cr = φ₁·φ₂·φ₃·(σs/Es)·20·(3.5 - 100μ)·√(d/d_eq)
    term1 = phi_1 * phi_2 * phi_3
    term2 = sigma_s / Es
    term3 = 20 * (3.5 - 100 * mu)
    term4 = np.sqrt(d / d_eq)
    
    a_cr = term1 * term2 * term3 * term4  # mm
    
    # Allowable crack width per TCVN Table 8.2
    if environment == 'normal':
        a_limit = 0.4  # mm for normal exposure
    elif environment == 'aggressive':
        a_limit = 0.3  # mm for aggressive environment
    else:
        a_limit = 0.3  # Conservative default
    
    # Utilization ratio
    UR = a_cr / a_limit
    status = 'PASS' if UR <= 1.0 else 'FAIL'
    
    # Detailed calculation narrative
    details = f"""
Crack Width Check (TCVN 5574:2018 Section 8.2.1)
{'='*55}

Section: b×h = {b:.0f}×{section.h:.0f} mm, h₀ = {h0:.0f} mm
Steel: As = {As:.0f} mm², φ = {bar_diameter:.0f} mm
Environment: {environment.upper()}

Service Loads:
  Service moment M_ser = {M_ser:.2f} kN·m
  Internal lever arm z = {z:.1f} mm

Steel Stress:
  σs = M_ser / (As·z) = {sigma_s:.1f} MPa
  
Reinforcement Ratio:
  μ = As / (b·h₀) = {mu:.4f}

Factors:
  φ₁ (long-term) = {phi_1:.1f}
  φ₂ (profile) = {phi_2:.1f}
  φ₃ (load pattern) = {phi_3:.1f}

Crack Width Calculation:
  a_cr = φ₁·φ₂·φ₃·(σs/Es)·20·(3.5-100μ)·√(d/d_eq)
  a_cr = {term1:.1f} × {term2:.6f} × {term3:.2f} × {term4:.2f}
  a_cr = {a_cr:.3f} mm

Allowable:
  a_limit = {a_limit:.1f} mm ({environment} environment)

Result:
  UR = a_cr / a_limit = {UR:.3f}
  Status: {status}
"""
    
    return {
        'a_cr': a_cr,
        'a_limit': a_limit,
        'sigma_s': sigma_s,
        'mu': mu,
        'UR': UR,
        'status': status,
        'details': details
    }


def check_deflection(
    L: float,  # Span length (mm)
    M_ser: float,  # Service moment (kN·m)
    section: BeamSection,
    context: DesignContext,
    As: float,  # Tension steel area (mm²)
    load_duration: str = 'long_term'  # 'short_term' or 'long_term'
) -> Dict[str, Any]:
    """
    TCVN 5574:2018 Section 8.2.2 deflection check.
    
    Method:
    1. Calculate cracking moment M_cr
    2. Determine effective moment of inertia Ie (Branson equation)
    3. Calculate curvature κ = M / (E·I)
    4. Integrate for deflection (simplified for uniform loading)
    5. Apply long-term factor for creep
    6. Compare with allowable limit (L/250)
    
    Args:
        L: Span length (mm)
        M_ser: Service moment (kN·m)
        section: BeamSection instance
        context: DesignContext with material properties
        As: Tension reinforcement area (mm²)
        load_duration: 'short_term' or 'long_term'
    
    Returns:
        {
            'delta': float,  # Calculated deflection (mm)
            'delta_limit': float,  # Allowable deflection (mm)
            'Ie': float,  # Effective moment of inertia (mm⁴)
            'M_cr': float,  # Cracking moment (kN·m)
            'UR': float,
            'status': str,
            'details': str
        }
    """
    Eb = context.concrete.Eb  # Modulus of elasticity (MPa)
    Rbt = context.concrete.Rbt  # Tensile strength (MPa)
    b = section.b
    h = section.h
    h0 = section.h0
    
    # Gross moment of inertia (uncracked section)
    Ig = (b * h**3) / 12  # mm⁴
    
    # Modular ratio
    n = context.steel.Es / Eb  # Dimensionless
    
    # Cracked section analysis (transformed section method)
    # Neutral axis depth for cracked section
    # Equilibrium: b·c²/2 = n·As·(h₀ - c)
    # Rearranging: c² + 2(n·As/b)·c - 2(n·As·h₀/b) = 0
    
    rho = As / (b * h0)  # Reinforcement ratio
    k = np.sqrt((n * rho)**2 + 2 * n * rho) - n * rho  # Neutral axis ratio
    c = k * h0  # Neutral axis depth (mm)
    
    # Cracked moment of inertia
    Icr = (b * c**3) / 3 + n * As * (h0 - c)**2  # mm⁴
    
    # Cracking moment (TCVN 8.2.2)
    # M_cr = (fr · Ig) / yt, where fr = Rbt, yt = h/2
    yt = h / 2  # Distance to extreme tension fiber (mm)
    M_cr = (Rbt * Ig) / yt  # N·mm
    M_cr_kNm = M_cr / 1e6  # Convert to kN·m
    
    # Effective moment of inertia (Branson equation)
    # If M_ser < M_cr, section is uncracked
    # If M_ser >= M_cr, use weighted average
    M_ser_Nmm = M_ser * 1e6  # N·mm
    
    if M_ser_Nmm < M_cr:
        Ie = Ig  # Uncracked
        section_state = "uncracked"
    else:
        # Branson equation: Ie = Icr + (Ig - Icr)·(M_cr/M_ser)³
        ratio = M_cr / M_ser_Nmm
        Ie = Icr + (Ig - Icr) * ratio**3  # mm⁴
        section_state = "cracked"
    
    # Curvature at service load
    kappa = M_ser_Nmm / (Eb * Ie)  # 1/mm
    
    # Deflection for simply-supported beam with uniform moment
    # δ = κ · L² / 8 (simplified, exact requires load distribution)
    delta_immediate = kappa * L**2 / 8  # mm
    
    # Apply long-term factor for creep and shrinkage
    if load_duration == 'long_term':
        xi = 2.0  # Creep coefficient (TCVN default)
        delta = delta_immediate * (1 + xi)  # Total long-term deflection
    else:
        xi = 0
        delta = delta_immediate  # Short-term only
    
    # Allowable deflection per TCVN Table 8.4
    # L/250 for typical structures (total deflection)
    # L/350 for supporting brittle finishes
    delta_limit = L / 250  # mm
    
    # Utilization ratio
    UR = delta / delta_limit
    status = 'PASS' if UR <= 1.0 else 'FAIL'
    
    # Detailed calculation narrative
    details = f"""
Deflection Check (TCVN 5574:2018 Section 8.2.2)
{'='*55}

Section: b×h = {b:.0f}×{h:.0f} mm, h₀ = {h0:.0f} mm
Span: L = {L:.0f} mm = {L/1000:.2f} m
Steel: As = {As:.0f} mm²

Materials:
  Concrete Eb = {Eb:.0f} MPa, Rbt = {Rbt:.2f} MPa
  Modular ratio n = Es/Eb = {n:.1f}

Section Properties:
  Gross inertia Ig = {Ig:.0f} mm⁴
  Neutral axis depth c = {c:.1f} mm
  Cracked inertia Icr = {Icr:.0f} mm⁴

Cracking Analysis:
  Cracking moment M_cr = {M_cr_kNm:.2f} kN·m
  Service moment M_ser = {M_ser:.2f} kN·m
  Section state: {section_state.upper()}

Effective Properties:
  Effective inertia Ie = {Ie:.0f} mm⁴
  Curvature κ = {kappa:.6e} mm⁻¹

Deflection Calculation:
  Immediate deflection δ₀ = {delta_immediate:.2f} mm
  Load duration: {load_duration}
  Creep factor ξ = {xi:.1f}
  Total deflection δ = {delta:.2f} mm

Allowable:
  δ_limit = L/250 = {delta_limit:.2f} mm
  
Result:
  UR = δ / δ_limit = {UR:.3f}
  Status: {status}
"""
    
    return {
        'delta': delta,
        'delta_limit': delta_limit,
        'delta_immediate': delta_immediate,
        'Ie': Ie,
        'Ig': Ig,
        'Icr': Icr,
        'M_cr': M_cr_kNm,
        'section_state': section_state,
        'UR': UR,
        'status': status,
        'details': details
    }


def run_sls_check_from_ui(
    beam_id: int,
    b: float, h: float, cover: float,
    As: float,  # Tension steel
    bar_diameter: float,
    concrete_name: str,
    steel_name: str,
    M_ser: float,  # Service moment (kN·m)
    L: float,  # Span length (m)
    environment: str = 'normal',
    load_duration: str = 'long_term'
) -> Dict[str, Any]:
    """
    Wrapper function for Streamlit UI integration.
    
    Performs complete serviceability checks:
    1. Crack width check
    2. Deflection check
    
    Args:
        beam_id: Beam element ID
        b, h, cover: Section dimensions (mm)
        As: Tension reinforcement area (mm²)
        bar_diameter: Reinforcing bar diameter (mm)
        concrete_name: Concrete grade (e.g., "B25")
        steel_name: Steel grade (e.g., "CB400V")
        M_ser: Service moment (kN·m, unfactored)
        L: Span length (m)
        environment: 'normal' or 'aggressive'
        load_duration: 'short_term' or 'long_term'
    
    Returns:
        {
            'beam_id': int,
            'crack_width': dict,
            'deflection': dict
        }
    """
    # Create section and context
    section = BeamSection(b=b, h=h, cover=cover)
    context = create_design_context_from_streamlit(
        concrete_name,
        steel_name,
        cover
    )
    
    # Convert span to mm
    L_mm = L * 1000
    
    # Run crack width check
    crack_result = check_crack_width(
        M_ser=M_ser,
        section=section,
        context=context,
        As=As,
        bar_diameter=bar_diameter,
        environment=environment
    )
    
    # Run deflection check
    deflection_result = check_deflection(
        L=L_mm,
        M_ser=M_ser,
        section=section,
        context=context,
        As=As,
        load_duration=load_duration
    )
    
    return {
        'beam_id': beam_id,
        'crack_width': crack_result,
        'deflection': deflection_result
    }
