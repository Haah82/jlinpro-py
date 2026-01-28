"""
Eurocode 2 (EN 1992-1-1:2004) Design Code Implementation.

Reference: EN 1992-1-1:2004 Eurocode 2: Design of concrete structures
Part 1-1: General rules and rules for buildings

Units: SI (MPa, mm, kN, kN·m)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, field_validator
import plotly.graph_objects as go
import yaml
from pathlib import Path

from .base import DesignCode


# ============================================================================
# MATERIAL MODELS (Eurocode 2)
# ============================================================================

class ConcreteEC2(BaseModel):
    """
    Eurocode 2 concrete material properties.
    
    Reference: EN 1992-1-1 Table 3.1
    
    Strength class notation: C20/25, C30/37, etc.
    Format: C(fck_cylinder)/(fck_cube)
    """
    strength_class: str  # e.g., "C30/37"
    fck: float  # Characteristic cylinder strength (MPa)
    fck_cube: float  # Characteristic cube strength (MPa)
    alpha_cc: float = 1.0  # Long-term strength reduction (default 1.0 per 3.1.6)
    gamma_c: float = 1.5  # Partial safety factor for concrete
    
    class Config:
        frozen = False
    
    @field_validator('fck')
    @classmethod
    def validate_fck(cls, v):
        if v < 12 or v > 90:
            raise ValueError("fck must be between 12 and 90 MPa")
        return v
    
    @property
    def fcm(self) -> float:
        """Mean compressive strength (MPa) per Table 3.1."""
        return self.fck + 8
    
    @property
    def fcd(self) -> float:
        """
        Design compressive strength (MPa).
        
        fcd = αcc · fck / γc
        
        Returns:
            Design strength in MPa
        """
        return self.alpha_cc * self.fck / self.gamma_c
    
    @property
    def fctm(self) -> float:
        """
        Mean tensile strength (MPa) per Table 3.1.
        
        fctm = 0.30 · fck^(2/3) for fck ≤ 50 MPa
        fctm = 2.12 · ln(1 + fcm/10) for fck > 50 MPa
        """
        if self.fck <= 50:
            return 0.30 * (self.fck ** (2/3))
        else:
            return 2.12 * np.log(1 + self.fcm / 10)
    
    @property
    def fctk_005(self) -> float:
        """Characteristic tensile strength (5% fractile) per 3.1.6(2)."""
        return 0.7 * self.fctm
    
    @property
    def fctk_095(self) -> float:
        """Characteristic tensile strength (95% fractile) per 3.1.6(2)."""
        return 1.3 * self.fctm
    
    @property
    def fctd(self) -> float:
        """Design tensile strength (MPa)."""
        return self.alpha_cc * self.fctk_005 / self.gamma_c
    
    @property
    def Ecm(self) -> float:
        """
        Secant modulus of elasticity (MPa) per Table 3.1.
        
        Ecm = 22 · (fcm/10)^0.3 (GPa) → converted to MPa
        """
        return 22000 * ((self.fcm / 10) ** 0.3)
    
    @property
    def epsilon_c1(self) -> float:
        """Strain at peak stress per Table 3.1."""
        if self.fck <= 50:
            return 0.0007 * (self.fcm ** 0.31)
        else:
            return 0.0007 * (self.fcm ** 0.31)
    
    @property
    def epsilon_cu1(self) -> float:
        """Ultimate compressive strain per Table 3.1."""
        if self.fck <= 50:
            return 0.0035
        else:
            return 0.0026 + 0.035 * ((90 - self.fck) / 100) ** 4
    
    @property
    def lambda_factor(self) -> float:
        """
        Stress block depth factor λ per 3.1.7(3).
        
        λ = 0.8 for fck ≤ 50 MPa
        λ = 0.8 - (fck - 50)/400 for fck > 50 MPa
        """
        if self.fck <= 50:
            return 0.8
        else:
            return 0.8 - (self.fck - 50) / 400
    
    @property
    def eta_factor(self) -> float:
        """
        Stress block intensity factor η per 3.1.7(3).
        
        η = 1.0 for fck ≤ 50 MPa
        η = 1.0 - (fck - 50)/200 for fck > 50 MPa
        """
        if self.fck <= 50:
            return 1.0
        else:
            return 1.0 - (self.fck - 50) / 200


class SteelEC2(BaseModel):
    """
    Eurocode 2 reinforcing steel properties.
    
    Reference: EN 1992-1-1 Table C.1
    """
    grade: str  # e.g., "B500B", "B500A"
    fyk: float  # Characteristic yield strength (MPa)
    ftk: float  # Characteristic tensile strength (MPa)
    epsilon_uk: float = 0.05  # Characteristic strain at maximum load
    gamma_s: float = 1.15  # Partial safety factor for steel
    Es: float = 200000  # Modulus of elasticity (MPa)
    
    class Config:
        frozen = False
    
    @field_validator('fyk')
    @classmethod
    def validate_fyk(cls, v):
        if v < 400 or v > 600:
            raise ValueError("fyk must be between 400 and 600 MPa")
        return v
    
    @property
    def fyd(self) -> float:
        """Design yield strength (MPa)."""
        return self.fyk / self.gamma_s
    
    @property
    def epsilon_yd(self) -> float:
        """Design yield strain."""
        return self.fyd / self.Es


# ============================================================================
# MATERIAL LOADER FROM YAML
# ============================================================================

def _load_ec2_materials() -> Dict[str, Any]:
    """Load Eurocode 2 material properties from YAML file."""
    yaml_path = Path(__file__).parent.parent.parent / "data" / "standards" / "materials_ec2.yaml"
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


# Load materials from YAML on module import
_EC2_MATERIALS = _load_ec2_materials()


class MaterialLoaderEC2:
    """Factory for Eurocode 2 material properties from YAML data."""
    
    @staticmethod
    def get_concrete(grade: str) -> ConcreteEC2:
        """
        Get concrete material by grade.
        
        Args:
            grade: Strength class (e.g., "C20_25", "C30/37")
        
        Returns:
            ConcreteEC2 instance with properties from YAML
        """
        # Normalize grade name: "C30/37" -> "C30_37" for YAML lookup
        normalized = grade.replace('/', '_')
        
        if normalized not in _EC2_MATERIALS['concrete']:
            available = [k.replace('_', '/') for k in _EC2_MATERIALS['concrete'].keys()]
            raise ValueError(f"Unknown concrete grade: {grade}. Available: {available}")
        
        props = _EC2_MATERIALS['concrete'][normalized]
        return ConcreteEC2(
            strength_class=grade.replace('_', '/'),  # Display format with slash
            fck=props['fck'],
            fck_cube=props['fck_cube']
        )
    
    @staticmethod
    def get_steel(grade: str) -> SteelEC2:
        """
        Get steel material by grade.
        
        Args:
            grade: Steel class (e.g., "B400", "B500A", "B500B")
        
        Returns:
            SteelEC2 instance with properties from YAML
        """
        if grade not in _EC2_MATERIALS['steel']:
            available = list(_EC2_MATERIALS['steel'].keys())
            raise ValueError(f"Unknown steel grade: {grade}. Available: {available}")
        
        props = _EC2_MATERIALS['steel'][grade]
        return SteelEC2(
            grade=grade,
            fyk=props['fyk'],
            ftk=props['ftk'],
            epsilon_uk=props['epsilon_uk'] / 100  # Convert % to decimal
        )
    
    @staticmethod
    def list_concrete_grades() -> List[str]:
        """Get list of available concrete grades."""
        return list(_EC2_MATERIALS['concrete'].keys())
    
    @staticmethod
    def list_steel_grades() -> List[str]:
        """Get list of available steel grades."""
        return list(_EC2_MATERIALS['steel'].keys())
    
    @staticmethod
    def get_parameters() -> Dict[str, Any]:
        """Get design parameters from YAML."""
        return _EC2_MATERIALS.get('parameters', {})
    
    # Backward compatibility: Class attributes for existing code
    @property
    def CONCRETE_GRADES(self) -> Dict[str, Dict[str, float]]:
        """Legacy attribute for backward compatibility."""
        return {
            grade.replace('_', '/'): {"fck": props['fck'], "fck_cube": props['fck_cube']}
            for grade, props in _EC2_MATERIALS['concrete'].items()
        }
    
    @property
    def STEEL_GRADES(self) -> Dict[str, Dict[str, float]]:
        """Legacy attribute for backward compatibility."""
        return {
            grade: {"fyk": props['fyk'], "ftk": props['ftk'], "epsilon_uk": props['epsilon_uk'] / 100}
            for grade, props in _EC2_MATERIALS['steel'].items()
        }


# ============================================================================
# SECTION GEOMETRY
# ============================================================================

class BeamSectionEC2(BaseModel):
    """Beam cross-section geometry."""
    b: float  # Width (mm)
    h: float  # Height (mm)
    cover: float  # Concrete cover to reinforcement centroid (mm)
    
    @property
    def d(self) -> float:
        """Effective depth (mm)."""
        return self.h - self.cover


class ColumnSectionEC2(BaseModel):
    """Column cross-section geometry."""
    b: float  # Width (mm)
    h: float  # Height (mm)
    cover: float  # Cover (mm)
    L_eff: float  # Effective length (mm)
    
    @property
    def i(self) -> float:
        """Radius of gyration (mm)."""
        return self.h / np.sqrt(12)
    
    @property
    def slenderness(self) -> float:
        """Slenderness ratio λ = L_eff / i."""
        return self.L_eff / self.i


# ============================================================================
# EUROCODE 2 IMPLEMENTATION
# ============================================================================

class Eurocode2Code(DesignCode):
    """Eurocode 2 (EN 1992-1-1:2004) implementation."""
    
    @property
    def code_name(self) -> str:
        return "Eurocode 2 EN 1992-1-1:2004"
    
    @property
    def code_units(self) -> str:
        return "SI"
    
    # ========================================================================
    # BEAM FLEXURE (Section 6.1)
    # ========================================================================
    
    def check_beam_flexure(
        self,
        M_Ed: float,  # Design moment (kN·m)
        section: BeamSectionEC2,
        concrete: ConcreteEC2,
        steel: SteelEC2,
        As: float,  # Tension reinforcement area (mm²)
        As_prime: float = 0.0  # Compression reinforcement (mm²)
    ) -> Dict[str, Any]:
        """
        Check beam flexural capacity per EN 1992-1-1 Section 6.1.
        
        Uses rectangular stress block (3.1.7).
        
        Args:
            M_Ed: Design moment (kN·m)
            section: Beam geometry
            concrete: Concrete properties
            steel: Steel properties
            As: Tension reinforcement area (mm²)
            As_prime: Compression reinforcement area (mm²)
        
        Returns:
            dict with UR, status, M_Rd, details
        """
        b = section.b
        d = section.d
        h = section.h
        cover = section.cover
        d_prime = cover  # Compression steel depth
        
        fcd = concrete.fcd
        fyd = steel.fyd
        lambda_f = concrete.lambda_factor
        eta_f = concrete.eta_factor
        epsilon_cu = concrete.epsilon_cu1
        epsilon_yd = steel.epsilon_yd
        
        # Convert moment to N·mm
        M_Ed_Nmm = M_Ed * 1e6
        
        # Calculate neutral axis depth (assuming tension-controlled)
        # Equilibrium: Cc = T
        # eta·fcd·lambda·x·b = As·fyd
        if As > 0:
            x_balance = (As * fyd) / (eta_f * fcd * lambda_f * b)
        else:
            x_balance = 0
        
        # Check if section is tension-controlled
        # x_limit corresponds to balanced failure (εs = εyd)
        x_limit = (epsilon_cu / (epsilon_cu + epsilon_yd)) * d
        
        if x_balance > x_limit:
            # Over-reinforced section (compression failure)
            status_rebar = "OVER-REINFORCED"
            x = x_limit  # Use limited neutral axis
        else:
            x = x_balance
            status_rebar = "OK"
        
        # Calculate moment capacity
        # M_Rd = Cc · z
        # z = d - lambda·x/2 (lever arm)
        z = d - lambda_f * x / 2
        
        # Concrete compression force
        Cc = eta_f * fcd * lambda_f * x * b  # N
        
        # Moment capacity
        M_Rd_Nmm = Cc * z  # N·mm
        M_Rd = M_Rd_Nmm / 1e6  # kN·m
        
        # Utilization ratio
        UR = M_Ed / M_Rd if M_Rd > 0 else 999
        
        # Status
        if UR <= 1.0:
            status = "PASS"
        else:
            status = "FAIL"
        
        # Minimum/maximum reinforcement check (9.2.1.1)
        rho_min = max(0.26 * concrete.fctm / steel.fyk, 0.0013)
        rho_max = 0.04  # 4% per 9.2.1.1(3)
        rho_actual = As / (b * d)
        
        if rho_actual < rho_min:
            status_rebar = f"FAIL - ρ < ρ_min ({rho_min:.4f})"
        elif rho_actual > rho_max:
            status_rebar = f"FAIL - ρ > ρ_max ({rho_max:.2f})"
        
        details = f"""
EC2 Flexural Check (EN 1992-1-1 Section 6.1)
=============================================
Geometry:
  b = {b:.0f} mm, h = {h:.0f} mm, d = {d:.0f} mm
  
Materials:
  fck = {concrete.fck:.1f} MPa, fcd = {fcd:.2f} MPa
  fyk = {steel.fyk:.0f} MPa, fyd = {fyd:.1f} MPa
  λ = {lambda_f:.3f}, η = {eta_f:.3f}
  
Reinforcement:
  As = {As:.0f} mm²
  ρ = {rho_actual:.4f} (min: {rho_min:.4f}, max: {rho_max:.2f})
  Status: {status_rebar}
  
Analysis:
  Neutral axis x = {x:.1f} mm (limit: {x_limit:.1f} mm)
  Lever arm z = {z:.1f} mm
  Concrete force Cc = {Cc/1000:.1f} kN
  
Capacity:
  M_Ed = {M_Ed:.2f} kN·m (demand)
  M_Rd = {M_Rd:.2f} kN·m (capacity)
  UR = {UR:.3f}
  Status: {status}
"""
        
        return {
            'UR': UR,
            'status': status,
            'M_Rd': M_Rd,
            'x': x,
            'z': z,
            'rho': rho_actual,
            'rho_min': rho_min,
            'rho_max': rho_max,
            'details': details
        }
    
    # ========================================================================
    # BEAM SHEAR (Section 6.2)
    # ========================================================================
    
    def check_beam_shear(
        self,
        V_Ed: float,  # Design shear force (kN)
        section: BeamSectionEC2,
        concrete: ConcreteEC2,
        steel: SteelEC2,
        Asw_s: float = 0.0,  # Shear reinforcement ratio Asw/s (mm²/mm)
        theta: float = 21.8  # Strut angle (degrees), default cot(θ)=2.5
    ) -> Dict[str, Any]:
        """
        Check beam shear capacity per EN 1992-1-1 Section 6.2.
        
        Variable strut inclination method (6.2.3).
        
        Args:
            V_Ed: Design shear force (kN)
            section: Beam geometry
            concrete: Concrete properties
            steel: Steel properties
            Asw_s: Shear reinforcement ratio Asw/s (mm²/mm)
            theta: Strut angle (degrees), 21.8° ≤ θ ≤ 45° (cot θ: 2.5 to 1.0)
        
        Returns:
            dict with UR, status, V_Rd, details
        """
        b = section.b
        d = section.d
        
        fck = concrete.fck
        fcd = concrete.fcd
        fctd = concrete.fctd
        fyd = steel.fyd
        
        # Convert to N
        V_Ed_N = V_Ed * 1000
        
        # Shear without shear reinforcement (6.2.2)
        # V_Rd,c = [C_Rd,c · k · (100 · ρ_l · fck)^(1/3)] · b_w · d
        C_Rdc = 0.18 / concrete.gamma_c
        k = min(1 + np.sqrt(200 / d), 2.0)  # Size effect
        
        # Assume longitudinal reinforcement ratio
        rho_l = 0.02  # 2% (conservative estimate, should be calculated from As)
        
        V_Rdc_N = C_Rdc * k * ((100 * rho_l * fck) ** (1/3)) * b * d
        V_Rdc_min = 0.035 * (k ** 1.5) * (fck ** 0.5) * b * d
        V_Rdc = max(V_Rdc_N, V_Rdc_min)
        
        # Shear with shear reinforcement (6.2.3)
        theta_rad = np.deg2rad(theta)
        cot_theta = 1 / np.tan(theta_rad)
        
        # V_Rd,s = (Asw/s) · z · fyd · cot(θ)
        z = 0.9 * d  # Lever arm
        V_Rds_N = Asw_s * z * fyd * cot_theta
        
        # Crushing capacity (6.2.3, Eq. 6.9)
        alpha_cw = 1.0  # For non-prestressed members
        nu1 = 0.6 * (1 - fck / 250)  # Strength reduction factor
        V_Rdmax_N = alpha_cw * b * z * nu1 * fcd / (cot_theta + np.tan(theta_rad))
        
        # Total capacity
        if Asw_s > 0:
            V_Rd_N = min(V_Rds_N, V_Rdmax_N)
            capacity_type = "With stirrups"
        else:
            V_Rd_N = V_Rdc
            capacity_type = "Concrete only"
        
        V_Rd = V_Rd_N / 1000  # kN
        
        # Utilization
        UR = V_Ed / V_Rd if V_Rd > 0 else 999
        status = "PASS" if UR <= 1.0 else "FAIL"
        
        # Minimum shear reinforcement (9.2.2)
        rho_w_min = 0.08 * np.sqrt(fck) / steel.fyk
        Asw_s_min = rho_w_min * b
        
        if Asw_s > 0 and Asw_s < Asw_s_min:
            status_stirrups = f"FAIL - Asw/s < min ({Asw_s_min:.3f} mm²/mm)"
        else:
            status_stirrups = "OK"
        
        details = f"""
EC2 Shear Check (EN 1992-1-1 Section 6.2)
==========================================
Geometry:
  b = {b:.0f} mm, d = {d:.0f} mm, z = {z:.0f} mm
  
Materials:
  fck = {fck:.1f} MPa, fcd = {fcd:.2f} MPa
  fyd = {fyd:.1f} MPa
  
Shear reinforcement:
  Asw/s = {Asw_s:.3f} mm²/mm (min: {Asw_s_min:.3f} mm²/mm)
  θ = {theta:.1f}° (cot θ = {cot_theta:.2f})
  Status: {status_stirrups}
  
Capacity ({capacity_type}):
  V_Rd,c = {V_Rdc/1000:.2f} kN (concrete)
  V_Rd,s = {V_Rds_N/1000:.2f} kN (stirrups)
  V_Rd,max = {V_Rdmax_N/1000:.2f} kN (crushing)
  V_Rd = {V_Rd:.2f} kN (governing)
  
Check:
  V_Ed = {V_Ed:.2f} kN (demand)
  UR = {UR:.3f}
  Status: {status}
"""
        
        return {
            'UR': UR,
            'status': status,
            'V_Rd': V_Rd,
            'V_Rdc': V_Rdc / 1000,
            'V_Rds': V_Rds_N / 1000,
            'V_Rdmax': V_Rdmax_N / 1000,
            'Asw_s_min': Asw_s_min,
            'details': details
        }
    
    # ========================================================================
    # COLUMN (Section 6.1 + Slenderness)
    # ========================================================================
    
    def check_column(
        self,
        N_Ed: float,  # Design axial force (kN, compression positive)
        M_Ed: float,  # Design moment (kN·m)
        section: ColumnSectionEC2,
        concrete: ConcreteEC2,
        steel: SteelEC2,
        As_total: float,  # Total reinforcement area (mm²)
        n_points: int = 50
    ) -> Dict[str, Any]:
        """
        Check column capacity per EN 1992-1-1 Section 6.1.
        
        Generates N-M interaction diagram and checks slenderness effects (5.8).
        
        Args:
            N_Ed: Design axial force (kN, compression positive)
            M_Ed: Design moment (kN·m)
            section: Column geometry
            concrete: Concrete properties
            steel: Steel properties
            As_total: Total reinforcement area (mm²)
            n_points: Number of points on interaction curve
        
        Returns:
            dict with UR, status, figure, eta (slenderness factor), details
        """
        b = section.b
        h = section.h
        cover = section.cover
        d = h - cover
        d_prime = cover
        
        fcd = concrete.fcd
        fyd = steel.fyd
        lambda_f = concrete.lambda_factor
        eta_f = concrete.eta_factor
        epsilon_cu = concrete.epsilon_cu1
        epsilon_yd = steel.epsilon_yd
        
        # Check slenderness (5.8.2)
        lambda_ratio = section.slenderness
        lambda_lim = 20  # Simplified (5.8.3.1)
        
        if lambda_ratio > lambda_lim:
            # Slender column - apply moment magnification
            # Simplified: M_Ed,tot = M_Ed · (1 + beta / (N_Ed / N_B))
            # For this implementation, use approximate factor
            eta = 1.0 + 0.01 * (lambda_ratio - lambda_lim)
            M_Ed_total = M_Ed * eta
            slenderness_note = f"Slender (λ = {lambda_ratio:.1f}), η = {eta:.2f}"
        else:
            eta = 1.0
            M_Ed_total = M_Ed
            slenderness_note = f"Short (λ = {lambda_ratio:.1f})"
        
        # Generate interaction diagram
        N_cap, M_cap = self._generate_interaction_curve_ec2(
            b, h, cover, concrete, steel, As_total, n_points
        )
        
        # Check if applied loads are inside envelope
        # Simple distance check
        distances = [np.sqrt((N_Ed - Nc)**2 + (M_Ed_total - Mc)**2)
                     for Nc, Mc in zip(N_cap, M_cap)]
        min_dist_idx = np.argmin(distances)
        
        # Approximate UR (ratio to capacity envelope)
        if len(N_cap) > 0:
            max_N = max(abs(n) for n in N_cap)
            max_M = max(abs(m) for m in M_cap)
            demand_norm = np.sqrt((N_Ed / max_N)**2 + (M_Ed_total / max_M)**2) if max_N > 0 and max_M > 0 else 0
            capacity_norm = 1.0
            UR = demand_norm / capacity_norm
        else:
            UR = 999
        
        status = "PASS" if UR <= 1.0 else "FAIL"
        
        # Create interaction diagram plot
        fig = self._plot_interaction_diagram_ec2(
            N_cap, M_cap, N_Ed, M_Ed_total,
            title=f"EC2 Column Interaction Diagram ({section.b}×{section.h}mm)"
        )
        
        details = f"""
EC2 Column Check (EN 1992-1-1 Section 6.1 + 5.8)
=================================================
Geometry:
  b = {b:.0f} mm, h = {h:.0f} mm, L_eff = {section.L_eff:.0f} mm
  {slenderness_note}
  
Materials:
  fck = {concrete.fck:.1f} MPa, fcd = {fcd:.2f} MPa
  fyk = {steel.fyk:.0f} MPa, fyd = {fyd:.1f} MPa
  
Reinforcement:
  As,total = {As_total:.0f} mm²
  ρ = {As_total/(b*h):.4f}
  
Applied loads:
  N_Ed = {N_Ed:.2f} kN
  M_Ed = {M_Ed:.2f} kN·m (amplified: {M_Ed_total:.2f} kN·m)
  
Check:
  UR = {UR:.3f}
  Status: {status}
"""
        
        return {
            'UR': UR,
            'status': status,
            'eta': eta,
            'N_capacity': N_cap,
            'M_capacity': M_cap,
            'figure': fig,
            'details': details
        }
    
    # ========================================================================
    # SERVICEABILITY (Section 7)
    # ========================================================================
    
    def check_serviceability(
        self,
        M_ser: float,  # Service moment (kN·m)
        section: BeamSectionEC2,
        concrete: ConcreteEC2,
        steel: SteelEC2,
        As: float,  # Tension reinforcement (mm²)
        L_span: float,  # Span length (mm)
        exposure_class: str = "XC1",  # Exposure class per EN 206-1
        load_duration: str = "long_term"
    ) -> Dict[str, Any]:
        """
        Check serviceability (crack width and deflection) per EN 1992-1-1 Section 7.
        
        Args:
            M_ser: Service moment (unfactored, kN·m)
            section: Beam geometry
            concrete: Concrete properties
            steel: Steel properties
            As: Tension reinforcement area (mm²)
            L_span: Span length (mm)
            exposure_class: Exposure class (XC1, XC2, XC3, XC4, XD1, XS1, etc.)
            load_duration: 'short_term' or 'long_term'
        
        Returns:
            dict with crack_width check, deflection check, details
        """
        b = section.b
        d = section.d
        h = section.h
        
        fck = concrete.fck
        fctm = concrete.fctm
        Ecm = concrete.Ecm
        Es = steel.Es
        fyk = steel.fyk
        
        M_ser_Nmm = M_ser * 1e6
        
        # ====================================================================
        # CRACK WIDTH CHECK (7.3)
        # ====================================================================
        
        # Allowable crack width (Table 7.1N)
        crack_limits = {
            "XC1": 0.4,  # mm - dry environment
            "XC2": 0.3,
            "XC3": 0.3,
            "XC4": 0.3,
            "XD1": 0.3,
            "XS1": 0.3,
        }
        w_max = crack_limits.get(exposure_class, 0.3)
        
        # Calculate stress in steel at service load
        # Assume cracked section
        n = Es / Ecm  # Modular ratio
        rho = As / (b * d)
        
        # Neutral axis depth (cracked section)
        k = np.sqrt((n * rho)**2 + 2 * n * rho) - n * rho
        x = k * d
        
        # Lever arm
        z = d - x / 3
        
        # Steel stress
        sigma_s = M_ser_Nmm / (As * z) if As > 0 else 0  # MPa
        
        # Crack width per 7.3.4
        # w_k = s_r,max · (ε_sm - ε_cm)
        
        # Maximum crack spacing (7.3.4, Eq. 7.11)
        phi = 12  # mm, assumed bar diameter
        k1 = 0.8  # High bond bars
        k2 = 0.5  # Bending
        c = section.cover  # mm
        s = 150  # mm, assumed bar spacing
        
        s_r_max = 3.4 * c + 0.425 * k1 * k2 * phi / rho if rho > 0 else 0
        
        # Strain difference (simplified, 7.3.4)
        epsilon_sm = sigma_s / Es
        
        # Minimum crack-inducing stress
        sigma_sc = k * fctm * (1 + n * rho) / rho if rho > 0 else 0
        
        if sigma_s > sigma_sc:
            epsilon_cm = 0.6 * sigma_s / Es
        else:
            epsilon_cm = 0
        
        # Crack width
        w_k = s_r_max * (epsilon_sm - epsilon_cm)
        
        UR_crack = w_k / w_max if w_max > 0 else 999
        status_crack = "PASS" if UR_crack <= 1.0 else "FAIL"
        
        # ====================================================================
        # DEFLECTION CHECK (7.4)
        # ====================================================================
        
        # Allowable deflection (7.4.1)
        delta_max = L_span / 250  # mm
        
        # Moment of inertia
        Ig = (b * h**3) / 12  # Gross section (mm⁴)
        
        # Cracked moment of inertia
        Icr = (b * x**3) / 3 + n * As * (d - x)**2
        
        # Cracking moment
        M_cr = (Ecm * Ig) / (h / 2) * fctm / 1e6  # kN·m
        
        # Effective moment of inertia (Branson)
        if M_ser < M_cr:
            Ie = Ig
        else:
            Ie = Icr + (Ig - Icr) * (M_cr / M_ser)**3
        
        # Curvature
        kappa = M_ser_Nmm / (Ecm * Ie) if Ie > 0 else 0  # 1/mm
        
        # Deflection (simply-supported, uniform load approximation)
        delta = kappa * L_span**2 / 8  # mm
        
        # Long-term deflection (creep)
        if load_duration == "long_term":
            phi_creep = 2.0  # Creep factor (simplified)
            delta_total = delta * (1 + phi_creep)
        else:
            delta_total = delta
        
        UR_deflection = delta_total / delta_max if delta_max > 0 else 999
        status_deflection = "PASS" if UR_deflection <= 1.0 else "FAIL"
        
        # ====================================================================
        # COMBINED STATUS
        # ====================================================================
        
        if status_crack == "PASS" and status_deflection == "PASS":
            status_overall = "PASS"
        else:
            status_overall = "FAIL"
        
        details = f"""
EC2 Serviceability Check (EN 1992-1-1 Section 7)
=================================================
Crack Width Check (7.3):
  Exposure class: {exposure_class}
  Allowable w_max = {w_max:.2f} mm
  Steel stress σ_s = {sigma_s:.1f} MPa
  Max crack spacing s_r,max = {s_r_max:.1f} mm
  Calculated w_k = {w_k:.3f} mm
  UR = {UR_crack:.3f}
  Status: {status_crack}
  
Deflection Check (7.4):
  Span L = {L_span:.0f} mm
  Allowable δ_max = {delta_max:.2f} mm (L/250)
  Cracking moment M_cr = {M_cr:.2f} kN·m
  Effective inertia I_e = {Ie:.0f} mm⁴
  Short-term deflection = {delta:.2f} mm
  Long-term deflection = {delta_total:.2f} mm
  UR = {UR_deflection:.3f}
  Status: {status_deflection}
  
Overall: {status_overall}
"""
        
        return {
            'UR_crack': UR_crack,
            'UR_deflection': UR_deflection,
            'status_crack': status_crack,
            'status_deflection': status_deflection,
            'status': status_overall,
            'w_k': w_k,
            'w_max': w_max,
            'delta': delta_total,
            'delta_max': delta_max,
            'details': details
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _generate_interaction_curve_ec2(
        self,
        b: float, h: float, cover: float,
        concrete: ConcreteEC2,
        steel: SteelEC2,
        As_total: float,
        n_points: int = 50
    ) -> Tuple[List[float], List[float]]:
        """Generate N-M interaction diagram for rectangular column."""
        fcd = concrete.fcd
        fyd = steel.fyd
        lambda_f = concrete.lambda_factor
        eta_f = concrete.eta_factor
        epsilon_cu = concrete.epsilon_cu1
        epsilon_yd = steel.epsilon_yd
        
        d = h - cover
        d_prime = cover
        
        N_values = []
        M_values = []
        
        # Point 1: Pure compression
        N_max = eta_f * fcd * b * h + (fyd - eta_f * fcd) * As_total
        N_values.append(N_max / 1000)  # kN
        M_values.append(0)
        
        # Iterate through neutral axis depths
        for x in np.linspace(h, 0.1*h, n_points):
            # Strain at extreme compression fiber
            epsilon_c = epsilon_cu
            
            # Strain in tension steel
            epsilon_s = epsilon_c * (d - x) / x if x > 0 else epsilon_yd
            
            # Concrete force
            a = lambda_f * x  # Stress block depth
            Cc = eta_f * fcd * b * a  # N
            
            # Steel forces (assume uniform distribution for simplicity)
            # Tension steel
            if epsilon_s > epsilon_yd:
                fs = fyd
            elif epsilon_s > 0:
                fs = epsilon_s * steel.Es
            else:
                fs = -fyd  # Compression
            
            Ts = fs * As_total / 2  # Half in tension
            
            # Compression steel (approximate)
            Cs = fyd * As_total / 2  # Half in compression
            
            # Total axial force (compression positive)
            N = Cc + Cs - Ts
            
            # Moment about centroid
            Mc = Cc * (h/2 - a/2)
            Ms = Ts * (d - h/2) + Cs * (h/2 - d_prime)
            M = Mc + Ms
            
            N_values.append(N / 1000)  # kN
            M_values.append(M / 1e6)  # kN·m
        
        # Point N: Pure tension
        N_min = -fyd * As_total
        N_values.append(N_min / 1000)
        M_values.append(0)
        
        return N_values, M_values
    
    def _plot_interaction_diagram_ec2(
        self,
        N_capacity: List[float],
        M_capacity: List[float],
        N_applied: float,
        M_applied: float,
        title: str = "EC2 Column Interaction Diagram"
    ) -> go.Figure:
        """Create Plotly interaction diagram."""
        fig = go.Figure()
        
        # Capacity envelope
        fig.add_trace(go.Scatter(
            x=M_capacity,
            y=N_capacity,
            mode='lines',
            name='Capacity Envelope',
            line=dict(color='blue', width=3),
            fill='toself',
            fillcolor='rgba(100, 150, 255, 0.2)'
        ))
        
        # Applied load point
        fig.add_trace(go.Scatter(
            x=[M_applied],
            y=[N_applied],
            mode='markers',
            name='Applied Load',
            marker=dict(size=12, color='red', symbol='star')
        ))
        
        # Layout
        fig.update_layout(
            title=title,
            xaxis_title='Moment M_Ed (kN·m)',
            yaxis_title='Axial Force N_Ed (kN)',
            hovermode='closest',
            width=700,
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
        
        return fig
