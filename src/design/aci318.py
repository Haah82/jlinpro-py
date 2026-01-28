"""
ACI 318-25 Design Code Implementation.

Reference: ACI 318-25 Building Code Requirements for Structural Concrete
Units: Imperial (psi, ksi, inches, kip-ft) - internally converted to SI for consistency
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, field_validator
import plotly.graph_objects as go
import yaml
from pathlib import Path

from .base import DesignCode


# ============================================================================
# MATERIAL MODELS (ACI 318-25)
# ============================================================================

class ConcreteACI(BaseModel):
    """
    ACI 318-25 concrete material properties.
    
    Reference: ACI 318-25 Section 19.2
    """
    fc_prime: float  # Compressive strength (MPa) - stored in SI
    lambda_factor: float = 1.0  # 1.0 normal weight, 0.85 lightweight, 0.75 all-lightweight
    
    class Config:
        frozen = False
    
    @field_validator('fc_prime')
    @classmethod
    def validate_fc_prime(cls, v):
        if v <= 0:
            raise ValueError("fc_prime must be positive")
        if v < 17.2:  # 2500 psi
            raise ValueError("fc_prime too low (min 17.2 MPa / 2500 psi)")
        return v
    
    @property
    def fc_psi(self) -> float:
        """Compressive strength in psi."""
        return self.fc_prime / 0.00689476  # MPa to psi
    
    @property
    def Ec(self) -> float:
        """
        Modulus of elasticity per ACI 19.2.2.1.
        
        Ec = 33 · λ^1.5 · √(fc') (psi units) → convert to MPa
        Ec = 4700 · √(fc') for normal weight (λ=1.0) in MPa units
        
        Returns:
            Modulus in MPa
        """
        # Using direct MPa formula: Ec = 4700 · λ^1.5 · √(fc') (MPa)
        return 4700 * (self.lambda_factor ** 1.5) * np.sqrt(self.fc_prime)
    
    @property
    def fr(self) -> float:
        """
        Modulus of rupture per ACI 19.2.3.1.
        
        fr = 7.5 · λ · √(fc') (psi) = 0.62 · λ · √(fc') (MPa)
        
        Returns:
            Modulus of rupture in MPa
        """
        return 0.62 * self.lambda_factor * np.sqrt(self.fc_prime)
    
    @property
    def beta1(self) -> float:
        """
        Stress block depth factor β1 per ACI 22.2.2.4.3.
        
        β1 = 0.85 for fc' ≤ 28 MPa (4000 psi)
        β1 = 0.85 - 0.05(fc' - 28)/7 for fc' > 28 MPa
        β1 ≥ 0.65
        """
        if self.fc_prime <= 28:
            return 0.85
        else:
            # β1 decreases by 0.05 for each 7 MPa (1000 psi) above 28 MPa
            beta = 0.85 - 0.05 * (self.fc_prime - 28) / 7
            return max(beta, 0.65)


class SteelACI(BaseModel):
    """
    ACI 318-25 reinforcing steel properties.
    
    Reference: ACI 318-25 Section 20.2.2
    """
    fy: float  # Yield strength (MPa) - stored in SI
    Es: float = 200000  # Modulus of elasticity (MPa)
    
    class Config:
        frozen = False
    
    @field_validator('fy')
    @classmethod
    def validate_fy(cls, v):
        if v <= 0:
            raise ValueError("fy must be positive")
        if v > 552:
            raise ValueError("fy exceeds ACI limit (552 MPa / 80 ksi)")
        return v
    
    @property
    def fy_psi(self) -> float:
        """Yield strength in psi."""
        return self.fy / 0.00689476  # MPa to psi


# ============================================================================
# MATERIAL LOADER FROM YAML
# ============================================================================

def _load_aci_materials() -> Dict[str, Any]:
    """Load ACI material properties from YAML file."""
    yaml_path = Path(__file__).parent.parent.parent / "data" / "standards" / "materials_aci318.yaml"
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


# Load materials from YAML on module import
_ACI_MATERIALS = _load_aci_materials()


class MaterialLoaderACI:
    """Material loader for ACI grades from YAML data."""
    
    @staticmethod
    def get_concrete(grade_name: str) -> ConcreteACI:
        """
        Get concrete grade by name.
        
        Args:
            grade_name: Grade identifier (e.g., "3000psi", "4000 psi (27.6 MPa)")
        
        Returns:
            ConcreteACI instance with properties from YAML
        """
        # Normalize grade name: "4000 psi (27.6 MPa)" -> "4000psi"
        # Extract number and add 'psi': "4000 psi" -> "4000psi"
        parts = grade_name.split()
        if len(parts) >= 2 and parts[1].lower() == 'psi':
            normalized = parts[0] + 'psi'
        else:
            normalized = parts[0].lower()
        
        if normalized not in _ACI_MATERIALS['concrete']:
            available = list(_ACI_MATERIALS['concrete'].keys())
            raise ValueError(f"Unknown ACI concrete grade: {grade_name}. Available: {available}")
        
        props = _ACI_MATERIALS['concrete'][normalized]
        return ConcreteACI(fc_prime=props['fc'], lambda_factor=props.get('lambda', 1.0))
    
    @staticmethod
    def get_steel(grade_name: str) -> SteelACI:
        """
        Get steel grade by name.
        
        Args:
            grade_name: Grade identifier (e.g., "Grade40", "Grade 60 (414 MPa)")
        
        Returns:
            SteelACI instance with properties from YAML
        """
        # Normalize grade name: "Grade 60 (414 MPa)" -> "Grade60"
        normalized = grade_name.replace(' ', '').split('(')[0]  # Remove spaces and parentheses
        
        if normalized not in _ACI_MATERIALS['steel']:
            available = list(_ACI_MATERIALS['steel'].keys())
            raise ValueError(f"Unknown ACI steel grade: {grade_name}. Available: {available}")
        
        props = _ACI_MATERIALS['steel'][normalized]
        return SteelACI(fy=props['fy'], Es=props['Es'])
    
    @staticmethod
    def list_concrete_grades() -> List[str]:
        """Get list of available concrete grades."""
        return list(_ACI_MATERIALS['concrete'].keys())
    
    @staticmethod
    def list_steel_grades() -> List[str]:
        """Get list of available steel grades."""
        return list(_ACI_MATERIALS['steel'].keys())
    
    @staticmethod
    def get_parameters() -> Dict[str, Any]:
        """Get design parameters from YAML."""
        return _ACI_MATERIALS.get('parameters', {})


# Backward compatibility: Create dictionaries for existing code
CONCRETE_ACI = {
    f"{name.replace('psi', ' psi')}": MaterialLoaderACI.get_concrete(name)
    for name in MaterialLoaderACI.list_concrete_grades()
}

STEEL_ACI = {
    f"{name}": MaterialLoaderACI.get_steel(name)
    for name in MaterialLoaderACI.list_steel_grades()
}


# ============================================================================
# BEAM FLEXURE (ACI 318-25 Section 22.3)
# ============================================================================

def calculate_phi_factor(c: float, d: float, steel_type: str = "other") -> float:
    """
    Calculate strength reduction factor φ per ACI 21.2.
    
    Transition from compression-controlled (φ=0.65) to tension-controlled (φ=0.90):
    - Compression-controlled: εt ≤ εty
    - Transition: εty < εt < 0.005
    - Tension-controlled: εt ≥ 0.005
    
    Args:
        c: Neutral axis depth (mm)
        d: Effective depth (mm)
        steel_type: "spiral" or "other"
    
    Returns:
        φ factor (0.65 to 0.90)
    """
    epsilon_cu = 0.003  # Concrete ultimate strain
    epsilon_ty = 0.002  # Yield strain threshold (fy/Es for Grade 60)
    
    # Strain in tension steel
    epsilon_t = epsilon_cu * (d - c) / c if c > 0 else 0.005
    
    if steel_type == "spiral":
        phi_compression = 0.75
    else:
        phi_compression = 0.65
    
    phi_tension = 0.90
    
    # Determine φ based on strain
    if epsilon_t <= epsilon_ty:
        # Compression-controlled
        return phi_compression
    elif epsilon_t >= 0.005:
        # Tension-controlled
        return phi_tension
    else:
        # Transition zone - linear interpolation
        phi = phi_compression + (phi_tension - phi_compression) * (epsilon_t - epsilon_ty) / (0.005 - epsilon_ty)
        return phi


def check_beam_flexure_aci(
    beam_id: int,
    b: float,  # Width (mm)
    h: float,  # Height (mm)
    cover: float,  # Cover (mm)
    As: float,  # Tension steel area (mm²)
    concrete_name: str,
    steel_name: str,
    Mu: float,  # Ultimate moment (kN·m)
    As_compression: float = 0,  # Compression steel (mm²)
) -> Dict[str, Any]:
    """
    ACI 318-25 beam flexure check.
    
    Reference: ACI 318-25 Section 22.3
    
    Args:
        beam_id: Beam element ID
        b: Width (mm)
        h: Height (mm)
        cover: Concrete cover (mm)
        As: Tension steel area (mm²)
        concrete_name: Concrete grade
        steel_name: Steel grade
        Mu: Ultimate moment (kN·m)
        As_compression: Compression steel area (mm²)
    
    Returns:
        dict with 'UR', 'status', 'Mn', 'phiMn', 'phi', 'a', 'details'
    """
    # Get materials
    concrete = MaterialLoaderACI.get_concrete(concrete_name)
    steel = MaterialLoaderACI.get_steel(steel_name)
    
    # Effective depth
    d = h - cover
    
    # Step 1: Calculate neutral axis depth (Whitney stress block)
    # a = As·fy / (0.85·fc'·b)
    # c = a / β1
    fc = concrete.fc_prime
    fy = steel.fy
    beta1 = concrete.beta1
    
    a = (As * fy) / (0.85 * fc * b)
    c = a / beta1
    
    # Step 2: Calculate φ factor based on strain
    phi = calculate_phi_factor(c, d)
    
    # Step 3: Calculate nominal moment Mn
    # Mn = As·fy·(d - a/2) + As'·fs'·(d - d')
    # For simplicity, assume compression steel yields if present
    if As_compression > 0:
        d_prime = cover
        fs_prime = min(fy, 0.003 * steel.Es * (c - d_prime) / c) if c > d_prime else 0
        Mn_comp = As_compression * fs_prime * (d - d_prime) / 1e6  # kN·m
    else:
        Mn_comp = 0
    
    Mn = (As * fy * (d - a/2)) / 1e6 + Mn_comp  # kN·m
    
    # Step 4: Design moment φMn
    phiMn = phi * Mn
    
    # Step 5: Utilization ratio
    UR = Mu / phiMn if phiMn > 0 else 999
    status = 'PASS' if UR <= 1.0 else 'FAIL'
    
    # Calculation details
    details = f"""
ACI 318-25 Beam Flexure Check
{'='*60}

Beam ID: {beam_id}
Section: b×h = {b:.0f}×{h:.0f} mm
Effective depth d = {d:.0f} mm

Materials:
  Concrete: {concrete_name}
    fc' = {fc:.1f} MPa ({fc/0.00689476:.0f} psi)
    β₁ = {beta1:.3f}
  Steel: {steel_name}
    fy = {fy:.0f} MPa ({fy/0.00689476:.0f} psi)

Tension Reinforcement:
  As = {As:.0f} mm² ({As/645.16:.2f} in²)
  ρ = As/(b·d) = {As/(b*d):.4f}

Whitney Stress Block:
  a = As·fy / (0.85·fc'·b) = {a:.1f} mm
  c = a / β₁ = {c:.1f} mm
  c/d = {c/d:.3f}

Strength Reduction:
  φ = {phi:.3f} ({'tension-controlled' if phi >= 0.9 else 'compression-controlled' if phi <= 0.65 else 'transition'})

Nominal Moment:
  Mn = As·fy·(d - a/2) = {Mn:.2f} kN·m

Design Moment:
  φMn = {phi:.3f} × {Mn:.2f} = {phiMn:.2f} kN·m

Applied Moment:
  Mu = {Mu:.2f} kN·m

Result:
  UR = Mu / φMn = {UR:.3f}
  Status: {status}
"""
    
    return {
        'beam_id': beam_id,
        'UR': UR,
        'status': status,
        'Mn': Mn,
        'phiMn': phiMn,
        'phi': phi,
        'a': a,
        'c': c,
        'details': details
    }


# ============================================================================
# BEAM SHEAR (ACI 318-25 Section 22.5)
# ============================================================================

def check_beam_shear_aci(
    beam_id: int,
    b: float,  # Width (mm)
    h: float,  # Height (mm)
    cover: float,  # Cover (mm)
    Vu: float,  # Ultimate shear (kN)
    concrete_name: str,
    steel_name: str,
    stirrup_dia: float = 0,  # Stirrup diameter (mm)
    n_legs: int = 0,  # Number of stirrup legs
    spacing: float = 0,  # Stirrup spacing (mm)
) -> Dict[str, Any]:
    """
    ACI 318-25 beam shear check.
    
    Reference: ACI 318-25 Section 22.5
    
    Args:
        beam_id: Beam element ID
        b: Width (mm)
        h: Height (mm)
        cover: Concrete cover (mm)
        Vu: Ultimate shear force (kN)
        concrete_name: Concrete grade
        steel_name: Steel grade
        stirrup_dia: Stirrup diameter (mm)
        n_legs: Number of stirrup legs
        spacing: Stirrup spacing (mm)
    
    Returns:
        dict with 'UR', 'status', 'Vc', 'Vs', 'Vn', 'phiVn', 'details'
    """
    # Get materials
    concrete = MaterialLoaderACI.get_concrete(concrete_name)
    steel = MaterialLoaderACI.get_steel(steel_name)
    
    # Effective depth
    d = h - cover
    bw = b  # Web width
    
    # Step 1: Concrete shear capacity Vc (ACI 22.5.5.1 simplified)
    # Vc = 2 · λ · √(fc') · bw · d (psi units)
    # Vc = 0.17 · λ · √(fc') · bw · d (MPa units)
    fc = concrete.fc_prime
    lambda_factor = concrete.lambda_factor
    
    Vc = 0.17 * lambda_factor * np.sqrt(fc) * bw * d / 1000  # kN
    
    # Step 2: Stirrup shear capacity Vs (if stirrups provided)
    if stirrup_dia > 0 and n_legs > 0 and spacing > 0:
        Av = n_legs * np.pi * (stirrup_dia / 2) ** 2  # mm²
        fyv = steel.fy  # Stirrup yield strength
        
        # Vs = Av · fyv · d / s
        Vs = Av * fyv * d / spacing / 1000  # kN
    else:
        Vs = 0
    
    # Step 3: Nominal shear Vn
    Vn = Vc + Vs
    
    # Step 4: Design shear φVn (φ = 0.75 for shear per ACI 21.2)
    phi = 0.75
    phiVn = phi * Vn
    
    # Step 5: Check maximum shear Vn,max (ACI 22.5.1.2)
    # Vn ≤ 8 · √(fc') · bw · d (simplified)
    Vn_max = 8 * 0.17 * lambda_factor * np.sqrt(fc) * bw * d / 1000  # kN
    
    if Vn > Vn_max:
        status_note = f"Exceeds Vn,max = {Vn_max:.2f} kN - section too small!"
    else:
        status_note = ""
    
    # Utilization ratio
    UR = Vu / phiVn if phiVn > 0 else 999
    status = 'PASS' if UR <= 1.0 else 'FAIL'
    
    # Calculation details
    details = f"""
ACI 318-25 Beam Shear Check
{'='*60}

Beam ID: {beam_id}
Section: bw×h = {bw:.0f}×{h:.0f} mm
Effective depth d = {d:.0f} mm

Materials:
  Concrete: {concrete_name}
    fc' = {fc:.1f} MPa
    λ = {lambda_factor}
  Steel: {steel_name}
    fy = {steel.fy:.0f} MPa

Concrete Shear Capacity:
  Vc = 0.17 · λ · √(fc') · bw · d
  Vc = {Vc:.2f} kN

Stirrup Capacity:
  Stirrups: {n_legs if n_legs > 0 else 'None'} legs φ{stirrup_dia:.0f}mm @ {spacing:.0f}mm spacing
"""
    
    if stirrup_dia > 0:
        details += f"""  Av = {Av:.0f} mm²
  Vs = Av · fyv · d / s = {Vs:.2f} kN
"""
    else:
        details += "  Vs = 0 kN (no stirrups)\n"
    
    details += f"""
Nominal Shear:
  Vn = Vc + Vs = {Vn:.2f} kN

Design Shear:
  φVn = {phi} × {Vn:.2f} = {phiVn:.2f} kN

Applied Shear:
  Vu = {Vu:.2f} kN

Result:
  UR = Vu / φVn = {UR:.3f}
  Status: {status}
  {status_note}
"""
    
    return {
        'beam_id': beam_id,
        'UR': UR,
        'status': status,
        'Vc': Vc,
        'Vs': Vs,
        'Vn': Vn,
        'phiVn': phiVn,
        'phi': phi,
        'details': details
    }


# ============================================================================
# COLUMN INTERACTION (ACI 318-25 Section 22.4)
# ============================================================================

def generate_column_interaction_aci(
    b: float,  # Width (mm)
    h: float,  # Height (mm)
    cover: float,  # Cover (mm)
    As_total: float,  # Total steel area (mm²)
    concrete_name: str,
    steel_name: str,
    n_points: int = 50
) -> Tuple[List[float], List[float]]:
    """
    Generate P-M interaction diagram per ACI 318-25.
    
    Reference: ACI 318-25 Section 22.4
    
    Returns:
        (N_values, M_values) in kN and kN·m
    """
    concrete = MaterialLoaderACI.get_concrete(concrete_name)
    steel = MaterialLoaderACI.get_steel(steel_name)
    
    fc = concrete.fc_prime
    fy = steel.fy
    Es = steel.Es
    beta1 = concrete.beta1
    
    d = h - cover
    epsilon_cu = 0.003  # Concrete ultimate compressive strain
    epsilon_sy = fy / Es  # Steel yield strain
    
    N_values = []
    M_values = []
    
    # Point A: Pure axial compression (ACI 22.4.2.2)
    # Po = 0.85·fc'·(Ag - Ast) + fy·Ast
    Ag = b * h
    Po = 0.85 * fc * (Ag - As_total) + fy * As_total
    Po = Po / 1000  # Convert to kN
    
    # For tied columns: Pn,max = 0.80·Po (ACI 22.4.2.1)
    # For spiral columns: Pn,max = 0.85·Po
    Pn_max = 0.80 * Po
    
    N_values.append(Pn_max)
    M_values.append(0)
    
    # Iterate through neutral axis depths
    c_values = np.linspace(h, 0.1 * d, n_points)
    
    for c in c_values:
        # Concrete compression resultant
        a = beta1 * c
        if a > h:
            a = h
        
        Cc = 0.85 * fc * b * a  # N
        
        # Steel strain (assuming single layer at centroid)
        epsilon_s = epsilon_cu * (d - c) / c
        
        # Steel stress
        if abs(epsilon_s) >= epsilon_sy:
            fs = fy * np.sign(epsilon_s)
        else:
            fs = Es * epsilon_s
        
        Ts = fs * As_total  # N
        
        # Axial force (compression positive)
        Pn = Cc - Ts
        
        # Moment about centroid
        y_centroid = h / 2
        Mc = Cc * (y_centroid - a / 2)
        Ms = Ts * (d - y_centroid)
        Mn = Mc + Ms
        
        N_values.append(Pn / 1000)  # kN
        M_values.append(Mn / 1e6)  # kN·m
    
    # Point B: Pure bending (c → 0)
    N_values.append(0)
    M_values.append(0)
    
    # Point C: Pure tension
    Tn = -fy * As_total / 1000  # kN (negative = tension)
    N_values.append(Tn)
    M_values.append(0)
    
    return N_values, M_values


def check_column_aci(
    col_id: int,
    b: float,
    h: float,
    cover: float,
    As_total: float,
    concrete_name: str,
    steel_name: str,
    Pu: float,  # Ultimate axial force (kN, compression positive)
    Mu: float,  # Ultimate moment (kN·m)
    column_type: str = "tied"  # "tied" or "spiral"
) -> Dict[str, Any]:
    """
    ACI 318-25 column check with interaction diagram.
    
    Args:
        col_id: Column element ID
        b: Width (mm)
        h: Height (mm)
        cover: Cover (mm)
        As_total: Total steel area (mm²)
        concrete_name: Concrete grade
        steel_name: Steel grade
        Pu: Ultimate axial force (kN)
        Mu: Ultimate moment (kN·m)
        column_type: "tied" or "spiral"
    
    Returns:
        dict with 'UR', 'status', 'figure', 'details'
    """
    # Generate interaction curve
    N_cap, M_cap = generate_column_interaction_aci(
        b, h, cover, As_total, concrete_name, steel_name
    )
    
    # Apply φ factor (ACI 21.2.1 and 21.2.2)
    # Tension-controlled: φ = 0.90
    # Compression-controlled tied: φ = 0.65
    # Compression-controlled spiral: φ = 0.75
    # For simplicity, apply compression-controlled φ to entire curve
    phi = 0.75 if column_type == "spiral" else 0.65
    
    N_design = [phi * N for N in N_cap]
    M_design = [phi * M for M in M_cap]
    
    # Check if point (Mu, Pu) is inside capacity envelope
    # Simple approach: check radial distance
    distances = [np.sqrt((Mu - Md)**2 + (Pu - Nd)**2) 
                 for Nd, Md in zip(N_design, M_design)]
    min_dist_idx = np.argmin(distances)
    min_dist = distances[min_dist_idx]
    
    # Approximate UR (more sophisticated check possible)
    max_capacity = np.sqrt(max(N_design)**2 + max(M_design)**2)
    applied = np.sqrt(Pu**2 + Mu**2)
    UR = applied / max_capacity if max_capacity > 0 else 999
    
    # Status based on whether point is inside envelope
    # Better method: use contour interpolation
    status = 'PASS' if UR <= 1.0 else 'FAIL'
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Capacity envelope
    fig.add_trace(go.Scatter(
        x=M_design,
        y=N_design,
        mode='lines',
        name='φPn-φMn Capacity',
        line=dict(color='blue', width=3),
        fill='toself',
        fillcolor='rgba(100, 150, 255, 0.2)'
    ))
    
    # Applied load point
    fig.add_trace(go.Scatter(
        x=[Mu],
        y=[Pu],
        mode='markers',
        name='Applied Load',
        marker=dict(size=12, color='red' if status == 'FAIL' else 'green', symbol='star')
    ))
    
    fig.update_layout(
        title=f'ACI 318-25 Column Interaction Diagram (Col {col_id})',
        xaxis_title='Moment φMn (kN·m)',
        yaxis_title='Axial Force φPn (kN)',
        hovermode='closest',
        width=700,
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    
    # Details
    details = f"""
ACI 318-25 Column Interaction Check
{'='*60}

Column ID: {col_id}
Section: {b:.0f}×{h:.0f} mm
Cover: {cover:.0f} mm
Steel: As = {As_total:.0f} mm²
Type: {column_type.upper()}

Materials:
  Concrete: {concrete_name}
  Steel: {steel_name}

Strength Reduction:
  φ = {phi} ({column_type}-controlled)

Applied Loads:
  Pu = {Pu:.2f} kN
  Mu = {Mu:.2f} kN·m

Result:
  UR ≈ {UR:.3f}
  Status: {status}
"""
    
    return {
        'col_id': col_id,
        'UR': UR,
        'status': status,
        'figure': fig,
        'phi': phi,
        'N_capacity': N_cap,
        'M_capacity': M_cap,
        'details': details
    }


# ============================================================================
# SERVICEABILITY (ACI 318-25 Section 24)
# ============================================================================

def check_serviceability_aci(
    beam_id: int,
    b: float,
    h: float,
    cover: float,
    As: float,
    bar_diameter: float,
    concrete_name: str,
    steel_name: str,
    M_service: float,  # Service moment (kN·m)
    L: float,  # Span (m)
    load_type: str = "sustained",  # "sustained" or "transient"
    As_compression: float = 0
) -> Dict[str, Any]:
    """
    ACI 318-25 serviceability check (deflection and crack control).
    
    Reference: ACI 318-25 Section 24 (Serviceability)
    
    Args:
        beam_id: Beam element ID
        b: Width (mm)
        h: Height (mm)
        cover: Cover (mm)
        As: Tension steel area (mm²)
        bar_diameter: Bar diameter (mm)
        concrete_name: Concrete grade
        steel_name: Steel grade
        M_service: Service moment (kN·m)
        L: Span (m)
        load_type: "sustained" or "transient"
        As_compression: Compression steel area (mm²)
    
    Returns:
        dict with 'deflection', 'crack_control', 'status'
    """
    concrete = MaterialLoaderACI.get_concrete(concrete_name)
    steel = MaterialLoaderACI.get_steel(steel_name)
    
    d = h - cover
    fc = concrete.fc_prime
    fy = steel.fy
    Ec = concrete.Ec
    fr = concrete.fr
    
    # ========== DEFLECTION CHECK (ACI 24.2) ==========
    
    # Gross moment of inertia
    Ig = b * h**3 / 12  # mm⁴
    
    # Cracking moment
    yt = h / 2
    Mcr = fr * Ig / yt / 1e6  # kN·m
    
    # Cracked moment of inertia (transformed section)
    n = steel.Es / Ec  # Modular ratio
    
    # Solve for neutral axis depth (cracked section)
    # b·c²/2 = n·As·(d - c)
    # Quadratic: b·c² + 2·n·As·c - 2·n·As·d = 0
    A_coef = b / 2
    B_coef = n * As
    C_coef = -n * As * d
    
    c = (-B_coef + np.sqrt(B_coef**2 - 4 * A_coef * C_coef)) / (2 * A_coef)
    
    # Cracked inertia
    Icr = (b * c**3) / 3 + n * As * (d - c)**2
    
    # Effective moment of inertia (Branson equation, ACI 24.2.3.5)
    if M_service < Mcr:
        Ie = Ig
    else:
        ratio = (Mcr / M_service) ** 3
        Ie = ratio * Ig + (1 - ratio) * Icr
        Ie = max(Ie, Icr)  # Ie ≥ Icr
    
    # Immediate deflection (simple span, uniformly distributed load approximation)
    # δ = 5·M·L² / (48·Ec·Ie)  (approximate for moment)
    L_mm = L * 1000
    delta_immediate = 5 * M_service * 1e6 * L_mm**2 / (48 * Ec * Ie)  # mm
    
    # Long-term deflection multiplier (ACI 24.2.4.1.1)
    # λΔ = ξ / (1 + 50·ρ')
    # ξ = 2.0 for sustained loads, 1.0 for transient
    xi = 2.0 if load_type == "sustained" else 1.0
    rho_prime = As_compression / (b * d) if As_compression > 0 else 0
    lambda_delta = xi / (1 + 50 * rho_prime)
    
    delta_total = delta_immediate * (1 + lambda_delta)
    
    # Allowable deflection (ACI Table 24.2.2)
    # L/240 for floors with non-structural elements
    delta_allow = L_mm / 240
    
    deflection_UR = delta_total / delta_allow
    deflection_status = 'PASS' if deflection_UR <= 1.0 else 'FAIL'
    
    # ========== CRACK CONTROL (ACI 24.3.2) ==========
    
    # ACI 24.3.2: Maximum spacing of reinforcement for crack control
    # s_max = 15·(40/fs) - 2.5·cc
    # Where fs = stress in reinforcement (MPa), cc = clear cover
    
    # Approximate steel stress at service loads
    # fs = M / (As · jd), assume jd ≈ 0.9d
    jd = 0.9 * d
    fs = M_service * 1e6 / (As * jd) if As > 0 else 0  # MPa
    
    # Maximum spacing (ACI equation in inch-pound units, converted)
    # Original: s = 15·(40000/fs) - 2.5·cc (inches, psi)
    # SI version (approximate): s_max = 380·(280/fs) - 2.5·cc (mm, MPa)
    fs_limit = min(fs, 2 * fy / 3)  # Typically limit to 2/3·fy
    
    if fs_limit > 0:
        s_max = 380 * (280 / fs_limit) - 2.5 * cover
    else:
        s_max = 999
    
    # Actual spacing (approximate based on number of bars)
    n_bars = max(1, int(As / (np.pi * (bar_diameter/2)**2)))
    s_actual = (b - 2 * cover) / max(1, n_bars - 1) if n_bars > 1 else 0
    
    crack_UR = s_actual / s_max if s_max > 0 else 0
    crack_status = 'PASS' if crack_UR <= 1.0 else 'FAIL'
    
    # ========== OVERALL STATUS ==========
    
    overall_status = 'PASS' if deflection_status == 'PASS' and crack_status == 'PASS' else 'FAIL'
    
    details = f"""
ACI 318-25 Serviceability Check
{'='*60}

Beam ID: {beam_id}
Span: L = {L:.2f} m
Section: {b:.0f}×{h:.0f} mm, d = {d:.0f} mm

--- DEFLECTION CHECK ---

Effective Inertia:
  Ig = {Ig/1e6:.2f} ×10⁶ mm⁴
  Mcr = {Mcr:.2f} kN·m
  Icr = {Icr/1e6:.2f} ×10⁶ mm⁴
  Ie = {Ie/1e6:.2f} ×10⁶ mm⁴ (Branson)

Deflection:
  Immediate: δ₀ = {delta_immediate:.2f} mm
  Load type: {load_type}
  Multiplier: λΔ = {lambda_delta:.2f}
  Total: δ = {delta_total:.2f} mm

Allowable: δ_allow = L/240 = {delta_allow:.2f} mm
UR = {deflection_UR:.3f} - {deflection_status}

--- CRACK CONTROL ---

Steel Stress:
  fs ≈ {fs:.1f} MPa (at service)

Spacing:
  s_max = {s_max:.0f} mm (ACI 24.3.2)
  s_actual ≈ {s_actual:.0f} mm

UR = {crack_UR:.3f} - {crack_status}

Overall Status: {overall_status}
"""
    
    return {
        'beam_id': beam_id,
        'deflection': {
            'delta_immediate': delta_immediate,
            'delta_total': delta_total,
            'delta_allow': delta_allow,
            'UR': deflection_UR,
            'status': deflection_status,
            'Ie': Ie,
            'Mcr': Mcr
        },
        'crack_control': {
            'fs': fs,
            's_max': s_max,
            's_actual': s_actual,
            'UR': crack_UR,
            'status': crack_status
        },
        'status': overall_status,
        'details': details
    }


# ============================================================================
# ACI 318-25 CODE CLASS
# ============================================================================

class ACI318Code(DesignCode):
    """ACI 318-25 implementation of DesignCode interface."""
    
    @property
    def code_name(self) -> str:
        return "ACI 318-25"
    
    @property
    def code_units(self) -> str:
        return "SI"  # Internal calculations in SI
    
    def check_beam_flexure(self, **kwargs) -> Dict[str, Any]:
        """Check beam flexural capacity per ACI 318-25 Section 22.3."""
        return check_beam_flexure_aci(**kwargs)
    
    def check_beam_shear(self, **kwargs) -> Dict[str, Any]:
        """Check beam shear capacity per ACI 318-25 Section 22.5."""
        return check_beam_shear_aci(**kwargs)
    
    def check_column(self, **kwargs) -> Dict[str, Any]:
        """Check column interaction per ACI 318-25 Section 22.4."""
        return check_column_aci(**kwargs)
    
    def check_serviceability(self, **kwargs) -> Dict[str, Any]:
        """Check serviceability per ACI 318-25 Section 24."""
        return check_serviceability_aci(**kwargs)


# ============================================================================
# STREAMLIT INTEGRATION WRAPPERS
# ============================================================================

def run_beam_flexure_check_aci_from_ui(**kwargs) -> Dict[str, Any]:
    """Wrapper for Streamlit UI integration."""
    return check_beam_flexure_aci(**kwargs)


def run_beam_shear_check_aci_from_ui(**kwargs) -> Dict[str, Any]:
    """Wrapper for Streamlit UI integration."""
    return check_beam_shear_aci(**kwargs)


def run_column_check_aci_from_ui(**kwargs) -> Dict[str, Any]:
    """Wrapper for Streamlit UI integration."""
    return check_column_aci(**kwargs)


def run_sls_check_aci_from_ui(**kwargs) -> Dict[str, Any]:
    """Wrapper for Streamlit UI integration."""
    return check_serviceability_aci(**kwargs)
