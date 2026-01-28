"""
TCVN 5574:2018 Column Design Module

Implements:
- P-Delta second-order effects (η factor)
- N-M interaction diagrams
- Biaxial bending checks
- Plotly visualization

Reference: TCVN 5574:2018 Sections 7.3 & 8.1.2.4
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, field_validator
import plotly.graph_objects as go

from .tcvn_setup import DesignContext, create_design_context_from_streamlit


class ColumnSection(BaseModel):
    """Rectangular column section properties."""
    
    b: float  # Width (mm)
    h: float  # Height (mm)
    cover: float  # Concrete cover (mm)
    L_eff: float  # Effective length (mm)
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def h0(self) -> float:
        """Effective depth to centroid of tension steel."""
        return self.h - self.cover
    
    @property
    def slenderness_ratio(self) -> float:
        """
        Slenderness ratio λ = L_eff / i
        For rectangular section: i = h / √12
        """
        i = self.h / np.sqrt(12)
        return self.L_eff / i
    
    @field_validator('b', 'h')
    @classmethod
    def validate_dimensions(cls, v):
        if v < 150:
            raise ValueError(f"Column dimension {v} mm too small (min 150 mm)")
        if v > 1200:
            raise ValueError(f"Column dimension {v} mm excessive (max 1200 mm)")
        return v
    
    @field_validator('L_eff')
    @classmethod
    def validate_length(cls, v):
        if v < 500:
            raise ValueError(f"Effective length {v} mm too small (min 500 mm)")
        if v > 12000:
            raise ValueError(f"Effective length {v} mm excessive (max 12 m)")
        return v


def calculate_eta_factor(
    section: ColumnSection,
    context: DesignContext,
    N_u: float  # Ultimate axial force (kN)
) -> float:
    """
    Calculate moment magnification factor η for slender columns.
    
    TCVN 5574:2018 Section 7.3:
    - Short columns (λ ≤ 14): η = 1.0 (no second-order effects)
    - Slender columns (λ > 14): η = 1 / (1 - N_u / N_cr)
    
    Where:
        N_cr = π² · Eb · I / L_eff²  (Euler buckling load)
        Eb = Modulus of elasticity (MPa)
        I = Moment of inertia (mm⁴)
    
    Args:
        section: ColumnSection instance
        context: DesignContext with material properties
        N_u: Applied ultimate axial load (kN, positive = compression)
    
    Returns:
        eta: Magnification factor (≥ 1.0)
    
    Raises:
        ValueError: If N_u approaches N_cr (unstable)
    """
    lambda_ratio = section.slenderness_ratio
    
    # TCVN limit for short columns
    lambda_limit = 14  # For braced frames
    
    if lambda_ratio <= lambda_limit:
        return 1.0  # Short column, no magnification
    
    # Calculate critical buckling load
    Eb = context.concrete.Eb  # MPa
    I = (section.b * section.h**3) / 12  # mm⁴
    N_cr = (np.pi**2 * Eb * I) / (section.L_eff**2)  # N
    N_cr_kN = N_cr / 1000  # kN
    
    # Check stability
    if N_u >= 0.9 * N_cr_kN:
        raise ValueError(
            f"Applied load N_u = {N_u:.1f} kN too close to buckling load "
            f"N_cr = {N_cr_kN:.1f} kN. Column is unstable."
        )
    
    # Calculate magnification factor
    if N_u <= 0:
        # Tension or zero axial load
        return 1.0
    
    eta = 1.0 / (1.0 - N_u / N_cr_kN)
    
    return max(eta, 1.0)


def generate_interaction_curve(
    section: ColumnSection,
    context: DesignContext,
    As_total: float,  # Total longitudinal steel area (mm²)
    n_points: int = 50
) -> Tuple[List[float], List[float]]:
    """
    Generate P-M (N-M) interaction diagram for rectangular column.
    
    Algorithm (TCVN 5574:2018 Section 8.1.2.4):
    1. Point A: Pure compression (N_max, M=0)
    2. Points B-C: Vary neutral axis depth from h to 0.1h
    3. Point D: Pure tension (-N_max, M=0)
    
    For each neutral axis position c:
    - Assume strain distribution: ε_c at extreme fiber, linear to neutral axis
    - Calculate concrete stress block: rectangular stress block (Rb, β·c)
    - Calculate steel stress from strain
    - Integrate forces to get N and M
    
    Args:
        section: ColumnSection instance
        context: DesignContext with material properties
        As_total: Total longitudinal reinforcement (mm²)
        n_points: Number of points on curve
    
    Returns:
        (N_values, M_values): Lists of capacity points in (kN, kN·m)
    """
    Rb = context.concrete.Rb  # MPa
    Rs = context.steel.Rs  # MPa (tension)
    Rsc = context.steel.Rsc  # MPa (compression)
    Es = context.steel.Es  # MPa
    
    b = section.b
    h = section.h
    cover = section.cover
    h0 = h - cover
    
    # Strain limits
    epsilon_cu = 0.0035  # Ultimate compression strain (absolute value)
    epsilon_sy = Rs / Es  # Steel yield strain
    
    N_values = []
    M_values = []
    
    # Point A: Pure compression (entire section in compression)
    # Concrete contributes Rb·b·h, steel contributes (Rsc - Rb)·As_total
    Nc = Rb * b * h  # N
    Ns = (Rsc - Rb) * As_total  # Net steel contribution (N)
    N_max = (Nc + Ns) / 1000  # kN
    M_0 = 0  # No moment for pure compression
    N_values.append(N_max)
    M_values.append(M_0)
    
    # Points B-C: Iterate through neutral axis positions
    # c varies from h (full compression) to 0.1h (mostly tension)
    c_values = np.linspace(h, 0.1 * h, n_points)
    
    for c in c_values:
        # Extreme fiber compression strain
        epsilon_c = epsilon_cu
        
        # Strain at centroid of steel (assume single layer at h0 from compression face)
        # Strain distribution is linear: ε = ε_c · (c - x) / c
        epsilon_s = epsilon_c * (c - h0) / c
        
        # Concrete stress block (rectangular, TCVN Section 8.1.2.3)
        beta = 0.85  # Stress block depth factor
        a = beta * c  # Stress block depth (mm)
        
        # Concrete compression force
        Cc = Rb * b * a  # N
        
        # Steel stress (assume yielded if |ε_s| > ε_sy)
        if abs(epsilon_s) > epsilon_sy:
            # Yielded
            if epsilon_s > 0:
                # Tension
                fs = Rs  # MPa
            else:
                # Compression
                fs = -Rsc  # MPa (negative for compression)
        else:
            # Elastic
            fs = epsilon_s * Es  # MPa
        
        # Steel force (positive = tension)
        Cs = fs * As_total  # N
        
        # Total axial force (positive = compression)
        N = Cc - Cs  # N
        
        # Moment about section centroid (h/2 from compression face)
        # Concrete force acts at a/2 from compression face
        Mc = Cc * (h/2 - a/2)  # N·mm
        
        # Steel force acts at h0 from compression face
        Ms = Cs * (h0 - h/2)  # N·mm
        
        M = Mc + Ms  # N·mm
        
        # Convert to kN, kN·m
        N_values.append(N / 1000)
        M_values.append(M / 1e6)
    
    # Point D: Pure tension (entire section in tension)
    N_min = -(Rs * As_total) / 1000  # kN (negative = tension)
    M_0 = 0
    N_values.append(N_min)
    M_values.append(M_0)
    
    return N_values, M_values


def check_biaxial_bending(
    Nux: float, Mux: float,  # Uniaxial capacity about x-axis (kN, kN·m)
    Nuy: float, Muy: float,  # Uniaxial capacity about y-axis (kN, kN·m)
    N: float, Mx: float, My: float  # Applied loads (kN, kN·m)
) -> Dict[str, Any]:
    """
    Simplified biaxial bending check using contour method.
    
    Formula (TCVN 5574:2018 Section 8.1.2.4):
        (Mx / Mux)^α + (My / Muy)^α ≤ 1.0
    
    Where α = 1.5 (typical for columns)
    
    Args:
        Nux, Mux: Uniaxial capacity about x-axis at applied axial load
        Nuy, Muy: Uniaxial capacity about y-axis at applied axial load
        N, Mx, My: Applied loads
    
    Returns:
        {
            'UR': float,  # Combined utilization ratio
            'UR_x': float,  # x-axis component
            'UR_y': float,  # y-axis component
            'status': str  # 'PASS' or 'FAIL'
        }
    """
    # Exponent for interaction formula (typical range: 1.0-2.0)
    alpha = 1.5
    
    # Calculate utilization ratios for each axis
    UR_x = abs(Mx / Mux) if Mux > 0 else 0.0
    UR_y = abs(My / Muy) if Muy > 0 else 0.0
    
    # Combined utilization using contour method
    UR_combined = UR_x**alpha + UR_y**alpha
    UR = UR_combined**(1.0 / alpha)
    
    status = 'PASS' if UR <= 1.0 else 'FAIL'
    
    return {
        'UR': UR,
        'UR_x': UR_x,
        'UR_y': UR_y,
        'alpha': alpha,
        'status': status
    }


def plot_interaction_diagram(
    N_capacity: List[float],
    M_capacity: List[float],
    N_applied: float,
    M_applied: float,
    title: str = "Column Interaction Diagram"
) -> go.Figure:
    """
    Create interactive Plotly chart for P-M interaction diagram.
    
    Args:
        N_capacity: List of axial capacity values (kN)
        M_capacity: List of moment capacity values (kN·m)
        N_applied: Applied axial load (kN)
        M_applied: Applied moment (kN·m)
        title: Chart title
    
    Returns:
        Plotly Figure object ready for st.plotly_chart()
    """
    fig = go.Figure()
    
    # Capacity envelope (filled region)
    fig.add_trace(go.Scatter(
        x=M_capacity,
        y=N_capacity,
        mode='lines',
        name='Capacity Envelope',
        line=dict(color='blue', width=3),
        fill='toself',
        fillcolor='rgba(100, 150, 255, 0.2)',
        hovertemplate='<b>Capacity</b><br>' +
                      'M = %{x:.1f} kN·m<br>' +
                      'N = %{y:.1f} kN<br>' +
                      '<extra></extra>'
    ))
    
    # Applied load point
    fig.add_trace(go.Scatter(
        x=[M_applied],
        y=[N_applied],
        mode='markers',
        name='Applied Load',
        marker=dict(
            size=15,
            color='red',
            symbol='star',
            line=dict(color='darkred', width=2)
        ),
        hovertemplate='<b>Applied</b><br>' +
                      'M = %{x:.1f} kN·m<br>' +
                      'N = %{y:.1f} kN<br>' +
                      '<extra></extra>'
    ))
    
    # Layout configuration
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#333'}
        },
        xaxis_title='Moment M (kN·m)',
        yaxis_title='Axial Force N (kN)',
        hovermode='closest',
        width=750,
        height=650,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        )
    )
    
    return fig


def run_column_check_from_ui(
    col_id: int,
    b: float, h: float, L_eff: float, cover: float,
    As_total: float,
    concrete_name: str, steel_name: str,
    N_u: float, M_u: float  # Applied loads from analysis (kN, kN·m)
) -> Dict[str, Any]:
    """
    Wrapper function for Streamlit UI integration.
    
    Performs complete column design check including:
    1. P-Delta second-order effects (η factor)
    2. Interaction diagram generation
    3. Capacity check
    4. Plotly visualization
    
    Args:
        col_id: Column element ID
        b, h: Section dimensions (mm)
        L_eff: Effective length (m) - will convert to mm
        cover: Concrete cover (mm)
        As_total: Total longitudinal steel area (mm²)
        concrete_name: Concrete grade name (e.g., "B25")
        steel_name: Steel grade name (e.g., "CB400V")
        N_u: Applied ultimate axial load (kN, positive = compression)
        M_u: Applied ultimate moment (kN·m)
    
    Returns:
        {
            'col_id': int,
            'figure': go.Figure,  # Plotly interaction diagram
            'UR': float,  # Utilization ratio
            'status': str,  # 'PASS' or 'FAIL'
            'eta': float,  # Slenderness magnification factor
            'lambda': float,  # Slenderness ratio
            'M_amplified': float,  # Amplified moment (kN·m)
            'details': str  # Calculation narrative
        }
    """
    # Create section and context
    section = ColumnSection(
        b=b,
        h=h,
        cover=cover,
        L_eff=L_eff * 1000  # Convert m to mm
    )
    
    context = create_design_context_from_streamlit(
        concrete_name,
        steel_name,
        cover
    )
    
    # Calculate P-Delta magnification factor
    lambda_ratio = section.slenderness_ratio
    
    try:
        eta = calculate_eta_factor(section, context, N_u)
    except ValueError as e:
        # Column unstable
        return {
            'col_id': col_id,
            'figure': None,
            'UR': 999.0,
            'status': 'FAIL',
            'eta': float('inf'),
            'lambda': lambda_ratio,
            'M_amplified': M_u,
            'details': f"ERROR: {str(e)}"
        }
    
    # Amplify moment for slenderness
    M_u_amplified = M_u * eta
    
    # Generate interaction curve
    N_cap, M_cap = generate_interaction_curve(section, context, As_total)
    
    # Check if applied load is within capacity envelope
    # Method: Find interpolated capacity at applied N_u
    N_cap_array = np.array(N_cap)
    M_cap_array = np.array(M_cap)
    
    # Find capacity moment at applied axial load
    if N_u <= min(N_cap_array) or N_u >= max(N_cap_array):
        # Outside axial load range
        UR = 999.0
        status = 'FAIL'
        M_cap_at_Nu = 0
    else:
        # Interpolate capacity at N_u
        M_cap_at_Nu = np.interp(N_u, N_cap_array, np.abs(M_cap_array))
        
        # Calculate utilization ratio
        if M_cap_at_Nu > 0:
            UR = abs(M_u_amplified) / M_cap_at_Nu
            status = 'PASS' if UR <= 1.0 else 'FAIL'
        else:
            UR = 999.0
            status = 'FAIL'
    
    # Create interaction diagram
    fig = plot_interaction_diagram(
        N_cap, M_cap,
        N_u, M_u_amplified,
        title=f"Column {col_id} Interaction Diagram"
    )
    
    # Detailed calculation narrative
    details = f"""
Column Design Check (TCVN 5574:2018)
{'='*50}

Section Properties:
  b × h = {b:.0f} × {h:.0f} mm
  Effective length L_eff = {L_eff:.2f} m
  Cover = {cover:.0f} mm
  Total steel As = {As_total:.0f} mm²

Materials:
  Concrete: {concrete_name} (Rb = {context.concrete.Rb:.1f} MPa)
  Steel: {steel_name} (Rs = {context.steel.Rs:.0f} MPa)

Slenderness Check:
  λ = L_eff / i = {lambda_ratio:.1f}
  Classification: {'SLENDER' if lambda_ratio > 14 else 'SHORT'} column

Second-Order Effects:
  Magnification factor η = {eta:.3f}
  Original moment M_u = {M_u:.2f} kN·m
  Amplified moment M_amplified = {M_u_amplified:.2f} kN·m

Applied Loads:
  Axial force N_u = {N_u:.2f} kN
  Moment M_u = {M_u_amplified:.2f} kN·m (amplified)

Capacity:
  Moment capacity at N_u = {M_cap_at_Nu:.2f} kN·m
  
Result:
  Utilization Ratio UR = {UR:.3f}
  Status: {status}
"""
    
    return {
        'col_id': col_id,
        'figure': fig,
        'UR': UR,
        'status': status,
        'eta': eta,
        'lambda': lambda_ratio,
        'M_amplified': M_u_amplified,
        'M_cap_at_Nu': M_cap_at_Nu if M_cap_at_Nu > 0 else 0,
        'details': details
    }
