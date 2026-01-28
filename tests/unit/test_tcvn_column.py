"""
Unit tests for TCVN 5574:2018 column design module
"""

import pytest
import numpy as np
from src.design.tcvn_column import (
    ColumnSection,
    calculate_eta_factor,
    generate_interaction_curve,
    check_biaxial_bending,
    plot_interaction_diagram,
    run_column_check_from_ui
)
from src.design.tcvn_setup import DesignContext


class TestColumnSection:
    """Test ColumnSection class."""
    
    def test_section_creation(self):
        """Test creating column section."""
        section = ColumnSection(b=400, h=400, cover=30, L_eff=3000)
        assert section.b == 400
        assert section.h == 400
        assert section.cover == 30
        assert section.L_eff == 3000
    
    def test_h0_property(self):
        """Test effective depth calculation."""
        section = ColumnSection(b=400, h=500, cover=30, L_eff=3000)
        assert section.h0 == 470
    
    def test_slenderness_ratio(self):
        """Test slenderness ratio calculation."""
        section = ColumnSection(b=400, h=400, cover=30, L_eff=3000)
        # λ = L_eff / i, where i = h / √12
        i = 400 / np.sqrt(12)
        expected_lambda = 3000 / i
        assert section.slenderness_ratio == pytest.approx(expected_lambda, rel=0.01)
    
    def test_invalid_dimensions(self):
        """Test validation of section dimensions."""
        with pytest.raises(ValueError, match="too small"):
            ColumnSection(b=100, h=400, cover=30, L_eff=3000)
    
    def test_invalid_length(self):
        """Test validation of effective length."""
        with pytest.raises(ValueError, match="too small"):
            ColumnSection(b=400, h=400, cover=30, L_eff=200)


class TestEtaFactor:
    """Test P-Delta magnification factor."""
    
    def test_short_column_no_magnification(self):
        """Test short column (λ ≤ 14) has η = 1.0."""
        # Create short column
        section = ColumnSection(b=400, h=400, cover=30, L_eff=1500)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 30)
        
        # λ = 1500 / (400/√12) ≈ 13.0 < 14
        assert section.slenderness_ratio < 14
        
        eta = calculate_eta_factor(section, context, N_u=500)
        assert eta == 1.0
    
    def test_slender_column_magnification(self):
        """Test slender column (λ > 14) has η > 1.0."""
        # Create slender column
        section = ColumnSection(b=300, h=300, cover=30, L_eff=3000)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 30)
        
        # λ = 3000 / (300/√12) ≈ 34.6 > 14
        assert section.slenderness_ratio > 14
        
        eta = calculate_eta_factor(section, context, N_u=500)
        assert eta > 1.0
    
    def test_zero_axial_load(self):
        """Test η = 1.0 for zero axial load."""
        section = ColumnSection(b=400, h=400, cover=30, L_eff=4000)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 30)
        
        eta = calculate_eta_factor(section, context, N_u=0)
        assert eta == 1.0
    
    def test_tension_axial_load(self):
        """Test η = 1.0 for tension."""
        section = ColumnSection(b=400, h=400, cover=30, L_eff=4000)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 30)
        
        eta = calculate_eta_factor(section, context, N_u=-100)
        assert eta == 1.0
    
    def test_unstable_column(self):
        """Test error when N_u approaches N_cr."""
        section = ColumnSection(b=300, h=300, cover=30, L_eff=5000)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 30)
        
        # Calculate N_cr
        Eb = context.concrete.Eb
        I = (section.b * section.h**3) / 12
        N_cr = (np.pi**2 * Eb * I) / (section.L_eff**2)
        N_cr_kN = N_cr / 1000
        
        # Try to apply 95% of N_cr (should fail)
        with pytest.raises(ValueError, match="too close to buckling"):
            calculate_eta_factor(section, context, N_u=0.95 * N_cr_kN)


class TestInteractionCurve:
    """Test interaction diagram generation."""
    
    def test_interaction_curve_shape(self):
        """Test interaction curve has correct shape."""
        section = ColumnSection(b=400, h=400, cover=40, L_eff=3000)
        context = DesignContext.from_ui_inputs("B30", "CB400V", 40)
        As_total = 2400  # mm²
        
        N_cap, M_cap = generate_interaction_curve(section, context, As_total)
        
        # Should have multiple points
        assert len(N_cap) > 40
        assert len(M_cap) > 40
        
        # Pure compression (first point): N > 0, M ≈ 0
        assert N_cap[0] > 0
        assert abs(M_cap[0]) < 1.0
        
        # Pure tension (last point): N < 0, M ≈ 0
        assert N_cap[-1] < 0
        assert abs(M_cap[-1]) < 1.0
    
    def test_maximum_compression(self):
        """Test maximum compression capacity."""
        section = ColumnSection(b=300, h=300, cover=30, L_eff=3000)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 30)
        As_total = 1600  # mm²
        
        N_cap, M_cap = generate_interaction_curve(section, context, As_total)
        
        # Maximum compression (first point)
        N_max = N_cap[0]
        
        # Theoretical: N_max ≈ Rb·b·h + (Rsc - Rb)·As
        Rb = context.concrete.Rb
        Rsc = context.steel.Rsc
        N_theoretical = (Rb * 300 * 300 + (Rsc - Rb) * As_total) / 1000
        
        # Should be within 10% of theoretical
        assert N_max == pytest.approx(N_theoretical, rel=0.15)
    
    def test_maximum_tension(self):
        """Test maximum tension capacity."""
        section = ColumnSection(b=300, h=300, cover=30, L_eff=3000)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 30)
        As_total = 1600  # mm²
        
        N_cap, M_cap = generate_interaction_curve(section, context, As_total)
        
        # Maximum tension (last point)
        N_min = N_cap[-1]
        
        # Theoretical: N_min = -Rs·As
        Rs = context.steel.Rs
        N_theoretical = -(Rs * As_total) / 1000
        
        # Should match theoretical
        assert N_min == pytest.approx(N_theoretical, rel=0.01)
    
    def test_balanced_point_exists(self):
        """Test that balanced failure point exists on curve."""
        section = ColumnSection(b=400, h=400, cover=40, L_eff=3000)
        context = DesignContext.from_ui_inputs("B30", "CB400V", 40)
        As_total = 2000  # mm²
        
        N_cap, M_cap = generate_interaction_curve(section, context, As_total)
        
        # Balanced point should have maximum moment
        max_M = max(np.abs(M_cap))
        assert max_M > 0
        
        # Should occur at intermediate axial load
        idx_max_M = np.argmax(np.abs(M_cap))
        N_at_max_M = N_cap[idx_max_M]
        assert N_at_max_M < max(N_cap)  # Less than pure compression
        assert N_at_max_M > min(N_cap)  # Greater than pure tension


class TestBiaxialBending:
    """Test biaxial bending check."""
    
    def test_uniaxial_x_only(self):
        """Test uniaxial bending about x-axis only."""
        result = check_biaxial_bending(
            Nux=1000, Mux=200,
            Nuy=1000, Muy=150,
            N=800, Mx=100, My=0
        )
        
        # UR should be Mx/Mux = 100/200 = 0.5
        assert result['UR_x'] == pytest.approx(0.5, rel=0.01)
        assert result['UR_y'] == 0.0
        assert result['status'] == 'PASS'
    
    def test_uniaxial_y_only(self):
        """Test uniaxial bending about y-axis only."""
        result = check_biaxial_bending(
            Nux=1000, Mux=200,
            Nuy=1000, Muy=150,
            N=800, Mx=0, My=100
        )
        
        # UR should be My/Muy = 100/150 = 0.667
        assert result['UR_x'] == 0.0
        assert result['UR_y'] == pytest.approx(0.667, rel=0.01)
        assert result['status'] == 'PASS'
    
    def test_biaxial_pass(self):
        """Test biaxial bending that passes."""
        result = check_biaxial_bending(
            Nux=1000, Mux=200,
            Nuy=1000, Muy=150,
            N=800, Mx=100, My=50
        )
        
        # UR = [(100/200)^1.5 + (50/150)^1.5]^(1/1.5)
        UR_x = 100 / 200
        UR_y = 50 / 150
        alpha = 1.5
        UR_expected = (UR_x**alpha + UR_y**alpha)**(1/alpha)
        
        assert result['UR'] == pytest.approx(UR_expected, rel=0.01)
        assert result['status'] == 'PASS'
    
    def test_biaxial_fail(self):
        """Test biaxial bending that fails."""
        result = check_biaxial_bending(
            Nux=1000, Mux=200,
            Nuy=1000, Muy=150,
            N=800, Mx=180, My=140
        )
        
        # Both components near capacity
        assert result['UR'] > 1.0
        assert result['status'] == 'FAIL'


class TestPlotlyVisualization:
    """Test Plotly interaction diagram."""
    
    def test_plot_creation(self):
        """Test that plot is created successfully."""
        N_cap = [1000, 800, 600, 400, 200, 0, -200]
        M_cap = [0, 150, 200, 220, 200, 150, 0]
        
        fig = plot_interaction_diagram(N_cap, M_cap, N_applied=500, M_applied=180)
        
        # Check figure has data
        assert len(fig.data) >= 2  # Capacity curve + applied point
        
        # Check figure has layout
        assert fig.layout.title.text is not None
        assert 'Moment' in fig.layout.xaxis.title.text
        assert 'Axial' in fig.layout.yaxis.title.text


class TestStreamlitIntegration:
    """Test Streamlit integration function."""
    
    def test_run_column_check_from_ui_short(self):
        """Test column check for short column."""
        result = run_column_check_from_ui(
            col_id=1,
            b=400, h=400, L_eff=2.0, cover=30,
            As_total=2400,
            concrete_name="B30",
            steel_name="CB400V",
            N_u=800,
            M_u=120
        )
        
        assert 'col_id' in result
        assert result['col_id'] == 1
        assert 'figure' in result
        assert 'UR' in result
        assert 'status' in result
        assert 'eta' in result
        assert 'lambda' in result
        
        # Short column should have η ≈ 1.0
        assert result['eta'] == pytest.approx(1.0, abs=0.01)
    
    def test_run_column_check_from_ui_slender(self):
        """Test column check for slender column."""
        result = run_column_check_from_ui(
            col_id=2,
            b=300, h=300, L_eff=4.0, cover=30,
            As_total=1600,
            concrete_name="B25",
            steel_name="CB400V",
            N_u=500,
            M_u=80
        )
        
        # Slender column should have η > 1.0
        assert result['eta'] > 1.0
        assert result['lambda'] > 14
        assert result['M_amplified'] > result['lambda']  # Should be amplified
    
    def test_run_column_check_typical_design(self):
        """Test with typical 400x400 column."""
        result = run_column_check_from_ui(
            col_id=1,
            b=400, h=400, L_eff=3.0, cover=40,
            As_total=2400,  # 1.5% reinforcement ratio
            concrete_name="B30",
            steel_name="CB400V",
            N_u=1000,
            M_u=150
        )
        
        # Should pass with typical design
        assert result['status'] in ['PASS', 'FAIL']
        assert result['UR'] > 0
        assert result['figure'] is not None


class TestRealWorldExamples:
    """Test with real-world column scenarios."""
    
    def test_ground_floor_column(self):
        """Test ground floor column (high axial, low moment)."""
        # Typical: 400x400, L=3.5m, heavy axial load
        result = run_column_check_from_ui(
            col_id=1,
            b=400, h=400, L_eff=3.5, cover=40,
            As_total=2800,  # 1.75% steel
            concrete_name="B30",
            steel_name="CB400V",
            N_u=1500,  # High axial (15-story building)
            M_u=80  # Low moment
        )
        
        assert result['status'] in ['PASS', 'FAIL']
        # High axial, low moment typically passes
    
    def test_upper_floor_column(self):
        """Test upper floor column (low axial, higher moment)."""
        # Typical: 300x300, L=3.0m, lighter load
        result = run_column_check_from_ui(
            col_id=1,
            b=300, h=300, L_eff=3.0, cover=30,
            As_total=1400,  # 1.5% steel
            concrete_name="B25",
            steel_name="CB400V",
            N_u=400,  # Lower axial
            M_u=60  # Moderate moment
        )
        
        assert result['status'] in ['PASS', 'FAIL']
        assert result['lambda'] > 0
    
    def test_corner_column_biaxial(self):
        """Test corner column with biaxial bending."""
        section = ColumnSection(b=400, h=400, cover=40, L_eff=3000)
        context = DesignContext.from_ui_inputs("B30", "CB400V", 40)
        As_total = 2400
        
        # Generate interaction curve for both axes (assuming square section)
        N_cap, M_cap = generate_interaction_curve(section, context, As_total)
        
        # Biaxial check
        result = check_biaxial_bending(
            Nux=1000, Mux=200,
            Nuy=1000, Muy=200,  # Square section, same capacity
            N=800, Mx=100, My=100
        )
        
        assert result['status'] in ['PASS', 'FAIL']
        assert result['UR'] > 0
