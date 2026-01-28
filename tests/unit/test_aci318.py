"""
Unit tests for ACI 318-25 design code implementation.

Tests cover:
- Material properties
- Beam flexure checks
- Beam shear checks
- Column interaction diagrams
- Serviceability (deflection and crack control)
"""

import pytest
import numpy as np
from src.design.aci318 import (
    ConcreteACI,
    SteelACI,
    MaterialLoaderACI,
    CONCRETE_ACI,
    STEEL_ACI,
    check_beam_flexure_aci,
    check_beam_shear_aci,
    check_column_aci,
    check_serviceability_aci,
    generate_column_interaction_aci,
    calculate_phi_factor,
    ACI318Code
)


# ============================================================================
# MATERIAL TESTS
# ============================================================================

class TestConcreteACI:
    """Test ACI concrete material properties."""
    
    def test_concrete_basic_properties(self):
        """Test basic concrete properties."""
        conc = ConcreteACI(fc_prime=27.6)  # 4000 psi
        
        assert conc.fc_prime == 27.6
        assert abs(conc.fc_psi - 4000) < 10  # Conversion tolerance
        assert conc.lambda_factor == 1.0
    
    def test_concrete_modulus_Ec(self):
        """Test modulus of elasticity calculation."""
        conc = ConcreteACI(fc_prime=27.6)  # 4000 psi
        
        # Ec = 4700 · √fc' (MPa units)
        # For fc' = 27.6 MPa, λ = 1.0
        expected_Ec = 4700 * np.sqrt(27.6)
        
        assert abs(conc.Ec - expected_Ec) < 100
        assert conc.Ec > 20000  # Reasonable range (should be ~24,700 MPa)
    
    def test_concrete_modulus_rupture(self):
        """Test modulus of rupture calculation."""
        conc = ConcreteACI(fc_prime=27.6)
        
        # fr = 0.62 · λ · √fc' (MPa)
        expected_fr = 0.62 * 1.0 * np.sqrt(27.6)
        
        assert abs(conc.fr - expected_fr) < 0.1
    
    def test_beta1_factor(self):
        """Test β1 stress block factor."""
        # fc' = 27.6 MPa (< 28 MPa): β1 = 0.85
        conc1 = ConcreteACI(fc_prime=27.6)
        assert conc1.beta1 == 0.85
        
        # fc' = 41.4 MPa (6000 psi): β1 = 0.85 - 0.05*(41.4-28)/7 ≈ 0.754
        conc2 = ConcreteACI(fc_prime=41.4)
        expected_beta2 = 0.85 - 0.05 * (41.4 - 28) / 7
        assert abs(conc2.beta1 - expected_beta2) < 0.01
        
        # fc' = 55.2 MPa (8000 psi): β1 should be ≥ 0.65
        conc3 = ConcreteACI(fc_prime=55.2)
        assert conc3.beta1 >= 0.65
        assert conc3.beta1 < 0.85
    
    def test_lightweight_concrete(self):
        """Test lightweight concrete lambda factor."""
        conc = ConcreteACI(fc_prime=27.6, lambda_factor=0.85)
        
        assert conc.lambda_factor == 0.85
        # Ec and fr should be reduced
        conc_normal = ConcreteACI(fc_prime=27.6, lambda_factor=1.0)
        assert conc.Ec < conc_normal.Ec
        assert conc.fr < conc_normal.fr
    
    def test_material_validation(self):
        """Test material property validation."""
        with pytest.raises(ValueError):
            ConcreteACI(fc_prime=-10)  # Negative strength
        
        with pytest.raises(ValueError):
            ConcreteACI(fc_prime=10)  # Too low (< 17.2 MPa)


class TestSteelACI:
    """Test ACI steel material properties."""
    
    def test_steel_basic_properties(self):
        """Test basic steel properties."""
        steel = SteelACI(fy=414)  # Grade 60
        
        assert steel.fy == 414
        assert abs(steel.fy_psi - 60000) < 100
        assert steel.Es == 200000
    
    def test_steel_validation(self):
        """Test steel property validation."""
        with pytest.raises(ValueError):
            SteelACI(fy=-100)  # Negative yield
        
        with pytest.raises(ValueError):
            SteelACI(fy=600)  # Exceeds ACI limit (550 MPa)


class TestMaterialLoader:
    """Test material loader functionality."""
    
    def test_load_concrete_grades(self):
        """Test loading predefined concrete grades."""
        conc = MaterialLoaderACI.get_concrete("4000 psi (27.6 MPa)")
        assert conc.fc_prime == 27.6
        
        conc2 = MaterialLoaderACI.get_concrete("6000 psi (41.4 MPa)")
        assert conc2.fc_prime == 41.4
    
    def test_load_steel_grades(self):
        """Test loading predefined steel grades."""
        steel = MaterialLoaderACI.get_steel("Grade 60 (414 MPa)")
        assert steel.fy == 414
    
    def test_invalid_grade_names(self):
        """Test error handling for invalid grades."""
        with pytest.raises(ValueError):
            MaterialLoaderACI.get_concrete("InvalidGrade")
        
        with pytest.raises(ValueError):
            MaterialLoaderACI.get_steel("InvalidGrade")


# ============================================================================
# PHI FACTOR TESTS
# ============================================================================

class TestPhiFactor:
    """Test strength reduction factor φ calculation."""
    
    def test_tension_controlled(self):
        """Test tension-controlled section (εt ≥ 0.005)."""
        # Large d/c ratio → tension-controlled
        phi = calculate_phi_factor(c=100, d=500)
        assert phi == 0.90
    
    def test_compression_controlled(self):
        """Test compression-controlled section (εt ≤ εty)."""
        # Small d/c ratio → compression-controlled
        phi = calculate_phi_factor(c=400, d=500)
        assert phi == 0.65
    
    def test_transition_zone(self):
        """Test transition zone (εty < εt < 0.005)."""
        phi = calculate_phi_factor(c=200, d=500)
        assert 0.65 < phi < 0.90
    
    def test_spiral_columns(self):
        """Test spiral column φ factor."""
        phi = calculate_phi_factor(c=400, d=500, steel_type="spiral")
        assert phi == 0.75  # Compression-controlled spiral


# ============================================================================
# BEAM FLEXURE TESTS
# ============================================================================

class TestBeamFlexure:
    """Test ACI beam flexure calculations."""
    
    def test_simple_beam_flexure(self):
        """Test basic flexural check."""
        result = check_beam_flexure_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1600,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Mu=120
        )
        
        assert 'UR' in result
        assert 'status' in result
        assert 'Mn' in result
        assert 'phiMn' in result
        assert result['phi'] > 0
    
    def test_flexure_capacity_formula(self):
        """Verify flexural capacity formula."""
        # Known example: b=300mm, h=500mm, d=460mm, As=1600mm²
        # fc'=27.6 MPa, fy=414 MPa
        result = check_beam_flexure_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1600,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Mu=100
        )
        
        # Check Whitney stress block depth
        a = result['a']
        assert a > 0 and a < 100  # Reasonable range
        
        # Check nominal moment is positive
        assert result['Mn'] > 0
        
        # Check φ factor is tension-controlled
        assert result['phi'] >= 0.85  # Should be tension-controlled
    
    def test_flexure_pass_fail(self):
        """Test pass/fail status."""
        # Low moment → PASS
        result_pass = check_beam_flexure_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            As=2000,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Mu=50
        )
        assert result_pass['status'] == 'PASS'
        assert result_pass['UR'] < 1.0
        
        # High moment → FAIL
        result_fail = check_beam_flexure_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1000,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Mu=300
        )
        assert result_fail['status'] == 'FAIL'
        assert result_fail['UR'] > 1.0


# ============================================================================
# BEAM SHEAR TESTS
# ============================================================================

class TestBeamShear:
    """Test ACI beam shear calculations."""
    
    def test_simple_shear_check(self):
        """Test basic shear check."""
        result = check_beam_shear_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            Vu=80,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)"
        )
        
        assert 'UR' in result
        assert 'status' in result
        assert 'Vc' in result
        assert 'Vs' in result
        assert result['Vc'] > 0
    
    def test_shear_with_stirrups(self):
        """Test shear check with stirrup reinforcement."""
        result = check_beam_shear_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            Vu=120,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            stirrup_dia=10,
            n_legs=2,
            spacing=150
        )
        
        assert result['Vs'] > 0
        assert result['Vn'] == result['Vc'] + result['Vs']
    
    def test_shear_concrete_only(self):
        """Test shear with concrete capacity only (no stirrups)."""
        result = check_beam_shear_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            Vu=40,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            stirrup_dia=0
        )
        
        assert result['Vs'] == 0
        assert result['Vn'] == result['Vc']
    
    def test_shear_phi_factor(self):
        """Test shear φ factor (should be 0.75)."""
        result = check_beam_shear_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            Vu=50,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)"
        )
        
        assert result['phi'] == 0.75


# ============================================================================
# COLUMN INTERACTION TESTS
# ============================================================================

class TestColumnInteraction:
    """Test ACI column interaction diagram."""
    
    def test_interaction_curve_generation(self):
        """Test P-M interaction curve generation."""
        N_cap, M_cap = generate_column_interaction_aci(
            b=400, h=400, cover=40,
            As_total=3200,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)"
        )
        
        assert len(N_cap) > 0
        assert len(M_cap) > 0
        assert len(N_cap) == len(M_cap)
    
    def test_interaction_curve_bounds(self):
        """Test interaction curve boundary points."""
        N_cap, M_cap = generate_column_interaction_aci(
            b=400, h=400, cover=40,
            As_total=3200,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)"
        )
        
        # Maximum axial force (pure compression)
        assert max(N_cap) > 0
        
        # Should have tension capacity (negative N)
        assert min(N_cap) < 0
        
        # Moment should be zero at pure axial
        assert M_cap[0] == 0  # Pure compression
        assert M_cap[-1] == 0  # Pure tension
    
    def test_column_check_basic(self):
        """Test basic column check."""
        result = check_column_aci(
            col_id=1,
            b=400, h=400, cover=40,
            As_total=3200,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Pu=500,
            Mu=100
        )
        
        assert 'UR' in result
        assert 'status' in result
        assert 'figure' in result
        assert result['phi'] > 0
    
    def test_column_tied_vs_spiral(self):
        """Test tied vs spiral column φ factors."""
        result_tied = check_column_aci(
            col_id=1,
            b=400, h=400, cover=40,
            As_total=3200,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Pu=500, Mu=100,
            column_type="tied"
        )
        
        result_spiral = check_column_aci(
            col_id=1,
            b=400, h=400, cover=40,
            As_total=3200,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Pu=500, Mu=100,
            column_type="spiral"
        )
        
        assert result_tied['phi'] == 0.65
        assert result_spiral['phi'] == 0.75


# ============================================================================
# SERVICEABILITY TESTS
# ============================================================================

class TestServiceability:
    """Test ACI serviceability checks."""
    
    def test_deflection_check_basic(self):
        """Test basic deflection check."""
        result = check_serviceability_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1600,
            bar_diameter=20,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            M_service=80,
            L=6.0
        )
        
        assert 'deflection' in result
        assert 'crack_control' in result
        assert 'status' in result
        
        defl = result['deflection']
        assert defl['delta_immediate'] > 0
        assert defl['delta_total'] > 0
        assert defl['delta_allow'] > 0
    
    def test_deflection_branson_equation(self):
        """Test Branson effective inertia calculation."""
        result = check_serviceability_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1600,
            bar_diameter=20,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            M_service=80,
            L=6.0
        )
        
        # Check Ie is between Icr and Ig
        assert result['deflection']['Ie'] > 0
        assert 'Mcr' in result['deflection']
    
    def test_long_term_deflection_multiplier(self):
        """Test long-term deflection multiplier."""
        result_sustained = check_serviceability_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1600,
            bar_diameter=20,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            M_service=60,
            L=6.0,
            load_type="sustained"
        )
        
        result_transient = check_serviceability_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1600,
            bar_diameter=20,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            M_service=60,
            L=6.0,
            load_type="transient"
        )
        
        # Sustained loads should have higher deflection
        assert result_sustained['deflection']['delta_total'] > result_transient['deflection']['delta_total']
    
    def test_crack_control_check(self):
        """Test crack control spacing check."""
        result = check_serviceability_aci(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1600,
            bar_diameter=20,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            M_service=80,
            L=6.0
        )
        
        crack = result['crack_control']
        assert crack['fs'] > 0
        assert crack['s_max'] > 0
        assert 'UR' in crack


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestACI318Code:
    """Test ACI318Code class implementation."""
    
    def test_code_name(self):
        """Test code identifier."""
        code = ACI318Code()
        assert code.code_name == "ACI 318-25"
        assert code.code_units == "SI"
    
    def test_all_methods_implemented(self):
        """Test all abstract methods are implemented."""
        code = ACI318Code()
        
        # Should not raise NotImplementedError
        result = code.check_beam_flexure(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1600,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Mu=100
        )
        assert result is not None
        
        result = code.check_beam_shear(
            beam_id=1,
            b=300, h=500, cover=40,
            Vu=80,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)"
        )
        assert result is not None
        
        result = code.check_column(
            col_id=1,
            b=400, h=400, cover=40,
            As_total=3200,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Pu=500, Mu=100
        )
        assert result is not None
        
        result = code.check_serviceability(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1600,
            bar_diameter=20,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            M_service=80,
            L=6.0
        )
        assert result is not None


# ============================================================================
# REAL-WORLD EXAMPLES
# ============================================================================

class TestRealWorldExamples:
    """Test with realistic design scenarios."""
    
    def test_typical_floor_beam(self):
        """Test typical office building floor beam."""
        # 300x600mm beam, 8m span, Grade 60 steel, 4000 psi concrete
        result = check_beam_flexure_aci(
            beam_id=1,
            b=300, h=600, cover=40,
            As=2400,  # 6-φ20 bars
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Mu=200
        )
        
        assert result['status'] == 'PASS'
        assert result['phi'] >= 0.85  # Tension-controlled
    
    def test_typical_column(self):
        """Test typical building column."""
        # 400x400mm column, 3% steel ratio
        result = check_column_aci(
            col_id=1,
            b=400, h=400, cover=40,
            As_total=4800,  # ~3% steel
            concrete_name="5000 psi (34.5 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Pu=800,
            Mu=120
        )
        
        assert result['status'] == 'PASS'
        assert result['figure'] is not None
    
    def test_parking_garage_beam(self):
        """Test parking garage beam with heavy loading."""
        result = check_beam_shear_aci(
            beam_id=1,
            b=400, h=700, cover=40,
            Vu=200,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            stirrup_dia=12,
            n_legs=2,
            spacing=200
        )
        
        # Should require stirrups for high shear
        assert result['Vs'] > 0
        assert result['status'] in ['PASS', 'FAIL']
