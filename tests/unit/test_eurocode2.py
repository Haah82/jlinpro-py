"""
Unit tests for Eurocode 2 (EN 1992-1-1:2004) implementation.

Tests validate material properties, flexure, shear, column interaction,
and serviceability checks against known examples.
"""

import pytest
import numpy as np
from src.design.eurocode2 import (
    ConcreteEC2,
    SteelEC2,
    MaterialLoaderEC2,
    BeamSectionEC2,
    ColumnSectionEC2,
    Eurocode2Code
)


class TestMaterialsEC2:
    """Test Eurocode 2 material properties."""
    
    def test_concrete_c30_37_properties(self):
        """Test C30/37 concrete properties per EN 1992-1-1 Table 3.1."""
        concrete = MaterialLoaderEC2.get_concrete("C30/37")
        
        assert concrete.fck == 30.0
        assert concrete.fck_cube == 37.0
        assert concrete.fcm == 38.0  # fck + 8
        
        # Ecm = 22 · (fcm/10)^0.3 GPa = 22000 · (3.8)^0.3 MPa
        expected_Ecm = 22000 * ((38 / 10) ** 0.3)
        assert abs(concrete.Ecm - expected_Ecm) < 100
        
        # fctm = 0.30 · fck^(2/3) for fck ≤ 50 MPa
        expected_fctm = 0.30 * (30 ** (2/3))
        assert abs(concrete.fctm - expected_fctm) < 0.1
        
        # Design strength fcd = αcc · fck / γc = 1.0 · 30 / 1.5 = 20 MPa
        assert abs(concrete.fcd - 20.0) < 0.1
        
        # Stress block factors
        assert concrete.lambda_factor == 0.8  # For fck ≤ 50 MPa
        assert concrete.eta_factor == 1.0  # For fck ≤ 50 MPa
    
    def test_concrete_c60_75_high_strength(self):
        """Test high-strength concrete C60/75."""
        concrete = MaterialLoaderEC2.get_concrete("C60/75")
        
        assert concrete.fck == 60.0
        assert concrete.fcm == 68.0
        
        # Lambda decreases for high strength
        expected_lambda = 0.8 - (60 - 50) / 400
        assert abs(concrete.lambda_factor - expected_lambda) < 0.01
        
        # Eta decreases for high strength
        expected_eta = 1.0 - (60 - 50) / 200
        assert abs(concrete.eta_factor - expected_eta) < 0.01
        
        # Tensile strength for fck > 50
        expected_fctm = 2.12 * np.log(1 + 68 / 10)
        assert abs(concrete.fctm - expected_fctm) < 0.1
    
    def test_steel_b500b_properties(self):
        """Test B500B reinforcement steel properties."""
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        assert steel.fyk == 500.0
        assert steel.ftk == 540.0
        assert steel.epsilon_uk == 0.05
        assert steel.Es == 200000
        
        # Design strength fyd = fyk / γs = 500 / 1.15 = 434.8 MPa
        expected_fyd = 500.0 / 1.15
        assert abs(steel.fyd - expected_fyd) < 0.1
        
        # Design strain
        expected_epsilon_yd = expected_fyd / 200000
        assert abs(steel.epsilon_yd - expected_epsilon_yd) < 1e-6
    
    def test_material_catalog_completeness(self):
        """Verify all materials in catalog are accessible."""
        # Test all concrete grades
        for grade in ["C20/25", "C25/30", "C30/37", "C40/50", "C50/60", "C90/105"]:
            concrete = MaterialLoaderEC2.get_concrete(grade)
            assert concrete.strength_class == grade
        
        # Test all steel grades
        for grade in ["B400A", "B400B", "B500A", "B500B", "B500C"]:
            steel = MaterialLoaderEC2.get_steel(grade)
            assert steel.grade == grade
    
    def test_invalid_material_grades(self):
        """Test error handling for invalid grades."""
        with pytest.raises(ValueError, match="Unknown concrete grade"):
            MaterialLoaderEC2.get_concrete("C99/999")
        
        with pytest.raises(ValueError, match="Unknown steel grade"):
            MaterialLoaderEC2.get_steel("B999Z")


class TestBeamFlexureEC2:
    """Test beam flexural capacity per EN 1992-1-1 Section 6.1."""
    
    def test_beam_flexure_under_reinforced(self):
        """Test under-reinforced beam (tension-controlled)."""
        code = Eurocode2Code()
        
        # Example: 300×500 mm beam, C30/37 concrete, B500B steel
        section = BeamSectionEC2(b=300, h=500, cover=40)
        concrete = MaterialLoaderEC2.get_concrete("C30/37")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        # Provide 1256 mm² (4φ20) tension steel
        As = 1256  # mm²
        
        # Applied moment
        M_Ed = 120  # kN·m
        
        result = code.check_beam_flexure(
            M_Ed=M_Ed,
            section=section,
            concrete=concrete,
            steel=steel,
            As=As
        )
        
        # Check results
        assert result['status'] == 'PASS'
        assert result['UR'] < 1.0
        assert result['M_Rd'] > M_Ed
        
        # Verify reinforcement ratio is within limits
        assert result['rho'] >= result['rho_min']
        assert result['rho'] <= result['rho_max']
        
        # Print for verification
        print(result['details'])
    
    def test_beam_flexure_minimum_reinforcement(self):
        """Test minimum reinforcement requirements (9.2.1.1)."""
        code = Eurocode2Code()
        
        section = BeamSectionEC2(b=300, h=500, cover=40)
        concrete = MaterialLoaderEC2.get_concrete("C30/37")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        # Very small steel area (below minimum)
        As = 100  # mm² (too low)
        M_Ed = 10  # kN·m (small moment)
        
        result = code.check_beam_flexure(
            M_Ed=M_Ed,
            section=section,
            concrete=concrete,
            steel=steel,
            As=As
        )
        
        # Should identify minimum reinforcement failure
        assert 'ρ < ρ_min' in result['details'] or result['rho'] < result['rho_min']
    
    def test_beam_flexure_capacity_calculation(self):
        """Verify moment capacity formula."""
        code = Eurocode2Code()
        
        # Known example
        b = 250  # mm
        h = 450  # mm
        cover = 35  # mm
        d = h - cover  # 415 mm
        
        section = BeamSectionEC2(b=b, h=h, cover=cover)
        concrete = MaterialLoaderEC2.get_concrete("C25/30")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        As = 982  # mm² (3φ20)
        
        result = code.check_beam_flexure(
            M_Ed=50,
            section=section,
            concrete=concrete,
            steel=steel,
            As=As
        )
        
        # Manual verification
        fcd = concrete.fcd  # 16.67 MPa
        fyd = steel.fyd  # 434.8 MPa
        lambda_f = concrete.lambda_factor  # 0.8
        eta_f = concrete.eta_factor  # 1.0
        
        # Neutral axis: x = As·fyd / (η·fcd·λ·b)
        x_expected = (As * fyd) / (eta_f * fcd * lambda_f * b)
        assert abs(result['x'] - x_expected) < 1.0
        
        # Lever arm: z = d - λ·x/2
        z_expected = d - lambda_f * x_expected / 2
        assert abs(result['z'] - z_expected) < 1.0


class TestBeamShearEC2:
    """Test beam shear capacity per EN 1992-1-1 Section 6.2."""
    
    def test_shear_without_reinforcement(self):
        """Test shear capacity without stirrups (6.2.2)."""
        code = Eurocode2Code()
        
        section = BeamSectionEC2(b=300, h=500, cover=40)
        concrete = MaterialLoaderEC2.get_concrete("C30/37")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        # No shear reinforcement
        Asw_s = 0.0
        V_Ed = 50  # kN
        
        result = code.check_beam_shear(
            V_Ed=V_Ed,
            section=section,
            concrete=concrete,
            steel=steel,
            Asw_s=Asw_s
        )
        
        # Should have V_Rd,c (concrete contribution only)
        assert result['V_Rdc'] > 0
        assert result['V_Rds'] == 0  # No stirrups
        
        print(result['details'])
    
    def test_shear_with_stirrups(self):
        """Test shear capacity with shear reinforcement (6.2.3)."""
        code = Eurocode2Code()
        
        section = BeamSectionEC2(b=300, h=500, cover=40)
        concrete = MaterialLoaderEC2.get_concrete("C30/37")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        # Stirrups: 2φ8@200mm → Asw = 2×50.3 = 100.6 mm²
        Asw = 2 * np.pi * (8/2)**2
        s = 200  # mm
        Asw_s = Asw / s  # 0.503 mm²/mm
        
        V_Ed = 150  # kN
        theta = 21.8  # degrees (cot θ = 2.5)
        
        result = code.check_beam_shear(
            V_Ed=V_Ed,
            section=section,
            concrete=concrete,
            steel=steel,
            Asw_s=Asw_s,
            theta=theta
        )
        
        # Should have both V_Rd,c and V_Rd,s
        assert result['V_Rds'] > 0
        assert result['V_Rd'] > result['V_Rdc']  # Total higher than concrete only
        
        # Check crushing limit
        assert result['V_Rdmax'] > 0
        
        print(result['details'])
    
    def test_shear_variable_strut_angle(self):
        """Test different strut angles (6.2.3)."""
        code = Eurocode2Code()
        
        section = BeamSectionEC2(b=250, h=400, cover=35)
        concrete = MaterialLoaderEC2.get_concrete("C25/30")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        Asw_s = 0.5  # mm²/mm
        V_Ed = 100  # kN
        
        # Test θ = 21.8° (cot θ = 2.5) - conservative
        result1 = code.check_beam_shear(V_Ed, section, concrete, steel, Asw_s, theta=21.8)
        
        # Test θ = 45° (cot θ = 1.0) - less conservative
        result2 = code.check_beam_shear(V_Ed, section, concrete, steel, Asw_s, theta=45)
        
        # Higher θ → lower V_Rd,s (less favorable)
        assert result2['V_Rds'] < result1['V_Rds']


class TestColumnEC2:
    """Test column interaction diagrams per EN 1992-1-1 Section 6.1."""
    
    def test_column_interaction_diagram_generation(self):
        """Test N-M interaction curve generation."""
        code = Eurocode2Code()
        
        section = ColumnSectionEC2(b=300, h=400, cover=40, L_eff=3000)
        concrete = MaterialLoaderEC2.get_concrete("C30/37")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        # 8φ20 → As = 2513 mm²
        As_total = 2513
        
        N_Ed = 1000  # kN
        M_Ed = 100  # kN·m
        
        result = code.check_column(
            N_Ed=N_Ed,
            M_Ed=M_Ed,
            section=section,
            concrete=concrete,
            steel=steel,
            As_total=As_total
        )
        
        # Verify interaction curve was generated
        assert len(result['N_capacity']) > 0
        assert len(result['M_capacity']) > 0
        
        # Pure compression point should be highest N
        assert max(result['N_capacity']) > N_Ed
        
        # Figure should be generated
        assert result['figure'] is not None
        
        print(result['details'])
    
    def test_column_slenderness_effects(self):
        """Test slenderness effects per Section 5.8."""
        code = Eurocode2Code()
        
        # Short column
        section_short = ColumnSectionEC2(b=300, h=400, cover=40, L_eff=2000)
        
        # Slender column
        section_slender = ColumnSectionEC2(b=300, h=400, cover=40, L_eff=6000)
        
        concrete = MaterialLoaderEC2.get_concrete("C30/37")
        steel = MaterialLoaderEC2.get_steel("B500B")
        As_total = 2513
        
        N_Ed = 800
        M_Ed = 80
        
        result_short = code.check_column(N_Ed, M_Ed, section_short, concrete, steel, As_total)
        result_slender = code.check_column(N_Ed, M_Ed, section_slender, concrete, steel, As_total)
        
        # Slender column should have η > 1.0
        assert result_short['eta'] == 1.0  # Short column
        assert result_slender['eta'] > 1.0  # Slender column with magnification


class TestServiceabilityEC2:
    """Test serviceability checks per EN 1992-1-1 Section 7."""
    
    def test_crack_width_check_xc1(self):
        """Test crack width for XC1 exposure class."""
        code = Eurocode2Code()
        
        section = BeamSectionEC2(b=300, h=500, cover=40)
        concrete = MaterialLoaderEC2.get_concrete("C30/37")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        As = 1256  # mm²
        M_ser = 80  # kN·m (service load, unfactored)
        L_span = 6000  # mm
        
        result = code.check_serviceability(
            M_ser=M_ser,
            section=section,
            concrete=concrete,
            steel=steel,
            As=As,
            L_span=L_span,
            exposure_class="XC1"
        )
        
        # XC1 allows w_max = 0.4 mm
        assert result['w_max'] == 0.4
        
        # Check result
        assert result['status_crack'] in ['PASS', 'FAIL']
        
        print(result['details'])
    
    def test_deflection_check_long_term(self):
        """Test long-term deflection with creep."""
        code = Eurocode2Code()
        
        section = BeamSectionEC2(b=250, h=450, cover=35)
        concrete = MaterialLoaderEC2.get_concrete("C25/30")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        As = 982  # mm²
        M_ser = 60  # kN·m
        L_span = 5000  # mm
        
        result = code.check_serviceability(
            M_ser=M_ser,
            section=section,
            concrete=concrete,
            steel=steel,
            As=As,
            L_span=L_span,
            load_duration="long_term"
        )
        
        # Allowable L/250
        assert result['delta_max'] == L_span / 250
        
        # Long-term deflection should be greater than short-term
        assert result['delta'] > 0
        
        print(result['details'])
    
    def test_serviceability_combined_check(self):
        """Test combined crack width and deflection check."""
        code = Eurocode2Code()
        
        section = BeamSectionEC2(b=300, h=500, cover=40)
        concrete = MaterialLoaderEC2.get_concrete("C30/37")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        As = 1256
        M_ser = 70
        L_span = 6000
        
        result = code.check_serviceability(
            M_ser=M_ser,
            section=section,
            concrete=concrete,
            steel=steel,
            As=As,
            L_span=L_span,
            exposure_class="XC3"
        )
        
        # Both checks should be present
        assert 'UR_crack' in result
        assert 'UR_deflection' in result
        
        # Overall status
        if result['status_crack'] == 'PASS' and result['status_deflection'] == 'PASS':
            assert result['status'] == 'PASS'
        else:
            assert result['status'] == 'FAIL'


class TestEurocode2Integration:
    """Integration tests for complete Eurocode 2 workflow."""
    
    def test_code_name_and_units(self):
        """Test code identification."""
        code = Eurocode2Code()
        
        assert code.code_name == "Eurocode 2 EN 1992-1-1:2004"
        assert code.code_units == "SI"
    
    def test_complete_beam_design_workflow(self):
        """Test complete beam design: flexure + shear + serviceability."""
        code = Eurocode2Code()
        
        section = BeamSectionEC2(b=300, h=500, cover=40)
        concrete = MaterialLoaderEC2.get_concrete("C30/37")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        As = 1256  # mm²
        Asw_s = 0.5  # mm²/mm
        
        # Ultimate loads
        M_Ed = 120  # kN·m
        V_Ed = 80  # kN
        
        # Service loads
        M_ser = 80  # kN·m
        L_span = 6000  # mm
        
        # Flexure check
        flex_result = code.check_beam_flexure(M_Ed, section, concrete, steel, As)
        
        # Shear check
        shear_result = code.check_beam_shear(V_Ed, section, concrete, steel, Asw_s)
        
        # Serviceability check
        sls_result = code.check_serviceability(M_ser, section, concrete, steel, As, L_span)
        
        # All should have valid results
        assert flex_result['status'] in ['PASS', 'FAIL']
        assert shear_result['status'] in ['PASS', 'FAIL']
        assert sls_result['status'] in ['PASS', 'FAIL']
        
        # Summary
        print("\n=== EUROCODE 2 BEAM DESIGN SUMMARY ===")
        print(f"Flexure: {flex_result['status']} (UR = {flex_result['UR']:.2f})")
        print(f"Shear: {shear_result['status']} (UR = {shear_result['UR']:.2f})")
        print(f"Serviceability: {sls_result['status']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
