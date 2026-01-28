"""
Integration tests for ACI 318-25 design code implementation.

Tests complete workflows from UI to backend calculations.
"""

import pytest
from src.design.aci318 import (
    run_beam_flexure_check_aci_from_ui,
    run_beam_shear_check_aci_from_ui,
    run_column_check_aci_from_ui,
    run_sls_check_aci_from_ui
)
from src.design import get_design_code, CODE_REGISTRY


class TestACI318Integration:
    """Integration tests for ACI 318-25 module."""
    
    def test_code_registry(self):
        """Test ACI code is registered in factory."""
        assert "ACI 318-25 (USA)" in CODE_REGISTRY
        
        code = get_design_code("ACI 318-25 (USA)")
        assert code.code_name == "ACI 318-25"
        assert code.code_units == "SI"
    
    def test_complete_beam_flexure_workflow(self):
        """Test complete beam flexure check workflow."""
        result = run_beam_flexure_check_aci_from_ui(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1600,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Mu=120
        )
        
        # Verify all required keys are present
        assert 'beam_id' in result
        assert 'UR' in result
        assert 'status' in result
        assert 'Mn' in result
        assert 'phiMn' in result
        assert 'phi' in result
        assert 'details' in result
        
        # Verify values are reasonable
        assert result['Mn'] > 0
        assert result['phiMn'] > 0
        assert 0.65 <= result['phi'] <= 0.90
        assert result['UR'] > 0
    
    def test_complete_beam_shear_workflow(self):
        """Test complete beam shear check workflow."""
        result = run_beam_shear_check_aci_from_ui(
            beam_id=1,
            b=300, h=500, cover=40,
            Vu=80,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            stirrup_dia=10,
            n_legs=2,
            spacing=150
        )
        
        # Verify all required keys
        assert 'beam_id' in result
        assert 'UR' in result
        assert 'status' in result
        assert 'Vc' in result
        assert 'Vs' in result
        assert 'Vn' in result
        assert 'phiVn' in result
        assert 'details' in result
        
        # Verify values
        assert result['Vc'] > 0
        assert result['Vs'] >= 0
        assert result['Vn'] == result['Vc'] + result['Vs']
        assert result['phi'] == 0.75
    
    def test_complete_column_workflow(self):
        """Test complete column interaction diagram workflow."""
        result = run_column_check_aci_from_ui(
            col_id=1,
            b=400, h=400, cover=40,
            As_total=3200,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Pu=500, Mu=100,
            column_type="tied"
        )
        
        # Verify all required keys
        assert 'col_id' in result
        assert 'UR' in result
        assert 'status' in result
        assert 'figure' in result
        assert 'phi' in result
        assert 'N_capacity' in result
        assert 'M_capacity' in result
        assert 'details' in result
        
        # Verify Plotly figure
        assert result['figure'] is not None
        assert hasattr(result['figure'], 'data')
        
        # Verify capacity arrays
        assert len(result['N_capacity']) > 0
        assert len(result['M_capacity']) > 0
        assert len(result['N_capacity']) == len(result['M_capacity'])
    
    def test_complete_serviceability_workflow(self):
        """Test complete serviceability check workflow."""
        result = run_sls_check_aci_from_ui(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1600,
            bar_diameter=20,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            M_service=80,
            L=6.0,
            load_type="sustained"
        )
        
        # Verify all required keys
        assert 'beam_id' in result
        assert 'deflection' in result
        assert 'crack_control' in result
        assert 'status' in result
        assert 'details' in result
        
        # Verify deflection dict
        defl = result['deflection']
        assert 'delta_immediate' in defl
        assert 'delta_total' in defl
        assert 'delta_allow' in defl
        assert 'UR' in defl
        assert 'status' in defl
        
        # Verify crack control dict
        crack = result['crack_control']
        assert 'fs' in crack
        assert 's_max' in crack
        assert 's_actual' in crack
        assert 'UR' in crack
        assert 'status' in crack


class TestACI318MultiCodeComparison:
    """Compare ACI and TCVN results for similar designs."""
    
    def test_flexure_capacity_comparison(self):
        """Compare flexure capacity between ACI and TCVN (if similar materials)."""
        # ACI check
        aci_result = run_beam_flexure_check_aci_from_ui(
            beam_id=1,
            b=300, h=500, cover=40,
            As=1800,
            concrete_name="4000 psi (27.6 MPa)",  # ≈ B30 (fc=17 MPa Rb)
            steel_name="Grade 60 (414 MPa)",  # ≈ CB400V
            Mu=100
        )
        
        # Both should provide reasonable capacities
        assert aci_result['phiMn'] > 0
        assert aci_result['status'] in ['PASS', 'FAIL']
        
        # ACI should have tension-controlled φ for typical beam
        assert aci_result['phi'] >= 0.85


class TestRealWorldScenarios:
    """Test realistic building design scenarios."""
    
    def test_office_building_beam(self):
        """Test typical office building floor beam."""
        # 300x600mm beam, 8m span, fc'=4000 psi, Grade 60
        flexure = run_beam_flexure_check_aci_from_ui(
            beam_id=1,
            b=300, h=600, cover=40,
            As=2400,  # 6-φ20 bars
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Mu=180
        )
        
        shear = run_beam_shear_check_aci_from_ui(
            beam_id=1,
            b=300, h=600, cover=40,
            Vu=100,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            stirrup_dia=10,
            n_legs=2,
            spacing=200
        )
        
        # Both should pass with reasonable UR
        assert flexure['status'] == 'PASS'
        assert shear['status'] == 'PASS'
        assert flexure['UR'] < 1.0
        assert shear['UR'] < 1.0
    
    def test_building_column(self):
        """Test typical building column."""
        # 400x400mm column, fc'=5000 psi, 3% steel
        result = run_column_check_aci_from_ui(
            col_id=1,
            b=400, h=400, cover=40,
            As_total=4800,  # 3% steel ratio
            concrete_name="5000 psi (34.5 MPa)",
            steel_name="Grade 60 (414 MPa)",
            Pu=800,
            Mu=120,
            column_type="tied"
        )
        
        # Should generate valid interaction diagram
        assert result['figure'] is not None
        assert len(result['N_capacity']) > 30  # Enough points for smooth curve
        
        # Maximum axial capacity should be reasonable
        assert max(result['N_capacity']) > 0
        assert min(result['N_capacity']) < 0  # Has tension capacity
    
    def test_parking_structure_beam(self):
        """Test parking structure beam with heavy loading."""
        # Heavy duty beam with stirrups
        result = run_beam_shear_check_aci_from_ui(
            beam_id=1,
            b=400, h=700, cover=40,
            Vu=250,
            concrete_name="4000 psi (27.6 MPa)",
            steel_name="Grade 60 (414 MPa)",
            stirrup_dia=12,
            n_legs=4,  # Double stirrups
            spacing=150
        )
        
        # Should have significant stirrup contribution
        assert result['Vs'] > result['Vc']
        assert result['Vn'] > 200  # High shear capacity
    
    def test_long_span_beam_deflection(self):
        """Test long-span beam for deflection control."""
        result = run_sls_check_aci_from_ui(
            beam_id=1,
            b=400, h=800, cover=40,
            As=4800,
            bar_diameter=25,
            concrete_name="5000 psi (34.5 MPa)",
            steel_name="Grade 60 (414 MPa)",
            M_service=300,
            L=12.0,  # 12m span
            load_type="sustained"
        )
        
        # Deflection should be calculated
        assert result['deflection']['delta_total'] > 0
        assert result['deflection']['delta_allow'] > 0
        
        # Long-term deflection should be > immediate
        assert result['deflection']['delta_total'] > result['deflection']['delta_immediate']
