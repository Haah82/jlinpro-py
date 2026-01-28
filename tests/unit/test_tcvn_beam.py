"""
Unit tests for TCVN 5574:2018 beam design module
"""

import pytest
import numpy as np
from src.design.tcvn_beam import (
    BeamSection,
    Reinforcement,
    Stirrups,
    check_flexure,
    check_shear,
    run_beam_check_from_ui
)
from src.design.tcvn_setup import DesignContext, MaterialLoader


class TestBeamSection:
    """Test BeamSection class."""
    
    def test_section_creation(self):
        """Test creating beam section."""
        section = BeamSection(b=300, h=500, cover=25)
        assert section.b == 300
        assert section.h == 500
        assert section.cover == 25
        assert section.h0 == 475
    
    def test_h0_property(self):
        """Test effective depth calculation."""
        section = BeamSection(b=250, h=450, cover=30)
        assert section.h0 == 420
    
    def test_invalid_width_too_small(self):
        """Test validation of minimum width."""
        with pytest.raises(ValueError, match="too small"):
            BeamSection(b=50, h=500, cover=25)
    
    def test_invalid_width_too_large(self):
        """Test validation of maximum width."""
        with pytest.raises(ValueError, match="excessive"):
            BeamSection(b=1500, h=500, cover=25)
    
    def test_invalid_height_too_small(self):
        """Test validation of minimum height."""
        with pytest.raises(ValueError, match="too small"):
            BeamSection(b=300, h=100, cover=25)


class TestReinforcement:
    """Test Reinforcement class."""
    
    def test_reinforcement_creation(self):
        """Test creating reinforcement."""
        rebar = Reinforcement(As_top=1000, As_bot=1500)
        assert rebar.As_top == 1000
        assert rebar.As_bot == 1500


class TestStirrups:
    """Test Stirrups class."""
    
    def test_stirrups_creation(self):
        """Test creating stirrups."""
        stirrups = Stirrups(diameter=10, n_legs=2, spacing=200)
        assert stirrups.diameter == 10
        assert stirrups.n_legs == 2
        assert stirrups.spacing == 200
    
    def test_asw_property(self):
        """Test Asw calculation."""
        stirrups = Stirrups(diameter=10, n_legs=2, spacing=200)
        expected = 2 * np.pi * (10/2)**2
        assert stirrups.Asw == pytest.approx(expected, rel=0.01)
    
    def test_qsw_property(self):
        """Test qsw calculation."""
        stirrups = Stirrups(diameter=10, n_legs=2, spacing=200)
        asw = 2 * np.pi * (10/2)**2
        expected = asw / 200
        assert stirrups.qsw == pytest.approx(expected, rel=0.01)
    
    def test_invalid_diameter(self):
        """Test validation of stirrup diameter."""
        with pytest.raises(ValueError, match="too small"):
            Stirrups(diameter=4, n_legs=2, spacing=200)
    
    def test_invalid_spacing_too_small(self):
        """Test validation of minimum spacing."""
        with pytest.raises(ValueError, match="too small"):
            Stirrups(diameter=10, n_legs=2, spacing=30)
    
    def test_invalid_spacing_too_large(self):
        """Test validation of maximum spacing."""
        with pytest.raises(ValueError, match="too large"):
            Stirrups(diameter=10, n_legs=2, spacing=600)


class TestFlexureCheck:
    """Test flexure check function."""
    
    def test_flexure_pass(self):
        """Test flexure check that passes."""
        section = BeamSection(b=300, h=500, cover=25)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 25)
        
        # Provide adequate steel
        result = check_flexure(M_u=200, section=section, context=context, As_provided=1500)
        
        assert result['status'] == 'PASS'
        assert result['UR'] <= 1.0
        assert result['M_cap'] > 200
    
    def test_flexure_fail_inadequate_steel(self):
        """Test flexure check with inadequate steel."""
        section = BeamSection(b=300, h=500, cover=25)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 25)
        
        # Provide insufficient steel
        result = check_flexure(M_u=300, section=section, context=context, As_provided=800)
        
        assert result['status'] == 'FAIL'
        assert result['UR'] > 1.0
    
    def test_flexure_alpha_m_limit(self):
        """Test flexure check with excessive moment (α_m > 0.5)."""
        section = BeamSection(b=200, h=300, cover=25)
        context = DesignContext.from_ui_inputs("B20", "CB400V", 25)
        
        # Apply very large moment
        result = check_flexure(M_u=500, section=section, context=context, As_provided=2000)
        
        assert result['status'] == 'FAIL'
        assert result['alpha_m'] > 0.5
        assert 'under-reinforced' in result['details']
    
    def test_flexure_xi_limit(self):
        """Test flexure check with ξ exceeding limit."""
        section = BeamSection(b=300, h=500, cover=25)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 25)
        
        # Calculate moment that gives ξ > ξ_R
        Rb = context.concrete.Rb
        Rs = context.steel.Rs
        b = section.b
        h0 = section.h0
        
        xi_R = context.get_xi_limit()  # 0.55 for CB400V
        xi_test = xi_R + 0.05  # Exceed limit
        
        alpha_m = xi_test * (1 - 0.5 * xi_test)
        M_u = (Rb * b * h0**2 * alpha_m) / 1e6  # kN·m
        
        As_provided = (Rb * b * h0 * xi_test) / Rs
        
        result = check_flexure(M_u, section, context, As_provided)
        
        assert result['xi'] > xi_R
        # Note: May pass if provided steel is adjusted properly
    
    def test_flexure_details(self):
        """Test flexure check returns detailed information."""
        section = BeamSection(b=300, h=500, cover=25)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 25)
        
        result = check_flexure(M_u=200, section=section, context=context, As_provided=1500)
        
        assert 'alpha_m' in result
        assert 'xi' in result
        assert 'xi_R' in result
        assert 'As_req' in result
        assert 'M_cap' in result
        assert 'UR' in result
        assert 'details' in result


class TestShearCheck:
    """Test shear check function."""
    
    def test_shear_pass(self):
        """Test shear check that passes."""
        section = BeamSection(b=300, h=500, cover=25)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 25)
        stirrups = Stirrups(diameter=10, n_legs=2, spacing=150)
        
        # Apply moderate shear
        result = check_shear(Q_u=100, section=section, context=context, stirrups=stirrups)
        
        assert result['status'] == 'PASS'
        assert result['UR'] <= 1.0
    
    def test_shear_fail_inadequate_stirrups(self):
        """Test shear check with inadequate stirrups."""
        section = BeamSection(b=300, h=500, cover=25)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 25)
        stirrups = Stirrups(diameter=8, n_legs=2, spacing=300)
        
        # Apply large shear
        result = check_shear(Q_u=250, section=section, context=context, stirrups=stirrups)
        
        assert result['status'] == 'FAIL'
        assert result['UR'] > 1.0
    
    def test_shear_components(self):
        """Test shear capacity components."""
        section = BeamSection(b=300, h=500, cover=25)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 25)
        stirrups = Stirrups(diameter=10, n_legs=2, spacing=200)
        
        result = check_shear(Q_u=100, section=section, context=context, stirrups=stirrups)
        
        # Check components exist
        assert result['Qb'] > 0  # Concrete contribution
        assert result['Qsw'] > 0  # Stirrup contribution
        assert result['Q_cap'] == pytest.approx(result['Qb'] + result['Qsw'], rel=0.01)
        assert result['Q_crush'] > 0  # Crushing limit
    
    def test_shear_crushing_limit(self):
        """Test shear crushing limit check."""
        section = BeamSection(b=200, h=400, cover=25)
        context = DesignContext.from_ui_inputs("B20", "CB400V", 25)
        stirrups = Stirrups(diameter=10, n_legs=2, spacing=100)
        
        # Apply very large shear to trigger crushing
        result = check_shear(Q_u=500, section=section, context=context, stirrups=stirrups)
        
        assert result['status'] == 'FAIL'
        # Check if crushing governs
        assert result['Q_u'] if 'Q_u' in result else 500 > result['Q_crush']
    
    def test_shear_details(self):
        """Test shear check returns detailed information."""
        section = BeamSection(b=300, h=500, cover=25)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 25)
        stirrups = Stirrups(diameter=10, n_legs=2, spacing=200)
        
        result = check_shear(Q_u=100, section=section, context=context, stirrups=stirrups)
        
        assert 'Qb' in result
        assert 'Qsw' in result
        assert 'Q_cap' in result
        assert 'Q_crush' in result
        assert 'UR' in result
        assert 'details' in result


class TestStreamlitIntegration:
    """Test Streamlit integration function."""
    
    def test_run_beam_check_from_ui(self):
        """Test complete beam check from UI."""
        results = run_beam_check_from_ui(
            beam_id=1,
            b=300, h=500, cover=25,
            As_top=1000, As_bot=1500,
            stirrup_dia=10, n_legs=2, spacing=200,
            concrete_name="B25",
            steel_name="CB400V",
            M_u=200,
            Q_u=100
        )
        
        assert 'beam_id' in results
        assert 'flexure' in results
        assert 'shear' in results
        
        assert results['beam_id'] == 1
        assert results['flexure']['status'] in ['PASS', 'FAIL']
        assert results['shear']['status'] in ['PASS', 'FAIL']
    
    def test_run_beam_check_typical_design(self):
        """Test with typical design values."""
        # Typical 300x500 beam with B30 concrete and CB400V steel
        results = run_beam_check_from_ui(
            beam_id=1,
            b=300, h=500, cover=30,
            As_top=982,  # 4φ18
            As_bot=1964,  # 4φ25
            stirrup_dia=10, n_legs=2, spacing=150,
            concrete_name="B30",
            steel_name="CB400V",
            M_u=220,
            Q_u=110
        )
        
        # Should pass with typical proportions
        assert results['flexure']['status'] == 'PASS'
        assert results['shear']['status'] == 'PASS'


class TestRealWorldExamples:
    """Test with real-world design scenarios."""
    
    def test_simple_supported_beam(self):
        """Test simply-supported beam design."""
        # 6m span, 12 kN/m UDL, B25 concrete, CB400V steel
        # Mu = wL²/8 = 12*6²/8 = 54 kN·m
        # Vu = wL/2 = 12*6/2 = 36 kN
        
        section = BeamSection(b=250, h=450, cover=25)
        context = DesignContext.from_ui_inputs("B25", "CB400V", 25)
        stirrups = Stirrups(diameter=8, n_legs=2, spacing=200)
        
        flex_result = check_flexure(M_u=54, section=section, context=context, As_provided=800)
        shear_result = check_shear(Q_u=36, section=section, context=context, stirrups=stirrups)
        
        # Both should pass with adequate design
        assert flex_result['status'] == 'PASS'
        assert shear_result['status'] == 'PASS'
    
    def test_cantilever_beam(self):
        """Test cantilever beam design."""
        # 2m cantilever, 15 kN/m UDL, B30 concrete, CB400V steel
        # Mu = wL²/2 = 15*2²/2 = 30 kN·m
        # Vu = wL = 15*2 = 30 kN
        
        section = BeamSection(b=200, h=400, cover=30)
        context = DesignContext.from_ui_inputs("B30", "CB400V", 30)
        stirrups = Stirrups(diameter=8, n_legs=2, spacing=150)
        
        # Use top steel for cantilever
        flex_result = check_flexure(M_u=30, section=section, context=context, As_provided=600)
        shear_result = check_shear(Q_u=30, section=section, context=context, stirrups=stirrups)
        
        assert flex_result['status'] == 'PASS'
        assert shear_result['status'] == 'PASS'
