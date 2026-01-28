"""
Unit tests for TCVN 5574:2018 material setup module
"""

import pytest
import numpy as np
from src.design.tcvn_setup import (
    ConcreteGrade,
    SteelGrade,
    DesignContext,
    MaterialLoader,
    create_design_context_from_streamlit,
    CONCRETE_GRADES,
    STEEL_GRADES
)


class TestConcreteGrade:
    """Test ConcreteGrade class."""
    
    def test_concrete_creation(self):
        """Test creating concrete grade."""
        concrete = ConcreteGrade(name="B25", Rb=14.5, Rbt=1.05, Eb=30000)
        assert concrete.name == "B25"
        assert concrete.Rb == 14.5
        assert concrete.Rbt == 1.05
        assert concrete.Eb == 30000
        assert concrete.gamma_b == 1.0
    
    def test_concrete_fc_property(self):
        """Test fc property (approximate cylinder strength)."""
        concrete = ConcreteGrade(name="B20", Rb=11.5, Rbt=0.90, Eb=27500)
        fc = concrete.fc
        assert fc == pytest.approx(11.5 * 1.25, rel=0.01)
    
    def test_stress_strain_bilinear(self):
        """Test bilinear stress-strain model."""
        concrete = CONCRETE_GRADES["B25"]
        strains, stresses = concrete.get_stress_strain_model('bilinear')
        
        assert len(strains) == len(stresses)
        assert strains[0] == 0.0
        assert stresses[0] == 0.0
        assert max(stresses) == pytest.approx(concrete.Rb, rel=0.01)
    
    def test_stress_strain_parabolic(self):
        """Test parabolic stress-strain model."""
        concrete = CONCRETE_GRADES["B30"]
        strains, stresses = concrete.get_stress_strain_model('parabolic')
        
        assert len(strains) == 50
        assert max(stresses) <= concrete.Rb
    
    def test_invalid_rb(self):
        """Test validation of Rb."""
        with pytest.raises(ValueError, match="outside typical range"):
            ConcreteGrade(name="Invalid", Rb=100.0, Rbt=1.0, Eb=30000)
    
    def test_invalid_rbt(self):
        """Test validation of Rbt."""
        with pytest.raises(ValueError, match="seems too high"):
            ConcreteGrade(name="Invalid", Rb=20.0, Rbt=10.0, Eb=30000)


class TestSteelGrade:
    """Test SteelGrade class."""
    
    def test_steel_creation(self):
        """Test creating steel grade."""
        steel = SteelGrade(name="CB400V", Rs=350, Rsc=350, Es=200000)
        assert steel.name == "CB400V"
        assert steel.Rs == 350
        assert steel.Rsc == 350
        assert steel.Es == 200000
    
    def test_fy_property(self):
        """Test fy property (alias for Rs)."""
        steel = STEEL_GRADES["CB400V"]
        assert steel.fy == steel.Rs
        assert steel.fy == 350
    
    def test_epsilon_y(self):
        """Test yield strain calculation."""
        steel = SteelGrade(name="CB400V", Rs=350, Rsc=350, Es=200000)
        epsilon_y = steel.epsilon_y
        expected = 350 / 200000
        assert epsilon_y == pytest.approx(expected, rel=0.01)
    
    def test_invalid_rs(self):
        """Test validation of Rs."""
        with pytest.raises(ValueError, match="outside typical range"):
            SteelGrade(name="Invalid", Rs=1000, Rsc=1000, Es=200000)


class TestMaterialLoader:
    """Test MaterialLoader class."""
    
    def test_get_concrete_valid(self):
        """Test loading valid concrete grade."""
        concrete = MaterialLoader.get_concrete("B25")
        assert concrete.name == "B25"
        assert concrete.Rb == 14.5
    
    def test_get_concrete_invalid(self):
        """Test loading invalid concrete grade."""
        with pytest.raises(ValueError, match="Unknown concrete grade"):
            MaterialLoader.get_concrete("B100")
    
    def test_get_steel_valid(self):
        """Test loading valid steel grade."""
        steel = MaterialLoader.get_steel("CB400V")
        assert steel.name == "CB400V"
        assert steel.Rs == 350
    
    def test_get_steel_invalid(self):
        """Test loading invalid steel grade."""
        with pytest.raises(ValueError, match="Unknown steel grade"):
            MaterialLoader.get_steel("CB999X")
    
    def test_list_grades(self):
        """Test listing available grades."""
        concrete_list = MaterialLoader.list_concrete_grades()
        steel_list = MaterialLoader.list_steel_grades()
        
        assert "B25" in concrete_list
        assert "B30" in concrete_list
        assert "CB400V" in steel_list
        assert len(concrete_list) == 9
        assert len(steel_list) == 4


class TestDesignContext:
    """Test DesignContext class."""
    
    def test_context_creation(self):
        """Test creating design context."""
        concrete = MaterialLoader.get_concrete("B25")
        steel = MaterialLoader.get_steel("CB400V")
        
        context = DesignContext(
            concrete=concrete,
            steel=steel,
            cover=25.0
        )
        
        assert context.concrete.name == "B25"
        assert context.steel.name == "CB400V"
        assert context.cover == 25.0
        assert context.gamma_b == 1.0
        assert context.gamma_s == 1.15
    
    def test_context_from_ui_inputs(self):
        """Test factory method for UI integration."""
        context = DesignContext.from_ui_inputs("B30", "CB400V", 30.0)
        
        assert context.concrete.name == "B30"
        assert context.steel.name == "CB400V"
        assert context.cover == 30.0
    
    def test_xi_limit_cb400v(self):
        """Test ξ_R limit for CB400V."""
        context = DesignContext.from_ui_inputs("B25", "CB400V", 25.0)
        xi_limit = context.get_xi_limit()
        assert xi_limit == 0.55
    
    def test_xi_limit_cb500v(self):
        """Test ξ_R limit for CB500V."""
        context = DesignContext.from_ui_inputs("B30", "CB500V", 25.0)
        xi_limit = context.get_xi_limit()
        assert xi_limit == 0.50
    
    def test_xi_limit_cb240t(self):
        """Test ξ_R limit for CB240T."""
        context = DesignContext.from_ui_inputs("B20", "CB240T", 25.0)
        xi_limit = context.get_xi_limit()
        assert xi_limit == 0.60
    
    def test_invalid_cover_too_small(self):
        """Test validation of minimum cover."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="greater than 15"):
            DesignContext.from_ui_inputs("B25", "CB400V", 10.0)
    
    def test_invalid_cover_too_large(self):
        """Test validation of maximum cover."""
        with pytest.raises(ValueError, match="seems excessive"):
            DesignContext.from_ui_inputs("B25", "CB400V", 150.0)
    
    def test_invalid_environment(self):
        """Test validation of environment."""
        with pytest.raises(ValueError, match="not in allowed values"):
            DesignContext(
                concrete=MaterialLoader.get_concrete("B25"),
                steel=MaterialLoader.get_steel("CB400V"),
                cover=25.0,
                environment="unknown"
            )


class TestStreamlitIntegration:
    """Test Streamlit integration function."""
    
    def test_create_context_from_streamlit(self):
        """Test creating context from Streamlit inputs."""
        context = create_design_context_from_streamlit(
            concrete_name="B25",
            steel_name="CB400V",
            cover=25.0,
            environment='normal'
        )
        
        assert context.concrete.name == "B25"
        assert context.steel.name == "CB400V"
        assert context.cover == 25.0
        assert context.gamma_b == 1.0
        assert context.gamma_s == 1.15
        assert context.environment == 'normal'
    
    def test_create_context_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            create_design_context_from_streamlit("Invalid", "CB400V", 25.0)
        
        with pytest.raises(ValueError):
            create_design_context_from_streamlit("B25", "Invalid", 25.0)


class TestPredefinedGrades:
    """Test predefined material grades."""
    
    def test_all_concrete_grades_valid(self):
        """Test all predefined concrete grades are valid."""
        for grade_name, concrete in CONCRETE_GRADES.items():
            assert concrete.name == grade_name
            assert concrete.Rb > 0
            assert concrete.Rbt > 0
            assert concrete.Eb > 0
    
    def test_all_steel_grades_valid(self):
        """Test all predefined steel grades are valid."""
        for grade_name, steel in STEEL_GRADES.items():
            assert steel.name == grade_name
            assert steel.Rs > 0
            assert steel.Rsc > 0
            assert steel.Es == 200000
    
    def test_concrete_grades_ordered(self):
        """Test concrete grades are in ascending strength order."""
        grades = ["B15", "B20", "B25", "B30", "B35", "B40", "B45", "B50", "B60"]
        rb_values = [CONCRETE_GRADES[g].Rb for g in grades]
        
        # Check monotonically increasing
        for i in range(len(rb_values) - 1):
            assert rb_values[i] < rb_values[i+1]
    
    def test_steel_grades_ordered(self):
        """Test steel grades are in ascending strength order."""
        grades = ["CB240T", "CB300V", "CB400V", "CB500V"]
        rs_values = [STEEL_GRADES[g].Rs for g in grades]
        
        # Check monotonically increasing
        for i in range(len(rs_values) - 1):
            assert rs_values[i] < rs_values[i+1]
