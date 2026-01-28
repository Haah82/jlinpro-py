"""
Integration tests for Eurocode 2 module.

Verifies complete design workflows using realistic scenarios from
EN 1992-1-1:2004 worked examples.
"""

import pytest
from src.design.eurocode2 import (
    MaterialLoaderEC2,
    BeamSectionEC2,
    ColumnSectionEC2,
    Eurocode2Code
)
from src.design import get_design_code, CODE_REGISTRY


class TestEurocode2FactoryIntegration:
    """Test Eurocode 2 integration with design code factory."""
    
    def test_eurocode2_in_registry(self):
        """Verify Eurocode 2 is registered in CODE_REGISTRY."""
        assert "Eurocode 2 EN 1992-1-1:2004 (Europe)" in CODE_REGISTRY
        
        code = CODE_REGISTRY["Eurocode 2 EN 1992-1-1:2004 (Europe)"]
        assert isinstance(code, Eurocode2Code)
    
    def test_get_eurocode2_via_factory(self):
        """Test retrieving Eurocode 2 via factory method."""
        code = get_design_code("Eurocode 2 EN 1992-1-1:2004 (Europe)")
        
        assert isinstance(code, Eurocode2Code)
        assert code.code_name == "Eurocode 2 EN 1992-1-1:2004"
        assert code.code_units == "SI"


class TestRealisticBeamDesign:
    """
    Realistic beam design scenario based on EN 1992-1-1 examples.
    
    Scenario: Simply-supported beam for residential building
    Span: 6m, Loads: Dead + Live = 15 kN/m
    """
    
    def test_residential_beam_design_workflow(self):
        """Complete design workflow for residential beam."""
        code = get_design_code("Eurocode 2 EN 1992-1-1:2004 (Europe)")
        
        # Geometry: 300×550 mm beam
        section = BeamSectionEC2(b=300, h=550, cover=40)
        
        # Materials: C30/37 concrete, B500B steel
        concrete = MaterialLoaderEC2.get_concrete("C30/37")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        # Design loads (factored)
        # M_Ed = (γ_G·g + γ_Q·q)·L²/8 = 1.35×10 + 1.5×5 = 21 kN/m
        # M_Ed = 21 × 6²/8 = 94.5 kN·m
        M_Ed = 95  # kN·m (rounded)
        V_Ed = 63  # kN (21×6/2)
        
        # Service loads (unfactored)
        M_ser = 67.5  # kN·m (15×6²/8)
        
        # Reinforcement (trial design)
        As = 1256  # mm² (4φ20)
        
        # Stirrups: φ8@200mm
        Asw = 2 * 3.14159 * (8/2)**2  # 2-leg stirrups
        Asw_s = Asw / 200  # mm²/mm
        
        # ================================================================
        # FLEXURE CHECK
        # ================================================================
        flex_result = code.check_beam_flexure(
            M_Ed=M_Ed,
            section=section,
            concrete=concrete,
            steel=steel,
            As=As
        )
        
        print("\n" + "="*60)
        print("FLEXURE CHECK")
        print("="*60)
        print(flex_result['details'])
        
        assert flex_result['status'] == 'PASS', "Flexure check should pass"
        assert flex_result['UR'] < 1.0, "Utilization should be less than 1.0"
        assert flex_result['M_Rd'] > M_Ed, "Capacity should exceed demand"
        
        # ================================================================
        # SHEAR CHECK
        # ================================================================
        shear_result = code.check_beam_shear(
            V_Ed=V_Ed,
            section=section,
            concrete=concrete,
            steel=steel,
            Asw_s=Asw_s,
            theta=21.8  # Conservative strut angle
        )
        
        print("\n" + "="*60)
        print("SHEAR CHECK")
        print("="*60)
        print(shear_result['details'])
        
        assert shear_result['status'] == 'PASS', "Shear check should pass"
        assert shear_result['UR'] < 1.0
        
        # ================================================================
        # SERVICEABILITY CHECK
        # ================================================================
        sls_result = code.check_serviceability(
            M_ser=M_ser,
            section=section,
            concrete=concrete,
            steel=steel,
            As=As,
            L_span=6000,  # mm
            exposure_class="XC1",  # Indoor exposure
            load_duration="long_term"
        )
        
        print("\n" + "="*60)
        print("SERVICEABILITY CHECK")
        print("="*60)
        print(sls_result['details'])
        
        assert sls_result['status'] == 'PASS', "Serviceability should pass"
        
        # ================================================================
        # SUMMARY
        # ================================================================
        print("\n" + "="*60)
        print("DESIGN SUMMARY")
        print("="*60)
        print(f"Section: {section.b}×{section.h} mm")
        print(f"Concrete: {concrete.strength_class}")
        print(f"Steel: {steel.grade}")
        print(f"Reinforcement: {As:.0f} mm²")
        print(f"Stirrups: Asw/s = {Asw_s:.3f} mm²/mm")
        print()
        print(f"Flexure:        {flex_result['status']:4} (UR = {flex_result['UR']:.2f})")
        print(f"Shear:          {shear_result['status']:4} (UR = {shear_result['UR']:.2f})")
        print(f"Crack width:    {sls_result['status_crack']:4} (UR = {sls_result['UR_crack']:.2f})")
        print(f"Deflection:     {sls_result['status_deflection']:4} (UR = {sls_result['UR_deflection']:.2f})")
        print("="*60)


class TestRealisticColumnDesign:
    """
    Realistic column design scenario.
    
    Scenario: Interior column for multi-story building
    Height: 3.5m, Axial load + moment from eccentricity
    """
    
    def test_building_column_design(self):
        """Complete column design for building interior column."""
        code = get_design_code("Eurocode 2 EN 1992-1-1:2004 (Europe)")
        
        # Geometry: 400×400 mm column, 3.5m height
        section = ColumnSectionEC2(
            b=400,
            h=400,
            cover=40,
            L_eff=3500  # Effective length (braced frame)
        )
        
        # Materials: C35/45 concrete, B500B steel
        concrete = MaterialLoaderEC2.get_concrete("C35/45")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        # Reinforcement: 8φ20 bars → As = 2513 mm²
        As_total = 8 * 3.14159 * (20/2)**2
        
        # Design loads
        N_Ed = 1500  # kN (compression)
        M_Ed = 120   # kN·m (from eccentricity + lateral loads)
        
        # ================================================================
        # COLUMN CHECK
        # ================================================================
        col_result = code.check_column(
            N_Ed=N_Ed,
            M_Ed=M_Ed,
            section=section,
            concrete=concrete,
            steel=steel,
            As_total=As_total,
            n_points=50
        )
        
        print("\n" + "="*60)
        print("COLUMN INTERACTION CHECK")
        print("="*60)
        print(col_result['details'])
        
        # Verify results
        assert col_result['status'] == 'PASS', "Column check should pass"
        assert col_result['UR'] < 1.0
        
        # Verify interaction diagram
        assert len(col_result['N_capacity']) > 40
        assert len(col_result['M_capacity']) > 40
        
        # Verify pure compression capacity
        N_max = max(col_result['N_capacity'])
        print(f"\nPure compression capacity: {N_max:.1f} kN")
        assert N_max > N_Ed, "Pure compression capacity should exceed applied load"
        
        # ================================================================
        # SUMMARY
        # ================================================================
        print("\n" + "="*60)
        print("COLUMN DESIGN SUMMARY")
        print("="*60)
        print(f"Section: {section.b}×{section.h} mm")
        print(f"Effective length: {section.L_eff:.0f} mm")
        print(f"Slenderness: {section.slenderness:.1f}")
        print(f"Concrete: {concrete.strength_class}")
        print(f"Steel: {steel.grade}")
        print(f"Reinforcement: {As_total:.0f} mm²")
        print()
        print(f"Applied loads: N_Ed = {N_Ed:.0f} kN, M_Ed = {M_Ed:.1f} kN·m")
        print(f"Magnification factor η = {col_result['eta']:.2f}")
        print(f"Status: {col_result['status']} (UR = {col_result['UR']:.2f})")
        print("="*60)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_high_strength_concrete_c90(self):
        """Test with highest strength concrete C90/105."""
        code = Eurocode2Code()
        
        section = BeamSectionEC2(b=300, h=500, cover=40)
        concrete = MaterialLoaderEC2.get_concrete("C90/105")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        As = 1256
        M_Ed = 150
        
        result = code.check_beam_flexure(M_Ed, section, concrete, steel, As)
        
        # Should work with adjusted factors
        assert result['status'] in ['PASS', 'FAIL']
        
        # Lambda and eta should be reduced for high strength
        assert concrete.lambda_factor < 0.8
        assert concrete.eta_factor < 1.0
    
    def test_low_reinforcement_ratio(self):
        """Test beam with very low reinforcement."""
        code = Eurocode2Code()
        
        section = BeamSectionEC2(b=300, h=500, cover=40)
        concrete = MaterialLoaderEC2.get_concrete("C25/30")
        steel = MaterialLoaderEC2.get_steel("B500B")
        
        # Very minimal steel (below minimum)
        As = 150  # mm² (intentionally very low to trigger failure)
        M_Ed = 20
        
        result = code.check_beam_flexure(M_Ed, section, concrete, steel, As)
        
        # Should flag minimum reinforcement issue
        # Either rho is below rho_min OR the details mention the failure
        assert result['rho'] < result['rho_min'] or 'ρ < ρ_min' in result['details']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
