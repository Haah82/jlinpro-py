"""
Integration test: Verify serviceability check UI integration
"""

import pytest
from src.design.tcvn_sls import run_sls_check_from_ui


def test_complete_sls_workflow():
    """Test complete serviceability workflow."""
    # Typical 300x500 beam, 6m span, B30 concrete, CB400V steel
    # Service moment M_ser = 100 kN·m
    
    results = run_sls_check_from_ui(
        beam_id=1,
        b=300,
        h=500,
        cover=30,
        As=1800,  # mm²
        bar_diameter=20,  # mm
        concrete_name="B30",
        steel_name="CB400V",
        M_ser=100,  # kN·m
        L=6.0,  # m
        environment='normal',
        load_duration='long_term'
    )
    
    print("\n" + "="*70)
    print("SERVICEABILITY CHECK RESULTS")
    print("="*70)
    print(f"\nBeam ID: {results['beam_id']}")
    
    # Crack width results
    crack = results['crack_width']
    print("\n" + "="*70)
    print("CRACK WIDTH CHECK")
    print("="*70)
    print(f"Calculated width a_cr = {crack['a_cr']:.3f} mm")
    print(f"Allowable width a_limit = {crack['a_limit']:.1f} mm")
    print(f"Steel stress σs = {crack['sigma_s']:.1f} MPa")
    print(f"Reinforcement ratio μ = {crack['mu']:.4f}")
    print(f"Utilization Ratio UR = {crack['UR']:.3f}")
    print(f"Status: {crack['status']}")
    
    # Deflection results
    deflect = results['deflection']
    print("\n" + "="*70)
    print("DEFLECTION CHECK")
    print("="*70)
    print(f"Cracking moment M_cr = {deflect['M_cr']:.2f} kN·m")
    print(f"Section state: {deflect['section_state'].upper()}")
    print(f"Gross inertia Ig = {deflect['Ig']:.0f} mm⁴")
    print(f"Cracked inertia Icr = {deflect['Icr']:.0f} mm⁴")
    print(f"Effective inertia Ie = {deflect['Ie']:.0f} mm⁴")
    print(f"Immediate deflection δ₀ = {deflect['delta_immediate']:.2f} mm")
    print(f"Long-term deflection δ = {deflect['delta']:.2f} mm")
    print(f"Allowable δ_limit = {deflect['delta_limit']:.2f} mm (L/250)")
    print(f"Utilization Ratio UR = {deflect['UR']:.3f}")
    print(f"Status: {deflect['status']}")
    
    print("\n" + "="*70)
    print("DETAILED CALCULATIONS")
    print("="*70)
    
    print("\n--- CRACK WIDTH DETAILS ---")
    print(crack['details'])
    
    print("\n--- DEFLECTION DETAILS ---")
    print(deflect['details'])
    
    print("="*70 + "\n")
    
    # Assertions
    assert crack['status'] in ['PASS', 'FAIL']
    assert deflect['status'] in ['PASS', 'FAIL']
    assert crack['a_cr'] > 0
    assert deflect['delta'] > 0


def test_serviceability_comparison():
    """Compare different load durations and environments."""
    print("\n" + "="*70)
    print("SERVICEABILITY PARAMETER COMPARISON")
    print("="*70)
    
    # Base case: normal environment, short-term
    base_case = run_sls_check_from_ui(
        beam_id=1,
        b=300, h=500, cover=30,
        As=1800,
        bar_diameter=20,
        concrete_name="B30",
        steel_name="CB400V",
        M_ser=100,
        L=6.0,
        environment='normal',
        load_duration='short_term'
    )
    
    # Case 2: Long-term loading
    long_term = run_sls_check_from_ui(
        beam_id=1,
        b=300, h=500, cover=30,
        As=1800,
        bar_diameter=20,
        concrete_name="B30",
        steel_name="CB400V",
        M_ser=100,
        L=6.0,
        environment='normal',
        load_duration='long_term'
    )
    
    # Case 3: Aggressive environment
    aggressive = run_sls_check_from_ui(
        beam_id=1,
        b=300, h=500, cover=30,
        As=1800,
        bar_diameter=20,
        concrete_name="B30",
        steel_name="CB400V",
        M_ser=100,
        L=6.0,
        environment='aggressive',
        load_duration='short_term'
    )
    
    print("\n--- SHORT-TERM vs LONG-TERM DEFLECTION ---")
    print(f"Short-term: δ = {base_case['deflection']['delta']:.2f} mm")
    print(f"Long-term:  δ = {long_term['deflection']['delta']:.2f} mm")
    ratio = long_term['deflection']['delta'] / base_case['deflection']['delta']
    print(f"Ratio (long/short): {ratio:.2f} (should be ≈ 3.0 with creep factor ξ=2.0)")
    
    print("\n--- NORMAL vs AGGRESSIVE ENVIRONMENT ---")
    print(f"Normal:     a_limit = {base_case['crack_width']['a_limit']:.1f} mm")
    print(f"Aggressive: a_limit = {aggressive['crack_width']['a_limit']:.1f} mm")
    print(f"Same crack width: a_cr = {base_case['crack_width']['a_cr']:.3f} mm")
    
    print("\n--- UTILIZATION RATIOS ---")
    print(f"Base case (normal, short):  Crack UR = {base_case['crack_width']['UR']:.2f}, Deflect UR = {base_case['deflection']['UR']:.2f}")
    print(f"Long-term (creep):         Deflect UR = {long_term['deflection']['UR']:.2f}")
    print(f"Aggressive environment:    Crack UR = {aggressive['crack_width']['UR']:.2f}")
    
    print("\n" + "="*70 + "\n")
    
    # Verify creep factor application
    assert ratio == pytest.approx(3.0, rel=0.01)
    
    # Verify environment limits
    assert base_case['crack_width']['a_limit'] == 0.4
    assert aggressive['crack_width']['a_limit'] == 0.3


def test_varying_reinforcement():
    """Test effect of reinforcement ratio on serviceability."""
    print("\n" + "="*70)
    print("REINFORCEMENT RATIO EFFECT")
    print("="*70)
    
    steel_areas = [1200, 1800, 2400]  # mm²
    
    for As in steel_areas:
        result = run_sls_check_from_ui(
            beam_id=1,
            b=300, h=500, cover=30,
            As=As,
            bar_diameter=20,
            concrete_name="B30",
            steel_name="CB400V",
            M_ser=100,
            L=6.0,
            environment='normal',
            load_duration='short_term'
        )
        
        crack = result['crack_width']
        deflect = result['deflection']
        
        print(f"\n--- As = {As} mm² (μ = {crack['mu']:.4f}) ---")
        print(f"Crack width: {crack['a_cr']:.3f} mm (UR = {crack['UR']:.2f}, {crack['status']})")
        print(f"Deflection:  {deflect['delta']:.2f} mm (UR = {deflect['UR']:.2f}, {deflect['status']})")
        print(f"Stress:      {crack['sigma_s']:.1f} MPa")


if __name__ == "__main__":
    import pytest
    
    print("\n" + "="*70)
    print("RUNNING SERVICEABILITY INTEGRATION TESTS")
    print("="*70)
    
    try:
        print("\n[TEST 1] Complete SLS Workflow")
        test_complete_sls_workflow()
        print("\n✅ Test 1 passed!")
        
        print("\n[TEST 2] Serviceability Comparison")
        test_serviceability_comparison()
        print("\n✅ Test 2 passed!")
        
        print("\n[TEST 3] Varying Reinforcement")
        test_varying_reinforcement()
        print("\n✅ Test 3 passed!")
        
        print("\n" + "="*70)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        raise
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        raise
