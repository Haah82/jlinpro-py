"""
Integration test: Verify column design UI integration and visualization
"""

import numpy as np
from src.design.tcvn_column import (
    run_column_check_from_ui, 
    generate_interaction_curve,
    ColumnSection
)
from src.design.tcvn_setup import DesignContext


def test_typical_column_design():
    """Test a typical column design scenario with visualization."""
    # Typical 400x400 mm column, 3.5m effective length, B30 concrete, CB400V steel
    # Applied loads: N = 1000 kN, M = 120 kN·m
    
    results = run_column_check_from_ui(
        col_id=1,
        b=400,
        h=400,
        L_eff=3.5,  # m
        cover=40,
        As_total=2400,  # 1.5% reinforcement ratio
        concrete_name="B30",
        steel_name="CB400V",
        N_u=1000,  # kN (compression)
        M_u=120  # kN·m
    )
    
    print("\n" + "="*70)
    print("COLUMN DESIGN CHECK RESULTS")
    print("="*70)
    print(f"\nColumn ID: {results['col_id']}")
    
    print("\n--- SLENDERNESS ---")
    print(f"Slenderness ratio λ = {results['lambda']:.1f}")
    if results['lambda'] > 14:
        print(f"Classification: SLENDER column")
        print(f"Magnification factor η = {results['eta']:.3f}")
        print(f"Amplified moment M = {results['M_amplified']:.2f} kN·m")
    else:
        print(f"Classification: SHORT column")
        print(f"No second-order effects (η = 1.0)")
    
    print("\n--- CAPACITY CHECK ---")
    print(f"Applied axial load N_u = {1000:.1f} kN")
    print(f"Applied moment M_u = {results['M_amplified']:.1f} kN·m (amplified)")
    print(f"Moment capacity at N_u = {results['M_cap_at_Nu']:.1f} kN·m")
    print(f"Utilization Ratio UR = {results['UR']:.3f}")
    
    print("\n--- RESULT ---")
    if results['status'] == 'PASS':
        print(f"✅ Column check: PASS")
    else:
        print(f"❌ Column check: FAIL")
    
    print("\n" + "="*70)
    print("\nDETAILS:")
    print(results['details'])
    print("="*70 + "\n")
    
    # Verify figure was created
    assert results['figure'] is not None, "Interaction diagram should be created"
    
    # Verify figure has data
    fig = results['figure']
    assert len(fig.data) >= 2, "Figure should have capacity curve and applied point"
    
    # Print figure data summary
    print("--- INTERACTION DIAGRAM DATA ---")
    capacity_trace = fig.data[0]
    print(f"Capacity curve points: {len(capacity_trace.x)}")
    print(f"N range: [{min(capacity_trace.y):.1f}, {max(capacity_trace.y):.1f}] kN")
    print(f"M range: [{min(capacity_trace.x):.1f}, {max(capacity_trace.x):.1f}] kN·m")
    
    applied_trace = fig.data[1]
    print(f"\nApplied load point:")
    print(f"  N = {applied_trace.y[0]:.1f} kN")
    print(f"  M = {applied_trace.x[0]:.1f} kN·m")
    
    # Assertions
    assert results['status'] in ['PASS', 'FAIL'], "Status should be PASS or FAIL"
    assert results['UR'] > 0, "UR should be positive"
    assert results['lambda'] > 0, "Slenderness should be positive"
    assert results['eta'] >= 1.0, "Magnification factor should be ≥ 1.0"


def test_interaction_curve_properties():
    """Test properties of interaction curve."""
    from src.design.tcvn_column import ColumnSection
    
    section = ColumnSection(b=400, h=400, cover=40, L_eff=3000)
    context = DesignContext.from_ui_inputs("B30", "CB400V", 40)
    As_total = 2400  # mm²
    
    N_cap, M_cap = generate_interaction_curve(section, context, As_total, n_points=100)
    
    print("\n" + "="*70)
    print("INTERACTION CURVE ANALYSIS")
    print("="*70)
    
    # Convert to numpy arrays
    N_array = np.array(N_cap)
    M_array = np.array(M_cap)
    
    # Find key points
    idx_pure_compression = 0
    idx_pure_tension = len(N_cap) - 1
    idx_balanced = np.argmax(np.abs(M_array))
    
    print("\n--- KEY POINTS ON INTERACTION DIAGRAM ---")
    
    print(f"\nPoint A (Pure Compression):")
    print(f"  N = {N_cap[idx_pure_compression]:.1f} kN")
    print(f"  M = {M_cap[idx_pure_compression]:.1f} kN·m")
    
    print(f"\nPoint B (Balanced Failure):")
    print(f"  N = {N_cap[idx_balanced]:.1f} kN")
    print(f"  M = {M_cap[idx_balanced]:.1f} kN·m (maximum)")
    
    print(f"\nPoint C (Pure Tension):")
    print(f"  N = {N_cap[idx_pure_tension]:.1f} kN")
    print(f"  M = {M_cap[idx_pure_tension]:.1f} kN·m")
    
    # Verify theoretical values
    Rb = context.concrete.Rb
    Rsc = context.steel.Rsc
    Rs = context.steel.Rs
    b = section.b
    h = section.h
    
    # Pure compression
    N_pure_comp_theory = (Rb * b * h + (Rsc - Rb) * As_total) / 1000
    print(f"\n--- VALIDATION ---")
    print(f"Pure compression (theoretical): {N_pure_comp_theory:.1f} kN")
    print(f"Pure compression (calculated):  {N_cap[idx_pure_compression]:.1f} kN")
    print(f"Difference: {abs(N_cap[idx_pure_compression] - N_pure_comp_theory):.1f} kN")
    
    # Pure tension
    N_pure_tension_theory = -(Rs * As_total) / 1000
    print(f"\nPure tension (theoretical): {N_pure_tension_theory:.1f} kN")
    print(f"Pure tension (calculated):  {N_cap[idx_pure_tension]:.1f} kN")
    print(f"Difference: {abs(N_cap[idx_pure_tension] - N_pure_tension_theory):.1f} kN")
    
    print("\n" + "="*70 + "\n")
    
    # Assertions
    assert abs(N_cap[idx_pure_compression] - N_pure_comp_theory) < 50, \
        "Pure compression should match theory"
    assert abs(N_cap[idx_pure_tension] - N_pure_tension_theory) < 1, \
        "Pure tension should match theory"
    assert M_cap[idx_balanced] > 50, "Balanced moment should be significant"


def test_slender_vs_short_columns():
    """Compare slender and short columns side-by-side."""
    print("\n" + "="*70)
    print("SLENDER VS SHORT COLUMN COMPARISON")
    print("="*70)
    
    # Same section, different lengths
    test_cases = [
        {"name": "Short Column", "L_eff": 2.0, "lambda_expected": "<14"},
        {"name": "Slender Column", "L_eff": 5.0, "lambda_expected": ">14"}
    ]
    
    for case in test_cases:
        result = run_column_check_from_ui(
            col_id=1,
            b=300, h=300, L_eff=case['L_eff'], cover=30,
            As_total=1600,
            concrete_name="B25",
            steel_name="CB400V",
            N_u=500,
            M_u=80
        )
        
        print(f"\n--- {case['name']} (L = {case['L_eff']} m) ---")
        print(f"Slenderness λ = {result['lambda']:.1f} {case['lambda_expected']}")
        print(f"Magnification η = {result['eta']:.3f}")
        print(f"Original moment M = 80.0 kN·m")
        print(f"Amplified moment M = {result['M_amplified']:.2f} kN·m")
        print(f"Amplification = {(result['M_amplified']/80.0 - 1)*100:.1f}%")
        print(f"Utilization UR = {result['UR']:.3f}")
        print(f"Status: {result['status']}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING COLUMN DESIGN INTEGRATION TESTS")
    print("="*70)
    
    try:
        print("\n[TEST 1] Typical Column Design")
        test_typical_column_design()
        print("\n✅ Test 1 passed!")
        
        print("\n[TEST 2] Interaction Curve Properties")
        test_interaction_curve_properties()
        print("\n✅ Test 2 passed!")
        
        print("\n[TEST 3] Slender vs Short Columns")
        test_slender_vs_short_columns()
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
