"""
Integration test: Verify beam design UI integration
"""

from src.design.tcvn_beam import run_beam_check_from_ui


def test_typical_beam_design():
    """Test a typical beam design scenario."""
    # Typical 300x500 mm beam with B30 concrete, CB400V steel
    # Applied loads: M = 220 kN·m, V = 110 kN
    
    results = run_beam_check_from_ui(
        beam_id=1,
        b=300,
        h=500,
        cover=30,
        As_top=982,  # 4φ18 bars
        As_bot=1964,  # 4φ25 bars
        stirrup_dia=10,
        n_legs=2,
        spacing=150,
        concrete_name="B30",
        steel_name="CB400V",
        M_u=220,
        Q_u=110
    )
    
    print("\n" + "="*60)
    print("BEAM DESIGN CHECK RESULTS")
    print("="*60)
    print(f"\nBeam ID: {results['beam_id']}")
    
    print("\n--- FLEXURE CHECK ---")
    flex = results['flexure']
    print(f"Status: {flex['status']}")
    print(f"α_m = {flex['alpha_m']:.4f}")
    print(f"ξ = {flex['xi']:.4f} (limit = {flex['xi_R']:.2f})")
    print(f"As required = {flex['As_req']:.0f} mm²")
    print(f"M capacity = {flex['M_cap']:.1f} kN·m")
    print(f"Utilization Ratio = {flex['UR']:.2f}")
    
    print("\n--- SHEAR CHECK ---")
    shear = results['shear']
    print(f"Status: {shear['status']}")
    print(f"Qb (concrete) = {shear['Qb']:.1f} kN")
    print(f"Qsw (stirrups) = {shear['Qsw']:.1f} kN")
    print(f"Q capacity = {shear['Q_cap']:.1f} kN")
    print(f"Q crush limit = {shear['Q_crush']:.1f} kN")
    print(f"Utilization Ratio = {shear['UR']:.2f}")
    
    print("\n" + "="*60)
    print("\nDETAILS:")
    print(flex['details'])
    print("\n" + shear['details'])
    print("="*60 + "\n")
    
    # Assertions
    assert results['flexure']['status'] == 'PASS', "Flexure check should pass"
    assert results['shear']['status'] == 'PASS', "Shear check should pass"
    assert results['flexure']['UR'] < 1.0, "Flexure UR should be < 1.0"
    assert results['shear']['UR'] < 1.0, "Shear UR should be < 1.0"


if __name__ == "__main__":
    test_typical_beam_design()
    print("\n✅ Integration test passed successfully!")
