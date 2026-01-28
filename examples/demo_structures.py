"""
Demonstration of core data structures (Node, Material, Section).

This script shows examples of using the fundamental classes for FEM analysis.
"""

from src.core.structures import (
    Node,
    Material,
    RectangularSection,
    CircularSection,
    ISection,
    CustomSection,
)


def main():
    print("=" * 60)
    print("JLinPro Python Migration - Core Data Structures Demo")
    print("=" * 60)

    # ===== Node Examples =====
    print("\n1. Node Examples:")
    print("-" * 60)

    # Create 2D nodes
    node1 = Node(id=1, x=0.0, y=0.0)
    node2 = Node(id=2, x=5.0, y=0.0)
    node3 = Node(id=3, x=5.0, y=3.0)

    # Apply different support conditions
    node1.set_fixed_support()
    node2.set_pinned_support()
    node3.set_roller_support(direction=1)

    print(f"Node 1 (Fixed):   {node1.restraints[:3]} - {node1.get_dofs()} DOFs")
    print(f"Node 2 (Pinned):  {node2.restraints[:3]} - {node2.get_dofs()} DOFs")
    print(f"Node 3 (Roller-Y): {node3.restraints[:3]} - {node3.get_dofs()} DOFs")

    # Calculate distance
    distance = node1.distance_to(node2)
    print(f"\nDistance from Node 1 to Node 2: {distance:.2f} m")

    # ===== Material Examples =====
    print("\n2. Material Examples:")
    print("-" * 60)

    # Steel material
    steel = Material.from_steel("A36")
    print(f"Steel A36:")
    print(f"  E  = {steel.E/1e9:.0f} GPa")
    print(f"  G  = {steel.G/1e9:.0f} GPa")
    print(f"  ν  = {steel.nu}")
    print(f"  ρ  = {steel.rho} kg/m³")

    # Concrete material
    concrete = Material.from_concrete(fc=30.0)
    print(f"\nConcrete fc=30 MPa:")
    print(f"  E  = {concrete.E/1e9:.2f} GPa")
    print(f"  G  = {concrete.G/1e9:.2f} GPa")
    print(f"  ν  = {concrete.nu}")
    print(f"  ρ  = {concrete.rho} kg/m³")

    # Timber material
    timber = Material.from_timber(species="Pine", grade="C24")
    print(f"\nPine C24:")
    print(f"  E  = {timber.E/1e9:.0f} GPa")
    print(f"  G  = {timber.G/1e9:.2f} GPa")
    print(f"  ρ  = {timber.rho} kg/m³")

    # ===== Section Examples =====
    print("\n3. Section Examples:")
    print("-" * 60)

    # Rectangular section
    rect = RectangularSection(width=0.3, height=0.5)
    rect_props = rect.get_properties()
    print("Rectangular Section (300mm x 500mm):")
    print(f"  A  = {rect_props['A']*1e4:.1f} cm²")
    print(f"  Ix = {rect_props['Ix']*1e8:.1f} cm⁴")
    print(f"  Iy = {rect_props['Iy']*1e8:.1f} cm⁴")
    print(f"  J  = {rect_props['J']*1e8:.2f} cm⁴")

    # Circular section
    circular = CircularSection(diameter=0.4)
    circ_props = circular.get_properties()
    print("\nSolid Circular Section (D=400mm):")
    print(f"  A  = {circ_props['A']*1e4:.1f} cm²")
    print(f"  I  = {circ_props['Ix']*1e8:.1f} cm⁴")
    print(f"  J  = {circ_props['J']*1e8:.1f} cm⁴")

    # Hollow circular (pipe)
    pipe = CircularSection(diameter=0.4, thickness=0.01)
    pipe_props = pipe.get_properties()
    print("\nHollow Circular Section (D=400mm, t=10mm):")
    print(f"  A  = {pipe_props['A']*1e4:.1f} cm²")
    print(f"  I  = {pipe_props['Ix']*1e8:.1f} cm⁴")
    print(f"  J  = {pipe_props['J']*1e8:.1f} cm⁴")

    # I-section
    i_beam = ISection(
        flange_width=0.2, flange_thickness=0.015, web_height=0.4, web_thickness=0.01
    )
    i_props = i_beam.get_properties()
    print("\nI-Section (bf=200mm, tf=15mm, h=400mm, tw=10mm):")
    print(f"  A  = {i_props['A']*1e4:.1f} cm²")
    print(f"  Ix = {i_props['Ix']*1e8:.1f} cm⁴ (strong axis)")
    print(f"  Iy = {i_props['Iy']*1e8:.2f} cm⁴ (weak axis)")
    print(f"  J  = {i_props['J']*1e8:.2f} cm⁴")

    # Custom section
    custom = CustomSection(name="Special Profile", A=0.025, Ix=2e-4, Iy=1e-4, J=1.5e-4)
    custom_props = custom.get_properties()
    print("\nCustom Section (from catalog):")
    print(f"  A  = {custom_props['A']*1e4:.1f} cm²")
    print(f"  Ix = {custom_props['Ix']*1e8:.1f} cm⁴")
    print(f"  Iy = {custom_props['Iy']*1e8:.1f} cm⁴")

    # ===== Practical Example =====
    print("\n4. Practical Example: RC Column Design")
    print("-" * 60)

    # Define a reinforced concrete column
    column_node_bottom = Node(id=10, x=0.0, y=0.0)
    column_node_top = Node(id=11, x=0.0, y=3.5)

    column_node_bottom.set_fixed_support()

    concrete_c30 = Material.from_concrete(fc=30.0)
    column_section = RectangularSection(width=0.4, height=0.4)

    L = column_node_bottom.distance_to(column_node_top)
    props = column_section.get_properties()

    # Calculate slenderness ratio
    r = (props["Ix"] / props["A"]) ** 0.5  # Radius of gyration
    slenderness = L / r

    print(f"Column Length: {L:.2f} m")
    print(
        f"Section: {column_section.width*1000:.0f}mm x {column_section.height*1000:.0f}mm"
    )
    print(f"Area: {props['A']*1e4:.1f} cm²")
    print(f"Moment of Inertia: {props['Ix']*1e8:.1f} cm⁴")
    print(f"Radius of gyration: {r*100:.2f} cm")
    print(f"Slenderness ratio (λ): {slenderness:.1f}")

    if slenderness < 22:
        print("→ Short column (use simplified design)")
    else:
        print("→ Slender column (consider second-order effects)")

    # Calculate axial stiffness
    EA = concrete_c30.E * props["A"]
    print(f"\nAxial stiffness (EA): {EA/1e6:.1f} MN")

    # Calculate flexural stiffness
    EI = concrete_c30.E * props["Ix"]
    print(f"Flexural stiffness (EI): {EI/1e3:.1f} kN·m²")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
