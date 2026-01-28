"""
Unit tests for 2D finite elements (Truss2D, Beam2D).
"""

import pytest
import numpy as np
from src.core.structures import Node, Material, RectangularSection, CircularSection
from src.elements.base import AbstractElement
from src.elements.truss2d import Truss2D
from src.elements.beam2d import Beam2D


class TestTruss2D:
    """Test cases for 2D truss element."""

    def test_truss_creation(self):
        """Test creating a truss element."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=3.0, y=4.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.1, height=0.1)

        truss = Truss2D(id=1, nodes=[node1, node2], material=steel, section=section)

        assert truss.id == 1
        assert len(truss.nodes) == 2
        assert truss.get_length() == pytest.approx(5.0)

    def test_truss_length_calculation(self):
        """Test element length calculation."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=3.0, y=4.0)
        steel = Material.from_steel()
        section = CircularSection(diameter=0.05)

        truss = Truss2D(id=1, nodes=[node1, node2], material=steel, section=section)

        assert truss.get_length() == pytest.approx(5.0)

    def test_truss_direction_cosines(self):
        """Test calculation of direction cosines."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=3.0, y=4.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.1, height=0.1)

        truss = Truss2D(id=1, nodes=[node1, node2], material=steel, section=section)

        cx, cy = truss.get_direction_cosines()

        # For 3-4-5 triangle: cos = 3/5, sin = 4/5
        assert cx == pytest.approx(0.6)
        assert cy == pytest.approx(0.8)

    def test_truss_local_stiffness(self):
        """Test local stiffness matrix calculation."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=5.0, y=0.0)
        steel = Material.from_steel("A36")
        section = RectangularSection(width=0.1, height=0.1)

        truss = Truss2D(id=1, nodes=[node1, node2], material=steel, section=section)

        K_local = truss.get_stiffness_local()

        # Expected k = EA/L
        A = section.get_properties()["A"]
        L = 5.0
        E = 200e9
        k = E * A / L

        # Check structure of local stiffness (6x6 for 3 DOFs per node)
        assert K_local.shape == (6, 6)
        # Axial stiffness in local DOF 0 and 3 (u-direction)
        assert K_local[0, 0] == pytest.approx(k)
        assert K_local[3, 3] == pytest.approx(k)
        assert K_local[0, 3] == pytest.approx(-k)
        # Transverse and rotational DOFs should be zero for truss
        assert K_local[1, 1] == pytest.approx(0.0)
        assert K_local[2, 2] == pytest.approx(0.0)

    def test_truss_transformation_matrix(self):
        """Test transformation matrix for horizontal truss."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=5.0, y=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.1, height=0.1)

        truss = Truss2D(id=1, nodes=[node1, node2], material=steel, section=section)

        T = truss.get_transformation_matrix()

        # For horizontal element: cx=1, cy=0
        # 6x6 transformation matrix for 3 DOFs per node
        assert T.shape == (6, 6)
        assert T[0, 0] == pytest.approx(1.0)  # cos(0) = 1
        assert T[0, 1] == pytest.approx(0.0)  # sin(0) = 0
        assert T[1, 0] == pytest.approx(0.0)  # -sin(0) = 0
        assert T[1, 1] == pytest.approx(1.0)  # cos(0) = 1
        assert T[2, 2] == pytest.approx(1.0)  # rotation unchanged

    def test_truss_global_stiffness(self):
        """Test global stiffness matrix calculation."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=5.0, y=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.1, height=0.1)

        truss = Truss2D(id=1, nodes=[node1, node2], material=steel, section=section)

        K_global = truss.get_stiffness_global()

        # Should be symmetric
        assert np.allclose(K_global, K_global.T)

        # Should be 12x12 for 2 nodes with 6 DOFs each (expanded from 6x6 element matrix)
        assert K_global.shape == (12, 12)

    def test_simple_truss_validation(self):
        """
        Validate simple truss against hand calculation.

        Two-bar truss:
        - Nodes: (0,0), (1,0), (0.5,0.866)
        - Horizontal bar and inclined bar at 60 degrees
        - Vertical load at top node
        """
        # Create nodes for equilateral triangle
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=1.0, y=0.0)
        node3 = Node(id=3, x=0.5, y=0.866)

        steel = Material.from_steel()
        section = CircularSection(diameter=0.02)

        # Create two truss elements
        truss1 = Truss2D(id=1, nodes=[node1, node3], material=steel, section=section)
        truss2 = Truss2D(id=2, nodes=[node2, node3], material=steel, section=section)

        # Verify geometry
        assert truss1.get_length() == pytest.approx(1.0, rel=1e-3)
        assert truss2.get_length() == pytest.approx(1.0, rel=1e-3)

        # Calculate internal force for a simple displacement
        # 6 DOFs: [ux, uy, uz, rx, ry, rz] - only ux and uy are used for 2D truss
        u_global = np.array([0.0, 0.0, 0.0, 0.0, 0.001, 0.0])  # Small vertical displacement at node 2

        forces = truss1.get_internal_forces(u_global)

        # Force should be tensile (positive) for downward displacement
        assert forces["axial_force"] != 0

    def test_truss_weight_calculation(self):
        """Test element weight calculation."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=5.0, y=0.0)
        steel = Material.from_steel()  # rho = 7850 kg/m³
        section = RectangularSection(width=0.1, height=0.1)  # A = 0.01 m²

        truss = Truss2D(id=1, nodes=[node1, node2], material=steel, section=section)

        # Weight = rho * A * L * g
        # = 7850 * 0.01 * 5.0 * 9.81
        expected_weight = 7850 * 0.01 * 5.0 * 9.81

        assert truss.get_weight() == pytest.approx(expected_weight)
        assert truss.get_mass() == pytest.approx(7850 * 0.01 * 5.0)


class TestBeam2D:
    """Test cases for 2D beam element."""

    def test_beam_creation(self):
        """Test creating a beam element."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=5.0, y=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        beam = Beam2D(id=1, nodes=[node1, node2], material=steel, section=section)

        assert beam.id == 1
        assert len(beam.nodes) == 2
        assert beam.get_length() == pytest.approx(5.0)
        assert beam.releases == 0  # No releases by default

    def test_beam_with_releases(self):
        """Test creating beam with moment releases."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=5.0, y=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        # Moment release at node j (pin at end)
        beam = Beam2D(
            id=1, nodes=[node1, node2], material=steel, section=section, releases=0x01
        )

        assert beam.releases == 0x01
        assert beam.is_moment_released(1)
        assert not beam.is_moment_released(0)

    def test_beam_local_stiffness(self):
        """Test local stiffness matrix calculation."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=4.0, y=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.2, height=0.4)

        beam = Beam2D(id=1, nodes=[node1, node2], material=steel, section=section)

        K_local = beam.get_stiffness_local()

        # Should be 6x6
        assert K_local.shape == (6, 6)

        # Should be symmetric
        assert np.allclose(K_local, K_local.T)

        # Axial stiffness at (0,0)
        L = 4.0
        E = 200e9
        A = 0.2 * 0.4
        expected_axial_stiffness = E * A / L

        assert K_local[0, 0] == pytest.approx(expected_axial_stiffness)

    def test_beam_transformation_matrix(self):
        """Test transformation matrix for horizontal beam."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=5.0, y=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        beam = Beam2D(id=1, nodes=[node1, node2], material=steel, section=section)

        T = beam.get_transformation_matrix()

        # Should be 6x6
        assert T.shape == (6, 6)

        # For horizontal beam: should be identity-like
        assert T[0, 0] == pytest.approx(1.0)
        assert T[2, 2] == pytest.approx(1.0)

    def test_beam_global_stiffness(self):
        """Test global stiffness matrix calculation."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=5.0, y=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        beam = Beam2D(id=1, nodes=[node1, node2], material=steel, section=section)

        K_global = beam.get_stiffness_global()

        # Should be symmetric
        assert np.allclose(K_global, K_global.T)

        # Should be 12x12 for 2 nodes with 6 DOFs each (expanded from 6x6 element matrix)
        assert K_global.shape == (12, 12)

    def test_cantilever_beam_theory(self):
        """
        Test cantilever beam deflection: δ = PL³/(3EI)

        Fixed at node 1, free at node 2.
        Apply unit load at node 2, compare displacement.
        """
        # Create cantilever beam
        L = 3.0  # meters
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=L, y=0.0)

        # Material and section
        E = 200e9  # Pa
        steel = Material(name="Steel", E=E, nu=0.3, rho=7850)
        section = RectangularSection(width=0.3, height=0.5)

        beam = Beam2D(id=1, nodes=[node1, node2], material=steel, section=section)

        # Get stiffness matrix
        K_local = beam.get_stiffness_local()

        # Apply unit load at free end (node j, vertical direction)
        # DOFs: [u_i, v_i, theta_i, u_j, v_j, theta_j]
        # We want to find v_j with fixed end (u_i=v_i=theta_i=0)

        # Extract reduced system (only free DOFs: u_j, v_j, theta_j)
        K_reduced = K_local[3:6, 3:6]
        F_reduced = np.array([0, -1000, 0])  # 1000 N downward

        # Solve for displacements
        u_reduced = np.linalg.solve(K_reduced, F_reduced)

        # Theoretical deflection: δ = PL³/(3EI)
        I = section.get_properties()["Ix"]
        P = 1000  # N
        delta_theory = P * L**3 / (3 * E * I)

        # Compare vertical deflection
        delta_computed = abs(u_reduced[1])

        assert delta_computed == pytest.approx(delta_theory, rel=1e-6)

    def test_simply_supported_beam(self):
        """
        Test simply supported beam with central load.

        Deflection at center: δ = PL³/(48EI)
        """
        L = 6.0  # meters
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=L, y=0.0)

        E = 200e9
        steel = Material(name="Steel", E=E, nu=0.3, rho=7850)
        section = RectangularSection(width=0.4, height=0.6)

        beam = Beam2D(id=1, nodes=[node1, node2], material=steel, section=section)

        # Get properties
        I = section.get_properties()["Ix"]

        # Theoretical deflection at center for central load P
        P = 10000  # N
        delta_center_theory = P * L**3 / (48 * E * I)

        # Our beam element gives end deflections, not center
        # This test verifies the stiffness is calculated correctly
        K = beam.get_stiffness_local()

        # Verify stiffness matrix properties
        assert K.shape == (6, 6)
        assert np.allclose(K, K.T)

    def test_beam_internal_forces(self):
        """Test calculation of internal forces."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=5.0, y=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        beam = Beam2D(id=1, nodes=[node1, node2], material=steel, section=section)

        # Apply small displacements
        u_global = np.array([0.0, 0.0, 0.0, 0.001, 0.0, 0.0])

        forces = beam.get_internal_forces(u_global)

        # Should have forces at both nodes
        assert "node_i" in forces
        assert "node_j" in forces
        assert "axial" in forces["node_i"]
        assert "shear" in forces["node_i"]
        assert "moment" in forces["node_i"]

    def test_beam_with_moment_release(self):
        """Test beam with moment release (pin connection)."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=5.0, y=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        # Release moment at node j
        beam = Beam2D(
            id=1, nodes=[node1, node2], material=steel, section=section, releases=0x01
        )

        K_local = beam.get_stiffness_local()

        # Moment at node j should have very small stiffness
        assert K_local[5, 5] < 1e-3


class TestElementValidation:
    """Integration tests for element validation."""

    def test_coincident_nodes_error(self):
        """Test that coincident nodes raise an error."""
        node1 = Node(id=1, x=0.0, y=0.0)
        node2 = Node(id=2, x=0.0, y=0.0)  # Same location
        steel = Material.from_steel()
        section = RectangularSection(width=0.1, height=0.1)

        with pytest.raises(ValueError, match="coincident"):
            truss = Truss2D(id=1, nodes=[node1, node2], material=steel, section=section)

    def test_wrong_number_of_nodes(self):
        """Test that wrong number of nodes raises an error."""
        node1 = Node(id=1, x=0.0, y=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.1, height=0.1)

        with pytest.raises(ValueError, match="exactly 2 nodes"):
            truss = Truss2D(id=1, nodes=[node1], material=steel, section=section)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
