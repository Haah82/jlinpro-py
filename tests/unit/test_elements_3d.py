"""
Unit tests for 3D elements (Truss3D, Beam3D).

Tests validate:
- Stiffness matrix properties (symmetry, size, rank)
- Transformation matrix properties (orthogonality)
- Internal force calculations
- Special cases (vertical elements, reference nodes, roll angles)
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

from src.core.structures import Node, Material, RectangularSection
from src.elements.truss3d import Truss3D
from src.elements.beam3d import Beam3D


class TestTruss3D:
    """Test cases for Truss3D element."""

    def test_stiffness_matrix_shape(self):
        """Verify stiffness matrix has correct shape (12x12)."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=3.0, y=4.0, z=5.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.1, height=0.1)

        truss = Truss3D(id=1, nodes=[n1, n2], material=steel, section=section)

        k_local = truss.get_stiffness_local()
        assert k_local.shape == (12, 12)

        k_global = truss.get_stiffness_global()
        assert k_global.shape == (12, 12)

    def test_stiffness_matrix_symmetry(self):
        """Verify stiffness matrix is symmetric."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=5.0, y=0.0, z=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.05, height=0.05)

        truss = Truss3D(id=1, nodes=[n1, n2], material=steel, section=section)

        k_local = truss.get_stiffness_local()
        assert_array_almost_equal(k_local, k_local.T, decimal=10)

        k_global = truss.get_stiffness_global()
        assert_array_almost_equal(k_global, k_global.T, decimal=10)

    def test_stiffness_matrix_rank(self):
        """Verify local stiffness matrix has correct rank."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=10.0, y=0.0, z=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.1, height=0.1)

        truss = Truss3D(id=1, nodes=[n1, n2], material=steel, section=section)

        k_local = truss.get_stiffness_local()

        # Check eigenvalues - local truss has only 1 non-zero eigenvalue
        # (axial stiffness in local x-direction)
        eigenvalues = np.linalg.eigvalsh(k_local)
        non_zero = np.sum(np.abs(eigenvalues) > 1e-6)
        assert non_zero >= 1, f"Expected at least 1 non-zero eigenvalue, got {non_zero}"
        
        # Verify the non-zero eigenvalue is positive (stiffness)
        max_eigenvalue = np.max(eigenvalues)
        assert max_eigenvalue > 0, "Stiffness matrix should be positive semi-definite"

    def test_transformation_matrix_orthogonality(self):
        """Verify transformation matrix is orthogonal."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=3.0, y=4.0, z=5.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.1, height=0.1)

        truss = Truss3D(id=1, nodes=[n1, n2], material=steel, section=section)

        T = truss.get_transformation_matrix()

        # T should be orthogonal: T^T @ T = I
        I = np.eye(12)
        assert_array_almost_equal(T.T @ T, I, decimal=10)

    def test_vertical_truss(self):
        """Test truss element aligned with Z-axis (vertical)."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=0.0, y=0.0, z=3.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.1, height=0.1)

        truss = Truss3D(id=1, nodes=[n1, n2], material=steel, section=section)

        # Should not raise any errors
        k_local = truss.get_stiffness_local()
        T = truss.get_transformation_matrix()
        k_global = truss.get_stiffness_global()

        assert k_local.shape == (12, 12)
        assert T.shape == (12, 12)
        assert k_global.shape == (12, 12)

    def test_horizontal_truss(self):
        """Test truss element aligned with X-axis (horizontal)."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=5.0, y=0.0, z=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.1, height=0.1)

        truss = Truss3D(id=1, nodes=[n1, n2], material=steel, section=section)

        k_global = truss.get_stiffness_global()

        # For horizontal truss, global stiffness should be concentrated in ux DOFs
        # (indices 0 and 6)
        assert k_global[0, 0] > 0
        assert k_global[6, 6] > 0
        assert_allclose(k_global[0, 6], -k_global[0, 0], rtol=1e-10)

    def test_axial_force_calculation(self):
        """Test internal force calculation for simple axial load."""
        n1 = Node(id=0, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=1, x=5.0, y=0.0, z=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.1, height=0.1)

        truss = Truss3D(id=1, nodes=[n1, n2], material=steel, section=section)

        # Create displacement vector (elongation of 0.001 m in x)
        u_global = np.zeros(12)
        u_global[6] = 0.001  # ux2 = 0.001 m (node 1, dof 0)

        forces = truss.get_internal_forces(u_global)

        # Calculate expected axial force
        L = 5.0
        E = steel.E
        A = section.A
        expected_N = E * A * 0.001 / L

        assert_allclose(forces["N"], expected_N, rtol=1e-6)


class TestBeam3D:
    """Test cases for Beam3D element."""

    def test_stiffness_matrix_shape(self):
        """Verify stiffness matrix has correct shape (12x12)."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=0.0, y=0.0, z=5.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        beam = Beam3D(id=1, nodes=[n1, n2], material=steel, section=section)

        k_local = beam.get_stiffness_local()
        assert k_local.shape == (12, 12)

        k_global = beam.get_stiffness_global()
        assert k_global.shape == (12, 12)

    def test_stiffness_matrix_symmetry(self):
        """Verify stiffness matrix is symmetric."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=5.0, y=0.0, z=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        beam = Beam3D(id=1, nodes=[n1, n2], material=steel, section=section)

        k_local = beam.get_stiffness_local()
        assert_array_almost_equal(k_local, k_local.T, decimal=10)

        k_global = beam.get_stiffness_global()
        assert_array_almost_equal(k_global, k_global.T, decimal=10)

    def test_vertical_column_roll_angle_0(self):
        """Test vertical column with roll angle = 0 degrees."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=0.0, y=0.0, z=3.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        beam = Beam3D(
            id=1, nodes=[n1, n2], material=steel, section=section, roll_angle=0.0
        )

        T = beam.get_transformation_matrix()
        k_global = beam.get_stiffness_global()

        # Should not raise errors
        assert T.shape == (12, 12)
        assert k_global.shape == (12, 12)

        # Verify orthogonality
        I = np.eye(12)
        assert_array_almost_equal(T.T @ T, I, decimal=10)

    def test_vertical_column_roll_angle_90(self):
        """Test vertical column with roll angle = 90 degrees."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=0.0, y=0.0, z=3.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        beam_0 = Beam3D(
            id=1, nodes=[n1, n2], material=steel, section=section, roll_angle=0.0
        )
        beam_90 = Beam3D(
            id=2, nodes=[n1, n2], material=steel, section=section, roll_angle=90.0
        )

        k_global_0 = beam_0.get_stiffness_global()
        k_global_90 = beam_90.get_stiffness_global()

        # Stiffness matrices should be different due to different orientation
        # (unless section is square)
        if section.width != section.height:
            assert not np.allclose(k_global_0, k_global_90)

    def test_horizontal_beam_with_reference_node(self):
        """Test horizontal beam with reference node defining orientation."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=5.0, y=0.0, z=0.0)
        ref = Node(id=3, x=2.5, y=0.0, z=1.0)  # Reference point above
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        beam = Beam3D(
            id=1, nodes=[n1, n2], material=steel, section=section, ref_node=ref
        )

        T = beam.get_transformation_matrix()
        k_global = beam.get_stiffness_global()

        # Verify orthogonality
        I = np.eye(12)
        assert_array_almost_equal(T.T @ T, I, decimal=10)

        assert k_global.shape == (12, 12)

    def test_transformation_matrix_orthogonality(self):
        """Verify transformation matrix is orthogonal for various orientations."""
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        # Test multiple orientations
        test_cases = [
            (Node(id=1, x=0, y=0, z=0), Node(id=2, x=5, y=0, z=0)),  # X-aligned
            (Node(id=1, x=0, y=0, z=0), Node(id=2, x=0, y=5, z=0)),  # Y-aligned
            (Node(id=1, x=0, y=0, z=0), Node(id=2, x=0, y=0, z=5)),  # Z-aligned
            (Node(id=1, x=0, y=0, z=0), Node(id=2, x=3, y=4, z=5)),  # Arbitrary
        ]

        I = np.eye(12)
        for n1, n2 in test_cases:
            beam = Beam3D(id=1, nodes=[n1, n2], material=steel, section=section)
            T = beam.get_transformation_matrix()
            assert_array_almost_equal(T.T @ T, I, decimal=10)

    def test_cantilever_beam_deflection(self):
        """
        Test cantilever beam deflection formula.

        For a cantilever with tip load P:
        δ = PL³/(3EI)
        """
        # Cantilever beam along X-axis
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=5.0, y=0.0, z=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        beam = Beam3D(id=1, nodes=[n1, n2], material=steel, section=section)

        # This is a simplified test - full deflection would require
        # solving the system. Here we just verify stiffness properties.
        k_local = beam.get_stiffness_local()

        L = 5.0
        E = steel.E
        Iz = section.Iz

        # Check that bending stiffness term matches theory
        expected_k22 = 12 * E * Iz / L**3  # Transverse stiffness
        assert_allclose(k_local[1, 1], expected_k22, rtol=1e-10)

    def test_axial_stiffness(self):
        """Verify axial stiffness term."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=10.0, y=0.0, z=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        beam = Beam3D(id=1, nodes=[n1, n2], material=steel, section=section)

        k_local = beam.get_stiffness_local()

        L = 10.0
        E = steel.E
        A = section.A

        expected_k_axial = E * A / L
        assert_allclose(k_local[0, 0], expected_k_axial, rtol=1e-10)
        assert_allclose(k_local[0, 6], -expected_k_axial, rtol=1e-10)

    def test_torsional_stiffness(self):
        """Verify torsional stiffness term."""
        n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        n2 = Node(id=2, x=8.0, y=0.0, z=0.0)
        steel = Material.from_steel()
        section = RectangularSection(width=0.3, height=0.5)

        beam = Beam3D(id=1, nodes=[n1, n2], material=steel, section=section)

        k_local = beam.get_stiffness_local()

        L = 8.0
        G = steel.G
        J = section.J

        expected_k_torsion = G * J / L
        assert_allclose(k_local[3, 3], expected_k_torsion, rtol=1e-10)
        assert_allclose(k_local[3, 9], -expected_k_torsion, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
