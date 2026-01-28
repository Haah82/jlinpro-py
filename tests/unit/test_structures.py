"""
Unit tests for core data structures (Node, Material, Section).
"""

import pytest
import numpy as np
from pydantic import ValidationError
from src.core.structures import (
    Node,
    Material,
    Section,
    RectangularSection,
    CircularSection,
    ISection,
    CustomSection,
)


class TestNode:
    """Test cases for Node class."""

    def test_node_creation_2d(self):
        """Test creating a 2D node."""
        node = Node(id=1, x=0.0, y=5.0)
        assert node.id == 1
        assert node.x == 0.0
        assert node.y == 5.0
        assert node.z == 0.0
        assert node.get_dofs() == 3
        assert not node.is_3d

    def test_node_creation_3d(self):
        """Test creating a 3D node."""
        node = Node(id=2, x=1.0, y=2.0, z=3.0)
        assert node.z == 3.0
        assert node.get_dofs() == 6
        assert node.is_3d

    def test_restraints_default(self):
        """Test default restraints are all free."""
        node = Node(id=1, x=0.0, y=0.0)
        for i in range(6):
            assert not node.is_restrained(i)

    def test_set_restraint(self):
        """Test setting individual restraints."""
        node = Node(id=1, x=0.0, y=0.0)
        node.set_restraint(0, True)
        node.set_restraint(1, True)

        assert node.is_restrained(0)
        assert node.is_restrained(1)
        assert not node.is_restrained(2)

    def test_fixed_support(self):
        """Test fixed support (all DOFs restrained)."""
        node = Node(id=1, x=0.0, y=0.0)
        node.set_fixed_support()

        for i in range(6):
            assert node.is_restrained(i)

    def test_pinned_support(self):
        """Test pinned support (translations fixed, rotations free)."""
        node = Node(id=1, x=0.0, y=0.0)
        node.set_pinned_support()

        # Translations fixed
        assert node.is_restrained(0)
        assert node.is_restrained(1)
        assert node.is_restrained(2)

        # Rotations free
        assert not node.is_restrained(3)
        assert not node.is_restrained(4)
        assert not node.is_restrained(5)

    def test_roller_support_y(self):
        """Test roller support in Y direction."""
        node = Node(id=1, x=0.0, y=0.0)
        node.set_roller_support(direction=1)

        assert not node.is_restrained(0)
        assert node.is_restrained(1)
        assert not node.is_restrained(2)

    def test_distance_between_nodes(self):
        """Test distance calculation between nodes."""
        node1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        node2 = Node(id=2, x=3.0, y=4.0, z=0.0)

        distance = node1.distance_to(node2)
        assert np.isclose(distance, 5.0)

    def test_distance_3d(self):
        """Test distance calculation in 3D."""
        node1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        node2 = Node(id=2, x=1.0, y=1.0, z=1.0)

        distance = node1.distance_to(node2)
        assert np.isclose(distance, np.sqrt(3))

    def test_get_coordinates(self):
        """Test getting coordinates as numpy array."""
        node = Node(id=1, x=1.5, y=2.5, z=3.5)
        coords = node.get_coordinates()

        assert isinstance(coords, np.ndarray)
        assert coords.shape == (3,)
        assert np.allclose(coords, [1.5, 2.5, 3.5])

    def test_restraint_index_validation(self):
        """Test that invalid DOF indices raise errors."""
        node = Node(id=1, x=0.0, y=0.0)

        with pytest.raises(IndexError):
            node.is_restrained(-1)

        with pytest.raises(IndexError):
            node.is_restrained(6)

        with pytest.raises(IndexError):
            node.set_restraint(10, True)


class TestMaterial:
    """Test cases for Material class."""

    def test_material_creation(self):
        """Test creating a custom material."""
        mat = Material(name="Test Material", E=200e9, nu=0.3, rho=7850)

        assert mat.name == "Test Material"
        assert mat.E == 200e9
        assert mat.nu == 0.3
        assert mat.rho == 7850

    def test_shear_modulus_calculation(self):
        """Test automatic shear modulus calculation."""
        mat = Material(name="Steel", E=200e9, nu=0.3, rho=7850)

        expected_G = 200e9 / (2 * (1 + 0.3))
        assert np.isclose(mat.G, expected_G)

    def test_steel_factory_method(self):
        """Test steel material factory method."""
        steel = Material.from_steel("A36")

        assert steel.name == "Steel A36"
        assert steel.E == 200e9
        assert steel.nu == 0.3
        assert steel.rho == 7850
        assert steel.alpha == 12e-6

    def test_concrete_factory_method(self):
        """Test concrete material factory method."""
        concrete = Material.from_concrete(fc=30.0)

        assert "30" in concrete.name
        # ACI formula: Ec = 4700*sqrt(30) ≈ 25.7 GPa
        expected_E = 4700 * np.sqrt(30) * 1e6
        assert np.isclose(concrete.E, expected_E)
        assert concrete.nu == 0.2
        assert concrete.rho == 2400

    def test_timber_factory_method(self):
        """Test timber material factory method."""
        timber = Material.from_timber(species="Pine", grade="C24")

        assert "Pine" in timber.name
        assert "C24" in timber.name
        assert timber.E == 11e9
        assert timber.rho == 420

    def test_invalid_E(self):
        """Test that negative E raises error."""
        with pytest.raises(ValidationError):
            Material(name="Invalid", E=-100, nu=0.3, rho=1000)

    def test_invalid_nu_low(self):
        """Test that nu <= 0 raises error."""
        with pytest.raises(ValidationError):
            Material(name="Invalid", E=200e9, nu=0.0, rho=1000)

    def test_invalid_nu_high(self):
        """Test that nu >= 0.5 raises error."""
        with pytest.raises(ValidationError):
            Material(name="Invalid", E=200e9, nu=0.5, rho=1000)


class TestRectangularSection:
    """Test cases for RectangularSection."""

    def test_rectangular_section_creation(self):
        """Test creating rectangular section."""
        section = RectangularSection(width=0.3, height=0.5)

        assert section.width == 0.3
        assert section.height == 0.5

    def test_rectangular_properties(self):
        """Test rectangular section property calculations."""
        b, h = 0.3, 0.5
        section = RectangularSection(width=b, height=h)
        props = section.get_properties()

        # Area
        assert np.isclose(props["A"], b * h)

        # Moment of inertia (strong axis)
        expected_Ix = b * h**3 / 12
        assert np.isclose(props["Ix"], expected_Ix)

        # Moment of inertia (weak axis)
        expected_Iy = h * b**3 / 12
        assert np.isclose(props["Iy"], expected_Iy)

        # Polar moment
        assert np.isclose(props["Iz"], props["Ix"] + props["Iy"])

        # Torsional constant (approximate)
        assert props["J"] > 0

    def test_square_section(self):
        """Test square section (special case of rectangular)."""
        section = RectangularSection(width=0.4, height=0.4)
        props = section.get_properties()

        # For square, Ix = Iy
        assert np.isclose(props["Ix"], props["Iy"])

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise errors."""
        with pytest.raises(ValueError):
            section = RectangularSection(width=-0.3, height=0.5)
            section.validate_geometry()

        with pytest.raises(ValueError):
            section = RectangularSection(width=0.3, height=-0.5)
            section.validate_geometry()


class TestCircularSection:
    """Test cases for CircularSection."""

    def test_solid_circular_section(self):
        """Test solid circular section."""
        D = 0.2
        section = CircularSection(diameter=D)
        props = section.get_properties()

        # Area
        expected_A = np.pi * D**2 / 4
        assert np.isclose(props["A"], expected_A)

        # Moment of inertia
        expected_I = np.pi * D**4 / 64
        assert np.isclose(props["Ix"], expected_I)
        assert np.isclose(props["Iy"], expected_I)

        # Torsional constant
        expected_J = np.pi * D**4 / 32
        assert np.isclose(props["J"], expected_J)

    def test_hollow_circular_section(self):
        """Test hollow circular section (pipe)."""
        D = 0.2
        t = 0.01
        section = CircularSection(diameter=D, thickness=t)
        props = section.get_properties()

        d = D - 2 * t  # Inner diameter

        # Area
        expected_A = np.pi * (D**2 - d**2) / 4
        assert np.isclose(props["A"], expected_A)

        # Moment of inertia
        expected_I = np.pi * (D**4 - d**4) / 64
        assert np.isclose(props["Ix"], expected_I)

        # Hollow section should have less area than solid
        solid = CircularSection(diameter=D)
        assert props["A"] < solid.get_properties()["A"]

    def test_invalid_thickness(self):
        """Test that thickness >= radius raises error."""
        with pytest.raises(ValueError, match="less than radius"):
            section = CircularSection(diameter=0.2, thickness=0.15)
            section.validate_geometry()


class TestISection:
    """Test cases for ISection."""

    def test_i_section_creation(self):
        """Test creating I-section."""
        section = ISection(
            flange_width=0.2, flange_thickness=0.015, web_height=0.4, web_thickness=0.01
        )

        assert section.flange_width == 0.2
        assert section.web_height == 0.4

    def test_i_section_properties(self):
        """Test I-section property calculations."""
        bf = 0.2
        tf = 0.015
        h = 0.4
        tw = 0.01

        section = ISection(
            flange_width=bf, flange_thickness=tf, web_height=h, web_thickness=tw
        )
        props = section.get_properties()

        # Area = 2 flanges + web
        hw = h - 2 * tf
        expected_A = 2 * bf * tf + hw * tw
        assert np.isclose(props["A"], expected_A)

        # Ix (strong axis) should be much larger than Iy (weak axis)
        assert props["Ix"] > props["Iy"]

        # All properties should be positive
        assert all(props[k] > 0 for k in ["A", "Ix", "Iy", "J"])

    def test_invalid_i_section(self):
        """Test that invalid I-section dimensions raise errors."""
        with pytest.raises(ValueError, match="Flange thickness too large"):
            section = ISection(
                flange_width=0.2,
                flange_thickness=0.25,  # Too large
                web_height=0.4,
                web_thickness=0.01,
            )
            section.validate_geometry()


class TestCustomSection:
    """Test cases for CustomSection."""

    def test_custom_section_creation(self):
        """Test creating custom section with known properties."""
        section = CustomSection(name="Special Shape", A=0.02, Ix=1e-4, Iy=5e-5, J=8e-5)

        assert section.name == "Special Shape"
        assert section.A == 0.02

    def test_custom_section_properties(self):
        """Test that custom section returns user-defined properties."""
        A, Ix, Iy, J = 0.025, 2e-4, 1e-4, 1.5e-4

        section = CustomSection(A=A, Ix=Ix, Iy=Iy, J=J)
        props = section.get_properties()

        assert props["A"] == A
        assert props["Ix"] == Ix
        assert props["Iy"] == Iy
        assert props["J"] == J
        assert props["Iz"] == Ix + Iy

    def test_invalid_custom_section(self):
        """Test that negative properties raise errors."""
        with pytest.raises(ValueError):
            section = CustomSection(A=-0.01, Ix=1e-4, Iy=5e-5, J=8e-5)
            section.validate_geometry()


class TestIntegration:
    """Integration tests combining multiple classes."""

    def test_complete_element_definition(self):
        """Test defining a complete structural element."""
        # Create nodes
        node1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        node2 = Node(id=2, x=5.0, y=0.0, z=0.0)

        # Fix first node
        node1.set_fixed_support()

        # Create material
        steel = Material.from_steel("A36")

        # Create section
        section = RectangularSection(width=0.3, height=0.5)

        # Verify all components are created properly
        assert node1.is_restrained(0)
        assert steel.E == 200e9
        assert section.get_properties()["A"] > 0

        # Calculate element length
        length = node1.distance_to(node2)
        assert np.isclose(length, 5.0)

    def test_cantilever_beam_setup(self):
        """Test setting up a cantilever beam problem."""
        # Fixed support at origin
        fixed_node = Node(id=1, x=0.0, y=0.0)
        fixed_node.set_fixed_support()

        # Free end at x=3m
        free_node = Node(id=2, x=3.0, y=0.0)

        # Material and section
        concrete = Material.from_concrete(fc=25.0)
        section = RectangularSection(width=0.3, height=0.4)

        # Verify setup
        assert fixed_node.is_restrained(0)
        assert fixed_node.is_restrained(1)
        assert not free_node.is_restrained(0)

        L = fixed_node.distance_to(free_node)
        props = section.get_properties()

        # Can calculate deflection parameters
        E = concrete.E
        I = props["Ix"]
        EI = E * I

        assert EI > 0  # Valid stiffness


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
