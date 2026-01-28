"""
Core data structures for finite element analysis.

This module provides the fundamental building blocks for structural analysis:
- Node: Geometric points with degrees of freedom
- Material: Material properties (elastic modulus, Poisson's ratio, density)
- Section: Cross-sectional properties (area, moments of inertia)

Examples:
    >>> node = Node(id=1, x=0.0, y=0.0, z=0.0)
    >>> node.set_restraint(0, True)  # Fix translation in X
    >>>
    >>> steel = Material.from_steel()
    >>> concrete = Material.from_concrete(fc=30.0)
    >>>
    >>> rect = RectangularSection(width=0.3, height=0.5)
    >>> props = rect.get_properties()
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field, field_validator, computed_field


class Node(BaseModel):
    """
    Represents a node in the finite element model.

    A node is a point in space with associated degrees of freedom (DOFs).
    For 2D analysis: 3 DOFs (ux, uy, rotation_z)
    For 3D analysis: 6 DOFs (ux, uy, uz, rotation_x, rotation_y, rotation_z)

    Attributes:
        id: Unique node identifier
        x: X-coordinate in global coordinate system (m)
        y: Y-coordinate in global coordinate system (m)
        z: Z-coordinate in global coordinate system (m)
        restraints: List of boolean flags for DOF restraints (True = fixed)
        displacements: Current displacement values for each DOF
        reactions: Reaction forces at restrained DOFs

    Examples:
        >>> # Create a 2D node at origin
        >>> node_2d = Node(id=1, x=0.0, y=0.0)
        >>> node_2d.get_dofs()
        3

        >>> # Create a 3D node and fix all translations
        >>> node_3d = Node(id=2, x=5.0, y=3.0, z=2.0)
        >>> node_3d.set_restraint(0, True)  # Fix ux
        >>> node_3d.set_restraint(1, True)  # Fix uy
        >>> node_3d.set_restraint(2, True)  # Fix uz
        >>> node_3d.is_restrained(0)
        True
    """

    id: int = Field(..., description="Unique node identifier")
    x: float = Field(..., description="X-coordinate (m)")
    y: float = Field(..., description="Y-coordinate (m)")
    z: float = Field(default=0.0, description="Z-coordinate (m)")
    restraints: List[bool] = Field(
        default_factory=lambda: [False] * 6,
        description="DOF restraints [ux, uy, uz, rx, ry, rz]",
    )
    displacements: List[float] = Field(
        default_factory=lambda: [0.0] * 6, description="Current displacements"
    )
    reactions: List[float] = Field(
        default_factory=lambda: [0.0] * 6, description="Reaction forces/moments"
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("restraints")
    @classmethod
    def validate_restraints(cls, v: List[bool]) -> List[bool]:
        """Ensure restraints list has exactly 6 elements."""
        if len(v) != 6:
            raise ValueError("Restraints must have exactly 6 elements")
        return v

    @computed_field
    @property
    def is_3d(self) -> bool:
        """Check if node is used in 3D analysis."""
        return self.z != 0.0 or any(self.restraints[2:])

    def get_dofs(self) -> int:
        """
        Get the number of degrees of freedom for this node.

        Returns:
            3 for 2D nodes, 6 for 3D nodes
        """
        return 6 if self.is_3d else 3

    def is_restrained(self, dof_index: int) -> bool:
        """
        Check if a specific DOF is restrained.

        Args:
            dof_index: DOF index (0-5 for 3D, 0-2 for 2D)
                       0=ux, 1=uy, 2=uz, 3=rx, 4=ry, 5=rz

        Returns:
            True if the DOF is restrained (fixed)

        Raises:
            IndexError: If dof_index is out of range
        """
        if dof_index < 0 or dof_index >= 6:
            raise IndexError(f"DOF index {dof_index} out of range [0, 5]")
        return self.restraints[dof_index]

    def set_restraint(self, dof_index: int, value: bool) -> None:
        """
        Set restraint condition for a specific DOF.

        Args:
            dof_index: DOF index (0-5)
            value: True to fix, False to free

        Raises:
            IndexError: If dof_index is out of range
        """
        if dof_index < 0 or dof_index >= 6:
            raise IndexError(f"DOF index {dof_index} out of range [0, 5]")
        self.restraints[dof_index] = value
        if value:
            self.reactions[dof_index] = 0.0  # Initialize reaction

    def set_fixed_support(self) -> None:
        """Fix all DOFs (fully fixed support)."""
        for i in range(6):
            self.set_restraint(i, True)

    def set_pinned_support(self) -> None:
        """Pin support (fix translations, free rotations)."""
        for i in range(3):
            self.set_restraint(i, True)
        for i in range(3, 6):
            self.set_restraint(i, False)

    def set_roller_support(self, direction: int = 1) -> None:
        """
        Roller support (fix one translation, free others).

        Args:
            direction: Direction to fix (0=x, 1=y, 2=z)
        """
        for i in range(6):
            self.set_restraint(i, False)
        self.set_restraint(direction, True)

    def get_coordinates(self) -> np.ndarray:
        """
        Get node coordinates as numpy array.

        Returns:
            Array of [x, y, z]
        """
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: "Node") -> float:
        """
        Calculate Euclidean distance to another node.

        Args:
            other: Another Node instance

        Returns:
            Distance in same units as coordinates
        """
        coords1 = self.get_coordinates()
        coords2 = other.get_coordinates()
        return float(np.linalg.norm(coords2 - coords1))


class Material(BaseModel):
    """
    Material properties for structural elements.

    Attributes:
        name: Material name/identifier
        E: Young's modulus (Pa)
        G: Shear modulus (Pa)
        nu: Poisson's ratio (dimensionless)
        rho: Density (kg/m³)
        alpha: Thermal expansion coefficient (1/°C)

    Examples:
        >>> # Create custom material
        >>> custom = Material(
        ...     name="Custom Steel",
        ...     E=200e9,
        ...     nu=0.3,
        ...     rho=7850
        ... )

        >>> # Use factory methods
        >>> steel = Material.from_steel()
        >>> concrete = Material.from_concrete(fc=30.0)
        >>> timber = Material.from_timber(species="Pine")
    """

    name: str = Field(..., description="Material name")
    E: float = Field(..., description="Young's modulus (Pa)", gt=0)
    nu: float = Field(..., description="Poisson's ratio", gt=0, lt=0.5)
    rho: float = Field(default=0.0, description="Density (kg/m³)", ge=0)
    alpha: float = Field(default=0.0, description="Thermal expansion (1/°C)")

    @field_validator("E")
    @classmethod
    def validate_E(cls, v: float) -> float:
        """Ensure Young's modulus is positive."""
        if v <= 0:
            raise ValueError("Young's modulus E must be positive")
        return v

    @field_validator("nu")
    @classmethod
    def validate_nu(cls, v: float) -> float:
        """Ensure Poisson's ratio is in valid range."""
        if not 0 < v < 0.5:
            raise ValueError("Poisson's ratio must be between 0 and 0.5")
        return v

    @computed_field
    @property
    def G(self) -> float:
        """
        Calculate shear modulus from E and nu.

        Returns:
            Shear modulus G = E / (2(1 + nu))
        """
        return self.E / (2.0 * (1.0 + self.nu))

    @classmethod
    def from_steel(cls, grade: str = "A36") -> "Material":
        """
        Create steel material with standard properties.

        Args:
            grade: Steel grade ("A36", "A572-50", "S355")

        Returns:
            Material instance with steel properties
        """
        steel_grades = {
            "A36": {"E": 200e9, "nu": 0.3, "rho": 7850},
            "A572-50": {"E": 200e9, "nu": 0.3, "rho": 7850},
            "S355": {"E": 210e9, "nu": 0.3, "rho": 7850},
        }

        props = steel_grades.get(grade, steel_grades["A36"])
        return cls(
            name=f"Steel {grade}",
            E=props["E"],
            nu=props["nu"],
            rho=props["rho"],
            alpha=12e-6,
        )

    @classmethod
    def from_concrete(cls, fc: float = 25.0) -> "Material":
        """
        Create concrete material with properties based on strength.

        Args:
            fc: Concrete compressive strength (MPa)

        Returns:
            Material instance with concrete properties
        """
        # ACI 318 formula: Ec = 4700*sqrt(fc) MPa
        E = 4700.0 * np.sqrt(fc) * 1e6  # Convert to Pa

        return cls(name=f"Concrete fc={fc}MPa", E=E, nu=0.2, rho=2400, alpha=10e-6)

    @classmethod
    def from_timber(cls, species: str = "Pine", grade: str = "C24") -> "Material":
        """
        Create timber material with standard properties.

        Args:
            species: Wood species
            grade: Strength grade (e.g., "C24", "C30")

        Returns:
            Material instance with timber properties
        """
        timber_grades = {
            "C24": {"E": 11e9, "rho": 420},
            "C30": {"E": 12e9, "rho": 460},
        }

        props = timber_grades.get(grade, timber_grades["C24"])
        return cls(
            name=f"{species} {grade}",
            E=props["E"],
            nu=0.3,
            rho=props["rho"],
            alpha=5e-6,
        )


class Section(BaseModel, ABC):
    """
    Abstract base class for cross-sectional properties.

    All section types must implement methods to calculate:
    - A: Cross-sectional area
    - Ix, Iy: Moments of inertia about local axes
    - Iz: Polar moment of inertia (for torsion)
    - J: Torsional constant
    """

    name: Optional[str] = Field(default=None, description="Section identifier")

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    def get_properties(self) -> dict:
        """
        Calculate and return all section properties.

        Returns:
            Dictionary with keys: A, Ix, Iy, Iz, J
        """
        pass

    @abstractmethod
    def validate_geometry(self) -> bool:
        """
        Validate that geometric parameters are physically valid.

        Returns:
            True if geometry is valid

        Raises:
            ValueError: If geometry is invalid
        """
        pass


class RectangularSection(Section):
    """
    Rectangular cross-section.

    Attributes:
        width: Width (b) in meters
        height: Height (h) in meters

    Properties:
        - A = b * h
        - Ix = b * h³ / 12 (about strong axis)
        - Iy = h * b³ / 12 (about weak axis)
        - Iz = Ix + Iy (polar moment)
        - J ≈ k * b * h³ (torsional constant)

    Examples:
        >>> rect = RectangularSection(width=0.3, height=0.5)
        >>> props = rect.get_properties()
        >>> props['A']
        0.15
        >>> props['Ix']
        0.003125
    """

    width: float = Field(..., description="Width b (m)", gt=0)
    height: float = Field(..., description="Height h (m)", gt=0)

    def validate_geometry(self) -> bool:
        """Validate rectangular section dimensions."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
        return True

    def get_properties(self) -> dict:
        """Calculate rectangular section properties."""
        self.validate_geometry()

        b = self.width
        h = self.height

        A = b * h
        Ix = b * h**3 / 12.0  # Strong axis (about x-x)
        Iy = h * b**3 / 12.0  # Weak axis (about y-y)
        Iz = Ix + Iy  # Polar moment

        # Torsional constant (Saint-Venant)
        beta = min(b, h) / max(b, h)
        k = (1 / 3) * (1 - 0.63 * beta)
        J = k * max(b, h) * min(b, h) ** 3

        return {"A": A, "Ix": Ix, "Iy": Iy, "Iz": Iz, "J": J, "width": b, "height": h}


class CircularSection(Section):
    """
    Circular cross-section.

    Attributes:
        diameter: Outer diameter (m)
        thickness: Wall thickness (m), 0 for solid section

    Properties:
        - A = π/4 * (D² - d²) where d = D - 2*t
        - Ix = Iy = π/64 * (D⁴ - d⁴)
        - Iz = Ix + Iy (polar moment)
        - J = π/32 * (D⁴ - d⁴) (torsional constant)

    Examples:
        >>> # Solid circular section
        >>> solid = CircularSection(diameter=0.2)
        >>> props = solid.get_properties()

        >>> # Hollow circular section (pipe)
        >>> pipe = CircularSection(diameter=0.2, thickness=0.01)
    """

    diameter: float = Field(..., description="Outer diameter (m)", gt=0)
    thickness: float = Field(default=0.0, description="Wall thickness (m)", ge=0)

    def validate_geometry(self) -> bool:
        """Validate circular section dimensions."""
        if self.diameter <= 0:
            raise ValueError("Diameter must be positive")
        if self.thickness < 0:
            raise ValueError("Thickness cannot be negative")
        if self.thickness >= self.diameter / 2:
            raise ValueError("Thickness must be less than radius")
        return True

    def get_properties(self) -> dict:
        """Calculate circular section properties."""
        self.validate_geometry()

        D = self.diameter
        t = self.thickness
        d = D - 2 * t  # Inner diameter

        if t == 0:  # Solid section
            A = np.pi * D**2 / 4
            I = np.pi * D**4 / 64
            J = np.pi * D**4 / 32
        else:  # Hollow section
            A = np.pi * (D**2 - d**2) / 4
            I = np.pi * (D**4 - d**4) / 64
            J = np.pi * (D**4 - d**4) / 32

        return {
            "A": A,
            "Ix": I,
            "Iy": I,
            "Iz": 2 * I,
            "J": J,
            "diameter": D,
            "thickness": t,
        }


class ISection(Section):
    """
    I-shaped (wide flange) cross-section.

    Attributes:
        flange_width: Total flange width (m)
        flange_thickness: Flange thickness (m)
        web_height: Total height (m)
        web_thickness: Web thickness (m)

    Examples:
        >>> # W18x50 approximate dimensions
        >>> i_section = ISection(
        ...     flange_width=0.19,
        ...     flange_thickness=0.014,
        ...     web_height=0.457,
        ...     web_thickness=0.009
        ... )
    """

    flange_width: float = Field(..., description="Flange width bf (m)", gt=0)
    flange_thickness: float = Field(..., description="Flange thickness tf (m)", gt=0)
    web_height: float = Field(..., description="Total height h (m)", gt=0)
    web_thickness: float = Field(..., description="Web thickness tw (m)", gt=0)

    def validate_geometry(self) -> bool:
        """Validate I-section dimensions."""
        if any(
            x <= 0
            for x in [
                self.flange_width,
                self.flange_thickness,
                self.web_height,
                self.web_thickness,
            ]
        ):
            raise ValueError("All dimensions must be positive")
        if self.flange_thickness * 2 >= self.web_height:
            raise ValueError("Flange thickness too large for web height")
        if self.web_thickness >= self.flange_width:
            raise ValueError("Web thickness too large for flange width")
        return True

    def get_properties(self) -> dict:
        """Calculate I-section properties."""
        self.validate_geometry()

        bf = self.flange_width
        tf = self.flange_thickness
        h = self.web_height
        tw = self.web_thickness
        hw = h - 2 * tf  # Web height

        # Area
        A = 2 * bf * tf + hw * tw

        # Moment of inertia about strong axis (Ix)
        Ix = (bf * h**3 / 12) - ((bf - tw) * hw**3 / 12)

        # Moment of inertia about weak axis (Iy)
        Iy = (2 * tf * bf**3 / 12) + (hw * tw**3 / 12)

        # Approximate torsional constant
        J = (2 * bf * tf**3 + hw * tw**3) / 3

        Iz = Ix + Iy

        return {
            "A": A,
            "Ix": Ix,
            "Iy": Iy,
            "Iz": Iz,
            "J": J,
            "flange_width": bf,
            "flange_thickness": tf,
            "web_height": h,
            "web_thickness": tw,
        }


class CustomSection(Section):
    """
    Custom cross-section with user-defined properties.

    Useful for irregular shapes or when properties are known from
    external sources (catalogs, CAD software, etc.).

    Attributes:
        A: Cross-sectional area (m²)
        Ix: Moment of inertia about strong axis (m⁴)
        Iy: Moment of inertia about weak axis (m⁴)
        J: Torsional constant (m⁴)

    Examples:
        >>> custom = CustomSection(
        ...     name="Special Shape",
        ...     A=0.02,
        ...     Ix=1e-4,
        ...     Iy=5e-5,
        ...     J=8e-5
        ... )
    """

    A: float = Field(..., description="Area (m²)", gt=0)
    Ix: float = Field(..., description="Moment of inertia Ix (m⁴)", gt=0)
    Iy: float = Field(..., description="Moment of inertia Iy (m⁴)", gt=0)
    J: float = Field(..., description="Torsional constant (m⁴)", gt=0)

    def validate_geometry(self) -> bool:
        """Validate custom section properties."""
        if any(x <= 0 for x in [self.A, self.Ix, self.Iy, self.J]):
            raise ValueError("All properties must be positive")
        return True

    def get_properties(self) -> dict:
        """Return user-defined properties."""
        self.validate_geometry()

        return {
            "A": self.A,
            "Ix": self.Ix,
            "Iy": self.Iy,
            "Iz": self.Ix + self.Iy,
            "J": self.J,
        }
