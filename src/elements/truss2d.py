"""
2D Truss element implementation.

Truss elements can only carry axial forces (tension/compression).
"""

from typing import List
import numpy as np
from pydantic import Field

from src.elements.base import AbstractElement
from src.core.structures import Node, Material, Section


class Truss2D(AbstractElement):
    """
    2D truss (bar) element with 2 DOFs per node.

    Truss elements resist only axial forces. Each node has 2 DOFs:
    - Translation in X direction
    - Translation in Y direction

    The element stiffness is based on: K = (EA/L) * transformation

    Attributes:
        id: Unique element identifier
        nodes: Two nodes defining the element
        material: Material properties (E, rho)
        section: Cross-sectional properties (A)

    Examples:
        >>> node1 = Node(id=1, x=0.0, y=0.0)
        >>> node2 = Node(id=2, x=3.0, y=4.0)
        >>> steel = Material.from_steel()
        >>> section = RectangularSection(width=0.1, height=0.1)
        >>> truss = Truss2D(id=1, nodes=[node1, node2], material=steel, section=section)
        >>> K_global = truss.get_stiffness_global()
        >>> print(f"Element length: {truss.get_length():.2f} m")
        Element length: 5.00 m
    """

    axial_force: float = Field(default=0.0, description="Axial force (+ = tension)")
    stress: float = Field(default=0.0, description="Axial stress (Pa)")

    def model_post_init(self, __context):
        """Validate element after initialization."""
        if len(self.nodes) != 2:
            raise ValueError("Truss2D requires exactly 2 nodes")
        self.validate_connectivity()

    def get_stiffness_local(self) -> np.ndarray:
        """
        Calculate local stiffness matrix for 2D truss.

        In local coordinates (along element axis), the stiffness is:
        k_local = (E*A/L) * [[1, -1],
                             [-1, 1]]

        Expanded to 4x4 for 2 nodes with 2 DOFs each:
        [[k, 0, -k, 0],
         [0, 0,  0, 0],
         [-k, 0, k, 0],
         [0, 0,  0, 0]]

        Returns:
            4x4 local stiffness matrix
        """
        L = self.get_length()
        E = self.material.E
        A = self.section.get_properties()["A"]

        k = E * A / L

        # Local stiffness (only axial DOF is active)
        K_local = np.array([[k, 0, -k, 0], [0, 0, 0, 0], [-k, 0, k, 0], [0, 0, 0, 0]])

        return K_local

    def get_transformation_matrix(self) -> np.ndarray:
        """
        Calculate transformation matrix for 2D truss element.

        Transforms from local (along element) to global (XY) coordinates.

        T = [[c, s, 0, 0],
             [-s, c, 0, 0],
             [0, 0, c, s],
             [0, 0, -s, c]]

        where c = cos(theta), s = sin(theta)

        Returns:
            4x4 transformation matrix
        """
        cx, cy = self.get_direction_cosines()

        # Transformation matrix for 2 nodes, 2 DOFs each
        T = np.array(
            [
                [cx, cy, 0, 0],
                [-cy, cx, 0, 0],
                [0, 0, cx, cy],
                [0, 0, -cy, cx],
            ]
        )

        return T

    def get_dof_indices(self) -> List[int]:
        """
        Get global DOF indices for this truss element.

        For 2D truss: [node1_ux, node1_uy, node2_ux, node2_uy]

        Returns:
            List of 4 global DOF indices
        """
        dofs = []
        for node in self.nodes:
            # Assuming node.id can be used to calculate global DOF index
            # This will be properly implemented in the Structure class
            dofs.extend([node.id * 3, node.id * 3 + 1])

        return dofs

    def get_internal_forces(self, u_global: np.ndarray) -> dict:
        """
        Calculate internal forces from global displacements.

        Args:
            u_global: Global displacement vector for element DOFs

        Returns:
            Dictionary with axial_force and stress
        """
        # Extract element displacements (4 values)
        if len(u_global) != 4:
            raise ValueError("Expected 4 displacement values for 2D truss")

        # Transform to local coordinates
        T = self.get_transformation_matrix()
        u_local = T @ u_global

        # Axial force: F = (E*A/L) * (u_j - u_i)
        L = self.get_length()
        E = self.material.E
        A = self.section.get_properties()["A"]

        # u_local[0] = u_i_axial, u_local[2] = u_j_axial
        axial_elongation = u_local[2] - u_local[0]
        axial_force = (E * A / L) * axial_elongation

        # Stress = Force / Area
        stress = axial_force / A

        self.axial_force = float(axial_force)
        self.stress = float(stress)

        return {
            "axial_force": axial_force,
            "stress": stress,
            "elongation": axial_elongation,
        }

    def get_weight(self) -> float:
        """
        Calculate element weight.

        Returns:
            Weight in Newtons (rho * A * L * g)
        """
        L = self.get_length()
        A = self.section.get_properties()["A"]
        rho = self.material.rho

        # Assuming g = 9.81 m/s^2
        weight = rho * A * L * 9.81

        return weight

    def get_mass(self) -> float:
        """
        Calculate element mass.

        Returns:
            Mass in kg
        """
        L = self.get_length()
        A = self.section.get_properties()["A"]
        rho = self.material.rho

        mass = rho * A * L

        return mass
