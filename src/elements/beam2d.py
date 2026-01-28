"""
2D Beam element implementation.

Beam elements carry axial forces, shear forces, and bending moments.
"""

from typing import List, Optional
import numpy as np
from pydantic import Field

from src.elements.base import AbstractElement
from src.core.structures import Node, Material, Section


class Beam2D(AbstractElement):
    """
    2D beam (frame) element with 3 DOFs per node.

    Beam elements resist axial forces, shear forces, and bending moments.
    Each node has 3 DOFs:
    - Translation in X direction (axial)
    - Translation in Y direction (transverse)
    - Rotation about Z axis (bending)

    Attributes:
        id: Unique element identifier
        nodes: Two nodes defining the element
        material: Material properties (E, G, rho)
        section: Cross-sectional properties (A, Ix)
        releases: Release conditions (bit flags for moment/shear/axial releases)

    Release Flags (bit positions):
        0x20 (32): Release axial force at node i
        0x10 (16): Release shear force at node i
        0x08 (8):  Release moment at node i
        0x04 (4):  Release axial force at node j
        0x02 (2):  Release shear force at node j
        0x01 (1):  Release moment at node j

    Examples:
        >>> node1 = Node(id=1, x=0.0, y=0.0)
        >>> node2 = Node(id=2, x=5.0, y=0.0)
        >>> steel = Material.from_steel()
        >>> section = RectangularSection(width=0.3, height=0.5)
        >>> beam = Beam2D(id=1, nodes=[node1, node2], material=steel, section=section)
        >>>
        >>> # Create beam with moment release at node j (pin at end)
        >>> beam_pinned = Beam2D(
        ...     id=2, nodes=[node1, node2], material=steel,
        ...     section=section, releases=0x01
        ... )
    """

    releases: int = Field(default=0, description="Release conditions (bit flags)")

    axial_force_i: float = Field(default=0.0, description="Axial force at node i")
    shear_force_i: float = Field(default=0.0, description="Shear force at node i")
    moment_i: float = Field(default=0.0, description="Bending moment at node i")

    axial_force_j: float = Field(default=0.0, description="Axial force at node j")
    shear_force_j: float = Field(default=0.0, description="Shear force at node j")
    moment_j: float = Field(default=0.0, description="Bending moment at node j")

    def model_post_init(self, __context):
        """Validate element after initialization."""
        if len(self.nodes) != 2:
            raise ValueError("Beam2D requires exactly 2 nodes")
        self.validate_connectivity()

    def get_stiffness_local(self) -> np.ndarray:
        """
        Calculate local stiffness matrix for 2D beam element.

        In local coordinates, the stiffness matrix is 6x6:

        DOFs: [u_i, v_i, theta_i, u_j, v_j, theta_j]

        Where:
        - u: axial displacement
        - v: transverse displacement
        - theta: rotation

        The matrix includes:
        - Axial stiffness: EA/L
        - Bending stiffness: 12EI/L³, 6EI/L², 4EI/L, 2EI/L

        Returns:
            6x6 local stiffness matrix
        """
        L = self.get_length()
        E = self.material.E
        props = self.section.get_properties()
        A = props["A"]
        I = props["Ix"]  # Moment of inertia about strong axis

        # Stiffness coefficients
        EA_L = E * A / L
        EI_L = E * I / L
        EI_L2 = E * I / (L * L)
        EI_L3 = E * I / (L * L * L)

        # Local stiffness matrix (6x6)
        K = np.zeros((6, 6))

        # Axial terms (DOFs 0 and 3)
        K[0, 0] = EA_L
        K[0, 3] = -EA_L
        K[3, 0] = -EA_L
        K[3, 3] = EA_L

        # Bending terms (DOFs 1, 2, 4, 5)
        K[1, 1] = 12 * EI_L3
        K[1, 2] = 6 * EI_L2
        K[1, 4] = -12 * EI_L3
        K[1, 5] = 6 * EI_L2

        K[2, 1] = 6 * EI_L2
        K[2, 2] = 4 * EI_L
        K[2, 4] = -6 * EI_L2
        K[2, 5] = 2 * EI_L

        K[4, 1] = -12 * EI_L3
        K[4, 2] = -6 * EI_L2
        K[4, 4] = 12 * EI_L3
        K[4, 5] = -6 * EI_L2

        K[5, 1] = 6 * EI_L2
        K[5, 2] = 2 * EI_L
        K[5, 4] = -6 * EI_L2
        K[5, 5] = 4 * EI_L

        # Apply releases if specified
        if self.releases != 0:
            K = self._apply_releases(K)

        return K

    def _apply_releases(self, K: np.ndarray) -> np.ndarray:
        """
        Apply release conditions to stiffness matrix.

        Release conditions zero out rows and columns corresponding to
        released DOFs, effectively making them unconstrained internally.

        Args:
            K: Original stiffness matrix

        Returns:
            Modified stiffness matrix with releases applied
        """
        # Release flags
        RELEASED_NI = 0x20  # Axial at i
        RELEASED_QI = 0x10  # Shear at i
        RELEASED_MI = 0x08  # Moment at i
        RELEASED_NJ = 0x04  # Axial at j
        RELEASED_QJ = 0x02  # Shear at j
        RELEASED_MJ = 0x01  # Moment at j

        K_mod = K.copy()

        # Apply moment release at node i (DOF 2)
        if self.releases & RELEASED_MI:
            K_mod[2, :] = 0
            K_mod[:, 2] = 0
            K_mod[2, 2] = 1e-6  # Small value to avoid singularity

        # Apply moment release at node j (DOF 5)
        if self.releases & RELEASED_MJ:
            K_mod[5, :] = 0
            K_mod[:, 5] = 0
            K_mod[5, 5] = 1e-6

        # Apply shear release at node i (DOF 1)
        if self.releases & RELEASED_QI:
            K_mod[1, :] = 0
            K_mod[:, 1] = 0
            K_mod[1, 1] = 1e-6

        # Apply shear release at node j (DOF 4)
        if self.releases & RELEASED_QJ:
            K_mod[4, :] = 0
            K_mod[:, 4] = 0
            K_mod[4, 4] = 1e-6

        # Apply axial release at node i (DOF 0)
        if self.releases & RELEASED_NI:
            K_mod[0, :] = 0
            K_mod[:, 0] = 0
            K_mod[0, 0] = 1e-6

        # Apply axial release at node j (DOF 3)
        if self.releases & RELEASED_NJ:
            K_mod[3, :] = 0
            K_mod[:, 3] = 0
            K_mod[3, 3] = 1e-6

        return K_mod

    def get_transformation_matrix(self) -> np.ndarray:
        """
        Calculate transformation matrix for 2D beam element.

        Transforms from local (along element) to global (XY) coordinates.

        T = [[c, s, 0,  0, 0, 0],
             [-s, c, 0,  0, 0, 0],
             [0, 0, 1,  0, 0, 0],
             [0, 0, 0,  c, s, 0],
             [0, 0, 0, -s, c, 0],
             [0, 0, 0,  0, 0, 1]]

        where c = cos(theta), s = sin(theta)

        Returns:
            6x6 transformation matrix
        """
        cx, cy = self.get_direction_cosines()

        # Transformation matrix for 2 nodes, 3 DOFs each
        T = np.array(
            [
                [cx, cy, 0, 0, 0, 0],
                [-cy, cx, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, cx, cy, 0],
                [0, 0, 0, -cy, cx, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        return T

    def get_dof_indices(self) -> List[int]:
        """
        Get global DOF indices for this beam element.

        For 2D beam: [node1_ux, node1_uy, node1_rz, node2_ux, node2_uy, node2_rz]

        Returns:
            List of 6 global DOF indices
        """
        dofs = []
        for node in self.nodes:
            # For 2D: 3 DOFs per node
            dofs.extend([node.id * 3, node.id * 3 + 1, node.id * 3 + 2])

        return dofs

    def get_internal_forces(self, u_global: np.ndarray) -> dict:
        """
        Calculate internal forces from global displacements.

        Args:
            u_global: Global displacement vector for element DOFs
                     Can be 6 values (legacy) or 12 values (6 DOFs per node)

        Returns:
            Dictionary with forces at nodes i and j
        """
        # Handle both 6-value (element DOFs only) and 12-value (global 6 DOFs per node) inputs
        if len(u_global) == 12:
            # Extract relevant DOFs for 2D beam: [ux, uy, rz] at indices [0, 1, 5] per node
            # Node i: [0, 1, 5], Node j: [6, 7, 11]
            u_elem = np.array([
                u_global[0],  # ux_i
                u_global[1],  # uy_i
                u_global[5],  # rz_i
                u_global[6],  # ux_j
                u_global[7],  # uy_j
                u_global[11]  # rz_j
            ])
        elif len(u_global) == 6:
            u_elem = u_global
        else:
            raise ValueError(f"Expected 6 or 12 displacement values for 2D beam, got {len(u_global)}")

        # Transform to local coordinates
        T = self.get_transformation_matrix()
        u_local = T @ u_elem

        # Calculate forces: F = K * u
        K_local = self.get_stiffness_local()
        f_local = K_local @ u_local

        # Extract forces at nodes
        self.axial_force_i = float(f_local[0])
        self.shear_force_i = float(f_local[1])
        self.moment_i = float(f_local[2])

        self.axial_force_j = float(f_local[3])
        self.shear_force_j = float(f_local[4])
        self.moment_j = float(f_local[5])

        return {
            "node_i": {
                "axial": self.axial_force_i,
                "shear": self.shear_force_i,
                "moment": self.moment_i,
            },
            "node_j": {
                "axial": self.axial_force_j,
                "shear": self.shear_force_j,
                "moment": self.moment_j,
            },
        }

    def get_displacement_local(self, u_global: np.ndarray) -> np.ndarray:
        """
        Transform global displacements to local coordinate system.

        Args:
            u_global: Global displacement vector (6 values)

        Returns:
            Local displacement vector
        """
        T = self.get_transformation_matrix()
        return T @ u_global

    def get_weight(self) -> float:
        """
        Calculate element weight.

        Returns:
            Weight in Newtons (rho * A * L * g)
        """
        L = self.get_length()
        A = self.section.get_properties()["A"]
        rho = self.material.rho

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

    def is_moment_released(self, node_index: int) -> bool:
        """
        Check if moment is released at specified node.

        Args:
            node_index: 0 for node i, 1 for node j

        Returns:
            True if moment is released
        """
        if node_index == 0:
            return bool(self.releases & 0x08)
        elif node_index == 1:
            return bool(self.releases & 0x01)
        else:
            raise ValueError("node_index must be 0 or 1")
