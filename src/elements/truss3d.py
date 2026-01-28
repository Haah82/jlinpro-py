"""
Truss3D element for 3D space truss analysis.

This module implements a 3-dimensional truss (bar) element that only carries
axial forces. The element has 2 nodes with 6 DOFs each (12 total DOFs).

The transformation logic is ported from OOFEM's Truss3D implementation.

References:
    - OOFEM: /oofem-3.0/src/sm/Elements/Bars/truss3d.C
"""

from typing import List
import numpy as np
from pydantic import Field

from src.elements.base import AbstractElement
from src.core.structures import Node, Material, Section


class Truss3D(AbstractElement):
    """
    3D truss (bar) element for space truss analysis.

    The element has 12 DOFs total (6 per node):
        Node 1: ux1, uy1, uz1, rx1, ry1, rz1
        Node 2: ux2, uy2, uz2, rx2, ry2, rz2

    Only axial stiffness is considered (EA/L).
    Rotation DOFs have zero stiffness in local system.

    Attributes:
        id: Unique element identifier
        nodes: List of 2 nodes
        material: Material properties (E, G, rho)
        section: Cross-section properties (A)

    Examples:
        >>> from src.core.structures import Node, Material, RectangularSection
        >>> n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        >>> n2 = Node(id=2, x=3.0, y=4.0, z=5.0)
        >>> steel = Material.from_steel()
        >>> section = RectangularSection(width=0.1, height=0.1)
        >>> truss = Truss3D(id=1, nodes=[n1, n2], material=steel, section=section)
        >>> k_local = truss.get_stiffness_local()
        >>> k_local.shape
        (12, 12)
    """

    def get_stiffness_local(self) -> np.ndarray:
        """
        Calculate local stiffness matrix (12x12) in element coordinate system.

        Only the axial terms (ux1, ux2) are populated.
        All other DOFs have zero stiffness in the local system.

        Local coordinate system:
            x: Along element axis (node1 -> node2)
            y, z: Perpendicular to element axis

        Returns:
            12x12 local stiffness matrix

        Local DOF ordering:
            [ux1, uy1, uz1, rx1, ry1, rz1, ux2, uy2, uz2, rx2, ry2, rz2]
        """
        E = self.material.E
        A = self.section.A
        L = self.get_length()

        # Axial stiffness
        EA_L = E * A / L

        # Initialize 12x12 matrix
        k_local = np.zeros((12, 12))

        # Only axial terms (indices 0 and 6 for ux1 and ux2)
        k_local[0, 0] = EA_L
        k_local[0, 6] = -EA_L
        k_local[6, 0] = -EA_L
        k_local[6, 6] = EA_L

        return k_local

    def get_transformation_matrix(self) -> np.ndarray:
        """
        Calculate transformation matrix from local to global coordinates.

        Uses OOFEM's algorithm for constructing the local coordinate system:
        1. x_local = (node2 - node1).normalize()
        2. Construct y_local orthogonal to x_local using an arbitrary reference
        3. z_local = cross(x_local, y_local)
        4. Build 12x12 transformation matrix as block diagonal of 3x3 rotation

        Returns:
            12x12 transformation matrix T where K_global = T^T @ K_local @ T
        """
        # Get node coordinates
        n1 = self.nodes[0]
        n2 = self.nodes[1]

        # Calculate element direction vector
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        dz = n2.z - n1.z
        L = self.get_length()

        # x_local: unit vector along element
        lx = np.array([dx / L, dy / L, dz / L])

        # Construct y_local orthogonal to x_local
        # If element is approximately vertical, use different reference
        if abs(lx[2]) > 0.999:  # Nearly vertical (aligned with Z)
            # Use Y-axis as reference
            y_ref = np.array([0.0, 1.0, 0.0])
        else:
            # Use Z-axis as reference
            y_ref = np.array([0.0, 0.0, 1.0])

        # ly: orthogonal to both lx and y_ref
        ly = np.cross(lx, y_ref)
        ly = ly / np.linalg.norm(ly)  # Normalize

        # lz: complete the right-handed system
        lz = np.cross(lx, ly)

        # Build 3x3 rotation matrix R
        # Rows are the local axes in global coordinates
        R = np.array([lx, ly, lz])

        # Build 12x12 transformation matrix as block diagonal
        # T = block_diag(R, R, R, R) for 4 sets of 3 DOFs each
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = R

        return T

    def get_dof_indices(self) -> List[int]:
        """
        Get global DOF indices for this element.

        Returns:
            List of 12 global DOF indices
        """
        dof_indices = []

        for node in self.nodes:
            # DOF indices for 6 DOFs per node
            base_dof = node.id * 6
            for i in range(6):
                dof_indices.append(base_dof + i)

        return dof_indices

    def get_internal_forces(self, u_global: np.ndarray) -> dict:
        """
        Calculate internal forces from global displacement vector.

        Args:
            u_global: Global displacement vector

        Returns:
            Dictionary with axial force:
                {'N': axial_force (tension positive)}
        """
        # Get element DOF indices
        dof_indices = self.get_dof_indices()

        # Extract element displacements from global vector
        u_elem = u_global[dof_indices]

        # Transform to local coordinates
        T = self.get_transformation_matrix()
        u_local = T @ u_elem

        # Axial strain in local coordinates
        L = self.get_length()
        axial_strain = (u_local[6] - u_local[0]) / L

        # Axial force
        E = self.material.E
        A = self.section.A
        N = E * A * axial_strain

        return {"N": N}

    def validate_connectivity(self) -> bool:
        """
        Validate element connectivity.

        Returns:
            True if valid (2 nodes with different positions)
        """
        if len(self.nodes) != 2:
            return False

        # Check that nodes are not coincident
        if self.get_length() < 1e-10:
            return False

        return True
