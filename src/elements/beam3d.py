"""
Beam3D element for 3D space frame analysis.

This module implements a 3-dimensional Timoshenko beam element with full
coupling of axial, torsional, and bending behavior. The element has 2 nodes
with 6 DOFs each (12 total DOFs).

The transformation logic is ported from OOFEM's Beam3D implementation, supporting
reference nodes and roll angles for arbitrary element orientation.

References:
    - OOFEM: /oofem-3.0/src/sm/Elements/Beams/beam3d.C (Lines 487-550)
    - Matrix Analysis of Structures by Kassimali
"""

from typing import List, Optional
import numpy as np
from pydantic import Field

from src.elements.base import AbstractElement
from src.core.structures import Node, Material, Section


class Beam3D(AbstractElement):
    """
    3D Timoshenko beam element for space frame analysis.

    The element has 12 DOFs total (6 per node):
        Node 1: ux1, uy1, uz1, rx1, ry1, rz1
        Node 2: ux2, uy2, uz2, rx2, ry2, rz2

    Includes:
        - Axial stiffness (EA/L)
        - Torsional stiffness (GJ/L)
        - Bending stiffness about local y and z axes (EI/L^3)
        - Optional shear deformation (Timoshenko)

    Attributes:
        id: Unique element identifier
        nodes: List of 2 nodes
        material: Material properties (E, G, rho)
        section: Cross-section properties (A, Iy, Iz, J)
        ref_node: Optional reference node for defining local y-axis
        roll_angle: Roll angle in degrees (rotation about element axis)

    Examples:
        >>> from src.core.structures import Node, Material, RectangularSection
        >>> n1 = Node(id=1, x=0.0, y=0.0, z=0.0)
        >>> n2 = Node(id=2, x=0.0, y=0.0, z=5.0)  # Vertical column
        >>> steel = Material.from_steel()
        >>> section = RectangularSection(width=0.3, height=0.5)
        >>> beam = Beam3D(id=1, nodes=[n1, n2], material=steel, section=section,
        ...               roll_angle=0.0)
        >>> k_global = beam.get_stiffness_global()
        >>> k_global.shape
        (12, 12)
    """

    ref_node: Optional[Node] = Field(
        default=None, description="Reference node for orientation"
    )
    roll_angle: float = Field(
        default=0.0, description="Roll angle in degrees (rotation about x-axis)"
    )

    def get_stiffness_local(self) -> np.ndarray:
        """
        Calculate local stiffness matrix (12x12) using Timoshenko beam theory.

        Local coordinate system:
            x: Along element axis (node1 -> node2)
            y: Perpendicular (defined by ref_node or roll_angle)
            z: Completes right-handed system

        Includes:
            - Axial: EA/L
            - Torsion: GJ/L
            - Bending-Y (about z-axis): 12EIz/L^3, 6EIz/L^2, 4EIz/L, 2EIz/L
            - Bending-Z (about y-axis): 12EIy/L^3, 6EIy/L^2, 4EIy/L, 2EIy/L

        Shear deformation is neglected (Bernoulli beam).

        Returns:
            12x12 local stiffness matrix

        Local DOF ordering:
            [ux1, uy1, uz1, rx1, ry1, rz1, ux2, uy2, uz2, rx2, ry2, rz2]
        """
        E = self.material.E
        G = self.material.G
        A = self.section.A
        Iy = self.section.Iy  # Moment of inertia about local y-axis
        Iz = self.section.Iz  # Moment of inertia about local z-axis
        J = self.section.J  # Torsional constant
        L = self.get_length()

        # Initialize 12x12 matrix
        k = np.zeros((12, 12))

        # Axial stiffness (ux1, ux2)
        EA_L = E * A / L
        k[0, 0] = EA_L
        k[0, 6] = -EA_L
        k[6, 0] = -EA_L
        k[6, 6] = EA_L

        # Torsional stiffness (rx1, rx2)
        GJ_L = G * J / L
        k[3, 3] = GJ_L
        k[3, 9] = -GJ_L
        k[9, 3] = -GJ_L
        k[9, 9] = GJ_L

        # Bending about local z-axis (in x-y plane)
        # Affects: uy1, rz1, uy2, rz2
        EIz = E * Iz
        k[1, 1] = 12 * EIz / L**3
        k[1, 5] = 6 * EIz / L**2
        k[1, 7] = -12 * EIz / L**3
        k[1, 11] = 6 * EIz / L**2

        k[5, 1] = 6 * EIz / L**2
        k[5, 5] = 4 * EIz / L
        k[5, 7] = -6 * EIz / L**2
        k[5, 11] = 2 * EIz / L

        k[7, 1] = -12 * EIz / L**3
        k[7, 5] = -6 * EIz / L**2
        k[7, 7] = 12 * EIz / L**3
        k[7, 11] = -6 * EIz / L**2

        k[11, 1] = 6 * EIz / L**2
        k[11, 5] = 2 * EIz / L
        k[11, 7] = -6 * EIz / L**2
        k[11, 11] = 4 * EIz / L

        # Bending about local y-axis (in x-z plane)
        # Affects: uz1, ry1, uz2, ry2
        EIy = E * Iy
        k[2, 2] = 12 * EIy / L**3
        k[2, 4] = -6 * EIy / L**2
        k[2, 8] = -12 * EIy / L**3
        k[2, 10] = -6 * EIy / L**2

        k[4, 2] = -6 * EIy / L**2
        k[4, 4] = 4 * EIy / L
        k[4, 8] = 6 * EIy / L**2
        k[4, 10] = 2 * EIy / L

        k[8, 2] = -12 * EIy / L**3
        k[8, 4] = 6 * EIy / L**2
        k[8, 8] = 12 * EIy / L**3
        k[8, 10] = 6 * EIy / L**2

        k[10, 2] = -6 * EIy / L**2
        k[10, 4] = 2 * EIy / L
        k[10, 8] = 6 * EIy / L**2
        k[10, 10] = 4 * EIy / L

        return k

    def get_transformation_matrix(self) -> np.ndarray:
        """
        Calculate transformation matrix using OOFEM's orientation algorithm.

        The algorithm handles arbitrary beam orientation using either:
        1. A reference node to define the local z-axis
        2. An "up-vector" approach with optional roll angle

        Process (from OOFEM beam3d.C):
        1. x_local = (node2 - node1).normalize()
        2. If ref_node exists:
               z_local = cross(x_local, v_ref).normalize()
           Else:
               v_up = (0, 0, 1) or (0, 1, 0) if vertical
               y_temp = cross(x_local, v_up).normalize()
               Apply roll rotation around x_local
               z_local = cross(x_local, y_local).normalize()
        3. y_local = cross(z_local, x_local).normalize()
        4. Build R = [x_local, y_local, z_local] (row vectors)
        5. T = block_diag(R, R, R, R)

        Returns:
            12x12 transformation matrix T where K_global = T^T @ K_local @ T
        """
        # Get node coordinates
        n1 = self.nodes[0]
        n2 = self.nodes[1]

        # Step 1: Calculate x_local (along element)
        dx = n2.x - n1.x
        dy = n2.y - n1.y
        dz = n2.z - n1.z
        L = self.get_length()
        lx = np.array([dx / L, dy / L, dz / L])

        # Step 2: Determine local z-axis
        if self.ref_node is not None:
            # Use reference node
            v_ref = np.array(
                [
                    self.ref_node.x - n1.x,
                    self.ref_node.y - n1.y,
                    self.ref_node.z - n1.z,
                ]
            )
            lz = np.cross(lx, v_ref)
            lz = lz / np.linalg.norm(lz)
        else:
            # Use "up-vector" method
            v_up = np.array([0.0, 0.0, 1.0])

            # Check if element is vertical (aligned with z-axis)
            if abs(np.dot(lx, v_up)) > 0.999:
                # Element is vertical, use y-axis as reference
                v_up = np.array([0.0, 1.0, 0.0])

            # Temporary y-axis
            ly_temp = np.cross(lx, v_up)
            ly_temp = ly_temp / np.linalg.norm(ly_temp)

            # Apply roll angle (rotation about x_local)
            theta = np.radians(self.roll_angle)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            # Rodrigues' rotation formula to rotate ly_temp around lx
            # ly = ly_temp * cos(theta) + cross(lx, ly_temp) * sin(theta) + lx * dot(lx, ly_temp) * (1 - cos(theta))
            dot_product = np.dot(lx, ly_temp)
            ly = (
                ly_temp * cos_theta
                + np.cross(lx, ly_temp) * sin_theta
                + lx * dot_product * (1 - cos_theta)
            )

            # z_local completes the right-handed system
            lz = np.cross(lx, ly)
            lz = lz / np.linalg.norm(lz)

        # Step 3: Calculate y_local
        ly = np.cross(lz, lx)
        ly = ly / np.linalg.norm(ly)

        # Step 4: Build 3x3 rotation matrix
        # Rows are local axes expressed in global coordinates
        R = np.array([lx, ly, lz])

        # Step 5: Build 12x12 transformation matrix
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
            Dictionary with internal forces at node 1 (start):
                {
                    'N': axial force,
                    'Vy': shear force in local y direction,
                    'Vz': shear force in local z direction,
                    'T': torque,
                    'My': bending moment about local y-axis,
                    'Mz': bending moment about local z-axis
                }
        """
        # Get element DOF indices
        dof_indices = self.get_dof_indices()

        # Extract element displacements from global vector
        u_elem = u_global[dof_indices]

        # Transform to local coordinates
        T = self.get_transformation_matrix()
        u_local = T @ u_elem

        # Calculate local stiffness matrix
        k_local = self.get_stiffness_local()

        # Internal forces in local coordinates: f = k * u
        f_local = k_local @ u_local

        # Return forces at node 1 (first 6 components)
        return {
            "N": f_local[0],  # Axial force
            "Vy": f_local[1],  # Shear in y
            "Vz": f_local[2],  # Shear in z
            "T": f_local[3],  # Torque
            "My": f_local[4],  # Moment about y
            "Mz": f_local[5],  # Moment about z
        }

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
