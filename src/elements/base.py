"""
Base class for finite elements.

This module provides the abstract base class that all element types inherit from.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
from pydantic import BaseModel, Field

from src.core.structures import Node, Material, Section


class AbstractElement(BaseModel, ABC):
    """
    Abstract base class for all finite element types.

    Provides common functionality for element stiffness calculation,
    coordinate transformation, and DOF management.

    Attributes:
        id: Unique element identifier
        nodes: List of nodes that define the element
        material: Material properties
        section: Cross-sectional properties

    Examples:
        >>> # Elements are created through subclasses
        >>> from src.elements.truss2d import Truss2D
        >>> element = Truss2D(id=1, nodes=[node1, node2], material=steel, section=rect)
    """

    id: int = Field(..., description="Unique element identifier")
    nodes: List[Node] = Field(..., description="Element nodes")
    material: Material = Field(..., description="Material properties")
    section: Section = Field(..., description="Cross-sectional properties")

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    def get_stiffness_local(self) -> np.ndarray:
        """
        Calculate element stiffness matrix in local coordinate system.

        Returns:
            Local stiffness matrix as numpy array
        """
        pass

    @abstractmethod
    def get_transformation_matrix(self) -> np.ndarray:
        """
        Calculate transformation matrix from local to global coordinates.

        Returns:
            Transformation matrix T
        """
        pass

    @abstractmethod
    def get_dof_indices(self) -> List[int]:
        """
        Get global DOF indices for this element.

        Returns:
            List of global DOF indices
        """
        pass

    def get_length(self) -> float:
        """
        Calculate element length.

        Returns:
            Element length in same units as node coordinates
        """
        if len(self.nodes) != 2:
            raise ValueError("Length calculation requires exactly 2 nodes")

        return self.nodes[0].distance_to(self.nodes[1])

    def get_stiffness_global(self) -> np.ndarray:
        """
        Calculate element stiffness matrix in global coordinate system.

        Uses transformation: K_global = T^T @ K_local @ T
        
        For 2D elements (6x6), expands to 12x12 to match global DOF numbering
        (6 DOFs per node: ux, uy, uz, rx, ry, rz).

        Returns:
            Global stiffness matrix (12x12 for 2-node elements)
        """
        K_local = self.get_stiffness_local()
        T = self.get_transformation_matrix()

        # K_global = T^T @ K_local @ T (still 6x6 or element size)
        K_elem = T.T @ K_local @ T
        
        # Check if this is a 2D element that needs expansion to 6 DOFs per node
        n_nodes = len(self.nodes)
        if K_elem.shape[0] == 6 and n_nodes == 2:
            # Expand from 6x6 (2 nodes × 3 DOFs) to 12x12 (2 nodes × 6 DOFs)
            K_expanded = np.zeros((12, 12))
            
            # Mapping: element DOF [ux, uy, rz] -> global DOF [ux, uy, rz] at indices [0, 1, 5]
            # Node i: elem [0,1,2] -> global [0,1,5]
            # Node j: elem [3,4,5] -> global [6,7,11]
            elem_to_global = [0, 1, 5, 6, 7, 11]
            
            for i, gi in enumerate(elem_to_global):
                for j, gj in enumerate(elem_to_global):
                    K_expanded[gi, gj] = K_elem[i, j]
            
            return K_expanded
        else:
            # 3D element or other - return as is
            return K_elem

    def validate_connectivity(self) -> bool:
        """
        Validate that element has proper nodal connectivity.

        Returns:
            True if connectivity is valid

        Raises:
            ValueError: If connectivity is invalid
        """
        if len(self.nodes) < 2:
            raise ValueError("Element must have at least 2 nodes")

        # Check for coincident nodes
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if self.nodes[i].distance_to(self.nodes[j]) < 1e-6:
                    raise ValueError(
                        f"Nodes {self.nodes[i].id} and {self.nodes[j].id} "
                        "are coincident (distance < 1e-6)"
                    )

        return True

    def get_angle(self) -> float:
        """
        Calculate element orientation angle in global XY plane.

        For 2-node elements, returns angle from positive X-axis.

        Returns:
            Angle in radians
        """
        if len(self.nodes) != 2:
            raise ValueError("Angle calculation requires exactly 2 nodes")

        dx = self.nodes[1].x - self.nodes[0].x
        dy = self.nodes[1].y - self.nodes[0].y

        return np.arctan2(dy, dx)

    def get_direction_cosines(self) -> Tuple[float, float]:
        """
        Calculate direction cosines for 2D elements.

        Returns:
            Tuple of (cos(theta), sin(theta))
        """
        L = self.get_length()
        dx = self.nodes[1].x - self.nodes[0].x
        dy = self.nodes[1].y - self.nodes[0].y

        cx = dx / L
        cy = dy / L

        return cx, cy

    def __str__(self) -> str:
        """String representation of element."""
        node_ids = [node.id for node in self.nodes]
        return (
            f"{self.__class__.__name__}(id={self.id}, "
            f"nodes={node_ids}, L={self.get_length():.3f}m)"
        )
