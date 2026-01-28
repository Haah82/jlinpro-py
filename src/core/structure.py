"""
Structure class for finite element analysis.

This module provides the main Structure class that manages nodes, elements,
loads, and performs static analysis.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pandas as pd
from pydantic import BaseModel, Field

from src.core.structures import Node, Material, Section
from src.elements.base import AbstractElement


class Load(BaseModel):
    """
    Point load applied to a node.

    Attributes:
        node_id: ID of node where load is applied
        fx: Force in X direction (N)
        fy: Force in Y direction (N)
        mz: Moment about Z axis (N·m)
    """

    node_id: int = Field(..., description="Node ID")
    fx: float = Field(default=0.0, description="Force in X (N)")
    fy: float = Field(default=0.0, description="Force in Y (N)")
    mz: float = Field(default=0.0, description="Moment about Z (N·m)")


class Structure(BaseModel):
    """
    Main structure class for finite element analysis.

    Manages the structural model including nodes, elements, loads,
    and performs static analysis.

    Attributes:
        nodes: Dictionary of nodes (node_id -> Node)
        elements: Dictionary of elements (element_id -> AbstractElement)
        loads: Dictionary of loads (load_id -> Load)
        analysis_results: Dictionary storing analysis results

    Examples:
        >>> from src.core.structures import Node, Material, RectangularSection
        >>> from src.elements.beam2d import Beam2D
        >>>
        >>> structure = Structure()
        >>>
        >>> # Add nodes
        >>> structure.add_node(Node(id=1, x=0.0, y=0.0))
        >>> structure.add_node(Node(id=2, x=5.0, y=0.0))
        >>>
        >>> # Add element
        >>> steel = Material.from_steel()
        >>> section = RectangularSection(width=0.3, height=0.5)
        >>> beam = Beam2D(id=1, nodes=[n1, n2], material=steel, section=section)
        >>> structure.add_element(beam)
        >>>
        >>> # Apply loads and boundary conditions
        >>> structure.nodes[1].set_fixed_support()
        >>> structure.add_load(Load(node_id=2, fy=-10000))
        >>>
        >>> # Solve
        >>> structure.solve_static()
        >>> results = structure.get_results_summary()
    """

    nodes: Dict[int, Node] = Field(default_factory=dict, description="Nodes")
    elements: Dict[int, AbstractElement] = Field(
        default_factory=dict, description="Elements"
    )
    loads: Dict[int, Load] = Field(default_factory=dict, description="Loads")
    analysis_results: Dict = Field(default_factory=dict, description="Results")

    model_config = {"arbitrary_types_allowed": True}

    def add_node(self, node: Node) -> None:
        """Add a node to the structure."""
        self.nodes[node.id] = node

    def add_element(self, element: AbstractElement) -> None:
        """Add an element to the structure."""
        self.elements[element.id] = element

    def add_load(self, load: Load) -> None:
        """Add a load to the structure."""
        load_id = len(self.loads) + 1
        self.loads[load_id] = load

    def get_num_dofs(self) -> int:
        """
        Calculate total number of degrees of freedom in the structure.

        Returns:
            Total number of DOFs (6 * number of nodes for 3D)
            Supports 6 DOFs per node: ux, uy, uz, rx, ry, rz
        """
        if not self.nodes:
            return 0

        # Use 6 DOFs per node (3D frame analysis)
        # This supports both 2D and 3D elements
        return len(self.nodes) * 6

    def get_dof_map(self) -> Dict[int, List[int]]:
        """
        Create mapping from node ID to global DOF indices.

        Returns:
            Dictionary: node_id -> [dof_ux, dof_uy, dof_uz, dof_rx, dof_ry, dof_rz]
        """
        dof_map = {}
        for node_id in sorted(self.nodes.keys()):
            base_dof = node_id * 6
            dof_map[node_id] = [
                base_dof,
                base_dof + 1,
                base_dof + 2,
                base_dof + 3,
                base_dof + 4,
                base_dof + 5,
            ]

        return dof_map

    def assemble_global_stiffness(self) -> sparse.csr_matrix:
        """
        Assemble global stiffness matrix from element stiffness matrices.

        Uses scipy.sparse.lil_matrix for efficient assembly, then converts
        to csr_matrix for efficient solving.

        Returns:
            Global stiffness matrix as scipy.sparse.csr_matrix
        """
        num_dofs = self.get_num_dofs()

        # Use lil_matrix for efficient assembly
        K_global = sparse.lil_matrix((num_dofs, num_dofs))

        dof_map = self.get_dof_map()

        # Loop through elements and assemble
        for element in self.elements.values():
            # Get element stiffness in global coordinates
            K_elem = element.get_stiffness_global()

            # Get DOF indices for this element's nodes
            elem_dofs = []
            for node in element.nodes:
                node_dofs = dof_map[node.id]
                elem_dofs.extend(node_dofs)

            # Scatter element stiffness to global matrix
            for i, dof_i in enumerate(elem_dofs):
                for j, dof_j in enumerate(elem_dofs):
                    K_global[dof_i, dof_j] += K_elem[i, j]

        # Convert to CSR format for efficient solving
        return K_global.tocsr()

    def apply_boundary_conditions(
        self, K: sparse.csr_matrix, F: np.ndarray
    ) -> Tuple[sparse.csr_matrix, np.ndarray, Dict]:
        """
        Apply boundary conditions using penalty method.

        Multiplies diagonal terms by large number (1e12) for restrained DOFs.
        This effectively fixes those DOFs without removing rows/columns.

        Args:
            K: Global stiffness matrix
            F: Global force vector

        Returns:
            Tuple of (K_modified, F_modified, dof_info)
        """
        K_mod = K.tolil()  # Convert to lil for modification
        F_mod = F.copy()

        dof_map = self.get_dof_map()
        penalty = 1e6  # Reduced from 1e12 for better numerical stability

        dof_info = {"free": [], "restrained": []}

        # Apply penalty method for restrained DOFs
        for node_id, node in self.nodes.items():
            node_dofs = dof_map[node_id]

            # Check each DOF (ux, uy, uz, rx, ry, rz for 3D)
            for local_dof in range(6):
                global_dof = node_dofs[local_dof]

                if node.is_restrained(local_dof):
                    # Apply penalty method
                    # If DOF has stiffness, multiply it; otherwise set it
                    diag_value = K_mod[global_dof, global_dof]
                    if abs(diag_value) > 1e-10:
                        K_mod[global_dof, global_dof] *= penalty
                    else:
                        K_mod[global_dof, global_dof] = penalty
                    F_mod[global_dof] = 0.0  # Zero prescribed displacement
                    dof_info["restrained"].append(global_dof)
                else:
                    # Check if this DOF has zero stiffness (unconnected)
                    # This can happen for rotation DOFs in truss-only nodes
                    diag_value = K_mod[global_dof, global_dof]
                    if abs(diag_value) < 1e-10:
                        # Apply penalty to prevent singularity
                        K_mod[global_dof, global_dof] = penalty
                        F_mod[global_dof] = 0.0
                        dof_info["restrained"].append(global_dof)
                    else:
                        dof_info["free"].append(global_dof)

        return K_mod.tocsr(), F_mod, dof_info

    def assemble_load_vector(self) -> np.ndarray:
        """
        Assemble global load vector from applied loads.

        Returns:
            Global load vector
        """
        num_dofs = self.get_num_dofs()
        F = np.zeros(num_dofs)

        dof_map = self.get_dof_map()

        for load in self.loads.values():
            if load.node_id not in self.nodes:
                raise ValueError(f"Load applied to non-existent node {load.node_id}")

            node_dofs = dof_map[load.node_id]

            F[node_dofs[0]] += load.fx
            F[node_dofs[1]] += load.fy
            F[node_dofs[2]] += load.mz

        return F

    def solve_static(self) -> None:
        """
        Solve static equilibrium: K * U = F

        Assembles global stiffness matrix and load vector,
        applies boundary conditions, solves for displacements,
        and calculates reactions and internal forces.
        """
        # Assemble global system
        K = self.assemble_global_stiffness()
        F = self.assemble_load_vector()

        # Store original for reaction calculation
        K_original = K.copy()
        F_original = F.copy()

        # Apply boundary conditions
        K_mod, F_mod, dof_info = self.apply_boundary_conditions(K, F)

        # Solve for displacements
        U = spsolve(K_mod, F_mod)

        # Store displacements in nodes
        dof_map = self.get_dof_map()
        for node_id, node in self.nodes.items():
            node_dofs = dof_map[node_id]
            node.displacements[0] = U[node_dofs[0]]
            node.displacements[1] = U[node_dofs[1]]
            node.displacements[2] = U[node_dofs[2]]

        # Calculate reactions: R = K_original @ U - F_applied
        R = K_original @ U - F_original

        # Store reactions in nodes (only for restrained DOFs)
        for node_id, node in self.nodes.items():
            node_dofs = dof_map[node_id]

            for local_dof in range(3):
                global_dof = node_dofs[local_dof]
                if node.is_restrained(local_dof):
                    node.reactions[local_dof] = R[global_dof]

        # Calculate element internal forces
        element_forces = {}
        for elem_id, element in self.elements.items():
            # Get element displacements (all elements now use 3 DOFs per node)
            elem_dofs = []

            for node in element.nodes:
                node_dofs = dof_map[node.id]
                elem_dofs.extend(node_dofs)

            u_elem = U[elem_dofs]

            # Calculate internal forces
            forces = element.get_internal_forces(u_elem)
            element_forces[elem_id] = forces

        # Store results
        self.analysis_results = {
            "displacements": U,
            "reactions": R,
            "element_forces": element_forces,
            "dof_info": dof_info,
        }

    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary of analysis results as pandas DataFrame.

        Returns:
            DataFrame with node displacements, reactions, and element forces
        """
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run solve_static() first.")

        # Node results
        node_data = []
        for node_id, node in sorted(self.nodes.items()):
            node_data.append(
                {
                    "Type": "Node",
                    "ID": node_id,
                    "X": node.x,
                    "Y": node.y,
                    "Disp_X": node.displacements[0],
                    "Disp_Y": node.displacements[1],
                    "Rot_Z": node.displacements[2],
                    "React_X": node.reactions[0] if node.is_restrained(0) else 0.0,
                    "React_Y": node.reactions[1] if node.is_restrained(1) else 0.0,
                    "React_MZ": node.reactions[2] if node.is_restrained(2) else 0.0,
                }
            )

        df_nodes = pd.DataFrame(node_data)

        # Element results
        element_data = []
        for elem_id, forces in self.analysis_results["element_forces"].items():
            element = self.elements[elem_id]

            # Different format for truss vs beam
            if "axial_force" in forces:
                # Truss element
                element_data.append(
                    {
                        "Type": "Truss",
                        "ID": elem_id,
                        "Node_I": element.nodes[0].id,
                        "Node_J": element.nodes[1].id,
                        "Axial_Force": forces["axial_force"],
                        "Stress": forces["stress"],
                    }
                )
            else:
                # Beam element
                element_data.append(
                    {
                        "Type": "Beam",
                        "ID": elem_id,
                        "Node_I": element.nodes[0].id,
                        "Node_J": element.nodes[1].id,
                        "Axial_I": forces["node_i"]["axial"],
                        "Shear_I": forces["node_i"]["shear"],
                        "Moment_I": forces["node_i"]["moment"],
                        "Axial_J": forces["node_j"]["axial"],
                        "Shear_J": forces["node_j"]["shear"],
                        "Moment_J": forces["node_j"]["moment"],
                    }
                )

        df_elements = pd.DataFrame(element_data)

        # Combine results
        print("\n" + "=" * 80)
        print("STRUCTURAL ANALYSIS RESULTS")
        print("=" * 80)
        print("\nNODE RESULTS:")
        print("-" * 80)
        print(df_nodes.to_string(index=False))

        if not df_elements.empty:
            print("\nELEMENT RESULTS:")
            print("-" * 80)
            print(df_elements.to_string(index=False))

        print("=" * 80 + "\n")

        return df_nodes, df_elements

    def get_max_displacement(self) -> float:
        """Get maximum displacement magnitude in structure."""
        if not self.analysis_results:
            return 0.0

        max_disp = 0.0
        for node in self.nodes.values():
            disp_mag = np.sqrt(node.displacements[0] ** 2 + node.displacements[1] ** 2)
            max_disp = max(max_disp, disp_mag)

        return max_disp

    def __str__(self) -> str:
        """String representation of structure."""
        return (
            f"Structure("
            f"nodes={len(self.nodes)}, "
            f"elements={len(self.elements)}, "
            f"loads={len(self.loads)})"
        )
