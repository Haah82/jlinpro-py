"""
Structure Evaluator for Optimization
=====================================

This module provides the fitness evaluation function for structural optimization.

The evaluator:
1. Maps optimization variables (discrete catalog indices or continuous areas) to structure
2. Runs finite element analysis
3. Checks constraints (stress, displacement, design code checks)
4. Returns fitness value (weight + penalty for constraint violations)
"""

from typing import List, Dict, Tuple, Any, Optional, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum


class OptimizationMode(Enum):
    """Optimization variable encoding mode."""
    DISCRETE = "discrete"  # Catalog-based (integer indices)
    CONTINUOUS = "continuous"  # Parameter-based (real values)


@dataclass
class ConstraintLimits:
    """Constraint limit values for structural optimization."""
    max_stress: float = 150.0  # MPa
    max_displacement: float = 25.0  # mm
    max_utilization_ratio: float = 1.0  # Design code check UR limit
    
    def __post_init__(self):
        """Validate constraint limits."""
        if self.max_stress <= 0:
            raise ValueError("max_stress must be positive")
        if self.max_displacement <= 0:
            raise ValueError("max_displacement must be positive")
        if self.max_utilization_ratio <= 0:
            raise ValueError("max_utilization_ratio must be positive")


@dataclass
class OptimizationProblem:
    """Definition of a structural optimization problem."""
    mode: OptimizationMode
    n_variables: int
    bounds: List[Tuple[float, float]]
    constraints: ConstraintLimits
    material_density: float = 7850.0  # kg/m³ (steel default)
    
    # Catalog for discrete mode
    section_catalog: Optional[List[Dict[str, float]]] = None  # List of {A, I, name}
    
    # Penalty factors
    penalty_stress: float = 1000.0  # Penalty per MPa over limit
    penalty_displacement: float = 5000.0  # Penalty per mm over limit
    penalty_ur: float = 10000.0  # Penalty for design code violations
    
    def __post_init__(self):
        """Validate optimization problem definition."""
        if len(self.bounds) != self.n_variables:
            raise ValueError(f"bounds length {len(self.bounds)} != n_variables {self.n_variables}")
        
        if self.mode == OptimizationMode.DISCRETE:
            if self.section_catalog is None or len(self.section_catalog) == 0:
                raise ValueError("section_catalog required for DISCRETE mode")


class StructureEvaluator:
    """
    Evaluator for structural optimization fitness function.
    
    This class handles:
    - Mapping optimization variables to structural properties
    - Running FEA
    - Constraint checking
    - Fitness calculation with penalties
    
    Attributes:
        problem: OptimizationProblem definition
        structure: Reference to Structure object (to be analyzed)
        design_code: Optional design code checker (TCVN, ACI, EC2)
        analysis_func: Custom function for running FEA (if not using structure.solve_static)
    """
    
    def __init__(
        self,
        problem: OptimizationProblem,
        structure: Any,  # Structure object
        design_code: Optional[Any] = None,
        analysis_func: Optional[Callable] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            problem: Optimization problem definition
            structure: Structure object to optimize
            design_code: Design code checker (optional)
            analysis_func: Custom analysis function (optional)
        """
        self.problem = problem
        self.structure = structure
        self.design_code = design_code
        self.analysis_func = analysis_func
        
        # Statistics
        self.n_evaluations = 0
        self.n_feasible = 0
        self.n_infeasible = 0
    
    def decode_variables(self, individual: np.ndarray) -> List[Dict[str, float]]:
        """
        Decode optimization variables to structural properties.
        
        Args:
            individual: Array of optimization variables
        
        Returns:
            List of section properties for each element
        """
        if self.problem.mode == OptimizationMode.DISCRETE:
            # Map integer indices to catalog sections
            indices = individual.astype(int)
            return [self.problem.section_catalog[idx] for idx in indices]
        
        else:  # CONTINUOUS
            # Directly use continuous values (e.g., cross-sectional areas)
            return [{'A': A, 'I': None} for A in individual]
    
    def apply_sections_to_structure(self, sections: List[Dict[str, float]]) -> None:
        """
        Apply decoded section properties to structure elements.
        
        Args:
            sections: List of section properties
        """
        if len(sections) != len(self.structure.elements):
            raise ValueError(
                f"Number of sections {len(sections)} != number of elements {len(self.structure.elements)}"
            )
        
        for elem_id, section in zip(self.structure.elements.keys(), sections):
            element = self.structure.elements[elem_id]
            
            # Update element section properties
            if hasattr(element, 'section'):
                element.section.A = section['A']
                if section.get('I') is not None:
                    element.section.I = section['I']
    
    def calculate_weight(self, sections: List[Dict[str, float]]) -> float:
        """
        Calculate total structure weight.
        
        Args:
            sections: List of section properties
        
        Returns:
            Total weight in kg
        """
        weight = 0.0
        for elem_id, section in zip(self.structure.elements.keys(), sections):
            element = self.structure.elements[elem_id]
            length = element.get_length()  # mm
            area = section['A']  # mm²
            
            # Volume = A * L (mm³), convert to m³
            volume = area * length / 1e9  # m³
            
            # Weight = ρ * V (kg)
            weight += self.problem.material_density * volume
        
        return weight
    
    def check_constraints(self) -> Tuple[float, Dict[str, Any]]:
        """
        Check structural constraints after analysis.
        
        Returns:
            (constraint_violation, details_dict)
            constraint_violation: Sum of all violations (0 if feasible)
        """
        violation = 0.0
        details = {
            'max_stress': 0.0,
            'max_displacement': 0.0,
            'max_ur': 0.0,
            'stress_violation': 0.0,
            'disp_violation': 0.0,
            'ur_violation': 0.0
        }
        
        # Check stresses
        if hasattr(self.structure, 'analysis_results') and 'element_forces' in self.structure.analysis_results:
            for elem_id, forces in self.structure.analysis_results['element_forces'].items():
                element = self.structure.elements[elem_id]
                
                # Calculate stress: σ = N/A
                if 'N' in forces:
                    axial_force = abs(forces['N'])  # N
                    section = element.section
                    stress = axial_force / section.A  # N/mm² = MPa
                    details['max_stress'] = max(details['max_stress'], stress)
                    
                    if stress > self.problem.constraints.max_stress:
                        overstress = stress - self.problem.constraints.max_stress
                        details['stress_violation'] += overstress
                        violation += overstress * self.problem.penalty_stress
        
        # Check displacements
        if hasattr(self.structure, 'analysis_results') and 'displacements' in self.structure.analysis_results:
            displacements = self.structure.analysis_results['displacements']
            for node_id, disp in displacements.items():
                # Calculate magnitude
                disp_mag = np.linalg.norm(list(disp.values()))  # mm
                details['max_displacement'] = max(details['max_displacement'], disp_mag)
                
                if disp_mag > self.problem.constraints.max_displacement:
                    overdisp = disp_mag - self.problem.constraints.max_displacement
                    details['disp_violation'] += overdisp
                    violation += overdisp * self.problem.penalty_displacement
        
        # Check design code (if provided)
        if self.design_code is not None:
            # Simplified: check if any element has UR > limit
            # In practice, would run actual design checks
            pass  # Placeholder for future integration
        
        return violation, details
    
    def evaluate(self, individual: np.ndarray) -> Tuple[float]:
        """
        Main evaluation function for optimization.
        
        This is the fitness function called by optimization algorithms.
        
        Args:
            individual: Array of optimization variables (normalized or indices)
        
        Returns:
            Tuple containing single fitness value (weight + penalties)
            Note: DEAP requires tuple return even for single objective
        """
        self.n_evaluations += 1
        
        try:
            # 1. Decode variables
            sections = self.decode_variables(individual)
            
            # 2. Apply to structure
            self.apply_sections_to_structure(sections)
            
            # 3. Run FEA
            if self.analysis_func is not None:
                self.analysis_func(self.structure)
            else:
                self.structure.solve_static()
            
            # 4. Calculate weight
            weight = self.calculate_weight(sections)
            
            # 5. Check constraints
            violation, details = self.check_constraints()
            
            # 6. Calculate fitness
            if violation > 0:
                self.n_infeasible += 1
                fitness = weight + violation
            else:
                self.n_feasible += 1
                fitness = weight
            
            return (fitness,)  # Tuple for DEAP compatibility
        
        except Exception as e:
            # Handle analysis failures
            print(f"Evaluation failed: {e}")
            return (1e10,)  # Large penalty
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get evaluation statistics.
        
        Returns:
            Dictionary with evaluation counts
        """
        return {
            'total_evaluations': self.n_evaluations,
            'feasible': self.n_feasible,
            'infeasible': self.n_infeasible,
            'feasibility_rate': self.n_feasible / max(self.n_evaluations, 1)
        }
    
    def reset_statistics(self) -> None:
        """Reset evaluation counters."""
        self.n_evaluations = 0
        self.n_feasible = 0
        self.n_infeasible = 0


def create_truss_catalog() -> List[Dict[str, float]]:
    """
    Create a catalog of standard truss sections.
    
    Returns:
        List of section dictionaries with A, I, name
    """
    catalog = [
        {'A': 100, 'I': None, 'name': 'Sect-100'},
        {'A': 200, 'I': None, 'name': 'Sect-200'},
        {'A': 300, 'I': None, 'name': 'Sect-300'},
        {'A': 400, 'I': None, 'name': 'Sect-400'},
        {'A': 500, 'I': None, 'name': 'Sect-500'},
        {'A': 600, 'I': None, 'name': 'Sect-600'},
        {'A': 800, 'I': None, 'name': 'Sect-800'},
        {'A': 1000, 'I': None, 'name': 'Sect-1000'},
        {'A': 1200, 'I': None, 'name': 'Sect-1200'},
        {'A': 1500, 'I': None, 'name': 'Sect-1500'},
    ]
    return catalog


def create_beam_catalog() -> List[Dict[str, float]]:
    """
    Create a catalog of standard beam sections.
    
    Returns:
        List of section dictionaries with A, Ix, Iy, name
    """
    catalog = [
        {'A': 2000, 'I': 1.67e6, 'name': 'Beam-200x100'},
        {'A': 3000, 'I': 5.62e6, 'name': 'Beam-300x100'},
        {'A': 4500, 'I': 1.69e7, 'name': 'Beam-300x150'},
        {'A': 6000, 'I': 4.50e7, 'name': 'Beam-400x150'},
        {'A': 8000, 'I': 1.07e8, 'name': 'Beam-500x160'},
        {'A': 10000, 'I': 2.08e8, 'name': 'Beam-500x200'},
        {'A': 12000, 'I': 3.60e8, 'name': 'Beam-600x200'},
        {'A': 15000, 'I': 7.03e8, 'name': 'Beam-750x200'},
    ]
    return catalog
