"""
Unified Optimization Interface
================================

This module provides a facade for running structural optimization
with different strategies (GA, DE, CaDE).

Main function: run_optimization()
"""

from typing import Dict, Any, Optional, List, Tuple, Generator
from enum import Enum
import numpy as np

from .evaluator import (
    StructureEvaluator,
    OptimizationProblem,
    OptimizationMode,
    ConstraintLimits,
    create_truss_catalog,
    create_beam_catalog
)
from .ga_deap import GeneticAlgorithm, GAParameters
from .advanced_algorithms import (
    DifferentialEvolution,
    ClassificationAssistedDE,
    DEParameters,
    CaDEParameters
)


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    GA_DISCRETE = "ga_discrete"  # Genetic Algorithm (catalog-based)
    DE_CONTINUOUS = "de_continuous"  # Differential Evolution
    CADE_CONTINUOUS = "cade_continuous"  # Classification-assisted DE


def run_optimization(
    structure: Any,
    strategy: OptimizationStrategy,
    n_variables: Optional[int] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    constraints: Optional[ConstraintLimits] = None,
    section_catalog: Optional[List[Dict[str, float]]] = None,
    design_code: Optional[Any] = None,
    custom_params: Optional[Dict[str, Any]] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Run structural optimization with specified strategy.
    
    This is the main entry point for optimization. It automatically:
    1. Creates OptimizationProblem
    2. Sets up StructureEvaluator
    3. Initializes appropriate optimizer
    4. Runs optimization and yields results
    
    Args:
        structure: Structure object to optimize
        strategy: OptimizationStrategy enum value
        n_variables: Number of design variables (default: number of elements)
        bounds: Variable bounds (required for continuous strategies)
        constraints: ConstraintLimits object
        section_catalog: Catalog of sections (required for discrete strategies)
        design_code: Design code checker (optional)
        custom_params: Custom parameters for optimizer
    
    Yields:
        Dictionary with optimization results for each generation:
        - generation: int
        - best_solution: np.ndarray
        - best_fitness: float
        - n_evaluations: int
        - (strategy-specific fields)
    
    Example:
        >>> from src.optimization import run_optimization, OptimizationStrategy
        >>> for result in run_optimization(structure, OptimizationStrategy.GA_DISCRETE):
        ...     print(f"Gen {result['generation']}: Fitness = {result['best_fitness']:.2f}")
    """
    # Default values
    if n_variables is None:
        n_variables = len(structure.elements)
    
    if constraints is None:
        constraints = ConstraintLimits()
    
    # Create optimization problem based on strategy
    if strategy == OptimizationStrategy.GA_DISCRETE:
        # Discrete mode
        if section_catalog is None:
            # Use default catalog based on element type
            section_catalog = create_truss_catalog()  # Or create_beam_catalog()
        
        # Bounds are indices into catalog
        bounds = [(0, len(section_catalog) - 1) for _ in range(n_variables)]
        
        problem = OptimizationProblem(
            mode=OptimizationMode.DISCRETE,
            n_variables=n_variables,
            bounds=bounds,
            constraints=constraints,
            section_catalog=section_catalog
        )
        
    elif strategy in [OptimizationStrategy.DE_CONTINUOUS, OptimizationStrategy.CADE_CONTINUOUS]:
        # Continuous mode
        if bounds is None:
            raise ValueError("bounds required for continuous optimization")
        
        problem = OptimizationProblem(
            mode=OptimizationMode.CONTINUOUS,
            n_variables=n_variables,
            bounds=bounds,
            constraints=constraints
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Create evaluator
    evaluator = StructureEvaluator(
        problem=problem,
        structure=structure,
        design_code=design_code
    )
    
    # Initialize optimizer
    if strategy == OptimizationStrategy.GA_DISCRETE:
        # GA parameters
        ga_params = GAParameters(**(custom_params or {}))
        optimizer = GeneticAlgorithm(
            evaluator=evaluator,
            n_catalog=len(section_catalog),
            params=ga_params
        )
        
    elif strategy == OptimizationStrategy.DE_CONTINUOUS:
        # DE parameters
        de_params = DEParameters(**(custom_params or {}))
        optimizer = DifferentialEvolution(
            evaluator=evaluator,
            bounds=bounds,
            params=de_params
        )
        
    elif strategy == OptimizationStrategy.CADE_CONTINUOUS:
        # CaDE parameters
        cade_params = CaDEParameters(**(custom_params or {}))
        optimizer = ClassificationAssistedDE(
            evaluator=evaluator,
            bounds=bounds,
            params=cade_params
        )
    
    # Run optimization
    for result in optimizer.optimize():
        # Add evaluator statistics
        result['evaluator_stats'] = evaluator.get_statistics()
        
        yield result
    
    # Final statistics
    final_stats = evaluator.get_statistics()
    print(f"\nOptimization complete:")
    print(f"  Total evaluations: {final_stats['total_evaluations']}")
    print(f"  Feasible solutions: {final_stats['feasible']}")
    print(f"  Infeasible solutions: {final_stats['infeasible']}")
    print(f"  Feasibility rate: {final_stats['feasibility_rate']:.2%}")


def quick_optimize_discrete(
    structure: Any,
    catalog: Optional[List[Dict[str, float]]] = None,
    max_generations: int = 50,
    population_size: int = 100
) -> Dict[str, Any]:
    """
    Quick discrete optimization with GA (convenience function).
    
    Args:
        structure: Structure to optimize
        catalog: Section catalog (uses default if None)
        max_generations: Number of generations
        population_size: Population size
    
    Returns:
        Dictionary with final optimization result
    """
    params = {
        'max_generations': max_generations,
        'population_size': population_size
    }
    
    results = []
    for result in run_optimization(
        structure=structure,
        strategy=OptimizationStrategy.GA_DISCRETE,
        section_catalog=catalog,
        custom_params=params
    ):
        results.append(result)
        print(f"Gen {result['generation']:3d}: Fitness = {result['best_fitness']:10.2f} kg")
    
    return results[-1]


def quick_optimize_continuous(
    structure: Any,
    bounds: List[Tuple[float, float]],
    max_iterations: int = 50,
    use_cade: bool = False
) -> Dict[str, Any]:
    """
    Quick continuous optimization with DE or CaDE (convenience function).
    
    Args:
        structure: Structure to optimize
        bounds: Variable bounds
        max_iterations: Number of iterations
        use_cade: Use CaDE instead of DE
    
    Returns:
        Dictionary with final optimization result
    """
    strategy = OptimizationStrategy.CADE_CONTINUOUS if use_cade else OptimizationStrategy.DE_CONTINUOUS
    
    if use_cade:
        params = {
            'max_iterations': max_iterations,
            'learning_phase_iterations': min(20, max_iterations // 2)
        }
    else:
        params = {
            'max_iterations': max_iterations
        }
    
    results = []
    for result in run_optimization(
        structure=structure,
        strategy=strategy,
        bounds=bounds,
        custom_params=params
    ):
        results.append(result)
        
        if use_cade and 'n_skipped_fea' in result:
            print(f"Gen {result['generation']:3d}: Fitness = {result['best_fitness']:10.2f} "
                  f"(FEA skipped: {result['n_skipped_fea']})")
        else:
            print(f"Gen {result['generation']:3d}: Fitness = {result['best_fitness']:10.2f}")
    
    return results[-1]


def compare_strategies(
    structure: Any,
    bounds: List[Tuple[float, float]],
    catalog: Optional[List[Dict[str, float]]] = None,
    max_iterations: int = 50
) -> Dict[str, Dict[str, Any]]:
    """
    Compare all three optimization strategies on the same problem.
    
    Args:
        structure: Structure to optimize
        bounds: Variable bounds (for continuous)
        catalog: Section catalog (for discrete)
        max_iterations: Number of iterations for each strategy
    
    Returns:
        Dictionary mapping strategy name to final result
    """
    comparison = {}
    
    # GA Discrete
    print("\n" + "="*60)
    print("Running GA (Discrete)")
    print("="*60)
    ga_result = quick_optimize_discrete(structure, catalog, max_iterations)
    comparison['GA_Discrete'] = ga_result
    
    # DE Continuous
    print("\n" + "="*60)
    print("Running DE (Continuous)")
    print("="*60)
    de_result = quick_optimize_continuous(structure, bounds, max_iterations, use_cade=False)
    comparison['DE_Continuous'] = de_result
    
    # CaDE Continuous
    print("\n" + "="*60)
    print("Running CaDE (Continuous)")
    print("="*60)
    cade_result = quick_optimize_continuous(structure, bounds, max_iterations, use_cade=True)
    comparison['CaDE_Continuous'] = cade_result
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    for name, result in comparison.items():
        print(f"{name:20s}: Fitness = {result['best_fitness']:10.2f} kg, "
              f"Evaluations = {result['n_evaluations']}")
    
    return comparison
