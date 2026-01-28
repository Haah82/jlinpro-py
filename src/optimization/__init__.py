"""
Optimization module for structural design optimization.

This module provides various optimization algorithms including:
- Standard Genetic Algorithm (DEAP-based)
- Differential Evolution (DE)
- Classification-assisted Differential Evolution (CaDE)
"""

from .evaluator import (
    StructureEvaluator,
    OptimizationProblem,
    OptimizationMode,
    ConstraintLimits,
    create_truss_catalog,
    create_beam_catalog
)
from .ga_optimizer import (
    run_optimization,
    OptimizationStrategy,
    quick_optimize_discrete,
    quick_optimize_continuous,
    compare_strategies
)
from .ga_deap import GeneticAlgorithm, GAParameters
from .advanced_algorithms import (
    DifferentialEvolution,
    ClassificationAssistedDE,
    DEParameters,
    CaDEParameters
)

__all__ = [
    # Evaluator
    'StructureEvaluator',
    'OptimizationProblem',
    'OptimizationMode',
    'ConstraintLimits',
    'create_truss_catalog',
    'create_beam_catalog',
    # Main interface
    'run_optimization',
    'OptimizationStrategy',
    'quick_optimize_discrete',
    'quick_optimize_continuous',
    'compare_strategies',
    # GA
    'GeneticAlgorithm',
    'GAParameters',
    # DE/CaDE
    'DifferentialEvolution',
    'ClassificationAssistedDE',
    'DEParameters',
    'CaDEParameters'
]
