"""
Advanced Optimization Algorithms
=================================

This module implements advanced algorithms for structural optimization:
1. Differential Evolution (DE) - Standard implementation
2. Classification-assisted Differential Evolution (CaDE) - ML-accelerated variant

These algorithms are particularly suited for continuous parameter optimization
(e.g., optimizing cross-sectional areas, dimensions).

References:
- Storn & Price (1997): Differential Evolution – A Simple and Efficient Heuristic
- PYTHON-CODE-GEN-AI.py: Reference implementation with CaDE
- https://github.com/f0uriest/GASTOp
"""

from typing import List, Tuple, Callable, Optional, Generator, Dict, Any
import numpy as np
from pyDOE import lhs
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass

from .evaluator import StructureEvaluator


@dataclass
class DEParameters:
    """Parameters for Differential Evolution algorithm."""
    mutation_factor: float = 0.8  # F: mutation scale factor
    crossover_prob: float = 0.7  # CR: crossover probability
    population_size: int = 50  # NP: population size
    max_iterations: int = 100  # Maximum generations
    strategy: str = "best1bin"  # DE strategy
    
    def __post_init__(self):
        """Validate DE parameters."""
        if not (0 < self.mutation_factor <= 2):
            raise ValueError("mutation_factor must be in (0, 2]")
        if not (0 <= self.crossover_prob <= 1):
            raise ValueError("crossover_prob must be in [0, 1]")
        if self.population_size < 4:
            raise ValueError("population_size must be >= 4")


@dataclass
class CaDEParameters(DEParameters):
    """Parameters for Classification-assisted Differential Evolution."""
    learning_phase_iterations: int = 20  # Phase I: model building
    n_estimators: int = 50  # AdaBoost estimators
    max_tree_depth: int = 1  # Decision tree depth
    
    def __post_init__(self):
        """Validate CaDE parameters."""
        super().__post_init__()
        if self.learning_phase_iterations >= self.max_iterations:
            raise ValueError("learning_phase_iterations must be < max_iterations")


class DifferentialEvolution:
    """
    Standard Differential Evolution optimizer for continuous variables.
    
    This implementation uses the DE/best/1/bin strategy:
    - Mutation: mutant = best + F*(x1 - x2)
    - Crossover: Binomial
    - Selection: Greedy
    
    Attributes:
        evaluator: StructureEvaluator for fitness calculation
        params: DEParameters configuration
        bounds: List of (min, max) tuples for each variable
    """
    
    def __init__(
        self,
        evaluator: StructureEvaluator,
        bounds: List[Tuple[float, float]],
        params: Optional[DEParameters] = None
    ):
        """
        Initialize DE optimizer.
        
        Args:
            evaluator: StructureEvaluator instance
            bounds: Variable bounds [(min1, max1), (min2, max2), ...]
            params: DE parameters (uses defaults if None)
        """
        self.evaluator = evaluator
        self.bounds = np.asarray(bounds)
        self.params = params or DEParameters()
        
        self.n_vars = len(bounds)
        self.min_b = self.bounds[:, 0]
        self.max_b = self.bounds[:, 1]
        self.diff = np.abs(self.min_b - self.max_b)
        
        # Population (normalized to [0, 1])
        self.population = None
        self.population_denorm = None
        self.fitness = None
        self.best_idx = None
        self.best = None
        self.best_denorm = None
        
        # Statistics
        self.generation = 0
        self.history = []
    
    def initialize_population(self) -> None:
        """Initialize population using Latin Hypercube Sampling."""
        # LHS in [0, 1]
        self.population = lhs(self.n_vars, samples=self.params.population_size)
        
        # Denormalize to actual bounds
        self.population_denorm = self.min_b + self.population * self.diff
        
        # Evaluate initial population
        self.fitness = np.array([
            self.evaluator.evaluate(ind)[0]
            for ind in self.population_denorm
        ])
        
        # Track best
        self.best_idx = np.argmin(self.fitness)
        self.best = self.population[self.best_idx].copy()
        self.best_denorm = self.population_denorm[self.best_idx].copy()
    
    def mutate(self, idx: int) -> np.ndarray:
        """
        Generate mutant vector using DE/best/1 strategy.
        
        Args:
            idx: Index of target vector
        
        Returns:
            Mutant vector (normalized)
        """
        # Select two random vectors (different from idx and each other)
        candidates = [i for i in range(self.params.population_size) if i != idx]
        x1_idx, x2_idx = np.random.choice(candidates, 2, replace=False)
        
        x1 = self.population[x1_idx]
        x2 = self.population[x2_idx]
        
        # Mutant = target + F*(best - target) + F*(x1 - x2)
        F = self.params.mutation_factor
        mutant = self.population[idx] + F * (self.best - self.population[idx]) + F * (x1 - x2)
        
        # Clip to [0, 1]
        mutant = np.clip(mutant, 0, 1)
        
        return mutant
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Binomial crossover between target and mutant.
        
        Args:
            target: Target vector
            mutant: Mutant vector
        
        Returns:
            Trial vector
        """
        CR = self.params.crossover_prob
        
        # Random crossover points
        cross_points = np.random.rand(self.n_vars) < CR
        
        # Ensure at least one gene from mutant
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.n_vars)] = True
        
        # Create trial
        trial = np.where(cross_points, mutant, target)
        
        return trial
    
    def evolve_generation(self) -> None:
        """Execute one generation of DE."""
        for j in range(self.params.population_size):
            # Generate mutant
            mutant = self.mutate(j)
            
            # Crossover
            trial = self.crossover(self.population[j], mutant)
            
            # Denormalize
            trial_denorm = self.min_b + trial * self.diff
            
            # Evaluate
            trial_fitness = self.evaluator.evaluate(trial_denorm)[0]
            
            # Selection (greedy)
            if trial_fitness < self.fitness[j]:
                self.fitness[j] = trial_fitness
                self.population[j] = trial
                self.population_denorm[j] = trial_denorm
                
                # Update best
                if trial_fitness < self.fitness[self.best_idx]:
                    self.best_idx = j
                    self.best = trial.copy()
                    self.best_denorm = trial_denorm.copy()
    
    def optimize(self) -> Generator[Dict[str, Any], None, None]:
        """
        Run DE optimization.
        
        Yields:
            Dictionary with generation results:
            - generation: int
            - best_solution: np.ndarray
            - best_fitness: float
            - n_evaluations: int
        """
        # Initialize
        self.initialize_population()
        
        # Record initial best
        yield {
            'generation': 0,
            'best_solution': self.best_denorm.copy(),
            'best_fitness': self.fitness[self.best_idx],
            'n_evaluations': self.evaluator.n_evaluations
        }
        
        # Evolution
        for gen in range(1, self.params.max_iterations):
            self.generation = gen
            self.evolve_generation()
            
            # Record history
            result = {
                'generation': gen,
                'best_solution': self.best_denorm.copy(),
                'best_fitness': self.fitness[self.best_idx],
                'n_evaluations': self.evaluator.n_evaluations
            }
            self.history.append(result)
            
            yield result


class ClassificationAssistedDE:
    """
    Classification-assisted Differential Evolution (CaDE).
    
    This variant uses machine learning to predict constraint feasibility,
    skipping expensive FEA for likely-infeasible designs.
    
    Algorithm:
    - Phase I (Learning): Standard DE, collect training data
    - Phase II (Employing): Use classifier to filter trial vectors
    
    Benefits:
    - Reduced computational cost (fewer FEA evaluations)
    - Faster convergence for constrained problems
    
    Attributes:
        evaluator: StructureEvaluator for fitness calculation
        params: CaDEParameters configuration
        classifier: AdaBoost classifier for feasibility prediction
    """
    
    def __init__(
        self,
        evaluator: StructureEvaluator,
        bounds: List[Tuple[float, float]],
        params: Optional[CaDEParameters] = None
    ):
        """
        Initialize CaDE optimizer.
        
        Args:
            evaluator: StructureEvaluator instance
            bounds: Variable bounds
            params: CaDE parameters
        """
        self.evaluator = evaluator
        self.bounds = np.asarray(bounds)
        self.params = params or CaDEParameters()
        
        self.n_vars = len(bounds)
        self.min_b = self.bounds[:, 0]
        self.max_b = self.bounds[:, 1]
        self.diff = np.abs(self.min_b - self.max_b)
        
        # Population
        self.population = None
        self.population_denorm = None
        self.fitness = None
        self.best_idx = None
        self.best = None
        self.best_denorm = None
        
        # Training data for classifier
        self.X_train = []  # Design vectors
        self.y_train = []  # Labels: +1 = feasible, -1 = infeasible
        
        # Classifier
        self.classifier = None
        
        # Statistics
        self.generation = 0
        self.history = []
        self.n_skipped_fea = 0  # FEA calls saved by classifier
    
    @staticmethod
    def assign_label(constraint_violation: float) -> int:
        """
        Assign feasibility label based on constraint violation.
        
        Args:
            constraint_violation: Violation value (0 = feasible)
        
        Returns:
            +1 if feasible, -1 if infeasible
        """
        return 1 if constraint_violation == 0 else -1
    
    def initialize_population(self) -> None:
        """Initialize population and collect training data."""
        # LHS sampling
        self.population = lhs(self.n_vars, samples=self.params.population_size)
        self.population_denorm = self.min_b + self.population * self.diff
        
        # Evaluate and collect training data
        self.fitness = np.zeros(self.params.population_size)
        
        for j in range(self.params.population_size):
            # Evaluate
            fitness_val = self.evaluator.evaluate(self.population_denorm[j])[0]
            self.fitness[j] = fitness_val
            
            # Collect training data
            # Note: In practice, need to access constraint violation from evaluator
            # For now, use heuristic: high fitness = infeasible
            weight_estimate = self.calculate_weight(self.population_denorm[j])
            violation = fitness_val - weight_estimate
            label = self.assign_label(violation)
            
            self.X_train.append(self.population_denorm[j])
            self.y_train.append(label)
        
        # Track best
        self.best_idx = np.argmin(self.fitness)
        self.best = self.population[self.best_idx].copy()
        self.best_denorm = self.population_denorm[self.best_idx].copy()
    
    def calculate_weight(self, individual: np.ndarray) -> float:
        """
        Estimate weight from design variables (for labeling).
        
        Args:
            individual: Design vector
        
        Returns:
            Estimated weight
        """
        # Simplified: assume weight proportional to sum of areas
        return np.sum(individual)
    
    def train_classifier(self) -> None:
        """Train AdaBoost classifier on collected data."""
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        
        self.classifier = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=self.params.max_tree_depth),
            n_estimators=self.params.n_estimators,
            random_state=42
        )
        
        self.classifier.fit(X, y)
    
    def mutate(self, idx: int) -> np.ndarray:
        """Generate mutant vector (same as DE)."""
        candidates = [i for i in range(self.params.population_size) if i != idx]
        x1_idx, x2_idx = np.random.choice(candidates, 2, replace=False)
        
        x1 = self.population[x1_idx]
        x2 = self.population[x2_idx]
        
        F = self.params.mutation_factor
        mutant = self.population[idx] + F * (self.best - self.population[idx]) + F * (x1 - x2)
        mutant = np.clip(mutant, 0, 1)
        
        return mutant
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Binomial crossover (same as DE)."""
        CR = self.params.crossover_prob
        cross_points = np.random.rand(self.n_vars) < CR
        
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.n_vars)] = True
        
        trial = np.where(cross_points, mutant, target)
        return trial
    
    def evolve_learning_phase(self) -> None:
        """
        Phase I: Learning phase with standard DE.
        Collect training data for classifier.
        """
        for j in range(self.params.population_size):
            mutant = self.mutate(j)
            trial = self.crossover(self.population[j], mutant)
            trial_denorm = self.min_b + trial * self.diff
            
            # Evaluate (always in learning phase)
            trial_fitness = self.evaluator.evaluate(trial_denorm)[0]
            
            # Collect training data
            weight_estimate = self.calculate_weight(trial_denorm)
            violation = trial_fitness - weight_estimate
            label = self.assign_label(violation)
            
            self.X_train.append(trial_denorm)
            self.y_train.append(label)
            
            # Selection
            if trial_fitness < self.fitness[j]:
                self.fitness[j] = trial_fitness
                self.population[j] = trial
                self.population_denorm[j] = trial_denorm
                
                if trial_fitness < self.fitness[self.best_idx]:
                    self.best_idx = j
                    self.best = trial.copy()
                    self.best_denorm = trial_denorm.copy()
    
    def evolve_employing_phase(self) -> None:
        """
        Phase II: Employing phase with classifier-assisted evaluation.
        Skip FEA for predicted infeasible designs.
        """
        for j in range(self.params.population_size):
            mutant = self.mutate(j)
            trial = self.crossover(self.population[j], mutant)
            trial_denorm = self.min_b + trial * self.diff
            
            # Predict feasibility
            trial_pred = self.classifier.predict(trial_denorm.reshape(1, -1))[0]
            
            if trial_pred == 1:  # Predicted feasible
                # Evaluate with FEA
                trial_fitness = self.evaluator.evaluate(trial_denorm)[0]
                
                if trial_fitness < self.fitness[j]:
                    self.fitness[j] = trial_fitness
                    self.population[j] = trial
                    self.population_denorm[j] = trial_denorm
                    
                    if trial_fitness < self.fitness[self.best_idx]:
                        self.best_idx = j
                        self.best = trial.copy()
                        self.best_denorm = trial_denorm.copy()
            
            elif trial_pred == -1:  # Predicted infeasible
                # Only evaluate if lighter than current
                trial_weight = self.calculate_weight(trial_denorm)
                current_weight = self.calculate_weight(self.population_denorm[j])
                
                if trial_weight < current_weight:
                    # Worth checking
                    trial_fitness = self.evaluator.evaluate(trial_denorm)[0]
                    
                    if trial_fitness < self.fitness[j]:
                        self.fitness[j] = trial_fitness
                        self.population[j] = trial
                        self.population_denorm[j] = trial_denorm
                        
                        if trial_fitness < self.fitness[self.best_idx]:
                            self.best_idx = j
                            self.best = trial.copy()
                            self.best_denorm = trial_denorm.copy()
                else:
                    # Skip FEA
                    self.n_skipped_fea += 1
    
    def optimize(self) -> Generator[Dict[str, Any], None, None]:
        """
        Run CaDE optimization.
        
        Yields:
            Dictionary with generation results including FEA savings
        """
        # Initialize
        self.initialize_population()
        
        yield {
            'generation': 0,
            'best_solution': self.best_denorm.copy(),
            'best_fitness': self.fitness[self.best_idx],
            'n_evaluations': self.evaluator.n_evaluations,
            'n_skipped_fea': 0,
            'phase': 'initialization'
        }
        
        # Phase I: Learning
        for gen in range(1, self.params.learning_phase_iterations):
            self.generation = gen
            self.evolve_learning_phase()
            
            yield {
                'generation': gen,
                'best_solution': self.best_denorm.copy(),
                'best_fitness': self.fitness[self.best_idx],
                'n_evaluations': self.evaluator.n_evaluations,
                'n_skipped_fea': self.n_skipped_fea,
                'phase': 'learning'
            }
        
        # Train classifier
        self.train_classifier()
        
        # Phase II: Employing
        for gen in range(self.params.learning_phase_iterations, self.params.max_iterations):
            self.generation = gen
            self.evolve_employing_phase()
            
            result = {
                'generation': gen,
                'best_solution': self.best_denorm.copy(),
                'best_fitness': self.fitness[self.best_idx],
                'n_evaluations': self.evaluator.n_evaluations,
                'n_skipped_fea': self.n_skipped_fea,
                'phase': 'employing'
            }
            self.history.append(result)
            
            yield result
