"""
Genetic Algorithm Optimizer using DEAP
=======================================

This module implements discrete catalog-based optimization using DEAP library.

The GA encodes each structural element as an integer index into a catalog
of standard sections (e.g., steel profiles, concrete beam sizes).

References:
- DEAP Documentation: https://deap.readthedocs.io/
- https://github.com/f0uriest/GASTOp
"""

from typing import List, Tuple, Optional, Dict, Any, Generator
import numpy as np
from dataclasses import dataclass
import random

# DEAP imports
from deap import base, creator, tools, algorithms

from .evaluator import StructureEvaluator, OptimizationMode


@dataclass
class GAParameters:
    """Parameters for Genetic Algorithm."""
    population_size: int = 100
    max_generations: int = 100
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    tournament_size: int = 3
    elite_size: int = 2  # Number of elites to preserve
    
    def __post_init__(self):
        """Validate GA parameters."""
        if self.population_size < 4:
            raise ValueError("population_size must be >= 4")
        if not (0 <= self.crossover_prob <= 1):
            raise ValueError("crossover_prob must be in [0, 1]")
        if not (0 <= self.mutation_prob <= 1):
            raise ValueError("mutation_prob must be in [0, 1]")
        if self.elite_size >= self.population_size:
            raise ValueError("elite_size must be < population_size")


class GeneticAlgorithm:
    """
    Genetic Algorithm for discrete catalog-based optimization.
    
    Encoding: Each gene is an integer index into section catalog.
    
    Operators:
    - Selection: Tournament
    - Crossover: Two-point
    - Mutation: Uniform (random catalog selection)
    - Elitism: Preserve best individuals
    
    Attributes:
        evaluator: StructureEvaluator for fitness calculation
        params: GAParameters configuration
        n_catalog: Number of sections in catalog
    """
    
    def __init__(
        self,
        evaluator: StructureEvaluator,
        n_catalog: int,
        params: Optional[GAParameters] = None
    ):
        """
        Initialize GA optimizer.
        
        Args:
            evaluator: StructureEvaluator instance (must be DISCRETE mode)
            n_catalog: Number of sections in catalog
            params: GA parameters
        """
        if evaluator.problem.mode != OptimizationMode.DISCRETE:
            raise ValueError("GA requires DISCRETE optimization mode")
        
        self.evaluator = evaluator
        self.n_catalog = n_catalog
        self.params = params or GAParameters()
        self.n_vars = evaluator.problem.n_variables
        
        # DEAP setup
        self._setup_deap()
        
        # Population
        self.population = None
        self.halloffame = tools.HallOfFame(maxsize=10)  # Initialize here
        
        # Statistics
        self.logbook = tools.Logbook()  # Initialize here
        self.generation = 0
    
    def _setup_deap(self) -> None:
        """Configure DEAP framework."""
        # Create fitness class (minimize single objective)
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        
        # Create individual class
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Toolbox
        self.toolbox = base.Toolbox()
        
        # Attribute generator: random catalog index
        self.toolbox.register(
            "attr_int",
            random.randint,
            0,
            self.n_catalog - 1
        )
        
        # Individual generator
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_int,
            n=self.n_vars
        )
        
        # Population generator
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )
        
        # Operators
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate",
            tools.mutUniformInt,
            low=0,
            up=self.n_catalog - 1,
            indpb=0.2  # Independent probability per gene
        )
        self.toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.params.tournament_size
        )
        
        # Statistics
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        # Logbook
        self.logbook = tools.Logbook()
    
    def _evaluate(self, individual: List[int]) -> Tuple[float]:
        """
        Wrapper for evaluator to match DEAP interface.
        
        Args:
            individual: List of catalog indices
        
        Returns:
            Tuple with fitness value
        """
        # Convert to numpy array
        individual_array = np.array(individual, dtype=float)
        
        # Call evaluator
        return self.evaluator.evaluate(individual_array)
    
    def initialize_population(self) -> None:
        """Generate initial population."""
        self.population = self.toolbox.population(n=self.params.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, self.population))
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit
        
        # Update hall of fame
        self.halloffame.update(self.population)
    
    def evolve_generation(self) -> None:
        """Execute one generation of GA."""
        # Select next generation
        offspring = self.toolbox.select(self.population, len(self.population))
        offspring = list(map(self.toolbox.clone, offspring))
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.params.crossover_prob:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            if random.random() < self.params.mutation_prob:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Elitism: preserve best individuals
        if self.params.elite_size > 0:
            # Get elites from previous population
            elites = tools.selBest(self.population, self.params.elite_size)
            
            # Replace worst in offspring with elites
            offspring.sort(key=lambda x: x.fitness.values[0])
            offspring[-self.params.elite_size:] = elites
        
        # Update population
        self.population[:] = offspring
        
        # Update hall of fame
        self.halloffame.update(self.population)
    
    def optimize(self) -> Generator[Dict[str, Any], None, None]:
        """
        Run GA optimization.
        
        Yields:
            Dictionary with generation results
        """
        # Initialize
        self.initialize_population()
        
        # Record statistics
        record = self.stats.compile(self.population)
        self.logbook.record(gen=0, evals=len(self.population), **record)
        
        # Best individual
        best = self.halloffame[0]
        best_array = np.array(best, dtype=float)
        best_sections = self.evaluator.decode_variables(best_array)
        
        yield {
            'generation': 0,
            'best_solution': best_array,
            'best_fitness': best.fitness.values[0],
            'best_sections': best_sections,
            'n_evaluations': self.evaluator.n_evaluations,
            'stats': record
        }
        
        # Evolution
        for gen in range(1, self.params.max_generations):
            self.generation = gen
            self.evolve_generation()
            
            # Record statistics
            record = self.stats.compile(self.population)
            self.logbook.record(gen=gen, evals=len(self.population), **record)
            
            # Best individual
            best = self.halloffame[0]
            best_array = np.array(best, dtype=float)
            best_sections = self.evaluator.decode_variables(best_array)
            
            result = {
                'generation': gen,
                'best_solution': best_array,
                'best_fitness': best.fitness.values[0],
                'best_sections': best_sections,
                'n_evaluations': self.evaluator.n_evaluations,
                'stats': record
            }
            
            yield result
    
    def get_pareto_front(self) -> List[Any]:
        """
        Get Pareto-optimal solutions (for multi-objective, future extension).
        
        Returns:
            List of non-dominated individuals
        """
        # For single-objective, return hall of fame
        return list(self.halloffame)


class NSGA2Optimizer:
    """
    NSGA-II for multi-objective optimization (future extension).
    
    This is a placeholder for future multi-objective optimization:
    - Minimize weight
    - Minimize cost
    - Maximize reliability
    - etc.
    """
    
    def __init__(self):
        """Initialize NSGA-II."""
        raise NotImplementedError("NSGA-II not yet implemented")
    
    def optimize(self):
        """Run NSGA-II."""
        raise NotImplementedError("NSGA-II not yet implemented")
