"""
Unit Tests for Optimization Module
===================================

Tests for evaluator, GA, DE, and CaDE implementations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from src.optimization import (
    StructureEvaluator,
    OptimizationProblem,
    OptimizationMode,
    ConstraintLimits,
    create_truss_catalog,
    create_beam_catalog,
    OptimizationStrategy,
    run_optimization,
    GeneticAlgorithm,
    GAParameters,
    DifferentialEvolution,
    ClassificationAssistedDE,
    DEParameters,
    CaDEParameters
)


class TestEvaluator:
    """Test cases for StructureEvaluator."""
    
    def test_discrete_problem_creation(self):
        """Test creating discrete optimization problem."""
        catalog = create_truss_catalog()
        bounds = [(0, len(catalog) - 1) for _ in range(5)]
        
        problem = OptimizationProblem(
            mode=OptimizationMode.DISCRETE,
            n_variables=5,
            bounds=bounds,
            constraints=ConstraintLimits(),
            section_catalog=catalog
        )
        
        assert problem.mode == OptimizationMode.DISCRETE
        assert problem.n_variables == 5
        assert len(problem.section_catalog) == len(catalog)
    
    def test_continuous_problem_creation(self):
        """Test creating continuous optimization problem."""
        bounds = [(100, 1500) for _ in range(5)]
        
        problem = OptimizationProblem(
            mode=OptimizationMode.CONTINUOUS,
            n_variables=5,
            bounds=bounds,
            constraints=ConstraintLimits()
        )
        
        assert problem.mode == OptimizationMode.CONTINUOUS
        assert problem.n_variables == 5
        assert problem.section_catalog is None
    
    def test_constraint_limits_validation(self):
        """Test constraint limits validation."""
        # Valid constraints
        constraints = ConstraintLimits(
            max_stress=200.0,
            max_displacement=30.0,
            max_utilization_ratio=1.0
        )
        assert constraints.max_stress == 200.0
        
        # Invalid stress
        with pytest.raises(ValueError):
            ConstraintLimits(max_stress=-10)
        
        # Invalid displacement
        with pytest.raises(ValueError):
            ConstraintLimits(max_displacement=0)
    
    def test_decode_variables_discrete(self):
        """Test decoding discrete variables."""
        catalog = create_truss_catalog()
        problem = OptimizationProblem(
            mode=OptimizationMode.DISCRETE,
            n_variables=3,
            bounds=[(0, len(catalog)-1)] * 3,
            constraints=ConstraintLimits(),
            section_catalog=catalog
        )
        
        # Mock structure
        structure = Mock()
        structure.elements = {1: Mock(), 2: Mock(), 3: Mock()}
        
        evaluator = StructureEvaluator(problem, structure)
        
        # Test decoding
        individual = np.array([0, 5, 9])  # Indices
        sections = evaluator.decode_variables(individual)
        
        assert len(sections) == 3
        assert sections[0]['A'] == catalog[0]['A']
        assert sections[1]['A'] == catalog[5]['A']
        assert sections[2]['A'] == catalog[9]['A']
    
    def test_decode_variables_continuous(self):
        """Test decoding continuous variables."""
        problem = OptimizationProblem(
            mode=OptimizationMode.CONTINUOUS,
            n_variables=3,
            bounds=[(100, 1500)] * 3,
            constraints=ConstraintLimits()
        )
        
        structure = Mock()
        structure.elements = {1: Mock(), 2: Mock(), 3: Mock()}
        
        evaluator = StructureEvaluator(problem, structure)
        
        # Test decoding
        individual = np.array([300, 600, 1200])
        sections = evaluator.decode_variables(individual)
        
        assert len(sections) == 3
        assert sections[0]['A'] == 300
        assert sections[1]['A'] == 600
        assert sections[2]['A'] == 1200
    
    def test_calculate_weight(self):
        """Test weight calculation."""
        catalog = create_truss_catalog()
        problem = OptimizationProblem(
            mode=OptimizationMode.DISCRETE,
            n_variables=2,
            bounds=[(0, len(catalog)-1)] * 2,
            constraints=ConstraintLimits(),
            section_catalog=catalog,
            material_density=7850.0  # kg/m³
        )
        
        # Mock structure
        structure = Mock()
        elem1 = Mock()
        elem1.get_length.return_value = 1000  # mm
        elem2 = Mock()
        elem2.get_length.return_value = 2000  # mm
        structure.elements = {1: elem1, 2: elem2}
        
        evaluator = StructureEvaluator(problem, structure)
        
        # Sections: A1=300mm², A2=500mm²
        sections = [{'A': 300, 'I': None}, {'A': 500, 'I': None}]
        
        # Weight = ρ * (A1*L1 + A2*L2)
        # = 7850 * (300*1000 + 500*2000) / 1e9
        # = 7850 * 1300000 / 1e9
        # = 10.205 kg
        weight = evaluator.calculate_weight(sections)
        
        expected = 7850 * (300*1000 + 500*2000) / 1e9
        assert abs(weight - expected) < 0.01
    
    def test_evaluator_statistics(self):
        """Test evaluation statistics tracking."""
        problem = OptimizationProblem(
            mode=OptimizationMode.CONTINUOUS,
            n_variables=1,
            bounds=[(100, 1000)],
            constraints=ConstraintLimits()
        )
        
        structure = Mock()
        structure.elements = {1: Mock()}
        
        evaluator = StructureEvaluator(problem, structure)
        
        # Initial statistics
        stats = evaluator.get_statistics()
        assert stats['total_evaluations'] == 0
        
        # Reset and check
        evaluator.reset_statistics()
        assert evaluator.n_evaluations == 0


class TestGAParameters:
    """Test GA parameter validation."""
    
    def test_valid_parameters(self):
        """Test valid GA parameters."""
        params = GAParameters(
            population_size=50,
            max_generations=100,
            crossover_prob=0.8,
            mutation_prob=0.2
        )
        
        assert params.population_size == 50
        assert params.max_generations == 100
    
    def test_invalid_population_size(self):
        """Test invalid population size."""
        with pytest.raises(ValueError):
            GAParameters(population_size=2)
    
    def test_invalid_probabilities(self):
        """Test invalid probabilities."""
        with pytest.raises(ValueError):
            GAParameters(crossover_prob=1.5)
        
        with pytest.raises(ValueError):
            GAParameters(mutation_prob=-0.1)


class TestDEParameters:
    """Test DE parameter validation."""
    
    def test_valid_parameters(self):
        """Test valid DE parameters."""
        params = DEParameters(
            mutation_factor=0.8,
            crossover_prob=0.7,
            population_size=50
        )
        
        assert params.mutation_factor == 0.8
        assert params.crossover_prob == 0.7
    
    def test_invalid_mutation_factor(self):
        """Test invalid mutation factor."""
        with pytest.raises(ValueError):
            DEParameters(mutation_factor=0)
        
        with pytest.raises(ValueError):
            DEParameters(mutation_factor=3.0)
    
    def test_invalid_crossover_prob(self):
        """Test invalid crossover probability."""
        with pytest.raises(ValueError):
            DEParameters(crossover_prob=1.5)


class TestCaDEParameters:
    """Test CaDE parameter validation."""
    
    def test_valid_parameters(self):
        """Test valid CaDE parameters."""
        params = CaDEParameters(
            mutation_factor=0.8,
            population_size=50,
            max_iterations=100,
            learning_phase_iterations=20
        )
        
        assert params.learning_phase_iterations == 20
        assert params.max_iterations == 100
    
    def test_invalid_learning_phase(self):
        """Test invalid learning phase iterations."""
        with pytest.raises(ValueError):
            CaDEParameters(
                max_iterations=50,
                learning_phase_iterations=60
            )


class TestCatalogCreation:
    """Test catalog creation functions."""
    
    def test_create_truss_catalog(self):
        """Test truss catalog creation."""
        catalog = create_truss_catalog()
        
        assert len(catalog) > 0
        assert all('A' in section for section in catalog)
        assert all('name' in section for section in catalog)
        
        # Check ascending order
        areas = [section['A'] for section in catalog]
        assert areas == sorted(areas)
    
    def test_create_beam_catalog(self):
        """Test beam catalog creation."""
        catalog = create_beam_catalog()
        
        assert len(catalog) > 0
        assert all('A' in section for section in catalog)
        assert all('I' in section for section in catalog)
        assert all('name' in section for section in catalog)
        
        # Check ascending order by area
        areas = [section['A'] for section in catalog]
        assert areas == sorted(areas)


class TestDEAlgorithm:
    """Test Differential Evolution algorithm."""
    
    def test_initialization(self):
        """Test DE initialization."""
        bounds = [(100, 1000), (100, 1000), (100, 1000)]
        
        # Mock evaluator
        problem = OptimizationProblem(
            mode=OptimizationMode.CONTINUOUS,
            n_variables=3,
            bounds=bounds,
            constraints=ConstraintLimits()
        )
        
        structure = Mock()
        structure.elements = {1: Mock(), 2: Mock(), 3: Mock()}
        
        evaluator = StructureEvaluator(problem, structure)
        
        params = DEParameters(population_size=10, max_iterations=5)
        de = DifferentialEvolution(evaluator, bounds, params)
        
        assert de.n_vars == 3
        assert de.params.population_size == 10
    
    def test_mutation_clipping(self):
        """Test that mutant vectors are clipped to [0, 1]."""
        bounds = [(0, 100)] * 3
        
        problem = OptimizationProblem(
            mode=OptimizationMode.CONTINUOUS,
            n_variables=3,
            bounds=bounds,
            constraints=ConstraintLimits()
        )
        
        structure = Mock()
        structure.elements = {1: Mock(), 2: Mock(), 3: Mock()}
        
        evaluator = StructureEvaluator(problem, structure)
        
        de = DifferentialEvolution(evaluator, bounds)
        de.initialize_population()
        
        # Generate mutants
        for i in range(de.params.population_size):
            mutant = de.mutate(i)
            
            # Check bounds
            assert np.all(mutant >= 0)
            assert np.all(mutant <= 1)


class TestCaDEAlgorithm:
    """Test Classification-assisted DE algorithm."""
    
    def test_label_assignment(self):
        """Test feasibility label assignment."""
        # Feasible (no violation)
        label = ClassificationAssistedDE.assign_label(0.0)
        assert label == 1
        
        # Infeasible (violation > 0)
        label = ClassificationAssistedDE.assign_label(10.0)
        assert label == -1
    
    def test_initialization(self):
        """Test CaDE initialization."""
        bounds = [(100, 1000)] * 3
        
        problem = OptimizationProblem(
            mode=OptimizationMode.CONTINUOUS,
            n_variables=3,
            bounds=bounds,
            constraints=ConstraintLimits()
        )
        
        structure = Mock()
        structure.elements = {1: Mock(), 2: Mock(), 3: Mock()}
        
        evaluator = StructureEvaluator(problem, structure)
        
        params = CaDEParameters(
            population_size=10,
            max_iterations=30,
            learning_phase_iterations=10
        )
        
        cade = ClassificationAssistedDE(evaluator, bounds, params)
        
        assert cade.n_vars == 3
        assert cade.params.learning_phase_iterations == 10


class TestOptimizationStrategy:
    """Test OptimizationStrategy enum."""
    
    def test_strategy_values(self):
        """Test strategy enum values."""
        assert OptimizationStrategy.GA_DISCRETE.value == "ga_discrete"
        assert OptimizationStrategy.DE_CONTINUOUS.value == "de_continuous"
        assert OptimizationStrategy.CADE_CONTINUOUS.value == "cade_continuous"


# Integration test marker
@pytest.mark.integration
class TestOptimizationIntegration:
    """Integration tests (require full structure implementation)."""
    
    @pytest.mark.skip(reason="Requires full Structure implementation")
    def test_ga_optimization_run(self):
        """Test running GA optimization on simple structure."""
        # This would require a fully implemented Structure class
        pass
    
    @pytest.mark.skip(reason="Requires full Structure implementation")
    def test_de_optimization_run(self):
        """Test running DE optimization on simple structure."""
        pass
    
    @pytest.mark.skip(reason="Requires full Structure implementation")
    def test_cade_fea_savings(self):
        """Test that CaDE actually saves FEA evaluations."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
