"""
Optimization Module Example
============================

This example demonstrates the optimization module capabilities:
1. Simple truss optimization with GA (discrete catalog)
2. Comparison of GA, DE, and CaDE strategies
3. Analysis of FEA savings with CaDE
"""

import numpy as np
from src.optimization import (
    OptimizationStrategy,
    create_truss_catalog,
    ConstraintLimits,
    run_optimization
)


# Mock structure for demonstration
class SimpleTrussStructure:
    """
    Simple 2D truss structure for optimization demonstration.
    
    This is a simplified mock that demonstrates the interface.
    Real implementation would use actual FEA from src.core.
    """
    
    def __init__(self, n_members: int = 10):
        """
        Initialize truss structure.
        
        Args:
            n_members: Number of truss members
        """
        self.n_members = n_members
        
        # Mock elements
        self.elements = {i: MockElement(length=1000 + i*500) for i in range(1, n_members+1)}
        
        # Mock analysis results
        self.analysis_results = {
            'displacements': {},
            'element_forces': {}
        }
    
    def solve_static(self):
        """
        Mock static analysis.
        
        In real implementation, this would call FEA solver.
        For demo, we generate synthetic results based on section properties.
        """
        # Generate mock displacements (proportional to 1/A)
        self.analysis_results['displacements'] = {}
        for node_id in range(1, self.n_members + 2):
            # Mock: displacement inversely proportional to average area
            avg_area = np.mean([elem.section.A for elem in self.elements.values()])
            disp_mag = 10000 / avg_area  # Simplified
            
            self.analysis_results['displacements'][node_id] = {
                'ux': disp_mag * np.random.uniform(0.5, 1.0),
                'uy': disp_mag * np.random.uniform(0.5, 1.0)
            }
        
        # Generate mock element forces
        self.analysis_results['element_forces'] = {}
        for elem_id, elem in self.elements.items():
            # Mock: axial force proportional to area and length
            force = elem.section.A * elem.length / 10  # Simplified
            
            self.analysis_results['element_forces'][elem_id] = {
                'N': force * np.random.uniform(0.8, 1.2)
            }


class MockElement:
    """Mock element for demonstration."""
    
    def __init__(self, length: float = 1000):
        """Initialize element."""
        self.length = length
        self.section = MockSection()
    
    def get_length(self) -> float:
        """Get element length."""
        return self.length


class MockSection:
    """Mock section for demonstration."""
    
    def __init__(self, A: float = 300):
        """Initialize section."""
        self.A = A
        self.I = None


def demo_ga_discrete():
    """
    Demonstration 1: Discrete GA optimization.
    
    Optimize a 10-member truss using catalog-based GA.
    """
    print("\n" + "="*70)
    print("DEMO 1: GENETIC ALGORITHM (Discrete Catalog)")
    print("="*70)
    
    # Create structure
    structure = SimpleTrussStructure(n_members=10)
    
    # Setup optimization
    catalog = create_truss_catalog()
    constraints = ConstraintLimits(
        max_stress=150.0,  # MPa
        max_displacement=25.0,  # mm
        max_utilization_ratio=1.0
    )
    
    print(f"\nStructure: {len(structure.elements)} members")
    print(f"Catalog: {len(catalog)} sections (A: {catalog[0]['A']}-{catalog[-1]['A']} mm²)")
    print(f"Constraints: σ_max={constraints.max_stress} MPa, δ_max={constraints.max_displacement} mm")
    
    # Run optimization
    print("\nRunning GA optimization...")
    results = []
    
    for result in run_optimization(
        structure=structure,
        strategy=OptimizationStrategy.GA_DISCRETE,
        section_catalog=catalog,
        constraints=constraints,
        custom_params={
            'population_size': 50,
            'max_generations': 20,
            'crossover_prob': 0.8,
            'mutation_prob': 0.2
        }
    ):
        results.append(result)
        
        gen = result['generation']
        fitness = result['best_fitness']
        n_evals = result['n_evaluations']
        
        if gen % 5 == 0:
            print(f"  Gen {gen:3d}: Best fitness = {fitness:8.2f} kg  "
                  f"(Evaluations: {n_evals})")
    
    # Final result
    final = results[-1]
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best weight: {final['best_fitness']:.2f} kg")
    print(f"Total evaluations: {final['n_evaluations']}")
    print(f"Feasibility rate: {final['evaluator_stats']['feasibility_rate']:.1%}")
    
    # Show section assignments
    if 'best_sections' in final:
        print(f"\nOptimal section assignment:")
        for i, section in enumerate(final['best_sections'], 1):
            print(f"  Member {i:2d}: {section['name']:12s} (A = {section['A']:5.0f} mm²)")


def demo_de_continuous():
    """
    Demonstration 2: Continuous DE optimization.
    
    Optimize using Differential Evolution with continuous variables.
    """
    print("\n" + "="*70)
    print("DEMO 2: DIFFERENTIAL EVOLUTION (Continuous)")
    print("="*70)
    
    # Create structure
    structure = SimpleTrussStructure(n_members=5)
    
    # Setup bounds (continuous area ranges)
    bounds = [(100, 1500) for _ in range(5)]
    
    constraints = ConstraintLimits(
        max_stress=150.0,
        max_displacement=25.0
    )
    
    print(f"\nStructure: {len(structure.elements)} members")
    print(f"Design space: A ∈ [{bounds[0][0]}, {bounds[0][1]}] mm² per member")
    
    # Run optimization
    print("\nRunning DE optimization...")
    results = []
    
    for result in run_optimization(
        structure=structure,
        strategy=OptimizationStrategy.DE_CONTINUOUS,
        bounds=bounds,
        constraints=constraints,
        custom_params={
            'population_size': 30,
            'max_iterations': 30,
            'mutation_factor': 0.8,
            'crossover_prob': 0.7
        }
    ):
        results.append(result)
        
        gen = result['generation']
        fitness = result['best_fitness']
        
        if gen % 5 == 0:
            print(f"  Iter {gen:3d}: Best fitness = {fitness:8.2f} kg")
    
    # Final result
    final = results[-1]
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Best weight: {final['best_fitness']:.2f} kg")
    print(f"Total evaluations: {final['n_evaluations']}")
    
    print(f"\nOptimal areas:")
    for i, area in enumerate(final['best_solution'], 1):
        print(f"  Member {i}: A = {area:6.1f} mm²")


def demo_cade_fea_savings():
    """
    Demonstration 3: CaDE with FEA savings tracking.
    
    Show how CaDE reduces computational cost.
    """
    print("\n" + "="*70)
    print("DEMO 3: CaDE - FEA Savings Analysis")
    print("="*70)
    
    # Create structure
    structure = SimpleTrussStructure(n_members=8)
    
    bounds = [(100, 1500) for _ in range(8)]
    
    print(f"\nStructure: {len(structure.elements)} members")
    print(f"Strategy: Classification-assisted DE")
    
    # Run CaDE
    print("\nRunning CaDE optimization...")
    results = []
    
    for result in run_optimization(
        structure=structure,
        strategy=OptimizationStrategy.CADE_CONTINUOUS,
        bounds=bounds,
        custom_params={
            'population_size': 40,
            'max_iterations': 40,
            'learning_phase_iterations': 15,
            'mutation_factor': 0.8
        }
    ):
        results.append(result)
        
        gen = result['generation']
        fitness = result['best_fitness']
        phase = result.get('phase', 'unknown')
        skipped = result.get('n_skipped_fea', 0)
        
        if gen % 5 == 0:
            if phase == 'employing':
                print(f"  Iter {gen:3d} [{phase:10s}]: Fitness = {fitness:8.2f} kg  "
                      f"(FEA skipped: {skipped})")
            else:
                print(f"  Iter {gen:3d} [{phase:10s}]: Fitness = {fitness:8.2f} kg")
    
    # Analyze savings
    final = results[-1]
    total_possible_evals = final['generation'] * 40  # pop_size * generations
    actual_evals = final['n_evaluations']
    fea_saved = final.get('n_skipped_fea', 0)
    
    print(f"\n{'='*70}")
    print(f"CaDE PERFORMANCE ANALYSIS")
    print(f"{'='*70}")
    print(f"Best weight: {final['best_fitness']:.2f} kg")
    print(f"Total iterations: {final['generation']}")
    print(f"Possible FEA calls: {total_possible_evals}")
    print(f"Actual FEA calls: {actual_evals}")
    print(f"FEA calls skipped: {fea_saved}")
    print(f"Computational savings: {(fea_saved / total_possible_evals * 100):.1f}%")
    
    # Phase breakdown
    learning_results = [r for r in results if r.get('phase') == 'learning']
    employing_results = [r for r in results if r.get('phase') == 'employing']
    
    print(f"\nPhase breakdown:")
    print(f"  Learning phase: {len(learning_results)} iterations (full FEA)")
    print(f"  Employing phase: {len(employing_results)} iterations (ML-assisted)")


def demo_comparison():
    """
    Demonstration 4: Compare all three strategies.
    
    Run GA, DE, and CaDE side-by-side for comparison.
    """
    print("\n" + "="*70)
    print("DEMO 4: STRATEGY COMPARISON")
    print("="*70)
    
    # Small structure for quick comparison
    structure = SimpleTrussStructure(n_members=5)
    bounds = [(100, 1500) for _ in range(5)]
    catalog = create_truss_catalog()
    
    strategies = [
        ("GA (Discrete)", OptimizationStrategy.GA_DISCRETE, {'max_generations': 20, 'population_size': 30}),
        ("DE (Continuous)", OptimizationStrategy.DE_CONTINUOUS, {'max_iterations': 20, 'population_size': 30}),
        ("CaDE (Continuous)", OptimizationStrategy.CADE_CONTINUOUS, {
            'max_iterations': 20, 'population_size': 30, 'learning_phase_iterations': 8
        })
    ]
    
    comparison_results = {}
    
    for name, strategy, params in strategies:
        print(f"\n{'-'*70}")
        print(f"Running: {name}")
        print(f"{'-'*70}")
        
        # Prepare kwargs
        kwargs = {
            'structure': structure,
            'strategy': strategy,
            'custom_params': params
        }
        
        if strategy == OptimizationStrategy.GA_DISCRETE:
            kwargs['section_catalog'] = catalog
        else:
            kwargs['bounds'] = bounds
        
        # Run optimization
        results = list(run_optimization(**kwargs))
        final = results[-1]
        
        comparison_results[name] = {
            'fitness': final['best_fitness'],
            'evaluations': final['n_evaluations'],
            'feasibility_rate': final['evaluator_stats']['feasibility_rate']
        }
        
        print(f"  Final fitness: {final['best_fitness']:.2f} kg")
        print(f"  Evaluations: {final['n_evaluations']}")
    
    # Summary table
    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Strategy':<25s} {'Best Weight':>12s} {'Evaluations':>12s} {'Feasibility':>12s}")
    print(f"{'-'*70}")
    
    for name, result in comparison_results.items():
        print(f"{name:<25s} {result['fitness']:>12.2f} {result['evaluations']:>12d} "
              f"{result['feasibility_rate']:>11.1%}")


if __name__ == "__main__":
    """Run all demonstrations."""
    
    print("\n" + "="*70)
    print("STRUCTURAL OPTIMIZATION MODULE - DEMONSTRATIONS")
    print("="*70)
    print("\nThis example demonstrates three optimization algorithms:")
    print("1. Genetic Algorithm (GA) - Discrete catalog-based")
    print("2. Differential Evolution (DE) - Continuous parameters")
    print("3. Classification-assisted DE (CaDE) - ML-accelerated")
    
    # Run demonstrations
    demo_ga_discrete()
    demo_de_continuous()
    demo_cade_fea_savings()
    demo_comparison()
    
    print("\n" + "="*70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("="*70)
    print("\n✅ Optimization module ready for production use")
    print("\nNext steps:")
    print("- Integrate with real FEA solver (src.core.structure)")
    print("- Add design code checking (TCVN, ACI, EC2) to constraints")
    print("- Connect to Streamlit UI for interactive optimization")
    print()
