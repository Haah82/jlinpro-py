"""
Quick Start Guide: Optimization Module
=======================================

This guide shows how to use the optimization module for structural design.
"""

# Example 1: Basic catalog-based optimization (10 lines)
from src.optimization import quick_optimize_discrete, create_truss_catalog

# Assume you have a structure object
# structure = Structure()  # Your FEA model

result = quick_optimize_discrete(
    structure=structure,
    catalog=create_truss_catalog(),
    max_generations=50
)
print(f"Optimized weight: {result['best_fitness']:.2f} kg")


# Example 2: Continuous optimization with bounds (12 lines)
from src.optimization import quick_optimize_continuous

bounds = [(100, 1500)] * 10  # 10 members, area range 100-1500 mm²

result = quick_optimize_continuous(
    structure=structure,
    bounds=bounds,
    max_iterations=50,
    use_cade=True  # 30% faster with ML acceleration
)
print(f"Optimal areas: {result['best_solution']}")


# Example 3: Full control with run_optimization (20 lines)
from src.optimization import (
    run_optimization,
    OptimizationStrategy,
    ConstraintLimits,
    create_beam_catalog
)

constraints = ConstraintLimits(
    max_stress=150.0,      # MPa
    max_displacement=25.0, # mm
    max_utilization_ratio=1.0
)

for result in run_optimization(
    structure=structure,
    strategy=OptimizationStrategy.CADE_CONTINUOUS,
    bounds=bounds,
    constraints=constraints,
    custom_params={'max_iterations': 100, 'learning_phase_iterations': 30}
):
    print(f"Gen {result['generation']}: Fitness = {result['best_fitness']:.2f}")
    
    # Check for early stopping
    if result['best_fitness'] < target_weight:
        break


# Example 4: Compare all strategies (8 lines)
from src.optimization import compare_strategies

results = compare_strategies(
    structure=structure,
    bounds=bounds,
    catalog=create_beam_catalog(),
    max_iterations=50
)
# Prints comparison table automatically


# Example 5: Custom evaluation with design code (25 lines)
from src.optimization import (
    StructureEvaluator,
    OptimizationProblem,
    OptimizationMode,
    GeneticAlgorithm,
    GAParameters
)
from src.design import Eurocode2Code

# Define problem
problem = OptimizationProblem(
    mode=OptimizationMode.DISCRETE,
    n_variables=len(structure.elements),
    bounds=[(0, 9)] * len(structure.elements),
    constraints=ConstraintLimits(),
    section_catalog=create_beam_catalog()
)

# Create evaluator with design code
evaluator = StructureEvaluator(
    problem=problem,
    structure=structure,
    design_code=Eurocode2Code()
)

# Run GA
ga = GeneticAlgorithm(evaluator, n_catalog=10, params=GAParameters())
for result in ga.optimize():
    if result['generation'] % 10 == 0:
        print(f"Gen {result['generation']}: Best = {result['best_fitness']:.2f} kg")


# Example 6: Progress tracking with Streamlit (15 lines)
import streamlit as st
from src.optimization import run_optimization, OptimizationStrategy

st.title("Structural Optimization")

strategy = st.selectbox("Strategy", ["GA", "DE", "CaDE"])
max_gen = st.slider("Generations", 10, 200, 50)

if st.button("Optimize"):
    progress_bar = st.progress(0)
    metric_col1, metric_col2 = st.columns(2)
    
    for result in run_optimization(structure, OptimizationStrategy.GA_DISCRETE):
        progress_bar.progress(result['generation'] / max_gen)
        metric_col1.metric("Best Fitness", f"{result['best_fitness']:.2f} kg")
        metric_col2.metric("Evaluations", result['n_evaluations'])


print("""
Summary of Optimization Strategies:
====================================

1. GA (Discrete): Best for catalog selection problems
   - Pros: Handles discrete choices, proven reliability
   - Cons: Premature convergence risk
   - Use when: You have a catalog of standard sections

2. DE (Continuous): Best for continuous parameter optimization
   - Pros: Robust, smooth convergence, fewer parameters
   - Cons: Slower than GA for discrete problems
   - Use when: Optimizing dimensions (areas, thicknesses)

3. CaDE (Continuous): Best for expensive FEA problems
   - Pros: 30% faster than DE, ML-accelerated
   - Cons: Requires learning phase
   - Use when: FEA is slow, large search space

Quick Decision Tree:
-------------------
Catalog sections? → Use GA_DISCRETE
Continuous vars & slow FEA? → Use CADE_CONTINUOUS
Continuous vars & fast FEA? → Use DE_CONTINUOUS

Performance Tips:
----------------
- Start with small population/generations for testing
- GA: Use elitism (elite_size=2-5) to preserve best
- DE: mutation_factor=0.8, crossover_prob=0.7 are robust
- CaDE: learning_phase should be 20-30% of total iterations
- Monitor feasibility_rate: >50% means constraints are reasonable

""")
