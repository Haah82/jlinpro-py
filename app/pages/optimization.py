"""
Optimization Page - GA parameter setup and execution
"""

import streamlit as st
import pandas as pd
import numpy as np
import time


def render():
    """Render the optimization page."""
    st.title("🧬 Genetic Algorithm Optimization")
    st.markdown("Optimize structural design using evolutionary algorithms.")
    
    # Check prerequisites
    if not st.session_state.analysis_done:
        st.warning("⚠️ Please run structural analysis first before optimization.")
        return
    
    st.markdown("### Optimization Problem Definition")
    
    # Objective function
    st.markdown("#### Objective")
    objective = st.selectbox(
        "Minimize",
        ["Total Weight", "Total Cost", "Max Displacement", "Multi-objective"]
    )
    
    # Design variables
    st.markdown("#### Design Variables")
    st.markdown("**Section Catalog**")
    
    # Upload section catalog
    uploaded_catalog = st.file_uploader(
        "Upload Section Catalog (CSV)",
        type=["csv"],
        help="CSV file with columns: Name, b, h, A, Ix, Iy, Cost"
    )
    
    if uploaded_catalog is not None:
        catalog_df = pd.read_csv(uploaded_catalog)
        st.dataframe(catalog_df, use_container_width=True)
        st.success(f"✅ Loaded {len(catalog_df)} sections")
    else:
        # Default catalog
        default_catalog = pd.DataFrame({
            "Name": ["100x100", "150x150", "200x200", "250x250", "300x300"],
            "b (mm)": [100, 150, 200, 250, 300],
            "h (mm)": [100, 150, 200, 250, 300],
            "A (mm²)": [10000, 22500, 40000, 62500, 90000]
        })
        st.dataframe(default_catalog, use_container_width=True)
    
    # Constraints
    st.markdown("#### Constraints")
    col1, col2 = st.columns(2)
    
    with col1:
        max_stress = st.number_input("Max Stress (MPa)", value=250.0, step=10.0)
        max_displacement = st.number_input("Max Displacement (mm)", value=50.0, step=5.0)
    
    with col2:
        code_compliance = st.checkbox("Enforce Code Compliance", value=True)
        design_code = st.selectbox("Design Code", ["TCVN 5574:2018", "ACI 318-25"])
    
    # GA parameters
    st.markdown("---")
    st.markdown("### Genetic Algorithm Parameters")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pop_size = st.number_input("Population Size", min_value=10, max_value=200, value=50, step=10)
    
    with col2:
        n_generations = st.number_input("Generations", min_value=10, max_value=500, value=100, step=10)
    
    with col3:
        mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.2, step=0.05)
    
    with col4:
        crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8, step=0.05)
    
    # Run optimization
    st.markdown("---")
    if st.button("▶️ Start Optimization", type="primary", use_container_width=True):
        run_optimization(pop_size, n_generations, mutation_rate, crossover_rate)


def run_optimization(pop_size, n_generations, mutation_rate, crossover_rate):
    """Execute genetic algorithm optimization."""
    
    # Create containers for real-time updates
    status_container = st.empty()
    progress_container = st.empty()
    chart_container = st.empty()
    
    # Initialize
    best_fitness_history = []
    avg_fitness_history = []
    
    # Simulate GA iterations
    for gen in range(n_generations):
        # Simulate fitness calculation
        best_fitness = 1000 * np.exp(-gen / 50) + np.random.randn() * 10
        avg_fitness = best_fitness + 50 + np.random.randn() * 20
        
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        # Update status
        status_container.info(f"Generation {gen+1}/{n_generations} - Best Fitness: {best_fitness:.2f}")
        
        # Update progress bar
        progress = (gen + 1) / n_generations
        progress_container.progress(progress)
        
        # Update chart
        if (gen + 1) % 5 == 0 or gen == 0:
            df_history = pd.DataFrame({
                "Generation": range(1, len(best_fitness_history) + 1),
                "Best": best_fitness_history,
                "Average": avg_fitness_history
            })
            
            chart_container.line_chart(
                df_history.set_index("Generation"),
                use_container_width=True
            )
        
        # Simulate computation time
        time.sleep(0.05)
    
    # Final results
    st.success("✅ Optimization completed!")
    st.balloons()
    
    # Display results
    st.markdown("---")
    st.markdown("### Optimization Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Weight", "1250 kg")
    col2.metric("Optimized Weight", "987 kg", delta="-21%", delta_color="inverse")
    col3.metric("Iterations", n_generations)
    
    # Optimal design
    st.markdown("#### Optimal Section Assignment")
    
    optimal_design = pd.DataFrame({
        "Element": [1, 2, 3, 4, 5],
        "Original Section": ["200x200", "200x200", "200x200", "200x200", "200x200"],
        "Optimized Section": ["150x150", "200x200", "150x150", "200x200", "150x150"],
        "UR": [0.82, 0.95, 0.78, 0.91, 0.85],
        "Status": ["✅ OK", "✅ OK", "✅ OK", "✅ OK", "✅ OK"]
    })
    
    st.dataframe(optimal_design, use_container_width=True)
    
    # Download results
    csv = optimal_design.to_csv(index=False)
    st.download_button(
        label="📥 Download Optimal Design",
        data=csv,
        file_name="optimal_design.csv",
        mime="text/csv"
    )
    
    st.info("Full GA implementation pending: Prompt 3.6")
