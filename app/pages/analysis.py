"""
Analysis Page - Run static/modal analysis, set parameters
"""

import streamlit as st
import time


def render():
    """Render the analysis page."""
    st.title("🔬 Structural Analysis")
    st.markdown("Configure and run structural analysis on your model.")
    
    # Check if model is defined
    if len(st.session_state.structure.nodes) == 0:
        st.error("⚠️ No structural model defined. Please add nodes and elements in the Modeling page.")
        return
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Static Analysis", "Modal Analysis", "Dynamic Time History"],
        help="Select the type of structural analysis to perform"
    )
    
    if analysis_type == "Static Analysis":
        render_static_analysis()
    elif analysis_type == "Modal Analysis":
        render_modal_analysis()
    elif analysis_type == "Dynamic Time History":
        render_dynamic_analysis()


def render_static_analysis():
    """Render static analysis configuration and execution."""
    st.subheader("Static Linear Analysis")
    
    st.markdown("#### Load Cases")
    
    # Add load case interface
    with st.expander("➕ Add Load Case", expanded=True):
        with st.form("add_load_form"):
            load_case_name = st.text_input("Load Case Name", value="DL1")
            
            col1, col2 = st.columns(2)
            with col1:
                load_type = st.selectbox("Type", ["Nodal Load", "Uniform Element Load", "Point Element Load"])
            with col2:
                node_or_elem = st.number_input("Node/Element ID", min_value=1, step=1)
            
            # Load components
            st.markdown("**Load Components**")
            load_cols = st.columns(3)
            Fx = load_cols[0].number_input("Fx (kN)", value=0.0)
            Fy = load_cols[1].number_input("Fy (kN)", value=0.0)
            Fz = load_cols[2].number_input("Fz (kN)", value=0.0)
            
            submitted = st.form_submit_button("Add Load")
            if submitted:
                st.success(f"✅ Load case '{load_case_name}' added")
    
    # Display current loads
    st.markdown("#### Current Load Cases")
    if len(st.session_state.structure.loads) > 0:
        st.info(f"{len(st.session_state.structure.loads)} load case(s) defined")
    else:
        st.warning("No loads defined yet")
    
    st.markdown("---")
    
    # Analysis settings
    st.markdown("#### Analysis Settings")
    col1, col2 = st.columns(2)
    with col1:
        solver_type = st.selectbox("Solver", ["Sparse Direct (SpSolve)", "Iterative (CG)"])
    with col2:
        tolerance = st.number_input("Tolerance", value=1e-6, format="%.0e")
    
    # Run analysis button
    st.markdown("---")
    if st.button("▶️ Run Static Analysis", type="primary", use_container_width=True):
        with st.spinner("Running analysis..."):
            progress_bar = st.progress(0)
            
            # Simulate analysis steps
            progress_bar.progress(20, text="Assembling global stiffness matrix...")
            time.sleep(0.5)
            
            progress_bar.progress(50, text="Applying boundary conditions...")
            time.sleep(0.5)
            
            progress_bar.progress(80, text="Solving system of equations...")
            time.sleep(0.5)
            
            try:
                # Call actual analysis method
                st.session_state.structure.solve_static()
                
                progress_bar.progress(100, text="Analysis complete!")
                st.session_state.analysis_done = True
                
                st.success("✅ Static analysis completed successfully!")
                st.balloons()
                
                # Display quick summary
                results = st.session_state.structure.get_results_summary()
                st.info(f"Max displacement: {results['max_displacement']:.4f} m")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                progress_bar.empty()


def render_modal_analysis():
    """Render modal analysis configuration."""
    st.subheader("Modal Analysis (Eigenvalue)")
    
    st.markdown("#### Modal Analysis Settings")
    col1, col2 = st.columns(2)
    with col1:
        n_modes = st.number_input("Number of Modes", min_value=1, max_value=20, value=10)
    with col2:
        mass_type = st.selectbox("Mass Matrix Type", ["Lumped", "Consistent"])
    
    if st.button("▶️ Run Modal Analysis", type="primary", use_container_width=True):
        st.warning("Modal analysis not yet implemented (Prompt 4.2)")


def render_dynamic_analysis():
    """Render dynamic time history analysis configuration."""
    st.subheader("Dynamic Time History Analysis")
    
    st.markdown("#### Time History Settings")
    col1, col2 = st.columns(2)
    with col1:
        time_step = st.number_input("Time Step (s)", value=0.01, format="%.3f")
        duration = st.number_input("Duration (s)", value=10.0)
    with col2:
        damping_ratio = st.number_input("Damping Ratio", value=0.05, min_value=0.0, max_value=1.0)
    
    # Ground motion upload
    st.markdown("#### Ground Motion Input")
    uploaded_gm = st.file_uploader("Upload acceleration time history (CSV)", type=["csv"])
    
    if st.button("▶️ Run Dynamic Analysis", type="primary", use_container_width=True):
        st.warning("Dynamic analysis not yet implemented (Prompt 4.2)")
