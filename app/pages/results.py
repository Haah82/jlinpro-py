"""
Results Page - Display tables, diagrams, deformed shape
"""

import streamlit as st
import pandas as pd
import numpy as np


def render():
    """Render the results page."""
    st.title("📊 Analysis Results")
    st.markdown("View and export structural analysis results.")
    
    # Check if analysis has been run
    if not st.session_state.analysis_done:
        st.warning("⚠️ No analysis results available. Please run analysis first in the Analysis page.")
        return
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Displacements",
        "⚡ Reactions",
        "📐 Element Forces",
        "🎨 Visualizations"
    ])
    
    with tab1:
        render_displacements()
    
    with tab2:
        render_reactions()
    
    with tab3:
        render_element_forces()
    
    with tab4:
        render_visualizations()


def render_displacements():
    """Display nodal displacements table."""
    st.subheader("Nodal Displacements")
    
    # Get results from structure
    results = st.session_state.structure.get_results_summary()
    
    if 'displacements' in results:
        df = results['displacements']
        
        # Format numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_styled = df.style.format({col: "{:.4e}" for col in numeric_cols})
        
        st.dataframe(df_styled, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="displacements.csv",
            mime="text/csv"
        )
        
        # Summary statistics
        st.markdown("#### Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        if 'UX' in df.columns:
            col1.metric("Max UX", f"{df['UX'].abs().max():.4e} m")
        if 'UY' in df.columns:
            col2.metric("Max UY", f"{df['UY'].abs().max():.4e} m")
        if 'UZ' in df.columns:
            col3.metric("Max UZ", f"{df['UZ'].abs().max():.4e} m")
    
    else:
        st.info("No displacement data available")


def render_reactions():
    """Display support reactions table."""
    st.subheader("Support Reactions")
    
    results = st.session_state.structure.get_results_summary()
    
    if 'reactions' in results:
        df = results['reactions']
        
        # Only show nodes with non-zero reactions
        df_nonzero = df[(df.abs() > 1e-10).any(axis=1)]
        
        numeric_cols = df_nonzero.select_dtypes(include=[np.number]).columns
        df_styled = df_nonzero.style.format({col: "{:.4e}" for col in numeric_cols})
        
        st.dataframe(df_styled, use_container_width=True)
        
        # Download button
        csv = df_nonzero.to_csv(index=True)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="reactions.csv",
            mime="text/csv"
        )
    else:
        st.info("No reaction data available")


def render_element_forces():
    """Display element internal forces table."""
    st.subheader("Element Internal Forces")
    
    results = st.session_state.structure.get_results_summary()
    
    if 'element_forces' in results:
        df = results['element_forces']
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            force_type = st.multiselect(
                "Force Components",
                options=df.columns.tolist(),
                default=df.columns.tolist()
            )
        
        with col2:
            selected_elements = st.multiselect(
                "Elements",
                options=df.index.tolist(),
                default=df.index.tolist()[:10] if len(df) > 10 else df.index.tolist()
            )
        
        # Filter dataframe
        df_filtered = df.loc[selected_elements, force_type]
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        df_styled = df_filtered.style.format({col: "{:.4e}" for col in numeric_cols})
        
        st.dataframe(df_styled, use_container_width=True)
        
        # Download button
        csv = df_filtered.to_csv(index=True)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="element_forces.csv",
            mime="text/csv"
        )
    else:
        st.info("No element force data available")


def render_visualizations():
    """Display visual results (plots, diagrams)."""
    st.subheader("Result Visualizations")
    
    viz_type = st.selectbox(
        "Visualization Type",
        ["Deformed Shape", "Force Diagram", "Stress Contour"]
    )
    
    if viz_type == "Deformed Shape":
        st.markdown("#### Deformed Shape")
        
        scale = st.slider("Deformation Scale", 1, 100, 10)
        overlay = st.checkbox("Overlay Original", value=True)
        
        st.info("Deformed shape plot will be rendered here (Prompt 2.2)")
        
    elif viz_type == "Force Diagram":
        st.markdown("#### Internal Force Diagram")
        
        force_type = st.selectbox("Force Type", ["Axial (N)", "Shear (V)", "Moment (M)"])
        
        st.info("Force diagram will be rendered here (Prompt 2.2)")
        
    elif viz_type == "Stress Contour":
        st.markdown("#### Stress Contour")
        st.info("Stress contour plot will be rendered here (Prompt 2.2)")
