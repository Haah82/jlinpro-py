"""
PyLinPro - Python Linear Analysis & Design Platform
Main Streamlit Application Entry Point
"""

import streamlit as st
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.structure import Structure

# Page configuration
st.set_page_config(
    page_title="PyLinPro",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'structure' not in st.session_state:
    st.session_state.structure = Structure()

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

if 'design_results' not in st.session_state:
    st.session_state.design_results = None

# Sidebar navigation
with st.sidebar:
    st.title("PyLinPro")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏗️ Modeling", "🔬 Analysis", "📊 Results", "✅ Design Check", "🧬 Optimization"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Project Info")
    st.info(f"Nodes: {len(st.session_state.structure.nodes)}")
    st.info(f"Elements: {len(st.session_state.structure.elements)}")
    st.info(f"Analysis: {'✅ Complete' if st.session_state.analysis_done else '⏳ Pending'}")

# Main content area
if page == "🏗️ Modeling":
    from pages import modeling
    modeling.render()
    
elif page == "🔬 Analysis":
    from pages import analysis
    analysis.render()
    
elif page == "📊 Results":
    from pages import results
    results.render()
    
elif page == "✅ Design Check":
    from pages import design
    design.render()
    
elif page == "🧬 Optimization":
    from pages import optimization
    optimization.render()
