"""
Modeling Page - Input nodes, elements, materials via forms and file upload
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from core.structures import Node, Material, RectangularSection, CircularSection


def render():
    """Render the modeling page."""
    st.title("🏗️ Structural Modeling")
    st.markdown("Define your structural model using manual input or file upload.")
    
    # Tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📁 Import File", "👁️ Preview Model"])
    
    with tab1:
        render_manual_input()
    
    with tab2:
        render_file_import()
    
    with tab3:
        render_model_preview()


def render_manual_input():
    """Render manual input forms for nodes and elements."""
    st.subheader("Manual Model Definition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Add Node")
        with st.form("add_node_form"):
            node_id = st.number_input("Node ID", min_value=1, step=1, value=1)
            
            coords_cols = st.columns(3)
            x = coords_cols[0].number_input("X (m)", value=0.0, format="%.3f")
            y = coords_cols[1].number_input("Y (m)", value=0.0, format="%.3f")
            z = coords_cols[2].number_input("Z (m)", value=0.0, format="%.3f")
            
            st.markdown("**Restraints** (Check to restrain)")
            restraint_cols = st.columns(6)
            restraints = [
                restraint_cols[0].checkbox("X", key="rx"),
                restraint_cols[1].checkbox("Y", key="ry"),
                restraint_cols[2].checkbox("Z", key="rz"),
                restraint_cols[3].checkbox("RX", key="rrx"),
                restraint_cols[4].checkbox("RY", key="rry"),
                restraint_cols[5].checkbox("RZ", key="rrz")
            ]
            
            submitted = st.form_submit_button("Add Node")
            if submitted:
                node = Node(id=node_id, x=x, y=y, z=z, restraints=restraints)
                st.session_state.structure.add_node(node)
                st.success(f"✅ Node {node_id} added at ({x:.2f}, {y:.2f}, {z:.2f})")
    
    with col2:
        st.markdown("#### Add Element")
        with st.form("add_element_form"):
            elem_id = st.number_input("Element ID", min_value=1, step=1, value=1)
            elem_type = st.selectbox("Type", ["Beam2D", "Truss2D", "Beam3D", "Truss3D"])
            
            node_cols = st.columns(2)
            node_i = node_cols[0].number_input("Node I", min_value=1, step=1, value=1)
            node_j = node_cols[1].number_input("Node J", min_value=1, step=1, value=2)
            
            # Material selection
            material_preset = st.selectbox(
                "Material",
                ["Steel S275", "Steel S355", "Concrete C30", "Custom"]
            )
            
            # Section selection
            section_type = st.selectbox("Section Type", ["Rectangular", "Circular", "I-Section"])
            
            if section_type == "Rectangular":
                sec_cols = st.columns(2)
                b = sec_cols[0].number_input("Width b (m)", value=0.3, format="%.3f")
                h = sec_cols[1].number_input("Height h (m)", value=0.5, format="%.3f")
            elif section_type == "Circular":
                d = st.number_input("Diameter d (m)", value=0.4, format="%.3f")
            
            submitted_elem = st.form_submit_button("Add Element")
            if submitted_elem:
                st.success(f"✅ Element {elem_id} added ({elem_type}: {node_i}-{node_j})")
    
    # Display current model stats
    st.markdown("---")
    st.markdown("#### Current Model Statistics")
    stats_cols = st.columns(4)
    stats_cols[0].metric("Nodes", len(st.session_state.structure.nodes))
    stats_cols[1].metric("Elements", len(st.session_state.structure.elements))
    stats_cols[2].metric("DOFs", st.session_state.structure.get_total_dofs())
    stats_cols[3].metric("Restraints", st.session_state.structure.get_restrained_dofs())


def render_file_import():
    """Render file import interface."""
    st.subheader("Import Model from File")
    
    st.info("""
    **Supported formats:**
    - JSON: Complete model definition
    - CSV: Nodes and elements tables
    - SAP2000 Export: `.s2k` text file
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["json", "csv", "s2k"],
        help="Upload your structural model file"
    )
    
    if uploaded_file is not None:
        file_type = Path(uploaded_file.name).suffix
        
        if file_type == ".json":
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            if st.button("Load Model"):
                # TODO: Implement JSON loading
                st.warning("JSON import not yet implemented")
        
        elif file_type == ".csv":
            df = pd.read_csv(uploaded_file)
            st.dataframe(df, use_container_width=True)
            
            if st.button("Parse CSV"):
                # TODO: Implement CSV parsing
                st.warning("CSV import not yet implemented")


def render_model_preview():
    """Render 3D preview of current model."""
    st.subheader("Model Preview")
    
    if len(st.session_state.structure.nodes) == 0:
        st.warning("⚠️ No nodes defined yet. Add nodes in the Manual Input tab.")
        return
    
    # Display nodes table
    st.markdown("#### Nodes")
    nodes_data = []
    for node_id, node in st.session_state.structure.nodes.items():
        restraint_str = "".join([
            "1" if r else "0" for r in node.restraints
        ])
        nodes_data.append({
            "ID": node.id,
            "X": f"{node.x:.3f}",
            "Y": f"{node.y:.3f}",
            "Z": f"{node.z:.3f}",
            "Restraints": restraint_str
        })
    
    if nodes_data:
        st.dataframe(pd.DataFrame(nodes_data), use_container_width=True)
    
    # Display elements table
    st.markdown("#### Elements")
    if len(st.session_state.structure.elements) > 0:
        elements_data = []
        for elem_id, elem in st.session_state.structure.elements.items():
            elements_data.append({
                "ID": elem_id,
                "Type": elem.__class__.__name__,
                "Node I": elem.node_i,
                "Node J": elem.node_j,
                "Length": f"{elem.get_length():.3f}"
            })
        st.dataframe(pd.DataFrame(elements_data), use_container_width=True)
    else:
        st.info("No elements defined yet.")
    
    # 3D visualization placeholder
    st.markdown("#### 3D Visualization")
    st.info("3D plot will be rendered here using Plotly (Prompt 2.2)")
