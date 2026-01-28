"""
Design Check Page - Code checking interface (TCVN, ACI, EC2)
"""

import streamlit as st
import pandas as pd


def render():
    """Render the design check page."""
    st.title("✅ Design Code Checking")
    st.markdown("Verify structural members against design standards.")
    
    # Check if analysis has been run
    if not st.session_state.analysis_done:
        st.warning("⚠️ Please run structural analysis first before performing design checks.")
        return
    
    # Design code selection
    st.markdown("### Design Standard")
    design_code = st.selectbox(
        "Select Design Code",
        ["TCVN 5574:2018 (Vietnam)", "ACI 318-25 (USA)", "Eurocode 2 (Europe)"],
        help="Choose the design standard for member checking"
    )
    
    # Design tabs based on member type
    if "TCVN" in design_code:
        render_tcvn_design()
    elif "ACI" in design_code:
        render_aci_design()
    elif "Eurocode" in design_code:
        render_ec2_design()


def render_tcvn_design():
    """Render TCVN 5574:2018 design interface."""
    st.markdown("### TCVN 5574:2018 Design Checks")
    
    # Material selection
    st.markdown("#### Material Properties")
    col1, col2 = st.columns(2)
    
    with col1:
        concrete_grade = st.selectbox(
            "Concrete Grade",
            ["B15", "B20", "B25", "B30", "B35", "B40", "B45", "B50", "B60"]
        )
    
    with col2:
        steel_grade = st.selectbox(
            "Steel Grade",
            ["CB240T", "CB300V", "CB400V", "CB500V"]
        )
    
    # Store in session state for access by child functions
    st.session_state['concrete_grade'] = concrete_grade
    st.session_state['steel_grade'] = steel_grade
    
    # Design check tabs
    tab1, tab2, tab3 = st.tabs(["🔷 Beam Design", "🔶 Column Design", "📏 Serviceability"])
    
    with tab1:
        render_beam_design_tcvn()
    
    with tab2:
        render_column_design_tcvn()
    
    with tab3:
        render_sls_design_tcvn()


def render_beam_design_tcvn():
    """Render TCVN beam design interface."""
    st.subheader("Beam Flexure & Shear Design")
    
    # Beam selection
    beam_id = st.selectbox("Select Beam Element", options=list(range(1, 11)))
    
    st.markdown("#### Cross-Section Details")
    col1, col2 = st.columns(2)
    with col1:
        b = st.number_input("Width b (mm)", value=300, step=50)
        h = st.number_input("Height h (mm)", value=500, step=50)
    with col2:
        cover = st.number_input("Concrete Cover a (mm)", value=25, step=5)
        h0 = h - cover
        st.info(f"Effective depth h₀ = {h0} mm")
    
    st.markdown("#### Reinforcement")
    col1, col2 = st.columns(2)
    with col1:
        As_top = st.number_input("Top Steel As' (mm²)", value=1000, step=100)
    with col2:
        As_bot = st.number_input("Bottom Steel As (mm²)", value=1500, step=100)
    
    # Stirrups
    st.markdown("#### Stirrups")
    col1, col2, col3 = st.columns(3)
    with col1:
        stirrup_dia = st.number_input("Diameter (mm)", value=10, step=2)
    with col2:
        n_legs = st.number_input("No. of Legs", value=2, min_value=2, step=2)
    with col3:
        spacing = st.number_input("Spacing s (mm)", value=200, step=50)
    
    Asw = n_legs * 3.14159 * (stirrup_dia/2)**2
    st.info(f"Asw = {Asw:.1f} mm²")
    
    # Run check
    if st.button("▶️ Run TCVN Beam Check", type="primary"):
        # Get forces from analysis results (placeholder - will be replaced when analysis is implemented)
        # For now, use example values
        M_u = 250.0  # kN·m - TODO: get from st.session_state.structure.get_element_force(beam_id, 'M')
        Q_u = 120.0  # kN - TODO: get from st.session_state.structure.get_element_force(beam_id, 'V')
        
        with st.spinner("Performing design checks..."):
            try:
                from design.tcvn_beam import run_beam_check_from_ui
                
                results = run_beam_check_from_ui(
                    beam_id, b, h, cover,
                    As_top, As_bot,
                    stirrup_dia, n_legs, spacing,
                    concrete_grade, steel_grade,
                    M_u, Q_u
                )
                
                # Display results
                flex = results['flexure']
                shear = results['shear']
                
                st.markdown("---")
                st.markdown("### Design Check Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Flexure")
                    if flex['status'] == 'PASS':
                        st.success(f"✅ PASS (UR = {flex['UR']:.2f})")
                    else:
                        st.error(f"❌ FAIL (UR = {flex['UR']:.2f})")
                
                with col2:
                    st.markdown("#### Shear")
                    if shear['status'] == 'PASS':
                        st.success(f"✅ PASS (UR = {shear['UR']:.2f})")
                    else:
                        st.error(f"❌ FAIL (UR = {shear['UR']:.2f})")
                
                # Summary table
                st.markdown("#### Check Summary")
                
                results_data = {
                    "Check Type": ["Flexure", "Shear"],
                    "Demand": [f"{M_u:.2f} kN·m", f"{Q_u:.2f} kN"],
                    "Capacity": [
                        f"{flex['M_cap']:.2f} kN·m" if flex['M_cap'] else "N/A",
                        f"{shear['Q_cap']:.2f} kN"
                    ],
                    "UR": [flex['UR'], shear['UR']],
                    "Status": [
                        "✅ PASS" if flex['status'] == 'PASS' else "❌ FAIL",
                        "✅ PASS" if shear['status'] == 'PASS' else "❌ FAIL"
                    ]
                }
                
                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)
                
                # Detailed calculations
                with st.expander("📋 Flexure Calculation Details"):
                    st.text(flex['details'])
                
                with st.expander("📋 Shear Calculation Details"):
                    st.text(shear['details'])
                
            except Exception as e:
                st.error(f"Error during design check: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def render_column_design_tcvn():
    """Render TCVN column design interface."""
    st.subheader("Column Compression & Biaxial Bending")
    
    # Get material grades from parent scope
    # Access via session state set by parent function
    concrete_grade = st.session_state.get('concrete_grade', 'B25')
    steel_grade = st.session_state.get('steel_grade', 'CB400V')
    
    # Column selection
    col_id = st.selectbox("Select Column Element", options=list(range(1, 11)))
    
    st.markdown("#### Cross-Section")
    col1, col2 = st.columns(2)
    with col1:
        b = st.number_input("Width b (mm)", value=400, step=50, key="col_b")
        h = st.number_input("Height h (mm)", value=400, step=50, key="col_h")
        cover = st.number_input("Cover (mm)", value=30, step=5, key="col_cover")
    with col2:
        L_eff = st.number_input("Effective Length (m)", value=3.0, step=0.5)
        h0 = h - cover
        st.info(f"Effective depth h₀ = {h0} mm")
    
    st.markdown("#### Reinforcement Layout")
    total_As = st.number_input("Total Steel As (mm²)", value=2400, step=100,
                                help="Total longitudinal reinforcement area (all bars)")
    
    # Run check
    if st.button("▶️ Run TCVN Column Check", type="primary"):
        with st.spinner("Generating interaction diagram..."):
            try:
                # Placeholder forces (will come from analysis)
                N_u = 800  # kN (compression)
                M_u = 120  # kN·m
                
                from design.tcvn_column import run_column_check_from_ui
                
                result = run_column_check_from_ui(
                    col_id=col_id,
                    b=b, h=h, L_eff=L_eff, cover=cover,
                    As_total=total_As,
                    concrete_name=concrete_grade,
                    steel_name=steel_grade,
                    N_u=N_u,
                    M_u=M_u
                )
                
                st.markdown("### Design Check Results")
                
                # Status
                if result['status'] == 'PASS':
                    st.success(f"✅ Column check: PASS (UR = {result['UR']:.2f})")
                else:
                    st.error(f"❌ Column check: FAIL (UR = {result['UR']:.2f})")
                
                # Key parameters
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Slenderness λ", f"{result['lambda']:.1f}")
                with col2:
                    st.metric("Magnification η", f"{result['eta']:.3f}")
                with col3:
                    st.metric("Amplified M", f"{result['M_amplified']:.1f} kN·m")
                
                # Interaction diagram
                st.markdown("#### Interaction Diagram")
                if result['figure']:
                    st.plotly_chart(result['figure'], use_container_width=True)
                else:
                    st.error("Could not generate interaction diagram")
                
                # Summary table
                st.markdown("#### Capacity Summary")
                summary_data = {
                    "Parameter": ["Axial Force", "Moment (amplified)", "Moment Capacity"],
                    "Value": [
                        f"{N_u:.1f} kN",
                        f"{result['M_amplified']:.1f} kN·m",
                        f"{result.get('M_cap_at_Nu', 0):.1f} kN·m"
                    ],
                    "Status": [
                        "Applied",
                        "Applied",
                        f"UR = {result['UR']:.2f}"
                    ]
                }
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)
                
                # Calculation details
                with st.expander("📋 Calculation Details"):
                    st.text(result['details'])
                
            except Exception as e:
                st.error(f"Error during column design check: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def render_sls_design_tcvn():
    """Render TCVN serviceability checks."""
    st.subheader("Serviceability Limit State Checks")
    
    # Get material grades from session state
    concrete_grade = st.session_state.get('concrete_grade', 'B25')
    steel_grade = st.session_state.get('steel_grade', 'CB400V')
    
    # Beam selection
    beam_id = st.selectbox("Select Beam Element", options=list(range(1, 11)), key="sls_beam_id")
    
    st.markdown("#### Section & Reinforcement")
    col1, col2, col3 = st.columns(3)
    with col1:
        b = st.number_input("Width b (mm)", value=300, step=50, key="sls_b")
        h = st.number_input("Height h (mm)", value=500, step=50, key="sls_h")
    with col2:
        cover = st.number_input("Cover (mm)", value=25, step=5, key="sls_cover")
        As = st.number_input("Tension Steel As (mm²)", value=1500, step=100, key="sls_As")
    with col3:
        bar_dia = st.number_input("Bar Diameter φ (mm)", value=16, step=2, key="sls_bar_dia")
        L_span = st.number_input("Span Length (m)", value=6.0, step=0.5, key="sls_span")
    
    st.markdown("#### Loading & Environment")
    col1, col2, col3 = st.columns(3)
    with col1:
        M_ser = st.number_input("Service Moment M_ser (kN·m)", value=120.0, step=10.0,
                                help="Unfactored service load moment")
    with col2:
        environment = st.selectbox("Exposure Condition", 
                                   options=['normal', 'aggressive'],
                                   help="Affects allowable crack width")
    with col3:
        load_duration = st.selectbox("Load Duration",
                                     options=['long_term', 'short_term'],
                                     help="Affects deflection via creep factor")
    
    # Run SLS checks
    if st.button("▶️ Run TCVN Serviceability Checks", type="primary"):
        with st.spinner("Performing serviceability checks..."):
            try:
                from design.tcvn_sls import run_sls_check_from_ui
                
                results = run_sls_check_from_ui(
                    beam_id=beam_id,
                    b=b, h=h, cover=cover,
                    As=As,
                    bar_diameter=bar_dia,
                    concrete_name=concrete_grade,
                    steel_name=steel_grade,
                    M_ser=M_ser,
                    L=L_span,
                    environment=environment,
                    load_duration=load_duration
                )
                
                st.markdown("### Serviceability Check Results")
                
                # Crack width results
                crack = results['crack_width']
                deflect = results['deflection']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔍 Crack Width")
                    if crack['status'] == 'PASS':
                        st.success(f"✅ PASS (UR = {crack['UR']:.2f})")
                    else:
                        st.error(f"❌ FAIL (UR = {crack['UR']:.2f})")
                    
                    st.metric("Calculated Width", f"{crack['a_cr']:.3f} mm")
                    st.metric("Allowable Width", f"{crack['a_limit']:.1f} mm")
                    st.metric("Steel Stress", f"{crack['sigma_s']:.1f} MPa")
                
                with col2:
                    st.markdown("#### 📏 Deflection")
                    if deflect['status'] == 'PASS':
                        st.success(f"✅ PASS (UR = {deflect['UR']:.2f})")
                    else:
                        st.error(f"❌ FAIL (UR = {deflect['UR']:.2f})")
                    
                    st.metric("Calculated Deflection", f"{deflect['delta']:.2f} mm")
                    st.metric("Allowable (L/250)", f"{deflect['delta_limit']:.2f} mm")
                    st.metric("Cracking Moment", f"{deflect['M_cr']:.2f} kN·m")
                
                # Summary table
                st.markdown("#### Check Summary")
                summary_data = {
                    "Check Type": ["Crack Width", "Deflection"],
                    "Calculated": [
                        f"{crack['a_cr']:.3f} mm",
                        f"{deflect['delta']:.2f} mm"
                    ],
                    "Allowable": [
                        f"{crack['a_limit']:.1f} mm",
                        f"{deflect['delta_limit']:.2f} mm"
                    ],
                    "UR": [crack['UR'], deflect['UR']],
                    "Status": [
                        "✅ PASS" if crack['status'] == 'PASS' else "❌ FAIL",
                        "✅ PASS" if deflect['status'] == 'PASS' else "❌ FAIL"
                    ]
                }
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)
                
                # Detailed calculations
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("📋 Crack Width Details"):
                        st.text(crack['details'])
                
                with col2:
                    with st.expander("📋 Deflection Details"):
                        st.text(deflect['details'])
                
            except Exception as e:
                st.error(f"Error during serviceability checks: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def render_aci_design():
    """Render ACI 318-25 design interface."""
    st.markdown("### ACI 318-25 Design Checks")
    
    # Material selection
    st.markdown("#### Material Properties")
    col1, col2 = st.columns(2)
    
    with col1:
        concrete_grade = st.selectbox(
            "Concrete Grade",
            ["3000 psi (20.7 MPa)", "4000 psi (27.6 MPa)", "5000 psi (34.5 MPa)", 
             "6000 psi (41.4 MPa)", "8000 psi (55.2 MPa)", "10000 psi (69.0 MPa)"],
            key="aci_concrete"
        )
    
    with col2:
        steel_grade = st.selectbox(
            "Steel Grade",
            ["Grade 60 (414 MPa)", "Grade 75 (517 MPa)", "Grade 80 (552 MPa)"],
            key="aci_steel"
        )
    
    # Member type tabs
    tab1, tab2, tab3 = st.tabs(["Beam Flexure", "Beam Shear", "Column"])
    
    with tab1:
        render_beam_flexure_aci(concrete_grade, steel_grade)
    
    with tab2:
        render_beam_shear_aci(concrete_grade, steel_grade)
    
    with tab3:
        render_column_aci(concrete_grade, steel_grade)


def render_beam_flexure_aci(concrete_name: str, steel_name: str):
    """Render ACI beam flexure check interface."""
    st.markdown("#### Beam Flexural Capacity Check")
    
    from src.design.aci318 import run_beam_flexure_check_aci_from_ui
    
    col1, col2 = st.columns(2)
    
    with col1:
        beam_id = st.number_input("Beam ID", min_value=1, value=1, key="aci_flex_beam_id")
        b = st.number_input("Width b (mm)", min_value=100, value=300, step=50, key="aci_flex_b")
        h = st.number_input("Height h (mm)", min_value=200, value=500, step=50, key="aci_flex_h")
        cover = st.number_input("Cover (mm)", min_value=20, value=40, step=5, key="aci_flex_cover")
    
    with col2:
        As = st.number_input("Tension Steel As (mm²)", min_value=100, value=1600, step=100, key="aci_flex_As")
        As_compression = st.number_input("Compression Steel As' (mm²)", min_value=0, value=0, step=100, key="aci_flex_As_comp")
        Mu = st.number_input("Ultimate Moment Mu (kN·m)", min_value=0.0, value=120.0, step=10.0, key="aci_flex_Mu")
    
    if st.button("Check Flexure", key="aci_flex_check"):
        with st.spinner("Performing ACI 318-25 flexure check..."):
            result = run_beam_flexure_check_aci_from_ui(
                beam_id=int(beam_id),
                b=b, h=h, cover=cover,
                As=As,
                As_compression=As_compression,
                concrete_name=concrete_name,
                steel_name=steel_name,
                Mu=Mu
            )
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Nominal Mn", f"{result['Mn']:.2f} kN·m")
            col2.metric("Design φMn", f"{result['phiMn']:.2f} kN·m")
            col3.metric("φ Factor", f"{result['phi']:.3f}")
            col4.metric("UR", f"{result['UR']:.3f}", 
                       delta="PASS" if result['status'] == 'PASS' else "FAIL",
                       delta_color="normal" if result['status'] == 'PASS' else "inverse")
            
            # Summary
            if result['status'] == 'PASS':
                st.success(f"✅ Beam {beam_id} PASSES flexure check (UR = {result['UR']:.3f})")
            else:
                st.error(f"❌ Beam {beam_id} FAILS flexure check (UR = {result['UR']:.3f})")
            
            # Detailed calculation
            with st.expander("View Calculation Details"):
                st.code(result['details'], language='text')


def render_beam_shear_aci(concrete_name: str, steel_name: str):
    """Render ACI beam shear check interface."""
    st.markdown("#### Beam Shear Capacity Check")
    
    from src.design.aci318 import run_beam_shear_check_aci_from_ui
    
    col1, col2 = st.columns(2)
    
    with col1:
        beam_id = st.number_input("Beam ID", min_value=1, value=1, key="aci_shear_beam_id")
        b = st.number_input("Width bw (mm)", min_value=100, value=300, step=50, key="aci_shear_b")
        h = st.number_input("Height h (mm)", min_value=200, value=500, step=50, key="aci_shear_h")
        cover = st.number_input("Cover (mm)", min_value=20, value=40, step=5, key="aci_shear_cover")
    
    with col2:
        Vu = st.number_input("Ultimate Shear Vu (kN)", min_value=0.0, value=80.0, step=10.0, key="aci_shear_Vu")
        stirrup_dia = st.number_input("Stirrup Diameter φ (mm)", min_value=0, value=10, step=2, key="aci_shear_dia")
        n_legs = st.number_input("Number of Legs", min_value=0, value=2, step=1, key="aci_shear_legs")
        spacing = st.number_input("Spacing s (mm)", min_value=0, value=150, step=25, key="aci_shear_spacing")
    
    if st.button("Check Shear", key="aci_shear_check"):
        with st.spinner("Performing ACI 318-25 shear check..."):
            result = run_beam_shear_check_aci_from_ui(
                beam_id=int(beam_id),
                b=b, h=h, cover=cover,
                Vu=Vu,
                concrete_name=concrete_name,
                steel_name=steel_name,
                stirrup_dia=stirrup_dia if stirrup_dia > 0 else 0,
                n_legs=int(n_legs) if n_legs > 0 else 0,
                spacing=spacing if spacing > 0 else 0
            )
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Concrete Vc", f"{result['Vc']:.2f} kN")
            col2.metric("Stirrup Vs", f"{result['Vs']:.2f} kN")
            col3.metric("Design φVn", f"{result['phiVn']:.2f} kN")
            col4.metric("UR", f"{result['UR']:.3f}",
                       delta="PASS" if result['status'] == 'PASS' else "FAIL",
                       delta_color="normal" if result['status'] == 'PASS' else "inverse")
            
            # Summary
            if result['status'] == 'PASS':
                st.success(f"✅ Beam {beam_id} PASSES shear check (UR = {result['UR']:.3f})")
            else:
                st.error(f"❌ Beam {beam_id} FAILS shear check (UR = {result['UR']:.3f})")
            
            # Detailed calculation
            with st.expander("View Calculation Details"):
                st.code(result['details'], language='text')


def render_column_aci(concrete_name: str, steel_name: str):
    """Render ACI column interaction diagram interface."""
    st.markdown("#### Column Axial-Moment Interaction")
    
    from src.design.aci318 import run_column_check_aci_from_ui
    
    col1, col2 = st.columns(2)
    
    with col1:
        col_id = st.number_input("Column ID", min_value=1, value=1, key="aci_col_id")
        b = st.number_input("Width b (mm)", min_value=200, value=400, step=50, key="aci_col_b")
        h = st.number_input("Height h (mm)", min_value=200, value=400, step=50, key="aci_col_h")
        cover = st.number_input("Cover (mm)", min_value=20, value=40, step=5, key="aci_col_cover")
        As_total = st.number_input("Total Steel As (mm²)", min_value=500, value=3200, step=100, key="aci_col_As")
    
    with col2:
        Pu = st.number_input("Ultimate Axial Force Pu (kN)", value=500.0, step=50.0, key="aci_col_Pu",
                            help="Positive = compression, Negative = tension")
        Mu = st.number_input("Ultimate Moment Mu (kN·m)", min_value=0.0, value=100.0, step=10.0, key="aci_col_Mu")
        column_type = st.selectbox("Column Type", ["tied", "spiral"], key="aci_col_type")
    
    if st.button("Check Column", key="aci_col_check"):
        with st.spinner("Generating ACI 318-25 interaction diagram..."):
            result = run_column_check_aci_from_ui(
                col_id=int(col_id),
                b=b, h=h, cover=cover,
                As_total=As_total,
                concrete_name=concrete_name,
                steel_name=steel_name,
                Pu=Pu, Mu=Mu,
                column_type=column_type
            )
            
            # Display interaction diagram
            st.plotly_chart(result['figure'], use_container_width=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("φ Factor", f"{result['phi']:.3f}")
            col2.metric("UR", f"{result['UR']:.3f}")
            col3.metric("Status", result['status'],
                       delta="PASS" if result['status'] == 'PASS' else "FAIL",
                       delta_color="normal" if result['status'] == 'PASS' else "inverse")
            
            # Summary
            if result['status'] == 'PASS':
                st.success(f"✅ Column {col_id} is within capacity envelope")
            else:
                st.error(f"❌ Column {col_id} exceeds capacity envelope")
            
            # Detailed calculation
            with st.expander("View Calculation Details"):
                st.code(result['details'], language='text')


def render_ec2_design():
    """Render Eurocode 2 design interface."""
    st.markdown("### Eurocode 2 Design Checks")
    st.info("Eurocode 2 implementation pending: Prompt 3.6")

