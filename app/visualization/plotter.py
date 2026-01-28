"""
Interactive 3D Visualization using Plotly
Functions for plotting structures, deformed shapes, force diagrams, and mode shapes
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple


def plot_structure_3d(structure, show_labels: bool = True) -> go.Figure:
    """
    Plot 3D structural model with nodes and elements.
    
    Args:
        structure: Structure object containing nodes and elements
        show_labels: If True, display node IDs as text labels
    
    Returns:
        plotly Figure object
    
    Example:
        >>> fig = plot_structure_3d(structure, show_labels=True)
        >>> st.plotly_chart(fig, use_container_width=True)
    """
    fig = go.Figure()
    
    # Extract node coordinates
    node_ids = []
    x_coords = []
    y_coords = []
    z_coords = []
    marker_colors = []
    marker_symbols = []
    
    for node_id, node in structure.nodes.items():
        node_ids.append(str(node_id))
        x_coords.append(node.x)
        y_coords.append(node.y)
        z_coords.append(node.z)
        
        # Color code based on restraint type
        restraint_count = sum(node.restraints)
        if restraint_count == 6:  # Fixed
            marker_colors.append('red')
            marker_symbols.append('square')
        elif restraint_count >= 3:  # Pinned
            marker_colors.append('orange')
            marker_symbols.append('diamond')
        else:  # Free or partially restrained
            marker_colors.append('blue')
            marker_symbols.append('circle')
    
    # Plot nodes
    node_trace = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers+text' if show_labels else 'markers',
        marker=dict(
            size=8,
            color=marker_colors,
            symbol=marker_symbols,
            line=dict(color='black', width=1)
        ),
        text=node_ids if show_labels else None,
        textposition='top center',
        name='Nodes',
        hovertemplate='<b>Node %{text}</b><br>' +
                      'X: %{x:.3f}<br>' +
                      'Y: %{y:.3f}<br>' +
                      'Z: %{z:.3f}<br>' +
                      '<extra></extra>'
    )
    fig.add_trace(node_trace)
    
    # Plot elements
    for elem_id, element in structure.elements.items():
        node_i = structure.nodes[element.node_i]
        node_j = structure.nodes[element.node_j]
        
        # Color by element type
        elem_type = element.__class__.__name__
        if 'Beam' in elem_type:
            color = 'rgb(100, 100, 100)'
        elif 'Truss' in elem_type:
            color = 'rgb(50, 150, 200)'
        else:
            color = 'rgb(150, 150, 150)'
        
        element_trace = go.Scatter3d(
            x=[node_i.x, node_j.x],
            y=[node_i.y, node_j.y],
            z=[node_i.z, node_j.z],
            mode='lines',
            line=dict(color=color, width=4),
            name=f'{elem_type} {elem_id}',
            showlegend=False,
            hovertemplate=f'<b>Element {elem_id}</b><br>' +
                         f'Type: {elem_type}<br>' +
                         f'Nodes: {element.node_i}-{element.node_j}<br>' +
                         '<extra></extra>'
        )
        fig.add_trace(element_trace)
    
    # Layout settings
    fig.update_layout(
        title='Structural Model',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        showlegend=True,
        hovermode='closest',
        width=900,
        height=700
    )
    
    return fig


def plot_deformed_shape(structure, scale: float = 1.0, overlay_original: bool = True) -> go.Figure:
    """
    Plot deformed structure overlaid on original structure.
    
    Args:
        structure: Structure object with analysis results
        scale: Magnification factor for displacements
        overlay_original: If True, show original structure as semi-transparent
    
    Returns:
        plotly Figure object
    """
    fig = go.Figure()
    
    # Check if analysis results exist
    if not hasattr(structure, 'displacements') or structure.displacements is None:
        # Return empty figure with message
        fig.add_annotation(
            text="No analysis results available. Please run analysis first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get displacement data
    U = structure.displacements
    
    # Calculate displacement magnitudes for coloring
    disp_magnitudes = []
    deformed_coords = {}
    
    for node_id, node in structure.nodes.items():
        dof_indices = node.get_dofs()
        ux = U[dof_indices[0]] if len(dof_indices) > 0 else 0
        uy = U[dof_indices[1]] if len(dof_indices) > 1 else 0
        uz = U[dof_indices[2]] if len(dof_indices) > 2 else 0
        
        magnitude = np.sqrt(ux**2 + uy**2 + uz**2)
        disp_magnitudes.append(magnitude)
        
        deformed_coords[node_id] = {
            'x': node.x + scale * ux,
            'y': node.y + scale * uy,
            'z': node.z + scale * uz
        }
    
    max_disp = max(disp_magnitudes) if disp_magnitudes else 1.0
    
    # Plot original structure (semi-transparent)
    if overlay_original:
        for elem_id, element in structure.elements.items():
            node_i = structure.nodes[element.node_i]
            node_j = structure.nodes[element.node_j]
            
            fig.add_trace(go.Scatter3d(
                x=[node_i.x, node_j.x],
                y=[node_i.y, node_j.y],
                z=[node_i.z, node_j.z],
                mode='lines',
                line=dict(color='rgba(200, 200, 200, 0.3)', width=2, dash='dash'),
                name='Original',
                showlegend=(elem_id == list(structure.elements.keys())[0]),
                hoverinfo='skip'
            ))
    
    # Plot deformed structure with color gradient
    for elem_id, element in structure.elements.items():
        node_i_def = deformed_coords[element.node_i]
        node_j_def = deformed_coords[element.node_j]
        
        # Calculate average displacement for this element
        idx_i = list(structure.nodes.keys()).index(element.node_i)
        idx_j = list(structure.nodes.keys()).index(element.node_j)
        avg_disp = (disp_magnitudes[idx_i] + disp_magnitudes[idx_j]) / 2
        
        # Normalize color (0 to 1)
        color_val = avg_disp / max_disp if max_disp > 0 else 0
        
        fig.add_trace(go.Scatter3d(
            x=[node_i_def['x'], node_j_def['x']],
            y=[node_i_def['y'], node_j_def['y']],
            z=[node_i_def['z'], node_j_def['z']],
            mode='lines',
            line=dict(
                color=color_val,
                colorscale='Jet',
                width=6,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title="Displacement (m)",
                    tickvals=[0, 0.5, 1],
                    ticktext=[f"{0:.3e}", f"{max_disp/2:.3e}", f"{max_disp:.3e}"]
                )
            ),
            name='Deformed',
            showlegend=(elem_id == list(structure.elements.keys())[0]),
            hovertemplate=f'<b>Element {elem_id}</b><br>' +
                         f'Avg Disp: {avg_disp:.3e} m<br>' +
                         '<extra></extra>'
        ))
    
    # Plot deformed nodes
    x_def = [deformed_coords[nid]['x'] for nid in structure.nodes.keys()]
    y_def = [deformed_coords[nid]['y'] for nid in structure.nodes.keys()]
    z_def = [deformed_coords[nid]['z'] for nid in structure.nodes.keys()]
    
    fig.add_trace(go.Scatter3d(
        x=x_def,
        y=y_def,
        z=z_def,
        mode='markers',
        marker=dict(size=6, color='red'),
        name='Deformed Nodes',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Layout
    fig.update_layout(
        title=f'Deformed Shape (Scale: {scale}x)',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        width=900,
        height=700
    )
    
    return fig


def plot_internal_forces(structure, force_type: str = 'M', scale: float = 1.0) -> go.Figure:
    """
    Plot internal force diagrams on the structure.
    
    Args:
        structure: Structure object with analysis results
        force_type: 'N' (axial), 'V' (shear), 'M' (moment)
        scale: Scale factor for force diagram
    
    Returns:
        plotly Figure object
    """
    fig = go.Figure()
    
    # Check if analysis results exist
    if not hasattr(structure, 'element_forces') or structure.element_forces is None:
        fig.add_annotation(
            text="No force results available. Please run analysis first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    force_data = structure.element_forces
    force_label = {'N': 'Axial Force', 'V': 'Shear Force', 'M': 'Moment'}[force_type]
    
    # Plot structure outline
    for elem_id, element in structure.elements.items():
        node_i = structure.nodes[element.node_i]
        node_j = structure.nodes[element.node_j]
        
        fig.add_trace(go.Scatter3d(
            x=[node_i.x, node_j.x],
            y=[node_i.y, node_j.y],
            z=[node_i.z, node_j.z],
            mode='lines',
            line=dict(color='gray', width=3),
            name='Structure',
            showlegend=(elem_id == list(structure.elements.keys())[0]),
            hoverinfo='skip'
        ))
    
    # Extract force values and find max for normalization
    force_values = []
    for elem_id in structure.elements.keys():
        if elem_id in force_data and force_type in force_data[elem_id]:
            force_values.append(abs(force_data[elem_id][force_type]))
    
    max_force = max(force_values) if force_values else 1.0
    
    # Plot force diagrams
    for elem_id, element in structure.elements.items():
        if elem_id not in force_data or force_type not in force_data[elem_id]:
            continue
        
        force_val = force_data[elem_id][force_type]
        color_val = abs(force_val) / max_force if max_force > 0 else 0
        
        # Calculate perpendicular offset for force diagram
        node_i = structure.nodes[element.node_i]
        node_j = structure.nodes[element.node_j]
        
        # Element vector
        dx = node_j.x - node_i.x
        dy = node_j.y - node_i.y
        dz = node_j.z - node_i.z
        L = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Perpendicular vector (simplified for 2D/3D)
        if abs(dz) < 1e-6:  # 2D case
            perp_x = -dy / L
            perp_y = dx / L
            perp_z = 0
        else:  # 3D case
            perp_x = -dy
            perp_y = dx
            perp_z = 0
            perp_L = np.sqrt(perp_x**2 + perp_y**2)
            if perp_L > 1e-6:
                perp_x /= perp_L
                perp_y /= perp_L
        
        # Scale offset by force magnitude
        offset = scale * force_val / max_force if max_force > 0 else 0
        
        # Draw force diagram
        x_points = [
            node_i.x,
            node_i.x + offset * perp_x,
            node_j.x + offset * perp_x,
            node_j.x,
            node_i.x
        ]
        y_points = [
            node_i.y,
            node_i.y + offset * perp_y,
            node_j.y + offset * perp_y,
            node_j.y,
            node_i.y
        ]
        z_points = [
            node_i.z,
            node_i.z + offset * perp_z,
            node_j.z + offset * perp_z,
            node_j.z,
            node_i.z
        ]
        
        fig.add_trace(go.Scatter3d(
            x=x_points,
            y=y_points,
            z=z_points,
            mode='lines',
            fill='toself',
            line=dict(color=color_val, colorscale='RdBu', width=2),
            name=f'Elem {elem_id}',
            showlegend=False,
            hovertemplate=f'<b>Element {elem_id}</b><br>' +
                         f'{force_label}: {force_val:.2e}<br>' +
                         '<extra></extra>'
        ))
    
    # Layout
    fig.update_layout(
        title=f'{force_label} Diagram',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        width=900,
        height=700
    )
    
    return fig


def plot_mode_shape(structure, mode_number: int = 1, animate: bool = True) -> go.Figure:
    """
    Plot modal analysis mode shape.
    
    Args:
        structure: Structure object with modal analysis results
        mode_number: Mode number to plot (1-indexed)
        animate: If True, create animation of oscillating mode shape
    
    Returns:
        plotly Figure object
    """
    fig = go.Figure()
    
    # Check if modal results exist
    if not hasattr(structure, 'modal_results') or structure.modal_results is None:
        fig.add_annotation(
            text="No modal analysis results available. Please run modal analysis first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    modal_results = structure.modal_results
    mode_shapes = modal_results['mode_shapes']
    frequencies = modal_results['frequencies']
    periods = modal_results['periods']
    
    if mode_number > len(frequencies):
        fig.add_annotation(
            text=f"Mode {mode_number} not available. Only {len(frequencies)} modes computed.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    mode_idx = mode_number - 1  # Convert to 0-indexed
    mode_shape = mode_shapes[:, mode_idx]
    freq = frequencies[mode_idx]
    period = periods[mode_idx]
    
    # Normalize mode shape
    max_disp = np.max(np.abs(mode_shape))
    mode_shape_norm = mode_shape / max_disp if max_disp > 0 else mode_shape
    
    if animate:
        # Create animation frames
        n_frames = 30
        frames = []
        
        for i in range(n_frames):
            # Oscillation: sin wave from 0 to 2π
            phase = 2 * np.pi * i / n_frames
            scale = np.sin(phase)
            
            frame_data = []
            
            # Plot deformed elements for this frame
            for elem_id, element in structure.elements.items():
                node_i = structure.nodes[element.node_i]
                node_j = structure.nodes[element.node_j]
                
                # Get mode shape displacements
                dof_i = node_i.get_dofs()
                dof_j = node_j.get_dofs()
                
                ux_i = mode_shape_norm[dof_i[0]] * scale if len(dof_i) > 0 else 0
                uy_i = mode_shape_norm[dof_i[1]] * scale if len(dof_i) > 1 else 0
                uz_i = mode_shape_norm[dof_i[2]] * scale if len(dof_i) > 2 else 0
                
                ux_j = mode_shape_norm[dof_j[0]] * scale if len(dof_j) > 0 else 0
                uy_j = mode_shape_norm[dof_j[1]] * scale if len(dof_j) > 1 else 0
                uz_j = mode_shape_norm[dof_j[2]] * scale if len(dof_j) > 2 else 0
                
                frame_data.append(go.Scatter3d(
                    x=[node_i.x + ux_i, node_j.x + ux_j],
                    y=[node_i.y + uy_i, node_j.y + uy_j],
                    z=[node_i.z + uz_i, node_j.z + uz_j],
                    mode='lines',
                    line=dict(color='blue', width=4),
                    showlegend=False
                ))
            
            frames.append(go.Frame(data=frame_data, name=f'frame{i}'))
        
        # Initial frame
        for elem_id, element in structure.elements.items():
            node_i = structure.nodes[element.node_i]
            node_j = structure.nodes[element.node_j]
            
            fig.add_trace(go.Scatter3d(
                x=[node_i.x, node_j.x],
                y=[node_i.y, node_j.y],
                z=[node_i.z, node_j.z],
                mode='lines',
                line=dict(color='blue', width=4),
                showlegend=False
            ))
        
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 50}, 'fromcurrent': True}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
                    }
                ]
            }]
        )
    else:
        # Static plot of mode shape
        for elem_id, element in structure.elements.items():
            node_i = structure.nodes[element.node_i]
            node_j = structure.nodes[element.node_j]
            
            dof_i = node_i.get_dofs()
            dof_j = node_j.get_dofs()
            
            ux_i = mode_shape_norm[dof_i[0]] if len(dof_i) > 0 else 0
            uy_i = mode_shape_norm[dof_i[1]] if len(dof_i) > 1 else 0
            uz_i = mode_shape_norm[dof_i[2]] if len(dof_i) > 2 else 0
            
            ux_j = mode_shape_norm[dof_j[0]] if len(dof_j) > 0 else 0
            uy_j = mode_shape_norm[dof_j[1]] if len(dof_j) > 1 else 0
            uz_j = mode_shape_norm[dof_j[2]] if len(dof_j) > 2 else 0
            
            fig.add_trace(go.Scatter3d(
                x=[node_i.x + ux_i, node_j.x + ux_j],
                y=[node_i.y + uy_i, node_j.y + uy_j],
                z=[node_i.z + uz_i, node_j.z + uz_j],
                mode='lines',
                line=dict(color='blue', width=4),
                showlegend=False
            ))
    
    # Layout
    fig.update_layout(
        title=f'Mode {mode_number} - Frequency: {freq:.3f} Hz, Period: {period:.3f} s',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        width=900,
        height=700
    )
    
    return fig
