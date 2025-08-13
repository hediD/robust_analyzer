import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import tempfile
import glob
import json
import hashlib
from datetime import datetime
from pathlib import Path
import zipfile
from typing import List, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from robustness_analyzer import RobustnessAnalyzer
import utils


def load_imagenet_labels():
    """Load ImageNet 1000 class labels from the JSON file."""
    try:
        labels_path = os.path.join("data", "imagenet1000_clsidx_to_labels.json")
        if not os.path.exists(labels_path):
            labels_path = os.path.join("..", "data", "imagenet1000_clsidx_to_labels.json")

        with open(labels_path, 'r') as f:
            id_to_class = eval(f.read())

        class_options = [(idx, label) for idx, label in id_to_class.items()]
        class_options.sort(key=lambda x: x[1].lower())

        return class_options, id_to_class
    except Exception as e:
        st.error(f"❌ Error loading ImageNet labels: {str(e)}")
        return [], {}

def create_target_class_selector():
    """Create a searchable target class selector with ImageNet labels."""
    st.sidebar.subheader("🎯 Target Class Selection")

    class_options, id_to_class = load_imagenet_labels()

    if not class_options:
        st.sidebar.warning("⚠️ Could not load ImageNet labels. Using text input.")
        target_class = st.sidebar.text_input(
            "Target Class",
            value="tank, army tank, armored combat vehicle, armoured combat vehicle",
            help="ImageNet class name for the target"
        )
        return target_class

    all_display_options = [
        f"{idx}: {label}"
        for idx, label in class_options
    ]

    default_idx = 0
    for i, (idx, label) in enumerate(class_options):
        if 'tank' in label.lower() and 'army' in label.lower():
            default_idx = i
            break

    selected_display = st.sidebar.selectbox(
        "🔍 Search & Select ImageNet Class:",
        all_display_options,
        index=default_idx,
        help="Type to search through 1000 ImageNet classes, then select"
    )

    if selected_display:
        selected_idx = int(selected_display.split(':')[0])
        target_class = id_to_class[selected_idx]

        st.sidebar.success(f"✅ Selected: {target_class}")
        return target_class

    return "tank, army tank, armored combat vehicle, armoured combat vehicle"


def setup_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="3D Robustness Analyzer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎯 3D Adversarial Robustness Analyzer")
    st.markdown("""
    Upload 3D objects, textures, and environments to analyze adversarial robustness
    through camera position optimization and visualize results with polar heatmaps.
    """)


def create_sidebar():
    """Create the parameter configuration sidebar."""
    st.sidebar.header("📋 Configuration")

    st.sidebar.subheader("Model Parameters")

    target_class = create_target_class_selector()

    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=1,
        max_value=16,
        value=4,
        step=1,
        help="Number of viewpoints to optimize in parallel"
    )

    st.sidebar.subheader("Optimization Parameters")

    params_to_optimize = st.sidebar.multiselect(
        "Parameters to Optimize",
        options=["camera", "texture", "lighting"],
        default=["camera"],
        help="Select which parameters to optimize during adversarial attack"
    )

    num_runs = st.sidebar.number_input(
        "Number of Runs",
        min_value=1,
        max_value=1000,
        value=1,
        step=1,
        help="Total number of optimization runs"
    )

    num_iterations = st.sidebar.number_input(
        "Adversarial Optimization Steps",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        help="Number of optimization steps per adversarial run"
    )

    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        value=5e-3,
        format_func=lambda x: f"{x:.1e}"
    )

    st.sidebar.subheader("Constraints")

    positive_z = st.sidebar.checkbox(
        "Positive Z Constraint",
        value=True,
        help="Constrain camera to positive elevation (z > 0)"
    )

    targeted = st.sidebar.checkbox(
        "Targeted Attack",
        value=False,
        help="Whether this is a targeted adversarial attack"
    )

    with st.sidebar.expander("Advanced Rendering Settings", expanded=False):
        st.write("**Rendering Paramters**")

        image_size = st.select_slider(
            "Image Size",
            options=[224, 256, 320, 384, 448, 512],
            value=448,
            help="Resolution of rendered images (image_size x image_size)"
        )

        bin_size = st.slider(
            "Bin Size",
            min_value=16,
            max_value=64,
            value=32,
            help="Spatial partitioning for rasterization - larger values use less memory but may be slower"
        )

        max_faces_per_bin = st.number_input(
            "Max Faces per Bin",
            min_value=10000,
            max_value=200000,
            value=100000,
            step=10000,
            help="Maximum faces per spatial bin - increase for complex meshes, decrease to save memory"
        )

        st.info("💡 **Tip:** Use smaller bin sizes and fewer faces per bin if you encounter GPU memory issues.")

    return {
        'target_class': target_class,
        'batch_size': batch_size,
        'params_to_optimize': params_to_optimize,
        'num_runs': num_runs,
        'num_iterations': num_iterations,
        'learning_rate': learning_rate,
        'image_size': image_size,
        'bin_size': bin_size,
        'max_faces_per_bin': max_faces_per_bin,
        'positive_z': positive_z,
        'targeted': targeted
    }


def handle_file_uploads():
    """Handle file uploads for OBJ, MTL, texture, and environment files."""
    st.header("📁 File Uploads")

    tab1, tab2 = st.tabs(["📦 Complete 3D Package", "🎨 Individual Files"])

    with tab1:
        st.subheader("Upload Complete 3D Package")
        st.info("💡 Upload a ZIP file containing .obj, .mtl, and all texture files")

        zip_file = st.file_uploader(
            "Upload ZIP package",
            type=['zip'],
            help="ZIP file containing .obj, .mtl, and texture files with correct folder structure"
        )

        if zip_file:
            st.success(f"✅ Uploaded package: {zip_file.name}")
            return "zip", zip_file, None, None, None

    with tab2:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("3D Object Files")

            obj_file = st.file_uploader(
                "Upload OBJ file",
                type=['obj'],
                help="Upload the 3D mesh file (.obj format)"
            )

            mtl_file = st.file_uploader(
                "Upload MTL file (optional)",
                type=['mtl'],
                help="Upload material definition file (.mtl format)"
            )

            if obj_file:
                st.success(f"✅ OBJ: {obj_file.name}")
            if mtl_file:
                st.success(f"✅ MTL: {mtl_file.name}")

        with col2:
            st.subheader("Textures")

            single_texture = st.file_uploader(
                "Single texture override",
                type=['png', 'jpg', 'jpeg'],
                help="Single texture to override MTL materials (optional)"
            )

            texture_files = st.file_uploader(
                "Multiple texture files",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tga'],
                accept_multiple_files=True,
                help="Upload all texture files referenced in the MTL file"
            )

            if single_texture:
                st.success(f"✅ Single texture: {single_texture.name}")
                st.info("This will override MTL materials")
            elif texture_files:
                st.success(f"✅ Uploaded {len(texture_files)} texture(s)")
                for tex_file in texture_files:
                    st.text(f"  • {tex_file.name}")
            else:
                st.info("💡 Will use MTL materials if available")

        with col3:
            st.subheader("Environment Maps")
            env_files = st.file_uploader(
                "Upload environment maps",
                type=['png', 'jpg', 'jpeg', 'hdr', 'exr'],
                accept_multiple_files=True,
                help="Upload one or more environment map files"
            )

            if env_files:
                st.success(f"✅ Uploaded {len(env_files)} environment map(s)")
                for env_file in env_files:
                    st.text(f"  • {env_file.name}")

        return "individual", obj_file, mtl_file, single_texture or texture_files, env_files


def save_uploaded_files(upload_type, *files):
    """Save uploaded files to persistent cache directory with caching."""
    return save_files_to_cache(upload_type, *files)


def save_zip_package(zip_file, temp_dir):
    """Extract and organize ZIP package."""
    import zipfile

    with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
        zf.extractall(temp_dir)

    obj_files = glob.glob(os.path.join(temp_dir, "**/*.obj"), recursive=True)
    if not obj_files:
        raise ValueError("No .obj file found in ZIP package")

    obj_path = obj_files[0]

    mtl_path = None
    obj_dir = os.path.dirname(obj_path)

    # Look for MTL file referenced in OBJ
    with open(obj_path, 'r') as f:
        for line in f:
            if line.strip().startswith('mtllib '):
                mtl_filename = line.strip().split('mtllib ')[1]
                mtl_path = os.path.join(obj_dir, mtl_filename)
                if not os.path.exists(mtl_path):
                    mtl_files = glob.glob(os.path.join(obj_dir, "*.mtl"))
                    mtl_path = mtl_files[0] if mtl_files else None
                break

    # Find textures referenced in MTL file
    texture_paths = []
    if mtl_path and os.path.exists(mtl_path):
        mtl_dir = os.path.dirname(mtl_path)
        with open(mtl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('map_'):
                    parts = line.split()
                    if len(parts) >= 2:
                        texture_name = parts[-1]  # Last part is usually the filename

                        # Look for texture relative to MTL file location first
                        texture_path = os.path.join(mtl_dir, texture_name)

                        # If not found, also try relative to temp_dir root (fallback)
                        if not os.path.exists(texture_path):
                            texture_path = os.path.join(temp_dir, texture_name)

                        if os.path.exists(texture_path):
                            texture_paths.append(texture_path)

    # Find environment maps - prioritize HDR/EXR formats for environment maps
    envmap_paths = []

    # First, look for dedicated environment map formats (HDR/EXR) without filtering
    env_priority_extensions = ['*.hdr', '*.exr']
    for ext in env_priority_extensions:
        envmap_paths.extend(glob.glob(os.path.join(temp_dir, "**/" + ext), recursive=True))

    # If no HDR/EXR found, look for other formats but filter by name
    if not envmap_paths:
        other_extensions = ['*.png', '*.jpg', '*.jpeg']
        for ext in other_extensions:
            potential_envmaps = glob.glob(os.path.join(temp_dir, "**/" + ext), recursive=True)
            # Only include if filename suggests it's an environment map
            # AND it's not already in our texture list
            envmap_paths.extend([p for p in potential_envmaps
                               if ('env' in p.lower() or 'hdri' in p.lower())
                               and p not in texture_paths])

    texture_path = texture_paths[0] if texture_paths else None

    return obj_path, texture_path, envmap_paths, temp_dir


def save_individual_files(obj_file, mtl_file, texture_files, env_files, temp_dir):
    """Save individual uploaded files."""
    obj_path = None
    texture_path = None
    envmap_paths = []

    if obj_file:
        obj_path = os.path.join(temp_dir, obj_file.name)
        with open(obj_path, 'wb') as f:
            f.write(obj_file.read())

    if mtl_file:
        mtl_path = os.path.join(temp_dir, mtl_file.name)
        with open(mtl_path, 'wb') as f:
            f.write(mtl_file.read())

        # Update OBJ file to reference the MTL file if needed
        if obj_path:
            update_obj_mtl_reference(obj_path, mtl_file.name)

    if texture_files:
        if hasattr(texture_files, 'read'):
            texture_path = os.path.join(temp_dir, texture_files.name)
            with open(texture_path, 'wb') as f:
                f.write(texture_files.read())
        else:
            for tex_file in texture_files:
                tex_path = os.path.join(temp_dir, tex_file.name)
                with open(tex_path, 'wb') as f:
                    f.write(tex_file.read())

    if env_files:
        for env_file in env_files:
            env_path = os.path.join(temp_dir, env_file.name)
            with open(env_path, 'wb') as f:
                f.write(env_file.read())
            envmap_paths.append(env_path)

    return obj_path, texture_path, envmap_paths, temp_dir


def update_obj_mtl_reference(obj_path, mtl_filename):
    """Update OBJ file to reference the correct MTL file."""
    with open(obj_path, 'r') as f:
        lines = f.readlines()

    has_mtllib = any(line.strip().startswith('mtllib ') for line in lines)

    if not has_mtllib:
        lines.insert(0, f"mtllib {mtl_filename}\n")

        with open(obj_path, 'w') as f:
            f.writelines(lines)


def validate_mtl_textures(mtl_path, temp_dir):
    """Validate that all textures referenced in MTL file are available."""
    if not mtl_path or not os.path.exists(mtl_path):
        return True, []

    missing_textures = []
    mtl_dir = os.path.dirname(mtl_path)

    with open(mtl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('map_'):
                parts = line.split()
                if len(parts) >= 2:
                    texture_name = parts[-1]  # Last part is usually the filename

                    # Look for texture relative to MTL file location first
                    texture_path = os.path.join(mtl_dir, texture_name)

                    # If not found, also try relative to temp_dir root (fallback)
                    if not os.path.exists(texture_path):
                        texture_path = os.path.join(temp_dir, texture_name)

                    if not os.path.exists(texture_path):
                        missing_textures.append(texture_name)

    return len(missing_textures) == 0, missing_textures


def create_interactive_polar_plot(camera_positions, labels_correct, logits, target_class, topk_value, selected_indices=None, current_highlighted_index=None):
    """Create an interactive polar plot using Plotly."""

    azimuth, elevation, _ = utils.compute_spherical_coordinates(camera_positions)
    azimuth_rad = np.radians(azimuth)

    _, id_to_class = load_imagenet_labels()
    preds_top1 = torch.argmax(logits, dim=1).cpu().numpy()
    pred_probs = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()

    hover_text = []
    for i in range(len(camera_positions)):
        pred_class = id_to_class.get(int(preds_top1[i]), f"class {preds_top1[i]}")
        status = "✅ Correct" if labels_correct[i] else "❌ Incorrect"
        if current_highlighted_index is not None and i == current_highlighted_index:
            selection_status = "🎯 CURRENTLY DISPLAYED"
        elif selected_indices is not None and i in selected_indices:
            selection_status = "📍 RENDERED"
        else:
            selection_status = ""
        hover_text.append(
            f"Position {i}<br>"
            f"Azimuth: {azimuth[i]:.1f}°<br>"
            f"Elevation: {elevation[i]:.1f}°<br>"
            f"Prediction: {pred_class[:30]}...<br>"
            f"Confidence: {pred_probs[i]:.3f}<br>"
            f"Status: {status}<br>"
            f"{selection_status}<br>"
            f"Click to view rendered image"
        )

    fig = go.Figure()

    # Create masks for different highlighting levels
    if selected_indices is not None and len(selected_indices) > 0:
        selected_mask = np.isin(np.arange(len(camera_positions)), selected_indices)
    else:
        selected_mask = np.zeros(len(camera_positions), dtype=bool)

    if current_highlighted_index is not None:
        current_mask = np.arange(len(camera_positions)) == current_highlighted_index
    else:
        current_mask = np.zeros(len(camera_positions), dtype=bool)

    unselected_mask = ~selected_mask
    selected_not_current_mask = selected_mask & ~current_mask

    correct_mask = labels_correct.astype(bool)
    incorrect_mask = ~correct_mask

    unselected_correct = correct_mask & unselected_mask
    if unselected_correct.any():
        fig.add_trace(go.Scatterpolar(
            r=elevation[unselected_correct],
            theta=azimuth[unselected_correct],
            mode='markers',
            marker=dict(
                size=6,
                color='lightblue',
                opacity=0.4,
                line=dict(width=1, color='darkblue')
            ),
            name=f'Correct (Top-{topk_value})',
            hovertext=[hover_text[i] for i in range(len(hover_text)) if unselected_correct[i]],
            hoverinfo='text',
            customdata=np.arange(len(camera_positions))[unselected_correct]
        ))

    unselected_incorrect = incorrect_mask & unselected_mask
    if unselected_incorrect.any():
        fig.add_trace(go.Scatterpolar(
            r=elevation[unselected_incorrect],
            theta=azimuth[unselected_incorrect],
            mode='markers',
            marker=dict(
                size=6,
                color='lightcoral',
                opacity=0.4,
                line=dict(width=1, color='darkred')
            ),
            name=f'Incorrect (Top-{topk_value})',
            hovertext=[hover_text[i] for i in range(len(hover_text)) if unselected_incorrect[i]],
            hoverinfo='text',
            customdata=np.arange(len(camera_positions))[unselected_incorrect]
        ))

    selected_correct_not_current = correct_mask & selected_not_current_mask
    if selected_correct_not_current.any():
        fig.add_trace(go.Scatterpolar(
            r=elevation[selected_correct_not_current],
            theta=azimuth[selected_correct_not_current],
            mode='markers',
            marker=dict(
                size=8,
                color='darkblue',
                opacity=1.0,
                line=dict(width=2, color='navy')
            ),
            name=f'📍 Selected Correct',
            hovertext=[hover_text[i] for i in range(len(hover_text)) if selected_correct_not_current[i]],
            hoverinfo='text',
            customdata=np.arange(len(camera_positions))[selected_correct_not_current]
        ))

    selected_incorrect_not_current = incorrect_mask & selected_not_current_mask
    if selected_incorrect_not_current.any():
        fig.add_trace(go.Scatterpolar(
            r=elevation[selected_incorrect_not_current],
            theta=azimuth[selected_incorrect_not_current],
            mode='markers',
            marker=dict(
                size=8,
                color='darkred',
                opacity=1.0,
                line=dict(width=2, color='maroon')
            ),
            name=f'📍 Selected Incorrect',
            hovertext=[hover_text[i] for i in range(len(hover_text)) if selected_incorrect_not_current[i]],
            hoverinfo='text',
            customdata=np.arange(len(camera_positions))[selected_incorrect_not_current]
        ))

    current_correct = correct_mask & current_mask
    if current_correct.any():
        fig.add_trace(go.Scatterpolar(
            r=elevation[current_correct],
            theta=azimuth[current_correct],
            mode='markers',
            marker=dict(
                size=14,
                color='darkblue',
                opacity=1.0,
                line=dict(width=4, color='navy')
            ),
            name=f'🎯 Currently Displayed Correct',
            hovertext=[hover_text[i] for i in range(len(hover_text)) if current_correct[i]],
            hoverinfo='text',
            customdata=np.arange(len(camera_positions))[current_correct]
        ))

    current_incorrect = incorrect_mask & current_mask
    if current_incorrect.any():
        fig.add_trace(go.Scatterpolar(
            r=elevation[current_incorrect],
            theta=azimuth[current_incorrect],
            mode='markers',
            marker=dict(
                size=14,
                color='darkred',
                opacity=1.0,
                line=dict(width=4, color='maroon')
            ),
            name=f'🎯 Currently Displayed Incorrect',
            hovertext=[hover_text[i] for i in range(len(hover_text)) if current_incorrect[i]],
            hoverinfo='text',
            customdata=np.arange(len(camera_positions))[current_incorrect]
        ))

    title_text = f"Interactive Camera Position Analysis (Top-{topk_value})<br>Class: {target_class[:50]}..."
    if current_highlighted_index is not None:
        title_text += f"<br>🎯 Currently displaying position {current_highlighted_index}"
    elif selected_indices is not None and len(selected_indices) > 0:
        title_text += f"<br>📍 {len(selected_indices)} positions selected"

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 90],
                ticksuffix="°",
                title="Elevation"
            ),
            angularaxis=dict(
                ticksuffix="°",
                rotation=90,
                direction="clockwise"
            )
        ),
        title=title_text,
        showlegend=True,
        height=600,
        font=dict(size=12)
    )

    return fig

def render_image_at_position(robust_analyzer, results, position_idx, envmap_idx=0):
    """Render an image at a specific camera position."""
    try:
        total_envmaps = len(results.get("envmap_paths", []))
        if total_envmaps == 0:
            total_envmaps = 1

        # Calculate indices based on the flattened structure
        # Structure: (num_runs * batch_size * n_envmaps)
        positions_per_run = robust_analyzer.batch_size * total_envmaps
        run_idx = position_idx // positions_per_run
        remaining = position_idx % positions_per_run
        batch_idx = remaining // total_envmaps
        env_idx = remaining % total_envmaps

        if run_idx >= len(results["final_scene_params"]):
            run_idx = len(results["final_scene_params"]) - 1
        if batch_idx >= robust_analyzer.batch_size:
            batch_idx = robust_analyzer.batch_size - 1
        if env_idx >= total_envmaps:
            env_idx = total_envmaps - 1

        st.write(f"Debug: position_idx={position_idx} -> run_idx={run_idx}, batch_idx={batch_idx}, env_idx={env_idx}")

        # Update model with the scene parameters from this specific run
        scene_params = results["final_scene_params"][run_idx]
        robust_analyzer.model.update_scene_params(scene_params)

        with torch.no_grad():
            render_images = robust_analyzer.model.render(with_grad=False)

            if len(render_images.shape) == 5:  # [batch, env, H, W, C]
                image = render_images[batch_idx, env_idx]
            else:  # [batch, H, W, C] - single environment
                image = render_images[batch_idx]

            image_np = utils.to_numpy(image)
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)

            # Get prediction info - use the stored logits instead of re-computing
            stored_logits = torch.stack(results["final_logits"])  # (num_runs, batch_size, n_envmaps, 1000)
            if len(stored_logits.shape) == 4:  # [runs, batch, env, classes]
                pred_logits = stored_logits[run_idx, batch_idx, env_idx]
            else:  # [runs, batch, classes] - single environment
                pred_logits = stored_logits[run_idx, batch_idx]

            pred_class_idx = pred_logits.argmax().item()
            pred_prob = torch.softmax(pred_logits, dim=0)[pred_class_idx].item()

            _, id_to_class = load_imagenet_labels()
            pred_class = id_to_class.get(pred_class_idx, f"class {pred_class_idx}")

            camera_pos = scene_params["camera"][batch_idx:batch_idx+1]
            azimuth, elevation, distance = utils.compute_spherical_coordinates(camera_pos.cpu().numpy())

            return {
                'image': image_np,
                'prediction': pred_class,
                'confidence': pred_prob,
                'class_idx': pred_class_idx,
                'azimuth': azimuth[0],
                'elevation': elevation[0],
                'distance': distance[0],
                'position_idx': position_idx,
                'run_idx': run_idx,
                'batch_idx': batch_idx,
                'env_idx': env_idx
            }

    except Exception as e:
        st.error(f"Error rendering image at position {position_idx}: {str(e)}")
        return None

def render_multiple_images(robust_analyzer, results, position_indices):
    """Render images for multiple positions with progress tracking."""
    if not position_indices:
        return []

    rendered_images = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, position_idx in enumerate(position_indices):
        progress = (i + 1) / len(position_indices)
        progress_bar.progress(progress)
        status_text.info(f"🎨 Rendering image {i + 1}/{len(position_indices)} (Position {position_idx})...")

        try:
            image_info = render_image_at_position(robust_analyzer, results, position_idx)
            if image_info:
                rendered_images.append(image_info)
            else:
                st.warning(f"⚠️ Failed to render image at position {position_idx}")
        except Exception as e:
            st.error(f"❌ Error rendering position {position_idx}: {str(e)}")

    progress_bar.empty()
    status_text.empty()

    return rendered_images

def create_image_carousel(rendered_images):
    """Create a carousel interface for navigating through rendered images."""
    if not rendered_images:
        st.info("No images to display")
        return

    if 'carousel_index' not in st.session_state:
        st.session_state['carousel_index'] = 0

    if st.session_state['carousel_index'] >= len(rendered_images):
        st.session_state['carousel_index'] = 0

    current_idx = st.session_state['carousel_index']
    current_image = rendered_images[current_idx]

    col_prev, col_info, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button("◀️ Previous", key=f"prev_image_{len(rendered_images)}", use_container_width=True):
            st.session_state['carousel_index'] = (current_idx - 1) % len(rendered_images)
            st.rerun()  # This will update the heatmap highlight

    with col_info:
        st.markdown(f"<div style='text-align: center'><h4>Image {current_idx + 1} of {len(rendered_images)}</h4></div>",
                   unsafe_allow_html=True)

    with col_next:
        if st.button("Next ▶️", key=f"next_image_{len(rendered_images)}", use_container_width=True):
            st.session_state['carousel_index'] = (current_idx + 1) % len(rendered_images)
            st.rerun()  # This will update the heatmap highlight

    st.info("💡 **Tip:** Use the Previous/Next buttons to navigate")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(current_image['image'],
                caption=f"Position {current_image['position_idx']}",
                use_container_width=True)

    with col2:
        st.write("**Position Information:**")
        st.write(f"• **Position Index:** {current_image['position_idx']}")
        st.write(f"• **Azimuth:** {current_image['azimuth']:.1f}°")
        st.write(f"• **Elevation:** {current_image['elevation']:.1f}°")
        st.write(f"• **Distance:** {current_image['distance']:.2f}")

        st.write("**Prediction Information:**")
        st.write(f"• **Predicted Class:** {current_image['prediction'][:40]}...")
        st.write(f"• **Confidence:** {current_image['confidence']:.3f}")

        if 'target_class' in st.session_state:
            target_idx = utils.get_idx(st.session_state.get('target_class', ''))
            is_correct = current_image['class_idx'] == target_idx
            status = "✅ Correct" if is_correct else "❌ Incorrect"
            st.write(f"• **Status:** {status}")

        with st.expander("🔧 Debug Info", expanded=False):
            st.write(f"• **Run Index:** {current_image['run_idx']}")
            st.write(f"• **Batch Index:** {current_image['batch_idx']}")
            st.write(f"• **Environment Index:** {current_image['env_idx']}")

def visualize_results(results, config):
    """Create interactive polar heatmap visualization with clickable points."""
    st.header("📊 Interactive Results Visualization")

    try:
        # Process results - ensure logits and camera positions are aligned
        logits = torch.stack(results["final_logits"])  # Shape: (num_runs, batch_size, n_envmaps, 1000)

        camera_positions = torch.stack([x["camera"] for x in results["final_scene_params"]])  # Shape: (num_runs, batch_size, 3)

        # Handle environment maps properly
        if results.get("envmap_paths"):
            n_envmaps = len(results["envmap_paths"])

            # Expand camera positions to match logits structure
            # From (num_runs, batch_size, 3) to (num_runs, batch_size, n_envmaps, 3)
            camera_positions = camera_positions.unsqueeze(2).expand(-1, -1, n_envmaps, -1)

            logits = logits.reshape(-1, 1000)  # (num_runs * batch_size * n_envmaps, 1000)
            camera_positions = camera_positions.reshape(-1, 3)  # (num_runs * batch_size * n_envmaps, 3)
        else:
            logits = logits.reshape(-1, 1000)  # (num_runs * batch_size, 1000)
            camera_positions = camera_positions.reshape(-1, 3)  # (num_runs * batch_size, 3)

        total_positions = len(logits)

        assert len(camera_positions) == len(logits), f"Mismatch: {len(camera_positions)} camera positions vs {len(logits)} logit entries"

        with st.expander("🏷️ Most Predicted Classes (Top 5)", expanded=False):
            preds_top1 = torch.argmax(logits, dim=1).cpu().numpy()
            values, counts = np.unique(preds_top1, return_counts=True)
            order = np.argsort(-counts)
            top_n = min(5, len(order))

            _, id_to_class = load_imagenet_labels()

            for rank in range(top_n):
                cls_id = int(values[order[rank]])
                cnt = int(counts[order[rank]])
                pct = (cnt / total_positions) * 100.0
                label = id_to_class.get(cls_id, f"class {cls_id}")
                st.write(f"{rank+1}. {label[:30]}... — {cnt} ({pct:.1f}%)")

        st.subheader("🎯 Interactive Camera Position Heatmap")

        topk_value = st.session_state.get('topk_value', 1)
        labels_correct = utils.get_labels_correct(logits, config['target_class'], topk=topk_value)

        col_plot, col_manual = st.columns([2, 1])

        with col_plot:
            all_rendered_indices = None
            current_highlighted_index = None
            selected_but_not_rendered_indices = None

            if 'selected_positions' in st.session_state and st.session_state['selected_positions']:
                selected_but_not_rendered_indices = st.session_state['selected_positions']

            if ('rendered_images' in st.session_state and
                st.session_state['rendered_images']):
                rendered_images = st.session_state['rendered_images']

                all_rendered_indices = [img['position_idx'] for img in rendered_images]

                current_carousel_idx = st.session_state.get('carousel_index', 0)
                if 0 <= current_carousel_idx < len(rendered_images):
                    current_highlighted_index = rendered_images[current_carousel_idx]['position_idx']

            # Combine selected and rendered indices for highlighting
            # Priority: selected_but_not_rendered < all_rendered < current_highlighted
            combined_selected_indices = selected_but_not_rendered_indices

            fig = create_interactive_polar_plot(
                camera_positions.numpy(),
                labels_correct.numpy(),
                logits,
                config['target_class'],
                topk_value,
                selected_indices=combined_selected_indices,
                current_highlighted_index=current_highlighted_index
            )

            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="polar_plot")

            if st.checkbox("🔍 Debug Mode", value=False):
                st.write("**Event Debug Info:**")
                st.write(f"Event: {event}")
                if event:
                    st.write(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
                    if isinstance(event, dict) and 'selection' in event:
                        st.write(f"Selection: {event['selection']}")

            # Handle point selection with improved error handling
            if event and isinstance(event, dict):
                selection = None
                if 'selection' in event and event['selection']:
                    if 'points' in event['selection'] and event['selection']['points']:
                        selection = event['selection']['points']
                    elif 'point_indices' in event['selection']:
                        selection = event['selection']['point_indices']

                if selection:
                    try:
                        selected_positions = []

                        for point in selection:
                            position_idx = None

                            if isinstance(point, dict):
                                if 'customdata' in point:
                                    position_idx = int(point['customdata'])
                                elif 'pointIndex' in point:
                                    position_idx = int(point['pointIndex'])
                            elif isinstance(point, (int, float)):
                                position_idx = int(point)

                            if position_idx is not None:
                                selected_positions.append(position_idx)

                        if selected_positions:
                            st.session_state['selected_positions'] = selected_positions

                            if len(selected_positions) == 1:
                                st.success(f"✅ Selected position {selected_positions[0]}")
                            else:
                                st.success(f"✅ Selected {len(selected_positions)} positions: {selected_positions}")

                            st.session_state['trigger_render'] = True

                    except Exception as e:
                        st.error(f"❌ Error processing selection: {str(e)}")

        with col_manual:
            topk_value = int(
                st.number_input(
                    "Top-K Accuracy Threshold",
                    min_value=1,
                    max_value=5,
                    value=st.session_state.get('topk_value', 1),
                    step=1,
                    help="Consider prediction correct if target class appears in top K predictions"
                )
            )
            st.session_state['topk_value'] = topk_value

            labels_correct = utils.get_labels_correct(logits, config['target_class'], topk=topk_value)

            correct_count = labels_correct.sum().item()
            total_positions = len(labels_correct)
            accuracy = (correct_count / total_positions) * 100

            st.info(f"📊 **Top-{topk_value} Accuracy:** {correct_count}/{total_positions} ({accuracy:.1f}%) correct")

            st.subheader("🎯 Position Selection")

            target_idx = utils.get_idx(config['target_class'])

            target_probs = torch.softmax(logits, dim=1)[:, target_idx].cpu().numpy()

            col_type, col_count = st.columns(2)
            with col_type:
                selection_type = st.selectbox(
                    "Show positions with:",
                    options=["worst_confidence", "best_confidence", "incorrect_predictions", "correct_predictions"],
                    format_func = lambda x: {
                        "worst_confidence": "▼ Lowest confidence (worst)",
                        "best_confidence": "▲ Highest confidence (best)",
                        "incorrect_predictions": "❌ Incorrect predictions",
                        "correct_predictions": "✅ Correct predictions"
                    }[x],
                    help="Choose which type of positions to analyze"
                )

            with col_count:
                max_positions = min(20, len(camera_positions))
                num_positions = st.number_input(
                    "Number of positions",
                    min_value=1,
                    max_value=max_positions,
                    value=min(5, max_positions),
                    step=1,
                    help=f"Number of positions to render (max {max_positions})"
                )

            if selection_type == "worst_confidence":
                sorted_indices = np.argsort(target_probs)
                selected_indices = sorted_indices[:num_positions]
                description = f"Top {num_positions} positions with lowest confidence in '{config['target_class'][:30]}...'"

            elif selection_type == "best_confidence":
                sorted_indices = np.argsort(target_probs)[::-1]
                selected_indices = sorted_indices[:num_positions]
                description = f"Top {num_positions} positions with highest confidence in '{config['target_class'][:30]}...'"

            elif selection_type == "incorrect_predictions":
                preds_top1 = torch.argmax(logits, dim=1).cpu().numpy()
                incorrect_mask = preds_top1 != target_idx
                incorrect_indices = np.where(incorrect_mask)[0]

                if len(incorrect_indices) > 0:
                    incorrect_confidences = target_probs[incorrect_indices]
                    sorted_incorrect = incorrect_indices[np.argsort(incorrect_confidences)]
                    selected_indices = sorted_incorrect[:num_positions]
                    description = f"Top {min(num_positions, len(selected_indices))} incorrect predictions (lowest confidence)"
                else:
                    selected_indices = []
                    description = "No incorrect predictions found!"

            else:  # correct_predictions
                preds_top1 = torch.argmax(logits, dim=1).cpu().numpy()
                correct_mask = preds_top1 == target_idx
                correct_indices = np.where(correct_mask)[0]

                if len(correct_indices) > 0:
                    correct_confidences = target_probs[correct_indices]
                    sorted_correct = correct_indices[np.argsort(correct_confidences)[::-1]]
                    selected_indices = sorted_correct[:num_positions]
                    description = f"Top {min(num_positions, len(selected_indices))} correct predictions (highest confidence)"
                else:
                    selected_indices = []
                    description = "No correct predictions found!"

            st.info(f"📋 **Selection:** {description}")

            if len(selected_indices) > 0:
                selected_confidences = target_probs[selected_indices]
                min_conf = selected_confidences.min()
                max_conf = selected_confidences.max()
                avg_conf = selected_confidences.mean()

                st.caption(f"**Confidence range:** {min_conf:.3f} - {max_conf:.3f} (avg: {avg_conf:.3f})")

                with st.expander("🔍 Preview selected positions", expanded=False):
                    with st.container():
                        st.markdown(
                            """
                            <div style="height: 200px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                            """,
                            unsafe_allow_html=True
                        )

                        for i, pos_idx in enumerate(selected_indices):
                            conf = target_probs[pos_idx]
                            pred_idx = torch.argmax(logits[pos_idx]).item()
                            pred_class = load_imagenet_labels()[1].get(pred_idx, f"class {pred_idx}")
                            status = "✅" if pred_idx == target_idx else "❌"
                            st.write(f"{i+1}. **Position {pos_idx}:** {status} {conf:.3f} confidence → {pred_class[:25]}...")

                        st.markdown("</div>", unsafe_allow_html=True)

            if st.button("🎨 Render Selected Positions", type="primary", disabled=len(selected_indices)==0):
                if len(selected_indices) > 0:
                    # Immediately update the heatmap highlighting
                    st.session_state['selected_positions'] = selected_indices.tolist()
                    st.session_state['trigger_render'] = True
                    st.success(f"✅ Selected {len(selected_indices)} positions - updating heatmap and rendering...")
                    st.rerun()  # This will immediately show the highlighting and then proceed to render

        # Improved caching logic for multiple selected positions
        if (st.session_state.get('trigger_render', False) and
            'selected_positions' in st.session_state and
            'robust_analyzer' in st.session_state):

            st.subheader("🖼️ Rendered Images")

            selected_positions = st.session_state['selected_positions']
            robust_analyzer = st.session_state['robust_analyzer']

            valid_positions = [pos for pos in selected_positions if pos < len(camera_positions)]
            if len(valid_positions) != len(selected_positions):
                invalid_positions = [pos for pos in selected_positions if pos >= len(camera_positions)]
                st.warning(f"⚠️ Removed invalid positions: {invalid_positions}")

            if valid_positions:
                st.session_state['target_class'] = config['target_class']

                cache_key = f"{sorted(valid_positions)}_{id(robust_analyzer)}"

                # Only render if we don't have cached images for these exact positions
                if (st.session_state.get('image_cache_key') != cache_key):
                    with st.status("🎨 Rendering selected images...", expanded=True) as status:
                        rendered_images = render_multiple_images(robust_analyzer, results, valid_positions)
                        st.session_state['rendered_images'] = rendered_images
                        st.session_state['image_cache_key'] = cache_key
                        st.session_state['carousel_index'] = 0
                        status.update(label=f"✅ Rendered {len(rendered_images)} images!", state="complete")
                        # Add this line to update the polar plot with first image highlighted
                        st.rerun()  # This will regenerate the polar plot with the first image magnified

                if st.session_state.get('rendered_images'):
                    create_image_carousel(st.session_state['rendered_images'])

                    st.info(f"📦 **Cached:** {len(st.session_state['rendered_images'])} images in memory (no re-rendering needed)")
                else:
                    st.error("❌ No images were successfully rendered")

            # Reset trigger but keep images cached
            st.session_state['trigger_render'] = False

        # Display carousel if we have cached images (even without trigger)
        elif ('rendered_images' in st.session_state and
              st.session_state['rendered_images'] and
              'selected_positions' in st.session_state):

            st.subheader("🖼️ Rendered Images")
            create_image_carousel(st.session_state['rendered_images'])
            st.info(f"📦 **Cached:** {len(st.session_state['rendered_images'])} images in memory")

        else:
            if 'robust_analyzer' not in st.session_state:
                st.info("⚠️ No robust analyzer found. Please run the analysis first.")
            else:
                st.info("👆 Click on one or more points in the polar plot above, or use the position selection below to view rendered images!")

        with st.expander("📊 Static Polar Plot (fallback)", expanded=False):
            utils.visualize_positions_polar(
                camera_positions.numpy(),
                labels_correct.numpy(),
                title=f"Camera Position Analysis (Top-{topk_value}) - Target: {config['target_class'][:30]}..."
            )
            current_fig = plt.gcf()
            st.pyplot(current_fig)
            plt.close(current_fig)

        return {
            'camera_positions': camera_positions.numpy(),
            'labels_correct': labels_correct.numpy(),
            'target_class': config['target_class'],
            'topk': topk_value
        }

    except Exception as e:
        st.error(f"❌ Error in processing results: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


def run_analysis(obj_path, texture_path, envmap_paths, config):
    """Run the robustness analysis with the given configuration."""

    raster_settings = {
        "image_size": config['image_size'],
        "bin_size": config['bin_size'],
        "max_faces_per_bin": config['max_faces_per_bin'],
    }

    kwargs = {
        "obj_path": obj_path,
        "texture_path": texture_path,
        "envmap_paths": envmap_paths,
        "target_class": config['target_class'],
        "batch_size": config['batch_size'],
        "params_to_optimize": config['params_to_optimize'],
        "targeted": config['targeted'],
        "positive_z": config['positive_z'],
        "raster_settings": raster_settings
    }

    with st.spinner("Initializing robustness analyzer..."):
        robust_analyzer = RobustnessAnalyzer(**kwargs)

    # Store analyzer in session state for later use
    st.session_state['robust_analyzer'] = robust_analyzer

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(run_num, total_runs, iteration=None, total_iterations=None):
        """Update Streamlit progress indicators."""
        # Calculate overall progress (runs are the main progress indicator)
        if iteration is not None and total_iterations is not None:
            run_progress = (run_num + iteration / total_iterations) / total_runs
        else:
            run_progress = run_num / total_runs
        progress_bar.progress(run_progress)

        # Update status text with current run information
        if iteration is not None and total_iterations is not None:
            status_text.info(
                f"🎯 **Run {run_num + 1}/{total_runs}** | "
                f"Iteration {iteration + 1}/{total_iterations} | "
                f"Overall Progress: {run_progress:.1%}"
            )
        else:
            status_text.info(
                f"🎯 **Run {run_num + 1}/{total_runs}** | "
                f"Overall Progress: {run_progress:.1%}"
            )

    try:
        status_text.info(f"🚀 Starting {config['num_runs']} optimization runs...")

        results = robust_analyzer.run(
            num_runs=config['num_runs'],
            num_iterations=config['num_iterations'],
            lr=config['learning_rate'],
            progress_callback=progress_callback
        )

        progress_bar.progress(1.0)
        status_text.success(f"✅ Analysis completed successfully! Processed {config['num_runs']} runs.")

        return robust_analyzer, results

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        return None, None


def download_results(results, plot_data):
    """Provide download functionality for results and visualization."""
    st.header("💾 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Download Results as JSON"):
            serializable_results = {}
            for key, values in results.items():
                if isinstance(values, list) and len(values) > 0:
                    if torch.is_tensor(values[0]):
                        serializable_results[key] = [v.detach().cpu().numpy().tolist() for v in values]
                    elif isinstance(values[0], dict):
                        serializable_results[key] = [
                            {k: v.detach().cpu().numpy().tolist() if torch.is_tensor(v) else v
                             for k, v in item.items()}
                            for item in values
                        ]
                    else:
                        serializable_results[key] = values
                else:
                    serializable_results[key] = values

            import json
            json_str = json.dumps(serializable_results, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name="robustness_analysis_results.json",
                mime="application/json"
            )

    with col2:
        if plot_data and st.button("📊 Download Current Polar Heatmap"):
            # Call the visualization function with current settings (it creates its own figure)
            utils.visualize_positions_polar(
                plot_data['camera_positions'],
                plot_data['labels_correct'],
                title=f"Camera Position Analysis (Top-{plot_data['topk']}) - Target: {plot_data['target_class'][:30]}..."
            )

            current_fig = plt.gcf()

            img_buffer = io.BytesIO()
            current_fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)

            # Close the figure to free memory
            plt.close(current_fig)

            st.download_button(
                label="💾 Download PNG",
                data=img_buffer.getvalue(),
                file_name=f"robustness_polar_heatmap_top{plot_data['topk']}.png",
                mime="image/png"
            )


def get_cache_dir():
    """Get or create a persistent cache directory with absolute path."""
    # Use absolute path based on the script location, not current working directory
    script_dir = Path(__file__).parent.absolute()
    cache_dir = script_dir / "file_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def get_file_hash(file_content):
    """Generate a hash for file content to use as cache key."""
    return hashlib.md5(file_content).hexdigest()

def check_cached_files():
    """Check if we have any cached files and load them into session state."""
    # Always check for cached files, don't use the 'file_cache_checked' flag
    # This allows recovery from failed cache loads

    cache_dir = get_cache_dir()

    cache_entries = []

    for cache_subdir in cache_dir.iterdir():
        if cache_subdir.is_dir():
            metadata_file = cache_subdir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        cached_metadata = json.load(f)

                    all_files_exist = True
                    required_paths = [cached_metadata.get('obj_path'), cached_metadata.get('texture_path')] + cached_metadata.get('envmap_paths', [])

                    for file_path in required_paths:
                        if file_path and not Path(file_path).exists():
                            all_files_exist = False
                            break

                    if all_files_exist and cached_metadata.get('obj_path'):
                        timestamp_str = cached_metadata.get('timestamp', '')
                        try:
                            if timestamp_str.startswith('"') and timestamp_str.endswith('"'):
                                timestamp_str = timestamp_str[1:-1]
                            timestamp = datetime.fromisoformat(timestamp_str.replace('"', ''))
                        except:
                            timestamp = datetime.min

                        cache_entries.append({
                            'metadata': cached_metadata,
                            'timestamp': timestamp,
                            'cache_key': cache_subdir.name
                        })

                except Exception as e:
                    print(f"Error loading cache metadata from {metadata_file}: {e}")
                    continue

    if cache_entries:
        latest_entry = max(cache_entries, key=lambda x: x['timestamp'])
        cached_metadata = latest_entry['metadata']

        st.session_state['files_processed'] = True
        st.session_state['file_paths'] = cached_metadata
        st.session_state['using_cached_files'] = True
        st.session_state['cache_info'] = {
            'cache_key': latest_entry['cache_key'][:8],
            'timestamp': latest_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'file_names': cached_metadata.get('file_names', ['Unknown files'])
        }

        print(f"✅ Loaded cached files from {latest_entry['cache_key'][:8]}...")
        return True

    return False

def save_files_to_cache(upload_type, *files):
    """Save uploaded files to persistent cache directory."""
    cache_dir = get_cache_dir()

    file_contents = []
    file_names = []

    if upload_type == "zip":
        zip_file = files[0]
        file_content = zip_file.read()
        zip_file.seek(0)
        cache_key = get_file_hash(file_content)
        file_names = [zip_file.name]
    else:
        obj_file, mtl_file, texture_files, env_files = files

        if obj_file:
            content = obj_file.read()
            obj_file.seek(0)
            file_contents.append(content)
            file_names.append(obj_file.name)

        if mtl_file:
            content = mtl_file.read()
            mtl_file.seek(0)
            file_contents.append(content)
            file_names.append(mtl_file.name)

        if texture_files:
            if hasattr(texture_files, 'read'):
                content = texture_files.read()
                texture_files.seek(0)
                file_contents.append(content)
                file_names.append(texture_files.name)
            else:
                for tex_file in texture_files:
                    content = tex_file.read()
                    tex_file.seek(0)
                    file_contents.append(content)
                    file_names.append(tex_file.name)

        if env_files:
            for env_file in env_files:
                content = env_file.read()
                env_file.seek(0)
                file_contents.append(content)
                file_names.append(env_file.name)

        combined_content = b''.join(file_contents)
        cache_key = get_file_hash(combined_content)

    upload_cache_dir = cache_dir / cache_key

    metadata_file = upload_cache_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                cached_data = json.load(f)

            all_exist = True
            for file_path in [cached_data.get('obj_path'), cached_data.get('texture_path')] + cached_data.get('envmap_paths', []):
                if file_path and not Path(file_path).exists():
                    all_exist = False
                    break

            if all_exist:
                st.info(f"📦 Using cached files (hash: {cache_key[:8]}...)")
                return cached_data['obj_path'], cached_data['texture_path'], cached_data['envmap_paths'], str(upload_cache_dir)
        except Exception as e:
            print(f"Error loading existing cache: {e}")

    upload_cache_dir.mkdir(exist_ok=True)

    st.info(f"💾 Caching files for future use (hash: {cache_key[:8]}...)")

    try:
        if upload_type == "zip":
            obj_path, texture_path, envmap_paths, _ = save_zip_package(files[0], str(upload_cache_dir))
        else:
            obj_path, texture_path, envmap_paths, _ = save_individual_files(*files, str(upload_cache_dir))

        metadata = {
            'obj_path': obj_path,
            'texture_path': texture_path,
            'envmap_paths': envmap_paths,
            'temp_dir': str(upload_cache_dir),
            'cache_key': cache_key,
            'file_names': file_names,
            'upload_type': upload_type,
            'timestamp': datetime.now().isoformat()  # Use isoformat instead of json.dumps
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Files cached successfully to {cache_key[:8]}...")
        return obj_path, texture_path, envmap_paths, str(upload_cache_dir)

    except Exception as e:
        st.error(f"❌ Error caching files: {str(e)}")
        if upload_cache_dir.exists():
            import shutil
            shutil.rmtree(upload_cache_dir, ignore_errors=True)
        raise

def main():
    """Main application function."""
    setup_page()

    has_cached_files = check_cached_files()

    if st.session_state.get('using_cached_files', False):
        st.success("🔄 **Using cached files from previous session**")

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.session_state.get('cache_info'):
                cache_info = st.session_state['cache_info']
                st.info(f"📁 Cache: {', '.join(cache_info['file_names'])}")

        with col2:
            if st.button("🗑️ Clear Cache", help="Clear cached files and upload new ones"):
                st.session_state['files_processed'] = False
                st.session_state['file_paths'] = None
                st.session_state['using_cached_files'] = False
                st.session_state.pop('cache_info', None)

                # Optionally remove all cache files (uncomment if desired)
                # cache_dir = get_cache_dir()
                # import shutil
                # if cache_dir.exists():
                #     shutil.rmtree(cache_dir, ignore_errors=True)

                st.rerun()

    config = create_sidebar()

    # Only show file uploads if not using cached files
    if not st.session_state.get('using_cached_files', False):
        upload_result = handle_file_uploads()
        upload_type = upload_result[0]

        if 'files_processed' not in st.session_state:
            st.session_state['files_processed'] = False
            st.session_state['file_paths'] = None

        if upload_type == "zip":
            zip_file = upload_result[1]
            if zip_file:
                try:
                    with st.spinner("Processing ZIP package..."):
                        obj_path, texture_path, envmap_paths, temp_dir = save_uploaded_files(upload_type, zip_file)

                    st.success("✅ ZIP package processed successfully!")

                    st.subheader("📋 Processed Files")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write("**3D Object:**")
                        if obj_path:
                            st.write(f"✅ {os.path.basename(obj_path)}")
                        else:
                            st.write("❌ No OBJ file found")

                    with col2:
                        st.write("**Texture:**")
                        if texture_path:
                            st.write(f"✅ {os.path.basename(texture_path)}")
                        else:
                            st.write("💡 Will use MTL materials")

                    with col3:
                        st.write("**Environment Maps:**")
                        if envmap_paths:
                            for env_path in envmap_paths:
                                st.write(f"✅ {os.path.basename(env_path)}")
                        else:
                            st.write("❌ No environment maps found")

                    mtl_path = obj_path.replace('.obj', '.mtl') if obj_path else None
                    if mtl_path and os.path.exists(mtl_path):
                        is_valid, missing = validate_mtl_textures(mtl_path, temp_dir)
                        if not is_valid:
                            st.warning(f"⚠️ Missing texture files: {', '.join(missing)}")

                    if obj_path and envmap_paths:
                        st.session_state['files_processed'] = True
                        st.session_state['file_paths'] = {
                            'obj_path': obj_path,
                            'texture_path': texture_path,
                            'envmap_paths': envmap_paths,
                            'temp_dir': temp_dir
                        }
                    else:
                        st.error("❌ Missing required files (OBJ and environment maps)")

                except Exception as e:
                    st.error(f"❌ Error processing ZIP package: {str(e)}")

        elif upload_type == "individual":
            obj_file, mtl_file, texture_files, env_files = upload_result[1:]

            if obj_file and env_files:
                try:
                    with st.spinner("Processing uploaded files..."):
                        obj_path, texture_path, envmap_paths, temp_dir = save_uploaded_files(
                            upload_type, obj_file, mtl_file, texture_files, env_files
                        )

                    st.success("✅ Files processed successfully!")

                    st.subheader("📋 Processed Files")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write("**3D Object:**")
                        st.write(f"✅ {obj_file.name}")

                    with col2:
                        st.write("**Texture:**")
                        if texture_files:
                            if hasattr(texture_files, 'read'):
                                st.write(f"✅ {texture_files.name}")
                            else:
                                for tex_file in texture_files:
                                    st.write(f"✅ {tex_file.name}")
                        else:
                            st.write("💡 Will use MTL materials")

                    with col3:
                        st.write("**Environment Maps:**")
                        for env_file in env_files:
                            st.write(f"✅ {env_file.name}")

                    if mtl_file:
                        mtl_path = os.path.join(temp_dir, mtl_file.name)
                        is_valid, missing = validate_mtl_textures(mtl_path, temp_dir)
                        if not is_valid:
                            st.warning(f"⚠️ Missing texture files: {', '.join(missing)}")
                            st.info("💡 Please upload all texture files referenced in the MTL file")

                    st.session_state['files_processed'] = True
                    st.session_state['file_paths'] = {
                        'obj_path': obj_path,
                        'texture_path': texture_path,
                        'envmap_paths': envmap_paths,
                        'temp_dir': temp_dir
                    }

                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
            else:
                st.warning("⚠️ Please upload at least an OBJ file and environment map(s) to proceed.")

    if st.session_state.get('files_processed', False):
        st.header("🚀 Run Analysis")

        with st.expander("📋 Current Configuration", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Target Class:** {config['target_class'][:40]}...")
                st.write(f"**Batch Size:** {config['batch_size']}")
                st.write(f"**Parameters to Optimize:** {', '.join(config['params_to_optimize'])}")
            with col2:
                st.write(f"**Number of Runs:** {config['num_runs']}")
                st.write(f"**Optimization Steps:** {config['num_iterations']}")
                st.write(f"**Learning Rate:** {config['learning_rate']:.1e}")

        if st.button("🎯 **Run Robustness Analysis**", type="primary", use_container_width=True):
            file_paths = st.session_state['file_paths']

            robust_analyzer, results = run_analysis(
                file_paths['obj_path'],
                file_paths['texture_path'],
                file_paths['envmap_paths'],
                config
            )

            if results is not None:
                st.session_state['results'] = results
                st.session_state['config'] = config

    if st.session_state.get('results') is not None:
        plot_data = visualize_results(
            st.session_state['results'],
            st.session_state.get('config', config)
        )
        download_results(st.session_state['results'], plot_data)

    if not st.session_state.get('files_processed', False):
        st.header("📖 Example Usage")
        st.markdown("""
        **Typical workflow:**
        1. Upload your 3D object (.obj file)
        2. Optionally upload a texture image
        3. Upload one or more environment maps
        4. Configure optimization parameters in the sidebar
        5. Expand "Advanced Rendering Settings" if needed
        6. Click "Run Robustness Analysis"
        7. View the polar heatmap results
        8. Download results as JSON

        **File Requirements:**
        - **OBJ file**: 3D mesh in Wavefront OBJ format
        - **Texture**: PNG/JPG image (optional, will use MTL if not provided)
        - **Environment maps**: HDR/EXR or regular images for lighting
        """)


if __name__ == "__main__":
    main()
