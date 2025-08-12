import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import tempfile
import glob
import json
from pathlib import Path
import zipfile
from typing import List, Dict, Optional

from robustness_analyzer import RobustnessAnalyzer
import utils


def load_imagenet_labels():
    """Load ImageNet 1000 class labels from the JSON file."""
    try:
        # First try to load from the data directory
        labels_path = os.path.join("data", "imagenet1000_clsidx_to_labels.json")
        if not os.path.exists(labels_path):
            # Fallback to relative path from src directory
            labels_path = os.path.join("..", "data", "imagenet1000_clsidx_to_labels.json")

        with open(labels_path, 'r') as f:
            id_to_class = eval(f.read())

        # Convert to a list of (id, label) tuples for easier searching
        class_options = [(idx, label) for idx, label in id_to_class.items()]
        class_options.sort(key=lambda x: x[1].lower())  # Sort alphabetically by label

        return class_options, id_to_class
    except Exception as e:
        st.error(f"❌ Error loading ImageNet labels: {str(e)}")
        return [], {}

def create_target_class_selector():
    """Create a searchable target class selector with ImageNet labels."""
    st.sidebar.subheader("🎯 Target Class Selection")

    # Load ImageNet labels
    class_options, id_to_class = load_imagenet_labels()

    if not class_options:
        # Fallback to text input if labels can't be loaded
        st.sidebar.warning("⚠️ Could not load ImageNet labels. Using text input.")
        target_class = st.sidebar.text_input(
            "Target Class",
            value="tank, army tank, armored combat vehicle, armoured combat vehicle",
            help="ImageNet class name for the target"
        )
        return target_class

    # Create all display options for the selectbox
    all_display_options = [
        f"{idx}: {label}"
        for idx, label in class_options
    ]

    # Find default tank class
    default_idx = 0
    for i, (idx, label) in enumerate(class_options):
        if 'tank' in label.lower() and 'army' in label.lower():
            default_idx = i
            break

    # Single searchable selectbox
    selected_display = st.sidebar.selectbox(
        "🔍 Search & Select ImageNet Class:",
        all_display_options,
        index=default_idx,
        help="Type to search through 1000 ImageNet classes, then select"
    )

    # Extract the selected class information
    if selected_display:
        selected_idx = int(selected_display.split(':')[0])
        target_class = id_to_class[selected_idx]

        # Show the selected class name
        st.sidebar.success(f"✅ Selected: {target_class}")
        return target_class

    # Fallback
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

    # Model Parameters
    st.sidebar.subheader("Model Parameters")

    # Use the new target class selector
    target_class = create_target_class_selector()

    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=1,
        max_value=16,
        value=4,
        step=1,
        help="Number of viewpoints to optimize in parallel"
    )

    # Optimization Parameters
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

    # Constraints
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

    # Advanced Rendering Parameters (Collapsible)
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

    # Create tabs for different upload modes
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

            # Option 1: Single texture override
            single_texture = st.file_uploader(
                "Single texture override",
                type=['png', 'jpg', 'jpeg'],
                help="Single texture to override MTL materials (optional)"
            )

            # Option 2: Multiple texture files for MTL
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
    """Save uploaded files to temporary directory with proper structure."""
    temp_dir = tempfile.mkdtemp()

    if upload_type == "zip":
        zip_file = files[0]
        return save_zip_package(zip_file, temp_dir)
    else:
        obj_file, mtl_file, texture_files, env_files = files
        return save_individual_files(obj_file, mtl_file, texture_files, env_files, temp_dir)


def save_zip_package(zip_file, temp_dir):
    """Extract and organize ZIP package."""
    import zipfile

    # Extract ZIP file
    with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
        zf.extractall(temp_dir)

    # Find OBJ file
    obj_files = glob.glob(os.path.join(temp_dir, "**/*.obj"), recursive=True)
    if not obj_files:
        raise ValueError("No .obj file found in ZIP package")

    obj_path = obj_files[0]

    # MTL file should be in the same directory or referenced in OBJ
    mtl_path = None
    obj_dir = os.path.dirname(obj_path)

    # Look for MTL file referenced in OBJ
    with open(obj_path, 'r') as f:
        for line in f:
            if line.strip().startswith('mtllib '):
                mtl_filename = line.strip().split('mtllib ')[1]
                mtl_path = os.path.join(obj_dir, mtl_filename)
                if not os.path.exists(mtl_path):
                    # Look for any MTL file in the directory
                    mtl_files = glob.glob(os.path.join(obj_dir, "*.mtl"))
                    mtl_path = mtl_files[0] if mtl_files else None
                break

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
            envmap_paths.extend([p for p in potential_envmaps if 'env' in p.lower() or 'hdri' in p.lower()])

    return obj_path, None, envmap_paths, temp_dir


def save_individual_files(obj_file, mtl_file, texture_files, env_files, temp_dir):
    """Save individual uploaded files."""
    obj_path = None
    texture_path = None
    envmap_paths = []

    # Save OBJ file
    if obj_file:
        obj_path = os.path.join(temp_dir, obj_file.name)
        with open(obj_path, 'wb') as f:
            f.write(obj_file.read())

    # Save MTL file
    if mtl_file:
        mtl_path = os.path.join(temp_dir, mtl_file.name)
        with open(mtl_path, 'wb') as f:
            f.write(mtl_file.read())

        # Update OBJ file to reference the MTL file if needed
        if obj_path:
            update_obj_mtl_reference(obj_path, mtl_file.name)

    # Save texture files
    if texture_files:
        if hasattr(texture_files, 'read'):  # Single file
            texture_path = os.path.join(temp_dir, texture_files.name)
            with open(texture_path, 'wb') as f:
                f.write(texture_files.read())
        else:  # Multiple files
            for tex_file in texture_files:
                tex_path = os.path.join(temp_dir, tex_file.name)
                with open(tex_path, 'wb') as f:
                    f.write(tex_file.read())

    # Save environment maps
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

    # Check if mtllib line already exists
    has_mtllib = any(line.strip().startswith('mtllib ') for line in lines)

    if not has_mtllib:
        # Add mtllib reference at the beginning
        lines.insert(0, f"mtllib {mtl_filename}\n")

        with open(obj_path, 'w') as f:
            f.writelines(lines)


def validate_mtl_textures(mtl_path, temp_dir):
    """Validate that all textures referenced in MTL file are available."""
    if not mtl_path or not os.path.exists(mtl_path):
        return True, []

    missing_textures = []
    mtl_dir = os.path.dirname(mtl_path)  # Get the directory containing the MTL file

    with open(mtl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('map_'):
                # Extract texture filename
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

    # Initialize analyzer
    with st.spinner("Initializing robustness analyzer..."):
        robust_analyzer = RobustnessAnalyzer(**kwargs)

    # Run optimization
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        with st.spinner(f"Running {config['num_runs']} optimization runs..."):
            results = robust_analyzer.run(
                num_runs=config['num_runs'],
                num_iterations=config['num_iterations'],
                lr=config['learning_rate']
            )

        progress_bar.progress(100)
        status_text.success("✅ Analysis completed successfully!")

        return robust_analyzer, results

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        return None, None


def visualize_results(results, config):
    """Create interactive polar heatmap visualization with topk slider."""
    st.header("📊 Interactive Results Visualization")

    try:
        # Process results as in your original code
        logits = torch.stack(results["final_logits"]).reshape(-1, 1000)

        # Get camera positions
        camera_positions = torch.stack([x["camera"] for x in results["final_scene_params"]]).reshape(-1, 3)
        if results.get("envmap_paths"):
            camera_positions = camera_positions.repeat_interleave(len(results["envmap_paths"]) or 1, dim=0)

        # Interactive topk slider
        st.subheader("🎯 Camera Position Heatmap")

        topk_value = st.slider(
            "Top-K Accuracy Threshold",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Consider prediction correct if target class appears in top K predictions"
        )

        # Get labels correctness based on current topk value
        labels_correct = utils.get_labels_correct(logits, config['target_class'], topk=topk_value)

        # Show current accuracy stats in a compact format
        correct_count = labels_correct.sum().item()
        total_positions = len(labels_correct)
        accuracy = (correct_count / total_positions) * 100

        st.info(f"📊 **Top-{topk_value} Accuracy:** {correct_count}/{total_positions} ({accuracy:.1f}%) correct")

        # Call the visualization function directly (it creates its own figure)
        utils.visualize_positions_polar(
            camera_positions.numpy(),
            labels_correct.numpy(),
            title=f"Camera Position Analysis (Top-{topk_value}) - Target: {config['target_class'][:30]}..."
        )

        # The function above shows the plot with plt.show(), but we need to capture it for Streamlit
        # Get the current figure that was just created
        current_fig = plt.gcf()
        st.pyplot(current_fig)
        plt.close(current_fig)  # Close to free memory

        # Store plot data for download
        return {
            'camera_positions': camera_positions.numpy(),
            'labels_correct': labels_correct.numpy(),
            'target_class': config['target_class'],
            'topk': topk_value
        }

    except Exception as e:
        st.error(f"❌ Error in processing results: {str(e)}")
        return None


def download_results(results, plot_data):
    """Provide download functionality for results and visualization."""
    st.header("💾 Download Results")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📄 Download Results as JSON"):
            # Prepare serializable results
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

            # Create download
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

            # Get the current figure that was just created
            current_fig = plt.gcf()

            # Save plot to bytes buffer
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


def main():
    """Main application function."""
    setup_page()

    # Create sidebar with configuration
    config = create_sidebar()

    # Handle file uploads
    upload_result = handle_file_uploads()
    upload_type = upload_result[0]

    # Initialize session state for file processing
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

                # Show file summary
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

                # Validate MTL textures if MTL exists
                mtl_path = obj_path.replace('.obj', '.mtl') if obj_path else None
                if mtl_path and os.path.exists(mtl_path):
                    is_valid, missing = validate_mtl_textures(mtl_path, temp_dir)
                    if not is_valid:
                        st.warning(f"⚠️ Missing texture files: {', '.join(missing)}")

                # Store file paths in session state
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

                # Show file summary
                st.subheader("📋 Processed Files")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**3D Object:**")
                    st.write(f"✅ {obj_file.name}")

                with col2:
                    st.write("**Texture:**")
                    if texture_files:
                        if hasattr(texture_files, 'read'):  # Single file
                            st.write(f"✅ {texture_files.name}")
                        else:  # Multiple files
                            for tex_file in texture_files:
                                st.write(f"✅ {tex_file.name}")
                    else:
                        st.write("💡 Will use MTL materials")

                with col3:
                    st.write("**Environment Maps:**")
                    for env_file in env_files:
                        st.write(f"✅ {env_file.name}")

                # Validate MTL textures if MTL was uploaded
                if mtl_file:
                    mtl_path = os.path.join(temp_dir, mtl_file.name)
                    is_valid, missing = validate_mtl_textures(mtl_path, temp_dir)
                    if not is_valid:
                        st.warning(f"⚠️ Missing texture files: {', '.join(missing)}")
                        st.info("💡 Please upload all texture files referenced in the MTL file")

                # Store file paths in session state
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

    # Show Run Optimization button if files are processed
    if st.session_state.get('files_processed', False):
        st.header("🚀 Run Analysis")

        # Show current configuration summary
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

        # Big run button
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

    # Always render
    if st.session_state.get('results') is not None:
        plot_data = visualize_results(
            st.session_state['results'],
            st.session_state.get('config', config)
        )
        download_results(st.session_state['results'], plot_data)

    # Show example configuration if no files processed
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
