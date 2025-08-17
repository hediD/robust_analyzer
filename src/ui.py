# -*- coding: utf-8 -*-
"""3D Adversarial Robustness Analyzer (Streamlit App)"""

from __future__ import annotations

import io
import json
import glob
import os
import hashlib
import tempfile
import zipfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import shutil
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

import plotly.express as px  # noqa: F401
from plotly.subplots import make_subplots  # noqa: F401

from model import Model
from robustness_analyzer import RobustnessAnalyzer
import utils

from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt

# Constants
APP_TITLE = "🎯 3D Adversarial Robustness Analyzer"
APP_TAGLINE = (
    "Upload 3D objects, textures, and environments to analyze adversarial robustness "
    "through camera position optimization and visualize results with polar heatmaps."
)
DEFAULT_TARGET_LABEL = (
    "tank, army tank, armored combat vehicle, armoured combat vehicle"
)
IMAGENET_JSON_REL = ("data", "imagenet1000_clsidx_to_labels.json")
IMAGENET_JSON_ALT_REL = ("..", "data", "imagenet1000_clsidx_to_labels.json")

# Utilities
def _safe_json_or_eval_text(text: str) -> Dict[int, str]:
    try:
        return json.loads(text)
    except Exception:
        return eval(text)


def _imagenet_labels_path() -> Optional[Path]:
    p1 = Path(*IMAGENET_JSON_REL)
    if p1.exists():
        return p1
    p2 = Path(*IMAGENET_JSON_ALT_REL)
    if p2.exists():
        return p2
    return None


def _expand_for_envmaps(
    logits: torch.Tensor,
    cam_positions_stack: torch.Tensor,
    envmap_paths: Optional[Sequence[str]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand logits and camera positions to account for environment maps."""
    if envmap_paths:
        n_env = len(envmap_paths)
        cams = cam_positions_stack.unsqueeze(2).expand(-1, -1, n_env, -1)
        logits_2d = logits.reshape(-1, 1000)
        cameras_2d = cams.reshape(-1, 3)
    else:
        logits_2d = logits.reshape(-1, 1000)
        cameras_2d = cam_positions_stack.reshape(-1, 3)
    return logits_2d, cameras_2d


def _now_iso() -> str:
    return datetime.now().isoformat()


def _now_human() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _get_idx_safe(target: str) -> int:
    return utils.get_idx(target)


def _softmax_max_probs(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()


def _pred_top1(logits: torch.Tensor) -> np.ndarray:
    return torch.argmax(logits, dim=1).cpu().numpy()


# UI Setup
def setup_page() -> None:
    st.set_page_config(
        page_title="3D Robustness Analyzer",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title(APP_TITLE)
    st.markdown(APP_TAGLINE)


# ImageNet Labeling & Target Selection
def load_imagenet_labels() -> Tuple[List[Tuple[int, str]], Dict[int, str]]:
    try:
        labels_file = _imagenet_labels_path()
        if not labels_file:
            raise FileNotFoundError("imagenet1000_clsidx_to_labels.json not found")

        text = labels_file.read_text()
        id_to_class: Dict[int, str] = _safe_json_or_eval_text(text)

        class_options = [(idx, label) for idx, label in id_to_class.items()]
        class_options.sort(key=lambda x: x[1].lower())
        return class_options, id_to_class

    except Exception as e:
        st.error(f"❌ Error loading ImageNet labels: {str(e)}")
        return [], {}


def get_cached_imagenet_labels() -> Tuple[List[Tuple[int, str]], Dict[int, str]]:
    if "imagenet_labels_cache" not in st.session_state:
        st.session_state["imagenet_labels_cache"] = load_imagenet_labels()
    return st.session_state["imagenet_labels_cache"]


def create_target_class_selector() -> str:
    st.sidebar.subheader("🎯 Target Class Selection")

    class_options, id_to_class = get_cached_imagenet_labels()
    if not class_options:
        st.sidebar.warning("⚠️ Could not load ImageNet labels. Using text input.")
        return st.sidebar.text_input(
            "Target Class",
            value=DEFAULT_TARGET_LABEL,
            help="ImageNet class name for the target",
        )

    display_options = [f"{idx}: {label}" for idx, label in class_options]

    # Default to tank class
    default_idx = 0
    for i, (_idx, label) in enumerate(class_options):
        if "tank" in label.lower() and "army" in label.lower():
            default_idx = i
            break

    selected_display = st.sidebar.selectbox(
        "🔍 Search & Select ImageNet Class:",
        display_options,
        index=default_idx,
        help="Type to search through 1000 ImageNet classes, then select",
    )

    if selected_display:
        selected_idx = int(selected_display.split(":")[0])
        target_class = id_to_class[selected_idx]
        st.sidebar.success(f"✅ Selected: {target_class}")
        return target_class

    return DEFAULT_TARGET_LABEL


# Sidebar Config
def create_sidebar() -> Dict[str, Union[int, float, bool, str, List[str]]]:
    st.sidebar.header("📋 Configuration")
    st.sidebar.subheader("Model Parameters")

    target_class = create_target_class_selector()

    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        help="Number of viewpoints to optimize in parallel",
    )

    st.sidebar.subheader("Optimization Parameters")
    params_to_optimize = st.sidebar.multiselect(
        "Parameters to Optimize",
        options=["camera", "texture", "lighting"],
        default=["camera"],
        help="Select which parameters to optimize during adversarial attack",
    )

    num_runs = st.sidebar.number_input(
        "Number of Runs",
        min_value=1,
        max_value=20_000,
        value=1,
        step=1,
        help="Total number of optimization runs",
    )

    num_iterations = st.sidebar.number_input(
        "Adversarial Optimization Steps",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        help="Number of optimization steps per adversarial run",
    )

    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        value=5e-3,
        format_func=lambda x: f"{x:.1e}",
    )

    st.sidebar.subheader("Constraints")
    positive_z = st.sidebar.checkbox(
        "Positive Z Constraint",
        value=True,
        help="Constrain camera to positive elevation (z > 0)",
    )
    targeted = st.sidebar.checkbox(
        "Targeted Attack",
        value=False,
        help="Whether this is a targeted adversarial attack",
    )

    with st.sidebar.expander("Advanced Rendering Settings", expanded=False):
        st.write("**Rendering Paramters**")
        image_size = st.select_slider(
            "Image Size",
            options=[224, 256, 320, 384, 448, 512],
            value=448,
            help="Resolution of rendered images (image_size x image_size)",
        )
        bin_size = st.slider(
            "Bin Size",
            min_value=16,
            max_value=64,
            value=32,
            help="Spatial partitioning for rasterization - larger values use less memory but may be slower",
        )
        max_faces_per_bin = st.number_input(
            "Max Faces per Bin",
            min_value=10000,
            max_value=200000,
            value=100000,
            step=10000,
            help="Maximum faces per spatial bin - increase for complex meshes, decrease to save memory",
        )
        st.info("💡 **Tip:** Use smaller bin sizes and fewer faces per bin if you encounter GPU memory issues.")

    st.sidebar.subheader("Download Settings")
    include_heatmap = st.sidebar.checkbox(
        "📊 Include heatmap in downloads",
        value=True,
        help="When checked: downloads include rendered image + heatmap composite. When unchecked: downloads only the rendered images with separate metadata files.",
        key="global_include_heatmap"
    )
    st.session_state["include_heatmap_global"] = include_heatmap

    return {
        "target_class": target_class,
        "batch_size": int(batch_size),
        "params_to_optimize": list(params_to_optimize),
        "num_runs": int(num_runs),
        "num_iterations": int(num_iterations),
        "learning_rate": float(learning_rate),
        "image_size": int(image_size),
        "bin_size": int(bin_size),
        "max_faces_per_bin": int(max_faces_per_bin),
        "positive_z": bool(positive_z),
        "targeted": bool(targeted),
        "include_heatmap": include_heatmap,
    }


# File Uploads & Cache
def handle_file_uploads():
    st.header("📁 File Uploads")
    tab1, tab2 = st.tabs(["📦 Complete 3D Package", "🎨 Individual Files"])

    with tab1:
        st.subheader("Upload Complete 3D Package")
        st.info("💡 Upload a ZIP file containing .obj, .mtl, and all texture files")
        zip_file = st.file_uploader(
            "Upload ZIP package",
            type=["zip"],
            help="ZIP file containing .obj, .mtl, and texture files with correct folder structure",
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
                type=["obj"],
                help="Upload the 3D mesh file (.obj format)",
            )
            mtl_file = st.file_uploader(
                "Upload MTL file (optional)",
                type=["mtl"],
                help="Upload material definition file (.mtl format)",
            )
            if obj_file:
                st.success(f"✅ OBJ: {obj_file.name}")
            if mtl_file:
                st.success(f"✅ MTL: {mtl_file.name}")

        with col2:
            st.subheader("Textures")
            single_texture = st.file_uploader(
                "Single texture override",
                type=["png", "jpg", "jpeg"],
                help="Single texture to override MTL materials (optional)",
            )
            texture_files = st.file_uploader(
                "Multiple texture files",
                type=["png", "jpg", "jpeg", "bmp", "tga"],
                accept_multiple_files=True,
                help="Upload all texture files referenced in the MTL file",
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
                type=["png", "jpg", "jpeg", "hdr", "exr"],
                accept_multiple_files=True,
                help="Upload one or more environment map files",
            )
            if env_files:
                st.success(f"✅ Uploaded {len(env_files)} environment map(s)")
                for env_file in env_files:
                    st.text(f"  • {env_file.name}")

        return "individual", obj_file, mtl_file, (single_texture or texture_files), env_files


def get_cache_dir() -> Path:
    script_dir = Path(__file__).parent.absolute()
    cache_dir = script_dir / "file_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()


def _cache_entries_sorted_latest_first(cache_dir: Path) -> List[Dict]:
    entries: List[Dict] = []
    for sub in cache_dir.iterdir():
        if not sub.is_dir():
            continue
        meta = sub / "metadata.json"
        if not meta.exists():
            continue
        try:
            cached = json.loads(meta.read_text())
            req = [cached.get("obj_path"), cached.get("texture_path")] + cached.get("envmap_paths", [])
            if any(p and not Path(p).exists() for p in req):
                continue

            ts_raw = cached.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_raw.strip('"'))
            except Exception:
                ts = datetime.min

            entries.append({"metadata": cached, "timestamp": ts, "cache_key": sub.name})
        except Exception:
            continue

    entries.sort(key=lambda x: x["timestamp"], reverse=True)
    return entries


def check_cached_files() -> bool:
    cache_dir = get_cache_dir()
    entries = _cache_entries_sorted_latest_first(cache_dir)
    if not entries:
        return False

    latest = entries[0]
    cached_metadata = latest["metadata"]

    st.session_state["files_processed"] = True
    st.session_state["file_paths"] = cached_metadata
    st.session_state["using_cached_files"] = True
    st.session_state["cache_info"] = {
        "cache_key": latest["cache_key"][:8],
        "timestamp": latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        "file_names": cached_metadata.get("file_names", ["Unknown files"]),
    }
    print(f"✅ Loaded cached files from {latest['cache_key'][:8]}...")
    return True


def save_zip_package(zip_file, temp_dir: str) -> Tuple[str, Optional[str], List[str], str]:
    with zipfile.ZipFile(io.BytesIO(zip_file.read())) as zf:
        zf.extractall(temp_dir)

    obj_files = glob.glob(os.path.join(temp_dir, "**/*.obj"), recursive=True)
    if not obj_files:
        raise ValueError("No .obj file found in ZIP package")

    obj_path = obj_files[0]
    mtl_path: Optional[str] = None
    obj_dir = os.path.dirname(obj_path)

    # Find MTL referenced in OBJ
    with open(obj_path, "r") as f:
        for line in f:
            if line.strip().startswith("mtllib "):
                mtl_filename = line.strip().split("mtllib ")[1]
                mtl_path = os.path.join(obj_dir, mtl_filename)
                if not os.path.exists(mtl_path):
                    mtl_files = glob.glob(os.path.join(obj_dir, "*.mtl"))
                    mtl_path = mtl_files[0] if mtl_files else None
                break

    # Find textures referenced in MTL
    texture_paths: List[str] = []
    if mtl_path and os.path.exists(mtl_path):
        mtl_dir = os.path.dirname(mtl_path)
        with open(mtl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("map_"):
                    parts = line.split()
                    if len(parts) >= 2:
                        texture_name = parts[-1]
                        tex_path = os.path.join(mtl_dir, texture_name)
                        if not os.path.exists(tex_path):
                            tex_path = os.path.join(temp_dir, texture_name)
                        if os.path.exists(tex_path):
                            texture_paths.append(tex_path)

    # Find env maps
    envmap_paths: List[str] = []
    for ext in ("*.hdr", "*.exr"):
        envmap_paths.extend(glob.glob(os.path.join(temp_dir, "**", ext), recursive=True))
    if not envmap_paths:
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            cands = glob.glob(os.path.join(temp_dir, "**", ext), recursive=True)
            envmap_paths.extend([p for p in cands if ("env" in p.lower() or "hdri" in p.lower()) and p not in texture_paths])

    texture_path = texture_paths[0] if texture_paths else None
    return obj_path, texture_path, envmap_paths, temp_dir


def update_obj_mtl_reference(obj_path: str, mtl_filename: str) -> None:
    with open(obj_path, "r") as f:
        lines = f.readlines()

    if not any(line.strip().startswith("mtllib ") for line in lines):
        lines.insert(0, f"mtllib {mtl_filename}\n")
        with open(obj_path, "w") as f:
            f.writelines(lines)


def save_individual_files(
    obj_file, mtl_file, texture_files, env_files, temp_dir: str
) -> Tuple[Optional[str], Optional[str], List[str], str]:
    obj_path: Optional[str] = None
    texture_path: Optional[str] = None
    envmap_paths: List[str] = []

    if obj_file:
        obj_path = os.path.join(temp_dir, obj_file.name)
        with open(obj_path, "wb") as f:
            f.write(obj_file.read())

    if mtl_file:
        mtl_path = os.path.join(temp_dir, mtl_file.name)
        with open(mtl_path, "wb") as f:
            f.write(mtl_file.read())
        if obj_path:
            update_obj_mtl_reference(obj_path, mtl_file.name)

    if texture_files:
        if hasattr(texture_files, "read"):
            texture_path = os.path.join(temp_dir, texture_files.name)
            with open(texture_path, "wb") as f:
                f.write(texture_files.read())
        else:
            for tex_file in texture_files:
                tex_path = os.path.join(temp_dir, tex_file.name)
                with open(tex_path, "wb") as f:
                    f.write(tex_file.read())

    if env_files:
        for env_file in env_files:
            env_path = os.path.join(temp_dir, env_file.name)
            with open(env_path, "wb") as f:
                f.write(env_file.read())
            envmap_paths.append(env_path)

    return obj_path, texture_path, envmap_paths, temp_dir


def validate_mtl_textures(mtl_path: str, temp_dir: str) -> Tuple[bool, List[str]]:
    if not mtl_path or not os.path.exists(mtl_path):
        return True, []

    missing: List[str] = []
    mtl_dir = os.path.dirname(mtl_path)
    with open(mtl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("map_"):
                parts = line.split()
                if len(parts) >= 2:
                    texture_name = parts[-1]
                    tex_path = os.path.join(mtl_dir, texture_name)
                    if not os.path.exists(tex_path):
                        tex_path = os.path.join(temp_dir, texture_name)
                    if not os.path.exists(tex_path):
                        missing.append(texture_name)
    return (len(missing) == 0), missing


def save_files_to_cache(upload_type, *files):
    cache_dir = get_cache_dir()
    file_names: List[str] = []

    if upload_type == "zip":
        zip_file = files[0]
        content = zip_file.read()
        zip_file.seek(0)
        cache_key = get_file_hash(content)
        file_names = [zip_file.name]
    else:
        obj_file, mtl_file, texture_files, env_files = files
        blobs: List[bytes] = []
        if obj_file:
            c = obj_file.read(); obj_file.seek(0)
            blobs.append(c); file_names.append(obj_file.name)
        if mtl_file:
            c = mtl_file.read(); mtl_file.seek(0)
            blobs.append(c); file_names.append(mtl_file.name)
        if texture_files:
            if hasattr(texture_files, "read"):
                c = texture_files.read(); texture_files.seek(0)
                blobs.append(c); file_names.append(texture_files.name)
            else:
                for tf in texture_files:
                    c = tf.read(); tf.seek(0)
                    blobs.append(c); file_names.append(tf.name)
        if env_files:
            for ef in env_files:
                c = ef.read(); ef.seek(0)
                blobs.append(c); file_names.append(ef.name)
        cache_key = get_file_hash(b"".join(blobs))

    upload_cache = cache_dir / cache_key
    meta_path = upload_cache / "metadata.json"

    # Use cached data if available
    if meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text())
            req = [cached.get("obj_path"), cached.get("texture_path")] + cached.get("envmap_paths", [])
            if all((not p) or Path(p).exists() for p in req):
                st.info(f"📦 Using cached files (hash: {cache_key[:8]}...)")
                return cached["obj_path"], cached["texture_path"], cached["envmap_paths"], str(upload_cache)
        except Exception:
            pass

    upload_cache.mkdir(exist_ok=True)
    st.info(f"💾 Caching files for future use (hash: {cache_key[:8]}...)")

    try:
        if upload_type == "zip":
            obj_path, texture_path, envmap_paths, _ = save_zip_package(files[0], str(upload_cache))
        else:
            obj_path, texture_path, envmap_paths, _ = save_individual_files(*files, str(upload_cache))

        meta = {
            "obj_path": obj_path,
            "texture_path": texture_path,
            "envmap_paths": envmap_paths,
            "temp_dir": str(upload_cache),
            "cache_key": cache_key,
            "file_names": file_names,
            "upload_type": upload_type,
            "timestamp": _now_iso(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"✅ Files cached successfully to {cache_key[:8]}...")
        return obj_path, texture_path, envmap_paths, str(upload_cache)

    except Exception as e:
        st.error(f"❌ Error caching files: {str(e)}")
        if upload_cache.exists():
            shutil.rmtree(upload_cache, ignore_errors=True)
        raise


# Visualization
def create_interactive_polar_plot(
    camera_positions: np.ndarray,
    labels_correct: np.ndarray,
    logits: torch.Tensor,
    target_class: str,
    topk_value: int,
    selected_indices: Optional[Sequence[int]] = None,
    current_highlighted_index: Optional[int] = None,
) -> go.Figure:
    azimuth, elevation, _ = utils.compute_spherical_coordinates(camera_positions)
    azimuth_rad = np.radians(azimuth)

    _, id_to_class = get_cached_imagenet_labels()
    preds_top1 = _pred_top1(logits)
    pred_probs = _softmax_max_probs(logits)

    # Hover text
    hover_text: List[str] = []
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
            "Position {i}<br>"
            "Azimuth: {az:.1f}°<br>"
            "Elevation: {el:.1f}°<br>"
            "Prediction: {pc:.30}...<br>"
            "Confidence: {p:.3f}<br>"
            "Status: {st}<br>"
            "{sel}<br>"
            "Click to view rendered image".format(
                i=i, az=azimuth[i], el=elevation[i], pc=pred_class, p=pred_probs[i], st=status, sel=selection_status
            )
        )

    fig = go.Figure()

    # Masks
    selected_mask = np.isin(np.arange(len(camera_positions)), selected_indices) if selected_indices else np.zeros(len(camera_positions), dtype=bool)
    current_mask = (np.arange(len(camera_positions)) == current_highlighted_index) if current_highlighted_index is not None else np.zeros(len(camera_positions), dtype=bool)
    unselected_mask = ~selected_mask
    selected_not_current_mask = selected_mask & ~current_mask
    correct_mask = labels_correct.astype(bool)
    incorrect_mask = ~correct_mask

    # Unselected correct
    unselected_correct = correct_mask & unselected_mask
    if unselected_correct.any():
        fig.add_trace(go.Scatterpolar(
            r=elevation[unselected_correct],
            theta=azimuth[unselected_correct],
            mode="markers",
            marker=dict(size=6, color="lightblue", opacity=0.4, line=dict(width=1, color="darkblue")),
            name=f"Correct (Top-{topk_value})",
            hovertext=[hover_text[i] for i in np.where(unselected_correct)[0]],
            hoverinfo="text",
            customdata=np.where(unselected_correct)[0],
        ))

    # Unselected incorrect
    unselected_incorrect = incorrect_mask & unselected_mask
    if unselected_incorrect.any():
        fig.add_trace(go.Scatterpolar(
            r=elevation[unselected_incorrect],
            theta=azimuth[unselected_incorrect],
            mode="markers",
            marker=dict(size=6, color="lightcoral", opacity=0.4, line=dict(width=1, color="darkred")),
            name=f"Incorrect (Top-{topk_value})",
            hovertext=[hover_text[i] for i in np.where(unselected_incorrect)[0]],
            hoverinfo="text",
            customdata=np.where(unselected_incorrect)[0],
        ))

    # Selected correct (not current)
    selected_correct_not_current = correct_mask & selected_not_current_mask
    if selected_correct_not_current.any():
        fig.add_trace(go.Scatterpolar(
            r=elevation[selected_correct_not_current],
            theta=azimuth[selected_correct_not_current],
            mode="markers",
            marker=dict(size=7, color="darkblue", opacity=0.8, line=dict(width=2, color="navy")),
            name="📍 Selected Correct",
            hovertext=[hover_text[i] for i in np.where(selected_correct_not_current)[0]],
            hoverinfo="text",
            customdata=np.where(selected_correct_not_current)[0],
        ))

    # Selected incorrect (not current)
    selected_incorrect_not_current = incorrect_mask & selected_not_current_mask
    if selected_incorrect_not_current.any():
        fig.add_trace(go.Scatterpolar(
            r=elevation[selected_incorrect_not_current],
            theta=azimuth[selected_incorrect_not_current],
            mode="markers",
            marker=dict(size=7, color="darkred", opacity=0.8, line=dict(width=2, color="maroon")),
            name="📍 Selected Incorrect",
            hovertext=[hover_text[i] for i in np.where(selected_incorrect_not_current)[0]],
            hoverinfo="text",
            customdata=np.where(selected_incorrect_not_current)[0],
        ))

    # Current
    current_correct = correct_mask & current_mask
    if current_correct.any():
        fig.add_trace(go.Scatterpolar(
            r=elevation[current_correct],
            theta=azimuth[current_correct],
            mode="markers",
            marker=dict(size=14, color="darkblue", opacity=1.0, line=dict(width=4, color="navy")),
            name="🎯 Currently Displayed Correct",
            hovertext=[hover_text[i] for i in np.where(current_correct)[0]],
            hoverinfo="text",
            customdata=np.where(current_correct)[0],
        ))

    current_incorrect = incorrect_mask & current_mask
    if current_incorrect.any():
        fig.add_trace(go.Scatterpolar(
            r=elevation[current_incorrect],
            theta=azimuth[current_incorrect],
            mode="markers",
            marker=dict(size=14, color="darkred", opacity=1.0, line=dict(width=4, color="maroon")),
            name="🎯 Currently Displayed Incorrect",
            hovertext=[hover_text[i] for i in np.where(current_incorrect)[0]],
            hoverinfo="text",
            customdata=np.where(current_incorrect)[0],
        ))

    title_text = f"Interactive Camera Position Analysis (Top-{topk_value})<br>Class: {target_class[:50]}..."
    if current_highlighted_index is not None:
        title_text += f"<br>🎯 Currently displaying position {current_highlighted_index}"
    elif selected_indices:
        title_text += f"<br>📍 {len(selected_indices)} positions selected"

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 90], ticksuffix="°", title="Elevation"),
            angularaxis=dict(ticksuffix="°", rotation=90, direction="clockwise"),
        ),
        title=title_text,
        showlegend=True,
        height=600,
        font=dict(size=12),
    )
    return fig


def render_image_at_position(
    robust_analyzer: RobustnessAnalyzer,
    results: Dict,
    position_idx: int,
    envmap_idx: int = 0,
) -> Optional[Dict]:
    try:
        total_envmaps = len(results.get("envmap_paths", [])) or 1

        # Structure: (num_runs * batch_size * n_envmaps)
        positions_per_run = robust_analyzer.batch_size * total_envmaps
        run_idx = position_idx // positions_per_run
        remaining = position_idx % positions_per_run
        batch_idx = remaining // total_envmaps
        env_idx = remaining % total_envmaps

        run_idx = min(run_idx, len(results["final_scene_params"]) - 1)
        batch_idx = min(batch_idx, robust_analyzer.batch_size - 1)
        env_idx = min(env_idx, total_envmaps - 1)

        # Update model with scene parameters
        scene_params = results["final_scene_params"][run_idx]
        robust_analyzer.model.update_scene_params(scene_params)

        with torch.no_grad():
            render_images = robust_analyzer.model.render(with_grad=False)

            if len(render_images.shape) == 5:  # [batch, env, H, W, C]
                image = render_images[batch_idx, env_idx]
            else:  # [batch, H, W, C]
                image = render_images[batch_idx]

            image_np = utils.to_numpy(image)
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)

            stored_logits = torch.stack(results["final_logits"])
            if len(stored_logits.shape) == 4:  # [runs, batch, env, classes]
                pred_logits = stored_logits[run_idx, batch_idx, env_idx]
            else:  # [runs, batch, classes]
                pred_logits = stored_logits[run_idx, batch_idx]

            pred_class_idx = int(pred_logits.argmax().item())
            pred_prob = float(torch.softmax(pred_logits, dim=0)[pred_class_idx].item())

            # Calculate target class ranking
            target_class_ranking = None
            target_class_confidence = None
            if st.session_state.get("target_class"):
                target_idx = _get_idx_safe(st.session_state["target_class"])
                sorted_indices = torch.argsort(pred_logits, descending=True)
                target_class_ranking = int((sorted_indices == target_idx).nonzero(as_tuple=True)[0].item()) + 1
                target_class_confidence = float(torch.softmax(pred_logits, dim=0)[target_idx].item())

            _, id_to_class = get_cached_imagenet_labels()
            pred_class = id_to_class.get(pred_class_idx, f"class {pred_class_idx}")

            camera_pos = scene_params["camera"][batch_idx:batch_idx + 1]
            azimuth, elevation, distance = utils.compute_spherical_coordinates(camera_pos.cpu().numpy())

            return {
                "image": image_np,
                "prediction": pred_class,
                "confidence": pred_prob,
                "class_idx": pred_class_idx,
                "azimuth": float(azimuth[0]),
                "elevation": float(elevation[0]),
                "distance": float(distance[0]),
                "position_idx": position_idx,
                "run_idx": run_idx,
                "batch_idx": batch_idx,
                "env_idx": env_idx,
                "target_class_ranking": target_class_ranking,
                "target_class_confidence": target_class_confidence,
            }

    except Exception as e:
        st.error(f"Error rendering image at position {position_idx}: {str(e)}")
    return None


def render_multiple_images(
    robust_analyzer: RobustnessAnalyzer,
    results: Dict,
    position_indices: Sequence[int],
) -> List[Dict]:
    if not position_indices:
        return []

    rendered_images: List[Dict] = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, position_idx in enumerate(position_indices):
        progress_bar.progress((i + 1) / len(position_indices))
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


# Composite Image Creation
def create_individual_heatmap(
    camera_positions: np.ndarray,
    labels_correct: np.ndarray,
    logits: torch.Tensor,
    target_class: str,
    topk_value: int,
    highlighted_index: int,
    config: Dict,
    image_info: Optional[Dict] = None,
) -> go.Figure:
    fig = create_interactive_polar_plot(
        camera_positions,
        labels_correct,
        logits,
        target_class,
        topk_value,
        selected_indices=None,
        current_highlighted_index=highlighted_index,
    )

    azimuth, elevation, _ = utils.compute_spherical_coordinates(camera_positions)
    title_parts = [f"Azimuth: {azimuth[highlighted_index]:.1f}°, Elevation: {elevation[highlighted_index]:.1f}°"]

    if image_info:
        title_parts.append(f"Predicted: {image_info['prediction'][:25]}...")
        title_parts.append(f"Confidence: {image_info['confidence']:.3f}")
    title_parts.append(f"Target: {target_class[:30]}...")

    fig.update_layout(title="<br>".join(title_parts), height=500, width=500)
    return fig


def create_composite_image(
    image_info: Dict,
    camera_positions: np.ndarray,
    labels_correct: np.ndarray,
    logits: torch.Tensor,
    config: Dict,
) -> Image.Image:
    rendered_img = Image.fromarray(image_info["image"])

    topk_value = st.session_state.get("topk_value", 1)
    heatmap_img: Optional[Image.Image] = None

    # Try Plotly export first
    try:
        fig = create_individual_heatmap(
            camera_positions,
            labels_correct,
            logits,
            config["target_class"],
            topk_value,
            image_info["position_idx"],
            config,
            image_info,
        )
        heatmap_bytes = fig.to_image(format="png", width=500, height=500, scale=2)
        heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
        print("✅ Plotly heatmap created successfully")

    except Exception as e:
        print(f"❌ Plotly heatmap failed: {e}")
        # Matplotlib fallback
        try:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="polar")

            azimuth, elevation, _ = utils.compute_spherical_coordinates(camera_positions)
            azimuth_rad = np.radians(azimuth)

            correct_mask = labels_correct.astype(bool)
            incorrect_mask = ~correct_mask

            if incorrect_mask.any():
                ax.scatter(
                    azimuth_rad[incorrect_mask], elevation[incorrect_mask],
                    c="lightcoral", s=30, alpha=0.6, label=f"Incorrect (Top-{topk_value})",
                )
            if correct_mask.any():
                ax.scatter(
                    azimuth_rad[correct_mask], elevation[correct_mask],
                    c="lightblue", s=30, alpha=0.6, label=f"Correct (Top-{topk_value})",
                )

            hi = image_info["position_idx"]
            highlighted_azimuth = azimuth_rad[hi]
            highlighted_elevation = elevation[hi]
            is_correct = bool(labels_correct[hi])
            color, edge_color = ("darkblue", "navy") if is_correct else ("darkred", "maroon")

            ax.scatter(highlighted_azimuth, highlighted_elevation, c=color, s=150, alpha=1.0, edgecolors=edge_color, linewidth=3)
            ax.set_ylim(0, 90)
            ax.set_theta_direction(-1)
            ax.set_theta_zero_location("N")

            title_lines = [
                f'Azimuth: {azimuth[hi]:.1f}°, Elevation: {elevation[hi]:.1f}°',
                f'Predicted: {image_info["prediction"][:25]}... (Conf: {image_info["confidence"]:.3f})',
            ]
            ax.set_title("\n".join(title_lines), pad=20, fontsize=9)
            ax.legend(loc="upper left", bbox_to_anchor=(0, 1), fontsize="small")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            heatmap_img = Image.open(buf)
            plt.close(fig)

        except Exception as e2:
            print(f"Matplotlib fallback also failed: {e2}")
            # Last resort placeholder
            heatmap_img = Image.new("RGB", (500, 500), color="lightgray")
            draw = ImageDraw.Draw(heatmap_img)

            # Find a usable font
            font = None
            for fp in [
                "C:/Windows/Fonts/arial.ttf",
                "/System/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "arial.ttf",
            ]:
                try:
                    font = ImageFont.truetype(fp, 14)
                    break
                except Exception:
                    continue
            if font is None:
                font = ImageFont.load_default()

            azimuth, elevation, _ = utils.compute_spherical_coordinates(camera_positions)
            msg = (
                "Heatmap unavailable\n(Kaleido not installed)\n\n"
                f"Position: {image_info['position_idx']}\n"
                f"Azimuth: {azimuth[image_info['position_idx']]:.1f}°\n"
                f"Elevation: {elevation[image_info['position_idx']]:.1f}°\n"
                f"Predicted: {image_info['prediction'][:20]}...\n"
                f"Confidence: {image_info['confidence']:.3f}"
            )
            bbox = draw.textbbox((0, 0), msg, font=font)
            x = (500 - (bbox[2] - bbox[0])) // 2
            y = (500 - (bbox[3] - bbox[1])) // 2
            draw.text((x, y), msg, fill="black", font=font, align="center")

    # Standardize sizes and compose
    rendered_size = (400, 400)
    heatmap_size = (400, 400)

    rendered_img = rendered_img.resize(rendered_size, Image.Resampling.LANCZOS)
    heatmap_img = heatmap_img.resize(heatmap_size, Image.Resampling.LANCZOS)

    margin = 20
    total_w = rendered_size[0] + heatmap_size[0] + 3 * margin
    total_h = max(rendered_size[1], heatmap_size[1]) + 2 * margin

    composite = Image.new("RGB", (total_w, total_h), color="white")
    composite.paste(rendered_img, (margin, margin))
    composite.paste(heatmap_img, (margin + rendered_size[0] + margin, margin))

    draw = ImageDraw.Draw(composite)
    label = None
    for fp in [
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
    ]:
        try:
            label = ImageFont.truetype(fp, 14)
            break
        except Exception:
            continue
    if label is None:
        try:
            label = ImageFont.load_default()
        except Exception:
            label = None

    label_y = margin + max(rendered_size[1], heatmap_size[1]) + 5
    if label:
        r_text, h_text = "Rendered Image", "Map"
        r_w = draw.textbbox((0, 0), r_text, font=label)[2]
        h_w = draw.textbbox((0, 0), h_text, font=label)[2]
        r_x = margin + (rendered_size[0] - r_w) // 2
        h_x = margin + rendered_size[0] + margin + (heatmap_size[0] - h_w) // 2
        draw.text((r_x, label_y), r_text, fill="black", font=label)
        draw.text((h_x, label_y), h_text, fill="black", font=label)

    return composite


# Downloads
def download_single_image_package(
    image_info: Dict,
    camera_positions: np.ndarray,
    labels_correct: np.ndarray,
    logits: torch.Tensor,
    config: Dict,
) -> bytes:
    with tempfile.TemporaryDirectory() as temp_dir:
        pkg_dir = Path(temp_dir) / f"robustness_image_{image_info['position_idx']}"
        pkg_dir.mkdir(exist_ok=True)

        composite_img = create_composite_image(image_info, camera_positions, labels_correct, logits, config)
        composite_path = pkg_dir / f"position_{image_info['position_idx']}_analysis.png"
        composite_img.save(composite_path, format="PNG", dpi=(300, 300))

        json_metadata = {
            "position_idx": image_info["position_idx"],
            "azimuth": float(image_info["azimuth"]),
            "elevation": float(image_info["elevation"]),
            "distance": float(image_info["distance"]),
            "prediction": image_info["prediction"],
            "confidence": float(image_info["confidence"]),
            "class_idx": int(image_info["class_idx"]),
            "run_idx": image_info["run_idx"],
            "batch_idx": image_info["batch_idx"],
            "env_idx": image_info["env_idx"],
            "target_class": config.get("target_class", ""),
            "is_correct": image_info["class_idx"] == _get_idx_safe(config.get("target_class", "")),
            "analysis_config": {
                "batch_size": config.get("batch_size"),
                "params_to_optimize": config.get("params_to_optimize"),
                "num_runs": config.get("num_runs"),
                "num_iterations": config.get("num_iterations"),
                "learning_rate": config.get("learning_rate"),
                "image_size": config.get("image_size"),
            },
            "timestamp": _now_iso(),
        }
        (pkg_dir / f"position_{image_info['position_idx']}_metadata.json").write_text(json.dumps(json_metadata, indent=2))

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fp in pkg_dir.rglob("*"):
                if fp.is_file():
                    zipf.write(fp, fp.relative_to(pkg_dir))
        buf.seek(0)
        return buf.getvalue()


def download_all_images_package(
    rendered_images: Sequence[Dict],
    camera_positions: np.ndarray,
    labels_correct: np.ndarray,
    logits: torch.Tensor,
    config: Dict,
) -> bytes:
    with tempfile.TemporaryDirectory() as temp_dir:
        pkg_dir = Path(temp_dir) / "robustness_analysis_all_images"
        pkg_dir.mkdir(exist_ok=True)

        summary = {
            "total_images": len(rendered_images),
            "target_class": config.get("target_class", ""),
            "analysis_config": {
                "batch_size": config.get("batch_size"),
                "params_to_optimize": config.get("params_to_optimize"),
                "num_runs": config.get("num_runs"),
                "num_iterations": config.get("num_iterations"),
                "learning_rate": config.get("learning_rate"),
                "image_size": config.get("image_size"),
            },
            "images": [],
            "timestamp": _now_iso(),
        }

        target_idx = _get_idx_safe(config.get("target_class", ""))

        for image_info in rendered_images:
            composite_img = create_composite_image(image_info, camera_positions, labels_correct, logits, config)
            out_path = pkg_dir / f"position_{image_info['position_idx']:03d}_analysis.png"
            composite_img.save(out_path, format="PNG", dpi=(300, 300))
            summary["images"].append({
                "position_idx": image_info["position_idx"],
                "azimuth": float(image_info["azimuth"]),
                "elevation": float(image_info["elevation"]),
                "distance": float(image_info["distance"]),
                "prediction": image_info["prediction"],
                "confidence": float(image_info["confidence"]),
                "is_correct": (image_info["class_idx"] == target_idx),
                "filename": out_path.name,
            })

        (pkg_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))

        # Overall heatmap with all positions highlighted
        all_positions = [img["position_idx"] for img in rendered_images]
        overall_fig = create_interactive_polar_plot(
            camera_positions,
            labels_correct,
            logits,
            config["target_class"],
            st.session_state.get("topk_value", 1),
            selected_indices=all_positions,
            current_highlighted_index=None,
        )
        overall_fig.update_layout(
            title=f"All Rendered Positions Overview<br>Target: {config['target_class'][:30]}...<br>{len(rendered_images)} positions analyzed",
            height=800,
            width=800,
        )

        try:
            overall_fig.write_image(str(pkg_dir / "overview_heatmap.png"), width=800, height=800, scale=2)
        except Exception:
            overall_fig.write_html(str(pkg_dir / "overview_heatmap.html"))

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fp in pkg_dir.rglob("*"):
                if fp.is_file():
                    zipf.write(fp, fp.relative_to(pkg_dir))
        buf.seek(0)
        return buf.getvalue()


# Carousel
def create_image_carousel(rendered_images: Sequence[Dict]) -> None:
    if not rendered_images:
        st.info("No images to display")
        return

    st.session_state.setdefault("carousel_index", 0)
    if st.session_state["carousel_index"] >= len(rendered_images):
        st.session_state["carousel_index"] = 0

    idx = st.session_state["carousel_index"]
    current = rendered_images[idx]

    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀️ Previous", key=f"prev_image_{len(rendered_images)}", use_container_width=True):
            st.session_state["carousel_index"] = (idx - 1) % len(rendered_images)
            st.rerun()
    with col_info:
        st.markdown(f"<div style='text-align: center'><h4>Image {idx + 1} of {len(rendered_images)}</h4></div>", unsafe_allow_html=True)
    with col_next:
        if st.button("Next ▶️", key=f"next_image_{len(rendered_images)}", use_container_width=True):
            st.session_state["carousel_index"] = (idx + 1) % len(rendered_images)
            st.rerun()

    st.markdown("---")
    d1, d2, _, d4, _ = st.columns([1, 2, 0.5, 2, 1])

    include_heatmap = st.session_state.get("include_heatmap_global", True)

    with d2:
        if ("results" in st.session_state and "config" in st.session_state):
            try:
                results = st.session_state["results"]
                config = st.session_state["config"]

                logits = torch.stack(results["final_logits"])
                cameras = torch.stack([x["camera"] for x in results["final_scene_params"]])
                logits2d, cams2d = _expand_for_envmaps(logits, cameras, results.get("envmap_paths"))

                topk = st.session_state.get("topk_value", 1)
                labels_correct = utils.get_labels_correct(logits2d, config["target_class"], topk=topk)

                if include_heatmap:
                    composite = create_composite_image(
                        current, cams2d.numpy(), labels_correct.numpy(), logits2d, config
                    )
                    buf = io.BytesIO()
                    composite.save(buf, format="PNG", dpi=(300, 300))
                    buf.seek(0)
                    file_name = f"robustness_position_{current['position_idx']}_analysis.png"
                    button_label = "📥 Download Current Image"
                else:
                    rendered_img = Image.fromarray(current["image"])
                    buf = io.BytesIO()
                    rendered_img.save(buf, format="PNG", dpi=(300, 300))
                    buf.seek(0)
                    file_name = f"robustness_position_{current['position_idx']}_rendered.png"
                    button_label = "📥 Download Current Image"

                st.download_button(
                    label=button_label,
                    data=buf.getvalue(),
                    file_name=file_name,
                    mime="image/png",
                    key=f"dl_btn_single_{idx}_{include_heatmap}",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"❌ Error creating image: {str(e)}")
                st.code(traceback.format_exc())
        else:
            st.button("📥 Download Current Image", disabled=True, help="Analysis data not available", use_container_width=True)

    with d4:
        if ("results" in st.session_state and "config" in st.session_state and rendered_images):
            cache_key = f"carousel_zip_{len(rendered_images)}_{hash(tuple(img['position_idx'] for img in rendered_images))}_{include_heatmap}"

            if cache_key not in st.session_state:
                try:
                    with st.spinner("Preparing download..."):
                        results = st.session_state["results"]
                        config = st.session_state["config"]

                        logits = torch.stack(results["final_logits"])
                        cameras = torch.stack([x["camera"] for x in results["final_scene_params"]])
                        logits2d, cams2d = _expand_for_envmaps(logits, cameras, results.get("envmap_paths"))

                        topk = st.session_state.get("topk_value", 1)
                        labels_correct = utils.get_labels_correct(logits2d, config["target_class"], topk=topk)

                        zbuf = io.BytesIO()
                        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zipf:
                            for image_info in rendered_images:
                                if include_heatmap:
                                    composite = create_composite_image(
                                        image_info, cams2d.numpy(), labels_correct.numpy(), logits2d, config
                                    )
                                    ibuf = io.BytesIO()
                                    composite.save(ibuf, format="PNG", dpi=(300, 300))
                                    ibuf.seek(0)
                                    zipf.writestr(f"position_{image_info['position_idx']:03d}_analysis.png", ibuf.getvalue())
                                else:
                                    rendered_img = Image.fromarray(image_info["image"])
                                    ibuf = io.BytesIO()
                                    rendered_img.save(ibuf, format="PNG", dpi=(300, 300))
                                    ibuf.seek(0)
                                    zipf.writestr(f"position_{image_info['position_idx']:03d}_rendered.png", ibuf.getvalue())

                                    metadata = {
                                        "position_idx": image_info["position_idx"],
                                        "azimuth": float(image_info["azimuth"]),
                                        "elevation": float(image_info["elevation"]),
                                        "distance": float(image_info["distance"]),
                                        "prediction": image_info["prediction"],
                                        "confidence": float(image_info["confidence"]),
                                        "class_idx": int(image_info["class_idx"]),
                                        "run_idx": image_info["run_idx"],
                                        "batch_idx": image_info["batch_idx"],
                                        "env_idx": image_info["env_idx"],
                                        "target_class_ranking": image_info.get("target_class_ranking"),
                                        "target_class_confidence": image_info.get("target_class_confidence"),
                                    }
                                    zipf.writestr(f"position_{image_info['position_idx']:03d}_metadata.json",
                                                json.dumps(metadata, indent=2))

                        zbuf.seek(0)
                        st.session_state[cache_key] = zbuf.getvalue()
                except Exception as e:
                    st.error(f"❌ Error creating images: {str(e)}")
                    st.code(traceback.format_exc())
                    st.session_state[cache_key] = None

            if st.session_state.get(cache_key):
                file_suffix = "composite" if include_heatmap else "render"
                image_type = "composite images" if include_heatmap else "rendered images"
                st.download_button(
                    label=f"📥 Download {image_type.title()}",
                    data=st.session_state[cache_key],
                    file_name=f"robustness_all_images_{len(rendered_images)}_positions_{file_suffix}.zip",
                    mime="application/zip",
                    key=f"dl_btn_all_{len(rendered_images)}_{include_heatmap}",
                    use_container_width=True
                )
            else:
                st.button("📥 Download Rendered Images", disabled=True, help="Error preparing download", use_container_width=True)
        else:
            st.button("📥 Download Rendered Images", disabled=True, help="Analysis data not available", use_container_width=True)

    st.info("💡 **Tip:** Use the Previous/Next buttons to navigate, or download composite PNG images directly")

    c1, c2 = st.columns([0.8, 1.2])
    with c1:
        st.image(current["image"], caption=f"Position {current['position_idx']}", use_container_width=True)
    with c2:
        st.write("**Position Information:**")
        st.write(f"• **Position Index:** {current['position_idx']}")
        st.write(f"• **Azimuth:** {current['azimuth']:.1f}°")
        st.write(f"• **Elevation:** {current['elevation']:.1f}°")
        st.write(f"• **Distance:** {current['distance']:.2f}")

        st.write("**Prediction Information:**")
        st.write(f"• **Predicted Class:** {current['prediction'][:40]}...")
        st.write(f"• **Confidence:** {current['confidence']:.3f}")

        if "target_class" in st.session_state:
            target_idx = _get_idx_safe(st.session_state.get("target_class", ""))
            status = "✅ Correct" if current["class_idx"] == target_idx else "❌ Incorrect"
            st.write(f"• **Status:** {status}")

            if current.get("target_class_ranking") is not None:
                ranking = current["target_class_ranking"]
                target_confidence = current.get("target_class_confidence", 0.0)

                if ranking % 10 == 1 and ranking % 100 != 11:
                    suffix = "st"
                elif ranking % 10 == 2 and ranking % 100 != 12:
                    suffix = "nd"
                elif ranking % 10 == 3 and ranking % 100 != 13:
                    suffix = "rd"
                else:
                    suffix = "th"

                st.write(f"• **Target Class Ranking:** {ranking}{suffix} place (confidence: {target_confidence:.3f})")


# Results Visualization
def visualize_results(results: Dict, config: Dict):
    st.header("📊 Interactive Results Visualization")

    try:
        logits_stack = torch.stack(results["final_logits"])
        cams_stack = torch.stack([x["camera"] for x in results["final_scene_params"]])

        logits2d, cams2d = _expand_for_envmaps(logits_stack, cams_stack, results.get("envmap_paths"))

        total_positions = len(logits2d)
        assert len(cams2d) == total_positions, f"Mismatch: {len(cams2d)} camera positions vs {len(logits2d)} logit entries"

        with st.expander("🏷️ Most Predicted Classes (Top 5)", expanded=False):
            preds_top1 = _pred_top1(logits2d)
            values, counts = np.unique(preds_top1, return_counts=True)
            order = np.argsort(-counts)
            top_n = min(5, len(order))
            _, id_to_class = get_cached_imagenet_labels()
            for rank in range(top_n):
                cls_id = int(values[order[rank]])
                cnt = int(counts[order[rank]])
                pct = (cnt / total_positions) * 100.0
                label = id_to_class.get(cls_id, f"class {cls_id}")
                st.write(f"{rank+1}. {label[:30]}... — {cnt} ({pct:.1f}%)")

        st.subheader("🎯 Interactive Camera Position Heatmap")
        topk = st.session_state.get("topk_value", 1)
        labels_correct = utils.get_labels_correct(logits2d, config["target_class"], topk=topk)

        col_plot, col_manual = st.columns([2, 1])

        with col_plot:
            all_rendered_indices: Optional[List[int]] = None
            current_highlighted_index: Optional[int] = None
            selected_not_rendered: Optional[List[int]] = None

            if st.session_state.get("selected_positions"):
                selected_not_rendered = st.session_state["selected_positions"]

            if st.session_state.get("rendered_images"):
                rendered_images = st.session_state["rendered_images"]
                all_rendered_indices = [img["position_idx"] for img in rendered_images]
                cur_idx = st.session_state.get("carousel_index", 0)
                if 0 <= cur_idx < len(rendered_images):
                    current_highlighted_index = rendered_images[cur_idx]["position_idx"]

            combined_selected_indices = selected_not_rendered

            fig = create_interactive_polar_plot(
                cams2d.numpy(),
                labels_correct.numpy(),
                logits2d,
                config["target_class"],
                topk,
                selected_indices=combined_selected_indices,
                current_highlighted_index=current_highlighted_index,
            )

            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="polar_plot")

            if st.checkbox("🔍 Debug Mode", value=False):
                st.write("**Event Debug Info:**")
                st.write(f"Event: {event}")

            # Selection handling
            if event and isinstance(event, dict):
                selection = None
                if "selection" in event and event["selection"]:
                    if "points" in event["selection"] and event["selection"]["points"]:
                        selection = event["selection"]["points"]
                    elif "point_indices" in event["selection"]:
                        selection = event["selection"]["point_indices"]

                if selection:
                    try:
                        sel_positions: List[int] = []
                        for point in selection:
                            pos_idx = None
                            if isinstance(point, dict):
                                if "customdata" in point:
                                    pos_idx = int(point["customdata"])
                                elif "pointIndex" in point:
                                    pos_idx = int(point["pointIndex"])
                            elif isinstance(point, (int, float)):
                                pos_idx = int(point)
                            if pos_idx is not None:
                                sel_positions.append(pos_idx)

                        if sel_positions:
                            st.session_state["selected_positions"] = sel_positions
                            if len(sel_positions) == 1:
                                st.success(f"✅ Selected position {sel_positions[0]}")
                            else:
                                st.success(f"✅ Selected {len(sel_positions)} positions: {sel_positions}")
                            st.session_state["trigger_render"] = True

                    except Exception as e:
                        st.error(f"❌ Error processing selection: {str(e)}")

        with col_manual:
            topk = int(st.number_input(
                "Top-K Accuracy Threshold",
                min_value=1, max_value=5,
                value=st.session_state.get("topk_value", 1),
                step=1,
                help="Consider prediction correct if target class appears in top K predictions",
            ))
            st.session_state["topk_value"] = topk
            labels_correct = utils.get_labels_correct(logits2d, config["target_class"], topk=topk)

            correct_count = int(labels_correct.sum().item())
            total_positions = len(labels_correct)
            accuracy = (correct_count / total_positions) * 100
            st.info(f"📊 **Top-{topk} Accuracy:** {correct_count}/{total_positions} ({accuracy:.1f}%) correct")

            st.write(""); st.write("");
            st.subheader("🎯 Position Selection")
            target_idx = _get_idx_safe(config["target_class"])
            target_probs = torch.softmax(logits2d, dim=1)[:, target_idx].cpu().numpy()

            col_type, col_count = st.columns(2)
            with col_type:
                selection_type = st.selectbox(
                    "Show positions with:",
                    options=["worst_confidence", "best_confidence", "incorrect_predictions", "correct_predictions"],
                    format_func=lambda x: {
                        "worst_confidence": "▼ Lowest confidence (worst)",
                        "best_confidence": "▲ Highest confidence (best)",
                        "incorrect_predictions": "❌ Incorrect predictions",
                        "correct_predictions": "✅ Correct predictions",
                    }[x],
                    help="Choose which type of positions to analyze",
                )
            with col_count:
                max_positions = min(20, len(cams2d))
                num_positions = st.number_input(
                    "Number of positions",
                    min_value=1, max_value=max_positions, value=min(5, max_positions), step=1,
                    help=f"Number of positions to render (max {max_positions})",
                )

            if selection_type == "worst_confidence":
                sorted_idx = np.argsort(target_probs)
                selected_indices = sorted_idx[:num_positions]
                desc = f"Top {num_positions} positions with lowest confidence in '{config['target_class'][:30]}...'"
            elif selection_type == "best_confidence":
                sorted_idx = np.argsort(target_probs)[::-1]
                selected_indices = sorted_idx[:num_positions]
                desc = f"Top {num_positions} positions with highest confidence in '{config['target_class'][:30]}...'"
            elif selection_type == "incorrect_predictions":
                preds1 = _pred_top1(logits2d)
                incorrect = np.where(preds1 != target_idx)[0]
                if len(incorrect) > 0:
                    confs = target_probs[incorrect]
                    order = incorrect[np.argsort(confs)]
                    selected_indices = order[:num_positions]
                    desc = f"Top {min(num_positions, len(selected_indices))} incorrect predictions (lowest confidence)"
                else:
                    selected_indices = np.array([], dtype=int)
                    desc = "No incorrect predictions found!"
            else:  # correct_predictions
                preds1 = _pred_top1(logits2d)
                correct = np.where(preds1 == target_idx)[0]
                if len(correct) > 0:
                    confs = target_probs[correct]
                    order = correct[np.argsort(confs)[::-1]]
                    selected_indices = order[:num_positions]
                    desc = f"Top {min(num_positions, len(selected_indices))} correct predictions (highest confidence)"
                else:
                    selected_indices = np.array([], dtype=int)
                    desc = "No correct predictions found!"

            st.info(f"📋 **Selection:** {desc}")

            if len(selected_indices) > 0:
                confs = target_probs[selected_indices]
                st.caption(f"**Confidence range:** {confs.min():.3f} - {confs.max():.3f} (avg: {confs.mean():.3f})")
                with st.expander("🔍 Preview selected positions", expanded=False):
                    _, id_to_class = get_cached_imagenet_labels()
                    for i, pos_idx in enumerate(selected_indices):
                        conf = target_probs[pos_idx]
                        pred_idx = int(torch.argmax(logits2d[pos_idx]).item())
                        pred_class = id_to_class.get(pred_idx, f"class {pred_idx}")
                        status = "✅" if pred_idx == target_idx else "❌"
                        st.write(f"{i+1}. **Position {pos_idx}:** {status} {conf:.3f} confidence → {pred_class[:25]}...")

            if st.button("🎨 Render Selected Positions", type="primary", disabled=len(selected_indices) == 0):
                if len(selected_indices) > 0:
                    st.session_state["selected_positions"] = selected_indices.tolist()
                    st.session_state["trigger_render"] = True
                    st.success(f"✅ Selected {len(selected_indices)} positions - updating heatmap and rendering...")
                    st.rerun()

        # Render selected images
        if (st.session_state.get("trigger_render", False)
            and st.session_state.get("selected_positions")
            and st.session_state.get("robust_analyzer")):

            st.subheader("🖼️ Rendered Images")
            selected_positions = st.session_state["selected_positions"]
            robust_analyzer = st.session_state["robust_analyzer"]

            valid_positions = [pos for pos in selected_positions if pos < len(cams2d)]
            if len(valid_positions) != len(selected_positions):
                invalid = [pos for pos in selected_positions if pos >= len(cams2d)]
                st.warning(f"⚠️ Removed invalid positions: {invalid}")

            if valid_positions:
                st.session_state["target_class"] = config["target_class"]
                cache_key = f"{sorted(valid_positions)}_{id(robust_analyzer)}"

                if st.session_state.get("image_cache_key") != cache_key:
                    with st.status("🎨 Rendering selected images...", expanded=True) as status:
                        rendered_images = render_multiple_images(robust_analyzer, results, valid_positions)
                        st.session_state["rendered_images"] = rendered_images
                        st.session_state["image_cache_key"] = cache_key
                        st.session_state["carousel_index"] = 0
                        status.update(label=f"✅ Rendered {len(rendered_images)} images!", state="complete")
                        st.rerun()

                if st.session_state.get("rendered_images"):
                    create_image_carousel(st.session_state["rendered_images"])
                else:
                    st.error("❌ No images were successfully rendered")

            st.session_state["trigger_render"] = False

        elif st.session_state.get("rendered_images") and st.session_state.get("selected_positions"):
            st.subheader("🖼️ Rendered Images")
            create_image_carousel(st.session_state["rendered_images"])
            st.info(f"📦 **Cached:** {len(st.session_state['rendered_images'])} images in memory")
        else:
            if not st.session_state.get("robust_analyzer"):
                st.info("⚠️ No robust analyzer found. Please run the analysis first.")
            else:
                st.info("👆 Choose a point (or multiple points) on the polar plot or use the position selection to the right to see the rendered images!")

        with st.expander("📊 Static Polar Plot", expanded=False):
            utils.visualize_positions_polar(
                cams2d.numpy(),
                labels_correct.numpy(),
                title=f"Camera Position Analysis (Top-{topk}) - Target: {config['target_class'][:30]}...",
            )
            polar_fig = plt.gcf()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(polar_fig, use_container_width=True)
            plt.close()

        with st.expander("📈 Distribution Histograms", expanded=False):
            utils.visualize_positions_with_distributions(
                cams2d.numpy(),
                labels_correct.numpy(),
                title=f"Distribution Analysis (Top-{topk}) - Target: {config['target_class'][:30]}...",
                mode="distributions",
                show_distance=False
            )
            dist_fig = plt.gcf()
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(dist_fig, use_container_width=True)
            plt.close()

        return {
            "camera_positions": cams2d.numpy(),
            "labels_correct": labels_correct.numpy(),
            "target_class": config["target_class"],
            "topk": topk,
        }

    except Exception as e:
        st.error(f"❌ Error in processing results: {str(e)}")
        st.code(traceback.format_exc())
        return None


# Analysis
def run_analysis(
    obj_path: str,
    texture_path: Optional[str],
    envmap_paths: Sequence[str],
    config: Dict,
):
    raster_settings = {
        "image_size": config["image_size"],
        "bin_size": config["bin_size"],
        "max_faces_per_bin": config["max_faces_per_bin"],
    }

    kwargs = {
        "obj_path": obj_path,
        "texture_path": texture_path,
        "envmap_paths": envmap_paths,
        "target_class": config["target_class"],
        "batch_size": config["batch_size"],
        "params_to_optimize": config["params_to_optimize"],
        "targeted": config["targeted"],
        "positive_z": config["positive_z"],
        "raster_settings": raster_settings,
    }

    with st.spinner("Initializing robustness analyzer..."):
        robust_analyzer = RobustnessAnalyzer(**kwargs)

    st.session_state["robust_analyzer"] = robust_analyzer

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(run_num, total_runs, iteration=None, total_iterations=None):
        if iteration is not None and total_iterations is not None:
            run_progress = (run_num + iteration / total_iterations) / total_runs
        else:
            run_progress = run_num / total_runs
        progress_bar.progress(run_progress)
        if iteration is not None and total_iterations is not None:
            status_text.info(
                f"🎯 **Run {run_num + 1}/{total_runs}** | "
                f"Iteration {iteration + 1}/{total_iterations} | "
                f"Overall Progress: {run_progress:.1%}"
            )
        else:
            status_text.info(f"🎯 **Run {run_num + 1}/{total_runs}** | Overall Progress: {run_progress:.1%}")

    try:
        status_text.info(f"🚀 Starting {config['num_runs']} optimization runs...")
        results = robust_analyzer.run(
            num_runs=config["num_runs"],
            num_iterations=config["num_iterations"],
            lr=config["learning_rate"],
            progress_callback=progress_callback,
        )
        progress_bar.progress(1.0)
        status_text.success(f"✅ Analysis completed successfully! Processed {config['num_runs']} runs.")
        return robust_analyzer, results
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        return None, None


# Downloads Panel
def download_results(results: Dict, plot_data: Optional[Dict]) -> None:
    st.write(""); st.write(""); st.write("")
    st.header("💾 Download Results")
    st.write(""); st.write("")

    col1, col2, col3 = st.columns(3)
    st.write(""); st.write("")

    with col1:
        try:
            serializable_results: Dict[str, Union[List, Dict, None]] = {}
            for key, values in results.items():
                if isinstance(values, list) and values:
                    if torch.is_tensor(values[0]):
                        serializable_results[key] = [v.detach().cpu().numpy().tolist() for v in values]
                    elif isinstance(values[0], dict):
                        serializable_results[key] = [
                            {k: (v.detach().cpu().numpy().tolist() if torch.is_tensor(v) else v) for k, v in item.items()}
                            for item in values
                        ]
                    else:
                        serializable_results[key] = values
                else:
                    serializable_results[key] = values

            json_str = json.dumps(serializable_results, indent=2)
            st.download_button(
                label="📄 Download Results as JSON",
                data=json_str,
                file_name="robustness_analysis_results.json",
                mime="application/json",
            )
        except Exception as e:
            st.button("📄 Download Results as JSON", disabled=True, help=f"Error preparing JSON: {str(e)}")

    with col2:
        if plot_data:
            cache_key = f"heatmap_{hash(str(plot_data))}"
            if cache_key not in st.session_state:
                try:
                    utils.visualize_positions_polar(
                        plot_data["camera_positions"],
                        plot_data["labels_correct"],
                        title=f"Camera Position Analysis (Top-{plot_data['topk']}) - Target: {plot_data['target_class'][:30]}...",
                    )
                    polar_fig = plt.gcf()

                    polar_buf = io.BytesIO()
                    polar_fig.savefig(polar_buf, format="png", dpi=300, bbox_inches="tight")
                    polar_buf.seek(0)
                    polar_img = Image.open(polar_buf)
                    plt.close()

                    utils.visualize_positions_with_distributions(
                        plot_data["camera_positions"],
                        plot_data["labels_correct"],
                        title=f"Distribution Analysis (Top-{plot_data['topk']}) - Target: {plot_data['target_class'][:30]}...",
                        mode="distributions",
                        show_distance=False
                    )
                    dist_fig = plt.gcf()

                    dist_buf = io.BytesIO()
                    dist_fig.savefig(dist_buf, format="png", dpi=300, bbox_inches="tight")
                    dist_buf.seek(0)
                    dist_img = Image.open(dist_buf)
                    plt.close()

                    # Combine both images vertically
                    polar_width, polar_height = polar_img.size
                    dist_width, dist_height = dist_img.size

                    combined_width = max(polar_width, dist_width)
                    combined_height = polar_height + dist_height + 20

                    combined_img = Image.new('RGB', (combined_width, combined_height), 'white')

                    polar_x = (combined_width - polar_width) // 2
                    combined_img.paste(polar_img, (polar_x, 0))

                    dist_x = (combined_width - dist_width) // 2
                    combined_img.paste(dist_img, (dist_x, polar_height + 20))

                    combined_buf = io.BytesIO()
                    combined_img.save(combined_buf, format="PNG", dpi=(300, 300))
                    combined_buf.seek(0)
                    st.session_state[cache_key] = combined_buf.getvalue()
                except Exception as e:
                    st.session_state[cache_key] = None

            if st.session_state.get(cache_key):
                st.download_button(
                    label="📊 Download Plots",
                    data=st.session_state[cache_key],
                    file_name=f"polar_and_distributions_combined_top{plot_data['topk']}.png",
                    mime="image/png",
                    key="dl_combined_plots",
                )
            else:
                st.button("📊 Download Current Polar Heatmap", disabled=True, help="Error preparing heatmap")
        else:
            st.button("📊 Download Current Polar Heatmap", disabled=True, help="Plot data not available")

    with col3:
        if plot_data and all(k in st.session_state for k in ("robust_analyzer", "results", "config")):
            robust_analyzer = st.session_state["robust_analyzer"]
            config = st.session_state["config"]

            logits_stack = torch.stack(results["final_logits"])
            cams_stack = torch.stack([x["camera"] for x in results["final_scene_params"]])
            logits2d, cams2d = _expand_for_envmaps(logits_stack, cams_stack, results.get("envmap_paths"))

            total_positions = len(cams2d)

            include_heatmap = st.session_state.get("include_heatmap_global", True)

            if total_positions > 100:
                st.write(f"⚠️ Large dataset: {total_positions} positions")
                confirm_large = st.checkbox("I understand this will take time", key="confirm_large_download")
                if not confirm_large:
                    st.button("🖼️ Download All Images", disabled=True, help="Please confirm for large datasets")
                else:
                    if st.button("🖼️ Download All Images"):
                        with st.spinner("Rendering all images for download..."):
                            try:
                                all_indices = list(range(total_positions))
                                rendered_images = render_multiple_images(robust_analyzer, results, all_indices)

                                if rendered_images:
                                    topk = st.session_state.get("topk_value", 1)
                                    labels_correct = utils.get_labels_correct(logits2d, config["target_class"], topk=topk)

                                    zbuf = io.BytesIO()
                                    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zipf:
                                        summary = {
                                            "total_images": len(rendered_images),
                                            "target_class": config.get("target_class", ""),
                                            "image_format": "composite_with_heatmap" if include_heatmap else "render",
                                            "analysis_config": {
                                                "batch_size": config.get("batch_size"),
                                                "params_to_optimize": config.get("params_to_optimize"),
                                                "num_runs": config.get("num_runs"),
                                                "num_iterations": config.get("num_iterations"),
                                                "learning_rate": config.get("learning_rate"),
                                                "image_size": config.get("image_size"),
                                            },
                                            "timestamp": _now_iso(),
                                        }
                                        zipf.writestr("analysis_summary.json", json.dumps(summary, indent=2))

                                        pbar = st.progress(0)
                                        status = st.empty()
                                        for i, image_info in enumerate(rendered_images):
                                            pbar.progress((i + 1) / len(rendered_images))

                                            if include_heatmap:
                                                status.info(f"💾 Creating composite image {i + 1}/{len(rendered_images)}...")
                                                composite = create_composite_image(
                                                    image_info, cams2d.numpy(), labels_correct.numpy(), logits2d, config
                                                )
                                                ibuf = io.BytesIO()
                                                composite.save(ibuf, format="PNG", dpi=(300, 300))
                                                ibuf.seek(0)
                                                zipf.writestr(f"position_{image_info['position_idx']:03d}_analysis.png", ibuf.getvalue())
                                            else:
                                                status.info(f"💾 Saving rendered image {i + 1}/{len(rendered_images)}...")
                                                rendered_img = Image.fromarray(image_info["image"])
                                                ibuf = io.BytesIO()
                                                rendered_img.save(ibuf, format="PNG", dpi=(300, 300))
                                                ibuf.seek(0)
                                                zipf.writestr(f"position_{image_info['position_idx']:03d}_rendered.png", ibuf.getvalue())

                                                metadata = {
                                                    "position_idx": image_info["position_idx"],
                                                    "azimuth": float(image_info["azimuth"]),
                                                    "elevation": float(image_info["elevation"]),
                                                    "distance": float(image_info["distance"]),
                                                    "prediction": image_info["prediction"],
                                                    "confidence": float(image_info["confidence"]),
                                                    "class_idx": int(image_info["class_idx"]),
                                                    "run_idx": image_info["run_idx"],
                                                    "batch_idx": image_info["batch_idx"],
                                                    "env_idx": image_info["env_idx"],
                                                    "target_class_ranking": image_info.get("target_class_ranking"),
                                                    "target_class_confidence": image_info.get("target_class_confidence"),
                                                }
                                                zipf.writestr(f"position_{image_info['position_idx']:03d}_metadata.json",
                                                            json.dumps(metadata, indent=2))
                                        pbar.empty(); status.empty()

                                    zbuf.seek(0)
                                    file_suffix = "composite" if include_heatmap else "render"
                                    st.download_button(
                                        label="💾 Download ZIP",
                                        data=zbuf.getvalue(),
                                        file_name=f"robustness_all_{len(rendered_images)}_positions_{file_suffix}.zip",
                                        mime="application/zip",
                                        key="download_all_images_complete_small",
                                    )
                                    image_type = "composite images" if include_heatmap else "rendered images"
                                    st.success(f"✅ Created ZIP with {len(rendered_images)} {image_type}!")
                                else:
                                    st.error("❌ No images were successfully rendered")
                            except Exception as e:
                                st.error(f"❌ Error creating all images: {str(e)}")
                                st.code(traceback.format_exc())
            else:
                cache_key = f"all_images_zip_{total_positions}_{hash(str(config))}_{include_heatmap}"
                if cache_key not in st.session_state:
                    with st.spinner("Preparing all images..."):
                        try:
                            all_indices = list(range(total_positions))
                            rendered_images = render_multiple_images(robust_analyzer, results, all_indices)

                            if rendered_images:
                                topk = st.session_state.get("topk_value", 1)
                                labels_correct = utils.get_labels_correct(logits2d, config["target_class"], topk=topk)

                                zbuf = io.BytesIO()
                                with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zipf:
                                    summary = {
                                        "total_images": len(rendered_images),
                                        "target_class": config.get("target_class", ""),
                                        "image_format": "composite_with_heatmap" if include_heatmap else "render",
                                        "analysis_config": {
                                            "batch_size": config.get("batch_size"),
                                            "params_to_optimize": config.get("params_to_optimize"),
                                            "num_runs": config.get("num_runs"),
                                            "num_iterations": config.get("num_iterations"),
                                            "learning_rate": config.get("learning_rate"),
                                            "image_size": config.get("image_size"),
                                        },
                                        "timestamp": _now_iso(),
                                    }
                                    zipf.writestr("analysis_summary.json", json.dumps(summary, indent=2))

                                    for image_info in rendered_images:
                                        if include_heatmap:
                                            composite = create_composite_image(
                                                image_info, cams2d.numpy(), labels_correct.numpy(), logits2d, config
                                            )
                                            ibuf = io.BytesIO()
                                            composite.save(ibuf, format="PNG", dpi=(300, 300))
                                            ibuf.seek(0)
                                            zipf.writestr(f"position_{image_info['position_idx']:03d}_analysis.png", ibuf.getvalue())
                                        else:
                                            rendered_img = Image.fromarray(image_info["image"])
                                            ibuf = io.BytesIO()
                                            rendered_img.save(ibuf, format="PNG", dpi=(300, 300))
                                            ibuf.seek(0)
                                            zipf.writestr(f"position_{image_info['position_idx']:03d}_rendered.png", ibuf.getvalue())

                                            metadata = {
                                                "position_idx": image_info["position_idx"],
                                                "azimuth": float(image_info["azimuth"]),
                                                "elevation": float(image_info["elevation"]),
                                                "distance": float(image_info["distance"]),
                                                "prediction": image_info["prediction"],
                                                "confidence": float(image_info["confidence"]),
                                                "class_idx": int(image_info["class_idx"]),
                                                "run_idx": image_info["run_idx"],
                                                "batch_idx": image_info["batch_idx"],
                                                "env_idx": image_info["env_idx"],
                                                "target_class_ranking": image_info.get("target_class_ranking"),
                                                "target_class_confidence": image_info.get("target_class_confidence"),
                                            }
                                            zipf.writestr(f"position_{image_info['position_idx']:03d}_metadata.json",
                                                        json.dumps(metadata, indent=2))

                                zbuf.seek(0)
                                st.session_state[cache_key] = zbuf.getvalue()
                            else:
                                st.session_state[cache_key] = None
                        except Exception as e:
                            st.session_state[cache_key] = None

                if st.session_state.get(cache_key):
                    file_suffix = "composite" if include_heatmap else "render"
                    image_type = "composite images" if include_heatmap else "rendered images"
                    st.download_button(
                        label=f"🖼️ Download All {image_type.title()}",
                        data=st.session_state[cache_key],
                        file_name=f"robustness_all_{total_positions}_positions_{file_suffix}.zip",
                        mime="application/zip",
                        key="download_all_images_complete",
                    )
                else:
                    st.button("🖼️ Download All Images", disabled=True, help="Error preparing images")
        else:
            st.button("🖼️ Download All Images", disabled=True, help="Analysis data not available")


def create_distribution_histograms(
    camera_positions: np.ndarray,
    labels_correct: np.ndarray,
    target_class: str,
    topk_value: int,
) -> Tuple["plt.Figure", "plt.Figure"]:
    def plot_max_normalized_hist(ax, data, bins, color, label):
        counts, bin_edges = np.histogram(data, bins=bins)
        if counts.max() > 0:
            counts = counts / counts.max()
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]),
                  color=color, alpha=0.7, label=label, edgecolor='black', linewidth=0.5)

    azimuth, elevation, _ = utils.compute_spherical_coordinates(camera_positions)

    azimuth_correct = azimuth[labels_correct == 1]
    azimuth_incorrect = azimuth[labels_correct == 0]
    elevation_correct = elevation[labels_correct == 1]
    elevation_incorrect = elevation[labels_correct == 0]

    has_incorrect = len(azimuth_incorrect) > 0

    # Azimuth histogram
    azimuth_fig, azimuth_ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_max_normalized_hist(azimuth_ax, azimuth_correct, bins=20, color='lightblue',
                           label='Correct' if has_incorrect else 'All positions')
    if has_incorrect:
        plot_max_normalized_hist(azimuth_ax, azimuth_incorrect, bins=20, color='lightcoral',
                               label='Incorrect')

    azimuth_ax.set_xlabel('Azimuth (degrees)', fontsize=14)
    azimuth_ax.set_ylabel('Relative Frequency (max=1 per group)', fontsize=14)
    azimuth_ax.set_title(f'Azimuth Distribution (Top-{topk_value})\nTarget: {target_class[:40]}...', fontsize=16)
    azimuth_ax.legend(fontsize=12)
    azimuth_ax.grid(True, alpha=0.3)

    if has_incorrect:
        azimuth_ax.text(0.02, 0.98,
                       f"Correct: {len(azimuth_correct)} | Incorrect: {len(azimuth_incorrect)}",
                       transform=azimuth_ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        azimuth_ax.text(0.02, 0.98,
                       f"Total positions: {len(azimuth_correct)}",
                       transform=azimuth_ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    # Elevation histogram
    elevation_fig, elevation_ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_max_normalized_hist(elevation_ax, elevation_correct, bins=20, color='lightblue',
                           label='Correct' if has_incorrect else 'All positions')
    if has_incorrect:
        plot_max_normalized_hist(elevation_ax, elevation_incorrect, bins=20, color='lightcoral',
                               label='Incorrect')

    elevation_ax.set_xlabel('Elevation (degrees)', fontsize=14)
    elevation_ax.set_ylabel('Relative Frequency (max=1 per group)', fontsize=14)
    elevation_ax.set_title(f'Elevation Distribution (Top-{topk_value})\nTarget: {target_class[:40]}...', fontsize=16)
    elevation_ax.legend(fontsize=12)
    elevation_ax.grid(True, alpha=0.3)

    if has_incorrect:
        elevation_ax.text(0.02, 0.98,
                         f"Correct: {len(elevation_correct)} | Incorrect: {len(elevation_incorrect)}",
                         transform=elevation_ax.transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        elevation_ax.text(0.02, 0.98,
                         f"Total positions: {len(elevation_correct)}",
                         transform=elevation_ax.transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    return azimuth_fig, elevation_fig


def clear_cache_directory() -> bool:
    try:
        cache_dir = get_cache_dir()
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(exist_ok=True)
            print("✅ Cache directory cleared successfully")
            return True
    except Exception as e:
        print(f"❌ Error clearing cache directory: {str(e)}")
        return False
    return False


# Main App
def main() -> None:
    setup_page()

    has_cached = check_cached_files()
    if st.session_state.get("using_cached_files", False):
        st.success("🔄 **Using cached files from previous session**")
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.session_state.get("cache_info"):
                ci = st.session_state["cache_info"]
                st.info(f"📁 Cache: {', '.join(ci['file_names'])}")
        with col2:
            if st.button("🗑️ Clear Cache", help="Clear cached files and upload new ones"):
                st.session_state["files_processed"] = False
                st.session_state["file_paths"] = None
                st.session_state["using_cached_files"] = False
                st.session_state.pop("cache_info", None)

                if clear_cache_directory():
                    st.success("🗑️ Cache cleared successfully!")
                else:
                    st.error("❌ Error clearing cache directory")

                st.rerun()

    config = create_sidebar()

    if not st.session_state.get("using_cached_files", False):
        upload_result = handle_file_uploads()
        upload_type = upload_result[0]

        if "files_processed" not in st.session_state:
            st.session_state["files_processed"] = False
            st.session_state["file_paths"] = None

        if upload_type == "zip":
            zip_file = upload_result[1]
            if zip_file:
                try:
                    with st.spinner("Processing ZIP package..."):
                        obj_path, texture_path, envmap_paths, temp_dir = save_files_to_cache(upload_type, zip_file)
                    st.success("✅ ZIP package processed successfully!")

                    st.subheader("📋 Processed Files")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write("**3D Object:**")
                        st.write(f"✅ {os.path.basename(obj_path) if obj_path else '❌ No OBJ file found'}")
                    with c2:
                        st.write("**Texture:**")
                        st.write(f"✅ {os.path.basename(texture_path) if texture_path else '💡 Will use MTL materials'}")
                    with c3:
                        st.write("**Environment Maps:**")
                        if envmap_paths:
                            for env_path in envmap_paths:
                                st.write(f"✅ {os.path.basename(env_path)}")
                        else:
                            st.write("❌ No environment maps found")

                    mtl_path = obj_path.replace(".obj", ".mtl") if obj_path else None
                    if mtl_path and os.path.exists(mtl_path):
                        ok, missing = validate_mtl_textures(mtl_path, temp_dir)
                        if not ok:
                            st.warning(f"⚠️ Missing texture files: {', '.join(missing)}")

                    if obj_path and envmap_paths:
                        st.session_state["files_processed"] = True
                        st.session_state["file_paths"] = {
                            "obj_path": obj_path,
                            "texture_path": texture_path,
                            "envmap_paths": envmap_paths,
                            "temp_dir": temp_dir,
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
                        obj_path, texture_path, envmap_paths, temp_dir = save_files_to_cache(
                            upload_type, obj_file, mtl_file, texture_files, env_files
                        )
                    st.success("✅ Files processed successfully!")

                    st.subheader("📋 Processed Files")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write("**3D Object:**")
                        st.write(f"✅ {obj_file.name}")
                    with c2:
                        st.write("**Texture:**")
                        if texture_files:
                            if hasattr(texture_files, "read"):
                                st.write(f"✅ {texture_files.name}")
                            else:
                                for tf in texture_files:
                                    st.write(f"✅ {tf.name}")
                        else:
                            st.write("💡 Will use MTL materials")
                    with c3:
                        st.write("**Environment Maps:**")
                        for ef in env_files:
                            st.write(f"✅ {ef.name}")

                    if mtl_file:
                        mtl_path = os.path.join(temp_dir, mtl_file.name)
                        ok, missing = validate_mtl_textures(mtl_path, temp_dir)
                        if not ok:
                            st.warning(f"⚠️ Missing texture files: {', '.join(missing)}")
                            st.info("💡 Please upload all texture files referenced in the MTL file")

                    st.session_state["files_processed"] = True
                    st.session_state["file_paths"] = {
                        "obj_path": obj_path,
                        "texture_path": texture_path,
                        "envmap_paths": envmap_paths,
                        "temp_dir": temp_dir,
                    }
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
            else:
                st.warning("⚠️ Please upload at least an OBJ file and environment map(s) to proceed.")

    if st.session_state.get("files_processed", False):
        st.header("🚀 Run Analysis")
        with st.expander("📋 Current Configuration", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Target Class:** {config['target_class'][:40]}...")
                st.write(f"**Batch Size:** {config['batch_size']}")
                st.write(f"**Parameters to Optimize:** {', '.join(config['params_to_optimize'])}")
            with c2:
                st.write(f"**Number of Runs:** {config['num_runs']}")
                st.write(f"**Optimization Steps:** {config['num_iterations']}")
                st.write(f"**Learning Rate:** {config['learning_rate']:.1e}")

        if st.button("🎯 **Run Robustness Analysis**", type="primary", use_container_width=True):
            fp = st.session_state["file_paths"]
            robust_analyzer, results = run_analysis(fp["obj_path"], fp["texture_path"], fp["envmap_paths"], config)
            if results is not None:
                st.session_state["results"] = results
                st.session_state["config"] = config

    if st.session_state.get("results") is not None:
        plot_data = visualize_results(st.session_state["results"], st.session_state.get("config", config))
        download_results(st.session_state["results"], plot_data)

    if not st.session_state.get("files_processed", False):
        st.header("📖 Example Usage")
        st.markdown(
            """
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
            """
        )


if __name__ == "__main__":
    main()