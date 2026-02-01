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
import gc
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

from model import Model, MODEL_CONFIGS
from robustness_analyzer import RobustnessAnalyzer
import utils
from image_selection import handle_image_selection
from visualization import (
    create_interactive_polar_plot,
    render_image_at_position,
    render_multiple_images,
    create_individual_heatmap,
    create_composite_image,
    download_single_image_package,
    download_all_images_package,
    create_image_carousel,
    visualize_results,
)
from texture_atlas import (
    visualize_textures,
    create_texture_atlas,
    handle_texture_atlas,
    create_texture_atlas_from_zip,
)

from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
import subprocess
import sys
import base64

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


def _get_num_envmaps(envmap_paths: Optional[Sequence[str]]) -> int:
    """Get the number of environment maps from the paths."""
    if not envmap_paths:
        return 1
    # Filter for actual HDR/EXR files
    valid_paths = [p for p in envmap_paths if p.lower().endswith(('.hdr', '.exr'))]
    return len(valid_paths) if valid_paths else 1


def _expand_for_envmaps(
    logits: torch.Tensor,
    cam_positions_stack: torch.Tensor,
    envmap_paths: Optional[Sequence[str]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand logits and camera positions to account for environment maps."""
    # Infer num_classes from logits shape
    num_classes = logits.shape[-1]
    n_env = _get_num_envmaps(envmap_paths)
    if n_env > 1:
        cams = cam_positions_stack.unsqueeze(2).expand(-1, -1, n_env, -1)
        logits_2d = logits.reshape(-1, num_classes)
        cameras_2d = cams.reshape(-1, 3)
    else:
        logits_2d = logits.reshape(-1, num_classes)
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
def _download_imagenet_labels() -> Optional[Path]:
    """Download ImageNet labels from GitHub Gist if not found locally."""
    url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    target_path = Path(*IMAGENET_JSON_REL)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request
        urllib.request.urlretrieve(url, target_path)
        return target_path
    except Exception as e:
        print(f"Failed to download ImageNet labels: {e}")
        return None


def load_imagenet_labels() -> Tuple[List[Tuple[int, str]], Dict[int, str]]:
    try:
        labels_file = _imagenet_labels_path()
        if not labels_file:
            labels_file = _download_imagenet_labels()
        if not labels_file:
            raise FileNotFoundError("imagenet1000_clsidx_to_labels.json not found and download failed")

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


def load_custom_labels_metadata() -> Dict:
    """Load custom class labels metadata from cache."""
    cache_dir = get_cache_dir()
    metadata_file = cache_dir / "custom_labels_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading custom labels metadata: {e}")
    return {}


def save_custom_labels_metadata(metadata: Dict):
    """Save custom class labels metadata to cache."""
    try:
        cache_dir = get_cache_dir()
        metadata_file = cache_dir / "custom_labels_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving custom labels metadata: {e}")


def get_cached_custom_labels_for_model(model_name: str) -> Optional[Dict[int, str]]:
    """Get cached custom labels for a specific model."""
    metadata = load_custom_labels_metadata()
    if model_name in metadata:
        # Verify the file still exists
        labels_info = metadata[model_name]
        labels_path = Path(labels_info["path"])
        if labels_path.exists():
            try:
                with open(labels_path, 'r') as f:
                    labels = json.load(f)
                # Convert string keys to int keys if necessary
                return {int(k): v for k, v in labels.items()}
            except Exception as e:
                print(f"Error loading custom labels: {e}")
                return None
        else:
            # Clean up metadata for missing file
            del metadata[model_name]
            save_custom_labels_metadata(metadata)
    return None


def remove_cached_custom_labels_for_model(model_name: str) -> bool:
    """Remove cached custom labels for a specific model."""
    metadata = load_custom_labels_metadata()
    if model_name in metadata:
        labels_info = metadata[model_name]
        labels_path = Path(labels_info["path"])

        # Remove the file if it exists
        if labels_path.exists():
            try:
                labels_path.unlink()
            except Exception as e:
                print(f"Error removing labels file: {e}")
                return False

        # Remove from metadata
        del metadata[model_name]
        save_custom_labels_metadata(metadata)
        return True
    return False


def cache_custom_labels_file(labels_file, model_name: str) -> str:
    """
    Cache uploaded custom labels JSON file for reuse.

    Args:
        labels_file: Streamlit uploaded file object
        model_name: Name of the model architecture

    Returns:
        Path to cached labels file
    """
    cache_dir = get_cache_dir()
    labels_cache_dir = cache_dir / "custom_labels"
    labels_cache_dir.mkdir(exist_ok=True)

    # Create hash of file content for caching
    content = labels_file.read()
    labels_file.seek(0)  # Reset file pointer

    file_hash = hashlib.md5(content).hexdigest()
    cached_labels_path = labels_cache_dir / f"{model_name}_{file_hash}.json"

    # Validate JSON format
    try:
        labels_data = json.loads(content)
        # Ensure it's a dict with numeric keys (or string numeric keys)
        test_dict = {int(k): v for k, v in labels_data.items()}
    except Exception as e:
        raise ValueError(f"Invalid JSON format for class labels: {str(e)}")

    # Save if not already cached
    if not cached_labels_path.exists():
        with open(cached_labels_path, "wb") as f:
            f.write(content)
        print(f"✅ Custom labels cached to {cached_labels_path}")
    else:
        print(f"✅ Using cached labels from {cached_labels_path}")

    # Update metadata
    metadata = load_custom_labels_metadata()
    file_size_kb = len(content) / 1024
    metadata[model_name] = {
        "path": str(cached_labels_path),
        "original_name": labels_file.name,
        "file_hash": file_hash,
        "file_size_kb": file_size_kb,
        "timestamp": _now_iso(),
        "model_name": model_name,
        "num_labels": len(test_dict)
    }
    save_custom_labels_metadata(metadata)

    return str(cached_labels_path)


def get_class_label(class_idx: int, num_classes: int = 1000, custom_labels: Optional[Dict[int, str]] = None) -> str:
    """
    Get class label for a given index.
    For ImageNet (1000 classes), use actual labels.
    For custom models, use custom labels if provided, otherwise generic 'class_N' format.
    """
    # First check if custom labels are provided
    if custom_labels is not None and class_idx in custom_labels:
        return custom_labels[class_idx]

    # Fall back to ImageNet labels for 1000-class models
    if num_classes == 1000:
        _, id_to_class = get_cached_imagenet_labels()
        return id_to_class.get(class_idx, f"class_{class_idx}")
    else:
        return f"class_{class_idx}"


def create_target_class_selector(num_classes: int = 1000, custom_labels: Optional[Dict[int, str]] = None) -> str:
    """Create target class selector. Uses st.* (not st.sidebar.*) to work inside expander context."""
    st.markdown("**🎯 Target Class**")

    # For custom models with non-ImageNet classes, show simple numeric selection
    if num_classes != 1000:
        st.info(f"📊 Custom model with {num_classes} classes detected")

        # Use custom labels for display if available
        if custom_labels:
            st.success(f"🏷️ Using {len(custom_labels)} custom class labels")
            format_func = lambda x: f"{x}: {custom_labels.get(x, f'Class {x}')}"
        else:
            st.write("Select target class by index:")
            format_func = lambda x: f"Class {x}"

        selected_idx = st.selectbox(
            "🔢 Target Class Index:",
            options=list(range(num_classes)),
            index=0,
            format_func=format_func,
            help=f"Select target class index (0 to {num_classes-1})",
            key="target_class_selector"
        )

        display_label = custom_labels.get(selected_idx, f"Class {selected_idx}") if custom_labels else f"Class {selected_idx}"
        st.success(f"✅ Selected: {display_label}")
        # Return the index as a string so it can be used throughout the system
        return str(selected_idx)

    # For ImageNet-1000 models, show full label selection
    class_options, id_to_class = get_cached_imagenet_labels()
    if not class_options:
        st.warning("⚠️ Could not load ImageNet labels. Using text input.")
        return st.text_input(
            "Target Class",
            value=DEFAULT_TARGET_LABEL,
            help="ImageNet class name for the target",
            key="target_class_text_input"
        )

    display_options = [f"{idx}: {label}" for idx, label in class_options]

    # Default to tank class
    default_idx = 0
    for i, (_idx, label) in enumerate(class_options):
        if "tank" in label.lower() and "army" in label.lower():
            default_idx = i
            break

    selected_display = st.selectbox(
        "🔍 Search & Select Class:",
        display_options,
        index=default_idx,
        help="Type to search through ImageNet-1000 classes, then select",
        key="target_class_imagenet_selector"
    )

    if selected_display:
        selected_idx = int(selected_display.split(":")[0])
        target_class = id_to_class[selected_idx]
        st.success(f"✅ Selected: {target_class}")
        return target_class

    return DEFAULT_TARGET_LABEL


# Sidebar Config
def create_sidebar() -> Dict[str, Union[int, float, bool, str, List[str]]]:
    st.sidebar.header("📋 Configuration")

    st.sidebar.subheader("Model Parameters")

    # Model selection FIRST (before target class)
    model_options = list(MODEL_CONFIGS.keys())
    model_display_names = [MODEL_CONFIGS[key]["description"] for key in model_options]

    selected_model_display = st.sidebar.selectbox(
        "🤖 Classification Model:",
        model_display_names,
        index=2,  # Default to nth model from drop down menu
        help="Select the neural network model to use for classification"
    )

    # Get the model key from the display name
    selected_model = model_options[model_display_names.index(selected_model_display)]

    st.sidebar.success(f"✅ Selected: {MODEL_CONFIGS[selected_model]['description']}")

    # Custom weights upload - determine this BEFORE target class selector
    custom_weights_path = None
    with st.sidebar.expander("🔧 Custom Model Weights", expanded=False):
        st.write("**Upload custom pre-trained weights (optional)**")

        # Check for cached weights for current model
        cached_weights_info = get_cached_weights_for_model(selected_model)

        if cached_weights_info:
            st.success(f"📦 **Cached weights found for {MODEL_CONFIGS[selected_model]['description']}**")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**File:** {cached_weights_info['original_name']}")
                st.write(f"**Size:** {cached_weights_info['file_size_mb']:.1f} MB")
                cached_time = datetime.fromisoformat(cached_weights_info['timestamp'].strip('"'))
                st.write(f"**Cached:** {cached_time.strftime('%Y-%m-%d %H:%M')}")

            with col2:
                if st.button("❌", help="Remove cached weights", key="remove_cached_weights"):
                    if remove_cached_weights_for_model(selected_model):
                        st.success("✅ Cached weights removed")
                        st.rerun()
                    else:
                        st.error("❌ Error removing weights")

            use_cached_weights = st.checkbox(
                "Use cached weights",
                value=True,
                help="Use the cached weights for this model"
            )

            if use_cached_weights:
                custom_weights_path = cached_weights_info["path"]
                st.info("💡 Using cached weights")
            else:
                st.info("💡 Using default pre-trained weights")

            # Option to use different weights
            with st.expander("🔄 Use different weights", expanded=False):
                new_weights_source = st.radio(
                    "New weights source",
                    options=["📤 Upload file", "📁 Local path"],
                    horizontal=True,
                    key="cached_weights_source_mode"
                )

                if new_weights_source == "📤 Upload file":
                    weights_file = st.file_uploader(
                        "Upload model weights",
                        type=["pth", "pt", "bin", "safetensors"],
                        help="Supported formats: .pth, .pt, .bin, .safetensors",
                        key="new_weights_upload"
                    )

                    if weights_file:
                        try:
                            # Remove old cached weights first
                            remove_cached_weights_for_model(selected_model)

                            # Cache the new weights file
                            custom_weights_path = cache_weights_file(weights_file, selected_model)
                            st.success(f"✅ New weights cached: {weights_file.name}")

                            # Show file info
                            file_size = len(weights_file.getvalue()) / (1024 * 1024)  # MB
                            st.info(f"📊 File size: {file_size:.1f} MB")
                            st.rerun()  # Refresh to show new cached weights

                        except Exception as e:
                            st.error(f"❌ Error processing weights: {str(e)}")
                            custom_weights_path = None
                else:  # Local path
                    local_weights_input = st.text_input(
                        "Weights file path",
                        value="",
                        placeholder="/path/to/model_weights.pth",
                        help="Absolute path to weights file (.pth, .pt, .bin, .safetensors)",
                        key="cached_local_weights_path"
                    )

                    if local_weights_input:
                        local_weights_path = Path(local_weights_input)
                        if not local_weights_path.exists():
                            st.error(f"❌ File not found: {local_weights_input}")
                        elif local_weights_path.suffix.lower() not in ['.pth', '.pt', '.bin', '.safetensors']:
                            st.error(f"❌ Unsupported format: {local_weights_path.suffix}")
                        else:
                            file_size = local_weights_path.stat().st_size / (1024 * 1024)  # MB
                            st.success(f"✅ Weights found: {local_weights_path.name}")
                            st.info(f"📊 File size: {file_size:.1f} MB")
                            custom_weights_path = str(local_weights_path)
                    else:
                        st.info("👆 Enter the path to your weights file")
        else:
            # No cached weights - show upload interface
            use_custom_weights = st.checkbox(
                "Use custom weights",
                value=False,
                help="Upload your own model weights instead of using default pre-trained weights"
            )

            if use_custom_weights:
                weights_source = st.radio(
                    "Weights source",
                    options=["📤 Upload file", "📁 Local path"],
                    horizontal=True,
                    key="weights_source_mode"
                )

                if weights_source == "📤 Upload file":
                    weights_file = st.file_uploader(
                        "Upload model weights",
                        type=["pth", "pt", "bin", "safetensors"],
                        help="Supported formats: .pth, .pt, .bin, .safetensors"
                    )

                    if weights_file:
                        try:
                            # Cache the weights file
                            custom_weights_path = cache_weights_file(weights_file, selected_model)
                            st.success(f"✅ Weights cached: {weights_file.name}")

                            # Show file info
                            file_size = len(weights_file.getvalue()) / (1024 * 1024)  # MB
                            st.info(f"📊 File size: {file_size:.1f} MB")

                        except Exception as e:
                            st.error(f"❌ Error processing weights: {str(e)}")
                            custom_weights_path = None
                    else:
                        st.info("💡 Upload a weights file to use custom model")

                else:  # Local path
                    weights_path_input = st.text_input(
                        "Weights file path",
                        value="",
                        placeholder="/path/to/model_weights.pth",
                        help="Absolute path to weights file (.pth, .pt, .bin, .safetensors)",
                        key="local_weights_path"
                    )

                    if weights_path_input:
                        weights_path = Path(weights_path_input)
                        if not weights_path.exists():
                            st.error(f"❌ File not found: {weights_path_input}")
                        elif weights_path.suffix.lower() not in ['.pth', '.pt', '.bin', '.safetensors']:
                            st.error(f"❌ Unsupported format: {weights_path.suffix}")
                        else:
                            file_size = weights_path.stat().st_size / (1024 * 1024)  # MB
                            st.success(f"✅ Weights found: {weights_path.name}")
                            st.info(f"📊 File size: {file_size:.1f} MB")
                            custom_weights_path = str(weights_path)
                    else:
                        st.info("👆 Enter the path to your weights file")
            else:
                st.info("💡 Using default pre-trained weights")

    # Detect num_classes from custom weights FIRST (before showing UI)
    detected_num_classes = 1000  # Default to ImageNet
    detected_from_weights = False
    detection_key = None
    detection_error = None

    if custom_weights_path:
        try:
            # Load weights to detect num_classes
            if custom_weights_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(custom_weights_path)
            else:
                state_dict = torch.load(custom_weights_path, map_location='cpu', weights_only=False)

            # Handle nested dicts
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']

            # Look for classifier weight to detect num_classes
            # Check for different model architectures:
            # - ResNet/torchvision: 'fc.weight'
            # - Transformers/ViT: 'classifier.weight', 'head.weight'
            for key in state_dict.keys():
                if 'fc.weight' in key or 'classifier.weight' in key or 'classifier.1.weight' in key or 'head.weight' in key:
                    detected_num_classes = state_dict[key].shape[0]
                    detected_from_weights = True
                    detection_key = key
                    break
        except Exception as e:
            detection_error = str(e)
            detected_num_classes = 1000

    # Option to override number of classes (even without custom weights)
    num_classes_override = None
    with st.sidebar.expander("🔢 Output Classes", expanded=False):
        # Show current detected state
        if detected_from_weights:
            st.success(f"✅ **Detected {detected_num_classes} classes** from custom weights")
            st.caption(f"Source: `{detection_key}`")
        elif custom_weights_path and detection_error:
            st.warning(f"⚠️ Could not detect classes from weights: {detection_error}")
            st.info(f"📊 Current: **{detected_num_classes}** classes (default)")
        else:
            st.info(f"📊 Current: **{detected_num_classes}** classes (ImageNet default)")

        st.markdown("---")
        st.write("**Override the number of output classes**")
        st.write("Use this if you want to train a model with fewer classes than ImageNet-1000.")

        use_custom_num_classes = st.checkbox(
            "Override number of classes",
            value=False,
            help="Check to specify a custom number of output classes"
        )

        if use_custom_num_classes:
            num_classes_override = st.number_input(
                "Number of classes",
                min_value=2,
                max_value=10000,
                value=detected_num_classes if detected_num_classes != 1000 else 10,
                step=1,
                help="Number of output classes for the classifier head"
            )
            st.info(f"🎯 Model will have {num_classes_override} output classes with randomly initialized classifier")

    # Set final num_classes based on override or detection
    if num_classes_override is not None:
        num_classes = num_classes_override
    else:
        num_classes = detected_num_classes

    # Custom class labels upload (for non-ImageNet models)
    custom_labels = None
    if num_classes != 1000:
        with st.sidebar.expander("🏷️ Custom Class Labels", expanded=True):
            st.write("**Upload JSON file with class names**")
            st.write("Expected format: `{\"0\": \"class_name_0\", \"1\": \"class_name_1\", ...}`")

            # Check for cached labels for current model
            cached_labels = get_cached_custom_labels_for_model(selected_model)

            if cached_labels:
                metadata = load_custom_labels_metadata()
                labels_info = metadata.get(selected_model, {})

                st.success(f"📦 **Cached labels found for {MODEL_CONFIGS[selected_model]['description']}**")

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**File:** {labels_info.get('original_name', 'Unknown')}")
                    st.write(f"**Labels:** {labels_info.get('num_labels', len(cached_labels))}")
                    if 'timestamp' in labels_info:
                        cached_time = datetime.fromisoformat(labels_info['timestamp'].strip('"'))
                        st.write(f"**Cached:** {cached_time.strftime('%Y-%m-%d %H:%M')}")

                with col2:
                    if st.button("❌", help="Remove cached labels", key="remove_cached_labels"):
                        if remove_cached_custom_labels_for_model(selected_model):
                            st.success("✅ Cached labels removed")
                            st.rerun()
                        else:
                            st.error("❌ Error removing labels")

                use_cached_labels = st.checkbox(
                    "Use cached labels",
                    value=True,
                    help="Use the cached labels for this model"
                )

                if use_cached_labels:
                    custom_labels = cached_labels
                    st.info("💡 Using cached labels")
                else:
                    st.info("💡 Using generic class names")

                # Option to upload new labels
                with st.expander("📤 Upload new labels", expanded=False):
                    upload_new_labels = st.checkbox(
                        "Upload new labels (will replace cached)",
                        value=False,
                        help="Upload new labels to replace the current cached ones"
                    )

                    if upload_new_labels:
                        labels_file = st.file_uploader(
                            "Upload class labels JSON",
                            type=["json"],
                            help="JSON file mapping class indices to names",
                            key="new_labels_upload"
                        )

                        if labels_file:
                            try:
                                # Remove old cached labels first
                                remove_cached_custom_labels_for_model(selected_model)

                                # Cache the new labels file
                                labels_path = cache_custom_labels_file(labels_file, selected_model)
                                custom_labels = get_cached_custom_labels_for_model(selected_model)
                                st.success(f"✅ New labels cached: {labels_file.name}")
                                st.info(f"📊 {len(custom_labels)} labels loaded")
                                st.rerun()  # Refresh to show new cached labels
                            except Exception as e:
                                st.error(f"❌ Error processing labels: {str(e)}")
                                custom_labels = None
            else:
                # No cached labels - show upload interface
                use_custom_labels = st.checkbox(
                    "Use custom labels",
                    value=False,
                    help="Upload a JSON file with class names"
                )

                if use_custom_labels:
                    labels_file = st.file_uploader(
                        "Upload class labels JSON",
                        type=["json"],
                        help="JSON file mapping class indices to names: {\"0\": \"class_name\", ...}",
                        key="labels_upload"
                    )

                    if labels_file:
                        try:
                            # Cache the labels file
                            labels_path = cache_custom_labels_file(labels_file, selected_model)
                            custom_labels = get_cached_custom_labels_for_model(selected_model)
                            st.success(f"✅ Labels cached: {labels_file.name}")
                            st.info(f"📊 {len(custom_labels)} labels loaded")
                        except Exception as e:
                            st.error(f"❌ Error processing labels: {str(e)}")
                            custom_labels = None
                    else:
                        st.info("💡 Upload a JSON file to use custom labels")
                else:
                    st.info("💡 Using generic class names")

    # Robustness Analysis settings - always visible since it's the main feature
    with st.sidebar.expander("🎯 Robustness Analysis Settings", expanded=True):
            target_class = create_target_class_selector(num_classes, custom_labels)

            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=8,
                value=1,
                step=1,
                help="Number of viewpoints to optimize in parallel",
                key="robustness_batch_size"
            )

            st.subheader("Optimization Parameters")
            params_to_optimize = st.multiselect(
                "Parameters to Optimize",
                options=["camera"],
                default=["camera"],
                help="Select which parameters to optimize during adversarial attack",
                key="robustness_params_to_optimize"
            )

            num_runs = st.number_input(
                "Number of Runs",
                min_value=1,
                max_value=20_000,
                value=1,
                step=1,
                help="Total number of optimization runs",
                key="robustness_num_runs"
            )

            num_iterations = st.number_input(
                "Adversarial Optimization Steps",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                help="Number of optimization steps per adversarial run",
                key="robustness_num_iterations"
            )

            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
                value=5e-3,
                format_func=lambda x: f"{x:.1e}",
                key="robustness_learning_rate"
            )

            targeted = st.checkbox(
                "Targeted Attack",
                value=False,
                help="Whether this is a targeted adversarial attack",
                key="robustness_targeted"
            )

            st.markdown("---")
            st.markdown("**Camera Constraints**")
            positive_z = st.checkbox(
                "Positive Z",
                value=True,
                help="Constrain camera to positive elevation (z > 0)",
                key="robustness_positive_z"
            )

            min_max_proportion = st.slider(
                "Object proportion in image range",
                min_value=0.05,
                max_value=0.8,
                value=(0.3, 0.8),
                step=0.1,
                help="Camera distance range as proportion of bounding box size (min, max)",
                key="robustness_min_max_proportion"
            )
            st.caption(f"💡 object size proportion in image ({min_max_proportion[0]:.1f}x to {min_max_proportion[1]:.1f}x).")

            st.markdown("---")
            st.markdown("**Rendering Settings**")
            image_size = st.select_slider(
                "Image Size",
                options=[224, 256, 320, 384, 448, 512],
                value=448,
                help="Resolution of rendered images (image_size x image_size)",
                key="robustness_image_size"
            )
            bin_size = st.slider(
                "Bin Size",
                min_value=16,
                max_value=64,
                value=32,
                help="Spatial partitioning for rasterization - larger values use less memory but may be slower",
                key="robustness_bin_size"
            )
            max_faces_per_bin = st.number_input(
                "Max Faces per Bin",
                min_value=10000,
                max_value=200000,
                value=100000,
                step=10000,
                help="Maximum faces per spatial bin - increase for complex meshes, decrease to save memory",
                key="robustness_max_faces_per_bin"
            )
            st.info("💡 **Tip:** Use smaller bin sizes and fewer faces per bin if you encounter GPU memory issues.")

            st.markdown("---")
            st.markdown("**Download Settings**")
            include_heatmap = st.checkbox(
                "📊 Include heatmap in downloads",
                value=True,
                help="When checked: downloads include rendered image + heatmap composite. When unchecked: downloads only the rendered images with separate metadata files.",
                key="global_include_heatmap"
            )
            st.session_state["include_heatmap_global"] = include_heatmap

    # Store model config in session state for image selection panel
    st.session_state["model_config"] = {
        "model_name": selected_model,
        "custom_weights_path": custom_weights_path,
        "num_classes": int(num_classes),
        "num_classes_override": num_classes_override,  # User-specified override (even without weights)
        "custom_labels": custom_labels,  # Store custom labels for image selection
    }
    st.session_state["model_name"] = selected_model

    return {
        "target_class": target_class,
        "model_name": selected_model,
        "custom_weights_path": custom_weights_path,
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
        "min_max_proportion": min_max_proportion,
        "num_classes": int(num_classes),
        "custom_labels": custom_labels,
    }


# Image Selection Functions
def handle_file_uploads():
    st.header("📁 File Uploads")
    tab1, tab2, tab3 = st.tabs(["🎯 Robustness Analysis Files", "🔧 Texture Atlas Creator", "🖼️ Image Selection"])

    with tab1:
        st.session_state["active_upload_tab"] = "robustness"
        st.subheader("📦 3D Files for Robustness Analysis")

        # Help expander with documentation
        with st.expander("📖 How to Use Robustness Analysis", expanded=False):
            st.markdown("""
### Overview

**Robustness Analysis** tests how well a classifier recognizes your 3D object from different viewpoints and camera distances. It helps identify adversarial viewpoints where the model fails.

### File Requirements

| File Type | Format | Required | Description |
|-----------|--------|----------|-------------|
| 3D Mesh | `.obj` | ✅ Yes | Wavefront OBJ format mesh |
| Materials | `.mtl` | ⚠️ Optional | Material definitions |
| Textures | `.png`, `.jpg` | ⚠️ Optional | Texture images (referenced in MTL or single override) |
| Environment Maps | `.hdr`, `.exr`, `.png`, `.jpg` | ✅ Yes | background environments |

### Input Modes

**📦 ZIP Package Mode:**
- Upload a single ZIP file containing all your 3D assets
- Best for: Sharing complete models, archiving setups
```
model.zip/
├── model.obj           # 3D mesh
├── model.mtl           # Materials (optional)
└── textures/           # Texture folder (optional)
    ├── diffuse.png
    └── normal.png
```

**📤 Individual Files Mode:**
- Upload OBJ, MTL, textures, and environment maps separately via file uploader
- Single texture option overrides all MTL materials
- Best for: Quick testing, mixing and matching assets

**📁 Local Path Mode:**
- Specify absolute paths to files on the server filesystem
- No file upload needed - directly reference existing files
- Best for: Large files, repeated testing, server-side assets
- Required inputs:
  - **OBJ path**: `/path/to/model.obj`
  - **Environment maps directory**: `/path/to/envmaps/` (directory with .hdr, .exr, .png, .jpg files)
- Optional inputs:
  - **MTL path**: `/path/to/model.mtl`
  - **Textures directory**: `/path/to/textures/`

### Custom Model Weights

You can use your own fine-tuned model weights instead of pre-trained ImageNet weights:
- **Upload file**: Upload `.pth`, `.pt`, `.bin`, or `.safetensors` files
- **Local path**: Specify absolute path to weights file on the server
- The system auto-detects the number of output classes from the weights
- Cached weights persist across sessions

### Custom Class Labels

Upload a JSON file to use custom class names instead of generic indices:
```json
{"0": "class_a", "1": "class_b", "2": "class_c"}
```
- Labels are cached and persist across sessions
- Auto-matched to detected number of classes in weights

### Configuration Options (Sidebar)

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Target Class** | Class to test recognition for | ImageNet class |
| **Batch Size** | Viewpoints to optimize in parallel | 1 |
| **Number of Runs** | Total optimization runs | 1 |
| **Optimization Steps** | Steps per adversarial run | 1 |
| **Learning Rate** | Optimizer learning rate | 5e-3 |
| **Targeted Attack** | Whether attack is targeted | Off |

**Camera Constraints:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| **Positive Z** | Constrain to positive elevation | On |
| **Object Proportion Range** | How much of image the object fills | 0.3 - 0.8 |

**Rendering Settings:**
| Parameter | Description | Default |
|-----------|-------------|---------|
| **Image Size** | Resolution of rendered images | 448 |
| **Bin Size** | Spatial partitioning for rasterization | 32 |
| **Max Faces per Bin** | Max faces per spatial bin | 100,000 |

### Workflow Steps

1. **Select input mode** (ZIP, Individual files, or Local path)
2. **Provide 3D files** - upload or specify paths
3. **Configure** analysis parameters in sidebar
4. **Optionally** load custom weights and/or class labels
5. **Run Analysis** - generates viewpoint confidence map
6. **Explore results**:
   - Filter by environment map (if multiple)
   - View interactive polar heatmap
   - Click points to select positions for rendering
   - Use position selection panel for batch selection
7. **Render & download** selected positions

### Understanding Results

- **Interactive Polar Plot**: Click to select viewpoints for rendering
- **Environment Selection**: Filter results by specific environment map
- **Red zones**: Low confidence (adversarial viewpoints)
- **Green zones**: High confidence (robust viewpoints)
- **Top-k Accuracy**: Percentage of positions where target is in top-k predictions
- **Image Carousel**: Browse rendered images with adjustable display size
- **Batch Downloads**: Download all rendered images as ZIP

### Tips

- **Local paths** are faster for large files and repeated testing
- Use high-quality `.hdr` environment maps for realistic lighting
- More theta/phi points = finer angular resolution (but slower)
- Multiple environment maps test lighting robustness
- Check "worst-case" viewpoints for model vulnerabilities
- Custom weights enable testing domain-specific classifiers
            """)

        st.info("💡 Upload your 3D object files and environment maps for adversarial robustness testing")

        mode = st.radio(
            "Input source",
            options=["📦 ZIP package", "📤 Individual files", "📁 Local path"],
            horizontal=True,
            key="robustness_input_mode",
        )

        if mode == "📦 ZIP package":
            st.subheader("Upload Complete 3D Package")
            st.info("💡 Upload a ZIP file containing .obj, .mtl, and all texture files")
            zip_file = st.file_uploader(
                "Upload ZIP package",
                type=["zip"],
                help="ZIP file containing .obj, .mtl, and texture files with correct folder structure",
                key="robustness_zip_upload"
            )
            if zip_file:
                st.success(f"✅ Uploaded package: {zip_file.name}")
                return "zip", zip_file, None, None, None

        elif mode == "📁 Local path":
            st.subheader("Local File Paths")

            local_mode = st.radio(
                "Path specification mode",
                options=["🔍 Auto-detect from directory (Recommended)", "📝 Specify individual paths"],
                horizontal=True,
                key="local_path_mode",
                help="Auto-detect will find OBJ, MTL, textures, and envmaps automatically from a root directory"
            )

            if local_mode == "🔍 Auto-detect from directory (Recommended)":
                st.info("💡 Enter a directory path containing your 3D assets - files will be auto-detected")

                root_dir = st.text_input(
                    "Root directory path",
                    value="",
                    placeholder="/path/to/assets/",
                    help="Directory containing .obj, .mtl, texture (.png/.jpg), and environment map (.exr/.hdr) files",
                    key="local_root_dir"
                )

                if root_dir:
                    root_path = Path(root_dir)
                    if not root_path.exists():
                        st.error(f"❌ Directory not found: {root_dir}")
                    elif not root_path.is_dir():
                        st.error(f"❌ Not a directory: {root_dir}")
                    else:
                        # Auto-detect files
                        obj_files = list(root_path.glob("*.obj")) + list(root_path.glob("*.OBJ"))
                        mtl_files = list(root_path.glob("*.mtl")) + list(root_path.glob("*.MTL"))
                        texture_files = []
                        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tga']:
                            texture_files.extend(root_path.glob(ext))
                            texture_files.extend(root_path.glob(ext.upper()))
                        # Exclude envmaps from textures (will be detected separately)
                        env_files = []
                        for ext in ['*.hdr', '*.exr']:
                            env_files.extend(root_path.glob(ext))
                            env_files.extend(root_path.glob(ext.upper()))

                        # Show summary and collapsible details
                        non_env_textures = [t for t in texture_files if t.suffix.lower() not in ['.hdr', '.exr']]
                        obj_name = obj_files[0].name if obj_files else "None"
                        tex_count = len(non_env_textures)
                        env_count = len(env_files)

                        if obj_files and env_files:
                            st.success(f"✅ **Detected:** 1 model ({obj_name}), {tex_count} texture(s), {env_count} environment map(s)")
                        elif not obj_files:
                            st.error("❌ No OBJ file found in directory")
                        elif not env_files:
                            st.error("❌ No environment maps (.hdr/.exr) found in directory")

                        with st.expander("📁 View detected files", expanded=False):
                            col1, col2 = st.columns(2)

                            with col1:
                                if obj_files:
                                    st.write(f"✅ **OBJ:** {obj_files[0].name}")
                                    if len(obj_files) > 1:
                                        st.warning(f"⚠️ Multiple OBJ files found, using first one")
                                else:
                                    st.error("❌ No OBJ file found")

                                if mtl_files:
                                    st.write(f"✅ **MTL:** {mtl_files[0].name}")
                                else:
                                    st.info("ℹ️ No MTL file found (optional)")

                                if non_env_textures:
                                    st.write(f"✅ **Textures:** {len(non_env_textures)} file(s)")
                                    for tex in non_env_textures[:3]:
                                        st.text(f"  • {tex.name}")
                                    if len(non_env_textures) > 3:
                                        st.text(f"  ... and {len(non_env_textures) - 3} more")

                            with col2:
                                if env_files:
                                    st.write(f"✅ **Environment maps:** {len(env_files)} file(s)")
                                    for env_f in sorted(env_files)[:5]:
                                        st.text(f"  • {env_f.name}")
                                    if len(env_files) > 5:
                                        st.text(f"  ... and {len(env_files) - 5} more")
                                else:
                                    st.error("❌ No environment maps (.hdr/.exr) found")

                        # Return if valid
                        if obj_files and env_files:
                            return "local_path", {
                                "obj_path": str(obj_files[0]),
                                "mtl_path": str(mtl_files[0]) if mtl_files else None,
                                "texture_paths": [str(t) for t in texture_files if t.suffix.lower() not in ['.hdr', '.exr']],
                                "env_paths": [str(e) for e in sorted(env_files)],
                            }, None, None, None
                else:
                    st.info("👆 Enter the path to your assets directory to get started")

            else:  # Specify individual paths
                st.info("💡 Enter absolute paths to files on the server filesystem")

                col_obj, col_env = st.columns(2)

                with col_obj:
                    st.markdown("**3D Object Files**")
                    obj_path = st.text_input(
                        "OBJ file path",
                        value="",
                        placeholder="/path/to/model.obj",
                        help="Absolute path to the .obj mesh file",
                        key="local_obj_path"
                    )
                    mtl_path = st.text_input(
                        "MTL file path (optional)",
                        value="",
                        placeholder="/path/to/model.mtl",
                        help="Absolute path to the .mtl material file",
                        key="local_mtl_path"
                    )
                    texture_dir = st.text_input(
                        "Textures directory (optional)",
                        value="",
                        placeholder="/path/to/textures/",
                        help="Directory containing texture files referenced in MTL",
                        key="local_texture_dir"
                    )

                with col_env:
                    st.markdown("**Environment Maps**")
                    env_dir = st.text_input(
                        "Environment maps directory",
                        value="",
                        placeholder="/path/to/envmaps/",
                        help="Directory containing environment map files (.hdr, .exr, .png, .jpg)",
                        key="local_env_dir"
                    )

                # Validate paths
                if obj_path:
                    obj_file_path = Path(obj_path)
                    if not obj_file_path.exists():
                        st.error(f"❌ OBJ file not found: {obj_path}")
                    elif not obj_file_path.suffix.lower() == '.obj':
                        st.error(f"❌ File is not an OBJ: {obj_path}")
                    else:
                        st.success(f"✅ OBJ: {obj_file_path.name}")

                        # Validate MTL if provided
                        mtl_file_path = None
                        if mtl_path:
                            mtl_file_path = Path(mtl_path)
                            if not mtl_file_path.exists():
                                st.warning(f"⚠️ MTL file not found: {mtl_path}")
                                mtl_file_path = None
                            else:
                                st.success(f"✅ MTL: {mtl_file_path.name}")

                        # Validate texture directory if provided
                        texture_files_list = []
                        if texture_dir:
                            texture_dir_path = Path(texture_dir)
                            if not texture_dir_path.exists():
                                st.warning(f"⚠️ Texture directory not found: {texture_dir}")
                            elif not texture_dir_path.is_dir():
                                st.warning(f"⚠️ Not a directory: {texture_dir}")
                            else:
                                # Find texture files
                                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tga']:
                                    texture_files_list.extend(texture_dir_path.glob(ext))
                                    texture_files_list.extend(texture_dir_path.glob(ext.upper()))
                                if texture_files_list:
                                    st.success(f"✅ Found {len(texture_files_list)} texture(s)")
                                else:
                                    st.info("💡 No texture files found in directory")

                        # Validate environment maps directory
                        env_files_list = []
                        if env_dir:
                            env_dir_path = Path(env_dir)
                            if not env_dir_path.exists():
                                st.error(f"❌ Environment maps directory not found: {env_dir}")
                            elif not env_dir_path.is_dir():
                                st.error(f"❌ Not a directory: {env_dir}")
                            else:
                                # Find environment map files
                                for ext in ['*.hdr', '*.exr', '*.png', '*.jpg', '*.jpeg']:
                                    env_files_list.extend(env_dir_path.glob(ext))
                                    env_files_list.extend(env_dir_path.glob(ext.upper()))
                                if env_files_list:
                                    st.success(f"✅ Found {len(env_files_list)} environment map(s)")
                                    for env_f in sorted(env_files_list)[:5]:  # Show first 5
                                        st.text(f"  • {env_f.name}")
                                    if len(env_files_list) > 5:
                                        st.text(f"  ... and {len(env_files_list) - 5} more")
                                else:
                                    st.error("❌ No environment map files found in directory")

                        # Return local paths if valid
                        if obj_file_path.exists() and env_files_list:
                            return "local_path", {
                                "obj_path": str(obj_file_path),
                                "mtl_path": str(mtl_file_path) if mtl_file_path else None,
                                "texture_paths": [str(t) for t in texture_files_list],
                                "env_paths": [str(e) for e in sorted(env_files_list)],
                            }, None, None, None
                else:
                    st.info("👆 Enter the path to your OBJ file to get started")

        else:  # Individual files (📤 Individual files)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("3D Object Files")
                obj_file = st.file_uploader(
                    "Upload OBJ file",
                    type=["obj"],
                    help="Upload the 3D mesh file (.obj format)",
                    key="robustness_obj_upload"
                )
                mtl_file = st.file_uploader(
                    "Upload MTL file (optional)",
                    type=["mtl"],
                    help="Upload material definition file (.mtl format)",
                    key="robustness_mtl_upload"
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
                    key="robustness_single_texture_upload"
                )
                texture_files = st.file_uploader(
                    "Multiple texture files",
                    type=["png", "jpg", "jpeg", "bmp", "tga"],
                    accept_multiple_files=True,
                    help="Upload all texture files referenced in the MTL file",
                    key="robustness_texture_files_upload"
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
                    key="robustness_env_files_upload"
                )
                if env_files:
                    st.success(f"✅ Uploaded {len(env_files)} environment map(s)")
                    for env_file in env_files:
                        st.text(f"  • {env_file.name}")

            return "individual", obj_file, mtl_file, (single_texture or texture_files), env_files

    with tab2:
        st.session_state["active_upload_tab"] = "texture_atlas"
        handle_texture_atlas()

    with tab3:
        st.session_state["active_upload_tab"] = "image_selection"
        handle_image_selection()

    return "individual", None, None, None, None


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

    # Only print the message once per session to avoid spam
    if not st.session_state.get("cache_message_shown", False):
        print(f"✅ Loaded cached files from {latest['cache_key'][:8]}...")
        st.session_state["cache_message_shown"] = True

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

    # Sort envmap paths to ensure deterministic order (0->N-1)
    envmap_paths.sort()

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

    # Sort envmap paths to ensure deterministic order (0->N-1)
    envmap_paths.sort()

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
                # Sort envmap paths to ensure deterministic order (0->N-1)
                cached_envmaps = cached.get("envmap_paths", [])
                cached_envmaps.sort()
                return cached["obj_path"], cached["texture_path"], cached_envmaps, str(upload_cache)
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
        "model_name": config["model_name"],
        "custom_weights_path": config["custom_weights_path"],
        "min_max_proportion": config["min_max_proportion"],
        "custom_labels": config.get("custom_labels"),
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
                f"Overall Progress: {run_progress:.1%} | "
                f"Model: {MODEL_CONFIGS[config['model_name']]['description']}"
            )
        else:
            status_text.info(
                f"🎯 **Run {run_num + 1}/{total_runs}** | "
                f"Overall Progress: {run_progress:.1%} | "
                f"Model: {MODEL_CONFIGS[config['model_name']]['description']}"
            )

    try:
        status_text.info(
            f"🚀 Starting {config['num_runs']} optimization runs with "
            f"{MODEL_CONFIGS[config['model_name']]['description']}..."
        )
        results = robust_analyzer.run(
            num_runs=config["num_runs"],
            num_iterations=config["num_iterations"],
            lr=config["learning_rate"],
            progress_callback=progress_callback,
        )

        # Add envmap_paths to results so we can properly detect number of environments
        results["envmap_paths"] = envmap_paths

        progress_bar.progress(1.0)
        status_text.success(
            f"✅ Analysis completed successfully! Processed {config['num_runs']} runs "
            f"with {MODEL_CONFIGS[config['model_name']]['description']}."
        )
        return robust_analyzer, results
    except Exception as e:
        error_message = str(e)

        # Check if it's an OOM-related error
        is_oom_error = (
            "out of memory" in error_message.lower() or
            "oom" in error_message.lower() or
            "gpu" in error_message.lower()
        )

        if is_oom_error:
            st.error("🚨 **GPU Out of Memory Error!**")
            st.markdown(error_message)

            st.warning("### 🔄 Action Required")
            st.markdown("""
            The analysis failed due to insufficient GPU memory.

            **Recommended: Restart the Streamlit App**

            To clear all GPU memory, you need to restart the Streamlit server:

            1. Press `Ctrl+C` in your terminal where Streamlit is running
            2. Run `streamlit run src/ui.py` again

            **Alternative: Adjust Settings and Retry**

            You can also try adjusting these settings before running the analysis again:
            - Reduce the **Batch Size** in the sidebar
            - Reduce the **Image Size** in Rendering Settings (try 256 or 128)
            - Use fewer environment maps
            - Close other applications using GPU memory
            """)

            if st.button("🧹 Clear Session & Try Again", help="Clear session state and try with adjusted settings"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    if key not in ["files_processed", "file_paths", "using_cached_files", "cache_info"]:
                        del st.session_state[key]
                st.success("✅ Session cleared! Adjust your settings and try again.")
                st.rerun()
        else:
            st.error(f"❌ Error during analysis: {error_message}")
            st.code(traceback.format_exc())

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
                        title=f"Azimuth-Elevation Heatmap",
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
                        title=f"Analysis of 3D Spherical Distribution of Model Classification",
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

            # Get envmap_paths from results
            envmap_paths = results.get("envmap_paths", [])

            logits2d, cams2d = _expand_for_envmaps(logits_stack, cams_stack, envmap_paths)

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


# Add this function near the other caching functions

def get_weights_metadata_path() -> Path:
    """Get path to weights metadata file."""
    cache_dir = get_cache_dir()
    weights_cache_dir = cache_dir / "model_weights"
    weights_cache_dir.mkdir(exist_ok=True)
    return weights_cache_dir / "weights_metadata.json"


def load_weights_metadata() -> Dict[str, Dict]:
    """Load weights metadata from cache."""
    metadata_path = get_weights_metadata_path()
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_weights_metadata(metadata: Dict[str, Dict]) -> None:
    """Save weights metadata to cache."""
    metadata_path = get_weights_metadata_path()
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving weights metadata: {e}")


def get_cached_weights_for_model(model_name: str) -> Optional[Dict]:
    """Get cached weights info for a specific model."""
    metadata = load_weights_metadata()
    if model_name in metadata:
        # Verify the file still exists
        weights_info = metadata[model_name]
        if Path(weights_info["path"]).exists():
            return weights_info
        else:
            # Clean up metadata for missing file
            del metadata[model_name]
            save_weights_metadata(metadata)
    return None


def remove_cached_weights_for_model(model_name: str) -> bool:
    """Remove cached weights for a specific model."""
    metadata = load_weights_metadata()
    if model_name in metadata:
        weights_info = metadata[model_name]
        weights_path = Path(weights_info["path"])

        # Remove the file if it exists
        if weights_path.exists():
            try:
                weights_path.unlink()
            except Exception as e:
                print(f"Error removing weights file: {e}")
                return False

        # Remove from metadata
        del metadata[model_name]
        save_weights_metadata(metadata)
        return True
    return False


def cache_weights_file(weights_file, model_name: str) -> str:
    """
    Cache uploaded weights file for reuse.

    Args:
        weights_file: Streamlit uploaded file object
        model_name: Name of the model architecture

    Returns:
        Path to cached weights file
    """
    cache_dir = get_cache_dir()
    weights_cache_dir = cache_dir / "model_weights"
    weights_cache_dir.mkdir(exist_ok=True)

    # Create hash of file content for caching
    content = weights_file.read()
    weights_file.seek(0)  # Reset file pointer

    file_hash = hashlib.md5(content).hexdigest()
    file_extension = Path(weights_file.name).suffix

    cached_weights_path = weights_cache_dir / f"{model_name}_{file_hash}{file_extension}"

    # Save if not already cached
    if not cached_weights_path.exists():
        with open(cached_weights_path, "wb") as f:
            f.write(content)
        print(f"✅ Weights cached to {cached_weights_path}")
    else:
        print(f"✅ Using cached weights from {cached_weights_path}")

    # Update metadata
    metadata = load_weights_metadata()
    file_size_mb = len(content) / (1024 * 1024)
    metadata[model_name] = {
        "path": str(cached_weights_path),
        "original_name": weights_file.name,
        "file_hash": file_hash,
        "file_size_mb": file_size_mb,
        "timestamp": _now_iso(),
        "model_name": model_name
    }
    save_weights_metadata(metadata)

    return str(cached_weights_path)


def validate_weights_compatibility(weights_path: str, model_name: str) -> Tuple[bool, str]:
    """
    Validate if uploaded weights are compatible with selected model.

    Args:
        weights_path: Path to weights file
        model_name: Name of the model architecture

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        import torch

        # Load weights to check format
        if weights_path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(weights_path)
            except ImportError:
                return False, "safetensors library not installed"
        else:
            state_dict = torch.load(weights_path, map_location='cpu')

        # Handle different state dict formats
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

        # Basic validation - check if it looks like a neural network
        if not isinstance(state_dict, dict):
            return False, "Invalid state dict format"

        if len(state_dict) == 0:
            return False, "Empty state dict"

        # Check for typical model keys
        keys = list(state_dict.keys())
        if not any('weight' in key or 'bias' in key for key in keys):
            return False, "No weight or bias parameters found"

        return True, "Weights appear valid"

    except Exception as e:
        return False, f"Error loading weights: {str(e)}"

def main() -> None:
    setup_page()

    # Initialize active_upload_tab early so sidebar config renders correctly
    if "active_upload_tab" not in st.session_state:
        st.session_state["active_upload_tab"] = "robustness"

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

    # Only show file uploads if not using cached files
    if not st.session_state.get("using_cached_files", False):
        upload_result = handle_file_uploads()
        upload_type = upload_result[0]

        # Regular file processing for robustness analysis
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

                    # Show summary and collapsible details
                    obj_name = os.path.basename(obj_path) if obj_path else "None"
                    tex_name = os.path.basename(texture_path) if texture_path else "MTL materials"
                    env_count = len(envmap_paths) if envmap_paths else 0
                    st.info(f"📋 **Package contents:** 1 model ({obj_name}), 1 texture ({tex_name}), {env_count} environment map(s)")

                    with st.expander("📁 View all extracted files", expanded=False):
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

                    # Show summary and collapsible details
                    tex_count = 1 if hasattr(texture_files, "read") else len(texture_files) if texture_files else 0
                    tex_name = texture_files.name if (texture_files and hasattr(texture_files, "read")) else f"{tex_count} file(s)" if tex_count else "MTL materials"
                    env_count = len(env_files)
                    st.info(f"📋 **Uploaded files:** 1 model ({obj_file.name}), texture ({tex_name}), {env_count} environment map(s)")

                    with st.expander("📁 View all uploaded files", expanded=False):
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
            elif st.session_state.get("active_upload_tab") == "robustness":
                st.warning("⚠️ Please upload at least an OBJ file and environment map(s) to proceed.")

        elif upload_type == "local_path":
            local_paths = upload_result[1]
            if local_paths:
                try:
                    obj_path = local_paths["obj_path"]
                    mtl_path = local_paths.get("mtl_path")
                    texture_paths = local_paths.get("texture_paths", [])
                    env_paths = local_paths.get("env_paths", [])

                    # Show summary and collapsible details
                    obj_name = Path(obj_path).name
                    tex_count = len(texture_paths)
                    tex_name = Path(texture_paths[0]).name if tex_count == 1 else f"{tex_count} file(s)" if tex_count else "MTL materials"
                    env_count = len(env_paths)
                    st.info(f"📋 **Local files:** 1 model ({obj_name}), texture ({tex_name}), {env_count} environment map(s)")

                    with st.expander("📁 View all local files", expanded=False):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.write("**3D Object:**")
                            st.write(f"✅ {Path(obj_path).name}")
                            if mtl_path:
                                st.write(f"✅ {Path(mtl_path).name}")
                        with c2:
                            st.write("**Textures:**")
                            if texture_paths:
                                for tp in texture_paths[:3]:
                                    st.write(f"✅ {Path(tp).name}")
                                if len(texture_paths) > 3:
                                    st.write(f"... +{len(texture_paths) - 3} more")
                            else:
                                st.write("💡 No textures specified")
                        with c3:
                            st.write("**Environment Maps:**")
                            for ep in env_paths[:3]:
                                st.write(f"✅ {Path(ep).name}")
                            if len(env_paths) > 3:
                                st.write(f"... +{len(env_paths) - 3} more")

                    # For local paths, we use files directly without copying to cache
                    # Determine texture path (first texture or None)
                    texture_path = texture_paths[0] if texture_paths else None

                    st.session_state["files_processed"] = True
                    st.session_state["file_paths"] = {
                        "obj_path": obj_path,
                        "texture_path": texture_path,
                        "envmap_paths": env_paths,
                        "temp_dir": None,  # No temp dir for local paths
                    }
                except Exception as e:
                    st.error(f"❌ Error processing local paths: {str(e)}")

    # Show robustness analysis sections
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
        # Check if target class has changed since analysis was run
        stored_config = st.session_state.get("config", {})
        current_target = config.get("target_class")
        stored_target = stored_config.get("target_class")

        if current_target != stored_target and stored_target is not None:
            st.warning(
                f"⚠️ **Target Class Changed!** The current target class (`{current_target}`) differs from "
                f"the one used during analysis (`{stored_target}`). This may cause errors or inconsistent results."
            )

            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🔄 Restart Analysis", type="primary", help="Clear results and restart with new target class"):
                    # Clear all analysis-related session state
                    st.session_state.pop("results", None)
                    st.session_state.pop("config", None)
                    st.session_state.pop("robust_analyzer", None)
                    st.session_state.pop("rendered_images", None)
                    st.session_state.pop("selected_positions", None)
                    st.session_state.pop("image_cache_key", None)
                    st.session_state.pop("carousel_index", None)
                    st.session_state.pop("trigger_render", None)
                    st.success("✅ Session cleared! Please run the analysis again with the new target class.")
                    st.rerun()
            with col2:
                st.info("💡 **Tip:** You can either restart the analysis with the new target class, or change the target class back to the original one used during analysis.")

        plot_data = visualize_results(st.session_state["results"], st.session_state.get("config", config))
        download_results(st.session_state["results"], plot_data)

if __name__ == "__main__":
    main()
