# -*- coding: utf-8 -*-
"""
Image Selection via TRAK (Tracing with Random Projections)

This module handles the image selection workflow for finding the most important
training samples based on TRAK influence scores.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import shutil
import tempfile
import traceback
import zipfile
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import MODEL_CONFIGS
from trak import (
    compute_trak_scores_simple,
    extract_gradients_from_images,
    project_gradients_random,
)

def load_classification_model_for_images(model_name, custom_weights_path=None, device='cuda'):
    """
    Load just the classification model for image selection, without 3D rendering components.

    Args:
        model_name: Model architecture name from MODEL_CONFIGS
        custom_weights_path: Path to custom weights file (optional)
        device: Device to use

    Returns:
        Loaded model ready for inference
    """
    from transformers import AutoConfig

    print(f"Loading model: {model_name}")
    print(f"Custom weights path: {custom_weights_path}")

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]

    if custom_weights_path is None:
        # Load pre-trained model with default 1000 classes
        model = config["model_class"].from_pretrained(
            config["model_name"],
            torch_dtype=torch.float32
        ).to(device).eval()
    else:
        # Load custom weights - detect number of classes
        # Try to load and detect num_classes
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

        # Detect num_classes from classifier head
        num_classes = None
        for key in ['classifier.weight', 'classifier.out_proj.weight', 'classifier.1.weight']:
            if key in state_dict:
                num_classes = state_dict[key].shape[0]
                break

        if num_classes is None:
            num_classes = 1000  # Default

        # Create model with correct num_classes
        model_config = AutoConfig.from_pretrained(config["model_name"])
        model_config.num_labels = num_classes
        model = config["model_class"](model_config).to(device).eval()

        # Load weights
        if not custom_weights_path.endswith('.safetensors'):
            # For .pt/.pth files, handle potential key mismatches
            model_state = model.state_dict()
            # Try direct load
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                st.warning(f"Some weights could not be loaded: {str(e)}")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


def train_and_evaluate_subset(
    model_name: str,
    custom_weights: str,
    subset_tensors: torch.Tensor,
    subset_labels: torch.Tensor,
    target_tensors: torch.Tensor,
    target_labels: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer_choice: str = "AdamW",
    freeze_backbone: bool = True,
    device: str = "cuda",
    progress_callback=None,
) -> Dict:
    """
    Train a model on a subset and evaluate on target data.

    Returns:
        Dict with final_train_loss, final_target_loss, final_target_acc
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Load fresh model
    model = load_classification_model_for_images(model_name, custom_weights, device)

    # Set up trainable parameters
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Optimizer
    weight_decay = 0.01
    if optimizer_choice == "AdamW":
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_choice == "Adam":
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    else:  # SGD
        optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    # Data loader
    subset_dataset = TensorDataset(subset_tensors, subset_labels)
    subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    final_train_loss = 0
    final_target_loss = 0
    final_target_acc = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for images, labels in subset_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        final_train_loss = total_train_loss / len(subset_loader)

        # Evaluation phase on target set
        if target_tensors is not None and len(target_tensors) > 0:
            model.eval()
            total_target_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for i in range(0, len(target_tensors), batch_size):
                    batch_images = target_tensors[i:i+batch_size].to(device)
                    batch_labels = target_labels[i:i+batch_size].to(device)

                    outputs = model(batch_images)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                    loss = criterion(logits, batch_labels)
                    total_target_loss += loss.item()

                    _, predicted = torch.max(logits, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

            num_batches = (len(target_tensors) + batch_size - 1) // batch_size
            final_target_loss = total_target_loss / num_batches
            final_target_acc = 100 * correct / total

        if progress_callback:
            progress_callback(epoch + 1, epochs)

    # Clean up
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "train_loss": final_train_loss,
        "target_loss": final_target_loss,
        "target_acc": final_target_acc,
    }


def handle_image_selection():
    """Handle the image selection workflow for finding most important training samples."""
    st.subheader("🖼️ Image Selection via TRAK")
    st.info("💡 Upload datasets to train a model and select the most important images based on TRAK influence scores")

    # Help expander with documentation
    with st.expander("📖 How to Use Image Selection", expanded=False):
        st.markdown("""
### Overview

**TRAK (Tracing with Random Projections)** identifies which training images have the highest influence on model predictions for a target domain. This helps you curate optimal training sets by finding the most impactful synthetic/simulated images for real-world performance.

### Dataset Input

**📦 Upload ZIP:**
- Upload a ZIP file containing `dataset.json` and an `images/` folder
- Best for: Sharing datasets, remote usage

**📁 Local Path:**
- Specify an absolute path to a dataset directory on the server
- No upload needed - directly reference existing files
- Best for: Large datasets, repeated testing, server-side data
- Example: `/mnt/data/my_dataset/`

### Dataset Structure

```
dataset_folder/
├── dataset.json          # Manifest file (required)
└── images/               # All images referenced in manifest
    ├── train_img_001.png
    ├── target_img_001.png
    └── ...
```

### Manifest Structure (`dataset.json`)

```json
{
  "dataset_info": {
    "name": "My Dataset",
    "target_class": 5,
    "target_class_name": "tank",
    "num_classes": 10
  },
  "class_mapping": {"0": "airplane", "5": "tank", ...},
  "entries": [
    {
      "filename": "images/train_img_001.png",
      "class_idx": 5,
      "purposes": ["train", "select"],
      "source": "simulated"
    },
    {
      "filename": "images/real_001.png",
      "class_idx": 5,
      "purposes": ["target"],
      "source": "real"
    }
  ]
}
```

### Image Purposes

| Purpose | Description | Required |
|---------|-------------|----------|
| `train` | Training images (target class + confusers) | ✅ Yes |
| `target` | Real-world images to optimize for | ✅ Yes |
| `select` | Candidate pool for TRAK selection | ⚠️ Optional (defaults to all train) |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Training epochs** | More epochs = better TRAK signal but slower | 5 |
| **Batch size** | Adjust based on GPU memory | 128 |
| **Learning rate** | Lower LR = slower convergence = more gradient signal | 1e-4 |

**Training Options** (expandable):
- **Reset classifier head**: Reinitialize classifier weights before training
- **Optimizer**: AdamW (default), Adam, or SGD

### TRAK Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| **JL projection dimension** | Higher = more accurate but slower/more memory | 1024 |
| **Number of projections** | More = more stable scores (averaged) | 4 |
| **Gradient source** | "Last layer only" (faster) or "Full model" | Last layer only |
| **Number of checkpoints** | Save multiple checkpoints during training, average TRAK across them | 1 |

**Ensemble TRAK**: Using multiple checkpoints × multiple projections gives more stable, reliable influence scores. For example, 3 checkpoints × 4 projections = 12 TRAK computations averaged.

### Workflow (3 Steps)

**Step 1: Train Initial Model**
- Trains a classifier on all training images (target class + confusers)
- Saves model checkpoints for TRAK computation
- Shows training/target loss and accuracy per epoch

**Step 2: Compute TRAK Scores**
- Extracts gradients from selection pool and target images
- Projects gradients using Johnson-Lindenstrauss random projection
- Computes influence scores for each selection image
- Displays sortable score table with image carousel

**Step 3: Select Subset & Retrain**
- Choose selection method: Top percentage or Top count
- Choose selection strategy: Top (highest influence), Bottom, or Random
- Optionally override training hyperparameters
- **Compare All Methods**: Run Top, Bottom, and Random selections and compare target accuracy
- Retrain model on selected subset + proportional confusers

### Understanding Results

- **Higher TRAK scores** = images more influential for target domain performance
- **Score table**: Sortable by rank, viewable in carousel or grid
- **Downloads**: CSV (scores only) or JSON (full manifest with scores added)
- **Comparison mode**: Trains all three methods and shows which performs best

### Example Use Case

- **Train images**: 500 simulated tank renders + 500 confuser images (other classes)
- **Target images**: 50 real-world tank photos
- **Selection pool**: The 500 simulated tank renders
- **TRAK config**: 3 checkpoints, 4 projections, JL dim=2048
- **Result**: Top 10% (50 images) most influential for real-world tank recognition

### Tips

- **Local paths** are faster for large datasets - no upload/extraction overhead
- **More checkpoints** give more stable scores but take longer to compute
- **Last layer gradients** are usually sufficient and much faster than full model
- Use **Compare All Methods** to verify TRAK selection outperforms random
- Higher **JL dimension** improves accuracy but uses more memory

""")

    # Get model selection from session state (set by sidebar)
    if "model_name" not in st.session_state:
        st.warning("⚠️ Please configure model settings in the sidebar first")
        return

    # File upload section
    st.markdown("### 📤 Upload Dataset")

    st.markdown("""
    **JSON Manifest Upload:**
    - Upload a **dataset folder (as ZIP)** containing images and a `dataset.json` manifest
    - The JSON specifies which images are for:
      - `target`: Evaluation/selection targets (real images)
      - `train`: Training images (tank + confusers)
      - `select`: Selection pool (tank images only)

    **Format example:** See the `ui_dataset` folder generated by `create_ui_dataset.py`
    """)

    # Dataset source selection
    dataset_source = st.radio(
        "Dataset source",
        options=["📦 Upload ZIP", "📁 Local path"],
        horizontal=True,
        help="Choose how to load the dataset - upload a ZIP file or specify a local directory path"
    )

    dataset_path = None  # Will be set to the path of the dataset directory

    if dataset_source == "📦 Upload ZIP":
        dataset_zip = st.file_uploader(
            "Upload dataset folder (as ZIP)",
            type=["zip"],
            help="ZIP file containing images/ folder and dataset.json manifest",
            key="dataset_zip"
        )

        if not dataset_zip:
            st.info("👆 Upload dataset ZIP file containing dataset.json and images/")
            return

        st.success(f"✅ Dataset uploaded: {dataset_zip.name}")

    else:  # Local path
        st.markdown("**Enter the path to your local dataset directory:**")
        local_dataset_path = st.text_input(
            "Local dataset path",
            value="",
            placeholder="/path/to/ui_dataset",
            help="Absolute path to a directory containing dataset.json and images/ folder"
        )

        if not local_dataset_path:
            st.info("👆 Enter the path to your dataset directory (e.g., `/mnt/xfs/home/user/ui_dataset`)")
            return

        # Validate the path
        local_path = Path(local_dataset_path)
        if not local_path.exists():
            st.error(f"❌ Path does not exist: {local_dataset_path}")
            return
        if not local_path.is_dir():
            st.error(f"❌ Path is not a directory: {local_dataset_path}")
            return

        manifest_path = local_path / "dataset.json"
        if not manifest_path.exists():
            st.error(f"❌ No dataset.json found in {local_dataset_path}")
            return

        images_path = local_path / "images"
        if not images_path.exists():
            st.error(f"❌ No images/ folder found in {local_dataset_path}")
            return

        st.success(f"✅ Dataset found at: {local_dataset_path}")
        dataset_path = local_path  # Store for later use
        dataset_zip = None  # No ZIP file to process
    st.info("📋 The JSON manifest will be parsed to identify target/train/select images")

    # Training parameters (selection parameters moved to Step 3)
    st.markdown("### ⚙️ Training Parameters")
    col_param1, col_param2, col_param3 = st.columns(3)

    with col_param1:
        train_epochs = st.number_input(
            "Training epochs",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            help="Number of epochs to train the model (more = better TRAK scores, slower)"
        )

    with col_param2:
        train_batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=2048,
            value=128,
            step=8,
            help="Batch size for training and gradient extraction"
        )

    with col_param3:
        learning_rate = st.select_slider(
            "Learning rate",
            options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1],
            value=1e-4,
            format_func=lambda x: f"{x:.0e}",
            help="Lower learning rate = slower convergence = more gradient signal for TRAK"
        )

    # Additional training options (hidden in expander)
    with st.expander("🔧 Training Options", expanded=False):
        col_opt1, col_opt2 = st.columns(2)

        with col_opt1:
            reset_classifier = st.checkbox(
                "Reset classifier head",
                value=False,
                help="Reinitialize the classifier weights before training. Use this if your custom weights already know the target class."
            )

        with col_opt2:
            step1_optimizer = st.selectbox(
                "Optimizer",
                options=["AdamW", "Adam", "SGD"],
                index=0,
                key="step1_optimizer",
                help="Optimization algorithm"
            )

        # Fixed defaults
        step1_weight_decay = 0.01
        step1_momentum = 0.9

    # ==================== TRAK CONFIGURATION (UNIFIED) ====================
    st.markdown("### 🔬 TRAK Configuration")
    st.caption("These settings apply to gradient extraction (Step 2) and are also used for training/retraining (Steps 1 & 3).")

    col_trak1, col_trak2, col_trak3 = st.columns(3)

    with col_trak1:
        jl_dim = st.select_slider(
            "JL projection dimension",
            options=[512, 1024, 2048, 4096, 8192, 16384, 32768],
            value=1024,
            key="trak_jl_dim",
            help="Johnson-Lindenstrauss projection dimension. Higher = more accurate but slower/more memory."
        )

    with col_trak2:
        num_projections = st.number_input(
            "Number of projections",
            min_value=1,
            max_value=128,
            value=4,
            step=1,
            key="trak_num_projections",
            help="Number of random projections to average. More = more stable scores but slower."
        )

    with col_trak3:
        gradient_source = st.selectbox(
            "Gradient source",
            options=["Last layer only", "Full model"],
            index=0,
            key="trak_gradient_source",
            help="Which parameters to extract gradients from. 'Last layer only' is faster and often sufficient."
        )

    # Derive freeze_backbone from gradient_source for consistency across all steps
    freeze_backbone = (gradient_source == "Last layer only")

    # Second row: checkpoint ensemble settings
    col_trak4, col_trak5 = st.columns(2)

    with col_trak4:
        num_checkpoints = st.number_input(
            "Number of checkpoints",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            key="trak_num_checkpoints",
            help="Save multiple checkpoints during training and average TRAK scores across them. More = more stable but slower."
        )

    with col_trak5:
        if num_checkpoints > 1:
            # Show which epochs will be saved
            checkpoint_epochs_preview = [int(train_epochs * (i + 1) / num_checkpoints) for i in range(num_checkpoints)]
            st.caption(f"📸 Checkpoints at epochs: {checkpoint_epochs_preview}")
        else:
            st.caption("📸 Single checkpoint at final epoch")

    # Summary info
    total_score_computations = num_checkpoints * num_projections
    if total_score_computations > 1:
        parts = []
        if num_checkpoints > 1:
            parts.append(f"**{num_checkpoints}** checkpoints")
        if num_projections > 1:
            parts.append(f"**{num_projections}** projections")
        st.info(f"💡 Ensemble TRAK: {' × '.join(parts)} = **{total_score_computations}** score computations averaged (JL dim={jl_dim})")
    else:
        st.caption(f"🔧 Single checkpoint, single projection (JL dim={jl_dim})")

    # ==================== STEP STATUS INDICATORS ====================
    st.markdown("---")
    st.markdown("### 📋 Workflow Steps")

    # Check what's already completed
    step1_complete = "step1_data" in st.session_state
    step2_complete = "trak_data" in st.session_state

    # Status indicators
    col_status1, col_status2, col_status3 = st.columns(3)
    with col_status1:
        if step1_complete:
            st.success("✅ Step 1: Model Trained")
        else:
            st.info("⏳ Step 1: Train Model")
    with col_status2:
        if step2_complete:
            st.success("✅ Step 2: TRAK Computed")
        elif step1_complete:
            st.info("⏳ Step 2: Compute TRAK")
        else:
            st.warning("🔒 Step 2: Waiting for Step 1")
    with col_status3:
        if step2_complete:
            st.info("⏳ Step 3: Ready to Select")
        else:
            st.warning("🔒 Step 3: Waiting for Step 2")

    # Reset button to start over
    if step1_complete or step2_complete:
        if st.button("🔄 Reset All Steps", type="secondary"):
            if "step1_data" in st.session_state:
                # Clean up temp directory if exists
                old_temp_dir = st.session_state["step1_data"].get("temp_dir")
                if old_temp_dir and os.path.exists(old_temp_dir):
                    shutil.rmtree(old_temp_dir, ignore_errors=True)
                del st.session_state["step1_data"]
            if "trak_data" in st.session_state:
                del st.session_state["trak_data"]
            st.rerun()

    # ==================== STEP 1: TRAIN INITIAL MODEL ====================
    st.markdown("---")
    step1_button_disabled = step1_complete  # Disable if already done (use reset to redo)

    if st.button("🎓 **Step 1: Train Initial Model**", type="primary", use_container_width=True, disabled=step1_button_disabled):
        with st.spinner("Training initial model..."):
            try:
                # Clean up previous temp directory if it exists
                if "trak_data" in st.session_state:
                    old_temp_dir = st.session_state["trak_data"].get("temp_dir")
                    if old_temp_dir and os.path.exists(old_temp_dir):
                        shutil.rmtree(old_temp_dir, ignore_errors=True)

                # Handle dataset loading (ZIP upload vs local path)
                temp_dir = None  # Only set if we extract a ZIP

                if dataset_path is not None:
                    # Local path - no extraction needed
                    st.info("📁 Loading dataset from local path...")
                    dataset_root = str(dataset_path)
                    manifest_file = dataset_path / "dataset.json"
                else:
                    # ZIP upload - extract to temp directory
                    st.info("📦 Extracting dataset from ZIP...")
                    temp_dir = tempfile.mkdtemp()

                    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)

                    # Find dataset.json
                    manifest_file = None
                    for root, dirs, files in os.walk(temp_dir):
                        if 'dataset.json' in files:
                            manifest_file = Path(os.path.join(root, 'dataset.json'))
                            break

                    if not manifest_file:
                        st.error("❌ No dataset.json found in the uploaded ZIP file")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return

                    # Get dataset root (parent of dataset.json)
                    dataset_root = str(manifest_file.parent)

                # Parse JSON manifest
                st.info("📋 Parsing JSON manifest...")
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)

                # Extract images by purpose with their class labels
                all_train_paths = []
                all_train_class_labels = []
                selection_pool_paths = []
                selection_pool_class_labels = []
                target_paths = []
                target_class_labels = []

                for entry in manifest['entries']:
                    # Build absolute path
                    img_path = os.path.join(dataset_root, entry['filename'])

                    if not os.path.exists(img_path):
                        st.warning(f"⚠️ Image not found: {entry['filename']}")
                        continue

                    # Add to appropriate lists based on purposes
                    purposes = entry.get('purposes', [])
                    class_idx = entry.get('class_idx', 0)

                    if 'train' in purposes:
                        all_train_paths.append(img_path)
                        all_train_class_labels.append(class_idx)

                    if 'select' in purposes:
                        selection_pool_paths.append(img_path)
                        selection_pool_class_labels.append(class_idx)

                    if 'target' in purposes:
                        target_paths.append(img_path)
                        target_class_labels.append(class_idx)

                # Validation
                if not all_train_paths:
                    st.error("❌ No training images found in manifest (missing 'train' purpose)")
                    if temp_dir:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    return

                if not target_paths:
                    st.error("❌ No target images found in manifest (missing 'target' purpose)")
                    if temp_dir:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    return

                if not selection_pool_paths:
                    st.warning("⚠️ No selection pool images found in manifest (missing 'select' purpose), using all training images")
                    selection_pool_paths = all_train_paths.copy()

                # Show loaded dataset info
                st.success(f"📊 Loaded dataset from manifest:")
                dataset_info = manifest.get('dataset_info', {})
                if dataset_info:
                    st.write(f"  - **Dataset**: {dataset_info.get('name', 'Unknown')}")
                    st.write(f"  - **Description**: {dataset_info.get('description', 'N/A')}")

                st.write(f"\n**Dataset breakdown:**")
                st.write(f"  - **Training images** (for initial model): {len(all_train_paths)} images")

                # Count by class in training set
                from collections import Counter
                train_class_counts = Counter(all_train_class_labels)

                # Get target class info from manifest
                target_class_idx = dataset_info.get('target_class', 5)
                target_class_name = dataset_info.get('target_class_name', 'target class')

                st.write(f"    - {target_class_name.capitalize()} images (class {target_class_idx}): {train_class_counts.get(target_class_idx, 0)}")
                st.write(f"    - Confuser images (other classes): {len(all_train_paths) - train_class_counts.get(target_class_idx, 0)}")

                # Show full class distribution in expander
                with st.expander("📊 Full class distribution in training set"):
                    for class_id in sorted(train_class_counts.keys()):
                        count = train_class_counts[class_id]
                        marker = "🎯" if class_id == target_class_idx else "  "
                        st.write(f"{marker} Class {class_id}: {count} images")

                st.write(f"  - **Target images** (for evaluation): {len(target_paths)} images")
                st.write(f"  - **Selection pool** (for TRAK selection): {len(selection_pool_paths)} images")

                # Show statistics by source if available
                stats = manifest.get('statistics', {})
                if stats.get('by_source'):
                    st.write(f"\n**By source:**")
                    for source, count in stats['by_source'].items():
                        st.write(f"  - {source}: {count}")

                pool_label = "Selection pool"

                # Create progress tracking
                st.markdown("---")
                st.markdown("### 🔄 Processing Pipeline")

                progress_bar = st.progress(0)
                status_text = st.empty()

                # STEP 1: Train Initial Model
                status_text.markdown("**Step 1/3:** 🎓 Training initial model on all training data...")
                progress_bar.progress(0)

                # Load model
                with st.spinner("Loading base model..."):
                    model_config = st.session_state.get("model_config", {})
                    model_name = model_config.get("model_name", "vit_l_16")
                    custom_weights = model_config.get("custom_weights_path")

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = load_classification_model_for_images(
                        model_name,
                        custom_weights_path=custom_weights,
                        device=device
                    )

                # Helper function to load and preprocess images
                def load_and_preprocess_images(image_paths, label_name):
                    from torchvision import transforms

                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

                    images_tensor = []
                    valid_paths = []

                    for img_path in image_paths:
                        try:
                            img = Image.open(img_path).convert('RGB')
                            img_tensor = preprocess(img)
                            images_tensor.append(img_tensor)
                            valid_paths.append(img_path)
                        except Exception as e:
                            st.warning(f"⚠️ Could not load {os.path.basename(img_path)} from {label_name}: {str(e)}")

                    if not images_tensor:
                        st.error(f"❌ No valid images in {label_name}")
                        return None, None

                    return torch.stack(images_tensor), valid_paths

                # Preprocess all datasets
                progress_bar.progress(5)
                with st.spinner("Preprocessing training images..."):
                    all_train_tensors, all_train_valid_paths = load_and_preprocess_images(all_train_paths, "training set")
                    if all_train_tensors is None:
                        return
                    # Use actual class labels from manifest
                    all_train_labels = torch.tensor(all_train_class_labels, dtype=torch.long)

                    # Separate confusers from target class for Step 3 retraining
                    target_class_idx = dataset_info.get('target_class', 0)
                    confuser_mask = all_train_labels != target_class_idx
                    confuser_tensors = all_train_tensors[confuser_mask]
                    confuser_labels = all_train_labels[confuser_mask]
                    confuser_paths = [p for i, p in enumerate(all_train_valid_paths) if confuser_mask[i]]

                progress_bar.progress(10)
                with st.spinner("Preprocessing selection pool images..."):
                    selection_tensors, selection_valid_paths = load_and_preprocess_images(selection_pool_paths, pool_label)
                    if selection_tensors is None:
                        return
                    # Use actual class labels from manifest
                    selection_labels = torch.tensor(selection_pool_class_labels, dtype=torch.long)

                progress_bar.progress(15)
                with st.spinner("Preprocessing target images..."):
                    target_tensors, target_valid_paths = load_and_preprocess_images(target_paths, "target set")
                    if target_tensors is None:
                        return
                    # Use actual class labels from manifest
                    target_labels = torch.tensor(target_class_labels, dtype=torch.long)

                # Train initial model
                progress_bar.progress(20)

                # Use all training images (including target class)
                train_tensors_for_model = all_train_tensors
                train_labels_for_model = all_train_labels

                with st.spinner(f"Training initial model for {train_epochs} epochs on {len(train_tensors_for_model)} images..."):
                    # Create simple training loop
                    from torch.utils.data import TensorDataset, DataLoader
                    import torch.optim as optim
                    import torch.nn as nn

                    train_dataset = TensorDataset(train_tensors_for_model, train_labels_for_model)
                    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

                    # Set up training mode based on unified TRAK config
                    model.train()

                    # Identify classifier layer (needed for both modes)
                    classifier_layer = None
                    classifier_name = None
                    if hasattr(model, 'fc'):
                        classifier_layer = model.fc
                        classifier_name = "model.fc"
                    elif hasattr(model, 'classifier'):
                        classifier_layer = model.classifier
                        classifier_name = "model.classifier"
                    elif hasattr(model, 'head'):
                        classifier_layer = model.head
                        classifier_name = "model.head"

                    if freeze_backbone:
                        # Freeze all parameters first
                        for param in model.parameters():
                            param.requires_grad = False

                        # Enable gradients only for the final classifier layer
                        if classifier_layer is not None:
                            for param in classifier_layer.parameters():
                                param.requires_grad = True
                            trainable_params = classifier_layer.parameters()
                            st.info(f"🔒 Backbone frozen, training classifier only ({classifier_name})")
                        else:
                            # Fallback: train all parameters
                            for param in model.parameters():
                                param.requires_grad = True
                            trainable_params = model.parameters()
                            st.warning("⚠️ Could not identify classifier layer, training entire model")
                    else:
                        # Full model training (all parameters)
                        for param in model.parameters():
                            param.requires_grad = True
                        trainable_params = model.parameters()
                        st.info("🔓 Training full model (all parameters)")

                    # Reset classifier weights if requested
                    if reset_classifier and classifier_layer is not None:
                        st.info(f"🔄 Resetting classifier weights ({classifier_name})...")
                        # Reinitialize classifier weights
                        for module in classifier_layer.modules():
                            if isinstance(module, nn.Linear):
                                nn.init.xavier_uniform_(module.weight)
                                if module.bias is not None:
                                    nn.init.zeros_(module.bias)
                        st.success(f"✅ Classifier weights reinitialized")

                    criterion = nn.CrossEntropyLoss()

                    # Create optimizer based on user selection
                    if step1_optimizer == "AdamW":
                        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=step1_weight_decay)
                    elif step1_optimizer == "Adam":
                        optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=step1_weight_decay)
                    else:  # SGD
                        optimizer = optim.SGD(trainable_params, lr=learning_rate, momentum=step1_momentum, weight_decay=step1_weight_decay)

                    # Get class mapping for nice output - prefer dataset.json manifest over model labels
                    # This ensures class indices in the dataset match the displayed class names
                    manifest_class_mapping = manifest.get('class_mapping', {})
                    if manifest_class_mapping:
                        # Use the dataset's own class mapping (convert int keys to str for consistent lookup)
                        class_mapping = {str(k): v for k, v in manifest_class_mapping.items()}
                    else:
                        # Fall back to model's sidebar custom labels if no dataset mapping
                        sidebar_labels = model_config.get("custom_labels")
                        if sidebar_labels:
                            class_mapping = {str(k): v for k, v in sidebar_labels.items()}
                        else:
                            class_mapping = {}

                    # Live-updating epoch status
                    epoch_status = st.empty()

                    # Calculate checkpoint epochs (evenly spaced)
                    checkpoint_epochs = set()
                    model_checkpoints = {}  # epoch -> state_dict
                    if num_checkpoints >= 1:
                        for i in range(num_checkpoints):
                            # Checkpoint at evenly spaced epochs (1-indexed)
                            ckpt_epoch = int(train_epochs * (i + 1) / num_checkpoints)
                            checkpoint_epochs.add(ckpt_epoch)

                    if num_checkpoints > 1:
                        st.info(f"📸 Will save checkpoints at epochs: {sorted(checkpoint_epochs)}")

                    for epoch in range(train_epochs):
                        # Training
                        model.train()
                        total_train_loss = 0
                        for batch_idx, (images, labels) in enumerate(train_loader):
                            images, labels = images.to(device), labels.to(device)

                            optimizer.zero_grad()
                            outputs = model(images)

                            # Handle both transformer models (return ImageClassifierOutput) and torchvision models (return tensor)
                            if hasattr(outputs, 'logits'):
                                logits = outputs.logits
                            else:
                                logits = outputs

                            loss = criterion(logits, labels)
                            loss.backward()
                            optimizer.step()

                            total_train_loss += loss.item()

                        avg_train_loss = total_train_loss / len(train_loader)

                        # Evaluate on target set
                        model.eval()
                        total_target_loss = 0
                        correct = 0
                        total = 0
                        all_predictions = []

                        with torch.no_grad():
                            for i in range(0, len(target_tensors), train_batch_size):
                                batch_images = target_tensors[i:i+train_batch_size].to(device)
                                batch_labels = target_labels[i:i+train_batch_size].to(device)

                                outputs = model(batch_images)
                                if hasattr(outputs, 'logits'):
                                    logits = outputs.logits
                                else:
                                    logits = outputs

                                loss = criterion(logits, batch_labels)
                                total_target_loss += loss.item()

                                _, predicted = torch.max(logits, 1)
                                total += batch_labels.size(0)
                                correct += (predicted == batch_labels).sum().item()

                                all_predictions.extend(predicted.cpu().tolist())

                        avg_target_loss = total_target_loss / ((len(target_tensors) + train_batch_size - 1) // train_batch_size)
                        target_acc = 100 * correct / total

                        # Save checkpoint if this epoch is a checkpoint epoch (1-indexed)
                        current_epoch_1indexed = epoch + 1
                        if current_epoch_1indexed in checkpoint_epochs:
                            # Deep copy the state dict to CPU
                            checkpoint_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                            model_checkpoints[current_epoch_1indexed] = checkpoint_state
                            epoch_status.text(f"  Epoch {epoch+1}/{train_epochs}: Train Loss = {avg_train_loss:.4f} | Target Loss = {avg_target_loss:.4f} | Target Acc = {target_acc:.1f}% 📸 checkpoint saved")
                        else:
                            # Update live status (overwrites previous)
                            epoch_status.text(f"  Epoch {epoch+1}/{train_epochs}: Train Loss = {avg_train_loss:.4f} | Target Loss = {avg_target_loss:.4f} | Target Acc = {target_acc:.1f}%")

                    model.eval()

                    # Log checkpoint summary
                    if len(model_checkpoints) > 1:
                        st.success(f"✅ Saved {len(model_checkpoints)} checkpoints at epochs: {sorted(model_checkpoints.keys())}")

                    # Final results with top 3 predictions using class names
                    from collections import Counter
                    pred_counts = Counter(all_predictions)
                    top_3 = pred_counts.most_common(3)

                    # Format top 3 with class names
                    top_3_str = ", ".join([
                        f"{class_mapping.get(str(cls), f'class_{cls}')} ({count})"
                        for cls, count in top_3
                    ])

                    epoch_status.empty()  # Clear the live status
                    st.write(f"  **Final**: Train Loss = {avg_train_loss:.4f} | Target Loss = {avg_target_loss:.4f} | Target Acc = {target_acc:.1f}%")
                    st.write(f"  **Predictions**: {top_3_str}")

                    # Debug: Show raw class indices to help diagnose mismatches
                    with st.expander("🔍 Debug: Class mapping details", expanded=False):
                        unique_labels = set(target_labels.tolist())
                        unique_preds = set(all_predictions)
                        st.write(f"**Target labels (ground truth):** {sorted(unique_labels)} → should be class {dataset_info.get('target_class', '?')} ({dataset_info.get('target_class_name', '?')})")
                        st.write(f"**Model predictions (unique):** {sorted(unique_preds)}")
                        st.write(f"**Class mapping being used:**")
                        for k, v in sorted(class_mapping.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
                            marker = "← target" if str(k) == str(dataset_info.get('target_class', -1)) else ""
                            st.write(f"  Class {k}: {v} {marker}")

                progress_bar.progress(100)
                st.success(f"✅ Step 1 Complete: Trained initial model for {train_epochs} epochs")

                # Save Step 1 data to session state for Step 2
                # If we have multiple checkpoints, store them all; otherwise just store final
                if len(model_checkpoints) > 1:
                    checkpoints_to_save = model_checkpoints
                else:
                    # Single checkpoint = final model state
                    checkpoints_to_save = {train_epochs: {k: v.cpu() for k, v in model.state_dict().items()}}

                st.session_state["step1_data"] = {
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},  # Keep for backward compat
                    "model_checkpoints": checkpoints_to_save,  # New: all checkpoints
                    "selection_tensors": selection_tensors.cpu(),
                    "selection_labels": selection_labels.cpu(),
                    "selection_valid_paths": selection_valid_paths,
                    "target_tensors": target_tensors.cpu(),
                    "target_labels": target_labels.cpu(),
                    "confuser_tensors": confuser_tensors.cpu(),
                    "confuser_labels": confuser_labels.cpu(),
                    "confuser_paths": confuser_paths,
                    "manifest": manifest,
                    "model_name": model_name,
                    "custom_weights": custom_weights,
                    "device": str(device),
                    "train_batch_size": train_batch_size,
                    "train_epochs": train_epochs,
                    "pool_label": pool_label,
                    "temp_dir": temp_dir,
                    "dataset_root": dataset_root,
                    "class_mapping": class_mapping,
                    "dataset_info": dataset_info,
                    "num_checkpoints": len(checkpoints_to_save),
                }

                st.info("💾 Model and data saved. Click **Step 2** to compute TRAK scores.")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during Step 1: {str(e)}")
                st.error(traceback.format_exc())
                return

    # ==================== STEP 2: COMPUTE TRAK SCORES ====================
    step2_button_disabled = not step1_complete or step2_complete

    if step1_complete and not step2_complete:
        st.markdown("#### 📊 Step 2: Compute TRAK Scores")
        # Show checkpoint info from Step 1 if available
        saved_checkpoints = st.session_state.get("step1_data", {}).get("num_checkpoints", 1)
        ckpt_info = f", {saved_checkpoints} checkpoint(s)" if saved_checkpoints > 1 else ""
        st.caption(f"Using JL dim={jl_dim}, {num_projections} projection(s){ckpt_info}, gradient source: {gradient_source}")

    if st.button("📊 **Step 2: Compute TRAK Scores**", type="primary", use_container_width=True, disabled=step2_button_disabled):
        with st.spinner("Computing TRAK scores..."):
            try:
                # Load data from Step 1
                step1_data = st.session_state["step1_data"]
                model_state_dict = step1_data["model_state_dict"]
                selection_tensors = step1_data["selection_tensors"]
                selection_labels = step1_data["selection_labels"]
                selection_valid_paths = step1_data["selection_valid_paths"]
                target_tensors = step1_data["target_tensors"]
                target_labels = step1_data["target_labels"]
                confuser_tensors = step1_data["confuser_tensors"]
                confuser_labels = step1_data["confuser_labels"]
                confuser_paths = step1_data["confuser_paths"]
                manifest = step1_data["manifest"]
                model_name = step1_data["model_name"]
                custom_weights = step1_data["custom_weights"]
                device = torch.device(step1_data["device"])
                train_batch_size = step1_data["train_batch_size"]
                train_epochs = step1_data["train_epochs"]
                pool_label = step1_data["pool_label"]
                temp_dir = step1_data["temp_dir"]
                dataset_root = step1_data["dataset_root"]
                class_mapping = step1_data["class_mapping"]
                dataset_info = step1_data["dataset_info"]
                # NOTE: jl_dim and num_projections come from sliders (defined above), not step1_data
                # This allows users to modify TRAK parameters after training without retraining

                # Get checkpoints from step1_data (support both old and new format)
                model_checkpoints = step1_data.get("model_checkpoints", {train_epochs: model_state_dict})
                num_ckpts = len(model_checkpoints)
                checkpoint_epochs_list = sorted(model_checkpoints.keys())

                # Move tensors to device
                selection_tensors = selection_tensors.to(device)
                selection_labels = selection_labels.to(device)
                target_tensors = target_tensors.to(device)
                target_labels = target_labels.to(device)

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.markdown("**Step 2/3:** 📊 Computing gradients and TRAK influence scores...")
                progress_bar.progress(5)

                total_computations = num_ckpts * num_projections
                st.info(f"🔄 Computing TRAK scores: {num_ckpts} checkpoint(s) × {num_projections} projection(s) = {total_computations} computation(s) (JL dim={jl_dim})")

                overall_progress = st.progress(0)
                status_text_proj = st.empty()

                # ===== ENSEMBLE TRAK: Loop over checkpoints and projections =====
                all_scores = []  # Will hold scores from all (checkpoint, projection) combinations

                for ckpt_idx, ckpt_epoch in enumerate(checkpoint_epochs_list):
                    ckpt_label = f"Checkpoint {ckpt_idx + 1}/{num_ckpts} (epoch {ckpt_epoch})"

                    # Load model with this checkpoint's weights
                    status_text_proj.text(f"{ckpt_label}: Loading model...")
                    model = load_classification_model_for_images(
                        model_name,
                        custom_weights_path=custom_weights,
                        device=device
                    )
                    ckpt_state_dict = model_checkpoints[ckpt_epoch]
                    model.load_state_dict({k: v.to(device) for k, v in ckpt_state_dict.items()})
                    model.eval()

                    # Calculate base progress for this checkpoint
                    ckpt_base_pct = int((ckpt_idx / num_ckpts) * 90)
                    ckpt_range = int(90 / num_ckpts)

                    # ===== Extract raw gradients for this checkpoint (expensive) =====
                    status_text_proj.text(f"{ckpt_label}: Extracting selection gradients...")
                    train_grad_progress = st.empty()

                    def train_progress_callback(current, total):
                        train_grad_progress.text(f"  Processing: {current}/{total} images")

                    selection_grads_raw = extract_gradients_from_images(
                        model,
                        selection_tensors,
                        selection_labels,
                        device=str(device),
                        batch_size=train_batch_size,
                        progress_callback=train_progress_callback,
                        project_dim=None,
                        projection_seed=None,
                        last_layer_only=freeze_backbone
                    )
                    train_grad_progress.empty()
                    overall_progress.progress(ckpt_base_pct + int(ckpt_range * 0.3))

                    # Free GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                    status_text_proj.text(f"{ckpt_label}: Extracting target gradients...")
                    target_grad_progress = st.empty()

                    def target_progress_callback(current, total):
                        target_grad_progress.text(f"  Processing: {current}/{total} images")

                    target_grads_raw = extract_gradients_from_images(
                        model,
                        target_tensors,
                        target_labels,
                        device=str(device),
                        batch_size=train_batch_size,
                        progress_callback=target_progress_callback,
                        project_dim=None,
                        projection_seed=None,
                        last_layer_only=freeze_backbone
                    )
                    target_grad_progress.empty()
                    overall_progress.progress(ckpt_base_pct + int(ckpt_range * 0.5))

                    # Free GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                    # ===== Apply projections and compute TRAK scores for this checkpoint =====
                    for proj_idx in range(num_projections):
                        proj_seed = 42 + proj_idx
                        proj_label = f"Proj {proj_idx + 1}/{num_projections}" if num_projections > 1 else ""
                        combined_label = f"{ckpt_label} | {proj_label}".strip(" |")

                        # Calculate progress within this checkpoint
                        proj_pct = int(ckpt_range * 0.5) + int((proj_idx / num_projections) * ckpt_range * 0.5)

                        # Project gradients
                        status_text_proj.text(f"{combined_label}: Projecting gradients (→{jl_dim}D)...")
                        selection_grads = project_gradients_random(selection_grads_raw, target_dim=jl_dim, seed=proj_seed)
                        target_grads = project_gradients_random(target_grads_raw, target_dim=jl_dim, seed=proj_seed)

                        # Compute TRAK scores
                        status_text_proj.text(f"{combined_label}: Computing influence scores...")
                        trak_status = st.empty()

                        def trak_progress_callback(step_name, pct):
                            trak_status.text(f"  {step_name}")

                        scores = compute_trak_scores_simple(
                            selection_grads,
                            target_grads,
                            lambda_reg=0.001,
                            device=str(device),
                            project_dim=jl_dim,
                            num_projections=1,
                            progress_callback=trak_progress_callback
                        )
                        trak_status.empty()
                        all_scores.append(scores)

                        overall_progress.progress(ckpt_base_pct + proj_pct)

                        # Free projected gradients
                        del selection_grads, target_grads

                    # Free raw gradients for this checkpoint
                    del selection_grads_raw, target_grads_raw, model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                # Average scores across all (checkpoints × projections)
                status_text_proj.text(f"Averaging {len(all_scores)} score computations...")
                scores = torch.stack(all_scores).mean(dim=0)

                # Clean up progress elements
                overall_progress.progress(90)
                overall_progress.empty()
                status_text_proj.empty()

                # Sum scores across all target samples to get total influence per selection candidate
                importance_scores = scores.sum(dim=1)

                progress_bar.progress(80)

                # Show TRAK computation summary with debug info
                score_min, score_max = importance_scores.min().item(), importance_scores.max().item()
                score_mean = importance_scores.mean().item()
                st.success(f"✅ Step 2 Complete: Computed TRAK scores for {len(selection_valid_paths)} images")
                with st.expander("📊 TRAK Score Statistics", expanded=False):
                    st.write(f"**Score matrix shape:** {scores.shape[0]} train × {scores.shape[1]} target")
                    st.write(f"**Aggregated scores:** min={score_min:.4f}, max={score_max:.4f}, mean={score_mean:.4f}")
                    st.write(f"**Score range:** {score_max - score_min:.4f} (higher range = better differentiation)")
                    # Show ensemble details
                    if num_ckpts > 1 or num_projections > 1:
                        st.write(f"**Ensemble configuration:**")
                        if num_ckpts > 1:
                            st.write(f"  • Checkpoints: {num_ckpts} (epochs {checkpoint_epochs_list})")
                        if num_projections > 1:
                            st.write(f"  • Projections: {num_projections}")
                        st.write(f"  • Total computations averaged: {len(all_scores)}")

                # Prepare scores dataframe with class information
                # Get class labels for each image in selection pool
                selection_class_indices = selection_labels.cpu().numpy().tolist()
                selection_class_names = [
                    class_mapping.get(str(idx), f"class_{idx}") for idx in selection_class_indices
                ]

                scores_data = {
                    "filename": [os.path.basename(p) for p in selection_valid_paths],
                    "trak_score": importance_scores.cpu().numpy().tolist(),
                    "class": selection_class_names,
                    "class_idx": selection_class_indices,
                    "source": [next((e['source'] for e in manifest['entries'] if e['filename'] in p), 'unknown')
                               for p in selection_valid_paths]
                }
                import pandas as pd
                scores_df = pd.DataFrame(scores_data)
                scores_df_sorted = scores_df.sort_values('trak_score', ascending=False).reset_index(drop=True)

                # Show class distribution in selection pool
                class_counts = scores_df['class'].value_counts()
                st.write(f"**Selection pool class distribution:** {dict(class_counts)}")

                # Store data in session state for Step 3
                st.session_state["trak_data"] = {
                    "importance_scores": importance_scores.cpu(),
                    "selection_tensors": selection_tensors.cpu(),
                    "selection_labels": selection_labels.cpu(),
                    "selection_valid_paths": selection_valid_paths,
                    "target_tensors": target_tensors.cpu(),  # For target evaluation during retraining
                    "target_labels": target_labels.cpu(),    # For target evaluation during retraining
                    "confuser_tensors": confuser_tensors.cpu(),  # Confusers to add back in Step 3
                    "confuser_labels": confuser_labels.cpu(),    # Confuser labels
                    "confuser_paths": confuser_paths,            # Confuser paths for display
                    "manifest": manifest,
                    "model_name": model_name,
                    "custom_weights": custom_weights,
                    "device": str(device),
                    "train_batch_size": train_batch_size,
                    "train_epochs": train_epochs,
                    "pool_label": pool_label,
                    "temp_dir": temp_dir,  # None if using local path
                    "dataset_root": dataset_root,  # Where dataset.json lives
                    "scores_df_sorted": scores_df_sorted,  # For image display
                    "jl_dim": jl_dim,  # TRAK projection dimension
                    "num_projections": num_projections,  # Number of projections averaged
                    "class_mapping": class_mapping,  # Class name mapping from dataset
                }

                st.success("✅ Step 2 Complete! TRAK scores computed.")
                st.info("💾 Results saved. You can now proceed to **Step 3** to select and retrain.")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during Step 2: {str(e)}")
                st.error(traceback.format_exc())
                return

    # STEP 2.5: Display TRAK Scores (separate from main button to prevent reset on interaction)
    if "trak_data" in st.session_state:
        trak_data = st.session_state["trak_data"]
        scores_df_sorted = trak_data.get("scores_df_sorted")
        selection_valid_paths = trak_data.get("selection_valid_paths")
        manifest = trak_data.get("manifest")

        if scores_df_sorted is not None:
            st.markdown("---")
            st.markdown("### 📊 Step 2: TRAK Scores Results")

            # View mode and K selection
            col_view, col_k = st.columns([2, 1])

            with col_view:
                view_mode = st.radio(
                    "View mode",
                    ["Top K (highest influence)", "Bottom K (lowest influence)", "All images"],
                    horizontal=True,
                    key="trak_view_mode",
                    help="Select which images to visualize"
                )

            with col_k:
                if view_mode != "All images":
                    k_value = st.number_input(
                        "K value",
                        min_value=1,
                        max_value=len(scores_df_sorted),
                        value=min(10, len(scores_df_sorted)),
                        key="trak_k_value",
                        help="Number of images to show"
                    )
                else:
                    k_value = len(scores_df_sorted)

            # Download buttons
            col_dl_csv, col_dl_json = st.columns(2)

            with col_dl_csv:
                scores_csv = scores_df_sorted.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=scores_csv,
                    file_name="trak_scores.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col_dl_json:
                # Create enriched manifest with TRAK scores
                # Build a lookup from filename to trak_score
                score_lookup = dict(zip(scores_df_sorted['filename'], scores_df_sorted['trak_score']))

                # Deep copy manifest and add trak_score to each entry
                import copy
                enriched_manifest = copy.deepcopy(manifest)
                for entry in enriched_manifest.get('entries', []):
                    filename = os.path.basename(entry.get('filename', ''))
                    entry['trak_score'] = score_lookup.get(filename, None)

                # Add metadata about TRAK computation
                enriched_manifest['trak_metadata'] = {
                    'computed_at': datetime.now().isoformat(),
                    'num_scored_images': len(scores_df_sorted),
                    'score_range': {
                        'min': float(scores_df_sorted['trak_score'].min()),
                        'max': float(scores_df_sorted['trak_score'].max()),
                        'mean': float(scores_df_sorted['trak_score'].mean()),
                    }
                }

                manifest_json = json.dumps(enriched_manifest, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=manifest_json,
                    file_name="dataset_with_trak_scores.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Original manifest with trak_score added to each entry"
                )

            # Filter dataframe based on view mode
            # IMPORTANT: reset_index(drop=True) ensures row.name matches position for highlighting
            if view_mode == "Top K (highest influence)":
                display_df = scores_df_sorted.head(k_value).copy().reset_index(drop=True)
                display_df['rank'] = range(1, len(display_df) + 1)
            elif view_mode == "Bottom K (lowest influence)":
                display_df = scores_df_sorted.tail(k_value).copy().reset_index(drop=True)
                display_df['rank'] = range(len(scores_df_sorted) - k_value + 1, len(scores_df_sorted) + 1)
            else:
                display_df = scores_df_sorted.copy().reset_index(drop=True)
                display_df['rank'] = range(1, len(display_df) + 1)

            # Initialize carousel index BEFORE building the table (so we can highlight current row)
            st.session_state.setdefault("trak_carousel_index", 0)
            if st.session_state["trak_carousel_index"] >= len(display_df):
                st.session_state["trak_carousel_index"] = 0
            carousel_idx = st.session_state["trak_carousel_index"]

            # Add visual indicator column for current selection
            display_df[''] = ['▶' if i == carousel_idx else '' for i in range(len(display_df))]

            # Reorder columns to show indicator first, then rank
            cols = ['', 'rank'] + [c for c in display_df.columns if c not in ['', 'rank']]
            display_df = display_df[cols]

            st.write(f"**Showing {len(display_df)} of {len(scores_df_sorted)} images** ({view_mode})")

            # Two-panel layout: Table on left, Visualizer on right
            col_table, col_viz = st.columns([1, 1])

            with col_table:
                st.markdown("#### 📋 Score Table")
                st.caption("💡 Click a row to view in carousel, or use navigation buttons")

                # Style the dataframe to highlight current row
                def highlight_current_row(row):
                    if row.name == carousel_idx:
                        return ['background-color: #1a472a; color: white'] * len(row)
                    return [''] * len(row)

                styled_df = display_df.style.apply(highlight_current_row, axis=1)

                # Display the table with selection and styling
                selected_rows = st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=450,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="trak_scores_table"
                )

                # Handle table selection - sync with carousel
                if selected_rows and "selection" in selected_rows and "rows" in selected_rows["selection"]:
                    selected_indices = selected_rows["selection"]["rows"]
                    if selected_indices:
                        table_idx = selected_indices[0]
                        if st.session_state.get("trak_carousel_index") != table_idx:
                            st.session_state["trak_carousel_index"] = table_idx
                            st.rerun()

            with col_viz:
                st.markdown("#### 🖼️ Image Carousel")

                # Navigation buttons
                nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                with nav_col1:
                    if st.button("◀️ Previous", key="trak_prev", use_container_width=True):
                        st.session_state["trak_carousel_index"] = (carousel_idx - 1) % len(display_df)
                        st.rerun()
                with nav_col2:
                    st.markdown(f"<div style='text-align: center'><b>Image {carousel_idx + 1} of {len(display_df)}</b></div>", unsafe_allow_html=True)
                with nav_col3:
                    if st.button("Next ▶️", key="trak_next", use_container_width=True):
                        st.session_state["trak_carousel_index"] = (carousel_idx + 1) % len(display_df)
                        st.rerun()

                # Display current image
                current_row = display_df.iloc[carousel_idx]
                current_filename = current_row['filename']
                current_score = current_row['trak_score']
                current_source = current_row['source']
                current_rank = current_row['rank']

                # Find and display image
                image_found = False
                for path in selection_valid_paths:
                    if current_filename in path:
                        try:
                            img = Image.open(path)
                            st.image(img, caption=f"#{current_rank}: {current_filename}", use_container_width=True)
                            image_found = True

                            # Score info below image
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric("TRAK Score", f"{current_score:.4f}")
                            with metric_col2:
                                st.metric("Global Rank", f"#{current_rank} / {len(scores_df_sorted)}")

                            st.caption(f"**Source:** {current_source} | **File:** {current_filename}")

                        except Exception as e:
                            st.error(f"Could not load image: {e}")
                        break

                if not image_found:
                    st.warning(f"Could not find image: {current_filename}")

            # Grid view expander
            with st.expander(f"🖼️ Grid View ({len(display_df)} images)", expanded=False):
                cols_per_row = st.slider("Images per row", 2, 6, 4, key="trak_grid_cols")

                for i in range(0, len(display_df), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(display_df):
                            row = display_df.iloc[i + j]
                            filename = row['filename']
                            score = row['trak_score']
                            rank = row['rank']

                            # Find image path
                            for path in selection_valid_paths:
                                if filename in path:
                                    try:
                                        img = Image.open(path)
                                        with col:
                                            st.image(img, caption=f"#{rank}", use_container_width=True)
                                            st.caption(f"Score: {score:.3f}")
                                    except:
                                        with col:
                                            st.warning(f"#{rank}: Load error")
                                    break

    # STEP 3: Select Subset and Retrain (separate from main button to prevent reset)
    if "trak_data" in st.session_state:
        st.markdown("---")
        st.markdown("### 🎯 Step 3: Select Training Subset & Retrain")

        trak_data = st.session_state["trak_data"]
        importance_scores = trak_data["importance_scores"]
        selection_tensors = trak_data["selection_tensors"]
        selection_labels = trak_data["selection_labels"]
        selection_valid_paths = trak_data["selection_valid_paths"]

        # Target tensors for evaluation (may not exist in old session data)
        target_tensors = trak_data.get("target_tensors")
        target_labels = trak_data.get("target_labels")
        has_target_data = target_tensors is not None and target_labels is not None

        if not has_target_data:
            st.warning("⚠️ Target evaluation data not available. Re-run Step 1 to enable target loss/accuracy tracking during retraining.")

        manifest = trak_data["manifest"]
        model_name = trak_data["model_name"]
        custom_weights = trak_data["custom_weights"]
        device = torch.device(trak_data["device"])
        train_batch_size = trak_data["train_batch_size"]
        train_epochs = trak_data["train_epochs"]
        pool_label = trak_data["pool_label"]

        # Confuser data for balanced retraining
        confuser_tensors = trak_data.get("confuser_tensors")
        confuser_labels = trak_data.get("confuser_labels")
        has_confuser_data = confuser_tensors is not None and confuser_labels is not None
        n_confusers_total = len(confuser_tensors) if has_confuser_data else 0

        # Selection parameters (moved here from initial config since TRAK scoring is independent)
        st.markdown("#### 📊 Selection Parameters")
        col_method, col_value, col_strategy = st.columns(3)

        with col_method:
            selection_method = st.radio(
                "Selection method",
                ["Top percentage", "Top count"],
                horizontal=True,
                key="step3_selection_method",
                help="Choose how to select images"
            )

        with col_value:
            if selection_method == "Top percentage":
                selection_pct = st.slider(
                    "Percentage to keep",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=1,
                    key="step3_selection_pct",
                    help="Percentage of most important images to keep"
                )
                n_select = max(1, int(len(selection_valid_paths) * selection_pct / 100.0))
            else:
                n_select = st.number_input(
                    "Number of images",
                    min_value=1,
                    max_value=len(selection_valid_paths),
                    value=min(10, len(selection_valid_paths)),
                    step=1,
                    key="step3_selection_count",
                    help="Number of images to keep"
                )

        with col_strategy:
            subset_method = st.radio(
                "Selection strategy",
                ["Top (highest influence)", "Bottom (lowest influence)", "Random"],
                key="step3_subset_method",
                help="Which images to select based on TRAK scores"
            )

        # Show selection summary
        pct_of_total = 100 * n_select / len(selection_valid_paths)
        # Calculate proportional confusers
        n_confusers_select = int(n_confusers_total * pct_of_total / 100) if has_confuser_data else 0
        n_confusers_select = max(1, n_confusers_select) if has_confuser_data else 0  # At least 1 confuser if available

        if has_confuser_data:
            st.info(f"📌 Will select **{n_select}** target images ({pct_of_total:.1f}%) + **{n_confusers_select}** random confusers ({pct_of_total:.1f}% of {n_confusers_total}) = **{n_select + n_confusers_select}** total")
        else:
            st.info(f"📌 Will select **{n_select}** images ({pct_of_total:.1f}% of {len(selection_valid_paths)}) using: **{subset_method}**")
            st.warning("⚠️ No confuser data available. Re-run Step 1 to include confusers in retraining.")

        # Training Parameters for Step 3 (can override Step 1 defaults)
        st.markdown("#### ⚙️ Training Parameters")
        col_epochs, col_bs, col_lr = st.columns(3)

        with col_epochs:
            step3_epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=500,
                value=train_epochs,  # Default from Step 1
                step=1,
                key="step3_train_epochs",
                help="Number of training epochs (default from Step 1)"
            )

        with col_bs:
            step3_batch_size = st.number_input(
                "Batch size",
                min_value=1,
                max_value=256,
                value=train_batch_size,  # Default from Step 1
                step=1,
                key="step3_batch_size",
                help="Training batch size (default from Step 1)"
            )

        with col_lr:
            step3_lr = st.select_slider(
                "Learning rate",
                options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                value=1e-4,
                format_func=lambda x: f"{x:.0e}",
                key="step3_learning_rate",
                help="Optimizer learning rate"
            )

        # Additional training options (hidden in expander)
        with st.expander("🔧 Training Options", expanded=False):
            optimizer_choice = st.selectbox(
                "Optimizer",
                options=["AdamW", "Adam", "SGD"],
                index=0,
                key="step3_optimizer",
                help="Optimization algorithm"
            )

            # Show current gradient source setting (from unified TRAK config)
            training_layer_mode = "Last layer only (frozen backbone)" if freeze_backbone else "Full model"
            st.caption(f"📌 Training mode: **{training_layer_mode}** (set in TRAK Configuration above)")

            # Fixed defaults for momentum and weight decay
            sgd_momentum = 0.9
            weight_decay = 0.01

        # Show training config summary
        training_mode = "classifier only (frozen backbone)" if freeze_backbone else "full model"
        st.caption(f"🔧 Training config: **{step3_epochs}** epochs, batch size **{step3_batch_size}**, lr **{step3_lr:.0e}**, {optimizer_choice}, {training_mode}")

        # Comparison mode: run all three methods and compare results
        col_retrain, col_compare = st.columns([1, 1])

        with col_compare:
            compare_btn = st.button("🔬 Compare All Methods", type="secondary", use_container_width=True, help="Run Top, Bottom, and Random selection methods and compare their results")

        if compare_btn:
            st.markdown("### 🔬 Method Comparison")
            st.info(f"Training and evaluating all three selection methods with **{selection_pct}%** of {pool_label} ({n_select} images each)")

            comparison_results = []
            methods = [
                ("Top (highest influence)", True),
                ("Bottom (lowest influence)", False),
                ("Random", None),
            ]

            overall_progress = st.progress(0)
            method_status = st.empty()

            for method_idx, (method_name, descending) in enumerate(methods):
                method_status.markdown(f"**Running {method_name}...**")

                # Select indices based on method
                if descending is not None:
                    selected_indices = torch.argsort(importance_scores, descending=descending)[:n_select]
                else:  # Random
                    import random
                    all_indices = list(range(len(selection_valid_paths)))
                    random.shuffle(all_indices)
                    selected_indices = torch.tensor(all_indices[:n_select])

                # Get subset data
                method_subset_tensors = selection_tensors[selected_indices]
                method_subset_labels = selection_labels[selected_indices]

                # Add confusers if applicable
                if has_confuser_data and n_confusers_select > 0:
                    import random
                    confuser_indices = list(range(n_confusers_total))
                    random.shuffle(confuser_indices)
                    selected_confuser_indices = confuser_indices[:n_confusers_select]

                    selected_confuser_tensors = confuser_tensors[selected_confuser_indices]
                    selected_confuser_labels = confuser_labels[selected_confuser_indices]

                    method_subset_tensors = torch.cat([method_subset_tensors, selected_confuser_tensors], dim=0)
                    method_subset_labels = torch.cat([method_subset_labels, selected_confuser_labels], dim=0)

                # Train and evaluate
                def progress_cb(epoch, total):
                    base_progress = method_idx / len(methods)
                    epoch_progress = epoch / total / len(methods)
                    overall_progress.progress(base_progress + epoch_progress)

                result = train_and_evaluate_subset(
                    model_name=model_name,
                    custom_weights=custom_weights,
                    subset_tensors=method_subset_tensors,
                    subset_labels=method_subset_labels,
                    target_tensors=target_tensors if has_target_data else None,
                    target_labels=target_labels if has_target_data else None,
                    epochs=step3_epochs,
                    batch_size=step3_batch_size,
                    lr=step3_lr,
                    optimizer_choice=optimizer_choice,
                    freeze_backbone=freeze_backbone,
                    device=device,
                    progress_callback=progress_cb,
                )

                comparison_results.append({
                    "Method": method_name,
                    "Train Loss": result["train_loss"],
                    "Target Loss": result["target_loss"],
                    "Target Acc (%)": result["target_acc"],
                })

            overall_progress.progress(1.0)
            method_status.markdown("**✅ Comparison Complete!**")

            # Display comparison table
            import pandas as pd
            df = pd.DataFrame(comparison_results)

            st.markdown("#### 📊 Comparison Results")
            st.dataframe(
                df.style.format({
                    "Train Loss": "{:.4f}",
                    "Target Loss": "{:.4f}",
                    "Target Acc (%)": "{:.2f}",
                }).highlight_min(subset=["Train Loss", "Target Loss"], color="lightgreen")
                 .highlight_max(subset=["Target Acc (%)"], color="lightgreen"),
                use_container_width=True,
                hide_index=True,
            )

            # Show which method performed best
            if has_target_data:
                best_idx = df["Target Acc (%)"].idxmax()
                best_method = df.loc[best_idx, "Method"]
                best_acc = df.loc[best_idx, "Target Acc (%)"]
                st.success(f"🏆 **Best Method**: {best_method} with **{best_acc:.2f}%** target accuracy")

            # Store comparison results in session state
            st.session_state["method_comparison_results"] = {
                "results": comparison_results,
                "selection_pct": selection_pct,
                "n_select": n_select,
                "epochs": step3_epochs,
            }

        with col_retrain:
            retrain_btn = st.button("🚀 Retrain Model on Selected Subset", type="primary", use_container_width=True)

        if retrain_btn:
            # Create new progress tracking for step 3
            st.markdown("### 🔄 Retraining Progress")
            progress_bar_retrain = st.progress(0)
            status_text_retrain = st.empty()

            status_text_retrain.markdown("**Step 3/3:** 🎯 Selecting subset and retraining model...")
            progress_bar_retrain.progress(5)

            # Select indices based on method (for target class images via TRAK)
            if subset_method == "Top (highest influence)":
                selected_indices = torch.argsort(importance_scores, descending=True)[:n_select]
            elif subset_method == "Bottom (lowest influence)":
                selected_indices = torch.argsort(importance_scores, descending=False)[:n_select]
            else:  # Random
                import random
                all_indices = list(range(len(selection_valid_paths)))
                random.shuffle(all_indices)
                selected_indices = torch.tensor(all_indices[:n_select])

            # Get TRAK-selected target class images
            subset_tensors = selection_tensors[selected_indices]
            subset_labels = selection_labels[selected_indices]

            # Add proportional random confusers to maintain class balance
            if has_confuser_data and n_confusers_select > 0:
                import random
                confuser_indices = list(range(n_confusers_total))
                random.shuffle(confuser_indices)
                selected_confuser_indices = confuser_indices[:n_confusers_select]

                selected_confuser_tensors = confuser_tensors[selected_confuser_indices]
                selected_confuser_labels = confuser_labels[selected_confuser_indices]

                # Combine target class + confusers
                subset_tensors = torch.cat([subset_tensors, selected_confuser_tensors], dim=0)
                subset_labels = torch.cat([subset_labels, selected_confuser_labels], dim=0)

                st.success(f"✅ Combined {n_select} TRAK-selected target images + {n_confusers_select} random confusers = {len(subset_tensors)} total")

            progress_bar_retrain.progress(10)

            # Retrain model
            with st.spinner(f"Retraining model on {len(subset_tensors)} selected images..."):
                from torch.utils.data import TensorDataset, DataLoader
                import torch.optim as optim
                import torch.nn as nn

                subset_dataset = TensorDataset(subset_tensors, subset_labels)
                subset_loader = DataLoader(subset_dataset, batch_size=step3_batch_size, shuffle=True)

                # Reset model (reload from scratch)
                model = load_classification_model_for_images(
                    model_name,
                    custom_weights_path=custom_weights,
                    device=device
                )

                # Setup training mode and parameters
                model.train()

                if freeze_backbone:
                    # Freeze all parameters first
                    for param in model.parameters():
                        param.requires_grad = False

                    # Enable gradients only for the final classifier layer
                    if hasattr(model, 'fc'):
                        for param in model.fc.parameters():
                            param.requires_grad = True
                        trainable_params = list(model.fc.parameters())
                    elif hasattr(model, 'classifier'):
                        for param in model.classifier.parameters():
                            param.requires_grad = True
                        trainable_params = list(model.classifier.parameters())
                    else:
                        # Fallback: train all parameters
                        for param in model.parameters():
                            param.requires_grad = True
                        trainable_params = list(model.parameters())
                else:
                    # Train all parameters (full fine-tuning)
                    for param in model.parameters():
                        param.requires_grad = True
                    trainable_params = list(model.parameters())

                # Create optimizer based on user selection
                if optimizer_choice == "AdamW":
                    optimizer = optim.AdamW(trainable_params, lr=step3_lr, weight_decay=weight_decay)
                elif optimizer_choice == "Adam":
                    optimizer = optim.Adam(trainable_params, lr=step3_lr, weight_decay=weight_decay)
                else:  # SGD
                    optimizer = optim.SGD(trainable_params, lr=step3_lr, momentum=sgd_momentum, weight_decay=weight_decay)

                criterion = nn.CrossEntropyLoss()

                progress_bar_retrain.progress(20)

                for epoch in range(step3_epochs):
                    # Training phase
                    model.train()
                    total_train_loss = 0
                    for batch_idx, (images, labels) in enumerate(subset_loader):
                        images, labels = images.to(device), labels.to(device)

                        optimizer.zero_grad()
                        outputs = model(images)

                        # Handle both transformer models (return ImageClassifierOutput) and torchvision models (return tensor)
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        else:
                            logits = outputs

                        loss = criterion(logits, labels)
                        loss.backward()
                        optimizer.step()

                        total_train_loss += loss.item()

                    avg_train_loss = total_train_loss / len(subset_loader)

                    # Evaluation phase on target set (if available)
                    if has_target_data:
                        model.eval()
                        total_target_loss = 0
                        correct = 0
                        total = 0

                        with torch.no_grad():
                            for i in range(0, len(target_tensors), step3_batch_size):
                                batch_images = target_tensors[i:i+step3_batch_size].to(device)
                                batch_labels = target_labels[i:i+step3_batch_size].to(device)

                                outputs = model(batch_images)
                                if hasattr(outputs, 'logits'):
                                    logits = outputs.logits
                                else:
                                    logits = outputs

                                loss = criterion(logits, batch_labels)
                                total_target_loss += loss.item()

                                _, predicted = torch.max(logits, 1)
                                total += batch_labels.size(0)
                                correct += (predicted == batch_labels).sum().item()

                        num_target_batches = (len(target_tensors) + step3_batch_size - 1) // step3_batch_size
                        avg_target_loss = total_target_loss / num_target_batches
                        target_acc = 100 * correct / total

                        if (epoch + 1) % max(1, step3_epochs // 5) == 0:
                            st.write(f"  Epoch {epoch+1}/{step3_epochs}: Train Loss = {avg_train_loss:.4f} | Target Loss = {avg_target_loss:.4f} | Target Acc = {target_acc:.2f}%")
                    else:
                        # No target data - just show train loss
                        if (epoch + 1) % max(1, step3_epochs // 5) == 0:
                            st.write(f"  Epoch {epoch+1}/{step3_epochs}: Train Loss = {avg_train_loss:.4f}")

                    # Update progress during training
                    progress_bar_retrain.progress(20 + int((epoch + 1) / step3_epochs * 80))

                model.eval()

            progress_bar_retrain.progress(100)
            status_text_retrain.markdown("**✅ All Steps Complete!**")
            st.success(f"✅ Step 3 Complete: Retrained model on {n_select} images using '{subset_method}' strategy")

            # Store selected images for display
            selected_paths = [selection_valid_paths[i] for i in selected_indices]
            selected_scores = [importance_scores[i].item() for i in selected_indices]

            selected_images = []
            selected_filenames = []

            for img_path in selected_paths:
                img = Image.open(img_path)
                img = img.convert('RGB')
                selected_images.append(img.copy())
                selected_filenames.append(os.path.basename(img_path))

            # Note: We keep temp_dir alive for the TRAK scores visualization
            # It will be cleaned up when starting a new analysis

            # Store results
            st.session_state["image_selection_results"] = {
                "selected_images": selected_images,
                "selected_filenames": selected_filenames,
                "selected_scores": selected_scores,
                "all_scores": importance_scores.tolist(),
                "n_total": len(selection_valid_paths),
                "n_selected": n_select,
                "pool_label": pool_label,
                "subset_method": subset_method
            }

    # Display results
    if "image_selection_results" in st.session_state:
        results = st.session_state["image_selection_results"]

        st.markdown("---")
        st.markdown("### 📊 Results")

        pool_label = results.get("pool_label", "Selection Pool")
        subset_method = results.get("subset_method", None)

        if subset_method:
            st.info(f"**Selection Strategy**: {subset_method} | **Pool**: {pool_label}")
        else:
            st.info(f"**Selection Pool**: {pool_label}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Total Images ({pool_label})", results["n_total"])
        with col2:
            st.metric("Selected Images", results["n_selected"])
        with col3:
            st.metric("Selection Rate", f"{results['n_selected']/results['n_total']*100:.1f}%")

        # Show selected images in expander
        with st.expander(f"🖼️ View Selected Images ({results['n_selected']} images)", expanded=False):
            st.info("Images are shown in order of importance (highest to lowest)")

            # Grid size selector
            cols_per_row = st.slider("Images per row", 2, 6, 4, key="selected_images_cols")

            # Display images in grid
            for i in range(0, len(results["selected_images"]), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(results["selected_images"]):
                        with cols[j]:
                            img = results["selected_images"][idx]
                            score = results["selected_scores"][idx]
                            filename = results["selected_filenames"][idx]

                            st.image(img, caption=f"#{idx+1}: {filename}", use_container_width=True)
                            st.caption(f"Score: {score:.4f}")

        # Download selected images
        st.markdown("### 💾 Download Selected Images")
        if st.button("📦 Create ZIP of Selected Images", use_container_width=True):
            with st.spinner("Creating ZIP file..."):
                # Create a new ZIP file with selected images
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                    for i, (img, filename) in enumerate(zip(results["selected_images"], results["selected_filenames"])):
                        # Save PIL image to bytes
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format='PNG')
                        img_bytes.seek(0)

                        # Add rank prefix
                        new_name = f"rank_{i+1:03d}_{filename}"
                        zip_out.writestr(new_name, img_bytes.getvalue())

                    # Add metadata JSON
                    metadata = {
                        "total_images": results["n_total"],
                        "selected_images": results["n_selected"],
                        "selection_rate": results["n_selected"] / results["n_total"],
                        "images": [
                            {
                                "rank": i+1,
                                "filename": results["selected_filenames"][i],
                                "importance_score": results["selected_scores"][i]
                            }
                            for i in range(len(results["selected_filenames"]))
                        ]
                    }
                    zip_out.writestr("selection_metadata.json", json.dumps(metadata, indent=2))

                zip_buffer.seek(0)

                st.download_button(
                    label="⬇️ Download Selected Images (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"selected_images_{results['n_selected']}_of_{results['n_total']}.zip",
                    mime="application/zip",
                    use_container_width=True
                )


# File Uploads & Cache
