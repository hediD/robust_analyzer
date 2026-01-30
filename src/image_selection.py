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
import io
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
    extract_and_project_gradients,
    project_gradients_random,
)
import base64
import random


def set_seed(seed: int = 42, cudnn_deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility across all random number generators.

    This ensures deterministic behavior for:
    - Python's random module (data shuffling, random selection)
    - NumPy's random number generator
    - PyTorch's CPU and CUDA random number generators (model initialization)

    Args:
        seed: The seed value to use (default: 42)
        cudnn_deterministic: If True, also enable cuDNN deterministic mode.
            This ensures bit-for-bit reproducibility but may reduce performance.
            Usually not necessary - data shuffling and init are the main variance sources.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # cuDNN determinism (optional - usually unnecessary, has performance cost)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch >= 1.8, enable deterministic algorithms globally
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass  # Some operations don't have deterministic implementations


def get_dataloader_generator(seed: int = 42) -> torch.Generator:
    """
    Create a seeded generator for DataLoader shuffling.

    Args:
        seed: The seed value to use

    Returns:
        A torch.Generator with the specified seed
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize worker seeds for deterministic multi-worker data loading.

    Each worker gets a unique but deterministic seed based on its ID and
    the initial seed set by the main process.

    Args:
        worker_id: The worker's ID (0 to num_workers-1)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def auto_download_json(data: dict, filename: str = "dataset.json") -> None:
    """
    Automatically trigger a file download in Streamlit using JavaScript.

    This creates a hidden anchor element with the data as a base64-encoded
    data URI and programmatically clicks it to trigger the download.

    Args:
        data: Dictionary to be downloaded as JSON
        filename: Name of the downloaded file
    """
    json_str = json.dumps(data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()

    # Create a unique ID for this download to avoid conflicts
    download_id = f"auto_download_{hash(json_str) % 10000}"

    # JavaScript to create and click a download link
    download_js = f'''
        <script>
            (function() {{
                // Only download once per page load
                if (window.{download_id}_downloaded) return;
                window.{download_id}_downloaded = true;

                var link = document.createElement('a');
                link.href = 'data:application/json;base64,{b64}';
                link.download = '{filename}';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }})();
        </script>
    '''

    st.components.v1.html(download_js, height=0)


def generate_dataset_json_from_folder(dataset_root: str, target_class_name: str = None) -> dict:
    """
    Auto-generate dataset.json from an ImageFolder-like directory structure.

    Expected structure:
        dataset_root/
        ├── train/           # Training images (used for both training and selection pool)
        │   ├── class1/
        │   │   ├── img1.png
        │   │   └── img2.jpg
        │   └── class2/
        │       └── img1.png
        └── target/          # Target/evaluation images (real-world test images)
            └── class1/
                └── img1.png

    Args:
        dataset_root: Path to the dataset root directory
        target_class_name: Name of the target class (auto-detected from target/ if not provided)

    Returns:
        dict: Generated dataset manifest (same format as dataset.json)
    """
    dataset_root = Path(dataset_root)
    train_dir = dataset_root / "train"
    target_dir = dataset_root / "target"

    # Validate required directories
    if not train_dir.exists():
        raise ValueError(f"Required 'train/' directory not found in {dataset_root}")
    if not target_dir.exists():
        raise ValueError(f"Required 'target/' directory not found in {dataset_root}")

    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}

    def find_images(directory: Path) -> List[Tuple[str, str]]:
        """Find all images in directory, return list of (relative_path, class_name)."""
        images = []
        if not directory.exists():
            return images

        for class_dir in sorted(directory.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name

            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in image_extensions:
                    # Relative path from dataset root
                    rel_path = img_file.relative_to(dataset_root)
                    images.append((str(rel_path), class_name))

        return images

    # Find all images
    train_images = find_images(train_dir)
    target_images = find_images(target_dir)

    if not train_images:
        raise ValueError(f"No images found in train/ directory")
    if not target_images:
        raise ValueError(f"No images found in target/ directory")

    # Build class mapping from all unique classes
    all_classes = set()
    for _, class_name in train_images + target_images:
        all_classes.add(class_name)
    all_classes = sorted(all_classes)

    # Auto-detect target classes from target/ folder (can have multiple)
    target_classes = sorted(set(class_name for _, class_name in target_images))

    # Create class mapping (target classes get highest indices)
    class_to_idx = {}
    idx = 0
    # First assign indices to non-target classes
    for class_name in all_classes:
        if class_name not in target_classes:
            class_to_idx[class_name] = idx
            idx += 1
    # Then assign indices to target classes
    target_class_indices = []
    for class_name in target_classes:
        class_to_idx[class_name] = idx
        target_class_indices.append(idx)
        idx += 1

    class_mapping = {str(v): k for k, v in class_to_idx.items()}

    # Build entries
    entries = []

    # Add train images (used for both training and selection pool)
    for rel_path, class_name in train_images:
        entries.append({
            "filename": rel_path,
            "purposes": ["select", "train"],  # Train images are both training data and selection pool
            "class_idx": class_to_idx[class_name],
            "class_name": class_name,
            "source": "train"
        })

    # Add target images
    for rel_path, class_name in target_images:
        entries.append({
            "filename": rel_path,
            "purposes": ["target"],
            "class_idx": class_to_idx[class_name],
            "class_name": class_name,
            "source": "real"
        })

    # Build manifest
    manifest = {
        "dataset_info": {
            "name": f"Auto-generated from {dataset_root.name}",
            "description": "Dataset auto-generated from ImageFolder structure",
            "num_classes": len(all_classes),
            "target_classes": target_class_indices,
            "target_class_names": target_classes
        },
        "class_mapping": class_mapping,
        "entries": entries
    }

    return manifest


def load_classification_model_for_images(model_name, custom_weights_path=None, device='cuda', num_classes=None):
    """
    Load just the classification model for image selection, without 3D rendering components.

    Args:
        model_name: Model architecture name from MODEL_CONFIGS
        custom_weights_path: Path to custom weights file (optional)
        device: Device to use
        num_classes: Override number of output classes (optional). If provided without
                    custom_weights_path, creates a model with random classifier weights
                    for this many classes.

    Returns:
        Loaded model ready for inference
    """
    from transformers import AutoConfig

    print(f"Loading model: {model_name}")
    print(f"Custom weights path: {custom_weights_path}")
    print(f"Num classes override: {num_classes}")

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]

    if custom_weights_path is None and num_classes is None:
        # Load pre-trained model with default 1000 classes
        model = config["model_class"].from_pretrained(
            config["model_name"],
            torch_dtype=torch.float32
        ).to(device).eval()
    elif custom_weights_path is None and num_classes is not None:
        # No custom weights but override num_classes - load pretrained backbone with new classifier
        print(f"Creating model with {num_classes} classes (pretrained backbone, random classifier)")
        model_config = AutoConfig.from_pretrained(config["model_name"])
        model_config.num_labels = num_classes

        # Load pretrained and modify classifier
        base_model = config["model_class"].from_pretrained(
            config["model_name"],
            torch_dtype=torch.float32
        )

        # Create model with correct num_classes
        model = config["model_class"](model_config).to(device).eval()

        # Copy backbone weights (everything except classifier)
        base_state = base_model.state_dict()
        model_state = model.state_dict()
        for key in base_state:
            if 'classifier' not in key and 'fc' not in key and 'head' not in key:
                if key in model_state and base_state[key].shape == model_state[key].shape:
                    model_state[key] = base_state[key]
        model.load_state_dict(model_state)

        del base_model
        print(f"✓ Loaded pretrained backbone with new {num_classes}-class classifier")
    else:
        # Load custom weights - detect number of classes if not explicitly provided
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

        # Only auto-detect num_classes from weights if not explicitly provided
        if num_classes is None:
            # Detect num_classes from classifier head in weights
            detected_classes = None
            for key in ['classifier.weight', 'classifier.out_proj.weight', 'classifier.1.weight']:
                if key in state_dict:
                    detected_classes = state_dict[key].shape[0]
                    break
            num_classes = detected_classes if detected_classes is not None else 1000
            print(f"Auto-detected {num_classes} classes from custom weights")

        # Create model with correct num_classes
        model_config = AutoConfig.from_pretrained(config["model_name"])
        model_config.num_labels = num_classes
        model = config["model_class"](model_config).to(device).eval()

        # Load weights
        if custom_weights_path.endswith('.safetensors'):
            # For safetensors files, load using safetensors method
            from safetensors.torch import load_model
            load_model(model, custom_weights_path, strict=False)
        else:
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
    num_classes: int = None,
    enable_reproducibility: bool = True,
    training_seed: int = 42,
    return_best: bool = False,
) -> Dict:
    """
    Train a model on a subset and evaluate on target data.

    Args:
        return_best: If True, return metrics from the epoch with best target loss.
                    If False, return metrics from the final epoch.

    Returns:
        Dict with train_loss, target_loss, target_acc (and best_epoch if return_best=True)
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Load fresh model
    model = load_classification_model_for_images(model_name, custom_weights, device, num_classes=num_classes)

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

    # Set seeds for reproducibility
    if enable_reproducibility:
        set_seed(training_seed)

    # Data loader with reproducible shuffling
    subset_dataset = TensorDataset(subset_tensors, subset_labels)
    if enable_reproducibility:
        subset_loader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=get_dataloader_generator(training_seed),
            worker_init_fn=worker_init_fn
        )
    else:
        subset_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    final_train_loss = 0
    final_target_loss = 0
    final_target_acc = 0

    # Track best metrics across epochs (for return_best mode)
    best_target_loss = float('inf')
    best_target_acc = 0
    best_train_loss = 0
    best_epoch = 0

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

            # Use reduction='sum' for correct averaging across variable batch sizes
            criterion_eval = nn.CrossEntropyLoss(reduction='sum')

            with torch.no_grad():
                for i in range(0, len(target_tensors), batch_size):
                    batch_images = target_tensors[i:i+batch_size].to(device)
                    batch_labels = target_labels[i:i+batch_size].to(device)

                    outputs = model(batch_images)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                    # Sum of losses for this batch (not mean)
                    loss = criterion_eval(logits, batch_labels)
                    total_target_loss += loss.item()

                    _, predicted = torch.max(logits, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

            # Correct mean: total sum / total number of samples
            final_target_loss = total_target_loss / len(target_tensors)
            final_target_acc = 100 * correct / total

            # Track best epoch based on target loss
            if final_target_loss < best_target_loss:
                best_target_loss = final_target_loss
                best_target_acc = final_target_acc
                best_train_loss = final_train_loss
                best_epoch = epoch + 1

        if progress_callback:
            progress_callback(epoch + 1, epochs)

    # Clean up
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if return_best:
        return {
            "train_loss": best_train_loss,
            "target_loss": best_target_loss,
            "target_acc": best_target_acc,
            "best_epoch": best_epoch,
        }
    else:
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

You can use either a JSON manifest or a simple folder structure:

**Option 1: JSON Manifest (Full Control)**
```
dataset_folder/
├── dataset.json          # Manifest file
└── images/               # All images referenced in manifest
    ├── train_img_001.png
    ├── target_img_001.png
    └── ...
```

**Option 2: ImageFolder Structure (Auto-Generates config)*
```
dataset_folder/
├── train/                # Training images (required)
│   ├── class_a/         # Class folders
│   │   ├── img1.png
│   │   └── img2.jpg
│   ├── class_b/         # Other classes
│   │   └── img1.png
│   └── target_class/    # Your target class
│       └── img1.png
└── target/               # Target/evaluation images (real-world test images)
    └── target_class/    # Must match a class in train/
        └── real_photo.png
```

When using ImageFolder structure:
- **Auto-generation**: `dataset.json` is created automatically
- **Auto-download**: The generated `dataset.json` is automatically downloaded to your browser
- **Manual download**: A "📥 Download dataset.json again" button is also available for re-downloading
- **Class detection**: Classes are discovered from train/ subfolder names
- **Target class(es)**: Auto-detected from target/ folder (can have multiple classes!)
- **Selection pool**: Train images are automatically used as the selection pool

### Manifest Structure (`dataset.json`)

```json
{
  "dataset_info": {
    "name": "My Dataset",
    "target_classes": [2],
    "target_class_names": ["target_class"],
    "num_classes": 3
  },
  "class_mapping": {"0": "class_a", "1": "class_b", "2": "target_class"},
  "entries": [
    {
      "filename": "images/train_img_001.png",
      "class_idx": 2,
      "purposes": ["train", "select"],
      "source": "synthetic"
    },
    {
      "filename": "images/real_001.png",
      "class_idx": 2,
      "purposes": ["target"],
      "source": "real"
    }
  ]
}
```

**Note**: Multiple target classes are supported! Put multiple class folders in `target/` and they'll all be treated as targets.

### Image Purposes

| Purpose | Description | Required |
|---------|-------------|----------|
| `train` | Training images (all classes) - also used as selection pool | ✅ Yes |
| `target` | Real-world images to optimize for | ✅ Yes |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Training epochs** | More epochs = better TRAK signal but slower | 10 |
| **Batch size** | Adjust based on GPU memory | 32 |
| **Learning rate** | Lower LR = slower convergence = more gradient signal | 5e-4 |

**Training Options** (expandable):
- **Reset classifier head**: Reinitialize classifier weights before training (default: enabled)
- **Optimizer**: AdamW (default), Adam, or SGD

### TRAK Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| **JL projection dimension** | Higher = more accurate but slower/more memory | 1024 |
| **Number of projections** | More = more stable scores (averaged) | 4 |
| **Gradient source** | "Last layer only" (faster) or "Full model" | Last layer only |
| **Number of checkpoints** | Save multiple checkpoints during training, average TRAK across them | 2 |

**Ensemble TRAK**: Using multiple checkpoints × multiple projections gives more stable, reliable influence scores. For example, 3 checkpoints × 4 projections = 12 TRAK computations averaged.

### Workflow (3 Steps)

**Step 1: Train Initial Model**
- Trains a classifier on `train` + `select` images combined
- Saves model checkpoints for TRAK computation
- Records initial class distribution for Step 3 balancing
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
- Select top-k% from `select`, then balance with `train` to match initial class ratios
- Final training set maintains same class proportions as Step 1

### Understanding Results

- **Higher TRAK scores** = images more influential for target domain performance
- **Score table**: Sortable by rank, viewable in carousel or grid
- **Downloads**: CSV (scores only) or JSON (full manifest with scores added)
- **Comparison mode**: Trains all three methods and shows which performs best

### Example Use Case

- **Train images**: 500 synthetic renders of target class + 500 other class images (also used as selection pool)
- **Target images**: 50 real-world photos of target class
- **TRAK config**: 3 checkpoints, 4 projections, JL dim=2048
- **Result**: Top 10% most influential synthetic images for real-world recognition

### Tips

- **Start with ImageFolder**: Just organize images in `train/<class>/` and `target/<class>/` folders - no JSON needed!
- **Auto-download**: The generated `dataset.json` is automatically downloaded when you upload the folder structure
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

    # Display current model configuration
    model_config = st.session_state.get("model_config", {})
    model_name = model_config.get("model_name", st.session_state.get("model_name", "vit_l_16"))
    custom_weights_path = model_config.get("custom_weights_path")
    num_classes_override = model_config.get("num_classes_override")
    custom_labels = model_config.get("custom_labels")

    # Check if dataset has been previewed (num_classes from dataset)
    dataset_preview = st.session_state.get("dataset_preview", {})
    dataset_info = dataset_preview.get("dataset_info", {})
    dataset_num_classes = dataset_info.get("num_classes")

    # Determine effective num_classes and source
    if num_classes_override is not None:
        num_classes = num_classes_override
        classes_source = "sidebar override"
    elif dataset_num_classes is not None:
        num_classes = dataset_num_classes
        classes_source = "from dataset"
    elif custom_weights_path:
        num_classes = model_config.get("num_classes", 1000)
        classes_source = "from weights"
    else:
        num_classes = 1000
        classes_source = "default (will auto-detect from dataset)"

    # Get model display name
    model_display = MODEL_CONFIGS.get(model_name, {}).get("description", model_name)

    # Show model info box
    st.markdown("### Current Model Configuration")
    col_model_info1, col_model_info2, col_model_info3 = st.columns(3)
    with col_model_info1:
        st.metric("Model", model_display.split(" (")[0] if "(" in model_display else model_display)
    with col_model_info2:
        st.metric("Output Classes", f"{num_classes}")
    with col_model_info3:
        if custom_labels:
            st.metric("Class Labels", f"{len(custom_labels)} custom")
        elif dataset_num_classes:
            st.metric("Class Labels", "From dataset")
        else:
            st.metric("Class Labels", "Auto-detect")

    # Show custom weights path if used
    if custom_weights_path:
        weights_name = Path(custom_weights_path).name
        st.caption(f"Weights: {weights_name[:40]}..." if len(weights_name) > 40 else f"Weights: {weights_name}")

    st.markdown("---")

    # File upload section
    st.markdown("### 📤 Upload Dataset")

    st.markdown("""
    **Two upload formats supported:**

    **Option 1: JSON Manifest** (`dataset.json` + images)
    - Upload a folder containing `dataset.json` manifest and `images/` folder
    - JSON specifies `train`, `select`, and `target` purposes per image

    **Option 2: ImageFolder Structure** (auto-generates manifest)
    - Upload a folder with `train/` and `target/` subfolders
    - Each subfolder contains class folders with images:
      ```
      dataset/
      ├── train/
      │   ├── class_a/       (images...)
      │   ├── class_b/       (images...)
      │   └── target_class/  (images...)
      └── target/
          └── target_class/  (real target images...)
      ```
    - Manifest will be auto-generated from folder structure
    """)

    # Dataset source selection
    dataset_source = st.radio(
        "Dataset source",
        options=["📦 Upload ZIP", "📁 Local path"],
        horizontal=True,
        help="Choose how to load the dataset - upload a ZIP file or specify a local directory path",
        key="image_selection_dataset_source",
    )

    dataset_path = None  # Will be set to the path of the dataset directory

    if dataset_source == "📦 Upload ZIP":
        dataset_zip = st.file_uploader(
            "Upload dataset folder (as ZIP)",
            type=["zip"],
            help="ZIP with either: (1) dataset.json + images/, or (2) train/ + target/ class folders",
            key="dataset_zip"
        )

        if not dataset_zip:
            st.info("Upload dataset ZIP file")
            return

        st.success(f"✅ Dataset uploaded: {dataset_zip.name}")

        # Preview ZIP contents
        try:
            with zipfile.ZipFile(dataset_zip, 'r') as zf:
                file_list = zf.namelist()

                # Find dataset.json
                manifest = None
                manifest_path = None
                for f in file_list:
                    if f.endswith('dataset.json'):
                        manifest_path = f
                        with zf.open(f) as mf:
                            manifest = json.load(mf)
                        break

                if manifest:
                    # Count images by purpose
                    train_count = 0
                    select_count = 0
                    target_count = 0
                    class_counts = {}
                    train_paths = []
                    select_paths = []

                    for entry in manifest.get('entries', []):
                        purposes = entry.get('purposes', [])
                        class_idx = entry.get('class_idx', 0)
                        img_path = entry.get('filename', '')

                        if 'train' in purposes:
                            train_count += 1
                            train_paths.append(img_path)
                            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
                        if 'select' in purposes:
                            select_count += 1
                            select_paths.append(img_path)
                        if 'target' in purposes:
                            target_count += 1

                    if select_count == 0:
                        select_count = train_count

                    # Calculate overlap
                    train_set = set(train_paths)
                    select_set = set(select_paths) if select_paths else train_set
                    overlap_count = len(train_set & select_set)
                    overlap_pct = 100 * overlap_count / len(select_set) if len(select_set) > 0 else 0

                    if overlap_pct == 100:
                        overlap_str = " (train=select)"
                    elif overlap_pct > 0:
                        overlap_str = f" ({overlap_pct:.0f}% overlap)"
                    else:
                        overlap_str = " (disjoint)"

                    dataset_info = manifest.get('dataset_info', {})
                    class_mapping = manifest.get('class_mapping', {})
                    st.info(f"📊 **Dataset**: {train_count} train, {select_count} select{overlap_str}, {target_count} target images")

                    with st.expander("📋 Dataset Details", expanded=False):
                        if dataset_info:
                            if dataset_info.get('name'):
                                st.write(f"**Name**: {dataset_info['name']}")
                            if dataset_info.get('description'):
                                st.write(f"**Description**: {dataset_info['description']}")
                            if 'num_classes' in dataset_info:
                                st.write(f"**Number of classes**: {dataset_info['num_classes']}")

                        st.write(f"\n**Dataset breakdown:**")
                        st.write(f"  - **Training images**: {train_count}")
                        st.write(f"  - **Selection pool**: {select_count}")
                        st.write(f"  - **Target images**: {target_count}")

                        if class_counts:
                            target_class_indices = dataset_info.get('target_classes', [dataset_info.get('target_class', 0)])
                            target_set = set(target_class_indices)
                            st.write(f"\n**Class distribution in training set:**")
                            for class_id in sorted(class_counts.keys()):
                                count = class_counts[class_id]
                                marker = "🎯" if class_id in target_set else "  "
                                class_name = class_mapping.get(str(class_id), f"class_{class_id}")
                                st.write(f"{marker} {class_name}: {count} images")

                    # Check for pre-computed TRAK scores in ZIP manifest
                    has_precomputed_trak = any(
                        entry.get('trak_score') is not None
                        for entry in manifest.get('entries', [])
                    )
                    if has_precomputed_trak:
                        train_entries_with_scores = [
                            e for e in manifest.get('entries', [])
                            if e.get('trak_score') is not None and 'train' in e.get('purposes', [])
                        ]
                        st.success(
                            f"🎯 **Pre-computed TRAK scores detected!** {len(train_entries_with_scores)} training entries have `trak_score` field. "
                            f"You can skip to Step 3 below to run selection comparison & retraining."
                        )
                        st.session_state["has_precomputed_trak"] = True
                        st.session_state["precomputed_zip"] = dataset_zip
                    else:
                        st.session_state["has_precomputed_trak"] = False

                else:
                    # Check for ImageFolder structure
                    has_train = any('train/' in f for f in file_list)
                    has_target = any('target/' in f for f in file_list)
                    if has_train and has_target:
                        # Count images per class folder
                        train_classes = {}
                        target_classes = set()
                        for f in file_list:
                            if '/train/' in f or f.startswith('train/'):
                                parts = f.split('/')
                                idx = parts.index('train') if 'train' in parts else -1
                                if idx >= 0 and len(parts) > idx + 2:
                                    class_name = parts[idx + 1]
                                    if class_name and not f.endswith('/'):
                                        train_classes[class_name] = train_classes.get(class_name, 0) + 1
                            if '/target/' in f or f.startswith('target/'):
                                parts = f.split('/')
                                idx = parts.index('target') if 'target' in parts else -1
                                if idx >= 0 and len(parts) > idx + 1:
                                    class_name = parts[idx + 1]
                                    if class_name and not f.endswith('/'):
                                        target_classes.add(class_name)

                        total_train = sum(train_classes.values())
                        st.info(f"📊 **ImageFolder**: {total_train} train images, {len(train_classes)} classes")

                        with st.expander("📋 Dataset Details (ImageFolder)", expanded=False):
                            st.write("**Structure detected**: ImageFolder (will auto-generate manifest)")
                            st.write(f"**Target class(es)**: {', '.join(sorted(target_classes))}")
                            st.write(f"\n**Class distribution in train/:**")
                            for class_name in sorted(train_classes.keys()):
                                count = train_classes[class_name]
                                marker = "🎯" if class_name in target_classes else "  "
                                st.write(f"{marker} {class_name}: {count} images")

        except Exception as e:
            st.warning(f"⚠️ Could not preview ZIP contents: {str(e)}")

    else:  # Local path
        st.markdown("**Enter the path to your local dataset directory:**")
        local_dataset_path = st.text_input(
            "Local dataset path",
            value="",
            placeholder="/path/to/ui_dataset",
            help="Path to directory with either: (1) dataset.json + images/, or (2) train/ + target/ folders"
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

        # Check for dataset_trak.json first (pre-computed TRAK scores)
        trak_manifest_path = local_path / "dataset_trak.json"
        manifest_path = local_path / "dataset.json"
        auto_generated_manifest = None
        using_trak_manifest = False

        # Use dataset_trak.json if it exists and user hasn't reset
        if trak_manifest_path.exists() and "force_reset_trak" not in st.session_state:
            manifest_path = trak_manifest_path
            using_trak_manifest = True
            st.info("📂 Found `dataset_trak.json` with pre-computed TRAK scores - will auto-load for Step 3.")

        if not manifest_path.exists():
            # Check if ImageFolder structure exists (train/ and target/ folders)
            train_dir = local_path / "train"
            target_dir = local_path / "target"

            if train_dir.exists() and target_dir.exists():
                st.info("📂 No dataset.json found, attempting to auto-generate from folder structure...")
                try:
                    auto_generated_manifest = generate_dataset_json_from_folder(str(local_path))
                    # Save the generated manifest
                    with open(manifest_path, 'w') as f:
                        json.dump(auto_generated_manifest, f, indent=2)
                    st.success(f"✅ Auto-generated dataset.json with {len(auto_generated_manifest['entries'])} entries")

                    # Auto-download the generated manifest (only once per path)
                    manifest_hash = hash(json.dumps(auto_generated_manifest, sort_keys=True))
                    download_key = f"auto_downloaded_local_manifest_{manifest_hash}"
                    if download_key not in st.session_state:
                        st.session_state[download_key] = True
                        auto_download_json(auto_generated_manifest, "dataset.json")
                        st.info("📥 **dataset.json auto-downloaded!** Check your downloads folder.")

                    # Also provide manual download button as fallback
                    st.download_button(
                        label="📥 Download dataset.json again",
                        data=json.dumps(auto_generated_manifest, indent=2),
                        file_name="dataset.json",
                        mime="application/json",
                        help="Download the auto-generated manifest to review or modify",
                        key="download_local_manifest"
                    )
                except Exception as e:
                    st.error(f"❌ Failed to auto-generate dataset.json: {str(e)}")
                    return
            else:
                st.error(f"❌ No dataset.json found in {local_dataset_path}")
                st.info("💡 Either provide a dataset.json or use ImageFolder structure with train/ and target/ folders")
                return

        # Check for images - either in images/ folder or in train/target folders
        images_path = local_path / "images"
        train_path = local_path / "train"
        if not images_path.exists() and not train_path.exists():
            st.error(f"❌ No images/ or train/ folder found in {local_dataset_path}")
            return

        st.success(f"✅ Dataset found at: {local_dataset_path}")
        dataset_path = local_path  # Store for later use
        dataset_zip = None  # No ZIP file to process

        # Parse and display dataset details immediately
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            # Count images by purpose
            train_count = 0
            select_count = 0
            target_count = 0
            train_paths_preview = []
            select_paths_preview = []
            target_paths_preview = []
            class_counts = {}

            for entry in manifest.get('entries', []):
                purposes = entry.get('purposes', [])
                class_idx = entry.get('class_idx', 0)
                img_path = entry.get('filename', '')

                if 'train' in purposes:
                    train_count += 1
                    train_paths_preview.append(img_path)
                    class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
                if 'select' in purposes:
                    select_count += 1
                    select_paths_preview.append(img_path)
                if 'target' in purposes:
                    target_count += 1
                    target_paths_preview.append(img_path)

            # If no select pool, it will use train images
            if select_count == 0:
                select_count = train_count

            # Calculate overlap
            train_set = set(train_paths_preview)
            select_set = set(select_paths_preview) if select_paths_preview else train_set
            overlap_count = len(train_set & select_set)
            overlap_pct = 100 * overlap_count / len(select_set) if len(select_set) > 0 else 0

            if overlap_pct == 100:
                overlap_str = " (train=select)"
            elif overlap_pct > 0:
                overlap_str = f" ({overlap_pct:.0f}% overlap)"
            else:
                overlap_str = " (disjoint)"

            dataset_info = manifest.get('dataset_info', {})
            class_mapping = manifest.get('class_mapping', {})
            st.info(f"📊 **Dataset**: {train_count} train, {select_count} select{overlap_str}, {target_count} target images")

            # Dataset details expander
            with st.expander("📋 Dataset Details", expanded=False):
                if dataset_info:
                    if dataset_info.get('name'):
                        st.write(f"**Name**: {dataset_info['name']}")
                    if dataset_info.get('description'):
                        st.write(f"**Description**: {dataset_info['description']}")
                    if 'num_classes' in dataset_info:
                        st.write(f"**Number of classes**: {dataset_info['num_classes']}")

                st.write(f"\n**Dataset breakdown:**")
                st.write(f"  - **Training images**: {train_count}")
                st.write(f"  - **Selection pool**: {select_count}")
                st.write(f"  - **Target images**: {target_count}")

                if class_counts:
                    target_class_indices = dataset_info.get('target_classes', [dataset_info.get('target_class', 0)])
                    target_set = set(target_class_indices)
                    st.write(f"\n**Class distribution in training set:**")
                    for class_id in sorted(class_counts.keys()):
                        count = class_counts[class_id]
                        marker = "🎯" if class_id in target_set else "  "
                        class_name = class_mapping.get(str(class_id), f"class_{class_id}")
                        st.write(f"{marker} {class_name}: {count} images")

            # Cache parsed data for Step 1
            st.session_state["dataset_preview"] = {
                "manifest": manifest,
                "train_count": train_count,
                "select_count": select_count,
                "target_count": target_count,
                "class_counts": class_counts,
                "dataset_info": dataset_info,
            }

            # Check if dataset.json already has pre-computed TRAK scores
            has_precomputed_trak = any(
                entry.get('trak_score') is not None
                for entry in manifest.get('entries', [])
            )

            if has_precomputed_trak:
                # Count entries with TRAK scores
                entries_with_scores = [e for e in manifest.get('entries', []) if e.get('trak_score') is not None]
                train_entries_with_scores = [e for e in entries_with_scores if 'train' in e.get('purposes', [])]

                st.success(
                    f"🎯 **Pre-computed TRAK scores detected!** {len(train_entries_with_scores)} training entries have `trak_score` field. "
                    f"You can skip to Step 3 below to run selection comparison & retraining."
                )

                # Store in session state for the checkbox to reference
                st.session_state["has_precomputed_trak"] = True
                st.session_state["precomputed_manifest_path"] = str(manifest_path)
                st.session_state["precomputed_dataset_root"] = str(local_path)

                # If this is from dataset_trak.json and trak_data not loaded, auto-trigger loading
                if using_trak_manifest and "trak_data" not in st.session_state:
                    st.session_state["auto_load_trak"] = True
            else:
                st.session_state["has_precomputed_trak"] = False

        except Exception as e:
            st.warning(f"⚠️ Could not preview dataset: {str(e)}")

    # Training parameters (selection parameters moved to Step 3)
    st.markdown("### ⚙️ Training Parameters")
    col_param1, col_param2, col_param3 = st.columns(3)

    with col_param1:
        train_epochs = st.number_input(
            "Training epochs",
            min_value=1,
            max_value=50,
            value=8,
            step=1,
            help="Number of epochs to train the model (more = better TRAK scores, slower)"
        )

    with col_param2:
        train_batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=2048,
            value=64,
            step=8,
            help="Batch size for training and gradient extraction"
        )

    with col_param3:
        learning_rate = st.select_slider(
            "Learning rate",
            options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1],
            value=5e-4,
            format_func=lambda x: f"{x:.0e}",
            help="Lower learning rate = slower convergence = more gradient signal for TRAK"
        )

    # Additional training options (hidden in expander)
    with st.expander("🔧 Training Options", expanded=False):
        col_opt1, col_opt2 = st.columns(2)

        with col_opt1:
            reset_classifier = st.checkbox(
                "Reset classifier head",
                value=True,
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

    # Reproducibility settings
    with st.expander("🎲 Reproducibility Settings", expanded=False):
        col_repro1, col_repro2 = st.columns(2)
        with col_repro1:
            enable_reproducibility = st.checkbox(
                "Enable deterministic training",
                value=True,
                key="enable_reproducibility",
                help="Fix random seeds for data shuffling and model initialization. Same seed = same results."
            )
        with col_repro2:
            training_seed = st.number_input(
                "Random seed",
                min_value=0,
                max_value=2**31 - 1,
                value=42,
                step=1,
                key="training_seed",
                help="Seed for random number generators.",
                disabled=not enable_reproducibility
            )
        if enable_reproducibility:
            st.caption("✓ Fixes data shuffling order and model weight initialization for reproducible results.")

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
            value=8,
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
            value=4,
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

    # ==================== SKIP STEPS 1 & 2 WITH PRE-COMPUTED TRAK SCORES ====================
    if st.session_state.get("has_precomputed_trak", False) and "trak_data" not in st.session_state:
        st.markdown("---")
        st.markdown("### ⚡ Skip to Step 3 (Pre-computed TRAK Scores)")

        # Check if we should auto-load (from dataset_trak.json)
        auto_load_trak = st.session_state.pop("auto_load_trak", False)

        if auto_load_trak:
            st.success(
                "🎯 **Found `dataset_trak.json`!** Auto-loading pre-computed TRAK scores. "
                "Click 'Reset All Steps' if you want to recompute."
            )
        else:
            st.info(
                "🎯 **Pre-computed TRAK scores detected!** Your `dataset.json` contains `trak_score` fields in entries.\n\n"
                "You can skip Steps 1 (Train Model) & 2 (Compute TRAK) and go directly to "
                "Step 3 to run **selection comparison** and **retrain with selected subsets**.\n\n"
                "This is useful when you've already computed TRAK scores and want to experiment with different "
                "selection percentages or methods without re-running the expensive computation."
            )

        # Auto-trigger if from dataset_trak.json, otherwise show button
        load_triggered = auto_load_trak or st.button("⚡ **Load Pre-computed TRAK Scores & Skip to Step 3**", type="primary", use_container_width=True)

        if load_triggered:
            with st.spinner("Loading pre-computed TRAK scores..."):
                try:
                    # Handle both local path and ZIP file sources
                    dataset_root = st.session_state.get("precomputed_dataset_root")
                    manifest_path = st.session_state.get("precomputed_manifest_path")
                    precomputed_zip = st.session_state.get("precomputed_zip")
                    temp_dir = None

                    if precomputed_zip is not None:
                        # Extract ZIP to temp directory
                        temp_dir = tempfile.mkdtemp(prefix="trak_precomputed_")
                        st.write(f"📦 Extracting ZIP to temporary directory...")
                        with zipfile.ZipFile(precomputed_zip, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)

                        # Find dataset.json in extracted files
                        for root, dirs, files in os.walk(temp_dir):
                            if 'dataset.json' in files:
                                manifest_path = os.path.join(root, 'dataset.json')
                                dataset_root = root
                                break

                        if not manifest_path or not os.path.exists(manifest_path):
                            st.error("❌ Could not find dataset.json in ZIP file")
                            if temp_dir:
                                shutil.rmtree(temp_dir, ignore_errors=True)
                            st.stop()

                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)

                    # Get class mapping
                    class_mapping = manifest.get('class_mapping', {})
                    dataset_info = manifest.get('dataset_info', {})
                    num_classes = dataset_info.get('num_classes', len(class_mapping))

                    # Separate entries by purpose
                    train_entries = []
                    select_entries = []
                    target_entries = []

                    for entry in manifest.get('entries', []):
                        purposes = entry.get('purposes', [])
                        if 'train' in purposes:
                            train_entries.append(entry)
                        if 'select' in purposes:
                            select_entries.append(entry)
                        if 'target' in purposes:
                            target_entries.append(entry)

                    # If no separate select pool, use train as select
                    if not select_entries:
                        select_entries = train_entries

                    # Filter to entries with TRAK scores
                    scored_entries = [e for e in select_entries if e.get('trak_score') is not None]

                    if not scored_entries:
                        st.error("❌ No entries with TRAK scores found in select/train pool")
                        st.stop()

                    st.write(f"📊 Loading {len(scored_entries)} scored entries, {len(train_entries)} train, {len(target_entries)} target...")

                    # Image transforms
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

                    # Load selection pool images and scores
                    selection_tensors_list = []
                    selection_labels_list = []
                    selection_valid_paths = []
                    importance_scores_list = []
                    selection_class_names = []
                    selection_sources = []

                    progress_bar = st.progress(0, text="Loading selection pool...")
                    for i, entry in enumerate(scored_entries):
                        img_path = os.path.join(dataset_root, entry['filename'])
                        if os.path.exists(img_path):
                            try:
                                img = Image.open(img_path).convert('RGB')
                                tensor = transform(img)
                                selection_tensors_list.append(tensor)
                                selection_labels_list.append(entry.get('class_idx', 0))
                                selection_valid_paths.append(img_path)  # Store full path, not relative
                                importance_scores_list.append(entry['trak_score'])
                                selection_class_names.append(entry.get('class_name', f"class_{entry.get('class_idx', 0)}"))
                                selection_sources.append(entry.get('source', 'train'))
                            except Exception:
                                pass
                        if i % 100 == 0:
                            progress_bar.progress(i / len(scored_entries), text=f"Loading selection pool... {i}/{len(scored_entries)}")
                    progress_bar.empty()

                    selection_tensors = torch.stack(selection_tensors_list)
                    selection_labels = torch.tensor(selection_labels_list, dtype=torch.long)
                    importance_scores = torch.tensor(importance_scores_list, dtype=torch.float32)

                    # Load train images
                    train_tensors_list = []
                    train_labels_list = []
                    train_paths = []

                    progress_bar = st.progress(0, text="Loading train images...")
                    for i, entry in enumerate(train_entries):
                        img_path = os.path.join(dataset_root, entry['filename'])
                        if os.path.exists(img_path):
                            try:
                                img = Image.open(img_path).convert('RGB')
                                tensor = transform(img)
                                train_tensors_list.append(tensor)
                                train_labels_list.append(entry.get('class_idx', 0))
                                train_paths.append(img_path)  # Store full path, not relative
                            except Exception:
                                pass
                        if i % 100 == 0:
                            progress_bar.progress(i / len(train_entries), text=f"Loading train images... {i}/{len(train_entries)}")
                    progress_bar.empty()

                    train_tensors = torch.stack(train_tensors_list) if train_tensors_list else None
                    train_labels = torch.tensor(train_labels_list, dtype=torch.long) if train_labels_list else None

                    # Load target images
                    target_tensors_list = []
                    target_labels_list = []

                    progress_bar = st.progress(0, text="Loading target images...")
                    for i, entry in enumerate(target_entries):
                        img_path = os.path.join(dataset_root, entry['filename'])
                        if os.path.exists(img_path):
                            try:
                                img = Image.open(img_path).convert('RGB')
                                tensor = transform(img)
                                target_tensors_list.append(tensor)
                                target_labels_list.append(entry.get('class_idx', 0))
                            except Exception:
                                pass
                        if i % 100 == 0:
                            progress_bar.progress(i / len(target_entries), text=f"Loading target images... {i}/{len(target_entries)}")
                    progress_bar.empty()

                    target_tensors = torch.stack(target_tensors_list) if target_tensors_list else None
                    target_labels = torch.tensor(target_labels_list, dtype=torch.long) if target_labels_list else None

                    # Calculate initial class ratios from train data
                    initial_class_counts = {}
                    initial_class_ratios = {}
                    if train_labels is not None:
                        for label in train_labels.tolist():
                            initial_class_counts[label] = initial_class_counts.get(label, 0) + 1
                        total = sum(initial_class_counts.values())
                        initial_class_ratios = {k: v / total for k, v in initial_class_counts.items()}

                    # Build scores dataframe (matching Step 2 format)
                    scores_data = {
                        'filename': [os.path.basename(p) for p in selection_valid_paths],
                        'trak_score': importance_scores.numpy().tolist(),
                        'class': selection_class_names,
                        'class_idx': selection_labels.numpy().tolist(),
                        'source': selection_sources,
                    }
                    scores_df = pd.DataFrame(scores_data)
                    scores_df_sorted = scores_df.sort_values('trak_score', ascending=False).reset_index(drop=True)

                    # Get model config from session state
                    model_config = st.session_state.get("model_config", {})
                    model_name = model_config.get("model_name", "resnet50")
                    custom_weights = model_config.get("custom_weights_path")
                    num_classes_override = model_config.get("num_classes_override")

                    # Determine target class for pool_label
                    target_classes = dataset_info.get('target_classes', [])
                    pool_label = target_classes[0] if target_classes else 0

                    # Store trak_data in session state
                    st.session_state["trak_data"] = {
                        "importance_scores": importance_scores,
                        "selection_tensors": selection_tensors,
                        "selection_labels": selection_labels,
                        "selection_valid_paths": selection_valid_paths,
                        "target_tensors": target_tensors,
                        "target_labels": target_labels,
                        "train_tensors": train_tensors,
                        "train_labels": train_labels,
                        "train_paths": train_paths,
                        "initial_class_ratios": initial_class_ratios,
                        "initial_class_counts": initial_class_counts,
                        "manifest": manifest,
                        "model_name": model_name,
                        "custom_weights": custom_weights,
                        "device": "cuda" if torch.cuda.is_available() else "cpu",
                        "train_batch_size": train_batch_size,
                        "train_epochs": train_epochs,
                        "pool_label": pool_label,
                        "temp_dir": temp_dir,  # Will be set if extracted from ZIP
                        "dataset_root": dataset_root,
                        "scores_df_sorted": scores_df_sorted,
                        "jl_dim": jl_dim,
                        "num_projections": num_projections,
                        "class_mapping": class_mapping,
                        "num_classes_override": num_classes_override or num_classes,
                        # Reproducibility settings
                        "enable_reproducibility": enable_reproducibility,
                        "training_seed": training_seed,
                    }

                    # Also mark step1 as complete (with minimal data)
                    st.session_state["step1_data"] = {
                        "model_name": model_name,
                        "custom_weights": custom_weights,
                        "dataset_info": dataset_info,
                        "num_classes_override": num_classes_override or num_classes,
                        "learning_rate": learning_rate,
                        "train_batch_size": train_batch_size,
                        "train_epochs": train_epochs,
                        "freeze_backbone": freeze_backbone,
                        "enable_reproducibility": enable_reproducibility,
                        "training_seed": training_seed,
                        "epoch_metrics": [],  # No training metrics since we skipped
                        "final_train_acc": 0,
                        "final_target_acc": 0,
                        "final_train_loss": 0,
                        "final_target_loss": 0,
                        "skipped": True,  # Flag to indicate steps were skipped
                    }

                    st.success(f"✅ Loaded {len(selection_valid_paths)} scored images! Ready for Step 3.")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error loading pre-computed scores: {str(e)}")
                    st.error(traceback.format_exc())
                    # Clean up temp directory on error
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)

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

    # Auto-run and Reset buttons
    col_auto, col_reset = st.columns([2, 1])

    with col_auto:
        # Auto-Run All button - only show if not already completed
        if not step2_complete:
            auto_run_btn = st.button(
                "🚀 **Auto-Run All Steps + Compare**",
                type="primary",
                use_container_width=True,
                help="Automatically run Step 1, Step 2, and Compare All Methods"
            )
            if auto_run_btn:
                st.session_state["auto_run_mode"] = True
                st.session_state["auto_run_step"] = 1

    with col_reset:
        # Reset button to start over
        if step1_complete or step2_complete:
            if st.button("🔄 Reset All Steps", type="secondary", use_container_width=True):
                if "step1_data" in st.session_state:
                    # Clean up temp directory if exists
                    old_temp_dir = st.session_state["step1_data"].get("temp_dir")
                    if old_temp_dir and os.path.exists(old_temp_dir):
                        shutil.rmtree(old_temp_dir, ignore_errors=True)
                    del st.session_state["step1_data"]
                if "trak_data" in st.session_state:
                    del st.session_state["trak_data"]
                if "auto_run_mode" in st.session_state:
                    del st.session_state["auto_run_mode"]
                if "auto_run_step" in st.session_state:
                    del st.session_state["auto_run_step"]
                # Set flag to prevent auto-loading dataset_trak.json on next load
                st.session_state["force_reset_trak"] = True
                # Clear precomputed trak flags
                if "has_precomputed_trak" in st.session_state:
                    del st.session_state["has_precomputed_trak"]
                if "auto_load_trak" in st.session_state:
                    del st.session_state["auto_load_trak"]
                st.rerun()

    # Check if we're in auto-run mode
    auto_run_mode = st.session_state.get("auto_run_mode", False)
    auto_run_step = st.session_state.get("auto_run_step", 0)

    # ==================== STEP 1: TRAIN INITIAL MODEL ====================
    st.markdown("---")
    step1_button_disabled = step1_complete  # Disable if already done (use reset to redo)

    # Show training configuration summary before Step 1 button
    if not step1_complete:
        model_config = st.session_state.get("model_config", {})
        model_name_display = model_config.get("model_name", "vit_l_16")
        custom_weights = model_config.get("custom_weights_path")

        # Get num_classes from dataset preview or sidebar override
        dataset_preview = st.session_state.get("dataset_preview", {})
        dataset_num_classes = dataset_preview.get("dataset_info", {}).get("num_classes")
        num_classes_override = model_config.get("num_classes_override")
        effective_classes = num_classes_override or dataset_num_classes or 1000

        st.info(
            f"**Training config:** {model_name_display} → {effective_classes} classes | "
            f"{train_epochs} epochs | batch {train_batch_size} | lr {learning_rate:.0e} | "
            f"{'Frozen backbone' if freeze_backbone else 'Full model'}"
            + (f" | Custom weights" if custom_weights else "")
        )

    # Trigger Step 1 via button OR auto-run mode
    step1_triggered = st.button("🎓 **Step 1: Train Initial Model**", type="primary", use_container_width=True, disabled=step1_button_disabled)
    if auto_run_mode and auto_run_step == 1 and not step1_complete:
        step1_triggered = True

    if step1_triggered:
        with st.spinner("Training initial model..."):
            try:
                # Set random seeds for reproducibility
                if enable_reproducibility:
                    set_seed(training_seed)
                    st.info(f"🎲 Reproducibility enabled with seed: {training_seed}")

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
                        # Try to find ImageFolder structure and auto-generate
                        st.info("📂 No dataset.json found, checking for ImageFolder structure...")

                        # Find root directory (look for train/ folder)
                        imagefolder_root = None
                        for root, dirs, files in os.walk(temp_dir):
                            if 'train' in dirs and 'target' in dirs:
                                imagefolder_root = Path(root)
                                break

                        if imagefolder_root:
                            try:
                                auto_manifest = generate_dataset_json_from_folder(str(imagefolder_root))
                                manifest_file = imagefolder_root / "dataset.json"
                                with open(manifest_file, 'w') as f:
                                    json.dump(auto_manifest, f, indent=2)
                                st.success(f"✅ Auto-generated dataset.json with {len(auto_manifest['entries'])} entries")

                                # Auto-download the generated manifest (only once per upload)
                                manifest_hash = hash(json.dumps(auto_manifest, sort_keys=True))
                                download_key = f"auto_downloaded_manifest_{manifest_hash}"
                                if download_key not in st.session_state:
                                    st.session_state[download_key] = True
                                    auto_download_json(auto_manifest, "dataset.json")
                                    st.info("📥 **dataset.json auto-downloaded!** Check your downloads folder.")

                                # Also provide manual download button as fallback
                                st.download_button(
                                    label="📥 Download dataset.json again",
                                    data=json.dumps(auto_manifest, indent=2),
                                    file_name="dataset.json",
                                    mime="application/json",
                                    help="Download the auto-generated manifest to review or modify",
                                    key="download_zip_manifest"
                                )
                            except Exception as e:
                                st.error(f"❌ Failed to auto-generate dataset.json: {str(e)}")
                                shutil.rmtree(temp_dir, ignore_errors=True)
                                return
                        else:
                            st.error("❌ No dataset.json found and no ImageFolder structure (train/ + target/) detected")
                            st.info("💡 Upload a ZIP with either:\n- A dataset.json manifest\n- ImageFolder structure: train/<class>/*.png and target/<class>/*.png")
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

                # Show loaded dataset info (collapsed by default)
                dataset_info = manifest.get('dataset_info', {})
                class_mapping = manifest.get('class_mapping', {})
                from collections import Counter
                train_class_counts = Counter(all_train_class_labels)

                # Support both multi-target and legacy single-target formats
                target_class_indices = dataset_info.get('target_classes', [dataset_info.get('target_class', 0)])
                target_class_names = dataset_info.get('target_class_names', [dataset_info.get('target_class_name', 'target')])
                target_set = set(target_class_indices)

                # Calculate overlap between train and select sets
                train_set = set(all_train_paths)
                select_set = set(selection_pool_paths)
                overlap_count = len(train_set & select_set)
                overlap_pct = 100 * overlap_count / len(select_set) if len(select_set) > 0 else 0

                # Brief summary outside expander
                if overlap_pct == 100:
                    overlap_str = " (train=select)"
                elif overlap_pct > 0:
                    overlap_str = f" ({overlap_pct:.0f}% overlap)"
                else:
                    overlap_str = " (disjoint)"
                st.success(f"📊 Loaded: {len(all_train_paths)} train, {len(selection_pool_paths)} select{overlap_str}, {len(target_paths)} target images")

                # Detailed info in expander
                with st.expander("📋 Dataset Details", expanded=False):
                    if dataset_info:
                        if dataset_info.get('name'):
                            st.write(f"**Name**: {dataset_info['name']}")
                        if dataset_info.get('description'):
                            st.write(f"**Description**: {dataset_info['description']}")

                    st.write(f"\n**Dataset breakdown:**")
                    st.write(f"  - **Training images** (for initial model): {len(all_train_paths)} images")
                    # Show target class(es) breakdown
                    target_count = sum(train_class_counts.get(idx, 0) for idx in target_class_indices)
                    target_names_str = ", ".join(target_class_names)
                    st.write(f"    - Target class(es) [{target_names_str}]: {target_count} images")
                    st.write(f"    - Other classes: {len(all_train_paths) - target_count} images")

                    st.write(f"  - **Target images** (for evaluation): {len(target_paths)} images")
                    st.write(f"  - **Selection pool** (for TRAK selection): {len(selection_pool_paths)} images")

                    # Show train/select overlap info
                    if overlap_pct == 100:
                        st.write(f"    - 🔄 *Train and select are identical* ({overlap_count} images)")
                    elif overlap_pct > 0:
                        st.write(f"    - 🔄 *{overlap_count} images overlap with train* ({overlap_pct:.1f}%)")
                    else:
                        st.write(f"    - ✂️ *No overlap with train set* (disjoint)")

                    # Show statistics by source if available
                    stats = manifest.get('statistics', {})
                    if stats.get('by_source'):
                        st.write(f"\n**By source:**")
                        for source, count in stats['by_source'].items():
                            st.write(f"  - {source}: {count}")

                    # Full class distribution
                    st.write(f"\n**Full class distribution in training set:**")
                    for class_id in sorted(train_class_counts.keys()):
                        count = train_class_counts[class_id]
                        marker = "🎯" if class_id in target_set else "  "
                        class_name = class_mapping.get(str(class_id), f"class_{class_id}")
                        st.write(f"{marker} {class_name}: {count} images")

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
                    num_classes_override = model_config.get("num_classes_override")

                    # Auto-detect num_classes from dataset manifest if not specified in sidebar
                    dataset_num_classes = dataset_info.get('num_classes')
                    if dataset_num_classes is not None and num_classes_override is None:
                        num_classes_override = dataset_num_classes
                        st.info(f"🎯 Auto-detected {num_classes_override} classes from dataset manifest")
                    elif dataset_num_classes is not None and num_classes_override != dataset_num_classes:
                        st.warning(f"⚠️ Sidebar specifies {num_classes_override} classes but dataset has {dataset_num_classes}. Using sidebar value.")

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = load_classification_model_for_images(
                        model_name,
                        custom_weights_path=custom_weights,
                        device=device,
                        num_classes=num_classes_override
                    )

                    # Verify model output classes match dataset
                    # Handle both simple Linear layers and Sequential classifiers
                    import torch.nn as nn
                    def get_classifier_out_features(layer):
                        """Get output features from a classifier layer (Linear or Sequential)."""
                        if hasattr(layer, 'out_features'):
                            return layer.out_features
                        elif isinstance(layer, nn.Sequential):
                            # For Sequential, check the last layer
                            for module in reversed(list(layer.modules())):
                                if hasattr(module, 'out_features'):
                                    return module.out_features
                        return None

                    # Detect classifier layer and its output size
                    classifier_layer = None
                    classifier_name = None
                    if hasattr(model, 'classifier'):
                        classifier_layer = model.classifier
                        classifier_name = "classifier"
                        model_num_classes = get_classifier_out_features(model.classifier)
                    elif hasattr(model, 'fc'):
                        classifier_layer = model.fc
                        classifier_name = "fc"
                        model_num_classes = get_classifier_out_features(model.fc)
                    elif hasattr(model, 'head'):
                        classifier_layer = model.head
                        classifier_name = "head"
                        model_num_classes = get_classifier_out_features(model.head)
                    else:
                        model_num_classes = None
                        classifier_name = "unknown"

                    # Always show the actual classifier info for verification
                    if model_num_classes is not None:
                        st.success(f"✅ **Model loaded:** {model_name} with **{model_num_classes} output classes** (layer: {classifier_name})")

                        if dataset_num_classes is not None and model_num_classes != dataset_num_classes:
                            st.error(f"🚨 CRITICAL MISMATCH: Model outputs {model_num_classes} classes but dataset has {dataset_num_classes}!")
                            st.error("Training will likely fail or produce wrong results. Check your configuration.")
                            return
                        elif dataset_num_classes is not None:
                            st.success(f"✅ Classes match: Model ({model_num_classes}) = Dataset ({dataset_num_classes})")
                    else:
                        st.warning(f"⚠️ Could not detect classifier output size for {model_name}")

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

                # Train on train + select combined
                train_tensors_for_model = torch.cat([all_train_tensors, selection_tensors], dim=0)
                train_labels_for_model = torch.cat([all_train_labels, selection_labels], dim=0)

                # Calculate initial class distribution for Step 3 balancing
                from collections import Counter
                initial_class_counts = Counter(train_labels_for_model.tolist())
                initial_total = len(train_labels_for_model)
                initial_class_ratios = {cls: count / initial_total for cls, count in initial_class_counts.items()}

                st.info(f"📊 Step 1: Training on {len(all_train_tensors)} train + {len(selection_tensors)} select = {initial_total} total")

                # Sanity check: verify label indices are valid for model output classes
                all_label_indices = set(train_labels_for_model.tolist()) | set(target_labels.tolist())
                target_label_set = set(target_labels.tolist())
                train_label_set = set(train_labels_for_model.tolist())
                max_label = max(all_label_indices)
                min_label = min(all_label_indices)
                if model_num_classes is not None:
                    if max_label >= model_num_classes:
                        st.error(f"🚨 LABEL ERROR: Dataset has label {max_label} but model only has {model_num_classes} outputs (0-{model_num_classes-1})!")
                        return
                    st.info(f"📋 Train labels: {sorted(train_label_set)} ({len(train_label_set)} classes) | Target labels: {sorted(target_label_set)} ({len(target_label_set)} classes)")

                    # Warn if target is single-class
                    if len(target_label_set) == 1:
                        target_class = list(target_label_set)[0]
                        target_class_name = class_mapping.get(str(target_class), f"class_{target_class}")
                        st.warning(f"⚠️ Target set is single-class ({target_class_name}). High target accuracy (~100%) is expected if model learns this class.")

                with st.status(f"🎓 Training initial model for {train_epochs} epochs on {len(train_tensors_for_model)} images...", expanded=True) as training_status:
                    # Create simple training loop
                    from torch.utils.data import TensorDataset, DataLoader
                    import torch.optim as optim
                    import torch.nn as nn

                    train_dataset = TensorDataset(train_tensors_for_model, train_labels_for_model)
                    # Use seeded generator for reproducible shuffling
                    if enable_reproducibility:
                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=train_batch_size,
                            shuffle=True,
                            generator=get_dataloader_generator(training_seed),
                            worker_init_fn=worker_init_fn
                        )
                    else:
                        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
                    num_batches = len(train_loader)

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
                    # Convert generator to list to ensure we can count params
                    trainable_params_list = list(trainable_params) if not isinstance(trainable_params, list) else trainable_params

                    if step1_optimizer == "AdamW":
                        optimizer = optim.AdamW(trainable_params_list, lr=learning_rate, weight_decay=step1_weight_decay)
                    elif step1_optimizer == "Adam":
                        optimizer = optim.Adam(trainable_params_list, lr=learning_rate, weight_decay=step1_weight_decay)
                    else:  # SGD
                        optimizer = optim.SGD(trainable_params_list, lr=learning_rate, momentum=step1_momentum, weight_decay=step1_weight_decay)

                    # Diagnostic: Show model configuration
                    num_trainable = sum(p.numel() for p in trainable_params_list)
                    num_total = sum(p.numel() for p in model.parameters())
                    num_requires_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)

                    if classifier_layer is not None:
                        # Handle both Linear and Sequential classifiers
                        if hasattr(classifier_layer, 'out_features'):
                            classifier_out_features = classifier_layer.out_features
                        elif isinstance(classifier_layer, nn.Sequential):
                            # Find the last Linear layer in Sequential
                            classifier_out_features = 'unknown'
                            for module in reversed(list(classifier_layer.modules())):
                                if hasattr(module, 'out_features'):
                                    classifier_out_features = module.out_features
                                    break
                        else:
                            classifier_out_features = 'unknown'

                        # CRITICAL: Verify num_classes matches expectations
                        if classifier_out_features != 'unknown' and dataset_num_classes is not None:
                            if classifier_out_features != dataset_num_classes:
                                st.error(f"🚨 CLASS MISMATCH: Model outputs {classifier_out_features} classes but dataset has {dataset_num_classes}!")
                            else:
                                st.success(f"✅ Classes match: Model and dataset both have {classifier_out_features} classes")

                        st.info(f"📊 Model config: {classifier_out_features} output classes, {num_trainable:,} trainable / {num_requires_grad:,} requires_grad / {num_total:,} total, lr={learning_rate:.0e}")
                    else:
                        st.info(f"📊 Model config: {num_trainable:,} trainable params / {num_total:,} total, lr={learning_rate:.0e}")

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

                    # Create a placeholder for epoch metrics table
                    epoch_metrics_placeholder = st.empty()
                    epoch_metrics_list = []

                    for epoch in range(train_epochs):
                        # Training
                        model.train()
                        total_train_loss = 0
                        train_correct = 0
                        train_total = 0

                        # Update progress bar: training is 20% -> 80% of total progress
                        # Each epoch contributes (80-20)/train_epochs = 60/train_epochs percent
                        epoch_progress = 20 + int(60 * epoch / train_epochs)
                        progress_bar.progress(epoch_progress)

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

                            # Track training accuracy
                            _, predicted = torch.max(logits.detach(), 1)
                            train_total += labels.size(0)
                            train_correct += (predicted == labels).sum().item()

                            # Debug: Check gradients on first batch of first epoch
                            if epoch == 0 and batch_idx == 0:
                                grad_norms = []
                                for p in trainable_params_list:
                                    if p.grad is not None:
                                        grad_norms.append(p.grad.norm().item())
                                if grad_norms:
                                    avg_grad = sum(grad_norms) / len(grad_norms)
                                    st.info(f"🔍 Gradient check: avg grad norm = {avg_grad:.4f}")
                                    if avg_grad < 1e-8:
                                        st.error("🚨 Gradients are near zero! Training will not converge.")
                                else:
                                    st.error("🚨 No gradients found! Check requires_grad settings.")

                            optimizer.step()

                            total_train_loss += loss.item()

                            # Batch-level status update (every 5 batches or on last batch)
                            if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
                                training_status.update(label=f"🎓 Epoch {epoch+1}/{train_epochs} | Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")

                        avg_train_loss = total_train_loss / len(train_loader)
                        train_acc = 100 * train_correct / train_total

                        # Evaluate on target set
                        model.eval()
                        total_target_loss = 0
                        correct = 0
                        total = 0
                        all_predictions = []

                        # Use reduction='sum' for correct averaging across variable batch sizes
                        criterion_eval = nn.CrossEntropyLoss(reduction='sum')

                        with torch.no_grad():
                            for i in range(0, len(target_tensors), train_batch_size):
                                batch_images = target_tensors[i:i+train_batch_size].to(device)
                                batch_labels = target_labels[i:i+train_batch_size].to(device)

                                outputs = model(batch_images)
                                if hasattr(outputs, 'logits'):
                                    logits = outputs.logits
                                else:
                                    logits = outputs

                                # Sum of losses for this batch (not mean)
                                loss = criterion_eval(logits, batch_labels)
                                total_target_loss += loss.item()

                                _, predicted = torch.max(logits, 1)
                                total += batch_labels.size(0)
                                correct += (predicted == batch_labels).sum().item()

                                all_predictions.extend(predicted.cpu().tolist())

                        # Correct mean: total sum / total number of samples
                        avg_target_loss = total_target_loss / len(target_tensors)
                        target_acc = 100 * correct / total

                        # Save checkpoint if this epoch is a checkpoint epoch (1-indexed)
                        current_epoch_1indexed = epoch + 1
                        ckpt_marker = ""
                        if current_epoch_1indexed in checkpoint_epochs:
                            # Deep copy the state dict to CPU
                            checkpoint_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                            model_checkpoints[current_epoch_1indexed] = checkpoint_state
                            ckpt_marker = "📸"

                        # Store metrics for this epoch
                        epoch_metrics_list.append({
                            "Epoch": f"{epoch+1}/{train_epochs}",
                            "Train Loss": f"{avg_train_loss:.4f}",
                            "Train Acc": f"{train_acc:.1f}%",
                            "Target Loss": f"{avg_target_loss:.4f}",
                            "Target Acc": f"{target_acc:.1f}%",
                            "": ckpt_marker  # Checkpoint marker column
                        })

                        # Update the metrics table (show last 10 epochs to avoid clutter)
                        display_metrics = epoch_metrics_list[-10:] if len(epoch_metrics_list) > 10 else epoch_metrics_list
                        df = pd.DataFrame(display_metrics)
                        epoch_metrics_placeholder.dataframe(df, use_container_width=True, hide_index=True)

                        # Update status with current epoch info
                        training_status.update(label=f"🎓 Epoch {epoch+1}/{train_epochs} | Train: {train_acc:.1f}% | Target: {target_acc:.1f}%")

                        # Final progress update for this epoch
                        epoch_progress = 20 + int(60 * (epoch + 1) / train_epochs)
                        progress_bar.progress(epoch_progress)

                    model.eval()
                    training_status.update(label="✅ Training complete!", state="complete")

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

                    st.write(f"  **Final**: Train Loss = {avg_train_loss:.4f} (Acc: {train_acc:.1f}%) | Target Loss = {avg_target_loss:.4f} (Acc: {target_acc:.1f}%)")
                    st.write(f"  **Predictions**: {top_3_str}")

                    # Debug: Show raw class indices to help diagnose mismatches
                    with st.expander("🔍 Debug: Class mapping details", expanded=False):
                        unique_labels = set(target_labels.tolist())
                        unique_preds = set(all_predictions)
                        target_indices_debug = dataset_info.get('target_classes', [dataset_info.get('target_class', 0)])
                        target_names_debug = dataset_info.get('target_class_names', [dataset_info.get('target_class_name', '?')])
                        st.write(f"**Target labels (ground truth):** {sorted(unique_labels)} → should be classes {target_indices_debug} ({target_names_debug})")
                        st.write(f"**Model predictions (unique):** {sorted(unique_preds)}")
                        st.write(f"**Class mapping being used:**")
                        target_set_debug = set(target_indices_debug)
                        for k, v in sorted(class_mapping.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
                            marker = "← target" if int(k) in target_set_debug else ""
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
                    # Store full train data for Step 3 class balancing
                    "train_tensors": all_train_tensors.cpu(),
                    "train_labels": all_train_labels.cpu(),
                    "train_paths": all_train_valid_paths,
                    # Initial class ratios from train+select for Step 3 balancing
                    "initial_class_ratios": initial_class_ratios,
                    "initial_class_counts": dict(initial_class_counts),
                    "manifest": manifest,
                    "model_name": model_name,
                    "custom_weights": custom_weights,
                    "num_classes_override": num_classes_override,  # User-specified class count override
                    "device": str(device),
                    "train_batch_size": train_batch_size,
                    "train_epochs": train_epochs,
                    "learning_rate": learning_rate,
                    "optimizer": step1_optimizer,
                    "freeze_backbone": freeze_backbone,
                    "enable_reproducibility": enable_reproducibility,
                    "training_seed": training_seed,
                    "pool_label": pool_label,
                    "temp_dir": temp_dir,
                    "dataset_root": dataset_root,
                    "class_mapping": class_mapping,
                    "dataset_info": dataset_info,
                    "num_checkpoints": len(checkpoints_to_save),
                    # Training results for display after rerun
                    "epoch_metrics": epoch_metrics_list,
                    "final_train_loss": avg_train_loss,
                    "final_target_loss": avg_target_loss,
                    "final_train_acc": train_acc,
                    "final_target_acc": target_acc,
                    "final_predictions": all_predictions,
                }

                # In auto-run mode, advance to Step 2
                if auto_run_mode:
                    st.session_state["auto_run_step"] = 2
                    st.info("💾 Model saved. Auto-running Step 2...")
                else:
                    st.info("💾 Model and data saved. Click **Step 2** to compute TRAK scores.")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during Step 1: {str(e)}")
                st.error(traceback.format_exc())
                return

    # ==================== STEP 2: COMPUTE TRAK SCORES ====================
    step2_button_disabled = not step1_complete or step2_complete

    if step1_complete and not step2_complete:
        # Display Step 1 training results in expander (persisted from training)
        step1_data = st.session_state.get("step1_data", {})
        epoch_metrics = step1_data.get("epoch_metrics", [])
        if epoch_metrics:
            with st.expander("📈 Step 1 Training Results", expanded=True):
                # Show model and training configuration
                st.markdown("**Run Configuration:**")
                model_name = step1_data.get("model_name", "unknown")
                num_classes = step1_data.get("num_classes_override") or step1_data.get("dataset_info", {}).get("num_classes", 1000)
                custom_weights = step1_data.get("custom_weights")
                lr = step1_data.get("learning_rate", 0)
                batch_size = step1_data.get("train_batch_size", 0)
                epochs = step1_data.get("train_epochs", 0)
                optimizer = step1_data.get("optimizer", "AdamW")
                freeze_backbone = step1_data.get("freeze_backbone", True)

                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                with col_c1:
                    st.write(f"**Model**: {model_name}")
                    st.write(f"**Output classes**: {num_classes}")
                with col_c2:
                    st.write(f"**Custom weights**: {'Yes' if custom_weights else 'No'}")
                    st.write(f"**Backbone**: {'Frozen' if freeze_backbone else 'Trainable'}")
                with col_c3:
                    st.write(f"**Epochs**: {epochs}")
                    st.write(f"**Batch size**: {batch_size}")
                with col_c4:
                    st.write(f"**Learning rate**: {lr:.0e}")
                    st.write(f"**Optimizer**: {optimizer}")

                st.markdown("---")
                st.markdown("**Training Results:**")

                # Show final metrics
                final_train_acc = step1_data.get("final_train_acc", 0)
                final_target_acc = step1_data.get("final_target_acc", 0)
                final_train_loss = step1_data.get("final_train_loss", 0)
                final_target_loss = step1_data.get("final_target_loss", 0)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train Accuracy", f"{final_train_acc:.1f}%")
                with col2:
                    st.metric("Target Accuracy", f"{final_target_acc:.1f}%")
                with col3:
                    st.metric("Train Loss", f"{final_train_loss:.4f}")
                with col4:
                    st.metric("Target Loss", f"{final_target_loss:.4f}")

                # Show epoch-by-epoch table
                st.markdown("**Epoch-by-epoch metrics:**")
                df = pd.DataFrame(epoch_metrics)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Show prediction distribution
                final_predictions = step1_data.get("final_predictions", [])
                class_mapping = step1_data.get("class_mapping", {})
                if final_predictions:
                    from collections import Counter
                    pred_counts = Counter(final_predictions)
                    top_3 = pred_counts.most_common(3)
                    top_3_str = ", ".join([
                        f"{class_mapping.get(str(cls), f'class_{cls}')} ({count})"
                        for cls, count in top_3
                    ])
                    st.markdown(f"**Top predictions on target set:** {top_3_str}")

        st.markdown("#### 📊 Step 2: Compute TRAK Scores")
        # Show checkpoint info from Step 1 if available
        saved_checkpoints = st.session_state.get("step1_data", {}).get("num_checkpoints", 1)
        ckpt_info = f", {saved_checkpoints} checkpoint(s)" if saved_checkpoints > 1 else ""
        st.caption(f"Using JL dim={jl_dim}, {num_projections} projection(s){ckpt_info}, gradient source: {gradient_source}")

    # Trigger Step 2 via button OR auto-run mode
    step2_triggered = st.button("📊 **Step 2: Compute TRAK Scores**", type="primary", use_container_width=True, disabled=step2_button_disabled)
    if auto_run_mode and auto_run_step == 2 and step1_complete and not step2_complete:
        step2_triggered = True

    if step2_triggered:
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
                # Full train data for Step 3 class balancing
                train_tensors = step1_data["train_tensors"]
                train_labels = step1_data["train_labels"]
                train_paths = step1_data["train_paths"]
                # Initial class ratios from train+select for Step 3 balancing
                initial_class_ratios = step1_data.get("initial_class_ratios", {})
                initial_class_counts = step1_data.get("initial_class_counts", {})
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
                num_classes_override = step1_data.get("num_classes_override")  # May be None
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
                        device=device,
                        num_classes=num_classes_override
                    )
                    ckpt_state_dict = model_checkpoints[ckpt_epoch]
                    model.load_state_dict({k: v.to(device) for k, v in ckpt_state_dict.items()})
                    model.eval()

                    # Calculate base progress for this checkpoint
                    ckpt_base_pct = int((ckpt_idx / num_ckpts) * 90)
                    ckpt_range = int(90 / num_ckpts)

                    # ===== Fused gradient extraction + projection (GPU-efficient) =====
                    # Extract gradients and project with all seeds in one pass
                    # Gradients stay on GPU during projection, avoiding CPU round-trip
                    proj_seeds = [42 + i for i in range(num_projections)]

                    # Selection gradients
                    status_text_proj.text(f"{ckpt_label}: Extracting & projecting selection gradients...")
                    selection_progress = st.empty()

                    def selection_progress_callback(current, total):
                        selection_progress.text(f"  Processing: {current}/{total} images (×{num_projections} projections)")

                    selection_grads_list = extract_and_project_gradients(
                        model,
                        selection_tensors,
                        selection_labels,
                        proj_dim=jl_dim,
                        seeds=proj_seeds,
                        device=str(device),
                        batch_size=train_batch_size,
                        progress_callback=selection_progress_callback,
                        last_layer_only=freeze_backbone
                    )
                    selection_progress.empty()
                    overall_progress.progress(ckpt_base_pct + int(ckpt_range * 0.4))

                    # Free GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                    # Target gradients
                    status_text_proj.text(f"{ckpt_label}: Extracting & projecting target gradients...")
                    target_progress = st.empty()

                    def target_progress_callback(current, total):
                        target_progress.text(f"  Processing: {current}/{total} images (×{num_projections} projections)")

                    target_grads_list = extract_and_project_gradients(
                        model,
                        target_tensors,
                        target_labels,
                        proj_dim=jl_dim,
                        seeds=proj_seeds,
                        device=str(device),
                        batch_size=train_batch_size,
                        progress_callback=target_progress_callback,
                        last_layer_only=freeze_backbone
                    )
                    target_progress.empty()
                    overall_progress.progress(ckpt_base_pct + int(ckpt_range * 0.7))

                    # Free GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

                    # ===== Compute TRAK scores for each projection =====
                    for proj_idx in range(num_projections):
                        proj_seed = proj_seeds[proj_idx]
                        proj_label = f"Proj {proj_idx + 1}/{num_projections}" if num_projections > 1 else ""
                        combined_label = f"{ckpt_label} | {proj_label}".strip(" |")

                        # Calculate progress within this checkpoint
                        proj_pct = int(ckpt_range * 0.7) + int((proj_idx / num_projections) * ckpt_range * 0.3)

                        # Get pre-projected gradients for this seed
                        selection_grads = selection_grads_list[proj_idx]
                        target_grads = target_grads_list[proj_idx]

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

                    # Free projected gradients for this checkpoint
                    del selection_grads_list, target_grads_list, model
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
                    # Full train data for Step 3 class balancing
                    "train_tensors": train_tensors.cpu(),
                    "train_labels": train_labels.cpu(),
                    "train_paths": train_paths,
                    # Initial class ratios from train+select for balancing
                    "initial_class_ratios": initial_class_ratios,
                    "initial_class_counts": initial_class_counts,
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
                    "num_classes_override": num_classes_override,  # User-specified class count override
                    # Reproducibility settings from Step 1
                    "enable_reproducibility": step1_data.get("enable_reproducibility", True),
                    "training_seed": step1_data.get("training_seed", 42),
                }

                st.success("✅ Step 2 Complete! TRAK scores computed.")

                # Auto-save dataset_trak.json to local folder (only if using local path, not ZIP)
                if temp_dir is None and dataset_root:
                    try:
                        # Build dataset_trak.json with TRAK scores
                        score_lookup = dict(zip(scores_df_sorted['filename'], scores_df_sorted['trak_score']))
                        manifest_with_trak = copy.deepcopy(manifest)
                        for entry in manifest_with_trak.get('entries', []):
                            filename = entry.get('filename', '')
                            entry['trak_score'] = score_lookup.get(filename, None)

                        # Add TRAK metadata
                        manifest_with_trak['trak_info'] = {
                            'computed_at': datetime.now().isoformat(),
                            'jl_dim': jl_dim,
                            'num_projections': num_projections,
                            'num_checkpoints': num_checkpoints,
                            'model_name': model_name,
                            'train_epochs': train_epochs,
                            'min_score': float(scores_df_sorted['trak_score'].min()),
                            'max_score': float(scores_df_sorted['trak_score'].max()),
                            'mean_score': float(scores_df_sorted['trak_score'].mean()),
                        }

                        trak_json_path = os.path.join(dataset_root, 'dataset_trak.json')
                        with open(trak_json_path, 'w') as f:
                            json.dump(manifest_with_trak, f, indent=2)
                        st.success(f"💾 Auto-saved `dataset_trak.json` to {dataset_root}")
                        # Clear force_reset_trak flag since we have fresh TRAK scores
                        if "force_reset_trak" in st.session_state:
                            del st.session_state["force_reset_trak"]
                    except Exception as e:
                        st.warning(f"⚠️ Could not auto-save dataset_trak.json: {str(e)}")

                # In auto-run mode, advance to Step 3 (compare all methods)
                if auto_run_mode:
                    st.session_state["auto_run_step"] = 3
                    st.info("💾 Results saved. Auto-running Compare All Methods...")
                else:
                    st.info("💾 Results saved. You can now proceed to **Step 3** to select and retrain.")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during Step 2: {str(e)}")
                st.error(traceback.format_exc())
                return

    # STEP 2.5: Display TRAK Scores (separate from main button to prevent reset on interaction)
    if "trak_data" in st.session_state:
        # Display Step 1 training results in expander (persisted)
        step1_data = st.session_state.get("step1_data", {})
        epoch_metrics = step1_data.get("epoch_metrics", [])
        if epoch_metrics:
            with st.expander("📈 Step 1 Training Results", expanded=False):
                # Show model and training configuration
                st.markdown("**Run Configuration:**")
                model_name = step1_data.get("model_name", "unknown")
                num_classes = step1_data.get("num_classes_override") or step1_data.get("dataset_info", {}).get("num_classes", 1000)
                custom_weights = step1_data.get("custom_weights")
                lr = step1_data.get("learning_rate", 0)
                batch_size = step1_data.get("train_batch_size", 0)
                epochs = step1_data.get("train_epochs", 0)
                optimizer = step1_data.get("optimizer", "AdamW")
                freeze_backbone = step1_data.get("freeze_backbone", True)

                col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                with col_c1:
                    st.write(f"**Model**: {model_name}")
                    st.write(f"**Output classes**: {num_classes}")
                with col_c2:
                    st.write(f"**Custom weights**: {'Yes' if custom_weights else 'No'}")
                    st.write(f"**Backbone**: {'Frozen' if freeze_backbone else 'Trainable'}")
                with col_c3:
                    st.write(f"**Epochs**: {epochs}")
                    st.write(f"**Batch size**: {batch_size}")
                with col_c4:
                    st.write(f"**Learning rate**: {lr:.0e}")
                    st.write(f"**Optimizer**: {optimizer}")

                st.markdown("---")
                st.markdown("**Training Results:**")

                final_train_acc = step1_data.get("final_train_acc", 0)
                final_target_acc = step1_data.get("final_target_acc", 0)
                final_train_loss = step1_data.get("final_train_loss", 0)
                final_target_loss = step1_data.get("final_target_loss", 0)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train Accuracy", f"{final_train_acc:.1f}%")
                with col2:
                    st.metric("Target Accuracy", f"{final_target_acc:.1f}%")
                with col3:
                    st.metric("Train Loss", f"{final_train_loss:.4f}")
                with col4:
                    st.metric("Target Loss", f"{final_target_loss:.4f}")

                df = pd.DataFrame(epoch_metrics)
                st.dataframe(df, use_container_width=True, hide_index=True)

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

            # Get dataset name for download filenames
            dataset_root = trak_data.get("dataset_root", "")
            dataset_name = Path(dataset_root).name if dataset_root else "dataset"
            # Remove common prefixes/suffixes for cleaner names
            if dataset_name.endswith(".zip"):
                dataset_name = dataset_name[:-4]

            with col_dl_csv:
                scores_csv = scores_df_sorted.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=scores_csv,
                    file_name=f"{dataset_name}_trak_scores.csv",
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

                # Auto-save enriched manifest to dataset folder (if we have a local path)
                dataset_root = trak_data.get("dataset_root")
                if dataset_root and os.path.isdir(dataset_root):
                    auto_save_path = os.path.join(dataset_root, "dataset.json")
                    save_key = f"trak_scores_saved_{hash(manifest_json) % 100000}"
                    if save_key not in st.session_state:
                        try:
                            with open(auto_save_path, 'w') as f:
                                f.write(manifest_json)
                            st.session_state[save_key] = True
                            st.success(f"✅ TRAK scores auto-saved to `{auto_save_path}`")
                        except Exception as e:
                            st.warning(f"⚠️ Could not auto-save: {e}")

                st.download_button(
                    label="📥 Download JSON",
                    data=manifest_json,
                    file_name=f"{dataset_name}_trak.json",
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

            # Download button for displayed subset (ZIP with images + JSON)
            displayed_filenames = set(display_df['filename'].tolist())
            score_lookup = dict(zip(display_df['filename'], display_df['trak_score']))
            rank_lookup = dict(zip(display_df['filename'], display_df['rank']))

            # Build subset manifest with only displayed images (paths relative to ZIP)
            subset_manifest = copy.deepcopy(manifest)
            subset_entries = []
            for entry in subset_manifest.get('entries', []):
                entry_filename = os.path.basename(entry.get('filename', ''))
                if entry_filename in displayed_filenames:
                    # Update path to be relative within the ZIP (just filename)
                    entry['filename'] = entry_filename
                    entry['trak_score'] = float(score_lookup.get(entry_filename, 0))
                    entry['trak_rank'] = int(rank_lookup.get(entry_filename, 0))
                    subset_entries.append(entry)
            subset_manifest['entries'] = subset_entries

            # Update dataset info
            subset_manifest['subset_info'] = {
                'selection_mode': view_mode,
                'k_value': k_value,
                'total_images': len(scores_df_sorted),
                'subset_size': len(display_df),
                'created_at': datetime.now().isoformat(),
                'score_range': {
                    'min': float(display_df['trak_score'].min()),
                    'max': float(display_df['trak_score'].max()),
                    'mean': float(display_df['trak_score'].mean()),
                }
            }

            # Build ZIP with images + JSON manifest
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add images
                for path in selection_valid_paths:
                    filename = os.path.basename(path)
                    if filename in displayed_filenames:
                        try:
                            with open(path, 'rb') as f:
                                zf.writestr(filename, f.read())
                        except Exception:
                            pass  # Skip files that can't be read

                # Add JSON manifest
                subset_json = json.dumps(subset_manifest, indent=2)
                zf.writestr("dataset.json", subset_json)

            zip_buffer.seek(0)
            view_suffix = "top" if "Top" in view_mode else ("bottom" if "Bottom" in view_mode else "all")
            # Get dataset name for filename
            dataset_root_dl = trak_data.get("dataset_root", "")
            dataset_name_dl = Path(dataset_root_dl).name if dataset_root_dl else "dataset"
            if dataset_name_dl.endswith(".zip"):
                dataset_name_dl = dataset_name_dl[:-4]
            zip_filename = f"{dataset_name_dl}_{view_suffix}_{len(display_df)}.zip"

            st.download_button(
                label=f"📥 Download {len(display_df)} Images",
                data=zip_buffer.getvalue(),
                file_name=zip_filename,
                mime="application/zip",
                use_container_width=True,
                help=f"Download ZIP with {len(display_df)} images and dataset.json containing TRAK scores/ranks"
            )

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
        num_classes_override = trak_data.get("num_classes_override")  # May be None
        device = torch.device(trak_data["device"])
        train_batch_size = trak_data["train_batch_size"]
        train_epochs = trak_data["train_epochs"]
        pool_label = trak_data["pool_label"]

        # Reproducibility settings from Step 1
        enable_reproducibility = trak_data.get("enable_reproducibility", True)
        training_seed = trak_data.get("training_seed", 42)

        # Full train data for class balancing
        train_tensors = trak_data.get("train_tensors")
        train_labels = trak_data.get("train_labels")
        train_paths = trak_data.get("train_paths")
        has_train_data = train_tensors is not None and train_labels is not None

        # Initial class ratios from Step 1 (train+select combined)
        initial_class_ratios = trak_data.get("initial_class_ratios", {})
        initial_class_counts = trak_data.get("initial_class_counts", {})

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

        # Per-class selection option
        per_class_selection = st.checkbox(
            "📊 Select per-class (balanced)",
            value=True,
            key="step3_per_class",
            help="If enabled, selects top/bottom k% from EACH class separately (maintaining class balance). Otherwise, selects globally regardless of class."
        )

        # Show selection summary
        pct_of_total = 100 * n_select / len(selection_valid_paths)

        # Calculate expected selection counts
        if per_class_selection and subset_method != "Random":
            # Per-class selection: k% from each class
            num_classes = len(set(selection_labels.tolist()))
            imgs_per_class = n_select // num_classes if num_classes > 0 else n_select
            st.info(f"📌 **Per-class selection**: ~{pct_of_total:.1f}% from each of {num_classes} classes = ~{imgs_per_class} per class, **{n_select}** total")
        elif initial_class_ratios and has_train_data:
            # Global selection with post-hoc balancing (legacy mode)
            total_initial = sum(initial_class_counts.values()) if initial_class_counts else len(train_labels)
            target_total = int(total_initial * pct_of_total / 100)
            st.info(f"📌 **Global selection**: {n_select} images ({pct_of_total:.1f}%), then balance to ~{target_total} total")
        else:
            st.info(f"📌 Will select **{n_select}** images ({pct_of_total:.1f}% of {len(selection_valid_paths)}) using: **{subset_method}**")
            if not initial_class_ratios:
                st.warning("⚠️ No initial class ratios available for balancing. Re-run Step 1.")

        # Training Parameters for Step 3 (can override Step 1 defaults)
        st.markdown("#### ⚙️ Training Parameters")
        col_epochs, col_bs, col_lr = st.columns(3)

        with col_epochs:
            step3_epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=500,
                value=20,  # Default 20 epochs for comparison
                step=1,
                key="step3_train_epochs",
                help="Number of training epochs (20 recommended for comparison)"
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
            col_opt1, col_opt2 = st.columns(2)

            with col_opt1:
                optimizer_choice = st.selectbox(
                    "Optimizer",
                    options=["AdamW", "Adam", "SGD"],
                    index=0,
                    key="step3_optimizer",
                    help="Optimization algorithm"
                )

            with col_opt2:
                step3_seed = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=999999,
                    value=training_seed,  # Default from Step 1
                    step=1,
                    key="step3_seed",
                    help="Random seed for reproducibility. Change this to run multiple experiments with different initializations."
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
        st.markdown("---")
        st.markdown("#### 🔬 Compare All Methods")

        # Multi-percentage selection for comprehensive comparison
        percentages_input = st.text_input(
            "Percentages to compare",
            value="10,20,30,40,50,60,70,80,90",
            key="compare_percentages",
            help="Enter percentages as comma-separated integers (e.g., 5, 25, 50, 75, 100)"
        )

        # Parse input into sorted list of unique percentages
        compare_percentages = []
        if percentages_input.strip():
            try:
                compare_percentages = sorted(set(
                    int(x.strip()) for x in percentages_input.split(",")
                    if x.strip() and 0 < int(x.strip()) <= 100
                ))
            except ValueError:
                st.warning("⚠️ Please enter comma-separated integers (e.g., 10, 25, 50).")

        col_retrain, col_compare = st.columns([1, 1])

        with col_compare:
            compare_btn = st.button("🔬 Compare All Methods", type="secondary", use_container_width=True, help="Run Top, Bottom, and Random selection methods across all selected percentages")

        # Trigger in auto-run mode
        if auto_run_mode and auto_run_step == 3 and step2_complete:
            compare_btn = True
            # Use default percentages in auto-run mode if none selected
            if not compare_percentages:
                compare_percentages = [10, 20, 30]

        if compare_btn and compare_percentages:
            # Set seeds for reproducibility at the start of comparison
            if enable_reproducibility:
                set_seed(step3_seed)
                st.info(f"🎲 Reproducibility enabled for comparison with seed: {step3_seed}")

            st.markdown("### 🔬 Multi-Percentage Method Comparison")

            # Auto-detect num_classes from manifest if not set
            if num_classes_override is None:
                dataset_info = manifest.get('dataset_info', {})
                dataset_num_classes = dataset_info.get('num_classes')
                if dataset_num_classes is not None:
                    num_classes_override = dataset_num_classes
                    st.info(f"🎯 Compare: Auto-detected {num_classes_override} classes from dataset manifest")

            st.info(f"Training and evaluating **{len(compare_percentages)} percentages** × **3 methods** = **{len(compare_percentages) * 3}** experiments (model: {num_classes_override or 1000} classes)")

            comparison_results = []
            methods = [
                ("Top", True),
                ("Bottom", False),
                ("Random", None),
            ]

            total_experiments = len(compare_percentages) * len(methods)
            overall_progress = st.progress(0)
            method_status = st.empty()
            experiment_idx = 0

            for pct in compare_percentages:
                n_select_pct = max(1, int(len(selection_valid_paths) * pct / 100.0))

                for method_name, descending in methods:
                    method_status.markdown(f"**Running {pct}% - {method_name}...** ({experiment_idx + 1}/{total_experiments})")

                    # Select indices based on method
                    if per_class_selection and descending is not None:
                        # Per-class selection: select top/bottom k% from EACH class
                        import random
                        from collections import defaultdict

                        # Group indices by class
                        class_to_indices = defaultdict(list)
                        for idx, label in enumerate(selection_labels.tolist()):
                            class_to_indices[label].append(idx)

                        # Select k% from each class
                        selected_indices_list = []
                        for cls, indices in class_to_indices.items():
                            n_per_class = max(1, int(len(indices) * pct / 100.0))
                            cls_scores = importance_scores[indices]
                            sorted_local = torch.argsort(cls_scores, descending=descending)[:n_per_class]
                            selected_indices_list.extend([indices[i] for i in sorted_local.tolist()])

                        selected_indices = torch.tensor(selected_indices_list)
                    elif descending is not None:
                        # Global selection (original behavior)
                        selected_indices = torch.argsort(importance_scores, descending=descending)[:n_select_pct]
                    else:  # Random
                        import random
                        all_indices = list(range(len(selection_valid_paths)))
                        random.shuffle(all_indices)
                        selected_indices = torch.tensor(all_indices[:n_select_pct])

                    # Get subset data from selection pool
                    method_subset_tensors = selection_tensors[selected_indices]
                    method_subset_labels = selection_labels[selected_indices]

                    # Balance to match initial train+select class ratios
                    if initial_class_ratios and has_train_data:
                        import random
                        from collections import Counter

                        # Count what we have from select
                        selected_class_counts = Counter(method_subset_labels.tolist())

                        # Calculate target total and per-class targets
                        total_initial = sum(initial_class_counts.values())
                        target_total = int(total_initial * pct / 100)

                        # For each class, calculate target count and add from train if needed
                        for cls, ratio in initial_class_ratios.items():
                            target_count = max(1, int(ratio * target_total))
                            current_count = selected_class_counts.get(cls, 0)
                            n_to_add = max(0, target_count - current_count)

                            if n_to_add > 0:
                                # Get indices of this class in train
                                cls_mask = (train_labels == cls)
                                cls_indices = torch.where(cls_mask)[0].tolist()
                                random.shuffle(cls_indices)
                                selected_cls_indices = cls_indices[:n_to_add]

                                if selected_cls_indices:
                                    cls_tensors = train_tensors[selected_cls_indices]
                                    cls_labels_tensor = train_labels[selected_cls_indices]
                                    method_subset_tensors = torch.cat([method_subset_tensors, cls_tensors], dim=0)
                                    method_subset_labels = torch.cat([method_subset_labels, cls_labels_tensor], dim=0)

                    # Train and evaluate
                    def progress_cb(epoch, total):
                        base_progress = experiment_idx / total_experiments
                        epoch_progress = epoch / total / total_experiments
                        overall_progress.progress(min(base_progress + epoch_progress, 1.0))

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
                        num_classes=num_classes_override,
                        enable_reproducibility=enable_reproducibility,
                        training_seed=step3_seed,
                        return_best=True,  # Return best epoch metrics based on target loss
                    )

                    comparison_results.append({
                        "Pct": f"{pct}%",
                        "Pct_num": pct,  # Numeric version for plotting
                        "N": n_select_pct,
                        "Method": method_name,
                        "Train Loss": result["train_loss"],
                        "Target Loss": result["target_loss"],
                        "Target Acc (%)": result["target_acc"],
                        "Best Epoch": result.get("best_epoch", step3_epochs),
                    })

                    experiment_idx += 1

            overall_progress.progress(1.0)
            method_status.markdown("**✅ Comparison Complete!**")

            # Clear auto-run mode after completion
            if auto_run_mode:
                st.session_state["auto_run_mode"] = False
                st.session_state["auto_run_step"] = 0

            # Display comparison table
            df = pd.DataFrame(comparison_results)

            st.markdown("#### 📊 Comparison Results")

            # Create pivot tables for both accuracy and loss
            pivot_acc = df.pivot(index="Pct", columns="Method", values="Target Acc (%)")
            pivot_acc = pivot_acc[["Top", "Bottom", "Random"]]  # Reorder columns

            pivot_loss = df.pivot(index="Pct", columns="Method", values="Target Loss")
            pivot_loss = pivot_loss[["Top", "Bottom", "Random"]]  # Reorder columns

            # Add N column (number of images) - same for all methods at each pct
            n_by_pct = df.groupby("Pct")["N"].first()

            # Build result dataframe with N, Acc and Loss for each method
            result_df = pd.DataFrame(index=pivot_acc.index)
            result_df["N"] = n_by_pct
            for method in ["Top", "Bottom", "Random"]:
                result_df[f"{method} Acc"] = pivot_acc[method]
                result_df[f"{method} Loss"] = pivot_loss[method]

            # Style: highlight max Acc and min Loss per row
            acc_cols = ["Top Acc", "Bottom Acc", "Random Acc"]
            loss_cols = ["Top Loss", "Bottom Loss", "Random Loss"]

            def highlight_best_in_row(row):
                styles = [""] * len(row)
                # Highlight max accuracy (green)
                acc_values = {col: row[col] for col in acc_cols if col in row.index}
                if acc_values:
                    max_acc = max(acc_values.values())
                    for i, col in enumerate(row.index):
                        if col in acc_cols and row[col] == max_acc:
                            styles[i] = "background-color: lightgreen"
                # Highlight min loss (light blue)
                loss_values = {col: row[col] for col in loss_cols if col in row.index}
                if loss_values:
                    min_loss = min(loss_values.values())
                    for i, col in enumerate(row.index):
                        if col in loss_cols and row[col] == min_loss:
                            styles[i] = "background-color: lightblue"
                return styles

            st.dataframe(
                result_df.style.format({
                    "N": "{:d}",
                    "Top Acc": "{:.2f}",
                    "Bottom Acc": "{:.2f}",
                    "Random Acc": "{:.2f}",
                    "Top Loss": "{:.4f}",
                    "Bottom Loss": "{:.4f}",
                    "Random Loss": "{:.4f}",
                }).apply(highlight_best_in_row, axis=1),
                use_container_width=True,
            )

            # Show best configuration overall
            if has_target_data:
                best_idx = df["Target Acc (%)"].idxmax()
                best_row = df.loc[best_idx]
                st.success(f"🏆 **Best Overall**: {best_row['Method']} at {best_row['Pct']} ({best_row['N']} images) with **{best_row['Target Acc (%)']:.2f}%** accuracy (loss: {best_row['Target Loss']:.4f})")

            # Plot comparison results
            st.markdown("#### 📈 Comparison Plots")
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Color scheme for methods
                method_colors = {"Top": "#2ecc71", "Bottom": "#e74c3c", "Random": "#3498db"}
                method_markers = {"Top": "o", "Bottom": "s", "Random": "^"}

                # Get data by method
                for method_name in ["Top", "Bottom", "Random"]:
                    method_data = df[df["Method"] == method_name].sort_values("Pct_num")
                    x = method_data["Pct_num"].values
                    acc = method_data["Target Acc (%)"].values
                    loss = method_data["Target Loss"].values

                    # Accuracy subplot
                    axes[0].plot(x, acc,
                                marker=method_markers[method_name],
                                color=method_colors[method_name],
                                label=method_name, linewidth=2, markersize=8)

                    # Loss subplot
                    axes[1].plot(x, loss,
                                marker=method_markers[method_name],
                                color=method_colors[method_name],
                                label=method_name, linewidth=2, markersize=8)

                # Configure accuracy subplot
                axes[0].set_xlabel("Data Percentage (%)", fontsize=12)
                axes[0].set_ylabel("Target Accuracy (%)", fontsize=12)
                axes[0].set_title("Target Accuracy vs Data Selection %", fontsize=14)
                axes[0].legend(loc="lower right")
                axes[0].grid(True, alpha=0.3)
                axes[0].set_xticks(sorted(df["Pct_num"].unique()))

                # Configure loss subplot
                axes[1].set_xlabel("Data Percentage (%)", fontsize=12)
                axes[1].set_ylabel("Target Loss", fontsize=12)
                axes[1].set_title("Target Loss vs Data Selection % (Best Epoch)", fontsize=14)
                axes[1].legend(loc="upper right")
                axes[1].grid(True, alpha=0.3)
                axes[1].set_xticks(sorted(df["Pct_num"].unique()))

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"⚠️ Could not generate plot: {str(e)}")

            # Store comparison results in session state
            st.session_state["method_comparison_results"] = {
                "results": comparison_results,
                "percentages": compare_percentages,
                "epochs": step3_epochs,
            }

        # Display persisted comparison results (survives reruns from download buttons, etc.)
        if "method_comparison_results" in st.session_state and not (compare_btn and compare_percentages):
            stored_results = st.session_state["method_comparison_results"]
            comparison_results = stored_results["results"]

            if comparison_results:
                st.markdown("---")
                st.markdown("#### 📊 Comparison Results (from previous run)")

                df = pd.DataFrame(comparison_results)

                # Create pivot tables for both accuracy and loss
                pivot_acc = df.pivot(index="Pct", columns="Method", values="Target Acc (%)")
                pivot_acc = pivot_acc[["Top", "Bottom", "Random"]]  # Reorder columns

                pivot_loss = df.pivot(index="Pct", columns="Method", values="Target Loss")
                pivot_loss = pivot_loss[["Top", "Bottom", "Random"]]  # Reorder columns

                # Add N column (number of images) - same for all methods at each pct
                n_by_pct = df.groupby("Pct")["N"].first()

                # Build result dataframe with N, Acc and Loss for each method
                result_df = pd.DataFrame(index=pivot_acc.index)
                result_df["N"] = n_by_pct
                for method in ["Top", "Bottom", "Random"]:
                    result_df[f"{method} Acc"] = pivot_acc[method]
                    result_df[f"{method} Loss"] = pivot_loss[method]

                # Style: highlight max Acc and min Loss per row
                acc_cols = ["Top Acc", "Bottom Acc", "Random Acc"]
                loss_cols = ["Top Loss", "Bottom Loss", "Random Loss"]

                def highlight_best_in_row_persisted(row):
                    styles = [""] * len(row)
                    # Highlight max accuracy (green)
                    acc_values = {col: row[col] for col in acc_cols if col in row.index}
                    if acc_values:
                        max_acc = max(acc_values.values())
                        for i, col in enumerate(row.index):
                            if col in acc_cols and row[col] == max_acc:
                                styles[i] = "background-color: lightgreen"
                    # Highlight min loss (light blue)
                    loss_values = {col: row[col] for col in loss_cols if col in row.index}
                    if loss_values:
                        min_loss = min(loss_values.values())
                        for i, col in enumerate(row.index):
                            if col in loss_cols and row[col] == min_loss:
                                styles[i] = "background-color: lightblue"
                    return styles

                st.dataframe(
                    result_df.style.format({
                        "N": "{:d}",
                        "Top Acc": "{:.2f}",
                        "Bottom Acc": "{:.2f}",
                        "Random Acc": "{:.2f}",
                        "Top Loss": "{:.4f}",
                        "Bottom Loss": "{:.4f}",
                        "Random Loss": "{:.4f}",
                    }).apply(highlight_best_in_row_persisted, axis=1),
                    use_container_width=True,
                    key="comparison_results_persisted"
                )

                # Show best configuration overall
                if "Target Acc (%)" in df.columns:
                    best_idx = df["Target Acc (%)"].idxmax()
                    best_row = df.loc[best_idx]
                    st.success(f"🏆 **Best Overall**: {best_row['Method']} at {best_row['Pct']} ({best_row['N']} images) with **{best_row['Target Acc (%)']:.2f}%** accuracy (loss: {best_row['Target Loss']:.4f})")

                # Plot comparison results (persisted)
                if "Pct_num" in df.columns:
                    st.markdown("#### 📈 Comparison Plots")
                    try:
                        import matplotlib.pyplot as plt

                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                        # Color scheme for methods
                        method_colors = {"Top": "#2ecc71", "Bottom": "#e74c3c", "Random": "#3498db"}
                        method_markers = {"Top": "o", "Bottom": "s", "Random": "^"}

                        # Get data by method
                        for method_name in ["Top", "Bottom", "Random"]:
                            method_data = df[df["Method"] == method_name].sort_values("Pct_num")
                            x = method_data["Pct_num"].values
                            acc = method_data["Target Acc (%)"].values
                            loss = method_data["Target Loss"].values

                            # Accuracy subplot
                            axes[0].plot(x, acc,
                                        marker=method_markers[method_name],
                                        color=method_colors[method_name],
                                        label=method_name, linewidth=2, markersize=8)

                            # Loss subplot
                            axes[1].plot(x, loss,
                                        marker=method_markers[method_name],
                                        color=method_colors[method_name],
                                        label=method_name, linewidth=2, markersize=8)

                        # Configure accuracy subplot
                        axes[0].set_xlabel("Data Percentage (%)", fontsize=12)
                        axes[0].set_ylabel("Target Accuracy (%)", fontsize=12)
                        axes[0].set_title("Target Accuracy vs Data Selection %", fontsize=14)
                        axes[0].legend(loc="lower right")
                        axes[0].grid(True, alpha=0.3)
                        axes[0].set_xticks(sorted(df["Pct_num"].unique()))

                        # Configure loss subplot
                        axes[1].set_xlabel("Data Percentage (%)", fontsize=12)
                        axes[1].set_ylabel("Target Loss", fontsize=12)
                        axes[1].set_title("Target Loss vs Data Selection % (Best Epoch)", fontsize=14)
                        axes[1].legend(loc="upper right")
                        axes[1].grid(True, alpha=0.3)
                        axes[1].set_xticks(sorted(df["Pct_num"].unique()))

                        plt.tight_layout()
                        st.pyplot(fig, key="comparison_plot_persisted")
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"⚠️ Could not generate plot: {str(e)}")

        with col_retrain:
            retrain_btn = st.button("🚀 Retrain Model on Selected Subset", type="primary", use_container_width=True)

        if retrain_btn:
            # Set seeds for reproducibility at the start of Step 3
            if enable_reproducibility:
                set_seed(step3_seed)
                st.info(f"🎲 Reproducibility enabled for Step 3 with seed: {step3_seed}")

            # Create new progress tracking for step 3
            st.markdown("### 🔄 Retraining Progress")
            progress_bar_retrain = st.progress(0)
            status_text_retrain = st.empty()

            status_text_retrain.markdown("**Step 3/3:** 🎯 Selecting subset and retraining model...")
            progress_bar_retrain.progress(5)

            # Select indices based on method (for target class images via TRAK)
            if per_class_selection and subset_method != "Random":
                # Per-class selection: select top/bottom k% from EACH class
                import random
                from collections import defaultdict

                descending = subset_method == "Top (highest influence)"

                # Group indices by class
                class_to_indices = defaultdict(list)
                for idx, label in enumerate(selection_labels.tolist()):
                    class_to_indices[label].append(idx)

                # Calculate selection percentage from n_select
                selection_pct_value = 100.0 * n_select / len(selection_valid_paths)

                # Select k% from each class
                selected_indices_list = []
                for cls, indices in class_to_indices.items():
                    n_per_class = max(1, int(len(indices) * selection_pct_value / 100.0))
                    cls_scores = importance_scores[indices]
                    sorted_local = torch.argsort(cls_scores, descending=descending)[:n_per_class]
                    selected_indices_list.extend([indices[i] for i in sorted_local.tolist()])

                selected_indices = torch.tensor(selected_indices_list)
                st.info(f"📊 Per-class selection: {len(selected_indices)} images across {len(class_to_indices)} classes (~{selection_pct_value:.1f}% from each)")
            elif subset_method == "Top (highest influence)":
                selected_indices = torch.argsort(importance_scores, descending=True)[:n_select]
            elif subset_method == "Bottom (lowest influence)":
                selected_indices = torch.argsort(importance_scores, descending=False)[:n_select]
            else:  # Random
                import random
                all_indices = list(range(len(selection_valid_paths)))
                random.shuffle(all_indices)
                selected_indices = torch.tensor(all_indices[:n_select])

            # Get TRAK-selected images from selection pool
            subset_tensors = selection_tensors[selected_indices]
            subset_labels = selection_labels[selected_indices]

            # Balance to match initial train+select class ratios
            if initial_class_ratios and has_train_data:
                import random
                from collections import Counter

                # Count what we have from select
                selected_class_counts = Counter(subset_labels.tolist())

                # Calculate target total and per-class targets
                total_initial = sum(initial_class_counts.values())
                target_total = int(total_initial * pct_of_total / 100)
                n_added_total = 0

                # For each class, calculate target count and add from train if needed
                for cls, ratio in initial_class_ratios.items():
                    target_count = max(1, int(ratio * target_total))
                    current_count = selected_class_counts.get(cls, 0)
                    n_to_add = max(0, target_count - current_count)

                    if n_to_add > 0:
                        # Get indices of this class in train
                        cls_mask = (train_labels == cls)
                        cls_indices = torch.where(cls_mask)[0].tolist()
                        random.shuffle(cls_indices)
                        selected_cls_indices = cls_indices[:n_to_add]

                        if selected_cls_indices:
                            cls_tensors = train_tensors[selected_cls_indices]
                            cls_labels_tensor = train_labels[selected_cls_indices]
                            subset_tensors = torch.cat([subset_tensors, cls_tensors], dim=0)
                            subset_labels = torch.cat([subset_labels, cls_labels_tensor], dim=0)
                            n_added_total += len(selected_cls_indices)

                st.success(f"✅ Selected {n_select} from select + {n_added_total} from train (balanced to {pct_of_total:.1f}% of initial mix) = {len(subset_tensors)} total")

            progress_bar_retrain.progress(10)

            # Retrain model
            with st.spinner(f"Retraining model on {len(subset_tensors)} selected images..."):
                from torch.utils.data import TensorDataset, DataLoader
                import torch.optim as optim
                import torch.nn as nn

                # Set seeds for reproducibility
                if enable_reproducibility:
                    set_seed(step3_seed)

                subset_dataset = TensorDataset(subset_tensors, subset_labels)
                # Use seeded generator for reproducible shuffling
                if enable_reproducibility:
                    subset_loader = DataLoader(
                        subset_dataset,
                        batch_size=step3_batch_size,
                        shuffle=True,
                        generator=get_dataloader_generator(step3_seed),
                        worker_init_fn=worker_init_fn
                    )
                else:
                    subset_loader = DataLoader(subset_dataset, batch_size=step3_batch_size, shuffle=True)

                # Reset model (reload from scratch)
                # Auto-detect num_classes from manifest if not set
                if num_classes_override is None:
                    dataset_info = manifest.get('dataset_info', {})
                    dataset_num_classes = dataset_info.get('num_classes')
                    if dataset_num_classes is not None:
                        num_classes_override = dataset_num_classes
                        st.info(f"🎯 Step 3: Auto-detected {num_classes_override} classes from dataset manifest")

                model = load_classification_model_for_images(
                    model_name,
                    custom_weights_path=custom_weights,
                    device=device,
                    num_classes=num_classes_override
                )

                # Verify and show model output classes
                import torch.nn as nn
                if hasattr(model, 'classifier'):
                    if hasattr(model.classifier, 'out_features'):
                        model_classes = model.classifier.out_features
                    elif isinstance(model.classifier, nn.Sequential):
                        for m in reversed(list(model.classifier.modules())):
                            if hasattr(m, 'out_features'):
                                model_classes = m.out_features
                                break
                    else:
                        model_classes = "unknown"
                elif hasattr(model, 'fc'):
                    model_classes = model.fc.out_features if hasattr(model.fc, 'out_features') else "unknown"
                else:
                    model_classes = "unknown"
                st.info(f"📊 Step 3 Model: {model_classes} output classes")

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

                        # Use reduction='sum' for correct averaging across variable batch sizes
                        criterion_eval = nn.CrossEntropyLoss(reduction='sum')

                        with torch.no_grad():
                            for i in range(0, len(target_tensors), step3_batch_size):
                                batch_images = target_tensors[i:i+step3_batch_size].to(device)
                                batch_labels = target_labels[i:i+step3_batch_size].to(device)

                                outputs = model(batch_images)
                                if hasattr(outputs, 'logits'):
                                    logits = outputs.logits
                                else:
                                    logits = outputs

                                # Sum of losses for this batch (not mean)
                                loss = criterion_eval(logits, batch_labels)
                                total_target_loss += loss.item()

                                _, predicted = torch.max(logits, 1)
                                total += batch_labels.size(0)
                                correct += (predicted == batch_labels).sum().item()

                        # Correct mean: total sum / total number of samples
                        avg_target_loss = total_target_loss / len(target_tensors)
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
                # Get dataset name for filename
                trak_data_for_zip = st.session_state.get("trak_data", {})
                dataset_root_for_zip = trak_data_for_zip.get("dataset_root", "")
                dataset_name_for_zip = Path(dataset_root_for_zip).name if dataset_root_for_zip else "dataset"
                if dataset_name_for_zip.endswith(".zip"):
                    dataset_name_for_zip = dataset_name_for_zip[:-4]

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
                    file_name=f"{dataset_name_for_zip}_selected_{results['n_selected']}.zip",
                    mime="application/zip",
                    use_container_width=True
                )


# File Uploads & Cache
