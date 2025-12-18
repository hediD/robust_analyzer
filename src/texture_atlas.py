# -*- coding: utf-8 -*-
"""
Texture Atlas module for 3D Adversarial Robustness Analyzer.

Contains functions for:
- Creating texture atlases from OBJ/MTL files
- Visualizing textures
- Handling texture uploads and processing
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

def visualize_textures(textures_info: List[Dict], title: str = "Textures", compact: bool = False, show_used_status: bool = False) -> None:
    """Display texture images in a grid layout."""
    if not textures_info:
        st.info("No textures to display")
        return

    st.subheader(f"🖼️ {title}")

    # Calculate grid layout - more columns if compact
    num_textures = len(textures_info)
    cols_per_row = min(4 if compact else 3, num_textures)
    rows = (num_textures + cols_per_row - 1) // cols_per_row

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            texture_idx = row * cols_per_row + col_idx
            if texture_idx < num_textures:
                with cols[col_idx]:
                    tex_info = textures_info[texture_idx]

                    # Show usage status if requested
                    if show_used_status:
                        status_icon = "✅" if tex_info.get("used_in_atlas", False) else "⚪"
                        caption = f"{status_icon} {tex_info['name']}\n{tex_info['size']}"
                    else:
                        caption = f"{tex_info['name']}\n{tex_info['size']}"

                    st.image(
                        tex_info["image"],
                        caption=caption,
                        use_container_width=True
                    )


def create_texture_atlas(obj_file, mtl_file, texture_files: List = None) -> Optional[Dict]:
    """
    Create texture atlas using the make_atlas.py script.

    Returns:
        Dict with paths to generated files or None if failed
    """
    if not obj_file or not mtl_file:
        st.error("Both OBJ and MTL files are required for atlas creation")
        return None

    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            obj_path = os.path.join(temp_dir, obj_file.name)
            mtl_path = os.path.join(temp_dir, mtl_file.name)

            with open(obj_path, "wb") as f:
                f.write(obj_file.read())
            with open(mtl_path, "wb") as f:
                f.write(mtl_file.read())

            # Save texture files if provided
            if texture_files:
                for tex_file in texture_files:
                    tex_path = os.path.join(temp_dir, tex_file.name)
                    with open(tex_path, "wb") as f:
                        f.write(tex_file.read())

            # Define output paths
            output_obj = os.path.join(temp_dir, "atlas_combined.obj")
            output_mtl = os.path.join(temp_dir, "atlas_combined.mtl")
            output_atlas = os.path.join(temp_dir, "atlas.png")

            # Get the script path relative to current file
            script_dir = Path(__file__).parent.absolute()
            atlas_script = script_dir / "make_atlas.py"

            # Call make_atlas.py script
            cmd = [
                sys.executable, str(atlas_script),
                obj_path, mtl_path, output_obj, output_mtl, output_atlas
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)

            if result.returncode != 0:
                st.error(f"Atlas creation failed: {result.stderr}")
                return None

            # Read generated files into memory
            atlas_data = {}

            if os.path.exists(output_obj):
                with open(output_obj, "rb") as f:
                    atlas_data["obj_content"] = f.read()
                    atlas_data["obj_name"] = "atlas_combined.obj"

            if os.path.exists(output_mtl):
                with open(output_mtl, "rb") as f:
                    atlas_data["mtl_content"] = f.read()
                    atlas_data["mtl_name"] = "atlas_combined.mtl"

            if os.path.exists(output_atlas):
                with open(output_atlas, "rb") as f:
                    atlas_data["atlas_content"] = f.read()
                    atlas_data["atlas_name"] = "atlas.png"
                # Also load for preview
                atlas_data["atlas_image"] = Image.open(output_atlas)

            st.success("✅ Texture atlas created successfully!")
            return atlas_data

    except Exception as e:
        st.error(f"Error creating texture atlas: {str(e)}")
        return None


def handle_texture_atlas():
    """Handle texture atlas creation interface."""
    st.subheader("🔧 Texture Atlas Creator")

    # Help expander with documentation
    with st.expander("📖 How to Use Texture Atlas Creator", expanded=False):
        st.markdown("""
### Overview

**Texture Atlas Creator** combines multiple texture files from a 3D model into a single unified atlas texture. This simplifies models and improves rendering performance.

### Why Use a Texture Atlas?

| Benefit | Description |
|---------|-------------|
| **Fewer draw calls** | Single texture = better GPU performance |
| **Simpler models** | One material instead of many |
| **Compatibility** | Some renderers only support single textures |
| **File management** | Fewer texture files to track |

### File Requirements

| File Type | Format | Required | Description |
|-----------|--------|----------|-------------|
| 3D Mesh | `.obj` | ✅ Yes | OBJ file with UV coordinates |
| Materials | `.mtl` | ✅ Yes | MTL file referencing multiple textures |
| Textures | `.png`, `.jpg`, `.bmp`, `.tga` | ✅ Yes | All textures referenced in MTL |

### Input Modes

**ZIP Package Mode:**
```
model.zip/
├── model.obj           # 3D mesh at root
├── model.mtl           # Materials at root
└── textures/           # Textures folder
    ├── body.png
    ├── wheels.png
    └── details.png
```

**Individual Files Mode:**
- Upload OBJ and MTL files separately
- Upload all texture files referenced in the MTL

### How It Works

1. **Parses MTL** to find all referenced textures
2. **Loads textures** and determines optimal atlas size
3. **Packs textures** into a single atlas image (bin packing)
4. **Remaps UVs** in the OBJ file to use atlas coordinates
5. **Generates** new OBJ, MTL, and atlas texture files

### Output Files

| File | Description |
|------|-------------|
| `model_atlas.obj` | Mesh with remapped UV coordinates |
| `model_atlas.mtl` | Single-material MTL referencing atlas |
| `atlas_texture.png` | Combined texture atlas |

### Tips

- Ensure UV coordinates don't overlap (per-material UVs are fine)
- Larger textures = larger atlas (memory consideration)
- Check atlas preview to verify texture placement
- Original UVs are preserved within their atlas region
        """)

    st.info("💡 Combine multiple textures from an OBJ/MTL into a single atlas texture")

    mode = st.radio(
        "Input source",
        options=["ZIP package", "Individual files"],
        horizontal=True,
        key="atlas_input_mode",
    )

    if mode == "ZIP package":
        atlas_zip_file = st.file_uploader(
            "Upload ZIP package (OBJ+MTL at root, textures/ folder included)",
            type=["zip"],
            key="atlas_zip_upload",
            help="ZIP should contain: root: *.obj, *.mtl; and a textures/ folder containing all referenced textures",
        )
        if atlas_zip_file:
            st.success(f"✅ Uploaded package: {atlas_zip_file.name}")

        if atlas_zip_file and st.button("🔧 Create Texture Atlas", type="primary", key="create_atlas_zip_btn"):
            with st.spinner("Creating texture atlas from ZIP..."):
                atlas_result = create_texture_atlas_from_zip(atlas_zip_file)
                if atlas_result:
                    st.session_state["atlas_result"] = atlas_result
                    st.rerun()

    else:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Required Files**")
            atlas_obj_file = st.file_uploader(
                "Upload OBJ file",
                type=["obj"],
                help="3D mesh file with multiple materials",
                key="atlas_obj_upload"
            )
            atlas_mtl_file = st.file_uploader(
                "Upload MTL file",
                type=["mtl"],
                help="Material file referencing multiple textures",
                key="atlas_mtl_upload"
            )

        with col2:
            st.write("**Texture Files (Optional)**")
            atlas_texture_files = st.file_uploader(
                "Upload texture files",
                type=["png", "jpg", "jpeg", "bmp", "tga"],
                accept_multiple_files=True,
                help="Only needed if textures referenced by MTL aren't available",
                key="atlas_texture_upload"
            )
            if atlas_texture_files:
                st.success(f"✅ Uploaded {len(atlas_texture_files)} texture(s)")
                for tex_file in atlas_texture_files:
                    st.text(f"  • {tex_file.name}")

        if atlas_obj_file:
            st.success(f"✅ OBJ: {atlas_obj_file.name}")
        if atlas_mtl_file:
            st.success(f"✅ MTL: {atlas_mtl_file.name}")

        if atlas_obj_file and atlas_mtl_file:
            if st.button("🔧 Create Texture Atlas", type="primary", key="create_atlas_btn"):
                with st.spinner("Creating texture atlas..."):
                    atlas_result = create_texture_atlas(atlas_obj_file, atlas_mtl_file, atlas_texture_files)
                    if atlas_result:
                        st.session_state["atlas_result"] = atlas_result
                        st.rerun()

    # Display results and download options
    if st.session_state.get("atlas_result"):
        atlas_result = st.session_state["atlas_result"]

        st.subheader("📥 Download Generated Files")

        # Individual downloads in a row
        col1, col2, col3 = st.columns(3)

        with col1:
            if "obj_content" in atlas_result:
                st.download_button(
                    label="📄 Download Combined OBJ",
                    data=atlas_result["obj_content"],
                    file_name=atlas_result["obj_name"],
                    mime="application/octet-stream",
                    key="download_atlas_obj"
                )

        with col2:
            if "mtl_content" in atlas_result:
                st.download_button(
                    label="📄 Download Combined MTL",
                    data=atlas_result["mtl_content"],
                    file_name=atlas_result["mtl_name"],
                    mime="application/octet-stream",
                    key="download_atlas_mtl"
                )

        with col3:
            if "atlas_content" in atlas_result:
                st.download_button(
                    label="🖼️ Download Atlas Texture",
                    data=atlas_result["atlas_content"],
                    file_name=atlas_result["atlas_name"],
                    mime="image/png",
                    key="download_atlas_texture"
                )

        # Reset button
        if st.button("🔄 Create Another Atlas", key="reset_atlas"):
            if "atlas_result" in st.session_state:
                del st.session_state["atlas_result"]
            st.rerun()


def _collect_original_textures_from_mtl(mtl_path: str) -> List[Dict]:
    """Parse MTL and load referenced textures for preview."""
    textures: List[Dict] = []
    used_texture_names = set()

    try:
        mtl_dir = os.path.dirname(mtl_path)

        # First pass: collect all referenced texture names
        with open(mtl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("map_"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    tex_name = parts[-1]
                    used_texture_names.add(os.path.basename(tex_name))

        # Second pass: find all texture files and mark which are used
        texture_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tga', '.tiff', '.gif'}

        # Check MTL directory and parent directory for textures
        search_dirs = [mtl_dir, os.path.dirname(mtl_dir)]
        if os.path.exists(os.path.join(os.path.dirname(mtl_dir), "textures")):
            search_dirs.append(os.path.join(os.path.dirname(mtl_dir), "textures"))

        found_textures = set()
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for file in os.listdir(search_dir):
                if any(file.lower().endswith(ext) for ext in texture_extensions):
                    tex_path = os.path.join(search_dir, file)
                    if file not in found_textures and os.path.isfile(tex_path):
                        found_textures.add(file)
                        is_used = file in used_texture_names
                        info = load_texture_for_preview(tex_path, file)
                        if info:
                            info["used_in_atlas"] = is_used
                            textures.append(info)

        # Sort so used textures appear first
        textures.sort(key=lambda x: (not x.get("used_in_atlas", False), x["name"]))

    except Exception:
        pass
    return textures


def create_texture_atlas_from_zip(zip_file) -> Optional[Dict]:
    """
    Create texture atlas from a ZIP that contains:
      - root: *.obj, *.mtl
      - textures/: all images referenced by MTL
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP
            data = zip_file.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(temp_dir)

            # Prefer root-level OBJ/MTL; else fall back to first found recursively
            root_files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
            root_objs = [os.path.join(temp_dir, f) for f in root_files if f.lower().endswith(".obj")]
            root_mtls = [os.path.join(temp_dir, f) for f in root_files if f.lower().endswith(".mtl")]

            if root_objs and root_mtls:
                obj_path = root_objs[0]
                mtl_path = root_mtls[0]
            else:
                obj_candidates = glob.glob(os.path.join(temp_dir, "**/*.obj"), recursive=True)
                mtl_candidates = glob.glob(os.path.join(temp_dir, "**/*.mtl"), recursive=True)
                if not obj_candidates or not mtl_candidates:
                    st.error("ZIP must contain an OBJ and MTL (preferably at root).")
                    return None
                # Choose shallowest path
                obj_path = min(obj_candidates, key=lambda p: len(Path(p).parts))
                mtl_path = min(mtl_candidates, key=lambda p: len(Path(p).parts))

            # Prepare outputs
            output_obj = os.path.join(temp_dir, "atlas_combined.obj")
            output_mtl = os.path.join(temp_dir, "atlas_combined.mtl")
            output_atlas = os.path.join(temp_dir, "atlas.png")

            # Run make_atlas.py
            script_dir = Path(__file__).parent.absolute()
            atlas_script = script_dir / "make_atlas.py"
            cmd = [sys.executable, str(atlas_script), obj_path, mtl_path, output_obj, output_mtl, output_atlas]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=temp_dir)
            if result.returncode != 0:
                st.error(f"Atlas creation failed: {result.stderr}")
                return None

            atlas_data: Dict = {}

            if os.path.exists(output_obj):
                with open(output_obj, "rb") as f:
                    atlas_data["obj_content"] = f.read()
                    atlas_data["obj_name"] = "atlas_combined.obj"

            if os.path.exists(output_mtl):
                with open(output_mtl, "rb") as f:
                    atlas_data["mtl_content"] = f.read()
                    atlas_data["mtl_name"] = "atlas_combined.mtl"

            if os.path.exists(output_atlas):
                with open(output_atlas, "rb") as f:
                    atlas_data["atlas_content"] = f.read()
                    atlas_data["atlas_name"] = "atlas.png"
                atlas_data["atlas_image"] = Image.open(io.BytesIO(atlas_data["atlas_content"]))

            # Load original textures referenced by MTL for preview
            atlas_data["original_textures"] = _collect_original_textures_from_mtl(mtl_path)

            st.success("✅ Texture atlas created successfully!")
            return atlas_data

    except Exception as e:
        st.error(f"Error creating texture atlas from ZIP: {str(e)}")
        return None


# Main App
