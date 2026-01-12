# -*- coding: utf-8 -*-
"""
Visualization module for 3D Adversarial Robustness Analyzer.

Contains functions for:
- Polar heatmap plots
- Image rendering and compositing
- Result visualization and carousels
- Download package generation
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import traceback
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from plotly.subplots import make_subplots

import utils


# ==========================
# Helper Functions
# ==========================
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


def _pred_top1(logits: torch.Tensor) -> np.ndarray:
    """Return argmax per row."""
    return torch.argmax(logits, dim=1).cpu().numpy()


def _softmax_max_probs(logits: torch.Tensor) -> np.ndarray:
    """Return max softmax probabilities per row."""
    return torch.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()


def _now_iso() -> str:
    """Return current timestamp in ISO format."""
    return datetime.now().isoformat()


def _get_idx_safe(target: str) -> int:
    """Safely get class index from target string."""
    from utils import get_idx
    return get_idx(target)


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
        from utils import id_to_class
        return id_to_class.get(class_idx, f"class_{class_idx}")
    else:
        return f"class_{class_idx}"


def create_interactive_polar_plot(
    camera_positions: np.ndarray,
    labels_correct: np.ndarray,
    logits: torch.Tensor,
    target_class: str,
    topk_value: int,
    selected_indices: Optional[Sequence[int]] = None,
    current_highlighted_index: Optional[int] = None,
    num_classes: int = 1000,
    custom_labels: Optional[Dict[int, str]] = None,
) -> go.Figure:
    azimuth, elevation, _ = utils.compute_spherical_coordinates(camera_positions)
    azimuth_rad = np.radians(azimuth)

    preds_top1 = _pred_top1(logits)
    pred_probs = _softmax_max_probs(logits)

    # Hover text
    hover_text: List[str] = []
    for i in range(len(camera_positions)):
        pred_class = get_class_label(int(preds_top1[i]), num_classes, custom_labels)
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

    title_text = f"Azimuth-Elevation"
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
        # Get actual number of environments from envmap_paths
        envmap_paths = results.get("envmap_paths", [])
        total_envmaps = _get_num_envmaps(envmap_paths)

        # Load stored logits for later use
        stored_logits = torch.stack(results["final_logits"])

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

            # Get config to determine num_classes
            config = st.session_state.get("config", {})
            num_classes = config.get("num_classes", 1000)
            custom_labels = config.get("custom_labels")
            pred_class = get_class_label(pred_class_idx, num_classes, custom_labels)

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
        num_classes=config.get("num_classes", 1000),
        custom_labels=config.get("custom_labels"),
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
        # Silent fallback to matplotlib when Plotly fails
        heatmap_img = None
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
                    c="lightcoral", s=30, alpha=0.6, label=f"Incorrect",
                )
            if correct_mask.any():
                ax.scatter(
                    azimuth_rad[correct_mask], elevation[correct_mask],
                    c="lightblue", s=30, alpha=0.6, label=f"Correct",
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
                "Heatmap unavailable\n(Plotting libraries unavailable)\n\n"
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
            num_classes=config.get("num_classes", 1000),
            custom_labels=config.get("custom_labels"),
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

    if current.get("env_idx") is not None and "results" in st.session_state:
        results = st.session_state["results"]
        envmap_paths = results.get("envmap_paths", [])
        if envmap_paths and current["env_idx"] < len(envmap_paths):
            env_name = Path(envmap_paths[current["env_idx"]]).name
            st.caption(f"🌍 **Environment:** {env_name}")

    d1, d2, _, d4, _ = st.columns([1, 2, 0.5, 2, 1])

    include_heatmap = st.session_state.get("include_heatmap_global", True)

    with d2:
        if ("results" in st.session_state and "config" in st.session_state):
            try:
                results = st.session_state["results"]
                config = st.session_state["config"]

                logits = torch.stack(results["final_logits"])
                cameras = torch.stack([x["camera"] for x in results["final_scene_params"]])

                # Get envmap_paths from results
                envmap_paths = results.get("envmap_paths", [])

                logits2d, cams2d = _expand_for_envmaps(logits, cameras, envmap_paths)

                topk = st.session_state.get("topk_value", 1)
                try:
                    labels_correct = utils.get_labels_correct(logits2d, config["target_class"], topk=topk)
                except Exception as e:
                    st.error(f"Error getting labels: {str(e)}")
                    # Create a fallback labels_correct array (assume all incorrect)
                    labels_correct = torch.zeros(len(logits2d), dtype=torch.bool)

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

                        # Get envmap_paths from results
                        envmap_paths = results.get("envmap_paths", [])

                        logits2d, cams2d = _expand_for_envmaps(logits, cameras, envmap_paths)

                        topk = st.session_state.get("topk_value", 1)
                        try:
                            labels_correct = utils.get_labels_correct(logits2d, config["target_class"], topk=topk)
                        except Exception as e:
                            st.error(f"Error getting labels: {str(e)}")
                            # Create a fallback labels_correct array (assume all incorrect)
                            labels_correct = torch.zeros(len(logits2d), dtype=torch.bool)

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
                                        "prediction": image_info.get("prediction", "N/A"),
                                        "confidence": image_info.get("confidence", 0),
                                        "azimuth": image_info.get("azimuth", 0),
                                        "elevation": image_info.get("elevation", 0),
                                        "distance": image_info.get("distance", 0),
                                        "env_idx": image_info.get("env_idx", 0),
                                    }

                        zbuf.seek(0)
                        st.session_state[cache_key] = zbuf.getvalue()

                except Exception as e:
                    st.error(f"❌ Error preparing download: {str(e)}")

            if cache_key in st.session_state:
                st.download_button(
                    label="📦 Download All Images (ZIP)",
                    data=st.session_state[cache_key],
                    file_name="robustness_positions_batch.zip",
                    mime="application/zip",
                    key=f"dl_btn_batch_{cache_key}",
                    use_container_width=True
                )
        else:
            st.button("📦 Download All Images (ZIP)", disabled=True, help="Analysis data not available", use_container_width=True)

    st.markdown("---")
    col_img, col_info = st.columns([2, 1])

    with col_img:
        render_img = Image.fromarray(current["image"])
        st.image(render_img, use_container_width=True)

    with col_info:
        st.write(""); st.write("")
        st.write("**Prediction Info:**")
        st.write(f"🎯 **Class:** {current['prediction']}")
        st.write(f"📊 **Confidence:** {current['confidence']:.3f}")
        if current.get("target_class_ranking"):
            st.write(f"🏆 **Target Ranking:** #{current['target_class_ranking']}")
        if current.get("target_class_confidence"):
            st.write(f"🎪 **Target Confidence:** {current['target_class_confidence']:.3f}")

        st.write(""); st.write("")
        st.write("**Camera Position:**")
        st.write(f"🧭 **Azimuth:** {current['azimuth']:.2f}°")
        st.write(f"📈 **Elevation:** {current['elevation']:.2f}°")
        st.write(f"📏 **Distance:** {current['distance']:.3f}")

        st.write(""); st.write("")
        st.write("**Metadata:**")
        st.write(f"📍 **Position Index:** {current['position_idx']}")
        st.write(f"🏃 **Run:** {current['run_idx']}")
        st.write(f"🖼️ **Batch:** {current['batch_idx']}")

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        include_heatmap = st.checkbox(
            "Include Heatmap Overlay",
            value=st.session_state.get("include_heatmap_global", True),
            help="Show heatmap overlay on top of rendered images"
        )
        st.session_state["include_heatmap_global"] = include_heatmap
    with col_b:
        st.write("")
        if st.button("🔄 Rerender All", help="Re-render all selected images"):
            st.session_state["rendered_images"] = []
            st.session_state["image_cache_key"] = None
            st.rerun()
    with col_c:
        st.write("")
        if st.button("Clear Selection", help="Clear selected positions and start over"):
            st.session_state["selected_positions"] = []
            st.session_state["rendered_images"] = []
            st.session_state["trigger_render"] = False
            st.rerun()

def visualize_results(results: Dict, config: Dict):
    st.header("📊 Results Visualization")

    try:
        logits_stack = torch.stack(results["final_logits"])
        cams_stack = torch.stack([x["camera"] for x in results["final_scene_params"]])

        # Get environment information from results
        envmap_paths = results.get("envmap_paths", [])
        n_envmaps = _get_num_envmaps(envmap_paths)

        # Debug: Show shapes
        if st.checkbox("🐛 Show Debug Info", value=False, key="debug_shapes"):
            st.write(f"**logits_stack shape:** {logits_stack.shape}")
            st.write(f"**cams_stack shape:** {cams_stack.shape}")
            st.write(f"**Number of environments:** {n_envmaps}")
            st.write(f"**envmap_paths:** {envmap_paths}")

        st.subheader("🌍 Environment Selection")

        if n_envmaps > 1:
            env_options = ["All Environments"] + [f"Environment {i+1}: {Path(envmap_paths[i]).name}" for i in range(n_envmaps)]
            selected_env = st.selectbox(
                "Select Environment to Visualize",
                options=range(len(env_options)),
                format_func=lambda x: env_options[x],
                index=0,
                help=f"Choose to view all environments together or filter by specific environment (Total: {n_envmaps})"
            )

            if selected_env == 0:  # All environments
                st.info(f"📊 Showing results from **all {n_envmaps} environments** combined")
                logits2d, cams2d = _expand_for_envmaps(logits_stack, cams_stack, envmap_paths)
                env_filter_idx = None
            else:  # Specific environment
                env_idx = selected_env - 1
                st.success(f"✅ Filtered to **{Path(envmap_paths[env_idx]).name}**")

                # First expand cameras for all environments, then filter
                # Expand cameras to match environment dimension
                cams_expanded = cams_stack.unsqueeze(2).expand(-1, -1, n_envmaps, -1)

                # Extract only the selected environment's data
                if len(logits_stack.shape) == 4:  # [runs, batch, env, classes]
                    filtered_logits = logits_stack[:, :, env_idx, :]  # [runs, batch, classes]
                    filtered_cams = cams_expanded[:, :, env_idx, :]  # [runs, batch, 3]
                else:
                    filtered_logits = logits_stack
                    filtered_cams = cams_stack

                # Reshape both to 2D
                num_classes = filtered_logits.shape[-1]  # Infer from shape
                logits2d = filtered_logits.reshape(-1, num_classes)  # [runs*batch, classes]
                cams2d = filtered_cams.reshape(-1, 3)  # [runs*batch, 3]
                env_filter_idx = env_idx
        else:
            st.info(f"📊 Using **single environment**: {Path(envmap_paths[0]).name if envmap_paths else 'No environment map'}")
            logits2d, cams2d = _expand_for_envmaps(logits_stack, cams_stack, envmap_paths)
            env_filter_idx = None

        # Store selected environment in session state for rendering
        st.session_state["selected_env_idx"] = env_filter_idx if env_filter_idx is not None else 0

        # Debug: Show processed shapes
        if st.session_state.get("debug_shapes", False):
            st.write(f"**After processing:**")
            st.write(f"- logits2d shape: {logits2d.shape}")
            st.write(f"- cams2d shape: {cams2d.shape}")
            st.write(f"- env_filter_idx: {env_filter_idx}")

        total_positions = len(logits2d)
        assert len(cams2d) == total_positions, f"Mismatch: {len(cams2d)} camera positions vs {len(logits2d)} logit entries"

        with st.expander("🏷️ Most Predicted Classes (Top 5)", expanded=False):
            preds_top1 = _pred_top1(logits2d)
            values, counts = np.unique(preds_top1, return_counts=True)
            order = np.argsort(-counts)
            top_n = min(5, len(order))

            # Get num_classes from config
            config = st.session_state.get("config", {})
            num_classes = config.get("num_classes", 1000)
            custom_labels = config.get("custom_labels")
            for rank in range(top_n):
                cls_id = int(values[order[rank]])
                cnt = int(counts[order[rank]])
                pct = (cnt / total_positions) * 100.0
                label = get_class_label(cls_id, num_classes, custom_labels)
                # Only truncate ImageNet labels (they're long), show full generic class names
                display_label = label[:30] + "..." if num_classes == 1000 and len(label) > 30 else label
                st.write(f"{rank+1}. {display_label} — {cnt} ({pct:.1f}%)")

        with st.expander("📊 Environment Statistics", expanded=False):
            # Show environment breakdown if multiple environments
            if n_envmaps > 1 and env_filter_idx is None:  # Only when viewing all environments
                st.subheader("📈 Results by Environment")

                # Split results by environment
                num_classes = logits_stack.shape[-1]  # Infer from shape
                logits_full = logits_stack.reshape(-1, n_envmaps, num_classes)

                env_stats = []
                topk = st.session_state.get("topk_value", 1)
                for i in range(n_envmaps):
                    env_logits = logits_full[:, i, :]
                    env_logits_flat = env_logits.reshape(-1, num_classes)

                    labels_correct_env = utils.get_labels_correct(env_logits_flat, config["target_class"], topk=topk)
                    accuracy = (labels_correct_env.sum() / len(labels_correct_env) * 100).item()

                    env_name = Path(envmap_paths[i]).name
                    env_stats.append({
                        "Environment": env_name,
                        "Accuracy": f"{accuracy:.1f}%",
                        "Correct": f"{int(labels_correct_env.sum())}/{len(labels_correct_env)}"
                    })

                import pandas as pd
                st.dataframe(pd.DataFrame(env_stats), use_container_width=True)
            else:
                st.info("ℹ️ Switch to 'All Environments' above to see per-environment statistics")

        st.subheader("🎯 Camera Position")
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
                num_classes=config.get("num_classes", 1000),
                custom_labels=config.get("custom_labels"),
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
            st.subheader("🎯 Render Selection")
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
                    # Get num_classes from config
                    config = st.session_state.get("config", {})
                    num_classes = config.get("num_classes", 1000)
                    custom_labels = config.get("custom_labels")
                    for i, pos_idx in enumerate(selected_indices):
                        conf = target_probs[pos_idx]
                        pred_idx = int(torch.argmax(logits2d[pos_idx]).item())
                        pred_class = get_class_label(pred_idx, num_classes, custom_labels)
                        status = "✅" if pred_idx == target_idx else "❌"
                        # Only truncate ImageNet labels (they're long), show full generic class names
                        display_class = pred_class[:25] + "..." if num_classes == 1000 and len(pred_class) > 25 else pred_class
                        st.write(f"{i+1}. **Position {pos_idx}:** {status} {conf:.3f} confidence → {display_class}")

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
                # Get selected environment index
                selected_env_idx = st.session_state.get("selected_env_idx", 0)
                cache_key = f"{sorted(valid_positions)}_{selected_env_idx}_{id(robust_analyzer)}"

                if st.session_state.get("image_cache_key") != cache_key:
                    with st.status("🎨 Rendering selected images...", expanded=True) as status:
                        # Pass environment index to render function
                        rendered_images = []
                        progress_bar_render = st.progress(0)
                        for i, pos_idx in enumerate(valid_positions):
                            progress_bar_render.progress((i + 1) / len(valid_positions))
                            image_info = render_image_at_position(
                                robust_analyzer,
                                results,
                                pos_idx,
                                envmap_idx=selected_env_idx
                            )
                            if image_info:
                                rendered_images.append(image_info)

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
                title=f"Azimuth-Elevation Heatmap",
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
                title=f"Analysis of 3D Spherical Distribution of Model Classification",
                mode="distributions",
                show_distance=True
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
