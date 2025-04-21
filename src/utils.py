import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torchvision.transforms.functional as F
from PIL import Image
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from typing import List, Optional
import torch
ch = torch

with open('../data/imagenet1000_clsidx_to_labels.json', 'r+') as f:
    id_to_class = eval(f.read())
    class_to_id = {v: k for k, v in id_to_class.items()}

# ==========================
# Image Processing Utilities
# ==========================
def to_numpy(tensor: torch.Tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise ValueError("Input must be a PyTorch tensor or a NumPy array")

def resize(image: np.ndarray, size: tuple):
    image = to_numpy(image)
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    return Image.fromarray(image).resize(size)

def imshow(tensor: torch.Tensor, ax: plt.Axes = None):
    if ax:
        ax.imshow(to_numpy(tensor))
    else:
        plt.imshow(to_numpy(tensor))
    plt.show()

# For camera position projection
def project(vector: torch.Tensor, ball_norm: float = 4.5):
    norm_vector = ch.norm(vector)
    if norm_vector >= ball_norm:
        return (vector / norm_vector) * ball_norm
    else:
        return vector

# ML models take gamma corrected images as input
def gamma_correct(tensor: torch.Tensor, gamma: float = 1/2.2, alpha: float = 0.5):
    return ch.pow(ch.clip(tensor, 1e-5), gamma) * alpha

# Show topk prediction classes from logits
def show_pred(logits: torch.Tensor, topk: int = 3):
    # Compute Probas
    with ch.no_grad():
        probas = ch.nn.functional.softmax(logits, dim=-1)
        # Sort probabilities in descending order
        top_probas, top_labels = ch.topk(probas, k=topk, dim=-1)

    # Convert to lists for processing
    top_labels = top_labels[0].tolist()
    top_probas = top_probas[0].tolist()

    # Create formatted string of predictions with truncation
    str_ = "\n".join([
        f"{(id_to_class[top_labels[i]][:27] + '...') if len(id_to_class[top_labels[i]]) > 30 else id_to_class[top_labels[i]]}:{top_probas[i]:.2f}"
        for i in range(topk)
    ])
    return str_

def get_topk_classes(logits: torch.Tensor, topk: int = 3):
    """
    Get the topk classes (and their counts) from batched logits.
    logits: (B, 1000)
    Returns: List[Tuple[class: str, count: int]]
    """
    list_classifications = (ch.argmax(logits, dim=1)).flatten().tolist()
    return sorted([(get_target(k), v)  for k, v in Counter(list_classifications).items()], key=lambda x: x[1], reverse=True)[:topk]

# Shows image after rendering
def show_img(img: np.ndarray):
    ### Img Plot
    plt.figure(figsize=(6, 6))
    plt.tick_params(left = False, right = False , labelleft = False,
                    labelbottom = False, bottom = False)
    if img.ndim == 4:
        img = img[0]
    plt.imshow(img.detach().cpu())
    plt.show()

# ==========================
# Class and Label Utilities
# ==========================
def get_idx(class_label: str):
    class_idx = class_to_id.get(class_label)
    if class_idx is None:
        raise ValueError(f"Class label '{class_label}' not found in id_to_class.")
    return class_idx

def get_target_idx(class_label: str):
    return get_idx(class_label)

def get_target(class_idx: int):
    class_label = id_to_class.get(class_idx)
    if class_label is None:
        raise ValueError(f"Class index '{class_idx}' not found in id_to_class.")
    return class_label

def get_targets(batched_logits: torch.Tensor):
    """
    Get the labels from the logits.
    """
    return [get_target(class_idx.item()) for class_idx in batched_logits.argmax(1)]

def substr_target_indices(substr: str):
    return [(i, label) for i, label in id_to_class.items() if substr in label]

def get_target_label(class_label: str, device: str = "cuda"):
    # Directly index the dictionary to get the class index
    class_idx = get_idx(class_label)

    target = ch.zeros((1, 1000), device=device)
    target[0, class_idx] = 1  # Use advanced indexing for clarity
    return target

# ==========================
# Attention and Mask Utilities
# ==========================
def get_attention_mask(mesh: torch.Tensor, cameras: torch.Tensor, image_size: tuple, patch_size: int):
    """
    Generate an attention mask for tokens that intersect with the object's bounding box.
    Args:
        mesh: Mesh object containing vertices and faces.
        cameras: Pych3D camera instance.
        image_size: Tuple (H, W) of the rendered image.
        patch_size: The size of each token (patch) in pixels.

    Returns:
        A binary mask of shape (B, num_tokens) where 1 indicates relevant tokens.
    """
    verts = mesh.verts_packed()  # All vertices
    projected_verts = cameras.transform_points_screen(verts, image_size=image_size).to("cpu")  # Project to screen space
    min_coords = projected_verts.min(dim=0)[0]
    max_coords = projected_verts.max(dim=0)[0]

    # Calculate bounding box in image space
    bbox_x_min, bbox_y_min = min_coords[:2]
    bbox_x_max, bbox_y_max = max_coords[:2]

    # Map bounding box to token indices
    H, W = image_size
    num_patches_x = W // patch_size
    num_patches_y = H // patch_size

    x_indices = ch.arange(num_patches_x) * patch_size
    y_indices = ch.arange(num_patches_y) * patch_size
    x_mask = (x_indices >= bbox_x_min) & (x_indices < bbox_x_max)
    y_mask = (y_indices >= bbox_y_min) & (y_indices < bbox_y_max)

    # Create a token-level mask
    mask = ch.outer(y_mask, x_mask).flatten()  # Flatten to match token indices
    return mask

def reconstruct_image_from_mask(image: torch.Tensor, mask: torch.Tensor, patch_size: int, background_value: int = 0):
    """
    Reconstruct the image based on the attention mask.

    Args:
        image: Original image tensor of shape (B, C, H, W).
        mask: Attention mask of shape (B, num_tokens).
        patch_size: Size of each patch in the original image.
        background_value: Value to fill for masked-out areas (default: 0).

    Returns:
        Reconstructed image tensor of shape (B, C, H, W).
    """
    B, C, H, W = image.shape
    num_patches_x = W // patch_size
    num_patches_y = H // patch_size

    # Reshape the mask to match the patch grid
    mask = mask.view(B, num_patches_y, num_patches_x)  # (B, H/patch_size, W/patch_size)

    # Divide the image into patches
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # (B, C, num_patches_y, num_patches_x, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # (B, num_patches_y, num_patches_x, C, patch_size, patch_size)

    # Apply the mask to zero out irrelevant patches
    mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, num_patches_y, num_patches_x, 1, 1, 1)
    patches = patches * mask + background_value * (1 - mask)  # Replace masked patches with background_value

    # Reassemble the patches into the full image
    reconstructed = ch.zeros_like(image) + background_value  # Start with background
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            reconstructed[:, :, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patches[:, i, j, :, :, :]

    return reconstructed

# ==========================
# Camera Utilities
# ==========================
def look_at_rotation(camera_position: torch.Tensor, target_position: torch.Tensor, up_vector: torch.Tensor = None):
    """
    Compute the rotation matrix for a camera to look at a target.

    Args:
        camera_position: (B, 3) tensor of camera positions.
        target_position: (B, 3) tensor of target positions.
        up_vector: Optional (3,) tensor specifying the up vector. Defaults to [0, 0, 1].

    Returns:
        R: (B, 3, 3) tensor of rotation matrices.
    """
    if up_vector is None:
        up_vector = ch.tensor([0.0, 0.0, 1.0], device=camera_position.device, dtype=camera_position.dtype)

    z_axis = F.normalize(camera_position - target_position, dim=-1)  # Camera direction (view axis)
    x_axis = F.normalize(ch.cross(up_vector.expand_as(z_axis), z_axis), dim=-1)  # Right vector
    y_axis = ch.cross(z_axis, x_axis)  # Orthogonal up vector

    R = ch.stack([x_axis, y_axis, z_axis], dim=-1)  # Combine to form rotation matrix
    return R

# ==========================
# Logits Analysis Utilities
# ==========================
def analyze_logits(logits: torch.Tensor, target_class: str = None, max_length: int = 15):
    """
    Analyze the logits to find the most common class, its count, percentage,
    and the average probability of that class across all elements.

    Args:
        logits (ch.Tensor): A tensor of shape (batch_size, num_classes).

    Returns:
        tuple: (most_common_class, count, percentage, average_probability)
    """

    # Get the most likely classes for each batch
    predictions = logits.argmax(1)

    # Count occurrences of each class
    class_counts = Counter(predictions.tolist())

    if target_class is None:
        # Find the most common class and its count
        class_idx, count = class_counts.most_common(1)[0]
    else:
        # Find the target class and its count
        class_idx = get_target_idx(target_class)
        class_count = class_counts[class_idx]

    # Calculate the average probability of the most common class
    probs = ch.nn.functional.softmax(logits, dim=1)
    average_probability = probs[:, class_idx].mean().item()

    truncate_string = lambda s: s if len(s)<=max_length else s[:max_length] + "." * (min(3, len(s)-max_length))
    return truncate_string(get_target(class_idx)), class_count, average_probability

def render_ims(model: torch.nn.Module, ims: Optional[List[int]] = None, envmaps: Optional[List[int]] = None, initial: bool = False):
    """
    Render images from the model.

    Args:
        model: The model to render from.
        ims: The indices of images to render, None for all.
        envmaps: The indices of environment maps to render, None for all.
        initial: Whether to render the initial image.
    """
    if ims is None:
        ims = range(model.batch_size)

    # Number of images and max per row
    total_images = len(ims)
    images_per_row = 5

    # Calculate the number of rows needed
    num_rows = math.ceil(total_images / images_per_row)

    render_func = getattr(model, "render_initial" if initial else "render")

    # If no envmaps is provided, only render one environment map at random
    if envmaps is None:
        envmaps = np.random.randint(0, rendered_images.shape[1])

    # Generate the rendered images
    rendered_images = to_numpy(render_func())
    rendered_images = rendered_images[ims, envmaps, :, :, :]

    # Plot the images
    _, axes = plt.subplots(num_rows, images_per_row, figsize=(images_per_row * 4, num_rows * 4))

    # Flatten axes for easier indexing (handles edge cases for incomplete rows)
    axes = axes.flatten()

    for i in range(total_images):
        axes[i].imshow(rendered_images[i])
        axes[i].axis("off")

    # Turn off any remaining axes if total_images < len(axes)
    for j in range(total_images, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def analyze_logits_detailed(logits: torch.Tensor, target_class: str = None, top_n: int = 5):
    """
    Analyze logits to provide detailed class statistics.

    Args:
        logits (torch.Tensor): Tensor of logits with shape (batch_size, num_classes)
        top_n (int, optional): Number of top classes to return. Defaults to 5.
        target_idx (int, optional): Index of the target class to highlight.

    Returns:
        pandas.DataFrame: DataFrame with columns ["Class", "Count", "Avg Probability"]
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)

    top_indices = torch.argmax(logits, dim=1)
    class_indices, counts = torch.unique(top_indices.cpu(), return_counts=True)
    softmax_probs = torch.softmax(logits, dim=1)
    top_probs = softmax_probs[range(softmax_probs.size(0)), top_indices]

    stats = [
        (id_to_class[idx], count, top_probs[top_indices == idx].mean().item())
        for idx, count in zip(class_indices.tolist(), counts.tolist())
    ]

    # Sort by count and take the top N
    sorted_stats = sorted(stats, key=lambda x: x[1], reverse=True)[:top_n]

    # Convert to DataFrame for better display
    df = pd.DataFrame(sorted_stats, columns=["Class", "Count", "Avg Probability"])

    # Highlight the target class if specified
    if target_class is not None:
        df.loc[df['Class'] == target_class, 'Class'] = df.loc[df['Class'] == target_class, 'Class'].apply(lambda x: f'**{x}**')

    return df

# ==========================
# Visualization Utilities
# ==========================
def compute_spherical_coordinates(positions: np.ndarray):
    """
    Compute spherical coordinates (azimuth, elevation, norm) from camera positions.
    """
    azimuth = np.arctan2(positions[:, 1], positions[:, 0]) * 180 / np.pi
    norm = np.linalg.norm(positions, axis=1)
    elevation = np.arcsin(positions[:, 2] / norm) * 180 / np.pi
    return azimuth, elevation, norm


def visualize_positions_with_distributions(
    positions: np.ndarray,
    labels_correct: np.ndarray,
    title: str = None,
    fontsize: int = 18,
    return_stats: bool = False,
    mode: str = "all"  # "all", "3d", or "distributions"
):
    """
    Visualize 3D positions and distributions of azimuth, elevation, and norm.

    Parameters:
    positions : numpy array or list of tuples
        Array of (x, y, z) camera positions.
    labels_correct : numpy array or list
        Array of labels corresponding to each camera position. 1 for "True", 0 for "Wrong".
    title : str, optional
        Base title for the plots
    fontsize : int, optional
        Font size for plot text. Default is 18.
    return_stats : bool, optional
        Whether to return statistics.
    mode : str, optional
        "all" (default): show both 3D scatter and histograms in one figure.
        "3d": show only the 3D scatter plot.
        "distributions": show only the histograms.
    """

    def plot_max_normalized_hist(ax, data, bins, color, label):
        counts, bin_edges = np.histogram(data, bins=bins)
        if counts.max() > 0:
            counts = counts / counts.max()  # Normalize so max bar is 1
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]), color=color, alpha=0.5, label=label)

    # Convert to numpy arrays if not already
    positions = np.array(positions)
    labels_correct = np.array(labels_correct)

    azimuth, elevation, norm = compute_spherical_coordinates(positions)

    # Separate data based on labels
    azimuth_true, azimuth_wrong = azimuth[labels_correct == 1], azimuth[labels_correct == 0]
    elevation_true, elevation_wrong = elevation[labels_correct == 1], elevation[labels_correct == 0]
    norm_true, norm_wrong = norm[labels_correct == 1], norm[labels_correct == 0]
    positions_true, positions_wrong = positions[labels_correct == 1], positions[labels_correct == 0]

    if mode == "3d":
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(positions_true[:, 0], positions_true[:, 1], positions_true[:, 2],
                    c='blue', label='True label', alpha=0.5)
        ax1.scatter(positions_wrong[:, 0], positions_wrong[:, 1], positions_wrong[:, 2],
                    c='red', label='Wrong label', alpha=0.5)
        ax1.set_xlabel('X', fontsize=fontsize)
        ax1.set_ylabel('Y', fontsize=fontsize)
        ax1.set_zlabel('Z', fontsize=fontsize)
        ax1.set_title("3D Positions", fontsize=fontsize)
        # Option 1: Legend on the right side
        ax1.legend(
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            fontsize=fontsize,
            frameon=False
        )
        # Option 2: Legend just under the title (uncomment to use)
        # ax1.legend(
        #     loc='upper center',
        #     bbox_to_anchor=(0.5, 0.88),
        #     fontsize=fontsize,
        #     frameon=False
        # )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    elif mode == "distributions":
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Azimuth Histogram
        plot_max_normalized_hist(axes[0], azimuth_true, bins=20, color='blue', label='True')
        plot_max_normalized_hist(axes[0], azimuth_wrong, bins=20, color='red', label='Wrong')
        axes[0].set_xlabel('Azimuth (degrees)', fontsize=fontsize)
        axes[0].set_ylabel('Relative Frequency\n(max=1 per true/wrong group)', fontsize=fontsize-3)
        axes[0].set_title("Azimuth", fontsize=fontsize)

        # Elevation Histogram
        plot_max_normalized_hist(axes[1], elevation_true, bins=20, color='blue', label='True')
        plot_max_normalized_hist(axes[1], elevation_wrong, bins=20, color='red', label='Wrong')
        axes[1].set_xlabel('Elevation (degrees)', fontsize=fontsize)
        axes[1].set_title("Elevation", fontsize=fontsize)
        # Norm/Distance Histogram
        plot_max_normalized_hist(axes[2], norm_true, bins=20, color='blue', label='True')
        plot_max_normalized_hist(axes[2], norm_wrong, bins=20, color='red', label='Wrong')
        axes[2].set_xlabel('Distance', fontsize=fontsize)
        axes[2].set_title("Distance", fontsize=fontsize)
        fig.suptitle(
            title or "Analysis of 3D Spherical Distribution of Model Classification",
            fontsize=fontsize + 2,
            y=1.05
        )
        fig.legend(
            ['True', 'Wrong'],
            loc='upper center',
            fontsize=fontsize,
            ncol=2,
            bbox_to_anchor=(0.5, 1.01),
            frameon=False
        )
        # Add total correct vs incorrect below the middle plot
        axes[1].text(
            0.5, -0.28,
            f"Total correct: {len(azimuth_true)}   Total wrong: {len(azimuth_wrong)}",
            ha='center', va='top', fontsize=fontsize - 2, transform=axes[1].transAxes
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    else:  # mode == "all"
        fig = plt.figure(figsize=(24, 6))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1, 1, 1])
        # 3D Scatter Plot
        ax1 = fig.add_subplot(gs[0], projection='3d')
        ax1.scatter(positions_true[:, 0], positions_true[:, 1], positions_true[:, 2],
                    c='blue', label='True label', alpha=0.5)
        ax1.scatter(positions_wrong[:, 0], positions_wrong[:, 1], positions_wrong[:, 2],
                    c='red', label='Wrong label', alpha=0.5)
        ax1.set_xlabel('X', fontsize=fontsize)
        ax1.set_ylabel('Y', fontsize=fontsize)
        ax1.set_zlabel('Z', fontsize=fontsize)
        ax1.set_title("3D Positions", fontsize=fontsize)
        # Azimuth Histogram
        ax2 = fig.add_subplot(gs[1])
        plot_max_normalized_hist(ax2, azimuth_true, bins=20, color='blue', label='True')
        plot_max_normalized_hist(ax2, azimuth_wrong, bins=20, color='red', label='Wrong')
        ax2.set_xlabel('Azimuth (degrees)', fontsize=fontsize)
        ax2.set_ylabel('Max-normalized Frequency\n(per true/wrong group)', fontsize=fontsize-3)
        ax2.set_title("Azimuth", fontsize=fontsize)
        # Add the second line below the y-label, with smaller font
        ax2.text(
            -0.25, 1.02,  # x, y in axes fraction coordinates (tweak as needed)
            "Each group is max-normalized\n(Tallest bar = 1 per true/wrong group)",
            ha='left', va='bottom', fontsize=fontsize-4, transform=ax2.transAxes
        )
        # Elevation Histogram
        ax3 = fig.add_subplot(gs[2])
        plot_max_normalized_hist(ax3, elevation_true, bins=20, color='blue', label='True')
        plot_max_normalized_hist(ax3, elevation_wrong, bins=20, color='red', label='Wrong')
        ax3.set_xlabel('Elevation (degrees)', fontsize=fontsize)
        ax3.set_title("Elevation", fontsize=fontsize)
        # Norm/Distance Histogram
        ax4 = fig.add_subplot(gs[3])
        plot_max_normalized_hist(ax4, norm_true, bins=20, color='blue', label='True')
        plot_max_normalized_hist(ax4, norm_wrong, bins=20, color='red', label='Wrong')
        ax4.set_xlabel('Distance', fontsize=fontsize)
        ax4.set_title("Distance", fontsize=fontsize)
        fig.suptitle(
            title or "Analysis of 3D Spherical Distribution of Model Classification",
            fontsize=fontsize + 2,
            y=1.05
        )
        handles, labels_ = ax1.get_legend_handles_labels()
        fig.legend(
            handles, labels_,
            loc='upper center',
            fontsize=fontsize,
            ncol=2,
            bbox_to_anchor=(0.5, 1.01),
            frameon=False
        )
        ax3.text(
            0.5, -0.18,
            "Each group is max-normalized\n(Tallest bar = 1 per true/wrong group)",
            ha='center', va='top', fontsize=fontsize - 4, transform=ax3.transAxes
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    if return_stats:
        return {
            'true_stats': {
                'azimuth_mean': np.mean(azimuth_true),
                'azimuth_std': np.std(azimuth_true),
                'elevation_mean': np.mean(elevation_true),
                'elevation_std': np.std(elevation_true),
                'norm_mean': np.mean(norm_true),
                'norm_std': np.std(norm_true)
            },
            'wrong_stats': {
                'azimuth_mean': np.mean(azimuth_wrong),
                'azimuth_std': np.std(azimuth_wrong),
                'elevation_mean': np.mean(elevation_wrong),
                'elevation_std': np.std(elevation_wrong),
                'norm_mean': np.mean(norm_wrong),
                'norm_std': np.std(norm_wrong)
            }
        }


def visualize_positions_polar(positions: np.ndarray, labels_correct: np.ndarray, title: str = None):
    """
    Visualize camera positions with azimuth-elevation heatmaps in a circular (polar) layout.

    Parameters:
    positions : numpy array or list of tuples
        Array of (x, y, z) positions
    labels_correct : numpy array or list
        Array of labels corresponding to each position. 1 for "correct", 0 for "incorrect".
    title : str, optional
        Custom title for the plot
    """
    # Convert to numpy arrays if not already
    positions = np.array(positions)
    labels_correct = np.array(labels_correct)

    # Compute azimuth (theta) and elevation (r)
    azimuth = np.arctan2(positions[:, 1], positions[:, 0])  # Radians
    elevation = np.arcsin(positions[:, 2] / np.linalg.norm(positions, axis=1))  # Radians

    # Normalize elevation to [0, 1] for radial plotting
    elevation_normalized = (elevation - elevation.min()) / (elevation.max() - elevation.min())

    # Combine data into a weighted histogram
    weights = np.where(labels_correct == 1, 1, -1)  # +1 for correct, -1 for misclassified
    heatmap, theta_edges, r_edges = np.histogram2d(
        azimuth, elevation_normalized, bins=(36, 18), range=[[-np.pi, np.pi], [0, 1]], weights=weights
    )
    counts, _, _ = np.histogram2d(
        azimuth, elevation_normalized, bins=(36, 18), range=[[-np.pi, np.pi], [0, 1]]
    )

    # Mask zero density areas
    heatmap_masked = np.ma.masked_where(counts == 0, heatmap)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

    # Calculate heatmap with safety checks
    heatmap = heatmap_masked.T
    heatmap_masked = np.ma.masked_where(np.isnan(heatmap), heatmap)

    valid_data = heatmap_masked.compressed()
    if len(valid_data) == 0:
        print("No valid data points to display")
        return

    data_min = valid_data.min()
    data_max = valid_data.max()
    abs_max = max(abs(data_min), abs(data_max))

    if data_min < 0 < data_max:
        colors = [(0.8, 0, 0), (1, 1, 1), (0, 0, 0.8)]  # Red -> White -> Blue
        cmap = LinearSegmentedColormap.from_list('RedWhiteBlue', colors)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        cbar_label = 'Point Density\n(Red: Misclassified, Blue: Well Classified)'
    elif data_max <= 0:
        cmap = LinearSegmentedColormap.from_list('Reds', ['#AA0000', '#FFFFFF'])
        norm = Normalize(vmin=-abs_max, vmax=0)
        cbar_label = 'Point Density\n(Red: Misclassified)'
    else:
        cmap = LinearSegmentedColormap.from_list('Blues', ['#FFFFFF', '#0000AA'])
        norm = Normalize(vmin=0, vmax=abs_max)
        cbar_label = 'Point Density\n(Blue: Well Classified)'

    pcm = ax.pcolormesh(
        theta_edges, r_edges, heatmap_masked,
        cmap=cmap,
        norm=norm,
        shading='auto'
    )

    cbar = plt.colorbar(pcm, ax=ax, pad=0.1)
    cbar.set_label(cbar_label)

    # Add radial (elevation) ticks
    elevation_ticks = np.linspace(0, 1, 5)
    elevation_labels = np.linspace(elevation.min(), elevation.max(), len(elevation_ticks))
    ax.set_yticks(elevation_ticks)
    ax.set_yticklabels([f"{np.degrees(e):.1f}°" for e in elevation_labels])

    ax.set_title(title or 'Azimuth-Elevation Heatmap', va='bottom')
    ax.set_theta_zero_location("N")  # Set 0° at the top
    ax.set_theta_direction(-1)  # Clockwise azimuth
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def create_logits_comparison_table(all_results: dict, target_class: str = None):
    """
    Create an HTML table comparing initial and final logits analysis for different result types.

    Args:
        all_results: Dictionary with keys as result types and values as dictionaries containing
                    'initial_logits' and 'final_logits'.
        target_class: Optional target class to highlight in the analysis.

    Returns:
        HTML string containing the styled comparison table.
    """
    # CSS styling for column widths and consistent heights
    style = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
            vertical-align: top; /* Align content to the top */
        }
        th {
            background-color: #f2f2f2;
        }
        td:first-child, th:first-child {
            width: 15%; /* Narrower first column */
            text-align: left; /* Align text to the left for readability */
        }
        td, th {
            width: 42.5%; /* Even distribution for the other columns */
        }
        .pandas-table {
            height: 200px; /* Set a fixed height for all tables */
            overflow: auto; /* Enable scrolling if content exceeds the height */
        }
    </style>
    """

    html_table = "<table>"
    html_table += """
    <tr>
        <th></th>
        <th style="text-align: center;">Initial</th>
        <th style="text-align: center;">Final</th>
    </tr>
    """

    for type_, results in all_results.items():
        initial_analysis = analyze_logits_detailed(
            np.concatenate(results["initial_logits"]).reshape(-1, 1000), target_class
        )
        final_analysis = analyze_logits_detailed(
            np.concatenate(results["final_logits"]).reshape(-1, 1000), target_class
        )

        html_table += f"""
        <tr>
            <td>{type_.capitalize()}</td>
            <td class="pandas-table">{initial_analysis.to_html(index=False, border=0)}</td>
            <td class="pandas-table">{final_analysis.to_html(index=False, border=0)}</td>
        </tr>
        """

    html_table += "</table>"

    # Return the styled HTML
    from IPython.display import HTML, display
    display(HTML(style + html_table))


def display_rendered_images(robust_analyzer, results, run_index=0, env_index=0, image_indices=None,
                           max_cols=2, figsize=None, raster_settings={}):
    """
    Display rendered images with camera position information and prediction correctness.

    Args:
        model: The 3D model to render with
        results: Dictionary containing results from RobustnessAnalyzer
        run_index: Index of the run to display results from
        env_index: Index of the environment map to use
        image_indices: Indices of images to display (defaults to all)
        max_cols: Maximum number of columns in the grid
        figsize: Figure size (width, height) tuple
        raster_settings: Raster settings for rendering
    """
    model = robust_analyzer.model

    # Ensure model has the updated parameters
    model.update_scene_params(results["final_scene_params"][run_index])

    # Render the images
    render_ims = model.render(with_grad=False, raster_settings=raster_settings)

    # Get predictions for the rendered images
    with torch.no_grad():
        logits = model()

    # Define image indices if not provided
    if image_indices is None:
        image_indices = range(robust_analyzer.batch_size)

    # Get target class index
    target_class_idx = get_idx(robust_analyzer.target_class) if hasattr(robust_analyzer, 'target_class') else None

    # Calculate grid dimensions
    n_images = len(image_indices)
    n_rows = math.ceil(n_images / max_cols)

    # Create default figsize if not provided
    if figsize is None:
        figsize = (max_cols * 7, n_rows * 5)

    # Create the figure and axes
    fig, axes = plt.subplots(n_rows, max_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    # Display each image
    for ax, im_idx in zip(axes, image_indices):
        # Get camera position metrics
        camera_pos = results["final_scene_params"][run_index]["camera"][im_idx][None]
        azimuth, elevation, distance = compute_spherical_coordinates(camera_pos)

        # Check if prediction is correct
        pred_idx = logits[im_idx, env_index].argmax().item()
        is_correct = (pred_idx == target_class_idx) if target_class_idx is not None else None
        pred_class = get_target(pred_idx)

        # Create title with position and prediction info
        position_info = " ".join([
            f"{metric}={value.item():.1f}"
            for value, metric in zip([azimuth, elevation, distance], ["azimuth", "elevation", "distance"])
        ])

        if is_correct is not None:
            # Add color codes
            prediction_info = f"\n{pred_class[:20]}... ({'✓' if is_correct else '✗'})"
        else:
            prediction_info = f"\n{pred_class[:20]}..."

        title = position_info + prediction_info

        # Display image with title
        ax.imshow(to_numpy(render_ims[im_idx][env_index]))
        ax.axis("off")
        ax.set_title(title)

    # Hide unused axes
    for ax in axes[len(image_indices):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

