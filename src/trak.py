"""
TRAK (Tracing with Random Projections) utilities for computing influence scores.

This module provides functions for:
- Extracting per-sample gradients from classification models
- Random projection for dimensionality reduction (Johnson-Lindenstrauss)
- Computing TRAK influence scores between training and target samples
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.func import vmap, grad, functional_call


def extract_gradients_from_images(
    model,
    images,
    labels,
    device='cuda',
    jl_dim=512,
    batch_size=32,
    progress_callback=None,
    project_dim=None,
    projection_seed=None,
    last_layer_only=True
):
    """
    Extract per-sample gradients from images using vectorized computation (vmap).

    Uses torch.func.vmap for efficient batched per-sample gradient extraction,
    which is significantly faster than the naive loop approach.

    Args:
        model: The classification model (transformers model)
        images: Tensor of images
        labels: Tensor of labels for the images
        device: Device to use for computation
        jl_dim: Dimension for JL projection (deprecated, use project_dim)
        batch_size: Batch size for processing
        progress_callback: Optional callback(processed, total) for progress updates
        project_dim: If set, project gradients to this dimension immediately (saves memory!)
        projection_seed: Random seed for projection matrix (for reproducibility/multiple projections)
        last_layer_only: If True, extract gradients only from classifier layer (default).
                        If False, extract from all model parameters (slower, higher memory).

    Returns:
        Tensor of gradients (N, D) or (N, project_dim) if projection is enabled

    Raises:
        ValueError: If model architecture is not supported or features cannot be extracted
    """
    # Route to appropriate implementation based on last_layer_only setting
    if not last_layer_only:
        return _extract_gradients_full_model(
            model, images, labels, device, batch_size,
            progress_callback, project_dim, projection_seed
        )
    # Convert images to tensors if needed
    if isinstance(images, list):
        images = torch.stack(images)

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_gradients = []
    total_samples = len(images)
    processed_samples = 0

    # Projection matrix - created lazily after we know gradient dimension
    projection_matrix = None
    grad_dim = None

    # Get classifier parameters
    model.eval()

    # Identify the classifier layer (different architectures use different names)
    # ResNet/torchvision: model.fc
    # Transformers/ViT: model.classifier
    if hasattr(model, 'fc'):
        classifier = model.fc
    elif hasattr(model, 'classifier'):
        classifier = model.classifier
    elif hasattr(model, 'head'):
        classifier = model.head
    else:
        raise ValueError("Could not identify classifier layer. Model must have 'fc', 'classifier', or 'head' attribute.")

    # Extract classifier parameters and buffers for functional_call
    params = {k: v.detach() for k, v in classifier.named_parameters()}
    buffers = {k: v for k, v in classifier.named_buffers()}

    # Define loss function for a single sample (used with vmap)
    def compute_loss(params, buffers, feature, label):
        """Compute cross-entropy loss for a single sample."""
        logits = functional_call(classifier, (params, buffers), (feature.unsqueeze(0),))
        return F.cross_entropy(logits, label.unsqueeze(0))

    # Create the per-sample gradient function using vmap
    # grad computes gradient w.r.t. first argument (params)
    # vmap vectorizes over the batch dimension of features and labels
    per_sample_grad_fn = vmap(
        grad(compute_loss),
        in_dims=(None, None, 0, 0)  # Don't vmap over params/buffers, vmap over features/labels
    )

    for batch_imgs, batch_labels in loader:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        # Extract features from the model (backbone is frozen, no gradients needed)
        with torch.no_grad():
            features = _extract_features(model, batch_imgs)

        # Compute per-sample gradients using vmap (vectorized, much faster!)
        # This computes gradients for ALL samples in the batch in one call
        per_sample_grads = per_sample_grad_fn(params, buffers, features, batch_labels)

        # per_sample_grads is a dict {param_name: tensor of shape (batch, *param_shape)}
        # Flatten and concatenate all parameters for each sample (batched operation)
        batch_size_actual = features.shape[0]

        # Stack all parameter gradients into one tensor per sample
        grad_tensors = []
        for key in params.keys():
            # Reshape from (batch, *param_shape) to (batch, -1)
            grad_tensors.append(per_sample_grads[key].view(batch_size_actual, -1))

        # Concatenate all parameters: (batch, total_grad_dim)
        batch_grads = torch.cat(grad_tensors, dim=1).cpu()  # Move to CPU immediately

        # Initialize projection matrix on first batch
        if project_dim is not None and projection_matrix is None:
            grad_dim = batch_grads.shape[1]
            if grad_dim > project_dim:
                if projection_seed is not None:
                    torch.manual_seed(projection_seed)
                projection_matrix = torch.randn(grad_dim, project_dim, dtype=torch.float32) / (project_dim ** 0.5)

        # Project entire batch at once on CPU (matrix multiplication is fast)
        if projection_matrix is not None:
            batch_projected = batch_grads @ projection_matrix  # (batch, project_dim)
            all_gradients.append(batch_projected)
        else:
            all_gradients.append(batch_grads)

        processed_samples += batch_size_actual
        if progress_callback:
            progress_callback(processed_samples, total_samples)

        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_gradients:
        return torch.empty(0)

    # Concatenate all batches (each is already a 2D tensor)
    return torch.cat(all_gradients, dim=0)


def _extract_features(model, batch_imgs):
    """
    Extract features from a model for a batch of images.

    Supports various architectures: ViT, ResNet, ConvNext, EfficientNet, etc.

    Args:
        model: The classification model
        batch_imgs: Batch of images tensor

    Returns:
        Features tensor of shape (batch, feature_dim)

    Raises:
        ValueError: If features cannot be extracted from the model
    """
    # Different feature extraction for different architectures
    if hasattr(model, 'vit'):
        # ViT models
        outputs = model.vit(batch_imgs)
        features = outputs.last_hidden_state[:, 0, :]  # CLS token
    elif hasattr(model, 'resnet'):
        # ResNet models
        outputs = model.resnet(batch_imgs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (batch, channels, H, W)
        features = last_hidden.mean(dim=[2, 3])  # Global average pooling
    elif hasattr(model, 'convnext'):
        # ConvNext models
        outputs = model.convnext(batch_imgs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        features = last_hidden.mean(dim=[2, 3])
    elif hasattr(model, 'efficientnet'):
        # EfficientNet models
        outputs = model.efficientnet(batch_imgs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        features = last_hidden.mean(dim=[2, 3])
    else:
        # Fallback: try to get pooler output or use the model directly
        if hasattr(model, 'base_model'):
            outputs = model.base_model(batch_imgs, output_hidden_states=True)
        else:
            outputs = model(batch_imgs, output_hidden_states=True)

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, 'hidden_states'):
            last_hidden = outputs.hidden_states[-1]
            if len(last_hidden.shape) == 4:  # Conv models
                features = last_hidden.mean(dim=[2, 3])
            else:  # Transformer models
                features = last_hidden[:, 0, :]
        else:
            raise ValueError("Could not extract features from model. This model architecture may not be supported.")

    return features


def project_gradients_random(gradients, target_dim=8192, seed=42, normalize=True):
    """
    Project high-dimensional gradients to lower dimension using random projection.
    Uses Johnson-Lindenstrauss lemma for dimensionality reduction.

    Args:
        gradients: Tensor of shape (N, D) where D is large
        target_dim: Target dimension (should be << D)
        seed: Random seed for reproducibility
        normalize: If True, L2-normalize gradients before projection (recommended)

    Returns:
        Projected gradients of shape (N, target_dim), normalized if normalize=True
    """
    N, D = gradients.shape

    # Normalize gradients before projection (L2 norm per sample)
    if normalize:
        grad_norm = gradients.norm(dim=1, keepdim=True)
        grad_norm[grad_norm == 0] = 1.0
        gradients = gradients / grad_norm

    if D <= target_dim:
        return gradients

    torch.manual_seed(seed)

    # Create random projection matrix: (D, target_dim)
    # Scale by 1/sqrt(target_dim) for variance preservation
    projection_matrix = torch.randn(D, target_dim, device=gradients.device) / (target_dim ** 0.5)

    # Project: (N, D) @ (D, target_dim) = (N, target_dim)
    projected = gradients @ projection_matrix

    return projected


def compute_trak_scores_simple(
    train_grads,
    target_grads,
    lambda_reg=0.001,
    device='cuda',
    project_dim=8192,
    num_projections=1,
    progress_callback=None
):
    """
    Simplified TRAK score computation for UI use.
    Computes: S = G_train @ (G_train.T @ G_train + lambda * I)^-1 @ G_target.T

    IMPORTANT: Gradients should already be projected to project_dim during extraction
    to avoid OOM. This function assumes gradients are already at the target dimension.

    Args:
        train_grads: Training gradients (N_train, D) - should be pre-projected and normalized
        target_grads: Target gradients (N_target, D) - should be pre-projected and normalized
        lambda_reg: Regularization parameter
        device: Device to use
        project_dim: Expected dimension (for logging only, projection happens during extraction)
        num_projections: Number of projections (for logging only, multiple projections require multiple extractions)
        progress_callback: Optional callback function(step_name, progress_pct) for progress updates

    Returns:
        Scores tensor (N_train, N_target)
    """
    N_train, D = train_grads.shape
    N_target = target_grads.shape[0]

    def report_progress(step_name, pct):
        if progress_callback:
            progress_callback(step_name, pct)

    # Gradients should already be projected and normalized
    report_progress(f"Processing gradients ({N_train} train, {N_target} target)", 10)

    # Move to device
    train_gpu = train_grads.to(device)
    target_gpu = target_grads.to(device)

    # Compute covariance matrix on GPU
    report_progress(f"Computing covariance matrix ({D}×{D})", 30)
    cov = train_gpu.T @ train_gpu

    # Add regularization and invert
    report_progress(f"Inverting covariance matrix ({D}×{D})", 50)
    cov += torch.eye(D, device=device) * lambda_reg
    inv_cov = torch.linalg.inv(cov)

    # Compute scores
    report_progress("Computing influence scores", 80)
    W_target = inv_cov @ target_gpu.T
    scores = (train_gpu @ W_target).cpu()

    report_progress("Complete", 100)
    return scores


def _extract_gradients_full_model(
    model,
    images,
    labels,
    device='cuda',
    batch_size=32,
    progress_callback=None,
    project_dim=None,
    projection_seed=None
):
    """
    Extract per-sample gradients from ALL model parameters (not just classifier).

    This is slower and more memory-intensive than last-layer-only extraction,
    but captures gradients from the entire model which can be useful for
    understanding feature learning dynamics.

    Args:
        model: The classification model
        images: Tensor of images
        labels: Tensor of labels
        device: Device to use
        batch_size: Batch size for processing
        progress_callback: Optional callback(processed, total)
        project_dim: If set, project gradients to this dimension
        projection_seed: Random seed for projection matrix

    Returns:
        Tensor of gradients (N, D) or (N, project_dim) if projection is enabled
    """
    import torch.nn as nn

    if isinstance(images, list):
        images = torch.stack(images)

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_gradients = []
    total_samples = len(images)
    processed_samples = 0

    # Projection matrix - created lazily
    projection_matrix = None

    # Get all trainable parameters
    model.eval()
    params_list = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    # Enable gradients for all parameters temporarily
    original_requires_grad = {}
    for name, p in model.named_parameters():
        original_requires_grad[name] = p.requires_grad
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    for batch_imgs, batch_labels in loader:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        batch_size_actual = batch_imgs.shape[0]

        batch_grads = []

        # Process each sample individually (slower but works for full model)
        for i in range(batch_size_actual):
            model.zero_grad()

            img = batch_imgs[i:i+1]
            label = batch_labels[i:i+1]

            outputs = model(img)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, label)
            loss.backward()

            # Collect gradients from all parameters
            grad_tensors = []
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad_tensors.append(p.grad.view(-1).clone())

            sample_grad = torch.cat(grad_tensors).cpu()
            batch_grads.append(sample_grad)

        # Stack batch gradients
        batch_grads = torch.stack(batch_grads)  # (batch, total_grad_dim)

        # Initialize projection matrix on first batch
        if project_dim is not None and projection_matrix is None:
            grad_dim = batch_grads.shape[1]
            if grad_dim > project_dim:
                if projection_seed is not None:
                    torch.manual_seed(projection_seed)
                projection_matrix = torch.randn(grad_dim, project_dim, dtype=torch.float32) / (project_dim ** 0.5)

        # Project if needed
        if projection_matrix is not None:
            batch_projected = batch_grads @ projection_matrix
            all_gradients.append(batch_projected)
        else:
            all_gradients.append(batch_grads)

        processed_samples += batch_size_actual
        if progress_callback:
            progress_callback(processed_samples, total_samples)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Restore original requires_grad settings
    for name, p in model.named_parameters():
        p.requires_grad = original_requires_grad[name]

    if not all_gradients:
        return torch.empty(0)

    return torch.cat(all_gradients, dim=0)
