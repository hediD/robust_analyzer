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

# Try to import fast_jl for efficient CUDA-based projection
# Falls back to naive implementation if not available
try:
    import fast_jl
    FAST_JL_AVAILABLE = True
except ImportError:
    FAST_JL_AVAILABLE = False


class Projector:
    """
    JL Random Projector for TRAK gradient projection.

    Supports two backends:
    - fast_jl: CUDA-optimized Rademacher (±1) projection, never materializes matrix
    - naive: Gaussian projection, materializes full D×k matrix (uses GPU if available)

    Initialize once with grad_dim (or from model), reuse for all projections.
    """

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int = 42,
        device: str = 'cuda',
        use_fast_jl: bool = True,
        batch_size: int = 1024,
        verbose: bool = True
    ):
        """
        Args:
            grad_dim: Input gradient dimension (number of parameters)
            proj_dim: Output projection dimension
            seed: Random seed for reproducibility
            device: Device for computation
            use_fast_jl: If True, use fast_jl when available (recommended)
            batch_size: Batch size for projection (to avoid OOM on large inputs)
            verbose: If True, print backend info on first projection
        """
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        self._logged_backend = False

        # Determine backend
        self.use_fast_jl = (
            use_fast_jl and
            FAST_JL_AVAILABLE and
            torch.cuda.is_available()
        )

        # For naive backend: lazily create projection matrix
        self._projection_matrix = None
        self._projection_matrix_gpu = None  # Cache GPU version

        # For fast_jl: get SM count once
        if self.use_fast_jl:
            self._num_sms = self._get_num_sms()

    @classmethod
    def from_model(cls, model, proj_dim: int, last_layer_only: bool = True, **kwargs):
        """
        Create projector by inferring grad_dim from model parameters.

        Args:
            model: PyTorch model
            proj_dim: Target projection dimension
            last_layer_only: If True, only count classifier parameters
            **kwargs: Additional args passed to __init__
        """
        if last_layer_only:
            # Find classifier layer
            classifier = None
            for name in ['fc', 'classifier', 'head']:
                if hasattr(model, name):
                    classifier = getattr(model, name)
                    break
            if classifier is None:
                raise ValueError("Could not find classifier layer (fc/classifier/head)")
            grad_dim = sum(p.numel() for p in classifier.parameters())
        else:
            grad_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return cls(grad_dim=grad_dim, proj_dim=proj_dim, **kwargs)

    @staticmethod
    def _get_num_sms():
        """Get streaming multiprocessor count for current CUDA device."""
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count

    @property
    def projection_matrix(self):
        """Lazily create Gaussian projection matrix for naive backend."""
        if self._projection_matrix is None:
            torch.manual_seed(self.seed)
            # Scale by 1/sqrt(k) for variance preservation
            self._projection_matrix = torch.randn(
                self.grad_dim, self.proj_dim, dtype=torch.float32
            ) / (self.proj_dim ** 0.5)
        return self._projection_matrix

    def project(self, grads: torch.Tensor, progress_callback=None) -> torch.Tensor:
        """
        Project gradients from grad_dim to proj_dim.

        Args:
            grads: Tensor of shape (N, grad_dim)
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Projected tensor of shape (N, proj_dim)
        """
        if grads.shape[1] != self.grad_dim:
            raise ValueError(f"Expected grad_dim={self.grad_dim}, got {grads.shape[1]}")

        if self.proj_dim >= self.grad_dim:
            return grads  # No projection needed

        # Log backend info on first projection
        if self.verbose and not self._logged_backend:
            backend = "fast_jl (CUDA Rademacher)" if self.use_fast_jl else "torch (Gaussian matmul)"
            device_info = f"GPU:{self.device}" if torch.cuda.is_available() else "CPU"
            print(f"[Projector] {self.grad_dim:,} -> {self.proj_dim:,} using {backend} on {device_info}")
            self._logged_backend = True

        if self.use_fast_jl:
            return self._project_fast_jl(grads, progress_callback)
        else:
            return self._project_naive(grads, progress_callback)

    def _project_fast_jl(self, grads: torch.Tensor, progress_callback=None) -> torch.Tensor:
        """Project using fast_jl Rademacher projection (CUDA-optimized)."""
        N, D = grads.shape

        # fast_jl requires batch size to be multiple of 8
        remainder = N % 8
        if remainder != 0:
            pad_size = 8 - remainder
            grads = torch.cat([
                grads,
                torch.zeros(pad_size, D, dtype=grads.dtype, device=grads.device)
            ], dim=0)

        # Ensure contiguous and on correct device
        grads_gpu = grads.to(self.device).contiguous()

        if progress_callback:
            progress_callback(N // 2, N)  # fast_jl is one call, report midpoint

        # Call fast_jl - never materializes the D×k matrix!
        projected = fast_jl.project_rademacher_8(
            grads_gpu,
            self.proj_dim,
            self.seed,
            self._num_sms
        )

        # Trim padding
        if remainder != 0:
            projected = projected[:N]

        if progress_callback:
            progress_callback(N, N)

        return projected.cpu()

    def _project_naive(self, grads: torch.Tensor, progress_callback=None) -> torch.Tensor:
        """
        Project using Gaussian matrix multiplication on GPU with batching.

        Batches the projection to avoid OOM and provide progress updates.
        """
        N, D = grads.shape
        use_gpu = torch.cuda.is_available() and self.device != 'cpu'

        # Get or create projection matrix on GPU
        if use_gpu:
            if self._projection_matrix_gpu is None:
                self._projection_matrix_gpu = self.projection_matrix.to(self.device)
            proj_matrix = self._projection_matrix_gpu
        else:
            proj_matrix = self.projection_matrix

        # If small enough, do it all at once
        if N <= self.batch_size:
            if use_gpu:
                result = (grads.to(self.device) @ proj_matrix).cpu()
            else:
                result = grads @ proj_matrix
            if progress_callback:
                progress_callback(N, N)
            return result

        # Batch projection for large inputs
        all_projected = []
        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch = grads[start:end]

            if use_gpu:
                batch_projected = (batch.to(self.device) @ proj_matrix).cpu()
            else:
                batch_projected = batch @ proj_matrix

            all_projected.append(batch_projected)

            if progress_callback:
                progress_callback(end, N)

        return torch.cat(all_projected, dim=0)

    def __repr__(self):
        backend = "fast_jl" if self.use_fast_jl else "naive"
        return f"Projector({self.grad_dim} -> {self.proj_dim}, backend={backend}, seed={self.seed})"


def extract_gradients_from_images(
    model,
    images,
    labels,
    device='cuda',
    batch_size=32,
    progress_callback=None,
    projector=None,
    project_dim=None,
    projection_seed=None,
    last_layer_only=True,
    normalize_grads=True,
    use_fast_jl=True
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
        batch_size: Batch size for processing
        progress_callback: Optional callback(processed, total) for progress updates
        projector: Optional Projector instance for gradient projection. If provided,
                  project_dim/projection_seed/use_fast_jl are ignored.
        project_dim: If set (and no projector), create projector with this dimension.
        projection_seed: Random seed for projection matrix (for reproducibility/multiple projections)
        last_layer_only: If True, extract gradients only from classifier layer (default).
                        If False, extract from all model parameters (slower, higher memory).
        normalize_grads: If True, normalize gradients by sqrt(num_params) as in official TRAK.
                        This ensures scores are bounded and numerically stable. (default: True)
        use_fast_jl: If True and fast_jl is available, use CUDA-optimized Rademacher projection.
                    This is much more memory-efficient as it never materializes the projection matrix.
                    Falls back to naive Gaussian projection if fast_jl is unavailable. (default: True)

    Returns:
        Tensor of gradients (N, D) or (N, project_dim) if projection is enabled

    Raises:
        ValueError: If model architecture is not supported or features cannot be extracted
    """
    # Route to appropriate implementation based on last_layer_only setting
    if not last_layer_only:
        return _extract_gradients_full_model(
            model, images, labels, device, batch_size,
            progress_callback, projector, project_dim, projection_seed, normalize_grads, use_fast_jl
        )

    # Convert images to tensors if needed
    if isinstance(images, list):
        images = torch.stack(images)

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_gradients = []
    total_samples = len(images)
    processed_samples = 0

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

    # Create projector if needed (using grad_dim from classifier)
    if projector is None and project_dim is not None:
        grad_dim = sum(p.numel() for p in classifier.parameters())
        projector = Projector(
            grad_dim=grad_dim,
            proj_dim=project_dim,
            seed=projection_seed or 42,
            device=device,
            use_fast_jl=use_fast_jl
        )

    # Track num_params for TRAK normalization
    num_params_for_grad = sum(p.numel() for p in classifier.parameters())

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
        all_gradients.append(batch_grads)

        processed_samples += batch_size_actual
        if progress_callback:
            progress_callback(processed_samples, total_samples)

        # Clear GPU cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_gradients:
        return torch.empty(0)

    # Concatenate all batches
    result = torch.cat(all_gradients, dim=0)

    # Project using Projector (handles fast_jl vs naive internally)
    if projector is not None:
        result = projector.project(result)

    # TRAK normalization: divide by sqrt(num_params_for_grad)
    # This is critical for bounded, numerically stable scores!
    # See: https://github.com/MadryLab/trak - grads /= self.normalize_factor
    if normalize_grads:
        normalize_factor = (num_params_for_grad ** 0.5)
        result = result / normalize_factor

    return result


def extract_and_project_gradients(
    model,
    images,
    labels,
    proj_dim: int,
    seeds: list,
    device='cuda',
    batch_size=32,
    progress_callback=None,
    last_layer_only=True,
    normalize_grads=True,
    use_fast_jl=True,
    verbose=True
):
    """
    Extract gradients and project with multiple seeds in a single pass (GPU-efficient).

    This fuses gradient extraction and projection so gradients stay on GPU:
    - For each batch: extract grads on GPU → project with all seeds → move to CPU
    - Avoids the GPU→CPU→GPU round-trip of separate extract then project

    Args:
        model: The classification model
        images: Tensor of images
        labels: Tensor of labels
        proj_dim: Projection dimension
        seeds: List of projection seeds (e.g., [42, 43, 44] for 3 projections)
        device: Device for computation
        batch_size: Batch size for processing
        progress_callback: Optional callback(current, total, seed_idx) for progress
        last_layer_only: If True, extract gradients only from classifier layer
        normalize_grads: If True, normalize by sqrt(num_params) as in TRAK
        use_fast_jl: If True, use fast_jl when available
        verbose: If True, print backend info

    Returns:
        List of projected gradient tensors, one per seed.
        Each tensor has shape (N, proj_dim).

    Example:
        >>> seeds = [42, 43, 44]  # 3 projection seeds
        >>> projected_list = extract_and_project_gradients(
        ...     model, images, labels, proj_dim=1024, seeds=seeds
        ... )
        >>> len(projected_list)  # 3 projections
        3
        >>> projected_list[0].shape  # (N, 1024)
    """
    if not seeds:
        raise ValueError("seeds list cannot be empty")

    # Convert images to tensors if needed
    if isinstance(images, list):
        images = torch.stack(images)

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize storage for each seed
    all_projected = {seed: [] for seed in seeds}
    total_samples = len(images)
    processed_samples = 0

    model.eval()

    # Identify classifier layer
    classifier = None
    for name in ['fc', 'classifier', 'head']:
        if hasattr(model, name):
            classifier = getattr(model, name)
            break
    if classifier is None:
        raise ValueError("Could not find classifier layer (fc/classifier/head)")

    # Get grad_dim and create projectors for each seed
    grad_dim = sum(p.numel() for p in classifier.parameters())
    num_params_for_grad = grad_dim

    projectors = {}
    for i, seed in enumerate(seeds):
        projectors[seed] = Projector(
            grad_dim=grad_dim,
            proj_dim=proj_dim,
            seed=seed,
            device=device,
            use_fast_jl=use_fast_jl,
            verbose=(verbose and i == 0)  # Only log for first projector
        )

    # Extract classifier parameters for functional_call
    params = {k: v.detach() for k, v in classifier.named_parameters()}
    buffers = {k: v for k, v in classifier.named_buffers()}

    def compute_loss(params, buffers, feature, label):
        logits = functional_call(classifier, (params, buffers), (feature.unsqueeze(0),))
        return F.cross_entropy(logits, label.unsqueeze(0))

    per_sample_grad_fn = vmap(
        grad(compute_loss),
        in_dims=(None, None, 0, 0)
    )

    for batch_imgs, batch_labels in loader:
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        # Extract features
        with torch.no_grad():
            features = _extract_features(model, batch_imgs)

        # Compute per-sample gradients (stays on GPU)
        per_sample_grads = per_sample_grad_fn(params, buffers, features, batch_labels)

        # Flatten and concatenate (stays on GPU!)
        batch_size_actual = features.shape[0]
        grad_tensors = []
        for key in params.keys():
            grad_tensors.append(per_sample_grads[key].view(batch_size_actual, -1))
        batch_grads = torch.cat(grad_tensors, dim=1)  # Still on GPU!

        # TRAK normalization (on GPU)
        if normalize_grads:
            batch_grads = batch_grads / (num_params_for_grad ** 0.5)

        # Project with each seed while still on GPU, then move to CPU
        for seed in seeds:
            proj = projectors[seed]
            # Project on GPU (or use fast_jl)
            if proj.use_fast_jl:
                projected = proj._project_fast_jl(batch_grads)
            else:
                # GPU projection
                if proj._projection_matrix_gpu is None:
                    proj._projection_matrix_gpu = proj.projection_matrix.to(device)
                projected = (batch_grads @ proj._projection_matrix_gpu).cpu()

            all_projected[seed].append(projected)

        processed_samples += batch_size_actual
        if progress_callback:
            progress_callback(processed_samples, total_samples)

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Concatenate all batches for each seed
    result = []
    for seed in seeds:
        if all_projected[seed]:
            result.append(torch.cat(all_projected[seed], dim=0))
        else:
            result.append(torch.empty(0, proj_dim))

    return result


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


def project_gradients_random(
    gradients,
    target_dim=8192,
    seed=42,
    normalize=True,
    use_fast_jl=True,
    progress_callback=None,
    verbose=True
):
    """
    Project high-dimensional gradients to lower dimension using random projection.
    Uses Johnson-Lindenstrauss lemma for dimensionality reduction.

    Uses the Projector class internally, which automatically uses fast_jl
    (CUDA-optimized Rademacher projection) when available.

    Args:
        gradients: Tensor of shape (N, D) where D is large
        target_dim: Target dimension (should be << D)
        seed: Random seed for reproducibility
        normalize: If True, L2-normalize gradients before projection (recommended)
        use_fast_jl: If True, use fast_jl when available (default: True)
        progress_callback: Optional callback(current, total) for progress updates
        verbose: If True, print backend info (default: True)

    Returns:
        Projected gradients of shape (N, target_dim), normalized if normalize=True
    """
    N, D = gradients.shape

    # L2-normalize gradients before projection (per sample)
    if normalize:
        grad_norm = gradients.norm(dim=1, keepdim=True)
        grad_norm[grad_norm == 0] = 1.0
        gradients = gradients / grad_norm

    if D <= target_dim:
        if progress_callback:
            progress_callback(N, N)
        return gradients

    # Use Projector class (handles fast_jl vs naive internally)
    projector = Projector(
        grad_dim=D,
        proj_dim=target_dim,
        seed=seed,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_fast_jl=use_fast_jl,
        verbose=verbose
    )

    return projector.project(gradients, progress_callback=progress_callback)


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

    # Gradients should already be projected (JL projection done during extraction)
    report_progress(f"Processing gradients ({N_train} train, {N_target} target)", 10)

    # Move to device
    train_gpu = train_grads.to(device)
    target_gpu = target_grads.to(device)

    # Unit sphere normalization (critical for consistent TRAK scores)
    # This ensures influence depends on gradient direction, not magnitude
    report_progress("Normalizing gradients to unit sphere", 20)
    train_norm = train_gpu.norm(dim=1, keepdim=True)
    train_norm[train_norm == 0] = 1.0  # Avoid division by zero
    train_gpu = train_gpu / train_norm

    target_norm = target_gpu.norm(dim=1, keepdim=True)
    target_norm[target_norm == 0] = 1.0
    target_gpu = target_gpu / target_norm

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
    projector=None,
    project_dim=None,
    projection_seed=None,
    normalize_grads=True,
    use_fast_jl=True
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
        projector: Optional Projector instance for gradient projection.
        project_dim: If set (and no projector), create projector with this dimension.
        projection_seed: Random seed for projection matrix
        normalize_grads: If True, normalize gradients by sqrt(num_params) as in official TRAK.
        use_fast_jl: If True and fast_jl is available, use CUDA-optimized Rademacher projection.

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

    # Get all trainable parameters
    model.eval()

    # Count num_params for normalization and projector creation
    num_params_for_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create projector if needed
    if projector is None and project_dim is not None:
        projector = Projector(
            grad_dim=num_params_for_grad,
            proj_dim=project_dim,
            seed=projection_seed or 42,
            device=device,
            use_fast_jl=use_fast_jl
        )

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

    # Concatenate all batches
    result = torch.cat(all_gradients, dim=0)

    # Project using Projector (handles fast_jl vs naive internally)
    if projector is not None:
        result = projector.project(result)

    # TRAK normalization: divide by sqrt(num_params_for_grad)
    # This is critical for bounded, numerically stable scores!
    if normalize_grads:
        normalize_factor = (num_params_for_grad ** 0.5)
        result = result / normalize_factor

    return result
