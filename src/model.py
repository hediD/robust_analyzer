import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms
import pyexr

from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    look_at_view_transform,
    BlendParams
)

from transformers import ViTForImageClassification, ViTImageProcessor
import imageio.v3 as iio

Image.MAX_IMAGE_PIXELS = None  # Ignore warning about Atlas texture can be very high resolution, will be downscaled
TEXTURE_MAX_IMAGE_PIXELS = 80_000_000  # Atlas texture can be very high resolution

def downscale_to_max_pixels(pil_img, max_pixels):
    """
    Downscale a PIL image so that its total number of pixels does not exceed max_pixels.
    Preserves aspect ratio. Returns the (possibly) resized image.
    """
    w, h = pil_img.size
    n_pixels = w * h
    if n_pixels > max_pixels:
        scale = (max_pixels / n_pixels) ** 0.5
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return pil_img

class Model(nn.Module):
    """
    A model that renders 3D objects and feeds them into an image classifier.

    Key features:
      1. Loads a 3D mesh from an .obj file (with optional texture)
      2. Creates and optionally optimizes a texture (using clustering for color reduction)
      3. Optimizes camera position in 3D space
      4. Optimizes lighting location and intensity
      5. Renders the 3D object and passes it to a ViT model for classification
    """

    def __init__(
        self,
        obj_path,
        texture_path=None,
        envmap_paths=None,
        camera_coords=None,
        device=None,
        optimize_kwargs=None,
        raster_settings=None,
        min_max_proportion=(0.3, 0.8),
        batch_size=1,
        nb_clusters=4,
        positive_z=True
    ):
        """
        Initialize the model with mesh, texture, and rendering parameters.

        Args:
            obj_path (str): Path to the .obj file of the mesh
            texture_path (str, optional): Path to the texture image file
            envmap_paths (list, optional): Paths to environment map image files
            camera_coords (torch.Tensor, optional): Initial camera coordinates
            device (str or torch.device): Device to use ('cuda' or 'cpu')
            optimize_kwargs (dict, optional): Dict with keys 'texture', 'camera', 'lighting' (bool values)
            raster_settings (dict, optional): Overrides for PyTorch3D RasterizationSettings
            min_max_proportion (tuple): Proportion for min/max distance from bounding box
            batch_size (int): Number of images to render in parallel
            nb_clusters (int): Number of color clusters for texture optimization
            positive_z (bool): If True, constrain camera to positive z coordinates
        """
        super().__init__()
        self.batch_size = batch_size
        self.positive_z = positive_z

        # ------ DEVICE SETUP ------
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------ OPTIMIZATION FLAGS ------
        self.optimize_kwargs = {
            "texture": False,
            "camera": False,
            "lighting": False
        }
        if optimize_kwargs is not None and isinstance(optimize_kwargs, dict):
            self.optimize_kwargs.update(optimize_kwargs)

        # ------ MESH LOADING ------
        use_mtl = texture_path is None

        # Load geometry from obj file
        verts, faces, aux = load_obj(obj_path, load_textures=use_mtl)
        verts = verts.to(self.device)
        faces_idx = faces.verts_idx.to(self.device)

        # ------ TEXTURE SETUP ------
        if texture_path is None:
            # Use first texture from MTL file
            texture_arr = list(aux.texture_images.values())[0].cpu().numpy()
            pil_texture = Image.fromarray((texture_arr * 255).astype(np.uint8))
        else:
            # Load texture from file
            pil_texture = Image.open(texture_path).convert('RGB')

        max_pixels = globals().get("TEXTURE_MAX_IMAGE_PIXELS", float('inf'))
        pil_texture = downscale_to_max_pixels(pil_texture, max_pixels)
        texture_arr = np.array(pil_texture).astype(np.float32) / 255.0
        texture_image = torch.from_numpy(texture_arr).unsqueeze(0).to(self.device)

        # Check if we can optimize texture (can't optimize with MTL)
        if use_mtl and self.optimize_kwargs["texture"]:
            print("Warning: can't optimize texture when using MTL files. You need to extract the texture from the material file "
                  "and feed it to the model via the texture_path argument.")
            self.optimize_kwargs["texture"] = False

        # Handle texture tensor based on optimization flag
        if self.optimize_kwargs["texture"]:
            texture_image = nn.Parameter(texture_image.repeat(batch_size, 1, 1, 1), requires_grad=True)
        else:
            texture_image = texture_image.expand(batch_size, -1, -1, -1).detach()
        self.texture_image = texture_image.detach().clone()

        # Create UV texture from loaded data
        faces_uvs = faces.textures_idx.to(self.device)
        verts_uvs = aux.verts_uvs.to(self.device)
        texture = TexturesUV(
            maps=texture_image,
            faces_uvs=[faces_uvs] * batch_size,
            verts_uvs=[verts_uvs] * batch_size
        )

        # ------ MESH CREATION ------
        mesh = load_objs_as_meshes([obj_path], device=device)
        # Scale mesh
        mesh._verts_list = [v / 5.0 for v in mesh.verts_list()]
        mesh = mesh.to(device).extend(batch_size)
        self.meshes = mesh

        # ------ BOUNDING BOX PROPERTIES ------
        self._calculate_bbox_properties(min_max_proportion)

        # ------ CAMERA SETUP ------
        if camera_coords is None:
            camera_coords = self._get_random_camera_coords()
        self.camera_coords = camera_coords.to(self.device)
        if self.camera_coords.ndim == 1:
            self.camera_coords = self.camera_coords.unsqueeze(0)

        # ------ LIGHTING SETUP ------
        self.light_color = torch.tensor([[1.0, 1.0, 1.0]], device=self.device)
        self.specular_strength = 0.5  # Reduced specular strength

        # Initialize light parameters
        self.init_light_location = self._get_random_camera_coords().to(self.device)
        self.init_light_intensity = torch.tensor([1.0] * batch_size, device=self.device)

        # ------ RENDERING SETTINGS ------
        default_raster_settings = {
            'image_size': 224,
            'blur_radius': 1e-6,
            'faces_per_pixel': 8,
            'bin_size': 16,
            'max_faces_per_bin': 20_000,
        }

        raster_settings = {**default_raster_settings, **raster_settings}

        self.raster_settings = RasterizationSettings(**raster_settings)

        # ------ TEXTURE CLUSTERING ------
        # Cluster the texture for color centroid optimization
        self.cluster_indices, self.cluster_colors = self._cluster_texture(nb_clusters=nb_clusters)

        # ------ SCENE PARAMETERS SETUP ------
        self.init_scene_params = {
            'camera': self.camera_coords[:self.batch_size].to(self.device),
            'texture_centroids': self.cluster_colors.to(self.device),
            'light_location': self.init_light_location.to(self.device),
            'light_intensity': self.init_light_intensity.to(self.device)
        }

        # Create scene_params dictionary with parameters that need gradients
        optimize_predicate = lambda k: (self.optimize_kwargs["camera"] and k=="camera") or \
                                        (k == 'texture_centroids' and self.optimize_kwargs['texture']) or \
                                        ((k == 'light_location' or k == 'light_intensity') and self.optimize_kwargs['lighting'])

        self.scene_params = {
            k: v.clone().detach().requires_grad_(optimize_predicate(k))
            for k, v in self.init_scene_params.items()
        }

        # Create initial lights
        self.lights = PointLights(
            device=self.device,
            location=self.scene_params['light_location'].detach(),
            ambient_color=self.light_color * self.scene_params['light_intensity'].detach().unsqueeze(-1),
            diffuse_color=self.light_color * self.scene_params['light_intensity'].detach().unsqueeze(-1),
            specular_color=self.light_color * self.scene_params['light_intensity'].detach().unsqueeze(-1) * self.specular_strength
        ).to(self.device)

        # ------ RENDERER SETUP ------
        camera = FoVPerspectiveCameras(device=self.device)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=camera,
                lights=self.lights,
                blend_params=BlendParams(
                    sigma=1e-2,  # more stable blending
                    gamma=1e-2,  # prevent z-fighting artifacts
                    background_color=(0.1, 0.1, 0.1)  # dark gray
                )
            )
        )

        # ------ IMAGE CLASSIFICATION MODEL ------
        # Vision Transformer model (frozen)
        self.ml_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(self.device).eval()
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        for param in self.ml_model.parameters():
            param.requires_grad = False

        # Image transformations
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.transforms = transforms.Compose([
            transforms.Lambda(
                lambda tensor: tensor[..., :3].permute(0, 3, 1, 2) if tensor.dim() == 4 else tensor.permute(2, 0, 1).unsqueeze(0)
            ),
            transforms.Normalize(mean, std),
            transforms.Resize((224, 224)),
        ])

        # Store lighting optimization flag for convenience
        self.optimize_lighting = self.optimize_kwargs["lighting"]

        # ------ ENVIRONMENT MAP SETUP ------
        self.use_envmap = envmap_paths is not None and len(envmap_paths) > 0
        if self.use_envmap:
            if isinstance(envmap_paths, str):
                envmap_paths = [envmap_paths]
            self.envmaps = torch.stack([self._load_envmap(path) for path in envmap_paths])
        self.n_envmaps = len(envmap_paths) if envmap_paths else 1

    # ------ SCENE PARAMETER METHODS ------

    def update_scene_params(self, scene_params: Dict[str, torch.Tensor]) -> None:
        """Update the scene parameters with new values."""
        for param, value in scene_params.items():
            setattr(self.scene_params[param], 'data', value.to(self.device))

    def _calculate_bbox_properties(self, min_max_proportion: Tuple[float, float]) -> None:
        """
        Calculate bounding box (center, size, min/max distance) for the loaded mesh.
        """
        verts = self.meshes.verts_packed()
        self.bbox_min = verts.min(dim=0)[0]
        self.bbox_max = verts.max(dim=0)[0]
        self.bbox_center = (self.bbox_min + self.bbox_max) / 2.0
        self.bbox_size = (self.bbox_max - self.bbox_min).max()

        # Min and max distances are derived from bounding box size
        self.min_distance = self.bbox_size / (2.0 * min_max_proportion[1])
        self.max_distance = self.bbox_size / (2.0 * min_max_proportion[0])

    # ------ CAMERA POSITION METHODS ------

    def _get_random_camera_coords(self, fov: torch.Tensor = torch.tensor(60.0)) -> torch.Tensor:
        """
        Generate random camera coordinates based on a spherical distribution.

        Args:
            fov: Field of view in degrees

        Returns:
            Tensor of camera coordinates with shape (batch_size, 3)
        """
        # Compute radius from bounding box size and FOV
        r = self.bbox_size.cpu() / (2.0 * np.random.uniform(0.5, 0.7)) / torch.tan(fov * (torch.pi / 180.0) / 2)

        # Random spherical angles: azimuth and elevation
        azimuth = 2 * np.pi * np.random.rand(self.batch_size)
        if self.positive_z:
            elevation = np.arccos(np.random.rand(self.batch_size))  # Only northern hemisphere
        else:
            elevation = np.arccos(2 * np.random.rand(self.batch_size) - 1)  # Full sphere

        # Convert spherical to Cartesian coordinates
        x = r * np.sin(elevation) * np.cos(azimuth)
        y = r * np.sin(elevation) * np.sin(azimuth)
        z = r * np.cos(elevation)

        coords = np.stack([x, y, z], axis=1).astype(np.float32)
        return torch.tensor(coords, device=self.device)

    def _constrain_position(self, position: torch.Tensor, min_distance: float, max_distance: float, positive_z: bool = True) -> torch.Tensor:
        """
        Clamp position to be within distance bounds while preserving direction.

        Args:
            position: Position tensor to constrain
            min_distance: Minimum allowed distance from bounding box center
            max_distance: Maximum allowed distance from bounding box center
            positive_z: If True, constrain z coordinates to be positive

        Returns:
            Constrained position tensor
        """
        if positive_z:
            position[:, 2] = position[:, 2].clamp(min=0)

        camera_vectors = position - self.bbox_center
        distances = torch.norm(camera_vectors, dim=-1, keepdim=True)
        normalized_vectors = camera_vectors / (distances + 1e-9)  # avoid division by zero

        clamped_distances = torch.clamp(distances, min_distance, max_distance)
        return self.bbox_center + normalized_vectors * clamped_distances

    def _constrain_camera(self) -> None:
        """Constrain camera position within allowed distance range."""
        self.scene_params['camera'].data = self._constrain_position(
            self.scene_params['camera'].data,
            self.min_distance,
            self.max_distance,
            self.positive_z
        )

    def distance_penalty(self) -> torch.Tensor:
        """
        Calculate penalty for camera distance outside allowed range.

        Returns:
            Tensor representing the distance penalty
        """
        dist = torch.norm(self.scene_params['camera'] - self.bbox_center, dim=-1)
        penalty = (
            (dist < self.min_distance).float() * (self.min_distance - dist) ** 2 +
            (dist > self.max_distance).float() * (dist - self.max_distance) ** 2
        )
        return penalty.mean()

    # ------ TEXTURE METHODS ------

    def _cluster_texture(self, nb_clusters: int = 3) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Cluster texture image into color groups for optimization.

        Args:
            nb_clusters: Number of color clusters to create

        Returns:
            Tuple containing:
            - Dictionary mapping cluster IDs to pixel indices
            - Tensor of cluster centroid colors
        """
        # Extract the first texture from the batch for clustering
        texture_arr = self.texture_image[0].detach().cpu().numpy()

        # Mask out near-black pixels (treated as unmapped or irrelevant)
        to_cluster_mask = (texture_arr.mean(axis=-1) > 1e-4)
        pixels_to_cluster = texture_arr[to_cluster_mask].reshape(-1, 3)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=nb_clusters, random_state=0)
        kmeans.fit(pixels_to_cluster)

        # Combine labels to skip a "0" cluster for black pixels
        cluster_labels = kmeans.labels_ + 1
        centroids_values = np.concatenate((np.zeros((1, 3)), kmeans.cluster_centers_), axis=0)

        # Recreate cluster array
        clusters_arr = np.zeros(texture_arr.shape[:2])
        clusters_arr[to_cluster_mask] = cluster_labels  # actual labels

        # Make a dictionary of per-cluster indices
        flattened_clusters = clusters_arr.flatten()
        per_cluster_indices = {
            cluster_id: torch.tensor(np.where(flattened_clusters == cluster_id)[0])
            for cluster_id in range(1, nb_clusters + 1)
        }

        cluster_colors = torch.tensor(centroids_values[1:], dtype=torch.float32, device=self.device)
        self.clusters_arr = clusters_arr
        return per_cluster_indices, cluster_colors

    def _fill_texture(self) -> torch.Tensor:
        """
        Reconstruct texture from optimized cluster centroids.

        Returns:
            Texture tensor with each pixel filled with its centroid color
        """
        texture = torch.zeros_like(self.texture_image)  # shape: (B, H, W, C)
        batch_size, height, width = texture.shape[:3]

        for cluster_id, flat_indices in self.cluster_indices.items():
            row_indices = flat_indices // width
            col_indices = flat_indices % width

            # For each image in the batch, fill pixels with the cluster centroid
            cluster_color = self.scene_params['texture_centroids'][cluster_id - 1]
            for b in range(batch_size):
                texture[b, row_indices, col_indices] = cluster_color

        return texture

    def _constrain_texture(self) -> None:
        """Ensure texture centroid colors stay within [0,1] range."""
        self.scene_params['texture_centroids'].data = torch.clamp(
            self.scene_params['texture_centroids'], 0.0, 1.0
        )

    # ------ LIGHTING METHODS ------

    def _constrain_lights(self) -> None:
        """Constrain light position and intensity to reasonable values."""
        # Constrain light position
        self.scene_params['light_location'].data = self._constrain_position(
            self.scene_params['light_location'].data,
            self.min_distance,
            self.max_distance,
            self.positive_z
        )

        # Keep light intensity in [0.3, 1.5]
        self.scene_params['light_intensity'].data = torch.clamp(
            self.scene_params['light_intensity'].data, 0.3, 1.5
        )

    # ------ ENVIRONMENT MAP METHODS ------

    def _load_envmap(self, path: str, gamma: float = 2.2, alpha: float = 0.5, target_size: Tuple[int, int] = (1024, 2048)) -> torch.Tensor:
        """
        Load and process environment map.

        Args:
            path: Path to environment map image
            gamma: Gamma correction value
            alpha: Scaling factor
            target_size: Desired (height, width) for resizing

        Returns:
            Processed environment map tensor
        """
        # Read image
        im = pyexr.open(path).get()[..., :3]

        # Convert to tensor and resize
        im_tensor = torch.tensor(im, device=self.device, dtype=torch.float32).squeeze(0)
        im_resized = F.interpolate(
            im_tensor.permute(2, 0, 1).unsqueeze(0),  # Add batch dimension
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        # Apply gamma correction and alpha scaling
        return (im_resized ** (1/gamma)) * alpha

    def _get_background_rays(self, R: torch.Tensor) -> torch.Tensor:
        """
        Generate world-space ray directions using Z-up convention.

        Args:
            R: Rotation matrix for camera

        Returns:
            Tensor of ray directions
        """
        H = W = self.raster_settings.image_size
        aspect_ratio = W / H
        fov_rad = torch.tensor(60.0 * np.pi / 180.0)

        # Create normalized device coordinates
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing="ij"
        )

        # Apply perspective projection
        x = x * aspect_ratio * torch.tan(fov_rad/2)
        y = y * torch.tan(fov_rad/2)

        # Camera space directions (forward is -Z in camera space)
        dirs = torch.stack([x, y, -torch.ones_like(x)], dim=-1)
        dirs = F.normalize(dirs, dim=-1)

        # Transform to world space using rotation matrix
        return F.normalize(dirs @ R.transpose(1, 2), dim=-1)

    def _sample_envmap(self, directions: torch.Tensor, envmap_idx: int = 0) -> torch.Tensor:
        """
        Sample environment map based on ray directions.

        Args:
            directions: Ray direction vectors
            envmap_idx: Index of environment map to sample

        Returns:
            Sampled colors from environment map
        """
        # Extract components
        x = directions[..., 0]  # right
        y = directions[..., 1]  # forward
        z = directions[..., 2]  # up

        # Calculate spherical coordinates
        phi = torch.atan2(x, y)  # azimuth around Z-axis
        theta = torch.acos(z.clamp(-1+1e-6, 1-1e-6))  # angle from Z-axis

        # Convert to UV coordinates
        u = (phi / (2 * np.pi) + 0.5) % 1.0
        v = 1.0 - (theta / np.pi)  # Invert V coordinate

        # Sample environment map
        H, W = self.envmaps.shape[1:3]
        u_idx = (u * (W - 1)).clamp(0, W-1).long()
        v_idx = (v * (H - 1)).clamp(0, H-1).long()

        return self.envmaps[envmap_idx, v_idx, u_idx]

    def _get_background_mask(self, meshes: torch.Tensor, R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Get mask indicating which pixels are background (not part of the mesh).

        Args:
            meshes: PyTorch3D meshes object
            R: Rotation matrices
            T: Translation vectors

        Returns:
            Boolean mask where True indicates background pixels
        """
        fragments = self.renderer.rasterizer(meshes_world=meshes, R=R, T=T)
        top_face_idx = fragments.pix_to_face[..., 0]  # shape (batch_size, H, W)
        background_mask = (top_face_idx < 0)[..., None]  # Add channel dimension
        return background_mask

    # ------ RENDERING METHODS ------

    def render(self, with_grad: bool = True, image_res: Optional[int] = None) -> torch.Tensor:
        """
        Render the object with current parameters.

        Args:
            with_grad: Whether to compute gradients during rendering
            image_res: Override default image resolution

        Returns:
            Rendered images tensor with shape (B, N_envmaps, H, W, C)
        """
        # Use context manager if gradients are not needed
        with torch.set_grad_enabled(with_grad):
            meshes = self.meshes.clone()

            # Handle custom resolution if specified
            if image_res is not None:
                raster_settings = RasterizationSettings(
                    **{**vars(self.raster_settings), 'image_size': image_res}
                )
            else:
                raster_settings = self.raster_settings

            # Apply constraints to parameters if computing gradients
            if with_grad:
                self._constrain_camera()
                self._constrain_texture()
                if self.optimize_kwargs["lighting"]:
                    self._constrain_lights()

            # Update texture if optimizing color centroids
            if with_grad and self.scene_params["texture_centroids"].requires_grad:
                texture_maps = self._fill_texture()
                new_textures = TexturesUV(
                    maps=texture_maps,
                    faces_uvs=meshes.textures.faces_uvs_list(),
                    verts_uvs=meshes.textures.verts_uvs_list()
                )
                meshes.textures = new_textures

            # Create lights based on gradient requirement
            lights = (
                PointLights(
                    device=self.device,
                    location=self.scene_params['light_location'],
                    ambient_color=self.light_color * self.scene_params['light_intensity'].unsqueeze(-1),
                    diffuse_color=self.light_color * self.scene_params['light_intensity'].unsqueeze(-1),
                    specular_color=self.light_color * self.scene_params['light_intensity'].unsqueeze(-1) * self.specular_strength
                ).to(self.device)
                if with_grad and self.optimize_kwargs["lighting"]
                else self.lights
            )

            # Compute view transforms
            up_vector = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(self.batch_size, -1)
            R, T = look_at_view_transform(
                eye=self.scene_params['camera'].to(self.device),
                at=self.bbox_center.to(self.device).unsqueeze(0).expand(self.batch_size, -1),
                up=up_vector
            )
            R, T = R.to(self.device), T.to(self.device)

            # Create PyTorch3D cameras with modified znear/zfar values
            camera = FoVPerspectiveCameras(
                device=self.device,
                znear=0.1,  # Increased znear (was likely 0.1 before)
                zfar=500.0,  # Adjusted zfar to maintain good depth precision
                aspect_ratio=1.0
            )

            # Create renderer with current parameters
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=camera,
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=camera,
                    lights=lights
                )
            )

            # Render base images with safety checks
            base_images = renderer(meshes_world=meshes, R=R, T=T)[..., :3]

            # Check for NaN/Inf values and replace them
            if torch.isnan(base_images).any() or torch.isinf(base_images).any():
                base_images = torch.where(
                    torch.isnan(base_images) | torch.isinf(base_images),
                    torch.tensor([0.1, 0.1, 0.1], device=self.device),
                    base_images
                )

            # Apply stricter clamping to the rendered colors
            base_images = base_images.clamp(0.0, 0.95)

            # Handle environment maps if enabled
            if self.use_envmap:
                # Initialize output tensor for all environment maps
                images = torch.zeros(self.batch_size, self.n_envmaps, *base_images.shape[1:],
                                   device=self.device)

                # Get background mask
                background = self._get_background_mask(meshes, R=R, T=T)

                # Apply environment maps
                for i in range(self.batch_size):
                    rays = self._get_background_rays(R[i:i+1])

                    for j in range(self.n_envmaps):
                        env_background = self._sample_envmap(rays, envmap_idx=j)
                        images[i, j] = torch.where(background[i], env_background, base_images[i])
            else:
                # No environment maps - just use rendered images
                images = base_images.unsqueeze(1)  # Add envmap dimension for consistency

            # Apply gamma correction with robust handling
            gamma = 2.2
            images = (torch.clamp(images, 1e-6, 0.95) + 1e-6) ** (1 / gamma)

            return images

    def render_initial(self) -> torch.Tensor:
        """
        Render mesh using initial parameters (original texture, initial camera coords).

        Returns:
            Rendered images tensor
        """
        with torch.no_grad():
            # Clone mesh and apply the initial texture
            meshes = self.meshes.clone()
            meshes.textures = TexturesUV(
                maps=self.texture_image,
                faces_uvs=meshes.textures.faces_uvs_list(),
                verts_uvs=meshes.textures.verts_uvs_list()
            )

            # Create initial lights
            orig_lights = PointLights(
                device=self.device,
                location=self.init_light_location.detach(),
                ambient_color=self.light_color * self.init_light_intensity.detach().unsqueeze(-1),
                diffuse_color=self.light_color * self.init_light_intensity.detach().unsqueeze(-1),
                specular_color=self.light_color * self.init_light_intensity.detach().unsqueeze(-1) * self.specular_strength
            ).to(self.device)

            # Create renderer with initial lighting
            camera = FoVPerspectiveCameras(device=self.device)
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=camera,
                    raster_settings=self.raster_settings
                ),
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=camera,
                    lights=orig_lights
                )
            )

            # Use initial camera coords
            up_vector = torch.tensor([[0.0, 0.0, 1.0]]).expand(self.batch_size, -1)
            R, T = look_at_view_transform(
                eye=self.camera_coords,
                at=self.bbox_center.unsqueeze(0).expand(self.batch_size, -1),
                up=up_vector
            )
            R, T = R.to(self.device), T.to(self.device)

            # Render base images
            base_images = renderer(meshes_world=meshes, R=R, T=T)[..., :3]

            if self.use_envmap:
                # Initialize output tensor for all environment maps
                images = torch.zeros(self.batch_size, self.n_envmaps, *base_images.shape[1:],
                                   device=self.device)

                # Get background mask
                background = self._get_background_mask(meshes, R=R, T=T)

                # Apply environment maps
                for i in range(self.batch_size):
                    rays = self._get_background_rays(R[i:i+1])

                    for j in range(self.n_envmaps):
                        env_background = self._sample_envmap(rays, envmap_idx=j)
                        bg_mask = background[i].unsqueeze(0)  # Add channel dim if needed
                        images[i, j] = torch.where(bg_mask, env_background, base_images[i])
            else:
                images = base_images.unsqueeze(1)  # Add envmap dimension for consistency

            # Gamma correction and clamping
            gamma = 2.2
            images = torch.clamp(images, 1e-6, 0.95) ** (1 / gamma)
            return images

    # ------ FORWARD PASS ------

    def forward(self, return_render: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: render object and classify with ViT.

        Args:
            return_render: If True, also return rendered images

        Returns:
            Tuple containing:
            - Classification logits with shape (B, N_envmaps, num_classes)
            - (Optional) Rendered images if return_render is True
        """
        images = self.render()  # (B, N_envmaps, H, W, C)
        images_orig = images.clone()

        # Reshape for batch processing through ViT
        B, N, H, W, C = images.shape
        images = images.reshape(B * N, H, W, C)
        images = images.permute(0, 3, 1, 2)  # (B*N, C, H, W)

        # Resize to ViT expected input size
        if images.shape[2:4] != (224, 224):
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize for ViT
        mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        pixel_values = (images - mean) / std

        # ViT forward pass
        outputs = self.ml_model(
            pixel_values=pixel_values,
            output_attentions=False
        )

        # Reshape logits back to (B, N_envmaps, num_classes)
        logits = outputs.logits.reshape(B, N, -1)

        if return_render:
            return logits, images_orig
        return logits


if __name__ == "__main__":
    from utils import show_pred

    # Load object mesh and texture
    obj = "airplane"
    obj_path = f"{obj}/mesh.obj"
    texture_path = f"{obj}/texture.png"

    # List of environment maps/backgrounds to use
    envmap_paths = [
        "environments_sky/farm_field_puresky_2k.exr",
        "environments/goegap_road_2k.exr"
    ]

    model = Model(
        obj_path,
        texture_path,
        envmap_paths=envmap_paths,
        batch_size=32,
        optimize_kwargs={"camera": True, "texture": True, "lighting": True},
        raster_settings={"image_size": 224},
        device="cuda"
    )

    logits, renders = model(return_render=True)

    env_idx = 0
    print(show_pred(logits[:, env_idx], topk=3))
