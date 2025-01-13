import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    look_at_view_transform
)

from transformers import ViTForImageClassification, ViTImageProcessor
import imageio.v3 as iio


class Model(nn.Module):
    """
    A model that:
      1. Loads a 3D mesh from an .obj file (with optional texture).
      2. Creates and optionally optimizes a texture (using clustering to reduce
         the number of colors).
      3. Optimizes camera position in 3D space.
      4. Optimizes lighting location and intensity.
      5. Renders the 3D object and passes it to a ViT model for classification.

    Args:
        obj_path (str): Path to the .obj file of the mesh.
        texture_path (str, optional): Path to the texture image file.
        envmap_paths (list, optional): Paths to the environment map image files.
        camera_coords (torch.Tensor, optional): Initial camera coordinates.
            If None, they will be randomly generated.
        device (str or torch.device, optional): Device to use ('cuda' or 'cpu').
        optimize_camera (bool, optional): If True, camera coordinates are optimized.
        optimize_texture (bool, optional): If True, texture (color centroids) is optimized.
        optimize_lighting (bool, optional): If True, light parameters are optimized.
        raster_settings (dict, optional): Overrides for PyTorch3D RasterizationSettings.
        min_max_proportion (tuple, optional): Proportion for min and max distance from bounding box.
        batch_size (int, optional): Number of images to render in parallel (batch size).
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
        nb_clusters=4
    ):
        super().__init__()
        self.batch_size = batch_size

        # -- Device setup --
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -- Load geometry --
        verts, faces, aux = load_obj(obj_path)
        verts = verts.to(self.device)
        faces_idx = faces.verts_idx.to(self.device)

        # -- Load or create default texture --
        if texture_path is None:
            # Default: gray texture of size 1024 x 1024
            texture_image = torch.ones((1, 1024, 1024, 3), device=self.device) * 0.5
        else:
            # Load texture from file
            pil_texture = Image.open(texture_path).convert('RGB')
            texture_arr = np.array(pil_texture).astype(np.float32) / 255.0
            texture_image = torch.from_numpy(texture_arr).unsqueeze(0).to(self.device)

        # Update the default optimize_kwargs to include texture_centroids and lighting parameters
        self.optimize_kwargs = {
            "texture": False,
            "camera": False,
            "lighting": False
        }
        if optimize_kwargs is not None and isinstance(optimize_kwargs, dict):
            self.optimize_kwargs.update(optimize_kwargs)

        # -- Manage texture for batch and optimize_texture flag --
        if self.optimize_kwargs["texture"]:
            # Create separate texture parameters for each item in the batch
            texture_image = nn.Parameter(texture_image.repeat(batch_size, 1, 1, 1), requires_grad=True)
        else:
            # Expand texture to match batch size but not optimized
            texture_image = texture_image.expand(batch_size, -1, -1, -1).detach()
        self.texture_image = texture_image.detach().clone()

        # -- Apply the texture to the mesh --
        faces_uvs = faces.textures_idx.to(self.device)
        verts_uvs = aux.verts_uvs.to(self.device)
        texture = TexturesUV(
            maps=texture_image,
            faces_uvs=[faces_uvs] * batch_size,
            verts_uvs=[verts_uvs] * batch_size
        )

        # -- Create the mesh structure --
        self.meshes = Meshes(
            verts=[verts] * batch_size,
            faces=[faces_idx] * batch_size,
            textures=texture
        ).to(self.device)

        # -- Compute bounding box properties (center, min, max distances) --
        self._calculate_bbox_properties(min_max_proportion)

        # -- Initialize camera coordinates --
        if camera_coords is None:
            camera_coords = self._get_random_camera_coords()
        self.camera_coords = camera_coords.to(self.device)
        if self.camera_coords.ndim == 1:
            self.camera_coords = self.camera_coords.unsqueeze(0)

        # Define a constant light color
        self.light_color = torch.tensor([[1.0, 1.0, 1.0]], device=self.device)

        # Initialize light parameters
        #self.init_light_location = torch.tensor([[0.0, 0.0, -3.0]] * batch_size, device=self.device)
        self.init_light_location = self._get_random_camera_coords().to(self.device)
        self.init_light_intensity = torch.tensor([1.0] * batch_size, device=self.device)

        # -- Rasterization settings --
        default_raster_settings = {
            'image_size': 224,
            'blur_radius': 1e-6,
            'faces_per_pixel': 1,
            'max_faces_per_bin': 100_000
        }
        if raster_settings is not None:
            default_raster_settings.update(raster_settings)
        self.raster_settings = RasterizationSettings(**default_raster_settings)

        # -- Cluster the texture for color centroid optimization --
        self.cluster_indices, self.cluster_colors = self._cluster_texture(nb_clusters=nb_clusters)

        self.init_scene_params = {
            'camera': self.camera_coords[:self.batch_size].to(self.device),
            'texture_centroids': self.cluster_colors.to(self.device),
            'light_location': self.init_light_location.to(self.device),
            'light_intensity': self.init_light_intensity.to(self.device)
        }

        optimize_predicate = lambda k: self.optimize_kwargs["camera"] and k=="camera" or \
                                        (k == 'texture_centroids' and self.optimize_kwargs['texture']) or \
                                        ((k == 'light_location' or k == 'light_intensity') and self.optimize_kwargs['lighting'])

        # Modify scene_params creation to handle texture_centroids and lighting parameters
        self.scene_params = {
            k: v.clone().detach().requires_grad_(optimize_predicate(k)) for k, v in self.init_scene_params.items()
        }

        # Initial lights creation
        self.lights = PointLights(
            device=self.device,
            location=self.scene_params['light_location'].detach(),
            ambient_color=self.light_color * self.scene_params['light_intensity'].detach().unsqueeze(-1),
            diffuse_color=self.light_color * self.scene_params['light_intensity'].detach().unsqueeze(-1),
            specular_color=self.light_color * self.scene_params['light_intensity'].detach().unsqueeze(-1)
        ).to(self.device)

        # -- Renderer --
        camera = FoVPerspectiveCameras(device=self.device)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=camera,
                lights=self.lights
            )
        )

        # -- Vision Transformer model (frozen) --
        self.ml_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(self.device).eval()
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        for param in self.ml_model.parameters():
            param.requires_grad = False

        # -- Optional transform (not currently used in forward pass) --
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.transforms = transforms.Compose([
            transforms.Lambda(
                lambda tensor: tensor[..., :3].permute(0, 3, 1, 2) if tensor.dim() == 4 else tensor.permute(2, 0, 1).unsqueeze(0)
            ),
            transforms.Normalize(mean, std),
            transforms.Resize((224, 224)),
        ])

        # For convenience if you want to quickly enable/disable
        # the lighting optimization, store a boolean:
        self.optimize_lighting = self.optimize_kwargs["lighting"]

        # -- Environment maps setup --
        self.use_envmap = envmap_paths is not None and len(envmap_paths) > 0
        if self.use_envmap:
            if isinstance(envmap_paths, str):
                envmap_paths = [envmap_paths]
            self.envmaps = torch.stack([self._load_envmap(path) for path in envmap_paths])
        self.n_envmaps = len(envmap_paths) or 1

    def update_scene_params(self, scene_params: Dict[str, torch.Tensor]) -> None:
        """
        Update the scene parameters with new values.
        """
        for param, value in scene_params.items():
            setattr(self.scene_params[param], 'data', value.to(self.device))

    def _get_random_camera_coords(self, fov: torch.Tensor = torch.tensor(60.0)) -> torch.Tensor:
        """
        Generates random camera coordinates (in Cartesian) for the batch based on
        a random spherical distribution, ensuring a suitable distance from the object
        (relative to bounding box size).
        """
        # Compute radius from bounding box size and FOV
        r = self.bbox_size.cpu() / (2.0 * np.random.uniform(0.5, 0.7)) / torch.tan(fov * (torch.pi / 180.0) / 2)

        # Random spherical angles theta, phi
        theta = 2 * np.pi * np.random.rand(self.batch_size)
        phi = np.arccos(2 * np.random.rand(self.batch_size) - 1)

        # Convert spherical to Cartesian
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        coords = np.stack([x, y, z], axis=1).astype(np.float32)
        return torch.tensor(coords, device=self.device)

    def _calculate_bbox_properties(self, min_max_proportion: Tuple[float, float]) -> None:
        """
        Calculate and store bounding box properties (center, size, min/max distance)
        for the loaded mesh.
        """
        verts = self.meshes.verts_packed()
        self.bbox_min = verts.min(dim=0)[0]
        self.bbox_max = verts.max(dim=0)[0]
        self.bbox_center = (self.bbox_min + self.bbox_max) / 2.0
        self.bbox_size = (self.bbox_max - self.bbox_min).max()

        # Min and max distances are derived from bounding box size
        self.min_distance = self.bbox_size / (2.0 * min_max_proportion[1])
        self.max_distance = self.bbox_size / (2.0 * min_max_proportion[0])

    def distance_penalty(self) -> torch.Tensor:
        """
        Returns a penalty term for camera distance if it goes out of [min_distance, max_distance].
        Encourages the camera to remain within a certain distance range from the bounding box center.
        """
        dist = torch.norm(self.scene_params['camera'] - self.bbox_center, dim=-1)
        penalty = (
            (dist < self.min_distance).float() * (self.min_distance - dist) ** 2 +
            (dist > self.max_distance).float() * (dist - self.max_distance) ** 2
        )
        return penalty.mean()

    def _cluster_texture(self, nb_clusters: int = 3) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Clusters the texture image into a predefined number of clusters (nb_clusters).
        Returns:
            - A dict with cluster indices for each cluster (as flattened indices).
            - A tensor of cluster centroid colors of shape [nb_clusters, 3].
        """
        # Extract the first texture from the batch for clustering
        texture_arr = self.texture_image[0].detach().cpu().numpy()

        # Mask out near-black pixels (treated as un-mapped or irrelevant)
        to_cluster_mask = (texture_arr.mean(axis=-1) > 1e-4)
        pixels_to_cluster = texture_arr[to_cluster_mask].reshape(-1, 3)

        # KMeans clustering
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
        Reconstruct the texture from the cluster centroids.
        Fills each cluster region with its centroid color for every image in the batch.
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

    def _constrain_position(self, position: torch.Tensor, min_distance: float, max_distance: float) -> torch.Tensor:
        """
        Clamps the position to be within [min_distance, max_distance]
        while preserving the direction from the bounding box center.
        """
        camera_vectors = position - self.bbox_center
        distances = torch.norm(camera_vectors, dim=-1, keepdim=True)
        normalized_vectors = camera_vectors / (distances + 1e-9)  # avoid /0

        clamped_distances = torch.clamp(distances, min_distance, max_distance)
        return self.bbox_center + normalized_vectors * clamped_distances

    def _constrain_camera(self) -> None:
        """
        Clamps the camera distance to be within [self.min_distance, self.max_distance]
        while preserving the direction from the bounding box center.
        """
        self.scene_params['camera'].data = self._constrain_position(
            self.scene_params['camera'].data,
            self.min_distance,
            self.max_distance
        )

    def _constrain_texture(self) -> None:
        """
        Ensures all texture centroid colors remain within [0, 1].
        """
        self.scene_params['texture_centroids'].data = torch.clamp(
            self.scene_params['texture_centroids'], 0.0, 1.0
        )

    def _constrain_lights(self) -> None:
        """
        Clamps the lighting location and intensity to prevent them from becoming extreme.
        """
        # Normalize the location vector to unit length
        self.scene_params['light_location'].data = self._constrain_position(
            self.scene_params['light_location'].data,
            self.min_distance,
            self.max_distance
        )

        # Keep light intensity in [0.5, 5]
        self.scene_params['light_intensity'].data = torch.clamp(
            self.scene_params['light_intensity'].data, 0.5, 5.0
        )

    def _load_envmap(self, path: str, gamma: float = 2.2, alpha: float = 1.0, target_size: Tuple[int, int] = (1024, 2048)) -> torch.Tensor:
        """
        Load and process environment map, resizing to a consistent size

        Args:
            path (str): Path to environment map
            gamma (float): Gamma correction value
            alpha (float): Scaling factor
            target_size (tuple): Desired (height, width) for resizing
        """
        # Read image
        im = iio.imread(path)[..., :3]

        # Convert to tensor and resize
        im_tensor = torch.tensor(im, device=self.device, dtype=torch.float32)
        im_resized = F.interpolate(
            im_tensor.permute(2, 0, 1).unsqueeze(0),  # Add batch dimension
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)

        # Apply gamma correction and alpha scaling
        return (im_resized ** (1/gamma)) * alpha

    def _get_background_rays(self, R: torch.Tensor) -> torch.Tensor:
        """Generate world-space ray directions for each pixel"""
        H = W = self.raster_settings.image_size
        tan = torch.tan(torch.tensor(60.0 * 3.14159 / 360.0))

        y, x = torch.meshgrid(
            torch.linspace(-tan, tan, H, device=self.device),
            torch.linspace(-tan * W/H, tan * W/H, W, device=self.device),
            indexing="ij"
        )

        dirs = torch.stack([x, y, torch.ones_like(x)], -1)
        return F.normalize(dirs @ R[0].T, dim=-1)

    def _sample_envmap(self, directions: torch.Tensor, envmap_idx: int = 0) -> torch.Tensor:
        """Sample environment map using ray directions"""
        x = directions[..., 0]
        y = directions[..., 1].clamp(-1 + 1e-6, 1 - 1e-6)
        z = directions[..., 2]

        # Convert to UV coordinates
        u = (-torch.atan2(x, z) / (2 * 3.14159) + 0.5) % 1.0
        v = (torch.asin(y) / 3.14159 + 0.5)

        # Convert to pixel coordinates
        H, W = self.envmaps.shape[1:3]
        u = (u * W).long() % W
        v = (v * H).long() % H

        return self.envmaps[envmap_idx, v, u]

    def _get_background_mask(self, meshes: torch.Tensor, R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Gets background mask using rasterizer fragments.
        Args:
            meshes: PyTorch3D meshes object
            R: Rotation matrices of shape (B, 3, 3)
            T: Translation vectors of shape (B, 3)
        Returns:
            background_mask: Boolean tensor of shape (B, H, W, 1) where True indicates background pixels
        """
        fragments = self.renderer.rasterizer(meshes_world=meshes, R=R, T=T)
        top_face_idx = fragments.pix_to_face[..., 0]  # shape (batch_size, H, W)
        background_mask = (top_face_idx < 0)[..., None]  # Add channel dimension
        return background_mask

    def render(self, with_grad: bool = True, image_res: Optional[int] = None) -> torch.Tensor:
        """
        Renders the object with multiple environment maps.

        Args:
            with_grad (bool): Whether to compute gradients during rendering
            image_res (int, optional): Override the default image resolution. 
                                       If None, uses the original image resolution from raster_settings.
        """
        # Use torch.set_grad_enabled(with_grad) context manager if gradients are not needed
        with torch.set_grad_enabled(with_grad):
            meshes = self.meshes.clone()

            if image_res is not None:
                # Create a temporary copy of raster settings with new image size
                raster_settings = RasterizationSettings(
                    **{**vars(self.raster_settings), 'image_size': image_res}
                )
            else:
                raster_settings = self.raster_settings

            # Constrain parameters
            if with_grad:
                self._constrain_camera()
                self._constrain_texture()
                if self.optimize_kwargs["lighting"]:
                    self._constrain_lights()

            # If we're optimizing texture centroids, fill the new texture:
            if with_grad and self.scene_params["texture_centroids"].requires_grad:
                # constrained or un-constrained texture
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
                    specular_color=self.light_color * self.scene_params['light_intensity'].unsqueeze(-1)
                ).to(self.device)
                if with_grad and self.optimize_kwargs["lighting"]
                else self.lights
            )

            # Compute view transforms
            up_vector = torch.tensor([[0.0, 1.0, 0.0]], device=self.device).expand(self.batch_size, -1)
            R, T = look_at_view_transform(
                eye=self.scene_params['camera'].to(self.device),
                at=self.bbox_center.to(self.device).unsqueeze(0).expand(self.batch_size, -1),
                up=up_vector
            )
            R, T = R.to(self.device), T.to(self.device)

            # Create a new renderer with the dynamic lights
            camera = FoVPerspectiveCameras(device=self.device)
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

            # Render
            base_images = renderer(meshes_world=meshes, R=R, T=T)[..., :3]

            if self.use_envmap:
                # Initialize output tensor for all environment maps
                images = torch.zeros(self.batch_size, self.n_envmaps, *base_images.shape[1:],
                                   device=self.device)

                # Get background mask using rasterizer fragments with camera parameters
                background = self._get_background_mask(meshes, R=R, T=T)

                # For each image in batch
                for i in range(self.batch_size):
                    rays = self._get_background_rays(R[i:i+1])

                    # For each environment map
                    for j in range(self.n_envmaps):
                        env_background = self._sample_envmap(rays, envmap_idx=j)
                        images[i, j] = torch.where(background[i], env_background, base_images[i])
            else:
                images = base_images.unsqueeze(1)  # Add envmap dimension for consistency

            # Gamma correction
            gamma = 2.2
            images = torch.clamp(images, 1e-6, 1.0) ** (1 / gamma)
            return images

    def forward(self, return_render: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Modified forward pass to handle multiple environment maps
        """
        images = self.render()  # (B, N_envmaps, H, W, C)
        images_orig = images.clone()

        # Reshape for batch processing through ViT
        B, N, H, W, C = images.shape
        images = images.reshape(B * N, H, W, C)
        images = images.permute(0, 3, 1, 2)  # (B*N, C, H, W)

        if images.shape[2:4] != (224, 224):
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalization
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

    def render_initial(self) -> torch.Tensor:
        """
        Render the mesh using its initial parameters (original texture, initial camera coords)
        without any optimization constraints.
        Returns:
            - If no envmaps: tensor of shape (B, H, W, C)
            - If envmaps: tensor of shape (B, N_envmaps, H, W, C)
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
                specular_color=self.light_color * self.init_light_intensity.detach().unsqueeze(-1)
            ).to(self.device)

            # Create a new renderer with initial lights
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
            up_vector = torch.tensor([[0.0, 1.0, 0.0]]).expand(self.batch_size, -1)

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

                # For each image in batch
                for i in range(self.batch_size):
                    rays = self._get_background_rays(R[i:i+1])

                    # For each environment map
                    for j in range(self.n_envmaps):
                        env_background = self._sample_envmap(rays, envmap_idx=j)
                        # Make sure background mask is properly broadcast
                        bg_mask = background[i].unsqueeze(0)  # Add channel dim if needed
                        images[i, j] = torch.where(bg_mask, env_background, base_images[i])
            else:
                images = base_images.unsqueeze(1)  # Add envmap dimension for consistency

            # Gamma correction
            gamma = 2.2
            images = torch.clamp(images, 1e-6, 1.0) ** (1 / gamma)
            return images


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
