import os
import numpy as np
import torch
from PIL import Image
from pytorch3d.io.mtl_io import make_mesh_texture_atlas
from typing import Dict, List, Tuple

class MTLParser:
    @staticmethod
    def parse_mtl_file(mtl_path: str) -> Tuple[Dict, Dict]:
        """
        Parse MTL file and extract material properties and texture references.
        
        Returns:
            - material_properties: Dict of material name to properties
            - texture_references: Dict of material name to texture file paths
        """
        material_properties = {}
        texture_references = {}
        current_material = None

        with open(mtl_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue

                if tokens[0] == 'newmtl':
                    current_material = tokens[1]
                    material_properties[current_material] = {}
                
                elif tokens[0] == 'Kd' and len(tokens) == 4:  # Diffuse color
                    material_properties[current_material]['diffuse_color'] = torch.tensor(
                        [float(x) for x in tokens[1:4]], dtype=torch.float32
                    )
                
                elif tokens[0] == 'map_Kd':  # Texture map
                    texture_path = ' '.join(tokens[1:])
                    texture_references[current_material] = texture_path

        return material_properties, texture_references

    @staticmethod
    def load_texture_images(
        texture_references: Dict, 
        base_path: str
    ) -> Dict[str, torch.Tensor]:
        """
        Load texture images for each material.
        
        Args:
            texture_references: Dict of material to texture file paths
            base_path: Base directory for relative texture paths
        
        Returns:
            Dict of material names to texture tensors
        """
        texture_images = {}
        for material, tex_path in texture_references.items():
            # Construct full path, handling both absolute and relative paths
            full_path = tex_path if os.path.isabs(tex_path) else os.path.join(base_path, tex_path)
            
            try:
                # Load image and convert to tensor
                image = Image.open(full_path)
                image_tensor = torch.from_numpy(np.array(image) / 255.0).float()
                texture_images[material] = image_tensor
            except Exception as e:
                print(f"Could not load texture for {material}: {e}")
        
        return texture_images

def prepare_texture_atlas(
    obj_path: str, 
    mtl_path: str, 
    faces_uvs: torch.Tensor, 
    verts_uvs: torch.Tensor,
    face_material_names: np.ndarray,
    texture_size: int = 256
) -> torch.Tensor:
    """
    High-level function to prepare texture atlas from MTL and OBJ files.
    
    Args:
        obj_path: Path to .obj file
        mtl_path: Path to .mtl file
        faces_uvs: Face UV indices
        verts_uvs: Vertex UV coordinates
        face_material_names: Material names for each face
        texture_size: Size of each face's texture map
    
    Returns:
        Texture atlas tensor
    """
    # Parse MTL file
    material_properties, texture_references = MTLParser.parse_mtl_file(mtl_path)
    
    # Load texture images (using obj directory as base path)
    base_path = os.path.dirname(obj_path)
    texture_images = MTLParser.load_texture_images(texture_references, base_path)
    
    # Ensure face_material_names is a numpy array
    if face_material_names is None:
        # If no material names provided, create default array with single material
        face_material_names = np.array(['default'] * len(faces_uvs))
    elif not isinstance(face_material_names, np.ndarray):
        face_material_names = np.array(face_material_names)
    
    # Create texture atlas
    atlas = make_mesh_texture_atlas(
        material_properties=material_properties,
        texture_images=texture_images,
        face_material_names=face_material_names,
        faces_uvs=faces_uvs,
        verts_uvs=verts_uvs,
        texture_size=texture_size,
        texture_wrap='repeat'
    )
    
    return atlas 