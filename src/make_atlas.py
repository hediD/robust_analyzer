#!/usr/bin/env python3

import sys
import os
from collections import OrderedDict
from PIL import Image

def parse_mtl(mtl_path):
    """Parse .mtl file to extract all material properties."""
    materials = OrderedDict()
    current_mtl = None

    with open(mtl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('newmtl '):
                current_mtl = line.split(None, 1)[1]
                materials[current_mtl] = {'map_Kd': None}

            elif current_mtl is not None and ' ' in line:
                key, value = line.split(' ', 1)
                materials[current_mtl][key] = value

    return materials


def parse_obj(obj_path):
    """
    Parse .obj file to extract vertex, normal, texture coordinates, and face data.
    Also collects material associations and object/group information.
    """
    vertices = []
    normals = []
    texcoords = []
    faces_per_material = OrderedDict()
    original_mtllib_lines = []
    object_groups = []

    vertex_objects = {}
    vertex_materials = {}

    current_material = None
    current_object = "default"
    current_group = "default"

    with open(obj_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            if stripped.startswith('mtllib '):
                original_mtllib_lines.append(stripped)

            elif stripped.startswith('o '):
                current_object = stripped.split(None, 1)[1]
                object_groups.append(('o', current_object))
                
            elif stripped.startswith('g '):
                current_group = stripped.split(None, 1)[1]
                object_groups.append(('g', current_group))

            elif stripped.startswith('usemtl '):
                current_material = stripped.split(None, 1)[1]
                if current_material not in faces_per_material:
                    faces_per_material[current_material] = []

            elif stripped.startswith('v '):
                vertex_index = len(vertices) + 1  # 1-based indexing in OBJ
                vertex_objects[vertex_index] = current_object
                vertex_materials[vertex_index] = current_material
                vertices.append(stripped)

            elif stripped.startswith('vn '):
                normals.append(stripped)

            elif stripped.startswith('vt '):
                texcoords.append(stripped)

            elif stripped.startswith('f '):
                if current_material is None:
                    current_material = "base"
                    if current_material not in faces_per_material:
                        faces_per_material[current_material] = []
                faces_per_material[current_material].append(stripped)

    return vertices, normals, texcoords, faces_per_material, original_mtllib_lines, object_groups, vertex_objects, vertex_materials


def load_textures(materials):
    """
    Load textures for all materials with appropriate fallback handling.
    Returns a list of tuples (material_name, image, is_fallback).
    """
    result = []
    max_width = 1
    max_height = 1
    
    # First pass: load images and find maximum dimensions
    for mat_name, props in materials.items():
        tex_path = props.get('map_Kd')
        if tex_path and os.path.isfile(tex_path):
            img = Image.open(tex_path).convert('RGBA')
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
            result.append((mat_name, img, False))
        else:
            try:
                kd = props.get('Kd', '0.5 0.5 0.5').split()
                color = (
                    int(float(kd[0]) * 255),
                    int(float(kd[1]) * 255),
                    int(float(kd[2]) * 255),
                    255
                )
            except (IndexError, ValueError):
                color = (128, 128, 128, 255)
                
            img = Image.new('RGBA', (16, 16), color)
            result.append((mat_name, img, True))
    
    # Second pass: resize textures appropriately
    final_result = []
    for i, (mat_name, img, is_fallback) in enumerate(result):
        if not is_fallback and (img.width < max_width or img.height < max_height):
            img = img.resize((max_width, max_height), Image.LANCZOS)
        
        final_result.append((mat_name, img, is_fallback))
    
    return final_result


def pack_textures_horizontal(mat_images, padding=5, fallback_size_factor=0.1):
    """
    Pack textures horizontally with smaller regions for fallback textures.
    """
    main_texture = next((img for _, img, is_fallback in mat_images if not is_fallback), None)
    if not main_texture:
        main_texture = mat_images[0][1]
    
    main_width = main_texture.width
    main_height = main_texture.height
    
    fallback_width = max(int(main_width * fallback_size_factor), 32)
    
    total_width = sum(
        main_width if not is_fallback else fallback_width 
        for _, _, is_fallback in mat_images
    )
    total_width += padding * (len(mat_images) - 1)
    
    atlas = Image.new('RGBA', (total_width, main_height))

    offsets_and_scales = {}
    current_x = 0
    
    for (mat_name, img, is_fallback) in mat_images:
        width = fallback_width if is_fallback else main_width
        
        if is_fallback or img.width != width or img.height != main_height:
            img = img.resize((width, main_height), Image.LANCZOS)
        
        flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        atlas.paste(flipped_img, (current_x, 0))

        offset_u = current_x / float(total_width)
        offset_v = 0.0
        scale_u = width / float(total_width)
        scale_v = 1.0

        offsets_and_scales[mat_name] = (offset_u, offset_v, scale_u, scale_v)
        current_x += width + padding

    return atlas, offsets_and_scales


def write_atlas_mtl(mtl_out_path, materials, atlas_texture_name="atlas.png"):
    """Generate MTL file for the atlas texture with appropriate material properties."""
    base_material = None
    for mat_name, props in materials.items():
        if 'map_Kd' not in props or props['map_Kd'] is None:
            base_material = props
            break
    
    with open(mtl_out_path, 'w', encoding='utf-8') as f:
        f.write("newmtl AtlasMaterial\n")
        f.write(f"map_Kd {atlas_texture_name}\n")
        
        if base_material:
            for key, value in base_material.items():
                if key != 'map_Kd':
                    f.write(f"{key} {value}\n")
        else:
            f.write("Kd 1.000 1.000 1.000\n")
            f.write("Ka 0.000 0.000 0.000\n")
            f.write("Ks 0.000 0.000 0.000\n")


def generate_atlas_obj(obj_in_path, obj_out_path, atlas_offsets, atlas_mtl_filename, unified_material="AtlasMaterial"):
    """
    Generate an OBJ file using the atlas texture, remapping all UV coordinates.
    Preserves original geometry while unifying all materials.
    """
    vertices, normals, old_texcoords, faces_per_material, original_mtllib_lines, object_groups, vertex_objects, vertex_materials = parse_obj(obj_in_path)

    old_vt_list = []
    for line in old_texcoords:
        parts = line.split()
        u = float(parts[1])
        v = float(parts[2])
        old_vt_list.append((u, v))

    new_vt_coords = []
    new_faces = []

    with open(obj_out_path, 'w', encoding='utf-8') as out_f:
        # Write basic header
        out_f.write(f"mtllib {atlas_mtl_filename}\n")
        
        # Write vertices without comments
        for v_line in vertices:
            out_f.write(f"{v_line}\n")
            
        # Write normals without comments
        for vn_line in normals:
            out_f.write(f"{vn_line}\n")

        # Single material declaration
        out_f.write(f"usemtl {unified_material}\n")

        # Process faces
        for mat_name, face_lines in faces_per_material.items():
            if mat_name not in atlas_offsets:
                offset_u, offset_v, scale_u, scale_v = (0, 0, 1, 1)
            else:
                offset_u, offset_v, scale_u, scale_v = atlas_offsets[mat_name]

            for face_line in face_lines:
                tokens = face_line.split()[1:]
                new_face_tokens = []

                for tok in tokens:
                    parts = tok.split('/')

                    if len(parts) == 1:
                        v_idx = parts[0]
                        vt_idx = None
                        vn_idx = None
                    elif len(parts) == 2:
                        v_idx, vt_idx = parts
                        vn_idx = None
                    else:
                        v_idx, vt_idx, vn_idx = parts

                    v_idx_int = int(v_idx) if v_idx else None
                    vt_idx_int = int(vt_idx) if (vt_idx and vt_idx != '') else None
                    vn_idx_int = int(vn_idx) if (vn_idx and vn_idx != '') else None

                    if vt_idx_int is not None:
                        old_u, old_v = old_vt_list[vt_idx_int - 1]

                        new_u = offset_u + old_u * scale_u
                        new_v = offset_v + (1.0 - old_v) * scale_v  # Invert V coordinate

                        new_vt_coords.append((new_u, new_v))
                        new_vt_id = len(new_vt_coords)
                    else:
                        new_vt_id = None

                    if new_vt_id is not None and vn_idx_int is not None:
                        new_tok = f"{v_idx_int}/{new_vt_id}/{vn_idx_int}"
                    elif new_vt_id is not None and vn_idx_int is None:
                        new_tok = f"{v_idx_int}/{new_vt_id}"
                    else:
                        if vn_idx_int is not None:
                            new_tok = f"{v_idx_int}//{vn_idx_int}"
                        else:
                            new_tok = f"{v_idx_int}"

                    new_face_tokens.append(new_tok)

                # Save face without comments
                new_faces.append(f"f {' '.join(new_face_tokens)}")

        # Write texture coordinates
        for (u, v) in new_vt_coords:
            out_f.write(f"vt {u} {v}\n")

        # Write faces without comments
        for f_line in new_faces:
            out_f.write(f"{f_line}\n")


def main():
    """Create a texture atlas from the given OBJ/MTL files."""
    if len(sys.argv) < 6:
        print("Usage: python make_atlas.py <input.obj> <input.mtl> <output.obj> <output.mtl> <atlas.png>")
        sys.exit(1)

    input_obj_path = sys.argv[1]
    input_mtl_path = sys.argv[2]
    output_obj_path = sys.argv[3]
    output_mtl_path = sys.argv[4]
    output_atlas_png = sys.argv[5]

    materials = parse_mtl(input_mtl_path)
    mat_images = load_textures(materials)
    atlas, offsets_scales = pack_textures_horizontal(mat_images)

    atlas.save(output_atlas_png)
    print(f"Saved atlas to {output_atlas_png}")

    write_atlas_mtl(output_mtl_path, materials, atlas_texture_name=os.path.basename(output_atlas_png))
    print(f"Saved new MTL to {output_mtl_path}")

    generate_atlas_obj(
        obj_in_path=input_obj_path,
        obj_out_path=output_obj_path,
        atlas_offsets=offsets_scales,
        atlas_mtl_filename=os.path.basename(output_mtl_path),
        unified_material="AtlasMaterial"
    )
    print(f"Saved new OBJ to {output_obj_path}")


if __name__ == "__main__":
    main()
