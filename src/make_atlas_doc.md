# Documentation: Using `make_atlas.py` for Texture Atlas Generation

## Introduction

The `make_atlas.py` script merges multiple textures referenced by a Wavefront OBJ (`.obj`) file and its corresponding Material Template Library (`.mtl`) file into a single texture atlas. It rewrites the OBJ and MTL files to reference this unified texture.

## Purpose and Benefits

- **Texture Optimization:** Combining textures into a single image reduces GPU texture bindings, enhancing rendering efficiency.
- **Simplified Asset Management:** A unified texture atlas simplifies asset handling, beneficial in applications such as game development and 3D visualization.
- **Compatibility:** The script maintains OBJ and MTL compatibility, facilitating easy integration into existing workflows.


## File Structure

Ensure you have:

- An `.obj` file containing object geometry (`v`, `vt`, `vn`) and face definitions (`f`).
- An associated `.mtl` file referencing material textures via `map_Kd` properties.
- Texture image files referenced in the `.mtl` file (e.g., PNG, JPG).

Typical structure:

```
object/
├── model.obj
├── model.mtl
└── textures/
    ├── wood.png
    ├── metal.png
    └── glass.png
```

## Using the Script

### Step-by-Step Guide

1. **Preparation**:

   Place your OBJ and MTL files clearly specifying paths to the associated textures. Ensure texture paths in the `.mtl` file are relative to the `.mtl` file location or absolute.

2. **Running the Script**:

   Execute the script from your terminal:

   ```bash
   python make_atlas.py <input.obj> <input.mtl> <output.obj> <output.mtl> <atlas.png>
   ```

   **Example:**

   ```bash
   python make_atlas.py models/model.obj models/model.mtl output/combined.obj output/combined.mtl output/atlas.png
   ```

3. **Output**:

   After execution, you'll have:

   - `combined.obj`: The rewritten OBJ file referencing a single material.
   - `combined.mtl`: An MTL file referencing the new texture atlas.
   - `atlas.png`: The generated texture atlas combining all input textures horizontally.

### Outputs

- **Texture Atlas (`atlas.png`)**:
  - Contains horizontally arranged textures used by materials from the original MTL.

- **Rewritten MTL (`combined.mtl`)**:
  - Contains a single material (`AtlasMaterial`) referencing the texture atlas.

- **Rewritten OBJ (`combined.obj`)**:
  - Contains updated texture coordinates (`vt`) recalculated to map correctly onto the texture atlas.

## Advanced Usage Tips

- **Multiple Objects and Groups**:
  The script correctly handles `.obj` files with multiple object groups or material definitions (`usemtl`). Each original material's texture is placed sequentially in the atlas, and UV coordinates are adjusted automatically.

- **Handling Missing Textures**:
  If a material doesn't have a texture (`map_Kd`), the script generates a 1x1 pixel white fallback texture.

## Best Practices

- Always backup original OBJ and MTL files before processing.
- Use descriptive names for output files to avoid confusion.
- Review the generated atlas visually to ensure textures are correctly mapped and combined.

## Troubleshooting

- **Textures Not Loading**:
  Ensure paths in the `.mtl` file are correct relative paths.

- **Incorrect UV Mapping**:
  Verify original `.obj` file correctness. UV remapping depends on initial UVs being accurate.

