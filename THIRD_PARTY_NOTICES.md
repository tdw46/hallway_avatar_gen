# Third-Party Notices

## Local development references

- Blender Extension Template
  - Local reference path used during development only
  - Runtime code is not copied verbatim from the template, but the project layout and registration flow were adapted from it

- Import Meshed Alpha
  - Local reference add-on used to inform the planned alpha-to-mesh service layer
  - License in the referenced add-on manifest: `GPL-3.0-or-later`
  - This project currently uses a fallback internal mesh generator while the fuller tracing-based adaptation is still pending

## Optional runtime dependency

- `psd-tools`
  - Loaded only when the user provides bundled wheels or vendored packages inside the extension folder and explicitly triggers PSD backend installation
  - Used for PSD traversal and layer rasterization

- `Pillow`
  - Loaded alongside `psd-tools`
  - Used indirectly for raster image handling when PSD layers are exported to cached PNG files

## Vendored runtime assets

- Quad Remesher runtime bundle
  - Vendored under `resources/quad_remesher_engine`
  - Used locally by Hallway for FBX-based remeshing through `qmesh`

- Voxel Skinning binaries
  - Vendored under `resources/voxel_skinning`
  - Used locally by Hallway for voxel heat diffuse weight binding
