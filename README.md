# Hallway Avatar Gen

Hallway Avatar Gen is a Blender 5.0 extension for importing See-through-style layered PSD avatars, converting visible layers into deformable meshes, and building a first-pass Stretchy-style rig inside Blender. The long-term goal is full 2.5-D avatar generation from See-through inputs, but the current release is focused on reliable PSD import, mesh generation, remeshing, and rig setup.

## Current scope

- PSD import via extension-local `psd_tools`, `Pillow`, and `vtracer` dependencies
- Recursive PSD traversal with hidden, empty, and effectively transparent layer skipping
- Fast traced alpha meshing adapted from the local Meshed Alpha workflow
- See-through-aware semantic classification and draw-order-preserving layer stacking
- Grounded mesh placement plus first-pass Stretchy-style armature generation and weighting
- Optional voxel-style binding and vendored Quad Remesher runtime support for imported sheets

## Current limitations

- Hallway is still positioned as an import-and-rig tool first, not a finished one-click 2.5-D avatar generator yet
- Gemini-assisted avatar generation and full See-through image parsing are still planned follow-up work
- Quad remeshing now uses the vendored `qmesh` runtime shipped inside this extension

## Usage

1. Add local dependency wheels to [wheels/README.md](/Users/tylerwalker/Library/Application%20Support/Blender/5.0/extensions/user_default/hallway_avatar_gen/wheels/README.md) or extracted packages to [vendor/README.md](/Users/tylerwalker/Library/Application%20Support/Blender/5.0/extensions/user_default/hallway_avatar_gen/vendor/README.md).
2. Open the add-on preferences or the `Hallway` sidebar panel and run `Install PSD Backend`.
3. Import a PSD with `Import PSD Avatar`.
4. Optionally remesh imported layers from the panel, then build the armature and bind weights.
5. Review the imported layers, skipped-layer reasons, and remesh/rig status in Blender.
