# Changelog

## 0.1.0 - 2026-04-13

- Created the Blender extension shell using the local template-style module layout
- Added PSD backend installation support from add-on preferences
- Reworked PSD dependency handling to use extension-local bundled wheels or vendored packages instead of `pip`
- Implemented recursive PSD traversal with empty-layer filtering and cached layer export
- Added traced alpha-mesh import, layer grounding, and draw-order-preserving layer stacking
- Added first-pass Stretchy-style autorigging, weighting, and voxel-binding integration
- Replaced the remesh runtime with a vendored QRemeshify runtime and vendored voxel-binding binaries
- Repositioned the extension publicly as `Hallway Avatar Gen`, with See-through-based 2.5-D generation marked as upcoming work
