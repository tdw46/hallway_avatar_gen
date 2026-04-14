# Blender Avatar AutoRig Extension Plan

## Codex Directive
Use this file as the living implementation plan for the Blender extension described here. Follow and adapt these rules while you work:

- Build the Blender extension described in this plan.
- Follow the local Blender Extension Template conventions first.
- Reuse/adapt the local Import Meshed Alpha code for alpha-to-mesh generation.
- Add **PSD layer import support** as a first-class feature.
- Add **special handling to ignore empty PSD layers** during import.
- Assume imported PSD files may come from **See-through** outputs.
- Use **See-through naming conventions** where practical to classify imported PSD layers for rigging.
- Keep the extension **heuristic-first** so the core pipeline works without ML downloads.
- Keep ML / inference optional and installable from add-on preferences.
- Support Windows, Linux, and macOS Apple Silicon. For optional inference, plan for Windows/Linux CUDA and Apple Metal/MPS.
- Let this document be edited, extended, and checked off as implementation proceeds.

---

## Source Repositories and Inputs

### Local authoritative inputs to inspect first
- Blender Extension Template:
  - `/Users/tylerwalker/Library/CloudStorage/GoogleDrive-tylerdillonwalker@gmail.com/My Drive/_ai_guidance/Blender Extenstion Template`
- Import Meshed Alpha add-on source:
  - `/Users/tylerwalker/Downloads/add-on-import-meshed-alpha-v1.0.2-macos-arm64`

### External source repos to inspect and borrow ideas from
- Stretchy Studio source code:
  - `https://github.com/MangoLion/stretchystudio`
- See-through source code:
  - `https://github.com/shitagaki-lab/see-through`
- Optional naming / output reference wrapper for See-through:
  - `https://github.com/jtydhr88/ComfyUI-See-through`

> Important: the local paths above are the primary development inputs. Do not assume their internal structure until audited.

---

## Status
- [ ] Phase 0 complete
- [ ] Phase 1 complete
- [ ] Phase 2 complete
- [ ] Phase 3 complete
- [ ] Phase 4 complete
- [ ] Phase 5 complete
- [ ] Phase 6 complete
- [ ] Phase 7 complete
- [ ] Phase 8 complete
- [ ] Phase 9 complete
- [ ] Phase 10 complete
- [ ] MVP shipped

## Implementation Notes
- [x] 2026-04-13 initial offline vertical slice created in this extension workspace
- [x] Product name retained as `Hallway Avatar Gen`
- [x] Public positioning updated so the current release is import-first: See-through PSD layer import now, 2.5-D avatar generation later
- [x] Extension shell, preferences, diagnostics, scene state, and UI panel scaffolded
- [x] PSD import path implemented with extension-local bundled wheel / vendored dependency support
- [x] Empty/transparent/hidden PSD layers are skipped with per-layer reasons recorded in scene state
- [x] Imported visible PSD layers are cached as PNGs and converted into traced alpha meshes aligned to the PSD canvas
- [x] See-through-aware name parsing plus geometry fallback classification implemented
- [x] Heuristic armature generation, grounding, and first-pass binding implemented
- [x] Hallway remesh flow migrated to a vendored Quad Remesher `qmesh` runtime inside the extension
- [ ] Blender runtime validation still needed
- [x] Import Meshed Alpha tracing path adapted into the Hallway PSD import pipeline

---

## Objective
Build a Blender extension that:
1. Imports layered avatar images from PNGs and PSDs.
2. Converts each layer's non-transparent pixels into a deformable mesh.
3. Ignores PSD layers that are empty, fully transparent, clipped away, hidden by rule, or semantically unusable.
4. Infers a usable 2D/2.5D character rig from those layer meshes.
5. Generates Blender armatures, constraints, and initial weights.
6. Optionally uses local inference for pose/keypoint initialization when heuristics are insufficient.
7. Supports Windows, Linux, and macOS Apple Silicon.
8. Manages optional dependencies/models from the add-on preferences UI.

This extension should be implemented **inside the structure and conventions of the local Blender Extension Template**.

It should also reuse/adapt the local **Import Meshed Alpha** codebase for the alpha-to-mesh stage.

It should borrow workflow and rigging ideas from **Stretchy Studio** and use **See-through** PSD naming conventions where helpful for automatic classification.

---

## Ground Rules
- [ ] Preserve the template's existing architectural style unless there is a concrete blocker.
- [ ] Prefer a **heuristic-first** MVP. The core rigging flow must work without any ML downloads.
- [ ] Make ML/inference optional and additive.
- [ ] Keep the generated data Blender-native: meshes, armatures, empties, constraints, vertex groups, custom properties.
- [ ] Avoid any network usage unless the user explicitly triggers dependency/model download from preferences.
- [ ] Cache downloaded models/wheels outside project source, in a deterministic per-user add-on cache directory.
- [ ] Every major pipeline stage must be invokable independently for debugging:
  - import layers
  - import PSD
  - flatten PSD layers to images
  - build alpha meshes
  - classify parts
  - estimate skeleton
  - build armature
  - bind weights
  - run optional ML assist
- [ ] Every stage must emit structured debug logs.
- [ ] PSD support must be a core path, not an afterthought.
- [ ] Empty PSD layers must never create empty Blender objects.
- [ ] See-through-compatible name parsing should happen before geometry-only heuristics.

---

## Product Scope

### In scope
- PNG layer import from a folder or batch selection.
- PSD layer import from a single PSD file or batch PSD import.
- Empty PSD layer detection and skipping.
- Alpha-to-mesh conversion using the existing local Import Meshed Alpha code as the starting point.
- Layer classification by filename, PSD layer name, and geometry.
- See-through-aware semantic label mapping.
- Heuristic joint placement from mesh bounds / centroids / overlap relationships.
- Blender armature creation for torso, head, arms, legs, and optional hair chains.
- Automatic parenting and first-pass weights.
- Optional pose-assist backends for difficult cases.
- Preferences UI for dependency install, model download, cache management, provider selection, and diagnostics.

### Out of scope for MVP
- Full Stretchy Studio-style direct mesh deformation editor.
- Full timeline animation toolset.
- Cloud inference.
- Commercial extension store packaging.
- Guaranteeing perfect deformation on highly stylized or non-humanoid art.

---

## Why this architecture
The target workflow matches the strongest parts of the tools already found:
1. **Import Meshed Alpha** already solves the critical alpha-driven geometry generation step.
2. **Stretchy Studio** shows the exact adjacent workflow we care about: decomposed PSD input, auto-rigging, heuristic fallback, and mesh-aware character setup.
3. **See-through** provides a realistic upstream PSD source for semantically decomposed avatar art.
4. See-through-following PSD naming is much more useful than pure geometry guesses when classifying limbs, clothing, hair, and facial parts.

For this add-on, the Blender-native plan is:
1. Import PNG layers or PSD layers.
2. Normalize layers into a common internal `LayerPart` representation.
3. Skip empty PSD layers aggressively.
4. Turn each usable layer into real mesh geometry.
5. Infer semantic body regions from names + layout.
6. Generate armature + weights.
7. Use optional inference only when heuristic confidence is low.

---

## See-through Naming Strategy

### Canonical implementation rule
Treat See-through-style names as the **first-pass semantic parser** for PSD imports.

### Working semantic tags to support
These names should be recognized directly or through aliases if they appear in PSD layer names:

#### Body / outfit tags
- [ ] `front hair`
- [ ] `back hair`
- [ ] `neck`
- [ ] `topwear`
- [ ] `handwear`
- [ ] `bottomwear`
- [ ] `legwear`
- [ ] `footwear`
- [ ] `tail`
- [ ] `wings`
- [ ] `objects`

#### Head / face tags
- [ ] `headwear`
- [ ] `face`
- [ ] `irides`
- [ ] `eyebrow`
- [ ] `eyewhite`
- [ ] `eyelash`
- [ ] `eyewear`
- [ ] `ears`
- [ ] `earwear`
- [ ] `nose`
- [ ] `mouth`

### Alias normalization rules
- [ ] Normalize spaces, underscores, and hyphens to a canonical internal form.
- [ ] Support both raw See-through names and Blender/artist variants like:
  - `front_hair`, `hair_front`, `bangs`
  - `back_hair`, `hair_back`
  - `topwear`, `body`, `torso`, `shirt`, `coat`
  - `handwear`, `arms`, `hands`, `sleeves`
  - `bottomwear`, `hips`, `pelvis`, `skirt`, `pants`
  - `legwear`, `legs`
  - `footwear`, `feet`, `shoes`
- [ ] Support See-through split suffixes when present, such as left/right or numbered fragments.
- [ ] Maintain the original PSD layer name as metadata even after normalization.

### Classification priority order
1. **Explicit See-through-compatible layer name match**
2. **Alias dictionary match**
3. **PSD group/folder context**
4. **Geometry + symmetry heuristics**
5. **Manual user correction**

---

## PSD Import Requirements

### Functional requirements
- [ ] Import a `.psd` file and enumerate all layers recursively.
- [ ] Support nested groups/folders.
- [ ] Preserve layer order and draw order metadata.
- [ ] Preserve original layer names.
- [ ] Preserve per-layer offsets / canvas placement.
- [ ] Support PSDs generated by See-through as a primary test case.
- [ ] Convert raster PSD layers to temporary RGBA buffers for alpha-mesh generation.
- [ ] Support optional import of PNG folders and PSDs through the same unified pipeline.

### Empty PSD layer handling
A PSD layer should be skipped if any of the following are true:
- [ ] zero width or zero height
- [ ] no pixel data
- [ ] fully transparent after rasterization
- [ ] blank mask result after combining layer alpha and mask
- [ ] hidden layer and the import settings say to ignore hidden layers
- [ ] clipped-away result produces no visible pixels

### Empty PSD logging
- [ ] Keep a skipped-layer report in memory and optionally save it to the debug log.
- [ ] Report skipped layers by name and reason.
- [ ] Never fail the whole import just because some PSD layers are empty.

### Suggested implementation approach
- [ ] Add a PSD parsing backend behind an optional dependency installed from preferences if it is not safe to vendor.
- [ ] Prefer a mature Python PSD parser that can rasterize layers in Blender's Python environment.
- [ ] Convert PSD layers into temporary RGBA images in a cache directory or memory buffer, then route them through the same alpha-mesh adapter used for PNGs.
- [ ] Keep PSD import and PNG import sharing as much downstream code as possible.

---

## High-Level Architecture

### Proposed modules
- [ ] `__init__.py`
- [ ] `manifest` / extension metadata files required by the template
- [ ] `preferences.py`
- [ ] `operators/import_layers.py`
- [ ] `operators/import_psd.py`
- [ ] `operators/generate_meshes.py`
- [ ] `operators/classify_parts.py`
- [ ] `operators/estimate_rig.py`
- [ ] `operators/build_armature.py`
- [ ] `operators/bind_weights.py`
- [ ] `operators/run_pipeline.py`
- [ ] `operators/install_dependencies.py`
- [ ] `operators/download_models.py`
- [ ] `ui/panels.py`
- [ ] `core/layer_io.py`
- [ ] `core/psd_io.py`
- [ ] `core/psd_layer_filters.py`
- [ ] `core/alpha_mesh_adapter.py`
- [ ] `core/mesh_cleanup.py`
- [ ] `core/part_classifier.py`
- [ ] `core/seethrough_naming.py`
- [ ] `core/rig_schema.py`
- [ ] `core/heuristic_rigger.py`
- [ ] `core/armature_builder.py`
- [ ] `core/weighting.py`
- [ ] `core/confidence.py`
- [ ] `core/debug_draw.py`
- [ ] `ml/provider_base.py`
- [ ] `ml/provider_onnx.py`
- [ ] `ml/provider_torch.py`
- [ ] `ml/model_registry.py`
- [ ] `ml/dwpose_adapter.py`
- [ ] `utils/env.py`
- [ ] `utils/paths.py`
- [ ] `utils/logging.py`
- [ ] `utils/subprocess_install.py`
- [ ] `tests/` or template-equivalent test area

### Data model
- [ ] Define a `LayerPart` structure:
  - source filepath
  - source type (`png`, `psd`)
  - psd document path if applicable
  - psd layer path / hierarchy path
  - layer name
  - normalized semantic token
  - imported object name
  - image size
  - canvas offset
  - alpha bbox
  - centroid
  - area
  - perimeter
  - side guess (`L`, `R`, `C`, `UNKNOWN`)
  - semantic label (`head`, `hair_front`, `upper_arm_l`, etc.)
  - parent semantic label
  - confidence
  - skipped flag
  - skip reason
- [ ] Define a `RigPlan` structure:
  - joint names
  - 2D joint positions in image/local plane space
  - bone parent map
  - layer-to-bone binding hints
  - weight rules
  - optional deform-chain rules
  - overall confidence

---

## Phase 0 — Audit the local and external inputs first

### Template audit
- [x] Inspect the local Blender Extension Template path.
- [x] Record its required packaging layout, registration pattern, naming conventions, preferences pattern, operator naming, panel layout, and build/release scripts.
- [x] Mirror those conventions in this project instead of inventing a parallel architecture.
- [ ] Identify where third-party dependency bootstrap logic is expected to live.
- [ ] Identify whether the template already has:
  - logging helpers
  - a preferences panel base class
  - environment bootstrap code
  - update/install helpers
  - tests
  - CI hooks

### Import Meshed Alpha audit
- [x] Inspect the local Import Meshed Alpha add-on path.
- [x] Identify the exact import operator, mesh generation entrypoints, properties, and helper modules.
- [ ] Determine whether the code is easier to:
  - vendor as-is into a submodule, or
  - adapt into a thin wrapper, or
  - extract into a reusable internal service layer.
- [ ] Keep a change log of every modification made to upstream behavior.

### Stretchy Studio audit
- [ ] Inspect the Stretchy Studio repo structure.
- [ ] Identify the rigging wizard concepts, mesh generation assumptions, and heuristic vs AI rigging split.
- [ ] Extract only the ideas and algorithms that translate cleanly into Blender.
- [ ] Do **not** attempt a direct code-port unless the repo licensing and architecture make that sensible.
- [ ] Write down the exact files or modules worth reviewing for rigging heuristics.

### See-through audit
- [ ] Inspect the See-through repo structure.
- [ ] Find the semantic tag definitions and any stratification / left-right split logic.
- [ ] Identify where PSD output tags and naming rules are defined.
- [ ] Record which naming conventions are safe to adopt directly in the Blender add-on.
- [ ] Check whether depth-based or left-right stratification logic is simple enough to replicate heuristically in Blender.

### Licensing audit
- [ ] Confirm licenses of all borrowed code and models before shipping.
- [x] Because the local alpha-mesh code is being reused from a GPL extension, treat this project as requiring GPL-compatible distribution unless proven otherwise.
- [x] Keep `THIRD_PARTY_NOTICES.md` from day one.
- [ ] Record the license status of Stretchy Studio-derived ideas/code and See-through-derived ideas/code separately.

**Exit criteria**
- [ ] Template conventions documented.
- [ ] Alpha mesh code entrypoints documented.
- [ ] Stretchy Studio audit notes written.
- [ ] See-through naming/tag audit written.
- [ ] Licensing decision written down.

---

## Phase 1 — Stand up the extension shell
- [x] Create the extension package following the local template.
- [x] Add a top-level product name placeholder, e.g. `Avatar AutoRig`.
- [x] Register minimal preferences, one panel, and one operator.
- [x] Add a diagnostics section in preferences showing:
  - Blender version
  - Python version
  - OS
  - add-on root
  - cache directory
  - dependency status
  - model status
  - PSD backend status
- [x] Add a basic logger with levels:
  - INFO
  - WARN
  - ERROR
  - DEBUG
- [x] Add a scene-level property group for pipeline state.

**Exit criteria**
- [ ] Add-on installs and enables cleanly.
- [ ] Preferences panel opens without errors.
- [ ] Diagnostic panel reports environment correctly.

---

## Phase 2 — Layer ingestion for PNG and PSD

### Goal
Support both folder-based PNG import and PSD import into one unified layer model.

### Tasks
- [ ] Implement a unified import entrypoint that accepts either:
  - a folder of PNGs
  - one PSD file
  - multiple PSD files
- [x] Build a PSD traversal routine for nested groups/folders.
- [x] Rasterize PSD layers into RGBA buffers or temp images.
- [x] Preserve canvas placement so all layers align in Blender.
- [x] Track source metadata for every imported layer.
- [ ] Add options:
  - ignore hidden PSD layers
  - ignore empty PSD layers
  - import groups as collections metadata only
  - flatten blend modes if needed
  - preserve original layer order
- [x] Create a skipped-layer diagnostics list for PSD imports.
- [x] Add a debug panel showing imported, skipped, and classified layers.

### Empty PSD detection helpers
- [x] Add a cheap pre-check using bounds and visibility before rasterization.
- [x] Add a post-raster check for any non-zero alpha.
- [x] Add a minimum visible-pixel threshold setting to reject accidental noise layers.
- [x] Add a setting to optionally keep tiny layers like pupils if clearly labeled.

**Exit criteria**
- [ ] PNG folder import works.
- [ ] PSD import works.
- [ ] Empty PSD layers are skipped reliably.
- [ ] Imported layers preserve draw order and positions.

---

## Phase 3 — Wrap alpha mesh generation as a service

### Goal
Turn the local Import Meshed Alpha behavior into a reusable internal function that can be called on many imported layer images programmatically.

### Tasks
- [ ] Extract or wrap the alpha-mesh operator so it can run headlessly from a batch pipeline.
- [ ] Ensure a deterministic output naming scheme:
  - image object
  - mesh object
  - material
  - collection placement
- [ ] Preserve UVs and source image material hookup.
- [ ] Add config settings for:
  - merge distance / cleanup tolerance
  - subdivision / triangulation options if available
  - per-layer scale
  - origin placement
  - collection naming
- [ ] Add post-import cleanup helpers:
  - remove doubles / merge by distance
  - limited dissolve where safe
  - recalc normals if needed
  - apply transforms consistently
  - place all meshes on a common rig plane
- [ ] Normalize coordinate system so all generated layers share one consistent 2D reference space.
- [ ] Ensure PSD-derived temporary images and folder PNGs use the same alpha-mesh path.

### Nice-to-have fallback
- [ ] If the local alpha-mesh code cannot be cleanly reused in batch form, add a fallback internal tracing path inspired by marching-squares style image-to-mesh workflows.

**Exit criteria**
- [ ] Folder of separated PNGs imports as aligned meshed layers in one click.
- [ ] PSD-derived layers import as aligned meshed layers in one click.
- [ ] Re-running the import is idempotent or safely replaceable.

---

## Phase 4 — Semantic classification with See-through-aware parsing

### Inputs expected
- front hair / back hair
- face / head / neck
- topwear / torso / body
- handwear / arm fragments / sleeves / hands
- bottomwear / pelvis / hips
- legwear / legs
- footwear / feet
- optional eyes / brows / mouth / nose / ears / accessories

### Classification strategy
Use a hybrid strategy:
1. **See-through layer naming rules first**
2. **General filename / alias rules second**
3. **Spatial rules third**
4. **Symmetry + overlap rules fourth**
5. **User correction UI last**

### Tasks
- [x] Build a See-through naming parser using aliases and regexes.
- [ ] Parse PSD group path as additional context.
- [ ] Support common synonyms and artist-friendly aliases.
- [ ] Compute per-layer geometry descriptors:
  - bbox
  - area
  - aspect ratio
  - centroid
  - contour length if available
  - overlap with other layers
  - relative vertical band in character space
- [x] Infer left/right by centroid relative to body centerline.
- [ ] Infer parent-child candidates by containment and proximity.
- [ ] Expose a part classification table in the UI.
- [ ] Allow user override before rig build.

### Required output labels for MVP
- [ ] root
- [ ] torso
- [ ] neck
- [ ] head
- [ ] upper_arm_l / lower_arm_l / hand_l
- [ ] upper_arm_r / lower_arm_r / hand_r
- [ ] upper_leg_l / lower_leg_l / foot_l
- [ ] upper_leg_r / lower_leg_r / foot_r
- [ ] hair_front
- [ ] hair_back
- [ ] eyes / mouth / brow as optional face controls

### Mapping notes
- [ ] `front hair` -> `hair_front`
- [ ] `back hair` -> `hair_back`
- [ ] `face` + `neck` + nearby head elements may imply `head`
- [ ] `topwear` usually maps to torso/chest anchor geometry
- [ ] `handwear` may need left/right split + arm chain inference
- [ ] `bottomwear` often helps locate pelvis/hip region
- [ ] `legwear` / `footwear` anchor lower-body rig chains

**Exit criteria**
- [ ] Common See-through-style PSDs classify with reasonable defaults.
- [ ] User can manually fix mislabeled layers quickly.

---

## Phase 5 — Build the heuristic rig estimator

### Core idea
Estimate joints from **mesh bounds + alpha silhouette geometry + semantic labels**, without requiring ML.

### Skeleton to generate
- [ ] root
- [ ] pelvis / torso pivot
- [ ] chest
- [ ] neck
- [ ] head
- [ ] shoulder_l / elbow_l / wrist_l
- [ ] shoulder_r / elbow_r / wrist_r
- [ ] hip_l / knee_l / ankle_l
- [ ] hip_r / knee_r / ankle_r
- [ ] optional eye controls
- [ ] optional hair chain starts

### Heuristic rules

#### Global frame
- [ ] Compute a character reference frame from torso/head/body layers.
- [ ] Establish centerline from torso/head centroids.
- [ ] Estimate overall character height from combined bounds.

#### Torso / pelvis / chest
- [ ] If `topwear` / `body` / `torso` exists, use its bbox and centroid as the main body anchor.
- [ ] Place pelvis near lower third of torso mesh.
- [ ] Place chest near upper third of torso mesh.
- [ ] Root defaults slightly below pelvis on centerline.
- [ ] Use `bottomwear` as a lower-torso stabilizer when present.

#### Neck / head
- [ ] If a `face` or `head` proxy region exists, set neck at the closest point between torso top and head lower bound.
- [ ] Set head pivot near lower-middle of head mesh, not geometric center.
- [ ] If `neck` exists as its own layer, use it as a strong anchor.

#### Shoulders / arms
- [ ] If arm layers exist, shoulder = nearest reasonable attachment point from upper-arm proximal edge to torso side.
- [ ] Elbow = weighted midpoint along upper/lower arm chain, biased by the bbox junction of upper/lower arm meshes.
- [ ] Wrist = proximal point of hand mesh or distal point of lower-arm mesh.
- [ ] If only one combined `handwear` region exists, estimate a 2-segment chain from the mesh long axis and left/right split.

#### Hips / legs
- [ ] Hip joints originate from lower torso width extrema, slightly inset.
- [ ] Knee = junction between upper-leg and lower-leg layers or the main bend point along the limb axis.
- [ ] Ankle = proximal point of foot mesh or distal end of lower leg.
- [ ] If `legwear` is fused, split the long axis into upper/lower segments using normalized height heuristics.
- [ ] Use `footwear` to improve ankle and foot tip placement.

#### Hair
- [ ] `hair_front` and `hair_back` should not become part of the main humanoid bone chain.
- [ ] Generate optional deform chains rooted near skull top/rear.
- [ ] Allow a simple chain count slider per hair layer.

#### Eyes / mouth
- [ ] For MVP, do not overbuild this.
- [ ] Optional control bones only if eye or mouth layers are present and clearly labeled.

### Confidence scoring
- [ ] Score each estimated joint and limb using:
  - semantic certainty
  - overlap consistency
  - symmetry quality
  - geometric plausibility
- [ ] Mark low-confidence limbs for optional ML assist or manual review.

### Debugging
- [ ] Draw temporary empties / gizmos / grease-pencil overlay at inferred joints.
- [ ] Show confidence colors.
- [ ] Allow joint drag correction before armature commit.

**Exit criteria**
- [ ] A usable skeleton can be previewed over imported layers without committing armature creation.

---

## Phase 6 — Optional ML assist layer

### Strategy
ML should help only when:
- classification confidence is low,
- limbs are missing/merged,
- or the user explicitly requests inference.

### Provider design
Implement a provider abstraction:
- [ ] `CPU`
- [ ] `ONNX CUDA` for Windows/Linux NVIDIA
- [ ] `Torch MPS` for Apple Silicon Metal fallback
- [ ] future: `CoreML` optimization path if packaging is practical

### Provider selection order
- [ ] User-selected provider if available
- [ ] Auto mode fallback chain:
  - CUDA
  - MPS
  - CPU

### DWPose integration plan
- [ ] Add a DWPose adapter that accepts a temporary flattened preview render or composite image.
- [ ] Run whole-body keypoint detection.
- [ ] Map 2D keypoints into the add-on's character plane space.
- [ ] Blend ML keypoints with heuristic joints using confidence-weighted fusion.
- [ ] Never let ML silently override high-confidence heuristic results.

### Dependency management
- [ ] Implement dependency installation only from preferences.
- [ ] Show exact package/model actions before install.
- [ ] Separate dependencies by backend.
- [ ] Keep PSD parsing backend install separate from ML install if needed.

#### Windows / Linux NVIDIA path
- [ ] ONNX Runtime GPU install option.
- [ ] Validate CUDA runtime presence before enabling provider.
- [ ] Clear error if CUDA/cuDNN mismatch is detected.

#### macOS Apple Silicon path
- [ ] Provide PyTorch MPS install option for Metal-backed execution.
- [ ] Keep CPU fallback available if MPS install fails.
- [ ] Treat CoreML provider support as optional future work unless Python packaging is proven clean in Blender's Python environment.

### Model cache
- [ ] Add model registry with checksum/version metadata.
- [ ] Store downloaded models in a dedicated cache dir.
- [ ] Provide buttons:
  - install backend deps
  - install PSD backend deps
  - download models
  - validate models
  - clear cache

**Exit criteria**
- [ ] User can install optional ML support from preferences.
- [ ] DWPose can run on supported machines when enabled.
- [ ] Core workflow still works without ML.

---

## Phase 7 — Armature generation in Blender

### Tasks
- [x] Convert the `RigPlan` into a Blender armature object.
- [x] Create edit bones in 2D plane space.
- [x] Build parent hierarchy.
- [ ] Set roll and local axes consistently for mirrored limbs.
- [ ] Add control bones only if they improve usability.
- [ ] Add basic IK option for arms/legs behind a toggle.
- [ ] Add custom shapes later, not in MVP.

### Bone naming convention
- [ ] Use a stable machine-readable naming convention.
- [ ] Keep left/right suffixes consistent.
- [ ] Avoid names that fight Rigify/other add-ons unless intentional.

### Optional extras
- [ ] Hair chain generation from selected hair meshes.
- [ ] Accessory parent bones for hats / props.
- [ ] Eye target controls.

**Exit criteria**
- [ ] Armature is created and aligned correctly over the character.
- [ ] Joint preview and committed armature match visually.

---

## Phase 8 — Weighting and binding

### Binding strategy
Because the source is cutout-style layer meshes, weighting can be simpler than full 3D skinning.

### Tasks
- [x] Create vertex groups per deform bone.
- [x] Assign whole rigid layers directly when appropriate:
  - head -> head bone
  - hand -> hand bone
  - foot -> foot bone
- [ ] For bendable limbs, compute gradient weights along the limb axis.
- [ ] Use projected distance to bone segment for initial weights.
- [x] Normalize weights per vertex.
- [ ] Add optional preserve-rigid toggle for rigid cutout parts.
- [ ] For hair meshes, assign root-heavy falloff along chain length.

### Layer-specific rules
- [ ] `hair_front` can either follow head rigidly or use chain deform.
- [ ] `hair_back` should usually parent near skull/rear head, not torso root.
- [ ] torso/body/topwear should mostly bind to torso/chest/root.
- [ ] face parts default to head.
- [ ] accessories/objects default to explicit parent rules or rigid parenting.

### Verification
- [ ] Provide one-click pose test operators:
  - raise left arm
  - bend right arm
  - bend left knee
  - head rotate
  - hair swing preview

**Exit criteria**
- [ ] Generated rigs survive a standard test pose without catastrophic deformation.

---

## Phase 9 — User workflow and UI

### Main panel
- [ ] Source folder/file picker
- [ ] Import PNG layers
- [ ] Import PSD
- [ ] Generate alpha meshes
- [ ] Classify parts
- [ ] Preview rig
- [ ] Build armature
- [ ] Bind weights
- [ ] Run full pipeline

### Secondary UI
- [ ] Layer classification table
- [ ] Joint confidence list
- [ ] Low-confidence warnings
- [ ] Optional ML controls
- [ ] Debug overlay toggles
- [ ] Skipped PSD layer report

### Preferences UI
- [ ] Cache directory
- [ ] Dependency install buttons
- [ ] PSD backend install button
- [ ] Model download buttons
- [ ] Provider priority
- [ ] Logging level
- [ ] Experimental feature toggles
- [ ] Offline mode toggle

### UX requirements
- [ ] Never auto-download anything at enable time.
- [ ] Never require ML to use the add-on.
- [ ] Every destructive action asks or safely replaces generated data.
- [ ] Errors should say what failed, where, and how to recover.
- [ ] PSD import errors should point to the exact layer if possible.

**Exit criteria**
- [ ] A new user can run the full pipeline from the panel without scripting.

---

## Phase 10 — Test matrix

### Functional tests
- [ ] clean humanoid anime PNG layers
- [ ] clean See-through PSD
- [ ] PSD with empty layers
- [ ] PSD with hidden layers
- [ ] PSD with nested groups
- [ ] missing hand layers
- [ ] combined arm layers
- [ ] combined leg layers
- [ ] asymmetrical pose-like illustration
- [ ] oversized hair_front / hair_back
- [ ] accessories overlapping head/body
- [ ] semi-transparent edges
- [ ] tiny image inputs
- [ ] large image inputs

### Platform tests
- [ ] Windows + CPU only
- [ ] Windows + NVIDIA CUDA
- [ ] Linux + CPU only
- [ ] Linux + NVIDIA CUDA
- [ ] macOS Apple Silicon + CPU only
- [ ] macOS Apple Silicon + MPS

### Blender tests
- [ ] install / enable / disable / re-enable
- [ ] rerun pipeline on same scene
- [ ] undo/redo on each major operator
- [ ] save/reopen blend file
- [ ] packaged extension install flow

### Performance tests
- [ ] many layers
- [ ] large alpha meshes
- [ ] repeated preview rebuilds
- [ ] optional inference cold start vs warm start
- [ ] PSD import speed on large decomposed files

**Exit criteria**
- [ ] No blocking regressions in the supported matrix.

---

## Phase 11 — Packaging, docs, and release prep
- [ ] Add `README.md`
- [ ] Add `THIRD_PARTY_NOTICES.md`
- [ ] Add `CHANGELOG.md`
- [ ] Add install docs for each backend
- [ ] Add troubleshooting docs
- [ ] Add sample See-through naming conventions doc
- [ ] Add known limitations doc
- [ ] Add development notes for maintaining vendored alpha-mesh code
- [ ] Add extension validation/build script if the template expects one

### Release gates
- [ ] Clean install from zip/package
- [ ] Offline heuristic pipeline works
- [ ] PSD import works
- [ ] Empty PSD layer handling works
- [ ] Dependency installs are opt-in and recoverable
- [ ] Errors are readable
- [ ] Licensing notice is complete

---

## Recommended MVP order
1. [ ] Audit template and local alpha-mesh code
2. [ ] Audit Stretchy Studio and See-through repos
3. [ ] Stand up extension shell
4. [ ] PNG + PSD ingestion
5. [ ] Empty PSD layer filtering
6. [ ] Batch alpha-mesh import from imported layers
7. [ ] Part classification UI + overrides
8. [ ] Heuristic skeleton preview
9. [ ] Armature commit
10. [ ] Basic weighting
11. [ ] Test-pose operators
12. [ ] Preferences-based dependency manager
13. [ ] Optional DWPose provider

Do **not** start with ML. Start with the fully offline heuristic path.

---

## Heuristic details Codex should implement first

### Part center estimates
- [ ] Use non-transparent pixel bounds if available from source import data.
- [ ] Otherwise use generated mesh local-space bounds.
- [ ] Cache centroid and extremal points for every mesh.

### Limb axis estimation
- [ ] For elongated meshes, estimate principal axis from covariance / PCA-style longest direction.
- [ ] Use that axis to split combined limb meshes into proximal/distal candidate segments.
- [ ] Prefer semantic labels over pure geometry when both exist.

### Attachment inference
- [ ] Arm attaches where upper-arm proximal end is closest to torso boundary.
- [ ] Leg attaches where upper-leg proximal end is closest to pelvis band.
- [ ] Head attaches where head lower boundary is closest to neck/top torso band.

### Symmetry repair
- [ ] If only one limb side is confidently detected, mirror from the other side as a fallback hypothesis.
- [ ] Mark mirrored results as lower confidence.

### Manual correction layer
- [ ] Joint preview points must be draggable before build.
- [ ] User edits should be stored and survive preview refresh when possible.

---

## Error-handling requirements
- [ ] If dependency install fails, show stderr/stdout summary and keep add-on usable.
- [ ] If PSD backend install fails, keep PNG workflow usable.
- [ ] If a model download fails, keep heuristic workflow usable.
- [ ] If layer classification is incomplete, allow partial rig generation or manual assignment.
- [ ] If alpha mesh import fails for one file/layer, continue the rest and report failures at the end.
- [ ] If PSD layer rasterization fails for one layer, continue the rest and report failures at the end.
- [ ] If weights fail, still keep the armature and imported meshes.

---

## File/path policy
- [ ] Never hardcode Tyler's absolute paths into shipped runtime logic.
- [ ] Use the provided local paths only during development and migration.
- [ ] Resolve runtime paths using Blender/user directories and add-on preferences.

---

## Definition of Done
The MVP is done when all of the following are true:
- [ ] A folder of separated avatar PNG layers can be imported in Blender.
- [ ] A See-through-style PSD can be imported in Blender.
- [ ] Empty PSD layers are ignored cleanly.
- [ ] Each usable layer becomes a usable mesh based on alpha.
- [ ] The add-on classifies the layers into major body parts.
- [ ] The add-on previews a skeleton in the right place.
- [ ] The user can tweak joints and then commit an armature.
- [ ] The add-on binds weights with acceptable first-pass deformation.
- [ ] The workflow runs with no ML installed.
- [ ] Optional ML support can be installed from preferences.
- [ ] Windows/Linux CUDA and macOS Apple Silicon paths are planned and gated behind clear diagnostics.

---

## References to review while implementing
- [ ] Local Blender Extension Template path
- [ ] Local Import Meshed Alpha path
- [ ] Stretchy Studio repo: `https://github.com/MangoLion/stretchystudio`
- [ ] See-through repo: `https://github.com/shitagaki-lab/see-through`
- [ ] Optional See-through wrapper reference: `https://github.com/jtydhr88/ComfyUI-See-through`
- [ ] DWPose repo / ONNX branch
- [ ] Blender AddonPreferences / extension packaging docs
- [ ] COA Tools and Tiny 2D Rig Tools for cutout rig UX ideas
