from __future__ import annotations

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import PropertyGroup

from .core.models import LayerPart


class HALLWAYAVATAR_PG_layer_item(PropertyGroup):
    source_path: StringProperty(name="Source Path")
    source_type: StringProperty(name="Source Type")
    document_path: StringProperty(name="Document Path")
    layer_path: StringProperty(name="Layer Path")
    layer_name: StringProperty(name="Layer Name")
    normalized_token: StringProperty(name="Normalized Token")
    imported_object_name: StringProperty(name="Imported Object")
    temp_image_path: StringProperty(name="Temp Image")
    image_width: IntProperty(name="Image Width")
    image_height: IntProperty(name="Image Height")
    canvas_width: IntProperty(name="Canvas Width")
    canvas_height: IntProperty(name="Canvas Height")
    offset_x: IntProperty(name="Offset X")
    offset_y: IntProperty(name="Offset Y")
    alpha_x0: IntProperty(name="Alpha X0")
    alpha_y0: IntProperty(name="Alpha Y0")
    alpha_x1: IntProperty(name="Alpha X1")
    alpha_y1: IntProperty(name="Alpha Y1")
    local_alpha_x0: IntProperty(name="Local Alpha X0")
    local_alpha_y0: IntProperty(name="Local Alpha Y0")
    local_alpha_x1: IntProperty(name="Local Alpha X1")
    local_alpha_y1: IntProperty(name="Local Alpha Y1")
    centroid_x: FloatProperty(name="Centroid X")
    centroid_y: FloatProperty(name="Centroid Y")
    area: IntProperty(name="Area")
    perimeter: FloatProperty(name="Perimeter")
    side_guess: StringProperty(name="Side Guess")
    semantic_label: StringProperty(name="Semantic Label")
    parent_semantic_label: StringProperty(name="Parent Semantic Label")
    confidence: FloatProperty(name="Confidence")
    skipped: BoolProperty(name="Skipped", description="Whether this layer part was skipped during import")
    skip_reason: StringProperty(name="Skip Reason")
    draw_index: IntProperty(name="Draw Index")


class HALLWAYAVATAR_PG_qremesh_settings(PropertyGroup):
    show_section: BoolProperty(
        name="Show Remesh Section",
        description="Expand or collapse the quad remesh settings section",
        default=True,
    )
    show_main_settings_section: BoolProperty(
        name="Show Main Remesh Settings",
        description="Expand or collapse the main qmesh settings section",
        default=True,
    )
    show_edge_loops_section: BoolProperty(
        name="Show Edge Loops Section",
        description="Expand or collapse the edge-loop guidance settings section",
        default=False,
    )
    show_misc_section: BoolProperty(
        name="Show Misc Section",
        description="Expand or collapse the symmetry and misc qmesh settings section",
        default=False,
    )
    auto_on_import: BoolProperty(
        name="Auto Remesh On Import",
        description="Run quad remesh automatically after PSD layers are imported",
        default=True,
    )
    target_quad_count: IntProperty(name="Quad Count", default=3000, soft_min=100, soft_max=10000, min=1, step=20)
    unsubdivide_iterations: IntProperty(name="Un-Subdivide", default=2, min=0, max=8, soft_max=6)
    unsubdivide_target_count: IntProperty(name="Un-Subdivide Target", default=1400, min=1, soft_min=100, soft_max=10000, step=20)
    show_advanced_filters: BoolProperty(
        name="Show Advanced Remesh Filters",
        description="Expand or collapse the See-through category remesh filter section",
        default=False,
    )
    remesh_front_hair: BoolProperty(name="Front Hair", description="Allow quad remesh on front hair layers", default=True)
    remesh_back_hair: BoolProperty(name="Back Hair", description="Allow quad remesh on back hair layers", default=True)
    remesh_face_head: BoolProperty(name="Face / Head", description="Allow quad remesh on face and other head-region layers", default=False)
    remesh_topwear: BoolProperty(name="Topwear", description="Allow quad remesh on topwear and torso-like layers", default=True)
    remesh_handwear: BoolProperty(name="Handwear", description="Allow quad remesh on arm and hand layers", default=True)
    remesh_bottomwear: BoolProperty(name="Bottomwear", description="Allow quad remesh on pelvis and bottomwear layers", default=False)
    remesh_legwear: BoolProperty(name="Legwear", description="Allow quad remesh on leg layers", default=True)
    remesh_footwear: BoolProperty(name="Footwear", description="Allow quad remesh on foot and shoe layers", default=True)
    remesh_tail: BoolProperty(name="Tail", description="Allow quad remesh on tail layers", default=False)
    remesh_wings: BoolProperty(name="Wings", description="Allow quad remesh on wing layers", default=False)
    remesh_objects: BoolProperty(name="Objects / Accessories", description="Allow quad remesh on prop and accessory layers", default=False)
    remesh_unclassified: BoolProperty(name="Unclassified", description="Allow quad remesh on layers that did not match a See-through category", default=False)
    target_count_as_input_percentage: BoolProperty(
        name="Target Count Is Input %",
        description="Interpret Quad Count as a percentage of the input face count instead of a fixed target",
        default=True,
    )
    target_edge_length: FloatProperty(name="Target Edge Length", default=0.02, min=0.0, precision=4, unit="LENGTH")
    adaptive_size: FloatProperty(name="Adaptive Size", default=100.0, min=0.0, max=100.0, precision=0, subtype="PERCENTAGE")
    adapt_quad_count: BoolProperty(
        name="Adapt Quad Count",
        description="Let qmesh exceed the target count when it needs extra topology to preserve detail",
        default=True,
    )
    max_quad_ratio: FloatProperty(name="Max Quad Ratio", default=6.0, min=1.0, soft_max=16.0, precision=2)
    use_vertex_color_map: BoolProperty(name="Use Vertex Color Map", description="Use painted vertex colors to drive local quad density", default=False)
    use_materials: BoolProperty(name="Use Materials", description="Use material borders as hints when generating quad flow", default=False)
    use_normals_splitting: BoolProperty(name="Use Normals Splitting", description="Use split normals as hard-edge guidance for quad flow", default=False)
    autodetect_hard_edges: BoolProperty(name="Detect Hard Edges", description="Automatically detect hard edges from geometry angles", default=True)
    enable_remesh: BoolProperty(name="Preprocess", description="Reserved upstream preprocess toggle", default=False)
    enable_smoothing: BoolProperty(name="Smoothing", description="Reserved upstream smoothing toggle", default=False)
    enable_sharp: BoolProperty(name="Sharp Detection", description="Reserved upstream sharp-feature toggle", default=True)
    sharp_angle: FloatProperty(name="Angle Threshold", min=0.0, max=180.0, default=35.0, precision=1, step=10, subtype="UNSIGNED")
    symmetry_x: BoolProperty(name="X Symmetry", description="Force local X-axis symmetry during quad remesh", default=False)
    symmetry_y: BoolProperty(name="Y Symmetry", description="Force local Y-axis symmetry during quad remesh", default=False)
    symmetry_z: BoolProperty(name="Z Symmetry", description="Force local Z-axis symmetry during quad remesh", default=False)
    scale_factor: FloatProperty(name="Dynamic Quad Size", min=0.01, max=10.0, default=1.0, subtype="FACTOR")
    fixed_chart_clusters: IntProperty(name="Fixed Chart Clusters", min=0, default=0)
    alpha: FloatProperty(name="Alpha", default=0.005, min=0.0, max=0.999, precision=3, step=0.5, subtype="FACTOR")
    ilp_method: EnumProperty(
        name="ILP Method",
        items=(
            ("LEASTSQUARES", "Least Squares", ""),
            ("ABS", "Absolute", ""),
        ),
        default="LEASTSQUARES",
    )
    time_limit: IntProperty(name="Time Limit", min=1, default=200)
    gap_limit: FloatProperty(name="Gap Limit", min=0.0, default=0.0)
    minimum_gap: FloatProperty(name="Minimum Gap", min=0.0, default=0.4)
    isometry: BoolProperty(name="Isometry", description="Favor even, isometric quad sizing during remeshing", default=True)
    regularity_quadrilaterals: BoolProperty(name="Regularity Quads", description="Favor regular quad layouts when solving the remesh", default=True)
    regularity_non_quadrilaterals: BoolProperty(name="Regularity Non-Quads", description="Favor regular layouts around non-quad patches", default=True)
    regularity_non_quadrilaterals_weight: FloatProperty(name="Non-Quad Weight", min=0.0, max=1.0, default=0.9)
    align_singularities: BoolProperty(name="Align Singularities", description="Try to align singularity placement with the solved field", default=True)
    align_singularities_weight: FloatProperty(name="Singularity Weight", min=0.0, max=1.0, default=0.1)
    repeat_losing_constraints_iterations: BoolProperty(name="Repeat Constraint Iterations", description="Retry the solver while relaxing losing constraints between passes", default=True)
    repeat_losing_constraints_quads: BoolProperty(name="Repeat Constraint Quads", description="Preserve quad regularity constraints during repeat passes", default=False)
    repeat_losing_constraints_non_quads: BoolProperty(name="Repeat Constraint Non-Quads", description="Preserve non-quad regularity constraints during repeat passes", default=False)
    repeat_losing_constraints_align: BoolProperty(name="Repeat Constraint Align", description="Preserve singularity alignment constraints during repeat passes", default=True)
    hard_parity_constraint: BoolProperty(name="Hard Parity Constraint", description="Force a stricter parity constraint when solving the quad layout", default=True)
    flow_config: EnumProperty(
        name="Flow Config",
        items=(
            ("SIMPLE", "Simple", ""),
            ("HALF", "Half", ""),
        ),
        default="SIMPLE",
    )
    satsuma_config: EnumProperty(
        name="Satsuma Config",
        items=(
            ("DEFAULT", "Default", ""),
            ("MST", "Approx-MST", ""),
            ("ROUND2EVEN", "Approx-Round2Even", ""),
            ("SYMMDC", "Approx-Symmdc", ""),
            ("EDGETHRU", "Edgethru", ""),
            ("LEMON", "Lemon", ""),
            ("NODETHRU", "Nodethru", ""),
        ),
        default="DEFAULT",
    )
    callback_time_limit: FloatVectorProperty(
        name="Callback Time Limit",
        size=8,
        default=(3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 90.0, 120.0),
    )
    callback_gap_limit: FloatVectorProperty(
        name="Callback Gap Limit",
        size=8,
        precision=3,
        default=(0.005, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3),
    )


class HALLWAYAVATAR_PG_state(PropertyGroup):
    source_psd_path: StringProperty(name="PSD Path")
    imported_collection_name: StringProperty(name="Imported Collection", default="Hallway Avatar Layers")
    rig_collection_name: StringProperty(name="Rig Collection", default="Hallway Avatar Rig")
    armature_object_name: StringProperty(name="Armature Object")
    show_source_section: BoolProperty(name="Show Source Section", description="Expand or collapse the source PSD file controls", default=True)
    show_backend_section: BoolProperty(name="Show Backend Section", description="Expand or collapse the local PSD backend status and install controls", default=False)
    show_import_section: BoolProperty(name="Show Import Section", description="Expand or collapse the main PSD import settings section", default=True)
    show_advanced_alpha_settings: BoolProperty(name="Show Advanced Alpha Settings", description="Expand or collapse advanced alpha filtering controls for PSD import", default=False)
    show_alpha_thresholds_section: BoolProperty(name="Show Transparency Controls", description="Expand or collapse the alpha threshold and noise filtering controls", default=False)
    show_trace_contrast_section: BoolProperty(name="Show Trace Contrast Section", description="Expand or collapse the meshed-alpha trace contrast remap controls", default=False)
    show_rigging_section: BoolProperty(name="Show Rigging Section", description="Expand or collapse the rig build and binding tools", default=True)
    show_roadmap_section: BoolProperty(name="Show Roadmap Section", description="Expand or collapse the roadmap notes for upcoming Hallway features", default=False)
    show_summary_section: BoolProperty(name="Show Summary Section", description="Expand or collapse the import summary and classification results", default=True)
    ignore_hidden_layers: BoolProperty(name="Ignore Hidden PSD Layers", description="Skip PSD layers that are hidden in the document", default=True)
    ignore_empty_layers: BoolProperty(name="Ignore Empty PSD Layers", description="Skip layers with no visible alpha after filtering", default=True)
    keep_tiny_named_parts: BoolProperty(name="Keep Tiny Named Parts", description="Keep very small named facial parts like mouth, nose, lashes, and irides", default=True)
    import_facial_features: BoolProperty(
        name="Import Facial Features",
        description="Import face-detail layers such as ears, nose, eyes, brows, lashes, and mouth. The base Face layer still imports when this is OFF",
        default=False,
    )
    auto_rig_on_import: BoolProperty(
        name="Auto Rig On Import",
        description="Build and bind the generated armature automatically after PSD import and optional remesh",
        default=True,
    )
    auto_setup_facial_video: BoolProperty(
        name="Facial Video Preview",
        description="After import, duplicate the Face layer UVs with the supplied transform txt and replace its material with a movie-backed background material",
        default=True,
    )
    facial_video_transform_path: StringProperty(
        name="Facial UV Transform",
        description="Text file containing [blender_uv_inverse_transform] and [full_frame_pixel_transform] sections",
    )
    facial_video_path: StringProperty(
        name="Facial Video",
        description="Movie file to use as the Face layer background material texture",
    )
    facial_video_frame_duration: IntProperty(
        name="Video Frames",
        description="Number of frames Blender should play from the movie texture",
        min=1,
        default=1000,
    )
    facial_video_start_frame: IntProperty(
        name="Start Frame",
        description="Timeline frame where the movie texture starts playing",
        min=0,
        default=0,
    )
    facial_video_frame_offset: IntProperty(
        name="Frame Offset",
        description="Offset into the movie texture playback",
        default=0,
    )
    facial_video_auto_refresh: BoolProperty(
        name="Auto Refresh Video",
        description="Update the movie texture when the timeline frame changes",
        default=True,
    )
    min_visible_pixels: IntProperty(name="Minimum Visible Pixels", min=0, default=8)
    alpha_noise_floor: IntProperty(name="Alpha Noise Floor", description="Treat layers below this maximum alpha value as transparent noise", min=0, max=255, default=64)
    visible_alpha_threshold: IntProperty(name="Visible Alpha Threshold", description="Base alpha cutoff for deciding which pixels count as visible", min=0, max=255, default=32)
    auto_alpha_threshold_boost: BoolProperty(name="Auto Boost Threshold", description="Automatically raise the visible alpha threshold for noisy faint layers", default=True)
    trace_contrast_low: FloatProperty(name="Trace Contrast Low", min=0.0, max=1.0, default=0.1, precision=3, subtype="FACTOR")
    trace_contrast_high: FloatProperty(name="Trace Contrast High", min=0.0, max=1.0, default=0.9, precision=3, subtype="FACTOR")
    mesh_grid_resolution: IntProperty(name="Mesh Grid Resolution", min=1, max=64, default=12)
    replace_existing: BoolProperty(name="Replace Existing Output", description="Clear the previous imported avatar output before creating a new one", default=True)
    auto_bind_on_build: BoolProperty(name="Auto Bind On Build", description="Bind imported meshes automatically when a rig is built", default=True)
    imported_count: IntProperty(name="Imported Count")
    remeshed_count: IntProperty(name="Remeshed Count")
    remesh_performed: BoolProperty(name="Remesh Performed", default=False)
    skipped_count: IntProperty(name="Skipped Count")
    classified_count: IntProperty(name="Classified Count")
    import_progress_visible: BoolProperty(name="Show Import Progress", default=False)
    import_progress: FloatProperty(name="Import Progress", min=0.0, max=1.0, default=0.0, subtype="FACTOR")
    import_progress_text: StringProperty(name="Import Progress Text")
    active_layer_index: IntProperty(name="Active Layer Index")
    last_report: StringProperty(name="Last Report")
    layer_items: CollectionProperty(type=HALLWAYAVATAR_PG_layer_item)
    qremesh_settings: PointerProperty(type=HALLWAYAVATAR_PG_qremesh_settings)


classes = (
    HALLWAYAVATAR_PG_layer_item,
    HALLWAYAVATAR_PG_qremesh_settings,
    HALLWAYAVATAR_PG_state,
)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


def register_properties() -> None:
    bpy.types.Scene.hallway_avatar_state = bpy.props.PointerProperty(type=HALLWAYAVATAR_PG_state)


def unregister_properties() -> None:
    del bpy.types.Scene.hallway_avatar_state


def clear_layer_items(scene: bpy.types.Scene) -> None:
    scene.hallway_avatar_state.layer_items.clear()


def set_layer_items(scene: bpy.types.Scene, parts: list[LayerPart]) -> None:
    state = scene.hallway_avatar_state
    state.layer_items.clear()
    state.imported_count = 0
    state.skipped_count = 0
    state.classified_count = 0

    for part in parts:
        item = state.layer_items.add()
        item.source_path = part.source_path
        item.source_type = part.source_type
        item.document_path = part.document_path or ""
        item.layer_path = part.layer_path
        item.layer_name = part.layer_name
        item.normalized_token = part.normalized_token
        item.imported_object_name = part.imported_object_name
        item.temp_image_path = part.temp_image_path or ""
        item.image_width = part.image_size[0]
        item.image_height = part.image_size[1]
        item.canvas_width = part.canvas_size[0]
        item.canvas_height = part.canvas_size[1]
        item.offset_x = part.canvas_offset[0]
        item.offset_y = part.canvas_offset[1]
        item.alpha_x0 = part.alpha_bbox[0]
        item.alpha_y0 = part.alpha_bbox[1]
        item.alpha_x1 = part.alpha_bbox[2]
        item.alpha_y1 = part.alpha_bbox[3]
        item.local_alpha_x0 = part.local_alpha_bbox[0]
        item.local_alpha_y0 = part.local_alpha_bbox[1]
        item.local_alpha_x1 = part.local_alpha_bbox[2]
        item.local_alpha_y1 = part.local_alpha_bbox[3]
        item.centroid_x = part.centroid[0]
        item.centroid_y = part.centroid[1]
        item.area = part.area
        item.perimeter = part.perimeter
        item.side_guess = part.side_guess
        item.semantic_label = part.semantic_label
        item.parent_semantic_label = part.parent_semantic_label
        item.confidence = part.confidence
        item.skipped = part.skipped
        item.skip_reason = part.skip_reason
        item.draw_index = part.draw_index

        if part.skipped:
            state.skipped_count += 1
        else:
            state.imported_count += 1
        if part.semantic_label and part.semantic_label != "unclassified":
            state.classified_count += 1


def get_parts(scene: bpy.types.Scene) -> list[LayerPart]:
    parts: list[LayerPart] = []
    for item in scene.hallway_avatar_state.layer_items:
        part = LayerPart(
            source_path=item.source_path,
            source_type=item.source_type,
            document_path=item.document_path or None,
            layer_path=item.layer_path,
            layer_name=item.layer_name,
            normalized_token=item.normalized_token,
            imported_object_name=item.imported_object_name,
            temp_image_path=item.temp_image_path or None,
            image_size=(item.image_width, item.image_height),
            canvas_size=(item.canvas_width, item.canvas_height),
            canvas_offset=(item.offset_x, item.offset_y),
            alpha_bbox=(item.alpha_x0, item.alpha_y0, item.alpha_x1, item.alpha_y1),
            local_alpha_bbox=(item.local_alpha_x0, item.local_alpha_y0, item.local_alpha_x1, item.local_alpha_y1),
            centroid=(item.centroid_x, item.centroid_y),
            area=item.area,
            perimeter=item.perimeter,
            side_guess=item.side_guess,
            semantic_label=item.semantic_label,
            parent_semantic_label=item.parent_semantic_label,
            confidence=item.confidence,
            skipped=item.skipped,
            skip_reason=item.skip_reason,
            draw_index=item.draw_index,
        )
        parts.append(part)
    return parts
