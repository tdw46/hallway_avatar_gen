from __future__ import annotations

import bpy
from bpy.props import BoolProperty, CollectionProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy.types import PropertyGroup

from .core.models import LayerPart
from .core.qremeshify_runtime.props import QRPropertyGroup, QWPropertyGroup


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


class HALLWAYAVATAR_PG_qremeshify_settings(PropertyGroup):
    show_section: BoolProperty(
        name="Show Remesh Section",
        description="Expand or collapse the quad remesh settings section",
        default=True,
    )
    show_main_settings_section: BoolProperty(
        name="Show Main Remesh Settings",
        description="Expand or collapse the main remesh settings section",
        default=True,
    )
    show_advanced_qremeshify: BoolProperty(
        name="Show Advanced Remesh Settings",
        description="Expand or collapse advanced remesh controls",
        default=False,
    )
    show_callback_limits: BoolProperty(name="Show Callback Limits", description="Expand or collapse remesh callback limit controls", default=False)
    auto_on_import: BoolProperty(
        name="Auto Remesh On Import",
        description="Run quad remesh automatically after PSD layers are imported",
        default=True,
    )
    use_fast_planar_strips: BoolProperty(
        name="Hallway Planar Remesh",
        description="Use the fast contour/quadrant remesher for flat PSD layer silhouettes",
        default=True,
    )
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
    setup_mouth_video_plane: BoolProperty(
        name="Mouth Video Plane",
        description="Create a separate movie-backed mouth plane from the relative mouth bbox in the facial transform txt",
        default=False,
    )
    facial_video_transform_path: StringProperty(
        name="Facial UV Transform",
        description="Text file containing [blender_uv_inverse_transform] and [full_frame_pixel_transform] sections",
    )
    facial_video_path: StringProperty(
        name="Facial Video",
        description="Movie file to use as the Face layer background material texture",
    )
    mouth_video_path: StringProperty(
        name="Mouth Video",
        description="Movie file to project onto the optional mouth plane",
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
    qremeshify_settings: PointerProperty(type=HALLWAYAVATAR_PG_qremeshify_settings)


classes = (
    QWPropertyGroup,
    QRPropertyGroup,
    HALLWAYAVATAR_PG_layer_item,
    HALLWAYAVATAR_PG_qremeshify_settings,
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
    bpy.types.Scene.quadwild_props = bpy.props.PointerProperty(type=QWPropertyGroup)
    bpy.types.Scene.quadpatches_props = bpy.props.PointerProperty(type=QRPropertyGroup)


def unregister_properties() -> None:
    del bpy.types.Scene.quadpatches_props
    del bpy.types.Scene.quadwild_props
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
