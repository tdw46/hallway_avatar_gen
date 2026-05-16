from __future__ import annotations

import bmesh
import bpy
from mathutils import Vector

from .. import properties
from ..utils import blender as blender_utils
from ..utils import env
from ..utils.logging import get_logger
from . import alpha_mesh_adapter, armature_builder, facial_video_preview, heuristic_rigger, mtoon_materials, part_classifier, psd_io, qremeshify, vrm_integration, weighting

logger = get_logger("pipeline")
ADDON_ID = env.addon_package_id(__package__)
LAYER_DEPTH_STEP_METERS = 0.0005
IMPORT_VERTEX_SCALE = 0.211
IMPORT_TARGET_MIN_Z = 1.31525
FACIAL_FEATURE_TOKENS = {
    "ears",
    "earwear",
    "nose",
    "mouth",
    "eyewhite",
    "irides",
    "eyelash",
    "eyebrow",
    "eyewear",
}


def _cache_dir_from_context(context: bpy.types.Context) -> str:
    addon = context.preferences.addons.get(ADDON_ID)
    if not addon:
        return ""
    prefs = addon.preferences
    return getattr(prefs, "cache_dir", "")


def _world_min_vertex_z(obj: bpy.types.Object) -> float | None:
    if obj.type != "MESH" or obj.data is None or not getattr(obj.data, "vertices", None):
        return None
    return min((obj.matrix_world @ vertex.co).z for vertex in obj.data.vertices)


def _ground_offset_from_parts(parts: list) -> float:
    offsets: list[float] = []
    for part in parts:
        if part.skipped or not part.imported_object_name:
            continue
        obj = bpy.data.objects.get(part.imported_object_name)
        if obj is None:
            continue
        value = float(obj.get("hallway_avatar_ground_offset_z", 0.0))
        if abs(value) > 1e-9:
            offsets.append(value)
    if not offsets:
        return 0.0
    return sum(offsets) / len(offsets)


def _import_scale_from_parts(parts: list) -> float:
    scales: list[float] = []
    for part in parts:
        if part.skipped or not part.imported_object_name:
            continue
        obj = bpy.data.objects.get(part.imported_object_name)
        if obj is None:
            continue
        value = float(obj.get("hallway_avatar_import_scale", 1.0))
        if value > 0.0:
            scales.append(value)
    if not scales:
        return 1.0
    return sum(scales) / len(scales)


def _apply_layer_depth_stack(parts: list, imported_objects: list[bpy.types.Object]) -> None:
    ordered = [(part, obj) for part, obj in zip(parts, imported_objects, strict=False) if obj is not None]
    if not ordered:
        return

    for depth_index, (part, obj) in enumerate(ordered):
        depth_offset = -depth_index * LAYER_DEPTH_STEP_METERS
        obj.location.y = depth_offset
        obj["hallway_avatar_depth_offset"] = depth_offset
        obj["hallway_avatar_depth_rank"] = depth_index
        obj["hallway_avatar_draw_index"] = part.draw_index
        logger.info(
            "Layer stack %s -> draw_index=%s depth_rank=%s world_y=%.6f",
            obj.name,
            part.draw_index,
            depth_index,
            obj.location.y,
        )


def _imported_mesh_objects_for_parts(parts: list) -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    seen: set[str] = set()
    for part in parts:
        if part.skipped or not part.imported_object_name or part.imported_object_name in seen:
            continue
        obj = bpy.data.objects.get(part.imported_object_name)
        if obj is None or obj.type != "MESH":
            continue
        seen.add(obj.name)
        objects.append(obj)
    return objects


def _skip_facial_features_when_disabled(parts: list, import_facial_features: bool) -> None:
    if import_facial_features:
        return
    for part in parts:
        if part.skipped:
            continue
        if part.normalized_token not in FACIAL_FEATURE_TOKENS:
            continue
        part.skipped = True
        part.skip_reason = "facial feature import disabled"


def _apply_import_geometry_transform(
    imported_objects: list[bpy.types.Object],
    *,
    import_scale: float = IMPORT_VERTEX_SCALE,
    target_min_z: float = IMPORT_TARGET_MIN_Z,
) -> float:
    min_values = [value for value in (_world_min_vertex_z(obj) for obj in imported_objects) if value is not None]
    if not min_values:
        return 0.0

    scale_value = max(float(import_scale), 1.0e-6)
    for obj in imported_objects:
        if obj.type != "MESH" or obj.data is None or not getattr(obj.data, "vertices", None):
            continue
        inverse_world = obj.matrix_world.inverted_safe()
        for vertex in obj.data.vertices:
            world_co = obj.matrix_world @ vertex.co
            vertex.co = inverse_world @ (world_co * scale_value)
        obj.data.update()
        obj["hallway_avatar_import_scale"] = scale_value
        obj["hallway_avatar_import_target_min_z"] = float(target_min_z)

    scaled_min_values = [value for value in (_world_min_vertex_z(obj) for obj in imported_objects) if value is not None]
    if not scaled_min_values:
        return 0.0

    min_z = min(scaled_min_values)
    z_offset = float(target_min_z) - min_z
    logger.info(
        "Import geometry transform -> scale=%.6f pre_lift_min_z=%.6f target_min_z=%.6f requested_offset=%.6f",
        scale_value,
        min_z,
        target_min_z,
        z_offset,
    )

    for obj in imported_objects:
        before_min_z = _world_min_vertex_z(obj)
        local_offset = obj.matrix_world.inverted_safe().to_3x3() @ Vector((0.0, 0.0, z_offset))
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bmesh.ops.translate(bm, verts=bm.verts[:], vec=local_offset)
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update()
        obj["hallway_avatar_ground_offset_z"] = z_offset
        obj["hallway_avatar_ground_min_z_before"] = min_z
        obj["hallway_avatar_import_final_min_z"] = float(target_min_z)
        after_min_z = _world_min_vertex_z(obj)
        logger.info(
            "Import geometry transform %s -> before_min_z=%s after_min_z=%s local_offset=(%.6f, %.6f, %.6f)",
            obj.name,
            f"{before_min_z:.6f}" if before_min_z is not None else "None",
            f"{after_min_z:.6f}" if after_min_z is not None else "None",
            local_offset.x,
            local_offset.y,
            local_offset.z,
        )
    return z_offset


def _lift_imported_meshes_to_ground(imported_objects: list[bpy.types.Object]) -> float:
    return _apply_import_geometry_transform(imported_objects)


def import_psd_scene(context: bpy.types.Context, filepath: str) -> list:
    scene = context.scene
    state = scene.hallway_avatar_state
    cache_dir = _cache_dir_from_context(context)
    state.remesh_performed = False

    parts = psd_io.load_psd_layer_parts(
        filepath,
        ignore_hidden_layers=state.ignore_hidden_layers,
        ignore_empty_layers=state.ignore_empty_layers,
        min_visible_pixels=state.min_visible_pixels,
        alpha_noise_floor=state.alpha_noise_floor,
        visible_alpha_threshold=state.visible_alpha_threshold,
        auto_alpha_threshold_boost=state.auto_alpha_threshold_boost,
        keep_tiny_named_parts=state.keep_tiny_named_parts,
        configured_cache_dir=cache_dir,
    )
    part_classifier.classify_parts(parts)
    _skip_facial_features_when_disabled(parts, state.import_facial_features)

    collection = blender_utils.clear_collection(state.imported_collection_name) if state.replace_existing else blender_utils.ensure_collection(state.imported_collection_name)
    imported_objects: list[bpy.types.Object] = []

    for part in parts:
        if part.skipped or part.area <= 0 or part.local_alpha_bbox[2] <= part.local_alpha_bbox[0] or part.local_alpha_bbox[3] <= part.local_alpha_bbox[1]:
            part.skipped = True
            if not part.skip_reason:
                part.skip_reason = "empty alpha after rasterization"
            continue
        obj = alpha_mesh_adapter.build_layer_mesh(
            context,
            part,
            collection,
            grid_resolution=state.mesh_grid_resolution,
            trace_contrast_remap=(state.trace_contrast_low, state.trace_contrast_high),
        )
        part.imported_object_name = obj.name
        obj["hallway_avatar_semantic_label"] = part.semantic_label
        obj["hallway_avatar_side_guess"] = part.side_guess
        obj["hallway_avatar_confidence"] = part.confidence
        imported_objects.append(obj)

    _apply_layer_depth_stack([part for part in parts if not part.skipped], imported_objects)
    context.view_layer.update()

    remeshed_count = 0
    if state.qremeshify_settings.auto_on_import and imported_objects:
        remeshed_count = qremeshify.remesh_parts(context, parts, qremeshify.QRemeshifySettings.from_scene_state(state))
        state.remesh_performed = remeshed_count > 0
        logger.info("Auto-remeshed %s imported layer objects", remeshed_count)

    final_imported_objects = _imported_mesh_objects_for_parts(parts)
    z_offset = _apply_import_geometry_transform(final_imported_objects)
    context.view_layer.update()
    if abs(z_offset) > 1e-9:
        logger.info(
            "Scaled imported layer mesh data by %.6f and translated by %.6fm so the lowest world-space vertex rests at Z=%.5f",
            IMPORT_VERTEX_SCALE,
            z_offset,
            IMPORT_TARGET_MIN_Z,
        )
        final_min_values = [value for value in (_world_min_vertex_z(obj) for obj in final_imported_objects) if value is not None]
        if final_min_values:
            logger.info("Import geometry post-pass -> global minimum world Z = %.6f", min(final_min_values))

    mtoon_count = mtoon_materials.configure_avatar_mtoon_materials(parts)
    logger.info("Configured MToon material settings on %s imported layer materials", mtoon_count)
    if state.auto_setup_facial_video:
        try:
            face_video_obj = facial_video_preview.setup_from_state(context, parts=parts, raise_on_missing=False)
            if face_video_obj is not None:
                mouth_plane_name = str(face_video_obj.get("hallway_avatar_mouth_video_plane_object", "") or "").strip()
                logger.info(
                    "Configured facial video preview on %s%s",
                    face_video_obj.name,
                    f"; mouth video plane {mouth_plane_name}" if mouth_plane_name else "",
                )
        except Exception as exc:
            logger.exception("Facial video preview setup failed during PSD import")
            state.last_report = f"Facial video preview failed: {exc}"

    state.source_psd_path = filepath
    properties.set_layer_items(scene, parts)
    state.remeshed_count = remeshed_count
    import_report = ""
    if state.qremeshify_settings.auto_on_import and imported_objects:
        import_report = f"Imported {state.imported_count} layers, remeshed {state.remeshed_count}, skipped {state.skipped_count}"
    else:
        state.remeshed_count = 0
        state.remesh_performed = False
        import_report = f"Imported {state.imported_count} layers, skipped {state.skipped_count}"
    state.last_report = import_report

    if state.auto_rig_on_import and imported_objects:
        armature_obj, rig_plan = build_armature_scene(context, bind_weights=state.auto_bind_on_build)
        state.last_report = f"{import_report}; built {armature_obj.name} with {len(rig_plan.bones)} bones"

    logger.info(state.last_report)
    return parts


def reclassify_scene(context: bpy.types.Context) -> list:
    scene = context.scene
    parts = properties.get_parts(scene)
    part_classifier.classify_parts(parts)
    properties.set_layer_items(scene, parts)
    scene.hallway_avatar_state.last_report = f"Classified {scene.hallway_avatar_state.classified_count} layers"
    return parts


def build_armature_scene(context: bpy.types.Context, *, bind_weights: bool = False):
    scene = context.scene
    state = scene.hallway_avatar_state
    parts = properties.get_parts(scene)
    if not parts:
        raise RuntimeError("No imported layers found. Import a PSD first.")

    part_classifier.classify_parts(parts)
    properties.set_layer_items(scene, parts)

    rig_plan = heuristic_rigger.estimate_rig(parts)
    if not rig_plan.bones:
        raise RuntimeError("Unable to estimate a rig from the current layers.")

    if state.replace_existing:
        blender_utils.clear_collection(state.rig_collection_name)

    ground_offset_z = _ground_offset_from_parts(parts)
    import_scale = _import_scale_from_parts(parts)
    armature_obj = armature_builder.build_armature(
        context,
        rig_plan,
        state.rig_collection_name,
        edit_bone_offset=(0.0, 0.0, ground_offset_z),
    )
    state.armature_object_name = armature_obj.name
    armature_obj["hallway_avatar_import_scale"] = import_scale
    humanoid_count, spring_count = vrm_integration.setup_vrm1_avatar(
        context,
        armature_obj,
    )
    logger.info(
        "Configured VRM 1.0 metadata on %s -> humanoid assignments=%s hair springs=%s",
        armature_obj.name,
        humanoid_count,
        spring_count,
    )

    if bind_weights:
        weighting.bind_parts(context, armature_obj, parts, rig_plan=rig_plan)

    logger.info(
        "Built rig with import scale %.6f and edit-bone ground offset %.6f while armature object stayed at world origin",
        import_scale,
        ground_offset_z,
    )
    state.last_report = f"Built rig with {len(rig_plan.bones)} bones (confidence {rig_plan.confidence:.2f})"
    logger.info(state.last_report)
    return armature_obj, rig_plan


def bind_weights_scene(context: bpy.types.Context) -> None:
    scene = context.scene
    state = scene.hallway_avatar_state
    armature_obj = bpy.data.objects.get(state.armature_object_name)
    if armature_obj is None:
        raise RuntimeError("No generated armature found. Build the armature first.")

    parts = properties.get_parts(scene)
    weighting.bind_parts(context, armature_obj, parts)
    state.last_report = f"Bound {len([part for part in parts if not part.skipped])} layers to {armature_obj.name}"


def remesh_imported_scene(context: bpy.types.Context, *, only_selected: bool = False) -> int:
    scene = context.scene
    state = scene.hallway_avatar_state
    parts = properties.get_parts(scene)
    if not parts:
        raise RuntimeError("No imported layers found. Import a PSD first.")

    count = qremeshify.remesh_parts(
        context,
        parts,
        qremeshify.QRemeshifySettings.from_scene_state(state),
        only_selected=only_selected,
    )
    mtoon_count = mtoon_materials.configure_avatar_mtoon_materials(parts)
    logger.info("Configured MToon material settings on %s remeshed layer materials", mtoon_count)
    properties.set_layer_items(scene, parts)
    state.remeshed_count = count
    state.remesh_performed = count > 0
    state.last_report = f"Remeshed {count} imported layer objects"
    logger.info(state.last_report)
    return count
