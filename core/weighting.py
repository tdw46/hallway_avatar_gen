from __future__ import annotations

import bpy
from mathutils import Vector

from ..utils import blender as blender_utils
from ..utils.logging import get_logger
from . import heuristic_rigger
from .models import LayerPart, RigPlan
from .voxel_binding import VoxelBindingSettings, run_voxel_heat_diffuse

logger = get_logger("weighting")

HAIR_SMOOTH_REPEAT = 80
SPLIT_FRONT_HAIR_PRE_BRIDGE_SMOOTH_REPEAT = 60
SPLIT_FRONT_HAIR_POST_BRIDGE_SMOOTH_REPEAT = 20
NECK_BLEND_SMOOTH_REPEAT = 40
OTHER_SMOOTH_REPEAT = 100
HAIR_BONE_PREFIXES = ("front_hair_", "back_hair_")
HEAD_PRIORITY_TOKENS = {
    "topwear",
    "face",
    "ears",
    "earwear",
    "mouth",
    "nose",
    "eyewhite",
    "eyelash",
    "eyebrow",
    "eyewear",
    "headwear",
}


def _ensure_armature_modifier(obj: bpy.types.Object, armature_obj: bpy.types.Object) -> None:
    armature_modifiers = [modifier for modifier in obj.modifiers if modifier.type == "ARMATURE"]
    preferred = obj.modifiers.get("HallwayAvatarArmature")
    if preferred is None:
        preferred = next(
            (modifier for modifier in armature_modifiers if getattr(modifier, "object", None) == armature_obj),
            armature_modifiers[0] if armature_modifiers else None,
        )
    if preferred is None:
        preferred = obj.modifiers.new("HallwayAvatarArmature", "ARMATURE")
    preferred.name = "HallwayAvatarArmature"
    preferred.object = armature_obj

    for modifier in list(obj.modifiers):
        if modifier.type != "ARMATURE" or modifier == preferred:
            continue
        obj.modifiers.remove(modifier)


def _set_armature_parent_keep_transform(obj: bpy.types.Object, armature_obj: bpy.types.Object) -> None:
    world_matrix = obj.matrix_world.copy()
    obj.parent = armature_obj
    obj.parent_type = "OBJECT"
    obj.matrix_parent_inverse = armature_obj.matrix_world.inverted_safe()
    obj.matrix_world = world_matrix


def _clear_generated_groups(obj: bpy.types.Object, armature_obj: bpy.types.Object) -> None:
    bone_names = {bone.name for bone in armature_obj.data.bones}
    for group in list(obj.vertex_groups):
        if group.name in bone_names:
            obj.vertex_groups.remove(group)


def _ensure_group(obj: bpy.types.Object, bone_name: str) -> bpy.types.VertexGroup:
    group = obj.vertex_groups.get(bone_name)
    if group is None:
        group = obj.vertex_groups.new(name=bone_name)
    return group


def _group_weight(group: bpy.types.VertexGroup | None, vertex_index: int) -> float:
    if group is None:
        return 0.0
    try:
        return group.weight(vertex_index)
    except RuntimeError:
        return 0.0


def _set_normalized_weights(
    vertex_index: int,
    weight_map: dict[bpy.types.VertexGroup, float],
) -> None:
    clamped = {group: max(0.0, weight) for group, weight in weight_map.items() if group is not None}
    total = sum(clamped.values())
    if total <= 1e-8:
        return
    for group, weight in clamped.items():
        group.add([vertex_index], weight / total, "REPLACE")


def _assign_rigid(obj: bpy.types.Object, bone_name: str) -> None:
    group = _ensure_group(obj, bone_name)
    indices = [vertex.index for vertex in obj.data.vertices]
    group.add(indices, 1.0, "REPLACE")


def _clear_armature_modifiers(obj: bpy.types.Object) -> None:
    for modifier in list(obj.modifiers):
        if modifier.type == "ARMATURE":
            obj.modifiers.remove(modifier)


def _triangulate_mesh(context: bpy.types.Context, obj: bpy.types.Object) -> None:
    if obj.type != "MESH" or not obj.data.polygons:
        return

    previous_active = context.view_layer.objects.active
    previous_selection = list(context.selected_objects)

    try:
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
        bpy.ops.object.mode_set(mode="OBJECT")
        logger.info("Triangulated %s before binding", obj.name)
    finally:
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        for selected in previous_selection:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            context.view_layer.objects.active = previous_active


def _parent_to_armature(context: bpy.types.Context, obj: bpy.types.Object, armature_obj: bpy.types.Object) -> None:
    previous_active = context.view_layer.objects.active
    previous_selection = list(context.selected_objects)

    try:
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        armature_obj.select_set(True)
        context.view_layer.objects.active = armature_obj
        bpy.ops.object.parent_set(type="ARMATURE")
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for selected in previous_selection:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            context.view_layer.objects.active = previous_active


def _smooth_weights(context: bpy.types.Context, obj: bpy.types.Object, repeat: int) -> None:
    if repeat <= 0 or len(obj.vertex_groups) == 0:
        return

    previous_active = context.view_layer.objects.active
    previous_selection = list(context.selected_objects)

    try:
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
        bpy.ops.object.vertex_group_smooth(
            group_select_mode="ALL",
            factor=0.5,
            repeat=repeat,
            expand=0.0,
        )
        bpy.ops.object.mode_set(mode="OBJECT")
        logger.info("Smoothed weights on %s with %s repeats", obj.name, repeat)
    finally:
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        for selected in previous_selection:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            context.view_layer.objects.active = previous_active


def _filtered_bone_names_for_part(
    part: LayerPart,
    armature_obj: bpy.types.Object,
    bone_names: tuple[str, ...],
) -> tuple[str, ...]:
    token = heuristic_rigger._canonical_token(part)
    valid = tuple(name for name in bone_names if armature_obj.data.bones.get(name) is not None)

    if token == "front hair":
        return tuple(name for name in valid if name.startswith("front_hair_"))
    if token == "back hair":
        return tuple(name for name in valid if name.startswith("back_hair_"))
    return tuple(name for name in valid if not name.startswith(HAIR_BONE_PREFIXES))


def _apply_voxel_weights(
    context: bpy.types.Context,
    part: LayerPart,
    obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    bone_names: tuple[str, ...],
) -> bool:
    valid_bones = _filtered_bone_names_for_part(part, armature_obj, bone_names)
    if not valid_bones:
        return False

    if context.object and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    try:
        _clear_generated_groups(obj, armature_obj)
        _clear_armature_modifiers(obj)
        if obj.parent == armature_obj:
            blender_utils.set_active_object(context, obj)
            bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

        run_voxel_heat_diffuse(
            context,
            armature_obj,
            [obj],
            valid_bones,
            settings=VoxelBindingSettings(),
        )
        _parent_to_armature(context, obj, armature_obj)
        _ensure_armature_modifier(obj, armature_obj)
        return True
    except Exception as exc:
        logger.warning("Voxel binding failed for %s with bones %s: %s", obj.name, valid_bones, exc)
        return False
    finally:
        bpy.ops.object.select_all(action="DESELECT")


def _override_head_weights(
    obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    bone_names: tuple[str, ...],
    *,
    mode: str = "region",
) -> None:
    if "head" not in bone_names or armature_obj.data.bones.get("head") is None:
        return

    head_group = obj.vertex_groups.get("head")
    if head_group is None:
        return

    other_groups = [obj.vertex_groups.get(name) for name in bone_names if name != "head"]
    other_groups = [group for group in other_groups if group is not None]
    head_threshold_z = (armature_obj.matrix_world @ armature_obj.data.bones["head"].head_local).z

    for vertex in obj.data.vertices:
        if mode == "head_weight":
            try:
                head_weight = head_group.weight(vertex.index)
            except RuntimeError:
                head_weight = 0.0
            if head_weight <= 1e-6:
                continue
        else:
            world_z = (obj.matrix_world @ vertex.co).z
            if world_z < head_threshold_z:
                continue
        head_group.add([vertex.index], 1.0, "REPLACE")
        for group in other_groups:
            group.add([vertex.index], 0.0, "REPLACE")


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - (2.0 * value))


def _blend_neck_head_weights(
    obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    bone_names: tuple[str, ...],
) -> bool:
    if "neck" not in bone_names or "head" not in bone_names:
        return False
    if armature_obj.data.bones.get("neck") is None or armature_obj.data.bones.get("head") is None:
        return False
    if obj.type != "MESH" or obj.data is None or not obj.data.vertices:
        return False

    neck_group = _ensure_group(obj, "neck")
    head_group = _ensure_group(obj, "head")
    blend_groups = [neck_group, head_group]
    spine_group = obj.vertex_groups.get("spine") if "spine" in bone_names else None
    if spine_group is not None:
        blend_groups.append(spine_group)

    world_positions = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    min_z = min(co.z for co in world_positions)
    max_z = max(co.z for co in world_positions)
    height = max(max_z - min_z, 1.0e-6)
    affected = 0

    for vertex, world_co in zip(obj.data.vertices, world_positions, strict=False):
        z_ratio = (world_co.z - min_z) / height
        head_weight = _smoothstep((z_ratio - 0.30) / 0.65)
        spine_weight = max(0.0, 1.0 - (z_ratio / 0.30)) * 0.20 if spine_group is not None else 0.0
        neck_weight = max(0.0, 1.0 - head_weight - spine_weight)
        weights: dict[bpy.types.VertexGroup, float] = {
            neck_group: neck_weight,
            head_group: head_weight,
        }
        if spine_group is not None and spine_weight > 0.0:
            weights[spine_group] = spine_weight
        for group in blend_groups:
            if group not in weights:
                weights[group] = 0.0
        _set_normalized_weights(vertex.index, weights)
        affected += 1

    logger.info(
        "Blended neck/head weights on %s -> affected=%s min_z=%.6f max_z=%.6f",
        obj.name,
        affected,
        min_z,
        max_z,
    )
    return affected > 0


def _apply_split_front_hair_head_bridge(
    obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    bone_names: tuple[str, ...],
) -> bool:
    if armature_obj.data.bones.get("head") is None:
        return False
    if not any(name.startswith("front_hair_left_") for name in bone_names):
        return False
    if not any(name.startswith("front_hair_right_") for name in bone_names):
        return False

    head_bone = armature_obj.data.bones["head"]
    head_head_world = armature_obj.matrix_world @ head_bone.head_local
    head_tail_world = armature_obj.matrix_world @ head_bone.tail_local
    center_x = (head_head_world.x + head_tail_world.x) * 0.5
    head_mid_z = (head_head_world.z + head_tail_world.z) * 0.5
    head_top_z = max(head_head_world.z, head_tail_world.z)
    head_length = max((head_tail_world - head_head_world).length, 1e-6)

    world_positions = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    if not world_positions:
        return False
    object_x_values = [co.x for co in world_positions]
    object_z_values = [co.z for co in world_positions]
    object_top_z = max(object_z_values)
    top_band_bottom_z = head_mid_z
    top_band_top_z = max(object_top_z, head_top_z)
    if top_band_top_z <= top_band_bottom_z:
        top_band_top_z = top_band_bottom_z + max(head_length * 0.25, 1e-4)

    center_half_width = max(head_length * 0.35, (max(object_x_values) - min(object_x_values)) * 0.12, 1e-4)
    head_group = _ensure_group(obj, "head")
    strand_groups = [
        obj.vertex_groups.get(name)
        for name in bone_names
        if name != "head" and (name.startswith("front_hair_left_") or name.startswith("front_hair_right_"))
    ]
    strand_groups = [group for group in strand_groups if group is not None]
    if not strand_groups:
        return False

    affected = 0
    for vertex, world_co in zip(obj.data.vertices, world_positions, strict=False):
        if world_co.z < top_band_bottom_z:
            continue
        x_factor = 1.0 - min(abs(world_co.x - center_x) / center_half_width, 1.0)
        if x_factor <= 0.0:
            continue
        z_factor = min(max((world_co.z - top_band_bottom_z) / (top_band_top_z - top_band_bottom_z), 0.0), 1.0)
        target_head_weight = x_factor * z_factor
        if target_head_weight <= 0.0:
            continue

        current_head_weight = _group_weight(head_group, vertex.index)
        new_head_weight = max(current_head_weight, target_head_weight)
        remaining_scale = max(0.0, 1.0 - new_head_weight)
        weight_map: dict[bpy.types.VertexGroup, float] = {head_group: new_head_weight}
        for group in strand_groups:
            current_weight = _group_weight(group, vertex.index)
            if current_weight > 0.0:
                weight_map[group] = current_weight * remaining_scale
        _set_normalized_weights(vertex.index, weight_map)
        affected += 1

    if affected:
        logger.info(
            "Applied split front hair head bridge on %s -> affected=%s center_x=%.6f top_band_bottom_z=%.6f top_band_top_z=%.6f center_half_width=%.6f",
            obj.name,
            affected,
            center_x,
            top_band_bottom_z,
            top_band_top_z,
            center_half_width,
        )
    return affected > 0


def bind_parts(
    context: bpy.types.Context,
    armature_obj: bpy.types.Object,
    parts: list[LayerPart],
    *,
    rig_plan: RigPlan | None = None,
) -> None:
    if rig_plan is None:
        rig_plan = heuristic_rigger.estimate_rig(parts)

    for part in parts:
        if part.skipped or not part.imported_object_name:
            continue
        obj = bpy.data.objects.get(part.imported_object_name)
        if obj is None or obj.type != "MESH":
            continue

        auto_bones = rig_plan.layer_auto_weight_bones.get(part.layer_path)
        token = heuristic_rigger._canonical_token(part)
        if auto_bones:
            filtered_auto_bones = _filtered_bone_names_for_part(part, armature_obj, auto_bones)
            success = _apply_voxel_weights(context, part, obj, armature_obj, filtered_auto_bones)
            if success:
                if token == "front hair" and _apply_split_front_hair_head_bridge(obj, armature_obj, filtered_auto_bones):
                    _smooth_weights(context, obj, SPLIT_FRONT_HAIR_PRE_BRIDGE_SMOOTH_REPEAT)
                    _apply_split_front_hair_head_bridge(obj, armature_obj, filtered_auto_bones)
                    _smooth_weights(context, obj, SPLIT_FRONT_HAIR_POST_BRIDGE_SMOOTH_REPEAT)
                elif token in heuristic_rigger.NECK_TOKENS and _blend_neck_head_weights(obj, armature_obj, filtered_auto_bones):
                    _smooth_weights(context, obj, NECK_BLEND_SMOOTH_REPEAT)
                else:
                    _smooth_weights(
                        context,
                        obj,
                        HAIR_SMOOTH_REPEAT if token in {"front hair", "back hair"} else OTHER_SMOOTH_REPEAT,
                    )
                if token == "topwear":
                    _override_head_weights(obj, armature_obj, filtered_auto_bones)
                elif token in HEAD_PRIORITY_TOKENS:
                    _override_head_weights(obj, armature_obj, filtered_auto_bones, mode="head_weight")
                continue

        _ensure_armature_modifier(obj, armature_obj)
        _clear_generated_groups(obj, armature_obj)
        bone_name = rig_plan.layer_bone_map.get(part.layer_path, "root")
        if armature_obj.data.bones.get(bone_name) is None:
            bone_name = "root" if armature_obj.data.bones.get("root") else next(iter(armature_obj.data.bones)).name
        _assign_rigid(obj, bone_name)
        if token in heuristic_rigger.NECK_TOKENS:
            _blend_neck_head_weights(
                obj,
                armature_obj,
                tuple(name for name in ("spine", "neck", "head") if armature_obj.data.bones.get(name)),
            )
        _smooth_weights(context, obj, OTHER_SMOOTH_REPEAT)
        _set_armature_parent_keep_transform(obj, armature_obj)

    logger.info("Bound %s layer objects to %s", len(parts), armature_obj.name)
