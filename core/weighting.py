from __future__ import annotations

import bpy

from ..utils import blender as blender_utils
from ..utils.logging import get_logger
from . import heuristic_rigger
from .models import LayerPart, RigPlan
from .voxel_binding import VoxelBindingSettings, run_voxel_heat_diffuse

logger = get_logger("weighting")

HAIR_SMOOTH_REPEAT = 80
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
        _smooth_weights(context, obj, OTHER_SMOOTH_REPEAT)

    logger.info("Bound %s layer objects to %s", len(parts), armature_obj.name)
