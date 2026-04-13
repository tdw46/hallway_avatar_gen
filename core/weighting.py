from __future__ import annotations

import bpy

from ..utils.logging import get_logger
from . import heuristic_rigger
from .models import LayerPart, RigPlan

logger = get_logger("weighting")


def _ensure_armature_modifier(obj: bpy.types.Object, armature_obj: bpy.types.Object) -> None:
    modifier = obj.modifiers.get("HallwayAvatarArmature")
    if modifier is None:
        modifier = obj.modifiers.new("HallwayAvatarArmature", "ARMATURE")
    modifier.object = armature_obj


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

        _ensure_armature_modifier(obj, armature_obj)
        _clear_generated_groups(obj, armature_obj)
        bone_name = rig_plan.layer_bone_map.get(part.layer_path, "root")
        if armature_obj.data.bones.get(bone_name) is None:
            bone_name = "root" if armature_obj.data.bones.get("root") else next(iter(armature_obj.data.bones)).name
        _assign_rigid(obj, bone_name)

    logger.info("Bound %s layer objects to %s", len(parts), armature_obj.name)
