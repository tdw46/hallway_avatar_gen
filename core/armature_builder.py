from __future__ import annotations

import bpy
from mathutils import Vector

from ..utils import blender as blender_utils
from ..utils.logging import get_logger
from .models import RigPlan

logger = get_logger("armature_builder")


def build_armature(
    context: bpy.types.Context,
    rig_plan: RigPlan,
    collection_name: str,
    *,
    object_name: str = "HallwayAvatarRig",
    edit_bone_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    collection = blender_utils.ensure_collection(collection_name)

    if context.object and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    armature_data = bpy.data.armatures.new(f"{object_name}_Data")
    armature_obj = bpy.data.objects.new(object_name, armature_data)
    armature_obj.show_in_front = True
    collection.objects.link(armature_obj)

    blender_utils.set_active_object(context, armature_obj)
    bpy.ops.object.mode_set(mode="EDIT")

    edit_bones: dict[str, bpy.types.EditBone] = {}
    bone_offset = Vector(edit_bone_offset)
    for bone_plan in rig_plan.bones.values():
        edit_bone = armature_data.edit_bones.new(bone_plan.name)
        edit_bone.head = Vector(bone_plan.head) + bone_offset
        edit_bone.tail = Vector(bone_plan.tail) + bone_offset
        if (edit_bone.tail - edit_bone.head).length < 0.001:
            edit_bone.tail.z += 0.05
        edit_bones[bone_plan.name] = edit_bone

    for bone_plan in rig_plan.bones.values():
        if bone_plan.parent and bone_plan.parent in edit_bones:
            edit_bones[bone_plan.name].parent = edit_bones[bone_plan.parent]
            edit_bones[bone_plan.name].use_connect = bone_plan.connected

    bpy.ops.object.mode_set(mode="OBJECT")
    bone_collections: dict[str, bpy.types.BoneCollection] = {}
    collection_names = list(rig_plan.bone_collection_names)
    for bone_plan in rig_plan.bones.values():
        if bone_plan.collection_name and bone_plan.collection_name not in collection_names:
            collection_names.append(bone_plan.collection_name)
    for collection_name in collection_names:
        collection = armature_data.collections.get(collection_name)
        if collection is None:
            collection = armature_data.collections.new(collection_name)
        bone_collections[collection_name] = collection
    for bone_plan in rig_plan.bones.values():
        collection_name = bone_plan.collection_name or "Body"
        bone = armature_data.bones.get(bone_plan.name)
        collection = bone_collections.get(collection_name)
        if bone is not None and collection is not None:
            collection.assign(bone)

    armature_obj["hallway_avatar_edit_bone_offset_x"] = bone_offset.x
    armature_obj["hallway_avatar_edit_bone_offset_y"] = bone_offset.y
    armature_obj["hallway_avatar_edit_bone_offset_z"] = bone_offset.z
    logger.info("Built armature %s with %s bones", armature_obj.name, len(rig_plan.bones))
    return armature_obj
