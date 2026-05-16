from __future__ import annotations

from collections.abc import Iterable
import importlib
import re

import bpy

from ..utils.logging import get_logger

logger = get_logger("vrm_integration")

HALLWAY_SPRING_PREFIX = "Hallway Hair"

SPRING_HIT_RADIUS = 0.02
SPRING_STIFFNESS = 0.3
SPRING_DRAG_FORCE = 0.6
SPRING_GRAVITY_POWER = 0.01
SPRING_GRAVITY_DIR = (0.0, 0.0, -1.0)


def _vrm_human_bones_type():
    for package_name in ("vrm", "bl_ext.user_default.vrm", "bl_ext.blender_org.vrm"):
        try:
            module = importlib.import_module(f"{package_name}.editor.vrm1.property_group")
        except Exception:
            continue
        human_bones_type = getattr(module, "Vrm1HumanBonesPropertyGroup", None)
        if human_bones_type is not None:
            return human_bones_type
    return None


def _set_spec_version_vrm1(armature_obj: bpy.types.Object) -> object | None:
    ext = getattr(armature_obj.data, "vrm_addon_extension", None)
    if ext is None:
        logger.info("VRM armature extension data unavailable; skipping VRM setup")
        return None

    vrm1_value = getattr(ext, "SPEC_VERSION_VRM1", "1.0")
    if getattr(ext, "spec_version", "") != vrm1_value:
        ext.spec_version = vrm1_value
    return ext


def _assign_if_exists(
    human_bones: object,
    human_bone_attr: str,
    armature_bones: bpy.types.bpy_prop_collection,
    bone_name: str,
) -> bool:
    if bone_name not in armature_bones:
        return False
    human_bone = getattr(human_bones, human_bone_attr, None)
    if human_bone is None or not hasattr(human_bone, "node"):
        return False
    human_bone.node.bone_name = bone_name
    return True


def assign_vrm1_humanoid_bones(
    context: bpy.types.Context,
    armature_obj: bpy.types.Object,
) -> int:
    if armature_obj.type != "ARMATURE":
        return 0

    ext = _set_spec_version_vrm1(armature_obj)
    human_bones_type = _vrm_human_bones_type()
    if ext is None or human_bones_type is None:
        if human_bones_type is None:
            logger.info(
                "VRM 1.0 human-bone helper API unavailable; skipping humanoid assignment"
            )
        armature_obj["hallway_avatar_vrm1_humanoid_assignments"] = 0
        return 0

    armature_data = armature_obj.data
    human_bones = ext.vrm1.humanoid.human_bones

    human_bones_type.fixup_human_bones(armature_obj)
    human_bones_type.update_all_bone_name_candidates(
        context, armature_data.name, force=True
    )

    for human_bone in human_bones.human_bone_name_to_human_bone().values():
        human_bone.node.bone_name = ""

    assignments = {
        "hips": "hips",
        "spine": "spine",
        "chest": "torso",
        "neck": "neck",
        "head": "head",
        "left_upper_arm": "leftArm",
        "left_lower_arm": "leftElbow",
        "left_hand": "leftHand",
        "right_upper_arm": "rightArm",
        "right_lower_arm": "rightElbow",
        "right_hand": "rightHand",
        "left_upper_leg": "leftLeg",
        "left_lower_leg": "leftKnee",
        "left_foot": "leftFoot",
        "right_upper_leg": "rightLeg",
        "right_lower_leg": "rightKnee",
        "right_foot": "rightFoot",
    }

    assigned = 0
    for human_bone_attr, bone_name in assignments.items():
        if _assign_if_exists(
            human_bones,
            human_bone_attr,
            armature_data.bones,
            bone_name,
        ):
            assigned += 1

    # Hallway's current 2.5-D rig often has no discrete hand or foot bones yet.
    if hasattr(human_bones, "allow_non_humanoid_rig"):
        human_bones.allow_non_humanoid_rig = True
    if hasattr(human_bones, "initial_automatic_bone_assignment"):
        human_bones.initial_automatic_bone_assignment = False
    human_bones_type.fixup_human_bones(armature_obj)
    human_bones_type.update_all_bone_name_candidates(
        context, armature_data.name, force=True
    )

    armature_obj["hallway_avatar_vrm1_humanoid_assignments"] = assigned
    logger.info("Assigned %s VRM 1.0 humanoid bones on %s", assigned, armature_obj.name)
    return assigned


def _strand_key(bone_name: str) -> str | None:
    if bone_name.startswith("front_hair_left_"):
        return "front_hair_left"
    if bone_name.startswith("front_hair_right_"):
        return "front_hair_right"
    if re.match(r"^front_hair_\d+", bone_name):
        return "front_hair"
    if re.match(r"^back_hair_\d+", bone_name):
        return "back_hair"
    return None


def _same_strand_children(bone: bpy.types.Bone, key: str) -> list[bpy.types.Bone]:
    return sorted(
        (child for child in bone.children if _strand_key(child.name) == key),
        key=lambda child: child.name,
    )


def _hair_chains(armature_obj: bpy.types.Object) -> list[tuple[str, ...]]:
    bones = armature_obj.data.bones
    top_bones: list[bpy.types.Bone] = []
    for bone in bones:
        key = _strand_key(bone.name)
        if key is None:
            continue
        parent = bone.parent
        if parent is None or _strand_key(parent.name) != key:
            top_bones.append(bone)

    chains: list[tuple[str, ...]] = []
    for top_bone in sorted(top_bones, key=lambda bone: bone.name):
        key = _strand_key(top_bone.name)
        if key is None:
            continue
        chain = [top_bone.name]
        current = top_bone
        while True:
            children = _same_strand_children(current, key)
            if not children:
                break
            current = children[0]
            chain.append(current.name)
        if len(chain) >= 2:
            chains.append(tuple(chain))
    return chains


def _remove_existing_hallway_springs(armature_obj: bpy.types.Object) -> None:
    ext = _set_spec_version_vrm1(armature_obj)
    if ext is None:
        return
    springs = ext.spring_bone1.springs
    for index in reversed(range(len(springs))):
        spring_name = getattr(springs[index], "vrm_name", "")
        if not spring_name.startswith(HALLWAY_SPRING_PREFIX):
            continue
        result = bpy.ops.vrm.remove_spring_bone1_spring(
            armature_object_name=armature_obj.name,
            spring_index=index,
        )
        if result != {"FINISHED"}:
            springs.remove(index)


def _add_spring(
    armature_obj: bpy.types.Object,
    name: str,
    joint_bone_names: Iterable[str],
) -> bool:
    result = bpy.ops.vrm.add_spring_bone1_spring(
        armature_object_name=armature_obj.name
    )
    if result != {"FINISHED"}:
        return False

    spring_bone1 = armature_obj.data.vrm_addon_extension.spring_bone1
    spring_index = len(spring_bone1.springs) - 1
    spring = spring_bone1.springs[spring_index]
    spring.vrm_name = name

    added = 0
    for bone_name in joint_bone_names:
        result = bpy.ops.vrm.add_spring_bone1_spring_joint(
            armature_object_name=armature_obj.name,
            spring_index=spring_index,
            guess_properties=False,
        )
        if result != {"FINISHED"}:
            continue
        joint = spring.joints[-1]
        joint.node.bone_name = bone_name
        joint.hit_radius = SPRING_HIT_RADIUS
        joint.stiffness = SPRING_STIFFNESS
        joint.drag_force = SPRING_DRAG_FORCE
        joint.gravity_power = SPRING_GRAVITY_POWER
        joint.gravity_dir = SPRING_GRAVITY_DIR
        added += 1

    if added == 0:
        bpy.ops.vrm.remove_spring_bone1_spring(
            armature_object_name=armature_obj.name,
            spring_index=spring_index,
        )
        return False

    return True


def generate_hair_spring_bones(armature_obj: bpy.types.Object) -> int:
    if armature_obj.type != "ARMATURE":
        return 0

    ext = _set_spec_version_vrm1(armature_obj)
    if ext is None:
        armature_obj["hallway_avatar_vrm1_hair_springs"] = 0
        return 0

    _remove_existing_hallway_springs(armature_obj)

    created = 0
    for chain in _hair_chains(armature_obj):
        joint_bone_names = chain[1:]
        if not joint_bone_names:
            continue
        spring_name = f"{HALLWAY_SPRING_PREFIX} {chain[0]}"
        if _add_spring(armature_obj, spring_name, joint_bone_names):
            created += 1

    ext.spring_bone1.enable_animation = True
    armature_obj["hallway_avatar_vrm1_hair_springs"] = created
    logger.info("Generated %s VRM 1.0 hair springs on %s", created, armature_obj.name)
    return created


def setup_vrm1_avatar(
    context: bpy.types.Context,
    armature_obj: bpy.types.Object,
) -> tuple[int, int]:
    humanoid_count = assign_vrm1_humanoid_bones(context, armature_obj)
    spring_count = generate_hair_spring_bones(armature_obj)
    return humanoid_count, spring_count
