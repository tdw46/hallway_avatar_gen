from __future__ import annotations

import bpy
from mathutils import Vector
from mathutils.kdtree import KDTree

from ..utils import blender as blender_utils
from ..utils.logging import get_logger
from . import heuristic_rigger
from .models import LayerPart, RigPlan
from .voxel_binding import VoxelBindingSettings, run_voxel_heat_diffuse

logger = get_logger("weighting")

HAIR_SMOOTH_REPEAT = 80
SPLIT_FRONT_HAIR_PRE_BRIDGE_SMOOTH_REPEAT = 60
SPLIT_FRONT_HAIR_POST_BRIDGE_SMOOTH_REPEAT = 20
NECK_BLEND_SMOOTH_REPEAT = 20
NECK_HEAD_SOFTEN_SMOOTH_REPEAT = 10
OTHER_SMOOTH_REPEAT = 20
TOPWEAR_NECK_BRIDGE_SMOOTH_REPEAT = 100
TOPWEAR_NECK_JOINT_SMOOTH_REPEAT = 4
TOPWEAR_NECK_JOINT_SMOOTH_FACTOR = 0.20
TOPWEAR_NECK_JOINT_SMOOTH_RADIUS_RATIO = 0.045
TOPWEAR_NECK_MAX_WEIGHT = 1.0
TOPWEAR_NECK_START_RATIO = 0.50
TOPWEAR_NECK_BLEND_RATIO = 0.32
TOPWEAR_NECK_CENTER_WIDTH_RATIO = 0.32
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


def _apply_joint_voxel_weights(
    context: bpy.types.Context,
    objs: list[bpy.types.Object],
    armature_obj: bpy.types.Object,
    bone_names: tuple[str, ...],
) -> bool:
    valid_bones = tuple(name for name in bone_names if armature_obj.data.bones.get(name) is not None)
    mesh_objs = [obj for obj in objs if obj.type == "MESH" and obj.data is not None and obj.data.vertices]
    if not valid_bones or not mesh_objs:
        return False

    if context.object and context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    try:
        for obj in mesh_objs:
            _clear_generated_groups(obj, armature_obj)
            _clear_armature_modifiers(obj)
            if obj.parent == armature_obj:
                blender_utils.set_active_object(context, obj)
                bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

        run_voxel_heat_diffuse(
            context,
            armature_obj,
            mesh_objs,
            valid_bones,
            settings=VoxelBindingSettings(),
        )
        for obj in mesh_objs:
            _parent_to_armature(context, obj, armature_obj)
            _ensure_armature_modifier(obj, armature_obj)
        return True
    except Exception as exc:
        logger.warning("Joint voxel binding failed for %s with bones %s: %s", ", ".join(obj.name for obj in mesh_objs), valid_bones, exc)
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


def _head_protected_vertex(
    obj: bpy.types.Object,
    vertex_index: int,
    *,
    head_threshold_z: float | None = None,
    world_co: Vector | None = None,
    min_head_weight: float = 0.45,
) -> bool:
    head_group = obj.vertex_groups.get("head")
    if _group_weight(head_group, vertex_index) >= min_head_weight:
        return True
    return head_threshold_z is not None and world_co is not None and world_co.z >= head_threshold_z


def _blend_neck_head_weights(
    obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    bone_names: tuple[str, ...],
    *,
    head_bottom_z: float | None = None,
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
    torso_group = obj.vertex_groups.get("torso") if "torso" in bone_names else None
    if torso_group is not None:
        blend_groups.append(torso_group)

    world_positions = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    min_z = min(co.z for co in world_positions)
    max_z = max(co.z for co in world_positions)
    height = max(max_z - min_z, 1.0e-6)
    head_fade_bottom_z = min(max(head_bottom_z if head_bottom_z is not None else max_z - (height * 0.18), min_z), max_z - 1.0e-6)
    head_full_bottom_z = max(head_fade_bottom_z + 1.0e-6, max_z - (height * 0.025))
    head_fade_height = max(head_full_bottom_z - head_fade_bottom_z, 1.0e-6)
    affected = 0

    for vertex, world_co in zip(obj.data.vertices, world_positions, strict=False):
        z_ratio = (world_co.z - min_z) / height
        if world_co.z >= head_full_bottom_z:
            head_weight = 1.0
        else:
            head_weight = min(max((world_co.z - head_fade_bottom_z) / head_fade_height, 0.0), 1.0)
        torso_weight = max(0.0, 1.0 - (z_ratio / 0.42)) * 0.28 if torso_group is not None else 0.0
        neck_weight = max(0.0, 1.0 - head_weight - torso_weight)
        weights: dict[bpy.types.VertexGroup, float] = {
            neck_group: neck_weight,
            head_group: head_weight,
        }
        if torso_group is not None and torso_weight > 0.0:
            weights[torso_group] = torso_weight
        for group in blend_groups:
            if group not in weights:
                weights[group] = 0.0
        _set_normalized_weights(vertex.index, weights)
        affected += 1

    logger.info(
        "Blended neck/head weights on %s -> affected=%s min_z=%.6f max_z=%.6f head_fade_bottom_z=%.6f head_full_bottom_z=%.6f",
        obj.name,
        affected,
        min_z,
        max_z,
        head_fade_bottom_z,
        head_full_bottom_z,
    )
    return affected > 0


def _topwear_neck_allowed_weight(
    world_co: Vector,
    *,
    min_z: float,
    height: float,
    neck_center_x: float,
    center_half_width: float,
    max_neck_weight: float,
) -> float:
    z_ratio = (world_co.z - min_z) / height
    if z_ratio < TOPWEAR_NECK_START_RATIO:
        return 0.0
    vertical_factor = _smoothstep((z_ratio - TOPWEAR_NECK_START_RATIO) / TOPWEAR_NECK_BLEND_RATIO)
    center_raw = 1.0 - min(abs(world_co.x - neck_center_x) / center_half_width, 1.0)
    if center_raw <= 0.0:
        return 0.0
    center_factor = 1.0 if center_raw >= 0.70 else _smoothstep((center_raw - 0.20) / 0.50)
    return max_neck_weight * vertical_factor * _smoothstep(center_factor)


def _apply_topwear_neck_bridge(
    obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    bone_names: tuple[str, ...],
) -> bool:
    if "neck" not in bone_names or armature_obj.data.bones.get("neck") is None:
        return False
    if obj.type != "MESH" or obj.data is None or not obj.data.vertices:
        return False

    neck_group = _ensure_group(obj, "neck")
    blend_groups = [
        group
        for group in (
            obj.vertex_groups.get("root"),
            obj.vertex_groups.get("hips"),
            obj.vertex_groups.get("torso"),
            obj.vertex_groups.get("spine"),
            neck_group,
            obj.vertex_groups.get("head"),
        )
        if group is not None
    ]
    if len(blend_groups) <= 1:
        return False

    world_positions = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    min_z = min(co.z for co in world_positions)
    max_z = max(co.z for co in world_positions)
    min_x = min(co.x for co in world_positions)
    max_x = max(co.x for co in world_positions)
    height = max(max_z - min_z, 1.0e-6)
    width = max(max_x - min_x, 1.0e-6)

    neck_bone = armature_obj.data.bones["neck"]
    neck_head_world = armature_obj.matrix_world @ neck_bone.head_local
    neck_tail_world = armature_obj.matrix_world @ neck_bone.tail_local
    neck_center_x = (neck_head_world.x + neck_tail_world.x) * 0.5
    center_half_width = max(width * TOPWEAR_NECK_CENTER_WIDTH_RATIO, 1.0e-6)
    max_neck_weight = TOPWEAR_NECK_MAX_WEIGHT
    affected = 0
    head_threshold_z = None
    if armature_obj.data.bones.get("head") is not None:
        head_threshold_z = (armature_obj.matrix_world @ armature_obj.data.bones["head"].head_local).z

    for vertex, world_co in zip(obj.data.vertices, world_positions, strict=False):
        if _head_protected_vertex(obj, vertex.index, head_threshold_z=head_threshold_z, world_co=world_co):
            continue
        target_neck_weight = _topwear_neck_allowed_weight(
            world_co,
            min_z=min_z,
            height=height,
            neck_center_x=neck_center_x,
            center_half_width=center_half_width,
            max_neck_weight=max_neck_weight,
        )
        if target_neck_weight <= 1.0e-5:
            continue

        current_neck_weight = _group_weight(neck_group, vertex.index)
        if current_neck_weight >= target_neck_weight:
            continue

        remaining_scale = max(0.0, 1.0 - target_neck_weight)
        existing_total = sum(
            _group_weight(group, vertex.index)
            for group in blend_groups
            if group != neck_group
        )
        weights: dict[bpy.types.VertexGroup, float] = {neck_group: target_neck_weight}
        if existing_total <= 1.0e-8:
            spine_group = obj.vertex_groups.get("spine") or obj.vertex_groups.get("torso")
            if spine_group is not None:
                weights[spine_group] = remaining_scale
        else:
            for group in blend_groups:
                if group == neck_group:
                    continue
                current_weight = _group_weight(group, vertex.index)
                if current_weight > 0.0:
                    weights[group] = (current_weight / existing_total) * remaining_scale
        _set_normalized_weights(vertex.index, weights)
        affected += 1

    if affected:
        logger.info(
            "Applied topwear neck bridge on %s -> affected=%s max_weight=%.2f top_band=%.2f center_half_width=%.6f",
            obj.name,
            affected,
            max_neck_weight,
            TOPWEAR_NECK_START_RATIO,
            center_half_width,
        )
    return affected > 0


def _smooth_topwear_neck_joint_weights(
    topwear_objs: list[bpy.types.Object],
    neck_objs: list[bpy.types.Object],
    armature_obj: bpy.types.Object,
    bone_names: tuple[str, ...],
) -> bool:
    mesh_objs = [
        obj
        for obj in topwear_objs + neck_objs
        if obj.type == "MESH" and obj.data is not None and obj.data.vertices
    ]
    group_names = tuple(name for name in ("root", "hips", "spine", "torso", "neck", "head") if name in bone_names)
    if len(mesh_objs) < 2 or not group_names:
        return False

    vertices: list[tuple[bpy.types.Object, bpy.types.MeshVertex, Vector, bool]] = []
    for obj in mesh_objs:
        is_topwear = obj in topwear_objs
        vertices.extend((obj, vertex, obj.matrix_world @ vertex.co, is_topwear) for vertex in obj.data.vertices)
    if len(vertices) < 2:
        return False

    min_x = min(world_co.x for _, _, world_co, _ in vertices)
    max_x = max(world_co.x for _, _, world_co, _ in vertices)
    min_z = min(world_co.z for _, _, world_co, _ in vertices)
    max_z = max(world_co.z for _, _, world_co, _ in vertices)
    radius = max(max_x - min_x, max_z - min_z, 1.0e-6) * TOPWEAR_NECK_JOINT_SMOOTH_RADIUS_RATIO
    if radius <= 1.0e-8:
        return False

    head_threshold_z = None
    if armature_obj.data.bones.get("head") is not None:
        head_threshold_z = (armature_obj.matrix_world @ armature_obj.data.bones["head"].head_local).z
    changed = False

    for _ in range(TOPWEAR_NECK_JOINT_SMOOTH_REPEAT):
        samples: list[dict[str, float]] = []
        kd_tree = KDTree(len(vertices))
        for sample_index, (obj, vertex, world_co, _) in enumerate(vertices):
            samples.append({name: _group_weight(obj.vertex_groups.get(name), vertex.index) for name in group_names})
            kd_tree.insert(world_co, sample_index)
        kd_tree.balance()

        updates: list[tuple[bpy.types.Object, int, dict[str, float]]] = []
        for obj, vertex, world_co, is_topwear in vertices:
            if is_topwear and _head_protected_vertex(obj, vertex.index, head_threshold_z=head_threshold_z, world_co=world_co):
                continue

            neighbors = kd_tree.find_range(world_co, radius)
            cross_object_neighbors = [
                sample_index
                for _, sample_index, _ in neighbors
                if vertices[sample_index][0] != obj
            ]
            if not cross_object_neighbors:
                continue

            current = {name: _group_weight(obj.vertex_groups.get(name), vertex.index) for name in group_names}
            averaged = {
                name: sum(samples[sample_index].get(name, 0.0) for sample_index in cross_object_neighbors) / len(cross_object_neighbors)
                for name in group_names
            }
            blended = {
                name: (current.get(name, 0.0) * (1.0 - TOPWEAR_NECK_JOINT_SMOOTH_FACTOR))
                + (averaged.get(name, 0.0) * TOPWEAR_NECK_JOINT_SMOOTH_FACTOR)
                for name in group_names
            }

            if is_topwear and "neck" in blended:
                blended["neck"] = max(blended["neck"], current.get("neck", 0.0) - 0.05)

            weight_map = {
                _ensure_group(obj, name): weight
                for name, weight in blended.items()
                if weight > 1.0e-6 or obj.vertex_groups.get(name) is not None
            }
            if weight_map:
                updates.append((obj, vertex.index, weight_map))

        if not updates:
            continue
        for obj, vertex_index, weight_map in updates:
            _set_normalized_weights(vertex_index, weight_map)
        changed = True

    if changed:
        logger.info(
            "Smoothed topwear/neck as joint mesh -> topwear=%s neck=%s repeats=%s radius=%.6f",
            len(topwear_objs),
            len(neck_objs),
            TOPWEAR_NECK_JOINT_SMOOTH_REPEAT,
            radius,
        )
    return changed


def _topwear_lowest_vertex_z(topwear_objs: list[bpy.types.Object]) -> float | None:
    values: list[float] = []
    for obj in topwear_objs:
        if obj.type != "MESH" or obj.data is None or not obj.data.vertices:
            continue
        values.extend((obj.matrix_world @ vertex.co).z for vertex in obj.data.vertices)
    return min(values) if values else None


def _move_neck_bone_head_to_topwear_head_midpoint(
    context: bpy.types.Context,
    armature_obj: bpy.types.Object,
    topwear_objs: list[bpy.types.Object],
) -> bool:
    if armature_obj.type != "ARMATURE" or armature_obj.data.bones.get("neck") is None:
        return False
    head_bone = armature_obj.data.bones.get("head")
    if head_bone is None:
        return False
    topwear_bottom_world_z = _topwear_lowest_vertex_z(topwear_objs)
    if topwear_bottom_world_z is None:
        return False
    head_head_world = armature_obj.matrix_world @ head_bone.head_local
    head_tail_world = armature_obj.matrix_world @ head_bone.tail_local
    head_bottom_world_z = min(head_head_world.z, head_tail_world.z)
    target_world_z = (topwear_bottom_world_z + head_bottom_world_z) * 0.5

    previous_active = context.view_layer.objects.active
    previous_selection = list(context.selected_objects)

    try:
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        bpy.ops.object.select_all(action="DESELECT")
        armature_obj.select_set(True)
        context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode="EDIT")
        edit_bone = armature_obj.data.edit_bones.get("neck")
        if edit_bone is None:
            return False

        current_head_world = armature_obj.matrix_world @ edit_bone.head
        target_local = armature_obj.matrix_world.inverted_safe() @ Vector(
            (current_head_world.x, current_head_world.y, target_world_z)
        )
        old_z = edit_bone.head.z
        edit_bone.head.z = min(target_local.z, edit_bone.tail.z - 0.001)
        moved = abs(edit_bone.head.z - old_z) > 1.0e-6
        bpy.ops.object.mode_set(mode="OBJECT")
    finally:
        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        for selected in previous_selection:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            context.view_layer.objects.active = previous_active

    if moved:
        logger.info(
            "Moved neck bone head to topwear/head midpoint -> armature=%s target_world_z=%.6f topwear_bottom_world_z=%.6f head_bottom_world_z=%.6f",
            armature_obj.name,
            target_world_z,
            topwear_bottom_world_z,
            head_bottom_world_z,
        )
    return moved


def _limit_topwear_neck_influence(
    obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    bone_names: tuple[str, ...],
) -> bool:
    if "neck" not in bone_names or armature_obj.data.bones.get("neck") is None:
        return False
    if obj.type != "MESH" or obj.data is None or not obj.data.vertices:
        return False

    neck_group = obj.vertex_groups.get("neck")
    if neck_group is None:
        return False
    torso_group = _ensure_group(obj, "torso") if "torso" in bone_names else None
    spine_group = _ensure_group(obj, "spine") if "spine" in bone_names else None
    fallback_group = torso_group or spine_group or obj.vertex_groups.get("hips") or obj.vertex_groups.get("root")
    if fallback_group is None:
        return False

    blend_groups = [
        group
        for group in (
            obj.vertex_groups.get("root"),
            obj.vertex_groups.get("hips"),
            spine_group,
            torso_group,
            neck_group,
            obj.vertex_groups.get("head"),
        )
        if group is not None
    ]

    world_positions = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    min_z = min(co.z for co in world_positions)
    max_z = max(co.z for co in world_positions)
    min_x = min(co.x for co in world_positions)
    max_x = max(co.x for co in world_positions)
    height = max(max_z - min_z, 1.0e-6)
    width = max(max_x - min_x, 1.0e-6)

    neck_bone = armature_obj.data.bones["neck"]
    neck_head_world = armature_obj.matrix_world @ neck_bone.head_local
    neck_tail_world = armature_obj.matrix_world @ neck_bone.tail_local
    neck_center_x = (neck_head_world.x + neck_tail_world.x) * 0.5
    center_half_width = max(width * TOPWEAR_NECK_CENTER_WIDTH_RATIO, 1.0e-6)
    max_neck_weight = TOPWEAR_NECK_MAX_WEIGHT
    affected = 0

    for vertex, world_co in zip(obj.data.vertices, world_positions, strict=False):
        allowed_neck_weight = _topwear_neck_allowed_weight(
            world_co,
            min_z=min_z,
            height=height,
            neck_center_x=neck_center_x,
            center_half_width=center_half_width,
            max_neck_weight=max_neck_weight,
        )
        current_neck_weight = _group_weight(neck_group, vertex.index)
        if current_neck_weight <= allowed_neck_weight + 1.0e-5:
            continue

        excess = current_neck_weight - allowed_neck_weight
        weights = {group: _group_weight(group, vertex.index) for group in blend_groups}
        weights[neck_group] = allowed_neck_weight
        weights[fallback_group] = weights.get(fallback_group, 0.0) + excess
        _set_normalized_weights(vertex.index, weights)
        affected += 1

    if affected:
        logger.info(
            "Limited topwear neck influence on %s -> affected=%s max_weight=%.2f top_band=%.2f center_half_width=%.6f",
            obj.name,
            affected,
            max_neck_weight,
            TOPWEAR_NECK_START_RATIO,
            center_half_width,
        )
    return affected > 0


def _apply_topwear_torso_bias(
    obj: bpy.types.Object,
    armature_obj: bpy.types.Object,
    bone_names: tuple[str, ...],
) -> bool:
    if "torso" not in bone_names or armature_obj.data.bones.get("torso") is None:
        return False
    if obj.type != "MESH" or obj.data is None or not obj.data.vertices:
        return False

    torso_group = _ensure_group(obj, "torso")
    blend_groups = [
        group
        for group in (
            obj.vertex_groups.get("root"),
            obj.vertex_groups.get("hips"),
            obj.vertex_groups.get("spine"),
            torso_group,
            obj.vertex_groups.get("neck"),
            obj.vertex_groups.get("head"),
        )
        if group is not None
    ]
    if len(blend_groups) <= 1:
        return False

    world_positions = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    min_z = min(co.z for co in world_positions)
    max_z = max(co.z for co in world_positions)
    height = max(max_z - min_z, 1.0e-6)
    affected = 0

    for vertex, world_co in zip(obj.data.vertices, world_positions, strict=False):
        z_ratio = (world_co.z - min_z) / height
        low_to_mid_factor = 1.0 - _smoothstep((z_ratio - 0.38) / 0.42)
        target_torso_weight = 0.68 * low_to_mid_factor
        if target_torso_weight <= 1.0e-5:
            continue

        current_torso_weight = _group_weight(torso_group, vertex.index)
        if current_torso_weight >= target_torso_weight:
            continue

        remaining_scale = max(0.0, 1.0 - target_torso_weight)
        existing_total = sum(
            _group_weight(group, vertex.index)
            for group in blend_groups
            if group != torso_group
        )
        weights: dict[bpy.types.VertexGroup, float] = {torso_group: target_torso_weight}
        if existing_total <= 1.0e-8:
            neck_group = obj.vertex_groups.get("neck") or obj.vertex_groups.get("spine")
            if neck_group is not None:
                weights[neck_group] = remaining_scale
        else:
            for group in blend_groups:
                if group == torso_group:
                    continue
                current_weight = _group_weight(group, vertex.index)
                if current_weight > 0.0:
                    weights[group] = (current_weight / existing_total) * remaining_scale
        _set_normalized_weights(vertex.index, weights)
        affected += 1

    if affected:
        logger.info(
            "Applied topwear torso bias on %s -> affected=%s max_weight=%.2f lower_band_end=%.2f",
            obj.name,
            affected,
            0.68,
            0.80,
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


def _head_mesh_bottom_z(parts: list[LayerPart]) -> float | None:
    preferred_tokens = ("face", "headwear")
    fallback_tokens = tuple(heuristic_rigger.HEAD_TOKENS)
    for tokens in (preferred_tokens, fallback_tokens):
        values: list[float] = []
        for part in parts:
            if part.skipped or not part.imported_object_name:
                continue
            if heuristic_rigger._canonical_token(part) not in tokens:
                continue
            obj = bpy.data.objects.get(part.imported_object_name)
            if obj is None or obj.type != "MESH" or obj.data is None or not obj.data.vertices:
                continue
            values.extend((obj.matrix_world @ vertex.co).z for vertex in obj.data.vertices)
        if values:
            return min(values)
    return None


def bind_parts(
    context: bpy.types.Context,
    armature_obj: bpy.types.Object,
    parts: list[LayerPart],
    *,
    rig_plan: RigPlan | None = None,
) -> None:
    if rig_plan is None:
        rig_plan = heuristic_rigger.estimate_rig(parts)

    head_bottom_z = _head_mesh_bottom_z(parts)
    joint_bound_paths: set[str] = set()
    topwear_entries: list[tuple[LayerPart, bpy.types.Object]] = []
    neck_entries: list[tuple[LayerPart, bpy.types.Object]] = []
    for part in parts:
        if part.skipped or not part.imported_object_name:
            continue
        obj = bpy.data.objects.get(part.imported_object_name)
        if obj is None or obj.type != "MESH":
            continue
        token = heuristic_rigger._canonical_token(part)
        if token == "topwear":
            topwear_entries.append((part, obj))
        elif token in heuristic_rigger.NECK_TOKENS:
            neck_entries.append((part, obj))

    joint_bones = tuple(name for name in ("root", "hips", "spine", "torso", "neck", "head") if armature_obj.data.bones.get(name))
    if topwear_entries and neck_entries and "neck" in joint_bones and "head" in joint_bones:
        joint_objs = [obj for _, obj in topwear_entries + neck_entries]
        if _apply_joint_voxel_weights(context, joint_objs, armature_obj, joint_bones):
            for part, obj in topwear_entries:
                _override_head_weights(obj, armature_obj, joint_bones)
                _apply_topwear_torso_bias(obj, armature_obj, joint_bones)
                if _apply_topwear_neck_bridge(obj, armature_obj, joint_bones):
                    _smooth_weights(context, obj, TOPWEAR_NECK_BRIDGE_SMOOTH_REPEAT)
                    _apply_topwear_torso_bias(obj, armature_obj, joint_bones)
                    _apply_topwear_neck_bridge(obj, armature_obj, joint_bones)
                _limit_topwear_neck_influence(obj, armature_obj, joint_bones)
                _override_head_weights(obj, armature_obj, joint_bones)
                joint_bound_paths.add(part.layer_path)
            for part, obj in neck_entries:
                if _blend_neck_head_weights(obj, armature_obj, joint_bones, head_bottom_z=head_bottom_z):
                    _smooth_weights(context, obj, NECK_BLEND_SMOOTH_REPEAT)
                    _blend_neck_head_weights(obj, armature_obj, joint_bones, head_bottom_z=head_bottom_z)
                    _smooth_weights(context, obj, NECK_HEAD_SOFTEN_SMOOTH_REPEAT)
                    _blend_neck_head_weights(obj, armature_obj, joint_bones, head_bottom_z=head_bottom_z)
                joint_bound_paths.add(part.layer_path)
            if _smooth_topwear_neck_joint_weights(
                [obj for _, obj in topwear_entries],
                [obj for _, obj in neck_entries],
                armature_obj,
                joint_bones,
            ):
                for part, obj in topwear_entries:
                    _apply_topwear_torso_bias(obj, armature_obj, joint_bones)
                    _apply_topwear_neck_bridge(obj, armature_obj, joint_bones)
                    _limit_topwear_neck_influence(obj, armature_obj, joint_bones)
                    _override_head_weights(obj, armature_obj, joint_bones)
                for part, obj in neck_entries:
                    _blend_neck_head_weights(obj, armature_obj, joint_bones, head_bottom_z=head_bottom_z)

    for part in parts:
        if part.skipped or not part.imported_object_name:
            continue
        if part.layer_path in joint_bound_paths:
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
                elif token in heuristic_rigger.NECK_TOKENS and _blend_neck_head_weights(obj, armature_obj, filtered_auto_bones, head_bottom_z=head_bottom_z):
                    _smooth_weights(context, obj, NECK_BLEND_SMOOTH_REPEAT)
                    _blend_neck_head_weights(obj, armature_obj, filtered_auto_bones, head_bottom_z=head_bottom_z)
                    _smooth_weights(context, obj, NECK_HEAD_SOFTEN_SMOOTH_REPEAT)
                    _blend_neck_head_weights(obj, armature_obj, filtered_auto_bones, head_bottom_z=head_bottom_z)
                else:
                    _smooth_weights(
                        context,
                        obj,
                        HAIR_SMOOTH_REPEAT if token in {"front hair", "back hair"} else OTHER_SMOOTH_REPEAT,
                    )
                if token == "topwear":
                    _override_head_weights(obj, armature_obj, filtered_auto_bones)
                    _apply_topwear_torso_bias(obj, armature_obj, filtered_auto_bones)
                    if _apply_topwear_neck_bridge(obj, armature_obj, filtered_auto_bones):
                        _smooth_weights(context, obj, TOPWEAR_NECK_BRIDGE_SMOOTH_REPEAT)
                        _apply_topwear_torso_bias(obj, armature_obj, filtered_auto_bones)
                        _apply_topwear_neck_bridge(obj, armature_obj, filtered_auto_bones)
                    _limit_topwear_neck_influence(obj, armature_obj, filtered_auto_bones)
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
                tuple(name for name in ("torso", "neck", "head") if armature_obj.data.bones.get(name)),
                head_bottom_z=head_bottom_z,
            )
        _smooth_weights(context, obj, OTHER_SMOOTH_REPEAT)
        _set_armature_parent_keep_transform(obj, armature_obj)

    if topwear_entries:
        _move_neck_bone_head_to_topwear_head_midpoint(context, armature_obj, [obj for _, obj in topwear_entries])

    logger.info("Bound %s layer objects to %s", len(parts), armature_obj.name)
