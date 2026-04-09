from __future__ import annotations

import json
from pathlib import Path

import bpy

from . import utils


def _load_result_info(result_dir: Path) -> dict:
    info_path = result_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json in {result_dir}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def _clear_collection(name: str, *, context=None) -> bpy.types.Collection:
    ctx = context or bpy.context
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        ctx.scene.collection.children.link(collection)
    else:
        for obj in list(collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    return collection


def _ensure_material(image_path: Path, material_name: str) -> bpy.types.Material:
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)
        material.use_nodes = True

    material.blend_method = "BLEND"
    material.shadow_method = "NONE"
    material.use_backface_culling = False

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (320, 0)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (80, 0)
    texture = nodes.new("ShaderNodeTexImage")
    texture.location = (-240, 0)
    texture.image = bpy.data.images.load(str(image_path), check_existing=True)
    texture.image.alpha_mode = "CHANNEL_PACKED"

    links.new(texture.outputs["Color"], shader.inputs["Base Color"])
    links.new(texture.outputs["Alpha"], shader.inputs["Alpha"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def import_result_directory(result_dir: str | Path, *, context=None) -> str:
    ctx = context or bpy.context
    prefs = utils.get_addon_preferences(ctx)
    result_path = Path(result_dir)
    info = _load_result_info(result_path)

    frame_h, frame_w = info.get("frame_size", [1, 1])
    scale = prefs.plane_scale if prefs else 0.005
    depth_spacing = prefs.depth_spacing if prefs else 0.05
    collection_name = result_path.parent.name
    collection = _clear_collection(collection_name, context=ctx)

    parent_name = f"{collection_name}_root"
    parent = bpy.data.objects.get(parent_name)
    if parent is None:
        parent = bpy.data.objects.new(parent_name, None)
        collection.objects.link(parent)

    parts = list(info.get("parts", {}).items())
    parts.sort(key=lambda item: item[1].get("depth_median", 0.0), reverse=True)

    for index, (tag, part) in enumerate(parts):
        xyxy = part.get("xyxy")
        if not xyxy:
            continue
        x1, y1, x2, y2 = xyxy
        image_path = result_path / f"{tag}.png"
        if not image_path.exists():
            continue

        width_px = max(1, x2 - x1)
        height_px = max(1, y2 - y1)
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        world_x = (center_x - (frame_w * 0.5)) * scale
        world_z = ((frame_h * 0.5) - center_y) * scale
        world_y = -index * depth_spacing

        mesh = bpy.data.meshes.new(f"{collection_name}_{tag}_mesh")
        obj = bpy.data.objects.new(f"{collection_name}_{tag}", mesh)
        collection.objects.link(obj)
        obj.parent = parent

        half_w = (width_px * scale) * 0.5
        half_h = (height_px * scale) * 0.5
        mesh.from_pydata(
            [
                (-half_w, 0.0, -half_h),
                (half_w, 0.0, -half_h),
                (half_w, 0.0, half_h),
                (-half_w, 0.0, half_h),
            ],
            [],
            [(0, 1, 2, 3)],
        )
        mesh.update()
        mesh.uv_layers.new(name="UVMap")
        uv_layer = mesh.uv_layers.active
        for loop_index, uv in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
            uv_layer.data[loop_index].uv = uv

        obj.location = (world_x, world_y, world_z)
        material = _ensure_material(image_path, f"{collection_name}_{tag}_mat")
        obj.data.materials.clear()
        obj.data.materials.append(material)
        obj["hallway_avatar_gen_result_dir"] = str(result_path)
        obj["hallway_avatar_gen_tag"] = tag

    return collection_name
