from __future__ import annotations

import importlib
import bmesh
import bpy
from math import radians
from mathutils import Vector
from pathlib import Path
from time import perf_counter

from ..utils import env
from ..utils.logging import get_logger
from .models import LayerPart

logger = get_logger("alpha_mesh_adapter")


def _pixel_to_plane(x: float, y: float, canvas_size: tuple[int, int]) -> tuple[float, float, float]:
    scale = 2.0 / max(1.0, float(max(canvas_size)))
    canvas_w, canvas_h = canvas_size
    return (
        (x - canvas_w * 0.5) * scale,
        0.0,
        (canvas_h * 0.5 - y) * scale,
    )


def _trace_pixels_to_bmesh(part: LayerPart, context: bpy.types.Context) -> bmesh.types.BMesh:
    return _trace_pixels_to_bmesh_with_contrast(part, context, contrast_remap=(0.1, 0.9))


def _trace_pixels_to_bmesh_with_contrast(
    part: LayerPart,
    context: bpy.types.Context,
    *,
    contrast_remap: tuple[float, float],
) -> bmesh.types.BMesh:
    env.import_optional("vtracer")
    mesher = importlib.import_module(f"{__package__}.import_meshed_alpha_vendor.alpha_mesher")
    pixels = mesher.preprocess_image(
        Path(part.temp_image_path),
        False,
        None,
        False,
        0,
        False,
        contrast_remap,
        1.0,
        None,
    )
    svg_data = mesher.trace_image(pixels, "spline")
    parsed = mesher.parse_trace(svg_data)
    bm = mesher.parsed_to_bmesh(parsed, context)
    mesher.post_process_mesh(
        bm,
        x_align="NONE",
        y_align="NONE",
        triangulate=False,
        xy_divisions=(1, 1),
        divide_ngons=False,
        remove_small_islands=0,
    )
    return bm


def _apply_canvas_transform(obj: bpy.types.Object, part: LayerPart) -> None:
    scale = 2.0 / max(1.0, float(max(part.canvas_size)))
    center_x = part.canvas_offset[0] + part.image_size[0] * 0.5
    center_y = part.canvas_offset[1] + part.image_size[1] * 0.5
    obj.location = Vector(_pixel_to_plane(center_x, center_y, part.canvas_size))
    obj.rotation_euler = (radians(90.0), 0.0, 0.0)
    obj.scale = (scale, scale, scale)


def _ensure_image_material(part: LayerPart) -> bpy.types.Material:
    material_name = f"HAVATAR_MAT_{part.layer_name}"
    material = bpy.data.materials.get(material_name)
    if material is None:
        material = bpy.data.materials.new(material_name)

    material.use_nodes = True
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"
    if hasattr(material, "blend_method"):
        material.blend_method = "BLEND"

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    tex.interpolation = "Closest"
    tex.image = bpy.data.images.load(part.temp_image_path, check_existing=True) if part.temp_image_path else None

    output.location = (300, 0)
    bsdf.location = (60, 0)
    tex.location = (-240, 0)

    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_layer_mesh(
    context: bpy.types.Context,
    part: LayerPart,
    collection: bpy.types.Collection,
    *,
    grid_resolution: int = 12,
    trace_contrast_remap: tuple[float, float] = (0.1, 0.9),
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(f"{part.layer_name}_mesh")
    trace_start = perf_counter()
    low, high = trace_contrast_remap
    contrast_remap = (min(low, high), max(low, high))
    bm = _trace_pixels_to_bmesh_with_contrast(part, context, contrast_remap=contrast_remap)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    trace_seconds = perf_counter() - trace_start

    material = _ensure_image_material(part)
    if mesh.materials:
        mesh.materials[0] = material
    else:
        mesh.materials.append(material)

    obj_name = f"{part.draw_index:03d}_{part.layer_name}"
    obj = bpy.data.objects.new(obj_name, mesh)
    collection.objects.link(obj)
    _apply_canvas_transform(obj, part)
    obj["hallway_avatar_generated"] = True
    obj["hallway_avatar_layer_path"] = part.layer_path
    obj["hallway_avatar_layer_name"] = part.layer_name
    obj["hallway_avatar_temp_image_path"] = part.temp_image_path or ""
    obj["hallway_avatar_image_width"] = part.image_size[0]
    obj["hallway_avatar_image_height"] = part.image_size[1]
    obj["hallway_avatar_canvas_width"] = part.canvas_size[0]
    obj["hallway_avatar_canvas_height"] = part.canvas_size[1]
    obj["hallway_avatar_canvas_offset_x"] = part.canvas_offset[0]
    obj["hallway_avatar_canvas_offset_y"] = part.canvas_offset[1]
    obj["hallway_avatar_semantic_label"] = part.semantic_label
    obj["hallway_avatar_side_guess"] = part.side_guess
    obj["hallway_avatar_confidence"] = part.confidence
    logger.info(
        "Traced %s -> mesh from %sx%s crop in %.3fs (contrast remap %.3f..%.3f)",
        part.layer_path,
        part.image_size[0],
        part.image_size[1],
        trace_seconds,
        contrast_remap[0],
        contrast_remap[1],
    )
    return obj
