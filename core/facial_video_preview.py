from __future__ import annotations

from dataclasses import dataclass
import ast
import re
from pathlib import Path

import bpy
from mathutils import Vector

from .. import properties
from ..utils.logging import get_logger
from . import seethrough_naming
from .models import LayerPart

logger = get_logger("facial_video_preview")

VIDEO_UV_LAYER_NAME = "Hallway_Facial_Video_UV"
BACKGROUND_MATERIAL_NAME = "HAVATAR_MAT_face_background_video"


@dataclass(frozen=True)
class BlenderUvInverseTransform:
    convention: str = "blender_bottom_left_uv"
    u_scale: float = 1.0
    u_offset: float = 0.0
    v_scale: float = 1.0
    v_offset: float = 0.0
    affine_3x3_row_major: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None

    def apply(self, uv: Vector) -> Vector:
        if self.affine_3x3_row_major is not None:
            matrix = self.affine_3x3_row_major
            a, b, tx = matrix[0]
            c, d, ty = matrix[1]
            return Vector((a * uv.x + b * uv.y + tx, c * uv.x + d * uv.y + ty))
        return Vector((uv.x * self.u_scale + self.u_offset, uv.y * self.v_scale + self.v_offset))


@dataclass(frozen=True)
class FullFramePixelTransform:
    uniform_scale: float = 1.0
    translate_x_px: float = 0.0
    translate_y_px: float = 0.0


@dataclass(frozen=True)
class FacialVideoTransform:
    uv_inverse: BlenderUvInverseTransform
    full_frame_pixels: FullFramePixelTransform


def _section_values(text: str, section_name: str) -> dict[str, str]:
    pattern = re.compile(rf"^\[{re.escape(section_name)}\]\s*$", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        return {}
    start = match.end()
    next_match = re.search(r"^\[[^\]]+\]\s*$", text[start:], re.MULTILINE)
    end = start + next_match.start() if next_match else len(text)
    values: dict[str, str] = {}
    for raw_line in text[start:end].splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def _float_value(values: dict[str, str], key: str, default: float) -> float:
    raw = values.get(key)
    if raw is None:
        return default
    return float(raw)


def _parse_affine(raw: str | None):
    if not raw:
        return None
    value = ast.literal_eval(raw)
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError("affine_3x3_row_major must be a 3x3 list")
    rows: list[tuple[float, float, float]] = []
    for row in value:
        if not isinstance(row, list | tuple) or len(row) != 3:
            raise ValueError("affine_3x3_row_major must be a 3x3 list")
        rows.append((float(row[0]), float(row[1]), float(row[2])))
    return (rows[0], rows[1], rows[2])


def parse_transform_text(text: str) -> FacialVideoTransform:
    uv_values = _section_values(text, "blender_uv_inverse_transform")
    pixel_values = _section_values(text, "full_frame_pixel_transform")
    if not uv_values:
        raise ValueError("Missing [blender_uv_inverse_transform] section")

    return FacialVideoTransform(
        uv_inverse=BlenderUvInverseTransform(
            convention=uv_values.get("convention", "blender_bottom_left_uv").strip() or "blender_bottom_left_uv",
            u_scale=_float_value(uv_values, "u_scale", 1.0),
            u_offset=_float_value(uv_values, "u_offset", 0.0),
            v_scale=_float_value(uv_values, "v_scale", 1.0),
            v_offset=_float_value(uv_values, "v_offset", 0.0),
            affine_3x3_row_major=_parse_affine(uv_values.get("affine_3x3_row_major")),
        ),
        full_frame_pixels=FullFramePixelTransform(
            uniform_scale=_float_value(pixel_values, "uniform_scale", 1.0),
            translate_x_px=_float_value(pixel_values, "translate_x_px", 0.0),
            translate_y_px=_float_value(pixel_values, "translate_y_px", 0.0),
        ),
    )


def parse_transform_file(filepath: str) -> FacialVideoTransform:
    path = Path(bpy.path.abspath(filepath))
    if not path.is_file():
        raise FileNotFoundError(f"Facial video transform file not found: {path}")
    return parse_transform_text(path.read_text(encoding="utf-8"))


def _face_part_object(parts: list[LayerPart]) -> bpy.types.Object | None:
    for part in parts:
        if part.skipped or not part.imported_object_name:
            continue
        token = part.normalized_token or seethrough_naming.classify_name(part.layer_name, part.layer_path)[0]
        if token != "face":
            continue
        obj = bpy.data.objects.get(part.imported_object_name)
        if obj is not None and obj.type == "MESH":
            return obj
    return None


def find_face_object(context: bpy.types.Context, parts: list[LayerPart] | None = None) -> bpy.types.Object | None:
    if parts:
        obj = _face_part_object(parts)
        if obj is not None:
            return obj
    scene_parts = properties.get_parts(context.scene)
    obj = _face_part_object(scene_parts)
    if obj is not None:
        return obj
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        layer_name = str(obj.get("hallway_avatar_layer_name", obj.name))
        layer_path = str(obj.get("hallway_avatar_layer_path", layer_name))
        token = seethrough_naming.classify_name(layer_name, layer_path)[0]
        if token == "face":
            return obj
    return None


def _face_plane_base_uvs(obj: bpy.types.Object, convention: str = "blender_bottom_left_uv") -> list[Vector]:
    world_positions = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    if not world_positions:
        return []
    canvas_w = float(obj.get("hallway_avatar_canvas_width", 0.0) or 0.0)
    canvas_h = float(obj.get("hallway_avatar_canvas_height", 0.0) or 0.0)
    if canvas_w <= 0.0 or canvas_h <= 0.0:
        for part in properties.get_parts(bpy.context.scene):
            if part.imported_object_name != obj.name:
                continue
            canvas_w = float(part.canvas_size[0])
            canvas_h = float(part.canvas_size[1])
            break
    if canvas_w > 0.0 and canvas_h > 0.0:
        world_scale = 2.0 / max(1.0, canvas_w, canvas_h)
        ground_offset_z = float(obj.get("hallway_avatar_ground_offset_z", 0.0) or 0.0)
        if convention == "blender_bottom_left_square_canvas_uv":
            canvas_square = max(canvas_w, canvas_h, 1.0)
            pad_x = (canvas_square - canvas_w) * 0.5
            pad_y = (canvas_square - canvas_h) * 0.5
            return [
                Vector((
                    (co.x / world_scale + canvas_w * 0.5 + pad_x) / canvas_square,
                    ((co.z - ground_offset_z) / world_scale + canvas_h * 0.5 + pad_y) / canvas_square,
                ))
                for co in world_positions
            ]
        return [
            Vector((
                (co.x / world_scale + canvas_w * 0.5) / canvas_w,
                ((co.z - ground_offset_z) / world_scale + canvas_h * 0.5) / canvas_h,
            ))
            for co in world_positions
        ]

    min_x = min(co.x for co in world_positions)
    max_x = max(co.x for co in world_positions)
    min_z = min(co.z for co in world_positions)
    max_z = max(co.z for co in world_positions)
    width = max(max_x - min_x, 1e-8)
    height = max(max_z - min_z, 1e-8)
    return [Vector(((co.x - min_x) / width, (co.z - min_z) / height)) for co in world_positions]


def duplicate_transformed_face_uv(obj: bpy.types.Object, transform: FacialVideoTransform) -> bpy.types.MeshUVLoopLayer:
    if obj.type != "MESH":
        raise TypeError(f"{obj.name} is not a mesh object")
    mesh = obj.data
    existing = mesh.uv_layers.get(VIDEO_UV_LAYER_NAME)
    if existing is None:
        target_uv = mesh.uv_layers.new(name=VIDEO_UV_LAYER_NAME)
    else:
        target_uv = existing

    base_uv_by_vertex = _face_plane_base_uvs(obj, transform.uv_inverse.convention)
    if not base_uv_by_vertex:
        raise RuntimeError(f"{obj.name} has no vertices for facial video UV generation")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            target_uv.data[loop_index].uv = transform.uv_inverse.apply(base_uv_by_vertex[vertex_index])

    for index, uv_layer in enumerate(mesh.uv_layers):
        if uv_layer.name == target_uv.name:
            mesh.uv_layers.active_index = index
            if hasattr(mesh.uv_layers, "active_render_index"):
                mesh.uv_layers.active_render_index = index
            if hasattr(target_uv, "active_render"):
                target_uv.active_render = True
            break
    obj["hallway_avatar_facial_video_uv_layer"] = target_uv.name
    obj["hallway_avatar_facial_video_uv_basis"] = "world_xz_bounds"
    obj["hallway_avatar_facial_video_uv_convention"] = transform.uv_inverse.convention
    obj["hallway_avatar_facial_video_uv_transform_direction"] = "target_uv_to_video_uv"
    obj["hallway_avatar_facial_video_u_scale"] = transform.uv_inverse.u_scale
    obj["hallway_avatar_facial_video_u_offset"] = transform.uv_inverse.u_offset
    obj["hallway_avatar_facial_video_v_scale"] = transform.uv_inverse.v_scale
    obj["hallway_avatar_facial_video_v_offset"] = transform.uv_inverse.v_offset
    obj["hallway_avatar_facial_video_uniform_scale"] = transform.full_frame_pixels.uniform_scale
    obj["hallway_avatar_facial_video_translate_x_px"] = transform.full_frame_pixels.translate_x_px
    obj["hallway_avatar_facial_video_translate_y_px"] = transform.full_frame_pixels.translate_y_px
    return target_uv


def _load_movie_image(video_path: str) -> bpy.types.Image:
    path = Path(bpy.path.abspath(video_path))
    if not path.is_file():
        raise FileNotFoundError(f"Facial video file not found: {path}")
    image = bpy.data.images.load(str(path), check_existing=True)
    try:
        image.source = "MOVIE"
    except Exception:
        pass
    return image


def _ensure_background_video_material(
    video_path: str,
    *,
    frame_duration: int,
    frame_start: int,
    frame_offset: int,
    auto_refresh: bool,
) -> bpy.types.Material:
    image = _load_movie_image(video_path)
    material = bpy.data.materials.get(BACKGROUND_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(BACKGROUND_MATERIAL_NAME)
    material.use_nodes = True
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"
    if hasattr(material, "blend_method"):
        material.blend_method = "BLEND"

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    tex = nodes.new("ShaderNodeTexImage")
    tex.name = "Image Texture"
    tex.label = "Image Texture"
    background = nodes.new("ShaderNodeBackground")
    tex.image = image
    tex.extension = "EXTEND"
    tex.image_user.frame_duration = max(1, int(frame_duration))
    tex.image_user.frame_start = int(frame_start)
    tex.image_user.frame_offset = int(frame_offset)
    tex.image_user.use_auto_refresh = bool(auto_refresh)
    background.inputs["Strength"].default_value = 1.0

    tex.location = (-500, 0)
    background.location = (-180, 0)
    output.location = (140, 0)

    links.new(tex.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])

    material["hallway_avatar_facial_video_path"] = str(Path(bpy.path.abspath(video_path)))
    material["hallway_avatar_facial_video_frames"] = int(frame_duration)
    material["hallway_avatar_facial_video_start_frame"] = int(frame_start)
    material["hallway_avatar_facial_video_offset"] = int(frame_offset)
    material["hallway_avatar_facial_video_auto_refresh"] = bool(auto_refresh)
    return material


def setup_facial_video_preview(
    context: bpy.types.Context,
    *,
    parts: list[LayerPart] | None = None,
    transform_path: str = "",
    video_path: str = "",
    frame_duration: int = 1000,
    frame_start: int = 0,
    frame_offset: int = 0,
    auto_refresh: bool = True,
) -> bpy.types.Object:
    transform = parse_transform_file(transform_path)
    obj = find_face_object(context, parts)
    if obj is None:
        raise RuntimeError("No imported See-through Face layer mesh found.")

    old_material_name = obj.active_material.name if obj.active_material else ""
    duplicate_transformed_face_uv(obj, transform)
    material = _ensure_background_video_material(
        video_path,
        frame_duration=frame_duration,
        frame_start=frame_start,
        frame_offset=frame_offset,
        auto_refresh=auto_refresh,
    )
    obj.data.materials.clear()
    obj.data.materials.append(material)
    obj.active_material_index = 0
    obj["hallway_avatar_facial_video_material"] = material.name
    if old_material_name:
        obj["hallway_avatar_facial_video_replaced_material"] = old_material_name
    logger.info("Configured facial video preview on %s using %s", obj.name, material.name)
    return obj


def setup_from_state(
    context: bpy.types.Context,
    *,
    parts: list[LayerPart] | None = None,
    raise_on_missing: bool = True,
) -> bpy.types.Object | None:
    state = context.scene.hallway_avatar_state
    transform_path = (state.facial_video_transform_path or "").strip()
    video_path = (state.facial_video_path or "").strip()
    if not transform_path or not video_path:
        message = "Facial video preview requires both a transform txt file and a video file."
        if raise_on_missing:
            raise RuntimeError(message)
        logger.info(message)
        return None
    return setup_facial_video_preview(
        context,
        parts=parts,
        transform_path=transform_path,
        video_path=video_path,
        frame_duration=state.facial_video_frame_duration,
        frame_start=state.facial_video_start_frame,
        frame_offset=state.facial_video_frame_offset,
        auto_refresh=state.facial_video_auto_refresh,
    )
