from __future__ import annotations

import math
import os
import platform
import re
import stat
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import bmesh
import bpy
import mathutils
from mathutils import Vector

from ..utils import paths
from ..utils.logging import get_logger
from . import seethrough_naming
from .models import LayerPart
from .qremeshify_runtime.util import bisect, exporter, importer

logger = get_logger("qremeshify")
_BLENDER_DUPLICATE_SUFFIX_RE = re.compile(r"^(?P<base>.+)\.(?P<suffix>\d{3})$")
_EXACT_CACHE_VERSION = "hallway-qremeshify-exact-v1"
_RUNTIME_FINGERPRINT: str | None = None


class QRemeshifyError(RuntimeError):
    pass


class QRemeshifyUnsupportedInput(QRemeshifyError):
    pass


@dataclass(frozen=True)
class _QRemeshifyPaths:
    mesh_path: str
    sharp_path: str
    field_path: str
    remeshed_path: str
    traced_path: str
    output_path: str
    output_smoothed_path: str

    @classmethod
    def from_mesh_path(cls, mesh_path: str) -> "_QRemeshifyPaths":
        mesh_path_without_ext, _ = os.path.splitext(mesh_path)
        return cls(
            mesh_path=mesh_path,
            sharp_path=f"{mesh_path_without_ext}_rem.sharp",
            field_path=f"{mesh_path_without_ext}_rem.rosy",
            remeshed_path=f"{mesh_path_without_ext}_rem.obj",
            traced_path=f"{mesh_path_without_ext}_rem_p0.obj",
            output_path=f"{mesh_path_without_ext}_rem_p0_0_quadrangulation.obj",
            output_smoothed_path=f"{mesh_path_without_ext}_rem_p0_0_quadrangulation_smooth.obj",
        )


@dataclass(frozen=True)
class QRemeshifySettings:
    auto_on_import: bool = True
    remesh_front_hair: bool = True
    remesh_back_hair: bool = True
    remesh_face_head: bool = False
    remesh_topwear: bool = True
    remesh_handwear: bool = True
    remesh_bottomwear: bool = False
    remesh_legwear: bool = True
    remesh_footwear: bool = True
    remesh_tail: bool = False
    remesh_wings: bool = False
    remesh_objects: bool = False
    remesh_unclassified: bool = False

    @classmethod
    def from_scene_state(cls, state) -> "QRemeshifySettings":
        props = state.qremeshify_settings
        return cls(
            auto_on_import=props.auto_on_import,
            remesh_front_hair=props.remesh_front_hair,
            remesh_back_hair=props.remesh_back_hair,
            remesh_face_head=props.remesh_face_head,
            remesh_topwear=props.remesh_topwear,
            remesh_handwear=props.remesh_handwear,
            remesh_bottomwear=props.remesh_bottomwear,
            remesh_legwear=props.remesh_legwear,
            remesh_footwear=props.remesh_footwear,
            remesh_tail=props.remesh_tail,
            remesh_wings=props.remesh_wings,
            remesh_objects=props.remesh_objects,
            remesh_unclassified=props.remesh_unclassified,
        )


def _library_filenames() -> tuple[str, str]:
    system = platform.system()
    if system == "Windows":
        return "lib_quadwild.dll", "lib_quadpatches.dll"
    if system == "Darwin":
        return "liblib_quadwild.dylib", "liblib_quadpatches.dylib"
    return "liblib_quadwild.so", "liblib_quadpatches.so"


def runtime_status() -> str:
    runtime_dir = paths.qremeshify_runtime_dir()
    missing = [name for name in _library_filenames() if not (runtime_dir / name).exists()]
    config_dir = runtime_dir / "config"
    if missing:
        return f"QRemeshify runtime missing {', '.join(missing)} ({runtime_platform_key()})"
    if not config_dir.exists():
        return f"QRemeshify runtime missing config ({runtime_platform_key()})"
    return f"QRemeshify runtime ready ({runtime_platform_key()})"


def runtime_platform_key() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        machine = "x64"
    elif machine in {"arm64", "aarch64"}:
        machine = "arm64"
    return f"{system or 'Unknown'}-{machine or 'unknown'}"


def _fix_runtime_permissions() -> None:
    runtime_dir = paths.qremeshify_runtime_dir()
    if platform.system() == "Darwin":
        os.system(f"xattr -dr com.apple.quarantine {str(runtime_dir)!r} >/dev/null 2>&1")
    for filename in _library_filenames():
        path = runtime_dir / filename
        if not path.exists():
            continue
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def ensure_runtime() -> None:
    runtime_dir = paths.qremeshify_runtime_dir()
    missing = [str(runtime_dir / name) for name in _library_filenames() if not (runtime_dir / name).exists()]
    if missing:
        raise QRemeshifyError("Vendored QRemeshify runtime is missing: " + ", ".join(missing))
    if not (runtime_dir / "config").exists():
        raise QRemeshifyError(f"Vendored QRemeshify runtime is missing config: {runtime_dir / 'config'}")
    _fix_runtime_permissions()


def _strip_import_prefix(name: str) -> str:
    return re.sub(r"^\d+[_\-\s]+", "", name or "").strip()


def _canonical_remesh_token(part: LayerPart) -> str:
    layer_name = _strip_import_prefix(part.layer_name)
    object_name = _strip_import_prefix(part.imported_object_name)
    for candidate_name, candidate_path in ((layer_name, part.layer_path), (object_name, "")):
        token, _, _ = seethrough_naming.classify_name(candidate_name, candidate_path)
        if token:
            return token
    if part.semantic_label == "hair_front":
        return "front hair"
    if part.semantic_label == "hair_back":
        return "back hair"
    if part.semantic_label == "torso":
        return "topwear"
    if part.semantic_label == "pelvis":
        return "bottomwear"
    if part.semantic_label.startswith("arm"):
        return "handwear"
    if part.semantic_label.startswith("leg"):
        return "legwear"
    if part.semantic_label.startswith("foot"):
        return "footwear"
    if part.semantic_label == "head":
        return "face"
    if part.semantic_label == "tail":
        return "tail"
    if part.semantic_label == "wings":
        return "wings"
    if part.semantic_label == "accessory":
        return "objects"
    return ""


def _remesh_filter_enabled(settings: QRemeshifySettings, token: str) -> bool:
    mapping = {
        "front hair": settings.remesh_front_hair,
        "back hair": settings.remesh_back_hair,
        "face": settings.remesh_face_head,
        "headwear": settings.remesh_face_head,
        "irides": settings.remesh_face_head,
        "eyebrow": settings.remesh_face_head,
        "eyewhite": settings.remesh_face_head,
        "eyelash": settings.remesh_face_head,
        "eyewear": settings.remesh_face_head,
        "ears": settings.remesh_face_head,
        "earwear": settings.remesh_face_head,
        "nose": settings.remesh_face_head,
        "mouth": settings.remesh_face_head,
        "topwear": settings.remesh_topwear,
        "neck": settings.remesh_topwear,
        "handwear": settings.remesh_handwear,
        "bottomwear": settings.remesh_bottomwear,
        "legwear": settings.remesh_legwear,
        "footwear": settings.remesh_footwear,
        "tail": settings.remesh_tail,
        "wings": settings.remesh_wings,
        "objects": settings.remesh_objects,
    }
    if not token:
        return settings.remesh_unclassified
    return mapping.get(token, settings.remesh_unclassified)


def _should_remesh_part(part: LayerPart, settings: QRemeshifySettings) -> bool:
    return _remesh_filter_enabled(settings, _canonical_remesh_token(part))


def _safe_mesh_stem(name: str) -> str:
    return "".join(c if c not in "\\/:*?<>|" else "_" for c in name).strip() or "hallway_qremeshify_mesh"


def _hash_file(path: Path, digest: "hashlib._Hash") -> None:
    digest.update(str(path.name).encode("utf-8", "surrogateescape"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def _runtime_fingerprint() -> str:
    global _RUNTIME_FINGERPRINT
    if _RUNTIME_FINGERPRINT is not None:
        return _RUNTIME_FINGERPRINT

    runtime_dir = paths.qremeshify_runtime_dir()
    digest = hashlib.sha256()
    digest.update(_EXACT_CACHE_VERSION.encode("utf-8"))
    digest.update(runtime_platform_key().encode("utf-8"))
    for filename in _library_filenames():
        _hash_file(runtime_dir / filename, digest)
    config_dir = runtime_dir / "config"
    for config_path in sorted(path for path in config_dir.rglob("*") if path.is_file()):
        digest.update(str(config_path.relative_to(config_dir)).encode("utf-8", "surrogateescape"))
        _hash_file(config_path, digest)
    _RUNTIME_FINGERPRINT = digest.hexdigest()
    return _RUNTIME_FINGERPRINT


def _exact_cache_enabled() -> bool:
    return os.environ.get("HALLWAY_QREMESHIFY_DISABLE_EXACT_CACHE", "").strip().lower() not in {"1", "true", "yes"}


def _exact_cache_key(mesh_filepath: str, payload: dict) -> str:
    digest = hashlib.sha256()
    digest.update(_runtime_fingerprint().encode("ascii"))
    with open(mesh_filepath, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stable_payload = dict(payload)
    stable_payload["mesh_path"] = "<mesh>"
    digest.update(json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _exact_cache_paths(qpaths: _QRemeshifyPaths) -> tuple[tuple[str, str], ...]:
    return (
        ("input.obj", qpaths.mesh_path),
        ("sharp.sharp", qpaths.sharp_path),
        ("field.rosy", qpaths.field_path),
        ("remeshed.obj", qpaths.remeshed_path),
        ("traced.obj", qpaths.traced_path),
        ("quadrangulation.obj", qpaths.output_path),
        ("quadrangulation_smooth.obj", qpaths.output_smoothed_path),
    )


def _restore_exact_cache(cache_key: str, qpaths: _QRemeshifyPaths, final_mesh_path: str) -> bool:
    if not _exact_cache_enabled():
        return False
    cache_entry = paths.ensure_cache_dir() / "qremeshify_exact" / cache_key
    final_cache_path = cache_entry / next(name for name, path in _exact_cache_paths(qpaths) if path == final_mesh_path)
    if not final_cache_path.is_file():
        return False
    for cache_name, output_path in _exact_cache_paths(qpaths):
        cache_path = cache_entry / cache_name
        if cache_path.is_file():
            shutil.copy2(cache_path, output_path)
    logger.info("QRemeshify exact cache hit -> %s", cache_key[:12])
    return True


def _store_exact_cache(cache_key: str, qpaths: _QRemeshifyPaths) -> None:
    if not _exact_cache_enabled():
        return
    cache_entry = paths.ensure_cache_dir() / "qremeshify_exact" / cache_key
    cache_entry.mkdir(parents=True, exist_ok=True)
    for cache_name, output_path in _exact_cache_paths(qpaths):
        output = Path(output_path)
        if output.is_file():
            shutil.copy2(output, cache_entry / cache_name)
    (cache_entry / "manifest.json").write_text(
        json.dumps({"version": _EXACT_CACHE_VERSION, "key": cache_key}, indent=2),
        encoding="utf-8",
    )


def _solve_linear_3x3(matrix: list[list[float]], vector: list[float]) -> tuple[float, float, float] | None:
    rows = [matrix[index][:] + [vector[index]] for index in range(3)]
    for pivot_index in range(3):
        pivot_row = max(range(pivot_index, 3), key=lambda row_index: abs(rows[row_index][pivot_index]))
        pivot_value = rows[pivot_row][pivot_index]
        if abs(pivot_value) <= 1e-12:
            return None
        if pivot_row != pivot_index:
            rows[pivot_index], rows[pivot_row] = rows[pivot_row], rows[pivot_index]
        pivot_value = rows[pivot_index][pivot_index]
        for column in range(pivot_index, 4):
            rows[pivot_index][column] /= pivot_value
        for row_index in range(3):
            if row_index == pivot_index:
                continue
            factor = rows[row_index][pivot_index]
            if abs(factor) <= 1e-12:
                continue
            for column in range(pivot_index, 4):
                rows[row_index][column] -= factor * rows[pivot_index][column]
    return (rows[0][3], rows[1][3], rows[2][3])


def _fit_affine_plane_map(samples: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    if len(samples) < 3:
        return None
    ata = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    atb = [0.0, 0.0, 0.0]
    for coord_a, coord_b, value in samples:
        vec = (coord_a, coord_b, 1.0)
        for row in range(3):
            atb[row] += vec[row] * value
            for col in range(3):
                ata[row][col] += vec[row] * vec[col]
    return _solve_linear_3x3(ata, atb)


def _fit_linear_axis_map(samples: list[tuple[float, float]]) -> tuple[float, float] | None:
    if len(samples) < 2:
        return None
    sum_coord = 0.0
    sum_value = 0.0
    sum_coord_sq = 0.0
    sum_coord_value = 0.0
    count = float(len(samples))
    for coord, value in samples:
        sum_coord += coord
        sum_value += value
        sum_coord_sq += coord * coord
        sum_coord_value += coord * value
    denominator = (count * sum_coord_sq) - (sum_coord * sum_coord)
    if abs(denominator) <= 1e-12:
        return None
    slope = ((count * sum_coord_value) - (sum_coord * sum_value)) / denominator
    intercept = (sum_value - (slope * sum_coord)) / count
    return (slope, intercept)


def _mesh_plane_axes_world(source_obj: bpy.types.Object) -> tuple[int, int]:
    if len(source_obj.data.vertices) < 2:
        return (0, 1)
    world_coords = [source_obj.matrix_world @ vert.co for vert in source_obj.data.vertices]
    spans = []
    for axis_index in range(3):
        axis_values = [coord[axis_index] for coord in world_coords]
        spans.append(max(axis_values) - min(axis_values))
    sorted_axes = sorted(range(3), key=lambda axis_index: spans[axis_index], reverse=True)
    return (sorted_axes[0], sorted_axes[1])


def _project_flat_uvs_from_source(source_obj: bpy.types.Object, target_obj: bpy.types.Object) -> bool:
    if source_obj.type != "MESH" or target_obj.type != "MESH":
        return False
    if not source_obj.data.uv_layers:
        return False
    if len(source_obj.data.loops) == 0 or len(target_obj.data.loops) == 0:
        return False

    plane_axis_a, plane_axis_b = _mesh_plane_axes_world(source_obj)
    source_world = [source_obj.matrix_world @ vert.co for vert in source_obj.data.vertices]
    target_world = [target_obj.matrix_world @ vert.co for vert in target_obj.data.vertices]

    success = False
    while len(target_obj.data.uv_layers) < len(source_obj.data.uv_layers):
        target_obj.data.uv_layers.new(name=source_obj.data.uv_layers[len(target_obj.data.uv_layers)].name)

    for layer_index, source_uv_layer in enumerate(source_obj.data.uv_layers):
        target_uv_layer = target_obj.data.uv_layers[layer_index]
        source_samples_u: list[tuple[float, float, float]] = []
        source_samples_v: list[tuple[float, float, float]] = []
        axis_samples_u: list[tuple[float, float]] = []
        axis_samples_v: list[tuple[float, float]] = []
        uv_values = [loop_data.uv.copy() for loop_data in source_uv_layer.data]
        uv_min = Vector((min(uv.x for uv in uv_values), min(uv.y for uv in uv_values)))
        uv_max = Vector((max(uv.x for uv in uv_values), max(uv.y for uv in uv_values)))

        for loop in source_obj.data.loops:
            world_co = source_world[loop.vertex_index]
            coord_a = float(world_co[plane_axis_a])
            coord_b = float(world_co[plane_axis_b])
            uv = source_uv_layer.data[loop.index].uv
            source_samples_u.append((coord_a, coord_b, float(uv.x)))
            source_samples_v.append((coord_a, coord_b, float(uv.y)))
            axis_samples_u.append((coord_a, float(uv.x)))
            axis_samples_v.append((coord_b, float(uv.y)))

        coeff_u = _fit_affine_plane_map(source_samples_u)
        coeff_v = _fit_affine_plane_map(source_samples_v)
        if coeff_u is None or coeff_v is None:
            linear_u = _fit_linear_axis_map(axis_samples_u)
            linear_v = _fit_linear_axis_map(axis_samples_v)
            if linear_u is None or linear_v is None:
                continue
            coeff_u = (linear_u[0], 0.0, linear_u[1])
            coeff_v = (0.0, linear_v[0], linear_v[1])

        for loop in target_obj.data.loops:
            world_co = target_world[loop.vertex_index]
            coord_a = float(world_co[plane_axis_a])
            coord_b = float(world_co[plane_axis_b])
            u_value = (coeff_u[0] * coord_a) + (coeff_u[1] * coord_b) + coeff_u[2]
            v_value = (coeff_v[0] * coord_a) + (coeff_v[1] * coord_b) + coeff_v[2]
            target_uv_layer.data[loop.index].uv = Vector((
                min(max(u_value, uv_min.x), uv_max.x),
                min(max(v_value, uv_min.y), uv_max.y),
            ))
        target_uv_layer.name = source_uv_layer.name
        success = True

    if success:
        target_obj.data.uv_layers.active_index = source_obj.data.uv_layers.active_index
        logger.info("Projected flat UVs from %s to %s", source_obj.name, target_obj.name)
    return success


def _copy_material_slots(source_obj: bpy.types.Object, target_obj: bpy.types.Object) -> None:
    target_obj.data.materials.clear()
    for material in source_obj.data.materials:
        target_obj.data.materials.append(material)
    target_obj.active_material_index = source_obj.active_material_index


def _copy_input_shading(source_obj: bpy.types.Object, target_obj: bpy.types.Object) -> None:
    if len(source_obj.data.polygons) == 0 or len(target_obj.data.polygons) == 0:
        return
    if source_obj.data.polygons[0].use_smooth:
        return
    for polygon in target_obj.data.polygons:
        polygon.use_smooth = False


def _prune_uv_layers(source_obj: bpy.types.Object, target_obj: bpy.types.Object) -> None:
    source_names = [layer.name for layer in source_obj.data.uv_layers]
    if not source_names:
        return
    while len(target_obj.data.uv_layers) > len(source_names):
        removable = next(
            (layer for layer in target_obj.data.uv_layers if layer.name not in source_names),
            target_obj.data.uv_layers[-1],
        )
        target_obj.data.uv_layers.remove(removable)
    for index, source_name in enumerate(source_names):
        if index < len(target_obj.data.uv_layers):
            target_obj.data.uv_layers[index].name = source_name


def _strip_duplicate_suffix(name: str) -> str:
    match = _BLENDER_DUPLICATE_SUFFIX_RE.match(name)
    if match:
        return match.group("base")
    return name


def _material_images(material: bpy.types.Material | None) -> set[bpy.types.Image]:
    images: set[bpy.types.Image] = set()
    if material is None or not material.use_nodes or material.node_tree is None:
        return images
    for node in material.node_tree.nodes:
        image = getattr(node, "image", None)
        if image is not None:
            images.add(image)
    return images


def _cleanup_transient_materials(materials: list[bpy.types.Material]) -> None:
    for material in materials:
        if material is None or material.name not in bpy.data.materials:
            continue
        if material.users == 0:
            bpy.data.materials.remove(material)


def _cleanup_duplicate_images_for_materials(materials: list[bpy.types.Material]) -> None:
    final_image_bases = {_strip_duplicate_suffix(image.name) for material in materials for image in _material_images(material)}
    if not final_image_bases:
        return
    for image in list(bpy.data.images):
        base_name = _strip_duplicate_suffix(image.name)
        if base_name == image.name or base_name not in final_image_bases:
            continue
        if image.use_fake_user and image.users <= 1:
            image.use_fake_user = False
        if image.users == 0:
            bpy.data.images.remove(image)


def _copy_custom_properties(source_obj: bpy.types.Object, target_obj: bpy.types.Object) -> None:
    for key in list(source_obj.keys()):
        if key == "_RNA_UI":
            continue
        target_obj[key] = source_obj[key]


def _copy_display_settings(source_obj: bpy.types.Object, target_obj: bpy.types.Object) -> None:
    target_obj.color = source_obj.color
    target_obj.display_type = source_obj.display_type
    target_obj.hide_render = source_obj.hide_render
    target_obj.show_name = source_obj.show_name
    target_obj.show_axis = source_obj.show_axis
    target_obj.show_wire = source_obj.show_wire
    target_obj.show_in_front = source_obj.show_in_front
    target_obj.hide_set(source_obj.hide_get())


def _set_active_object(context: bpy.types.Context, obj: bpy.types.Object | None) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    if obj is None:
        return
    obj.select_set(True)
    context.view_layer.objects.active = obj


def _apply_data_transfer_modifier(
    context: bpy.types.Context,
    source_obj: bpy.types.Object,
    target_obj: bpy.types.Object,
    *,
    transfer_uvs: bool = False,
    transfer_vertex_groups: bool = False,
) -> None:
    if not transfer_uvs and not transfer_vertex_groups:
        return
    previous_active = context.view_layer.objects.active
    previous_selection = list(context.selected_objects)
    try:
        _set_active_object(context, target_obj)
        modifier = target_obj.modifiers.new("HallwayAvatarDataTransfer", "DATA_TRANSFER")
        modifier.object = source_obj
        modifier.mix_mode = "REPLACE"
        modifier.mix_factor = 1.0
        modifier.use_object_transform = True
        if transfer_uvs:
            active_source_uv = source_obj.data.uv_layers.active or (source_obj.data.uv_layers[0] if source_obj.data.uv_layers else None)
            if active_source_uv is not None and len(target_obj.data.uv_layers) == 0:
                target_obj.data.uv_layers.new(name=active_source_uv.name)
            if active_source_uv is not None:
                target_obj.data.uv_layers.active_index = 0
            modifier.use_loop_data = True
            modifier.data_types_loops = {"UV"}
            modifier.loop_mapping = "POLYINTERP_LNORPROJ"
        if transfer_vertex_groups:
            for source_group in source_obj.vertex_groups:
                if target_obj.vertex_groups.get(source_group.name) is None:
                    target_obj.vertex_groups.new(name=source_group.name)
            modifier.use_vert_data = True
            modifier.data_types_verts = {"VGROUP_WEIGHTS"}
            modifier.layers_vgroup_select_src = "ALL"
            modifier.layers_vgroup_select_dst = "NAME"
            modifier.vert_mapping = "POLYINTERP_VNORPROJ"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for selected in previous_selection:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            context.view_layer.objects.active = previous_active


def _preserve_parent(source_obj: bpy.types.Object, target_obj: bpy.types.Object) -> None:
    if source_obj.parent is None:
        return
    world_matrix = target_obj.matrix_world.copy()
    target_obj.parent = source_obj.parent
    target_obj.parent_type = source_obj.parent_type
    target_obj.parent_bone = source_obj.parent_bone
    target_obj.matrix_parent_inverse = source_obj.matrix_parent_inverse.copy()
    target_obj.matrix_world = world_matrix


def _replace_source_object(context: bpy.types.Context, source_obj: bpy.types.Object, new_obj: bpy.types.Object) -> bpy.types.Object:
    original_name = source_obj.name
    original_mesh_name = source_obj.data.name if source_obj.data else f"{original_name}_mesh"
    source_was_selected = source_obj.select_get()
    source_was_active = context.view_layer.objects.active == source_obj
    transient_materials = [material for material in new_obj.data.materials if material is not None]
    _copy_material_slots(source_obj, new_obj)
    _copy_input_shading(source_obj, new_obj)
    _copy_custom_properties(source_obj, new_obj)
    _copy_display_settings(source_obj, new_obj)
    if source_obj.data and source_obj.data.uv_layers:
        if not _project_flat_uvs_from_source(source_obj, new_obj):
            _apply_data_transfer_modifier(context, source_obj, new_obj, transfer_uvs=True)
        _prune_uv_layers(source_obj, new_obj)
    if source_obj.vertex_groups:
        _apply_data_transfer_modifier(context, source_obj, new_obj, transfer_vertex_groups=True)
    _preserve_parent(source_obj, new_obj)
    bpy.data.objects.remove(source_obj, do_unlink=True)
    new_obj.name = original_name
    new_obj.data.name = original_mesh_name
    new_obj["hallway_avatar_qremeshify_remeshed"] = True
    _cleanup_transient_materials(transient_materials)
    _cleanup_duplicate_images_for_materials([material for material in new_obj.data.materials if material is not None])
    new_obj.select_set(source_was_selected)
    if source_was_active or context.view_layer.objects.active is None:
        context.view_layer.objects.active = new_obj
    return new_obj


def _scene_qremeshify_props(context: bpy.types.Context):
    return context.scene.quadwild_props, context.scene.quadpatches_props


def _export_input_mesh(context: bpy.types.Context, source_obj: bpy.types.Object, mesh_filepath: str, qpaths: _QRemeshifyPaths) -> tuple[bmesh.types.BMesh, bpy.types.Object]:
    props, _ = _scene_qremeshify_props(context)
    depsgraph = context.evaluated_depsgraph_get()
    evaluated_obj = source_obj.evaluated_get(depsgraph)
    mesh = evaluated_obj.to_mesh()
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        if evaluated_obj.rotation_mode == "QUATERNION":
            matrix = mathutils.Matrix.LocRotScale(None, evaluated_obj.rotation_quaternion, evaluated_obj.scale)
        else:
            matrix = mathutils.Matrix.LocRotScale(None, evaluated_obj.rotation_euler, evaluated_obj.scale)
        bmesh.ops.transform(bm, matrix=matrix, verts=bm.verts)

        if props.symmetryX or props.symmetryY or props.symmetryZ:
            bisect.bisect_on_axes(bm, props.symmetryX, props.symmetryY, props.symmetryZ)

        if props.enableSharp:
            face_set_data_layer = bm.faces.layers.int.get(".sculpt_face_set")
            bm.edges.ensure_lookup_table()
            for edge in bm.edges:
                is_sharp = math.degrees(edge.calc_face_angle(0)) > props.sharpAngle
                is_material_boundary = len(edge.link_faces) > 1 and edge.link_faces[0].material_index != edge.link_faces[1].material_index
                is_face_set_boundary = (
                    face_set_data_layer is not None
                    and len(edge.link_faces) > 1
                    and edge.link_faces[0][face_set_data_layer] != edge.link_faces[1][face_set_data_layer]
                )
                if is_sharp or edge.is_boundary or edge.seam or is_material_boundary or is_face_set_boundary:
                    edge.smooth = False

        bmesh.ops.triangulate(bm, faces=bm.faces, quad_method="SHORT_EDGE", ngon_method="BEAUTY")
        exporter.export_mesh(bm, mesh_filepath)
        if props.enableSharp:
            num_sharp_features = exporter.export_sharp_features(bm, qpaths.sharp_path, props.sharpAngle)
            logger.debug("QRemeshify found %s sharp edges for %s", num_sharp_features, source_obj.name)
    except Exception:
        bm.free()
        evaluated_obj.to_mesh_clear()
        raise
    return bm, evaluated_obj


def _python_executable() -> str:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(sys.prefix) / "bin" / f"python{version}",
        Path(sys.prefix) / "bin" / "python3",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable


def _worker_payload(mesh_filepath: str, props, qr_props) -> dict:
    return {
        "mesh_path": mesh_filepath,
        "useCache": bool(props.useCache),
        "enableRemesh": bool(props.enableRemesh),
        "enableSharp": bool(props.enableSharp),
        "sharpAngle": float(props.sharpAngle),
        "enableSmoothing": bool(props.enableSmoothing),
        "scaleFact": float(qr_props.scaleFact),
        "fixedChartClusters": int(qr_props.fixedChartClusters),
        "alpha": float(qr_props.alpha),
        "ilpMethod": str(qr_props.ilpMethod),
        "timeLimit": int(qr_props.timeLimit),
        "gapLimit": float(qr_props.gapLimit),
        "minimumGap": float(qr_props.minimumGap),
        "isometry": bool(qr_props.isometry),
        "regularityQuadrilaterals": bool(qr_props.regularityQuadrilaterals),
        "regularityNonQuadrilaterals": bool(qr_props.regularityNonQuadrilaterals),
        "regularityNonQuadrilateralsWeight": float(qr_props.regularityNonQuadrilateralsWeight),
        "alignSingularities": bool(qr_props.alignSingularities),
        "alignSingularitiesWeight": float(qr_props.alignSingularitiesWeight),
        "repeatLosingConstraintsIterations": bool(qr_props.repeatLosingConstraintsIterations),
        "repeatLosingConstraintsQuads": bool(qr_props.repeatLosingConstraintsQuads),
        "repeatLosingConstraintsNonQuads": bool(qr_props.repeatLosingConstraintsNonQuads),
        "repeatLosingConstraintsAlign": bool(qr_props.repeatLosingConstraintsAlign),
        "hardParityConstraint": bool(qr_props.hardParityConstraint),
        "flowConfig": str(qr_props.flowConfig),
        "satsumaConfig": str(qr_props.satsumaConfig),
        "callbackTimeLimit": [float(value) for value in qr_props.callbackTimeLimit],
        "callbackGapLimit": [float(value) for value in qr_props.callbackGapLimit],
    }


def _run_qremeshify_worker(mesh_filepath: str, qpaths: _QRemeshifyPaths, props, qr_props) -> str:
    payload = _worker_payload(mesh_filepath, props, qr_props)
    final_mesh_path = qpaths.output_smoothed_path if props.enableSmoothing else qpaths.output_path
    exact_cache_key = None
    if not payload["useCache"]:
        exact_cache_key = _exact_cache_key(mesh_filepath, payload)
        if _restore_exact_cache(exact_cache_key, qpaths, final_mesh_path):
            return final_mesh_path

    payload_path = Path(mesh_filepath).with_suffix(".qremeshify.json")
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    env = dict(os.environ)
    package_parent = str(paths.addon_root().parent)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = package_parent if not existing_pythonpath else f"{package_parent}{os.pathsep}{existing_pythonpath}"
    timeout_seconds = max(300.0, min(3600.0, float(qr_props.timeLimit) + 900.0))
    command = [
        _python_executable(),
        str(Path(__file__).with_name("qremeshify_worker.py")),
        str(payload_path),
    ]
    logger.info("QRemeshify worker start -> timeout=%.1fs payload=%s", timeout_seconds, payload_path)
    try:
        result = subprocess.run(
            command,
            cwd=str(paths.addon_root()),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise QRemeshifyError(f"QRemeshify worker timed out after {int(timeout_seconds)} seconds") from exc

    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout or "").splitlines()[-40:])
        if result.returncode < 0:
            signal_number = -result.returncode
            raise QRemeshifyError(f"QRemeshify native worker crashed with signal {signal_number}.\n{tail}")
        raise QRemeshifyError(f"QRemeshify worker failed with exit {result.returncode}.\n{tail}")

    if not os.path.isfile(final_mesh_path):
        raise QRemeshifyError(f"QRemeshify worker finished but did not produce {final_mesh_path}")
    if exact_cache_key is not None:
        _store_exact_cache(exact_cache_key, qpaths)
        logger.info("QRemeshify exact cache stored -> %s", exact_cache_key[:12])
    if result.stdout:
        logger.debug("QRemeshify worker stdout tail:\n%s", "\n".join(result.stdout.splitlines()[-20:]))
    return final_mesh_path


def _link_debug_mesh(context: bpy.types.Context, source_obj: bpy.types.Object, mesh_path: str, suffix: str) -> None:
    mesh = importer.import_mesh(mesh_path)
    obj = bpy.data.objects.new(f"{source_obj.name} {suffix}", mesh)
    context.collection.objects.link(obj)
    obj.hide_set(True)


def remesh_object(
    context: bpy.types.Context,
    source_obj: bpy.types.Object,
    settings: QRemeshifySettings,
) -> bpy.types.Object:
    source_name = source_obj.name
    if source_obj.type != "MESH":
        raise QRemeshifyError(f"{source_name} is not a mesh.")
    if len(source_obj.data.polygons) == 0:
        raise QRemeshifyError(f"{source_name} has no faces to remesh.")

    ensure_runtime()
    props, qr_props = _scene_qremeshify_props(context)
    cache_dir = paths.ensure_cache_dir() / "qremeshify"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mesh_filepath = str(cache_dir / f"{_safe_mesh_stem(source_name)}.obj")
    qpaths = _QRemeshifyPaths.from_mesh_path(mesh_filepath)
    source_location = source_obj.location.copy()
    collection_targets = list(source_obj.users_collection) or [context.scene.collection]

    logger.info(
        "QRemeshify input %s -> scaleFact=%.3f fixedChartClusters=%s remesh=%s smoothing=%s sharp=%s angle=%.1f flow=%s satsuma=%s timeLimit=%s",
        source_name,
        qr_props.scaleFact,
        qr_props.fixedChartClusters,
        props.enableRemesh,
        props.enableSmoothing,
        props.enableSharp,
        props.sharpAngle,
        qr_props.flowConfig,
        qr_props.satsumaConfig,
        qr_props.timeLimit,
    )

    bm = None
    evaluated_obj = None
    try:
        if not props.useCache:
            bm, evaluated_obj = _export_input_mesh(context, source_obj, mesh_filepath, qpaths)
        final_mesh_path = _run_qremeshify_worker(mesh_filepath, qpaths, props, qr_props)
        if props.debug and os.path.isfile(qpaths.remeshed_path):
            _link_debug_mesh(context, source_obj, qpaths.remeshed_path, "remeshAndField")
        if props.debug and os.path.isfile(qpaths.traced_path):
            _link_debug_mesh(context, source_obj, qpaths.traced_path, "trace")
        if props.debug and props.enableSmoothing:
            _link_debug_mesh(context, source_obj, qpaths.output_path, "quadrangulate")

        final_mesh = importer.import_mesh(final_mesh_path)
        final_obj = bpy.data.objects.new(f"{source_obj.name}__hallway_qremeshify", final_mesh)
        final_obj.location = source_location
        if props.symmetryX or props.symmetryY or props.symmetryZ:
            mirror_modifier = final_obj.modifiers.new("Mirror", "MIRROR")
            mirror_modifier.use_axis[0] = props.symmetryX
            mirror_modifier.use_axis[1] = props.symmetryY
            mirror_modifier.use_axis[2] = props.symmetryZ
            mirror_modifier.use_clip = True
            mirror_modifier.merge_threshold = 0.001
        for collection in collection_targets:
            collection.objects.link(final_obj)
        replaced = _replace_source_object(context, source_obj, final_obj)
        logger.info(
            "QRemeshify replaced %s with %s -> verts=%s faces=%s quads=%s",
            source_name,
            replaced.name,
            len(replaced.data.vertices),
            len(replaced.data.polygons),
            sum(1 for polygon in replaced.data.polygons if polygon.loop_total == 4),
        )
        return replaced
    finally:
        if bm is not None:
            bm.free()
        if evaluated_obj is not None:
            evaluated_obj.to_mesh_clear()


def remesh_parts(
    context: bpy.types.Context,
    parts: list[LayerPart],
    settings: QRemeshifySettings,
    *,
    only_selected: bool = False,
) -> int:
    selected_names = {obj.name for obj in context.selected_objects}
    candidate_parts = [part for part in parts if not part.skipped and part.imported_object_name and _should_remesh_part(part, settings)]
    if only_selected:
        intersected = [part for part in candidate_parts if part.imported_object_name in selected_names]
        if intersected:
            candidate_parts = intersected

    remeshed_count = 0
    for part in candidate_parts:
        obj = bpy.data.objects.get(part.imported_object_name)
        if obj is None:
            continue
        try:
            remeshed_obj = remesh_object(context, obj, settings)
        except QRemeshifyUnsupportedInput as exc:
            obj["hallway_avatar_qremeshify_skipped"] = str(exc)
            logger.warning("QRemeshify skipped %s: %s", obj.name, exc)
            continue
        except QRemeshifyError as exc:
            obj["hallway_avatar_qremeshify_error"] = str(exc)
            logger.error("QRemeshify failed for %s: %s", obj.name, exc)
            continue
        part.imported_object_name = remeshed_obj.name
        remeshed_count += 1
    context.view_layer.update()
    return remeshed_count
