from __future__ import annotations

import platform
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import bmesh
import bpy

from ..utils import paths
from ..utils.logging import get_logger
from .models import LayerPart

logger = get_logger("qremesh")

class QRemeshError(RuntimeError):
    pass


@dataclass(frozen=True)
class QRemeshSettings:
    auto_on_import: bool = False
    target_quad_count: int = 5000
    target_count_as_input_percentage: bool = False
    target_edge_length: float = 0.0
    adaptive_size: float = 50.0
    adapt_quad_count: bool = True
    use_vertex_color_map: bool = False
    use_materials: bool = False
    use_normals_splitting: bool = False
    autodetect_hard_edges: bool = True
    symmetry_x: bool = False
    symmetry_y: bool = False
    symmetry_z: bool = False

    @classmethod
    def from_scene_state(cls, state) -> "QRemeshSettings":
        props = state.qremesh_settings
        return cls(
            auto_on_import=props.auto_on_import,
            target_quad_count=props.target_quad_count,
            target_count_as_input_percentage=props.target_count_as_input_percentage,
            target_edge_length=props.target_edge_length,
            adaptive_size=props.adaptive_size,
            adapt_quad_count=props.adapt_quad_count,
            use_vertex_color_map=props.use_vertex_color_map,
            use_materials=props.use_materials,
            use_normals_splitting=props.use_normals_splitting,
            autodetect_hard_edges=props.autodetect_hard_edges,
            symmetry_x=props.symmetry_x,
            symmetry_y=props.symmetry_y,
            symmetry_z=props.symmetry_z,
        )


@dataclass(frozen=True)
class _RuntimePaths:
    engine_folder: Path
    engine_path: Path
    engine_support_paths: tuple[Path, ...]

def engine_folder() -> Path:
    return paths.quad_remesher_runtime_dir()


def engine_executable() -> Path:
    return engine_folder() / ("qmesh.exe" if platform.system() == "Windows" else "qmesh")


def _engine_support_paths() -> tuple[Path, ...]:
    folder = engine_folder()
    if platform.system() == "Darwin":
        return (
            folder / "qmeshlib.dylib",
            folder / "libfbxsdk.dylib",
            folder / "ChSolver.dylib",
            folder / "resources",
        )
    return (folder / "resources",)


def _clear_quarantine(path: Path) -> None:
    if platform.system() != "Darwin" or not path.exists():
        return
    subprocess.run(
        ["xattr", "-dr", "com.apple.quarantine", str(path)],
        capture_output=True,
        check=False,
        text=True,
    )


def _fix_executable_mode(path: Path) -> None:
    if not path.exists():
        return
    current_mode = path.stat().st_mode
    new_mode = current_mode | (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if current_mode != new_mode:
        path.chmod(new_mode)


def runtime_status() -> str:
    if engine_executable().exists() and all(support_path.exists() for support_path in _engine_support_paths()):
        return "vendored runtime ready"
    if not engine_executable().exists():
        return "vendored engine missing"
    return "vendored engine support files missing"


def ensure_runtime() -> _RuntimePaths:
    engine_path = engine_executable()
    support_paths = _engine_support_paths()

    if not engine_path.exists():
        raise QRemeshError(f"Vendored Quad Remesher engine not found at {engine_path}")
    missing_support = [path for path in support_paths if not path.exists()]
    if missing_support:
        missing_text = ", ".join(str(path) for path in missing_support)
        raise QRemeshError(f"Vendored Quad Remesher engine support files are missing: {missing_text}")

    _clear_quarantine(engine_folder())
    _clear_quarantine(engine_path)
    for support_path in support_paths:
        _clear_quarantine(support_path)
    _fix_executable_mode(engine_path)

    return _RuntimePaths(
        engine_folder=engine_folder(),
        engine_path=engine_path,
        engine_support_paths=support_paths,
    )


def _read_progress_status(progress_path: Path) -> tuple[float | None, str]:
    try:
        lines = progress_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, ""
    if not lines:
        return None, ""
    try:
        value = float(lines[0].strip())
    except ValueError:
        return None, ""
    message = lines[1].strip() if len(lines) > 1 else ""
    return value, message


def _mesh_debug_stats(bm: bmesh.types.BMesh) -> dict[str, object]:
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    if not bm.verts:
        return {
            "verts": 0,
            "faces": 0,
            "bounds": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            "surface_area": 0.0,
            "min_edge": 0.0,
            "median_edge": 0.0,
            "max_edge": 0.0,
            "zero_area_faces": 0,
        }

    xs = [vert.co.x for vert in bm.verts]
    ys = [vert.co.y for vert in bm.verts]
    zs = [vert.co.z for vert in bm.verts]
    edge_lengths = sorted(edge.calc_length() for edge in bm.edges) if bm.edges else [0.0]
    surface_area = sum(face.calc_area() for face in bm.faces)
    zero_area_faces = sum(1 for face in bm.faces if face.calc_area() <= 1e-12)
    return {
        "verts": len(bm.verts),
        "faces": len(bm.faces),
        "bounds": ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))),
        "surface_area": surface_area,
        "min_edge": edge_lengths[0],
        "median_edge": median(edge_lengths),
        "max_edge": edge_lengths[-1],
        "zero_area_faces": zero_area_faces,
    }


def _sanitize_bmesh_for_qremesh(bm: bmesh.types.BMesh) -> None:
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-5)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges[:], dist=1e-5)
    loose_verts = [vert for vert in bm.verts if not vert.link_faces]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")


def _effective_target_quad_count(stats: dict[str, object], settings: QRemeshSettings) -> int:
    if settings.target_edge_length > 0.0:
        surface_area = max(float(stats["surface_area"]), 1e-8)
        return max(1, int(round(surface_area / max(settings.target_edge_length * settings.target_edge_length, 1e-8))))
    if settings.target_count_as_input_percentage:
        return max(1, int(round(max(1, int(stats["faces"])) * (float(settings.target_quad_count) / 100.0))))
    return max(1, int(settings.target_quad_count))


def _symmetry_axis_text(settings: QRemeshSettings) -> str:
    parts = []
    if settings.symmetry_x:
        parts.append("X")
    if settings.symmetry_y:
        parts.append("Y")
    if settings.symmetry_z:
        parts.append("Z")
    return "".join(parts)


def _write_runtime_settings(
    *,
    settings_path: Path,
    input_fbx: Path,
    output_fbx: Path,
    progress_path: Path,
    target_quad_count: int,
    settings: QRemeshSettings,
) -> None:
    lines = [
        "HostApp=Blender",
        f'FileIn="{input_fbx.as_posix()}"',
        f'FileOut="{output_fbx.as_posix()}"',
        f'ProgressFile="{progress_path.as_posix()}"',
        f"TargetQuadCount={target_quad_count}",
        f"CurvatureAdaptivness={float(settings.adaptive_size):.6f}",
        f"ExactQuadCount={int(not settings.adapt_quad_count)}",
        f"UseVertexColorMap={str(bool(settings.use_vertex_color_map))}",
        f"UseMaterialIds={int(bool(settings.use_materials))}",
        f"UseIndexedNormals={int(bool(settings.use_normals_splitting))}",
        f"AutoDetectHardEdges={int(bool(settings.autodetect_hard_edges))}",
    ]
    sym_axis = _symmetry_axis_text(settings)
    if sym_axis:
        lines.append(f"SymAxis={sym_axis}")
        lines.append("SymLocal=1")
    settings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_active_object(context: bpy.types.Context, obj: bpy.types.Object | None) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    if obj is None:
        return
    obj.select_set(True)
    context.view_layer.objects.active = obj


def _mesh_debug_stats_for_object(context: bpy.types.Context, source_obj: bpy.types.Object) -> dict[str, object]:
    depsgraph = context.evaluated_depsgraph_get()
    evaluated_obj = source_obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_obj.to_mesh()
    bm = bmesh.new()
    try:
        bm.from_mesh(evaluated_mesh)
        bmesh.ops.transform(bm, matrix=source_obj.matrix_world, verts=bm.verts)
        return _mesh_debug_stats(bm)
    finally:
        bm.free()
        evaluated_obj.to_mesh_clear()


def _export_object_to_fbx(context: bpy.types.Context, obj: bpy.types.Object, filepath: Path) -> None:
    previous_active = context.view_layer.objects.active
    previous_selection = list(context.selected_objects)
    try:
        _set_active_object(context, obj)
        bpy.ops.export_scene.fbx(filepath=str(filepath), use_selection=True)
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for selected in previous_selection:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            context.view_layer.objects.active = previous_active


def _run_runtime_engine(
    *,
    runtime: _RuntimePaths,
    settings_path: Path,
    progress_path: Path,
    output_path: Path,
    timeout_seconds: float = 1800.0,
) -> None:
    output_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)

    process = subprocess.Popen(
        [str(runtime.engine_path), "-s", str(settings_path)],
        cwd=str(runtime.engine_folder),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    start_time = time.monotonic()
    last_logged_percent = -1

    try:
        while True:
            progress_value, progress_text = _read_progress_status(progress_path)
            if progress_value is not None:
                if 0.0 <= progress_value <= 1.0:
                    percent = int((99.0 * progress_value) + 1.0)
                    if percent // 10 != last_logged_percent // 10:
                        logger.info("Quad Remesher progress %s%%", percent)
                        last_logged_percent = percent
                elif progress_value == 2.0 and output_path.exists():
                    break
                elif progress_value < 0.0:
                    raise QRemeshError(progress_text or f"Quad Remesher failed with progress code {progress_value}")

            if process.poll() is not None:
                break

            if time.monotonic() - start_time > timeout_seconds:
                process.kill()
                raise QRemeshError(f"Quad Remesher timed out after {int(timeout_seconds)} seconds")

            time.sleep(0.25)

        return_code = process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()

    progress_value, progress_text = _read_progress_status(progress_path)
    if progress_value == 2.0 and output_path.exists():
        return
    if output_path.exists():
        return
    if progress_text:
        raise QRemeshError(progress_text)
    if return_code != 0:
        raise QRemeshError(f"Quad Remesher exited with code {return_code}")
    raise QRemeshError(f"Quad Remesher finished without producing {output_path}")


def _import_runtime_result(context: bpy.types.Context, filepath: Path) -> tuple[list[bpy.types.Object], bpy.types.Object]:
    existing_names = set(bpy.data.objects.keys())
    previous_active = context.view_layer.objects.active
    previous_selection = list(context.selected_objects)
    imported_objects: list[bpy.types.Object] = []
    try:
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.import_scene.fbx(filepath=str(filepath))
        imported_objects = [obj for obj in context.selected_objects if obj.name not in existing_names]
        imported_meshes = [obj for obj in imported_objects if obj.type == "MESH"]
        if not imported_meshes:
            imported_meshes = [obj for obj in context.selected_objects if obj.type == "MESH"]
        if not imported_meshes:
            raise QRemeshError(f"Quad Remesher imported {filepath} but no mesh object was created")
        result_obj = max(imported_meshes, key=lambda obj: len(obj.data.polygons))
        context.view_layer.objects.active = result_obj
        return imported_objects, result_obj
    except Exception:
        bpy.ops.object.select_all(action="DESELECT")
        for selected in previous_selection:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            context.view_layer.objects.active = previous_active
        raise


def _rehome_imported_result(
    context: bpy.types.Context,
    imported_objects: list[bpy.types.Object],
    result_obj: bpy.types.Object,
    collection_targets: list[bpy.types.Collection],
) -> None:
    target_collections = collection_targets or [context.scene.collection]
    for collection in target_collections:
        if collection.objects.get(result_obj.name) is None:
            collection.objects.link(result_obj)

    for collection in list(result_obj.users_collection):
        if collection not in target_collections:
            collection.objects.unlink(result_obj)

    for imported_obj in imported_objects:
        if imported_obj == result_obj:
            continue
        if imported_obj.name in bpy.data.objects:
            bpy.data.objects.remove(imported_obj, do_unlink=True)


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
            if active_source_uv is not None:
                modifier.layers_uv_select_src = active_source_uv.name
                modifier.layers_uv_select_dst = target_obj.data.uv_layers[0].name
            modifier.loop_mapping = "POLYINTERP_NEAREST"
        if transfer_vertex_groups:
            for source_group in source_obj.vertex_groups:
                if target_obj.vertex_groups.get(source_group.name) is None:
                    target_obj.vertex_groups.new(name=source_group.name)
            modifier.use_vert_data = True
            modifier.data_types_verts = {"VGROUP_WEIGHTS"}
            modifier.layers_vgroup_select_src = "ALL"
            modifier.layers_vgroup_select_dst = "NAME"
            modifier.vert_mapping = "POLYINTERP_NEAREST"
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for selected in previous_selection:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            context.view_layer.objects.active = previous_active


def _copy_modifiers(context: bpy.types.Context, source_obj: bpy.types.Object, target_obj: bpy.types.Object) -> None:
    if not source_obj.modifiers:
        return

    previous_active = context.view_layer.objects.active
    previous_selection = list(context.selected_objects)
    try:
        bpy.ops.object.select_all(action="DESELECT")
        source_obj.select_set(True)
        target_obj.select_set(True)
        context.view_layer.objects.active = source_obj
        for modifier in source_obj.modifiers:
            bpy.ops.object.modifier_copy_to_selected(modifier=modifier.name)
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for selected in previous_selection:
            if selected.name in bpy.data.objects:
                selected.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            context.view_layer.objects.active = previous_active


def _copy_material_slots(source_obj: bpy.types.Object, target_obj: bpy.types.Object) -> None:
    target_obj.data.materials.clear()
    for material in source_obj.data.materials:
        target_obj.data.materials.append(material)
    target_obj.active_material_index = source_obj.active_material_index


def _copy_input_shading(source_obj: bpy.types.Object, target_obj: bpy.types.Object) -> None:
    if source_obj.type != "MESH" or target_obj.type != "MESH":
        return
    if len(source_obj.data.polygons) == 0 or len(target_obj.data.polygons) == 0:
        return
    input_use_smooth = bool(source_obj.data.polygons[0].use_smooth)
    if input_use_smooth:
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


def _preserve_parent(source_obj: bpy.types.Object, target_obj: bpy.types.Object) -> None:
    if source_obj.parent is None:
        return
    world_matrix = target_obj.matrix_world.copy()
    target_obj.parent = source_obj.parent
    target_obj.parent_type = source_obj.parent_type
    target_obj.parent_bone = source_obj.parent_bone
    target_obj.matrix_parent_inverse = source_obj.matrix_parent_inverse.copy()
    target_obj.matrix_world = world_matrix


def _replace_source_object(
    context: bpy.types.Context,
    source_obj: bpy.types.Object,
    new_obj: bpy.types.Object,
    settings: QRemeshSettings,
) -> bpy.types.Object:
    original_name = source_obj.name
    original_mesh_name = source_obj.data.name if source_obj.data else f"{original_name}_mesh"
    source_was_selected = source_obj.select_get()
    source_was_active = context.view_layer.objects.active == source_obj
    _copy_material_slots(source_obj, new_obj)
    _copy_input_shading(source_obj, new_obj)
    _copy_custom_properties(source_obj, new_obj)
    _copy_display_settings(source_obj, new_obj)
    if source_obj.data and source_obj.data.uv_layers:
        _apply_data_transfer_modifier(context, source_obj, new_obj, transfer_uvs=True)
        _prune_uv_layers(source_obj, new_obj)
    if source_obj.vertex_groups:
        _apply_data_transfer_modifier(context, source_obj, new_obj, transfer_vertex_groups=True)
    _copy_modifiers(context, source_obj, new_obj)
    _preserve_parent(source_obj, new_obj)
    bpy.data.objects.remove(source_obj, do_unlink=True)
    new_obj.name = original_name
    new_obj.data.name = original_mesh_name
    new_obj["hallway_avatar_qremeshed"] = True
    new_obj.select_set(source_was_selected)
    if source_was_active or context.view_layer.objects.active is None:
        context.view_layer.objects.active = new_obj
    return new_obj


def remesh_object(
    context: bpy.types.Context,
    source_obj: bpy.types.Object,
    settings: QRemeshSettings,
) -> bpy.types.Object:
    source_name = source_obj.name
    source_matrix_world = source_obj.matrix_world.copy()
    if source_obj.type != "MESH":
        raise QRemeshError(f"{source_obj.name} is not a mesh.")
    if len(source_obj.data.polygons) == 0:
        raise QRemeshError(f"{source_obj.name} has no faces to remesh.")

    runtime = ensure_runtime()
    cache_dir = paths.ensure_cache_dir() / "qremesh"
    cache_dir.mkdir(parents=True, exist_ok=True)

    collection_targets = list(source_obj.users_collection)

    with tempfile.TemporaryDirectory(prefix="hallway_qr_runtime_", dir=str(cache_dir)) as temp_dir:
        temp_root = Path(temp_dir)
        input_fbx = temp_root / "inputMesh.fbx"
        output_fbx = temp_root / "retopo.fbx"
        progress_path = temp_root / "progress.txt"
        settings_path = temp_root / "RetopoSettings.txt"

        stats = _mesh_debug_stats_for_object(context, source_obj)
        _export_object_to_fbx(context, source_obj, input_fbx)

        target_quad_count = _effective_target_quad_count(stats, settings)
        _write_runtime_settings(
            settings_path=settings_path,
            input_fbx=input_fbx,
            output_fbx=output_fbx,
            progress_path=progress_path,
            target_quad_count=target_quad_count,
            settings=settings,
        )

        bounds_min, bounds_max = stats["bounds"]
        logger.info(
            "Quad Remesher input %s -> verts=%s faces=%s surface_area=%.6f target_quads=%s target_edge_length=%.6f adaptive_size=%.1f exact_quad_count=%s use_vertex_color_map=%s use_materials=%s use_normals_splitting=%s autodetect_hard_edges=%s symmetry=%s bounds_min=(%.6f, %.6f, %.6f) bounds_max=(%.6f, %.6f, %.6f)",
            source_obj.name,
            stats["verts"],
            stats["faces"],
            float(stats["surface_area"]),
            target_quad_count,
            settings.target_edge_length,
            settings.adaptive_size,
            not settings.adapt_quad_count,
            settings.use_vertex_color_map,
            settings.use_materials,
            settings.use_normals_splitting,
            settings.autodetect_hard_edges,
            _symmetry_axis_text(settings) or "-",
            bounds_min[0],
            bounds_min[1],
            bounds_min[2],
            bounds_max[0],
            bounds_max[1],
            bounds_max[2],
        )

        _run_runtime_engine(
            runtime=runtime,
            settings_path=settings_path,
            progress_path=progress_path,
            output_path=output_fbx,
        )

        imported_objects, remeshed_obj = _import_runtime_result(context, output_fbx)
        _rehome_imported_result(context, imported_objects, remeshed_obj, collection_targets)
        remeshed_obj.matrix_world = source_matrix_world

    replaced = _replace_source_object(context, source_obj, remeshed_obj, settings)
    logger.info("Quad Remesher replaced %s with %s", source_name, replaced.name)
    return replaced


def remesh_parts(
    context: bpy.types.Context,
    parts: list[LayerPart],
    settings: QRemeshSettings,
    *,
    only_selected: bool = False,
) -> int:
    selected_names = {obj.name for obj in context.selected_objects}
    candidate_parts = [part for part in parts if not part.skipped and part.imported_object_name]
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
        except QRemeshError as exc:
            obj["hallway_avatar_qremesh_error"] = str(exc)
            logger.error("Quad Remesher failed for %s: %s", obj.name, exc)
            continue
        part.imported_object_name = remeshed_obj.name
        remeshed_count += 1
    context.view_layer.update()
    return remeshed_count
