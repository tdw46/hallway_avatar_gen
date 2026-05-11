from __future__ import annotations

import os
import platform
import re
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import bmesh
import bpy
from mathutils import Vector

from ..utils import paths
from ..utils.logging import get_logger
from .models import LayerPart
from . import seethrough_naming

logger = get_logger("qremesh")
_BLENDER_DUPLICATE_SUFFIX_RE = re.compile(r"^(?P<base>.+)\.(?P<suffix>\d{3})$")

class QRemeshError(RuntimeError):
    pass


@dataclass(frozen=True)
class QRemeshSettings:
    auto_on_import: bool = True
    target_quad_count: int = 3000
    unsubdivide_iterations: int = 2
    unsubdivide_target_count: int = 1400
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
    target_count_as_input_percentage: bool = True
    target_edge_length: float = 0.02
    adaptive_size: float = 100.0
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
            unsubdivide_iterations=props.unsubdivide_iterations,
            unsubdivide_target_count=props.unsubdivide_target_count,
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


def _machine_key() -> str:
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64"}:
        return "x64"
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return machine or "unknown"


def runtime_platform_key() -> str:
    system = platform.system()
    if system == "Darwin":
        return f"Darwin-{_machine_key()}"
    if system == "Windows":
        return f"Windows-{_machine_key()}"
    if system == "Linux":
        return f"Linux-{_machine_key()}"
    return f"{system or 'Unknown'}-{_machine_key()}"


def _platform_engine_folder() -> Path:
    return paths.quad_remesher_runtime_dir() / runtime_platform_key()


def engine_folder() -> Path:
    platform_folder = _platform_engine_folder()
    if platform_folder.exists():
        return platform_folder
    legacy_folder = paths.quad_remesher_runtime_dir()
    if platform.system() == "Darwin" and (legacy_folder / "qmesh").exists():
        return legacy_folder
    return platform_folder


def engine_executable() -> Path:
    return engine_folder() / ("qmesh.exe" if platform.system() == "Windows" else "qmesh")


def _engine_support_paths() -> tuple[Path, ...]:
    folder = engine_folder()
    system = platform.system()
    if system == "Darwin":
        return (
            folder / "qmeshlib.dylib",
            folder / "libfbxsdk.dylib",
            folder / "ChSolver.dylib",
            folder / "resources",
        )
    if system == "Windows":
        return (
            folder / "qmeshlib.dll",
            folder / "libfbxsdk.dll",
            folder / "ChSolver.dll",
            folder / "resources",
        )
    if system == "Linux":
        return (
            folder / "libqmeshlib.so",
            folder / "libfbxsdk.so",
            folder / "libChSolver.so",
            folder / "resources",
        )
    return (folder / "resources",)


def _runtime_environment(runtime: _RuntimePaths) -> dict[str, str]:
    env = dict(os.environ)
    runtime_dir = str(runtime.engine_folder)
    system = platform.system()
    if system == "Linux":
        current = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = runtime_dir if not current else f"{runtime_dir}:{current}"
    elif system == "Windows":
        current = env.get("PATH")
        env["PATH"] = runtime_dir if not current else f"{runtime_dir};{current}"
    return env


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
    platform_key = runtime_platform_key()
    if engine_executable().exists() and all(support_path.exists() for support_path in _engine_support_paths()):
        return f"vendored runtime ready ({platform_key})"
    if not engine_executable().exists():
        return f"vendored engine missing ({platform_key})"
    return f"vendored engine support files missing ({platform_key})"


def ensure_runtime() -> _RuntimePaths:
    engine_path = engine_executable()
    support_paths = _engine_support_paths()

    if not engine_path.exists():
        raise QRemeshError(
            "Vendored Quad Remesher engine not found for "
            f"{runtime_platform_key()} at {engine_path}. Add the platform runtime under "
            f"{_platform_engine_folder()}."
        )
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


def _duplicate_object_for_export(
    context: bpy.types.Context,
    source_obj: bpy.types.Object,
) -> bpy.types.Object:
    export_obj = source_obj.copy()
    export_obj.data = source_obj.data.copy()
    export_obj.animation_data_clear()
    export_obj.name = f"{source_obj.name}__hallway_qmesh_export"
    export_obj.data.name = f"{source_obj.data.name}__hallway_qmesh_export"
    target_collections = list(source_obj.users_collection) or [context.scene.collection]
    for collection in target_collections:
        collection.objects.link(export_obj)
    export_obj.matrix_world = source_obj.matrix_world.copy()
    return export_obj


def _cleanup_export_duplicate(export_obj: bpy.types.Object | None) -> None:
    if export_obj is None or export_obj.name not in bpy.data.objects:
        return
    mesh = export_obj.data if export_obj.type == "MESH" else None
    bpy.data.objects.remove(export_obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def _strip_import_prefix(name: str) -> str:
    return re.sub(r"^\d+[_\-\s]+", "", name or "").strip()


def _canonical_remesh_token(part: LayerPart) -> str:
    layer_name = _strip_import_prefix(part.layer_name)
    object_name = _strip_import_prefix(part.imported_object_name)

    for candidate_name, candidate_path in (
        (layer_name, part.layer_path),
        (object_name, ""),
    ):
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


def _remesh_filter_enabled(settings: QRemeshSettings, token: str) -> bool:
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


def _should_remesh_part(part: LayerPart, settings: QRemeshSettings) -> bool:
    return _remesh_filter_enabled(settings, _canonical_remesh_token(part))


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
        env=_runtime_environment(runtime),
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


def _unsubdivide_mesh(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    *,
    iterations: int = 2,
    target_count: int = 1400,
) -> None:
    if obj.type != "MESH" or len(obj.data.polygons) == 0 or iterations <= 0:
        return
    face_count_before = len(obj.data.polygons)
    vertex_count_before = len(obj.data.vertices)
    total_iterations = 0
    target_count = max(1, int(target_count))
    max_iterations = 32

    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        if not bm.verts:
            logger.info("Un-subdivide skipped for %s because it has no vertices", obj.name)
            return

        def _current_counts() -> tuple[int, int]:
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            return (len(bm.verts), len(bm.faces))

        while total_iterations < iterations:
            verts_before_step, faces_before_step = _current_counts()
            bmesh.ops.unsubdivide(bm, verts=bm.verts[:], iterations=1)
            total_iterations += 1
            verts_after_step, faces_after_step = _current_counts()
            if (verts_after_step, faces_after_step) == (verts_before_step, faces_before_step):
                break

        while total_iterations < max_iterations:
            current_verts, current_faces = _current_counts()
            if current_verts <= target_count and current_faces <= target_count:
                break
            bmesh.ops.unsubdivide(bm, verts=bm.verts[:], iterations=1)
            total_iterations += 1
            next_verts, next_faces = _current_counts()
            if (next_verts, next_faces) == (current_verts, current_faces):
                break

        bm.to_mesh(obj.data)
        obj.data.update()
    finally:
        bm.free()

    face_count_after = len(obj.data.polygons)
    vertex_count_after = len(obj.data.vertices)
    obj["hallway_avatar_unsubdivide_iterations_requested"] = int(iterations)
    obj["hallway_avatar_unsubdivide_iterations"] = int(total_iterations)
    obj["hallway_avatar_unsubdivide_target_count"] = int(target_count)
    obj["hallway_avatar_unsubdivide_faces_before"] = int(face_count_before)
    obj["hallway_avatar_unsubdivide_faces_after"] = int(face_count_after)
    obj["hallway_avatar_unsubdivide_vertices_before"] = int(vertex_count_before)
    obj["hallway_avatar_unsubdivide_vertices_after"] = int(vertex_count_after)
    logger.info(
        "Un-subdivide %s -> requested=%s applied=%s target=%s faces %s->%s verts %s->%s",
        obj.name,
        iterations,
        total_iterations,
        target_count,
        face_count_before,
        face_count_after,
        vertex_count_before,
        vertex_count_after,
    )


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


def _solve_linear_3x3(system: list[list[float]], values: list[float]) -> tuple[float, float, float] | None:
    rows = [system[index][:] + [values[index]] for index in range(3)]
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

    ata = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
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
            if modifier.type == "ARMATURE":
                continue
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


def _material_images(material: bpy.types.Material | None) -> set[bpy.types.Image]:
    images: set[bpy.types.Image] = set()
    if material is None or not material.use_nodes or material.node_tree is None:
        return images
    for node in material.node_tree.nodes:
        image = getattr(node, "image", None)
        if image is not None:
            images.add(image)
    return images


def _strip_duplicate_suffix(name: str) -> str:
    match = _BLENDER_DUPLICATE_SUFFIX_RE.match(name)
    if match:
        return match.group("base")
    return name


def _cleanup_transient_materials(materials: list[bpy.types.Material]) -> None:
    for material in materials:
        if material is None or material.name not in bpy.data.materials:
            continue
        if material.users == 0:
            bpy.data.materials.remove(material)


def _cleanup_duplicate_images_for_materials(materials: list[bpy.types.Material]) -> None:
    final_image_bases = {
        _strip_duplicate_suffix(image.name)
        for material in materials
        for image in _material_images(material)
    }
    if not final_image_bases:
        return

    for image in list(bpy.data.images):
        if image is None:
            continue
        base_name = _strip_duplicate_suffix(image.name)
        if base_name == image.name or base_name not in final_image_bases:
            continue
        if image.use_fake_user and image.users <= 1:
            image.use_fake_user = False
        if image.users == 0:
            logger.info("Removed duplicate remesh image datablock %s because final materials already use %s", image.name, base_name)
            bpy.data.images.remove(image)


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
    _copy_modifiers(context, source_obj, new_obj)
    _preserve_parent(source_obj, new_obj)
    bpy.data.objects.remove(source_obj, do_unlink=True)
    new_obj.name = original_name
    new_obj.data.name = original_mesh_name
    new_obj["hallway_avatar_qremeshed"] = True
    _cleanup_transient_materials(transient_materials)
    _cleanup_duplicate_images_for_materials([material for material in new_obj.data.materials if material is not None])
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
        target_quad_count = _effective_target_quad_count(stats, settings)
        export_obj = _duplicate_object_for_export(context, source_obj)
        try:
            _export_object_to_fbx(context, export_obj, input_fbx)
        finally:
            _cleanup_export_duplicate(export_obj)

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
        _unsubdivide_mesh(
            context,
            remeshed_obj,
            iterations=settings.unsubdivide_iterations,
            target_count=settings.unsubdivide_target_count,
        )

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
    candidate_parts = [
        part
        for part in parts
        if not part.skipped and part.imported_object_name and _should_remesh_part(part, settings)
    ]
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
