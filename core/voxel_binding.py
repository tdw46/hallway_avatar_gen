from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import bpy

from ..utils.logging import get_logger
from ..utils import paths

logger = get_logger("voxel_binding")


@dataclass(frozen=True)
class VoxelBindingSettings:
    resolution: int = 128
    loops: int = 5
    samples: int = 64
    influence: int = 8
    falloff: float = 0.2
    detect_solidify: bool = False
    use_half_cpu_cores: bool = False


def _clear_quarantine(target: Path) -> None:
    if platform.system() != "Darwin" or not target.exists():
        return
    subprocess.run(
        ["xattr", "-dr", "com.apple.quarantine", str(target)],
        check=False,
        capture_output=True,
        text=True,
    )


def _runtime_root() -> Path:
    return paths.voxel_skinning_runtime_dir()


def _binary_relative_path() -> Path:
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Windows":
        if machine.endswith("amd64") or machine.endswith("x86_64"):
            return Path("bin") / "Windows" / "x64" / "vhd.exe"
        if machine.endswith("arm64") or machine.endswith("aarch64"):
            return Path("bin") / "Windows" / "arm64" / "vhd.exe"
        return Path("bin") / "Windows" / "x86" / "vhd.exe"

    if system == "Linux":
        if machine.endswith("amd64") or machine.endswith("x86_64"):
            return Path("bin") / "Linux" / "x64" / "vhd"
        if machine.endswith("arm64") or machine.endswith("aarch64"):
            return Path("bin") / "Linux" / "arm64" / "vhd"

    if system == "Darwin":
        return Path("bin") / "Darwin" / "vhd"

    raise FileNotFoundError(f"Unsupported platform for voxel binding: {system} {machine}")


def locate_voxel_binary() -> Path:
    relative = _binary_relative_path()
    root = _runtime_root()
    _clear_quarantine(root)
    binary_path = root / relative
    if binary_path.exists():
        _clear_quarantine(binary_path)
        if not os.access(binary_path, os.X_OK):
            os.chmod(binary_path, 0o755)
        return binary_path
    raise FileNotFoundError(f"Vendored voxel skinning binary not found at {binary_path}")


def _write_mesh_data(objs: list[bpy.types.Object], filepath: Path) -> None:
    with filepath.open("w", encoding="utf-8") as handle:
        handle.write("# voxel heat diffuse mesh export.\n")

        vertex_offset = 0
        for obj in objs:
            for vertex in obj.data.vertices:
                world_co = obj.matrix_world @ vertex.co
                handle.write(f"v,{world_co[0]},{world_co[1]},{world_co[2]}\n")

            for polygon in obj.data.polygons:
                handle.write("f")
                for loop_index in polygon.loop_indices:
                    vertex_index = obj.data.loops[loop_index].vertex_index
                    handle.write(f",{vertex_offset + vertex_index}")
                handle.write("\n")

            vertex_offset += len(obj.data.vertices)


def _write_bone_data(
    context: bpy.types.Context,
    armature_obj: bpy.types.Object,
    filepath: Path,
    bone_names: tuple[str, ...],
) -> tuple[str, ...]:
    valid_bones = tuple(name for name in bone_names if armature_obj.data.bones.get(name) is not None)
    if not valid_bones:
        return ()

    previous_active = context.view_layer.objects.active
    previous_selection = list(context.selected_objects)

    try:
        bpy.ops.object.select_all(action="DESELECT")
        armature_obj.select_set(True)
        context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode="EDIT")

        with filepath.open("w", encoding="utf-8") as handle:
            handle.write("# voxel heat diffuse bone export.\n")
            for name in valid_bones:
                bone = armature_obj.data.edit_bones.get(name)
                if bone is None or not bone.use_deform:
                    continue
                escaped_name = name.replace(",", "\\;")
                world_head = armature_obj.matrix_world @ bone.head
                world_tail = armature_obj.matrix_world @ bone.tail
                handle.write(
                    f"b,{escaped_name},{world_head[0]},{world_head[1]},{world_head[2]},"
                    f"{world_tail[0]},{world_tail[1]},{world_tail[2]}\n"
                )
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        for obj in previous_selection:
            if obj.name in bpy.data.objects:
                obj.select_set(True)
        if previous_active and previous_active.name in bpy.data.objects:
            context.view_layer.objects.active = previous_active

    return valid_bones


def _import_weight_data(objs: list[bpy.types.Object], filepath: Path) -> tuple[str, ...]:
    permutation: list[tuple[int, bpy.types.Object]] = []
    vertex_offset = 0
    for obj in objs:
        for vertex_index in range(len(obj.data.vertices)):
            permutation.append((vertex_index, obj))
        vertex_offset += len(obj.data.vertices)

    loaded_bones: list[str] = []

    with filepath.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line:
                continue
            tokens = line.strip("\r\n").split(",")
            if not tokens:
                continue
            if tokens[0] == "b":
                group_name = tokens[1].replace("\\;", ",")
                loaded_bones.append(group_name)
                for obj in objs:
                    existing = obj.vertex_groups.get(group_name)
                    if existing is not None:
                        obj.vertex_groups.remove(existing)
                    obj.vertex_groups.new(name=group_name)
                continue
            if tokens[0] != "w":
                continue

            group_name = loaded_bones[int(tokens[2])]
            flat_index = int(tokens[1])
            vertex_index, obj = permutation[flat_index]
            weight = float(tokens[3])
            obj.vertex_groups[group_name].add([vertex_index], weight, "REPLACE")

    return tuple(loaded_bones)


def run_voxel_heat_diffuse(
    context: bpy.types.Context,
    armature_obj: bpy.types.Object,
    objs: list[bpy.types.Object],
    bone_names: tuple[str, ...],
    *,
    settings: VoxelBindingSettings | None = None,
) -> tuple[str, ...]:
    if not objs:
        return ()

    settings = settings or VoxelBindingSettings()
    binary_path = locate_voxel_binary()
    mesh_objects = sorted(objs, key=lambda obj: obj.name)

    cache_root = paths.ensure_cache_dir() / "voxel_bind"
    cache_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hallway_voxel_", dir=str(cache_root)) as temp_dir:
        temp_path = Path(temp_dir)
        mesh_path = temp_path / "untitled-mesh.txt"
        bone_path = temp_path / "untitled-bone.txt"
        weight_path = temp_path / "untitled-weight.txt"

        _write_mesh_data(mesh_objects, mesh_path)
        valid_bones = _write_bone_data(context, armature_obj, bone_path, bone_names)
        if not valid_bones:
            raise RuntimeError("No valid deform bones were available for voxel binding.")

        command = [
            str(binary_path),
            mesh_path.name,
            bone_path.name,
            weight_path.name,
            str(settings.resolution),
            str(settings.loops),
            str(settings.samples),
            str(settings.influence),
            str(settings.falloff),
            "y" if settings.detect_solidify else "n",
            "y" if settings.use_half_cpu_cores else "n",
        ]

        logger.info(
            "Voxel binding %s with bones %s via %s",
            ", ".join(obj.name for obj in mesh_objects),
            valid_bones,
            binary_path,
        )

        process = subprocess.Popen(
            command,
            cwd=temp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output_lines: list[str] = []
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            output_lines.append(line)
            logger.info("Voxel solver | %s", line)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Voxel heat diffuse solver failed with exit code {return_code}: {' | '.join(output_lines)}"
            )
        if not weight_path.exists():
            raise RuntimeError("Voxel heat diffuse solver did not produce a weight file.")

        return _import_weight_data(mesh_objects, weight_path)
