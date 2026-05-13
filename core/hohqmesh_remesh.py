from __future__ import annotations

import math
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import bpy

from ..utils import paths
from ..utils.logging import get_logger
from .strip_remesh import (
    StripRemeshUnsupported,
    _boundary_loops,
    _clean_loop,
    _independent_loop_groups,
    _loop_area,
    _make_coord,
    _mesh_plane,
    _normalize_domain_loops,
    _point_in_loop,
    _project_loop,
    _source_rotation_scale_matrix,
    _target_edge_length,
)

logger = get_logger("hohqmesh")

_EPS = 1e-8
_HOHQMESH_VERSION = "1.5.5"


class HOHQMeshError(RuntimeError):
    pass


@dataclass(frozen=True)
class _BoundaryProfile:
    raw_vertices: int
    loop_count: int
    max_loop_vertices: int
    perimeter: float
    feature_count: int
    high_feature_count: int
    reflex_count: int
    vertex_density: float
    is_complex: bool
    require_containment: bool
    edge_length_ratio: float


def _env_int(name: str, default: int, minimum: int) -> int:
    value = os.environ.get(name, "").strip()
    if value:
        try:
            return max(minimum, int(value))
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", name, value)
    return default


def _env_float(name: str, default: float, minimum: float) -> float:
    value = os.environ.get(name, "").strip()
    if value:
        try:
            return max(minimum, float(value))
        except ValueError:
            logger.warning("Ignoring invalid %s=%r", name, value)
    return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return safe or "mesh"


def _platform_arch() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        arch = "arm64"
    elif machine in {"x86_64", "amd64"}:
        arch = "x86_64"
    else:
        arch = machine
    return system, arch


def _hohqmesh_executable() -> Path:
    override = os.environ.get("HALLWAY_HOHQMESH_EXECUTABLE", "").strip()
    if override:
        executable = Path(override).expanduser()
        if executable.exists():
            return executable
        raise HOHQMeshError(f"HALLWAY_HOHQMESH_EXECUTABLE does not exist: {executable}")

    system, arch = _platform_arch()
    executable_name = "HOHQMesh.exe" if system == "Windows" else "HOHQMesh"
    executable = paths.hohqmesh_runtime_dir() / system / arch / "bin" / executable_name
    if executable.exists():
        return executable
    raise HOHQMeshError(f"HOHQMesh runtime is not bundled for {system}/{arch}: {executable}")


def _line_block(name: str, start: tuple[float, float], end: tuple[float, float]) -> str:
    return (
        "   \\begin{END_POINTS_LINE}\n"
        f"      name = {name}\n"
        f"      xStart = [{start[0]:.12g},{start[1]:.12g},0.0]\n"
        f"      xEnd   = [{end[0]:.12g},{end[1]:.12g},0.0]\n"
        "   \\end{END_POINTS_LINE}\n"
    )


def _loop_chain_blocks(prefix: str, loop: list[tuple[float, float]]) -> str:
    blocks: list[str] = []
    for index, start in enumerate(loop):
        end = loop[(index + 1) % len(loop)]
        if math.hypot(end[0] - start[0], end[1] - start[1]) <= _EPS:
            continue
        blocks.append(_line_block(f"{prefix}_{index:04d}", start, end))
    if len(blocks) < 3:
        raise StripRemeshUnsupported("boundary loop has fewer than three usable edges")
    return "".join(blocks)


def _orient_ccw(loop: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return loop if _loop_area(loop) >= 0.0 else list(reversed(loop))


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    denom = (dx * dx) + (dy * dy)
    if denom <= _EPS:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (((px - ax) * dx) + ((py - ay) * dy)) / denom))
    qx = ax + (dx * t)
    qy = ay + (dy * t)
    return math.hypot(px - qx, py - qy)


def _rdp_open(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    best_index = 0
    best_distance = 0.0
    start = points[0]
    end = points[-1]
    for index in range(1, len(points) - 1):
        distance = _point_segment_distance(points[index], start, end)
        if distance > best_distance:
            best_distance = distance
            best_index = index
    if best_distance <= tolerance:
        return [start, end]
    return _rdp_open(points[: best_index + 1], tolerance)[:-1] + _rdp_open(points[best_index:], tolerance)


def _turn_angle_degrees(loop: list[tuple[float, float]], index: int) -> float:
    previous = loop[(index - 1) % len(loop)]
    current = loop[index]
    nxt = loop[(index + 1) % len(loop)]
    ax = current[0] - previous[0]
    ay = current[1] - previous[1]
    bx = nxt[0] - current[0]
    by = nxt[1] - current[1]
    if math.hypot(ax, ay) <= _EPS or math.hypot(bx, by) <= _EPS:
        return 0.0
    return math.degrees(math.atan2((ax * by) - (ay * bx), (ax * bx) + (ay * by)))


def _loop_perimeter(loop: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(
            loop[(index + 1) % len(loop)][0] - point[0],
            loop[(index + 1) % len(loop)][1] - point[1],
        )
        for index, point in enumerate(loop)
    )


def _boundary_profile(loops: list[list[tuple[float, float]]], edge_length: float) -> _BoundaryProfile:
    raw_vertices = sum(len(loop) for loop in loops)
    max_loop_vertices = max((len(loop) for loop in loops), default=0)
    perimeter = sum(_loop_perimeter(loop) for loop in loops)
    feature_count = 0
    high_feature_count = 0
    reflex_count = 0
    for loop in loops:
        ccw = _orient_ccw(loop)
        for index in range(len(ccw)):
            turn = _turn_angle_degrees(ccw, index)
            abs_turn = abs(turn)
            if abs_turn >= 15.0:
                feature_count += 1
            if abs_turn >= 60.0:
                high_feature_count += 1
            if turn < -15.0:
                reflex_count += 1
    vertex_density = raw_vertices / max(perimeter, _EPS)

    complex_raw_threshold = _env_int("HALLWAY_HOHQMESH_COMPLEX_RAW_VERTEX_THRESHOLD", 1600, 1)
    complex_loop_threshold = _env_int("HALLWAY_HOHQMESH_COMPLEX_MAX_LOOP_VERTEX_THRESHOLD", 1500, 1)
    complex_feature_threshold = _env_int("HALLWAY_HOHQMESH_COMPLEX_FEATURE_THRESHOLD", 120, 1)
    complex_feature_loop_threshold = _env_int("HALLWAY_HOHQMESH_COMPLEX_FEATURE_LOOP_VERTEX_THRESHOLD", 1000, 1)
    complex_density_threshold = _env_float("HALLWAY_HOHQMESH_COMPLEX_VERTEX_DENSITY_THRESHOLD", 1200.0, 0.0)
    is_complex = (
        raw_vertices >= complex_raw_threshold
        or max_loop_vertices >= complex_loop_threshold
        or (feature_count >= complex_feature_threshold and max_loop_vertices >= complex_feature_loop_threshold)
        or vertex_density >= complex_density_threshold
    )
    edge_length_ratio = _env_float("HALLWAY_HOHQMESH_COMPLEX_EDGE_LENGTH_RATIO", 0.75, 0.1) if is_complex else 1.0
    require_containment = _env_bool("HALLWAY_HOHQMESH_REQUIRE_CONTOUR_CONTAINMENT", is_complex)
    return _BoundaryProfile(
        raw_vertices=raw_vertices,
        loop_count=len(loops),
        max_loop_vertices=max_loop_vertices,
        perimeter=perimeter,
        feature_count=feature_count,
        high_feature_count=high_feature_count,
        reflex_count=reflex_count,
        vertex_density=vertex_density,
        is_complex=is_complex,
        require_containment=require_containment,
        edge_length_ratio=edge_length_ratio,
    )


def _simplify_loop_with_features(
    loop: list[tuple[float, float]],
    edge_length: float,
    *,
    angle_threshold_override: float | None = None,
    tolerance_ratio_override: float | None = None,
    max_segments_override: int | None = None,
) -> list[tuple[float, float]]:
    max_segments = max_segments_override or _env_int("HALLWAY_HOHQMESH_MAX_BOUNDARY_SEGMENTS", 384, 16)
    angle_threshold = angle_threshold_override
    if angle_threshold is None:
        angle_threshold = _env_float("HALLWAY_HOHQMESH_FEATURE_ANGLE_DEGREES", 15.0, 0.0)
    configured_ratio = os.environ.get("HALLWAY_HOHQMESH_SIMPLIFY_TOLERANCE_RATIO", "").strip()
    if tolerance_ratio_override is not None:
        tolerance_ratio = tolerance_ratio_override
    elif configured_ratio:
        tolerance_ratio = _env_float("HALLWAY_HOHQMESH_SIMPLIFY_TOLERANCE_RATIO", 0.08, 0.0)
    else:
        complex_vertex_threshold = _env_int("HALLWAY_HOHQMESH_COMPLEX_LOOP_VERTEX_THRESHOLD", 1500, 1)
        default_complex_ratio = _env_float("HALLWAY_HOHQMESH_COMPLEX_SIMPLIFY_TOLERANCE_RATIO", 0.24, 0.0)
        tolerance_ratio = default_complex_ratio if len(loop) >= complex_vertex_threshold else 0.08
    tolerance = max(
        _env_float("HALLWAY_HOHQMESH_SIMPLIFY_MIN_TOLERANCE", 0.0005, 0.0),
        edge_length * tolerance_ratio,
    )

    def simplify_at(current_tolerance: float) -> list[tuple[float, float]]:
        forced = {0}
        max_forced = max(4, max_segments - 2)
        scored_features = [
            (abs(_turn_angle_degrees(loop, index)), index)
            for index in range(len(loop))
            if abs(_turn_angle_degrees(loop, index)) >= angle_threshold
        ]
        scored_features.sort(reverse=True)
        for _score, index in scored_features[:max_forced]:
            forced.add(index)
        forced_indices = sorted(forced)
        simplified: list[tuple[float, float]] = []
        for start_index, end_index in zip(forced_indices, forced_indices[1:] + [forced_indices[0] + len(loop)]):
            segment = [loop[index % len(loop)] for index in range(start_index, end_index + 1)]
            simplified.extend(_rdp_open(segment, current_tolerance)[:-1])
        return _clean_loop(simplified)

    simplified = simplify_at(tolerance)
    max_tolerance = max(tolerance, edge_length * 0.5)
    while len(simplified) > max_segments and tolerance < max_tolerance:
        tolerance *= 1.5
        simplified = simplify_at(tolerance)
    return simplified if len(simplified) >= 3 else loop


def _outward_biased_loop(
    loop: list[tuple[float, float]],
    edge_length: float,
    *,
    is_outer: bool,
    bias_ratio_override: float | None = None,
) -> list[tuple[float, float]]:
    if bias_ratio_override is None:
        bias_ratio = _env_float("HALLWAY_HOHQMESH_OUTWARD_BIAS_RATIO", 0.04, 0.0)
    else:
        bias_ratio = bias_ratio_override
    bias = edge_length * bias_ratio
    if bias <= _EPS:
        return loop
    ccw = _orient_ccw(loop)
    result: list[tuple[float, float]] = []
    for index, point in enumerate(ccw):
        previous = ccw[(index - 1) % len(ccw)]
        nxt = ccw[(index + 1) % len(ccw)]
        normals: list[tuple[float, float]] = []
        for start, end in ((previous, point), (point, nxt)):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= _EPS:
                continue
            inward = (-dy / length, dx / length)
            normals.append((-inward[0], -inward[1]) if is_outer else inward)
        if not normals:
            result.append(point)
            continue
        nx = sum(normal[0] for normal in normals)
        ny = sum(normal[1] for normal in normals)
        length = math.hypot(nx, ny)
        if length <= _EPS:
            nx, ny = normals[-1]
            length = 1.0
        result.append((point[0] + (nx / length) * bias, point[1] + (ny / length) * bias))
    return _clean_loop(result)


def _loop_contains_point_with_tolerance(
    loop: list[tuple[float, float]],
    point: tuple[float, float],
    tolerance: float,
) -> bool:
    if _point_in_loop(point, loop):
        return True
    return any(
        _point_segment_distance(point, start, loop[(index + 1) % len(loop)]) <= tolerance
        for index, start in enumerate(loop)
    )


def _coverage_miss_count(
    raw_loop: list[tuple[float, float]],
    prepared_loop: list[tuple[float, float]],
    tolerance: float,
) -> int:
    return sum(1 for point in raw_loop if not _loop_contains_point_with_tolerance(prepared_loop, point, tolerance))


def _coverage_miss_indices(
    raw_loop: list[tuple[float, float]],
    prepared_loop: list[tuple[float, float]],
    tolerance: float,
) -> list[int]:
    return [
        index
        for index, point in enumerate(raw_loop)
        if not _loop_contains_point_with_tolerance(prepared_loop, point, tolerance)
    ]


def _nearest_loop_index(loop: list[tuple[float, float]], point: tuple[float, float]) -> int:
    best_index = 0
    best_distance = float("inf")
    for index, candidate in enumerate(loop):
        distance = (candidate[0] - point[0]) ** 2 + (candidate[1] - point[1]) ** 2
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def _repair_simplified_loop_containment(
    raw_loop: list[tuple[float, float]],
    simplified_loop: list[tuple[float, float]],
    missed_indices: list[int],
) -> list[tuple[float, float]]:
    if not missed_indices:
        return simplified_loop
    selected = {_nearest_loop_index(raw_loop, point) for point in simplified_loop}
    window = _env_int("HALLWAY_HOHQMESH_CONTAINMENT_REPAIR_VERTEX_WINDOW", 0, 0)
    max_vertices = _env_int("HALLWAY_HOHQMESH_CONTAINMENT_REPAIR_MAX_VERTICES", 140, 8)
    for missed_index in missed_indices:
        for offset in range(-window, window + 1):
            selected.add((missed_index + offset) % len(raw_loop))
        if len(selected) >= max_vertices:
            break
    if len(selected) > max_vertices:
        locked = sorted(selected)
        stride = math.ceil(len(locked) / max_vertices)
        selected = set(locked[::stride])
        for missed_index in missed_indices:
            selected.add(missed_index)
            if len(selected) >= max_vertices:
                break
    return _clean_loop([raw_loop[index] for index in sorted(selected)])


def _ensure_outer_loop_covers_raw(
    raw_loop: list[tuple[float, float]],
    simplified_loop: list[tuple[float, float]],
    edge_length: float,
    initial_bias_ratio: float,
) -> tuple[list[tuple[float, float]], int]:
    tolerance = max(edge_length * 0.04, 0.0008)
    bias_ratio = initial_bias_ratio
    max_bias_ratio = _env_float("HALLWAY_HOHQMESH_CONTAINMENT_MAX_BIAS_RATIO", 1.25, 0.0)
    repair_loops = _env_int("HALLWAY_HOHQMESH_CONTAINMENT_REPAIR_LOOPS", 1, 0)
    base_loop = simplified_loop
    best_loop = _outward_biased_loop(base_loop, edge_length, is_outer=True, bias_ratio_override=bias_ratio)
    best_misses = _coverage_miss_count(raw_loop, best_loop, tolerance)
    while best_misses and bias_ratio < max_bias_ratio:
        bias_ratio = min(max_bias_ratio, max(bias_ratio * 1.6, bias_ratio + 0.04))
        candidate = _outward_biased_loop(base_loop, edge_length, is_outer=True, bias_ratio_override=bias_ratio)
        misses = _coverage_miss_count(raw_loop, candidate, tolerance)
        if misses <= best_misses:
            best_loop = candidate
            best_misses = misses
        if misses == 0:
            return best_loop, best_misses
    for _ in range(repair_loops):
        if best_misses == 0:
            break
        missed_indices = _coverage_miss_indices(raw_loop, best_loop, tolerance)
        repaired = _repair_simplified_loop_containment(raw_loop, base_loop, missed_indices)
        if len(repaired) <= len(base_loop):
            break
        base_loop = repaired
        candidate = _outward_biased_loop(base_loop, edge_length, is_outer=True, bias_ratio_override=bias_ratio)
        misses = _coverage_miss_count(raw_loop, candidate, tolerance)
        if misses <= best_misses:
            best_loop = candidate
            best_misses = misses
    return best_loop, best_misses


def _prepare_group_loops_for_hohqmesh(
    loops: list[list[tuple[float, float]]],
    edge_length: float,
    *,
    angle_threshold: float | None = None,
    tolerance_ratio: float | None = None,
    max_segments: int | None = None,
    outward_bias_ratio: float | None = None,
    contain_original: bool = False,
) -> list[list[tuple[float, float]]]:
    prepared: list[list[tuple[float, float]]] = []
    raw_vertices = sum(len(loop) for loop in loops)
    containment_misses = 0
    for index, loop in enumerate(loops):
        simplified = _simplify_loop_with_features(
            _orient_ccw(loop),
            edge_length,
            angle_threshold_override=angle_threshold,
            tolerance_ratio_override=tolerance_ratio,
            max_segments_override=max_segments,
        )
        if contain_original and index == 0:
            initial_bias = outward_bias_ratio
            if initial_bias is None:
                initial_bias = _env_float(
                    "HALLWAY_HOHQMESH_CONTAINMENT_OUTWARD_BIAS_RATIO",
                    0.18,
                    0.0,
                )
            biased, containment_misses = _ensure_outer_loop_covers_raw(
                _orient_ccw(loop),
                simplified,
                edge_length,
                initial_bias,
            )
        else:
            biased = _outward_biased_loop(
                simplified,
                edge_length,
                is_outer=index == 0,
                bias_ratio_override=outward_bias_ratio,
            )
        if len(biased) < 3 or abs(_loop_area(biased)) <= _EPS:
            raise StripRemeshUnsupported("simplified HOHQMesh boundary became degenerate")
        prepared.append(biased)
    logger.info(
        "HOHQMesh boundary prep -> raw_vertices=%s prepared_vertices=%s max_segments=%s feature_angle=%s tolerance_ratio=%s containment_misses=%s",
        raw_vertices,
        sum(len(loop) for loop in prepared),
        max(len(loop) for loop in prepared),
        angle_threshold if angle_threshold is not None else "auto",
        tolerance_ratio if tolerance_ratio is not None else "auto",
        containment_misses,
    )
    return prepared


def _control_text(
    mesh_filename: str,
    edge_length: float,
    loops: list[list[tuple[float, float]]],
) -> str:
    smoothing_iterations = _env_int("HALLWAY_HOHQMESH_SMOOTHING_ITERATIONS", 20, 0)
    polynomial_order = _env_int("HALLWAY_HOHQMESH_POLYNOMIAL_ORDER", 2, 1)
    background = max(edge_length, _env_float("HALLWAY_HOHQMESH_MIN_BACKGROUND_GRID", 0.002, 0.0001))

    outer = _orient_ccw(loops[0])
    inner_loops = [_orient_ccw(loop) for loop in loops[1:]]
    inner_text = ""
    if inner_loops:
        chains = []
        for loop_index, loop in enumerate(inner_loops):
            chains.append(
                f"   \\begin{{CHAIN}}\n"
                f"      name = hole_{loop_index:03d}\n"
                f"{_loop_chain_blocks(f'hole_{loop_index:03d}', loop)}"
                "   \\end{CHAIN}\n"
            )
        inner_text = "   \\begin{INNER_BOUNDARIES}\n" + "".join(chains) + "   \\end{INNER_BOUNDARIES}\n"

    smoother = ""
    if smoothing_iterations > 0:
        smoother = (
            "   \\begin{SPRING_SMOOTHER}\n"
            "      smoothing            = ON\n"
            "      smoothing type       = LinearAndCrossbarSpring\n"
            f"      number of iterations = {smoothing_iterations}\n"
            "   \\end{SPRING_SMOOTHER}\n"
        )

    return (
        "\\begin{CONTROL_INPUT}\n"
        "   \\begin{RUN_PARAMETERS}\n"
        f"      mesh file name   = {mesh_filename}\n"
        "      plot file name   = none\n"
        "      stats file name  = none\n"
        "      mesh file format = ABAQUS\n"
        f"      polynomial order = {polynomial_order}\n"
        "      plot file format = skeleton\n"
        "   \\end{RUN_PARAMETERS}\n"
        "   \\begin{BACKGROUND_GRID}\n"
        f"      background grid size = [{background:.12g},{background:.12g},0.0]\n"
        "   \\end{BACKGROUND_GRID}\n"
        f"{smoother}"
        "\\end{CONTROL_INPUT}\n"
        "\\begin{MODEL}\n"
        "   \\begin{OUTER_BOUNDARY}\n"
        f"{_loop_chain_blocks('outer', outer)}"
        "   \\end{OUTER_BOUNDARY}\n"
        f"{inner_text}"
        "\\end{MODEL}\n"
        "\\end{FILE}\n"
    )


def _parse_abaqus_mesh(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    nodes: dict[int, tuple[float, float, float]] = {}
    faces: list[tuple[int, int, int, int]] = []
    mode = ""
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("**"):
            continue
        upper = line.upper()
        if upper.startswith("*NODE"):
            mode = "node"
            continue
        if upper.startswith("*ELEMENT"):
            mode = "element"
            continue
        if line.startswith("*"):
            mode = ""
            continue
        if mode == "node":
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif mode == "element":
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            faces.append((int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])))

    if not nodes or not faces:
        raise HOHQMeshError(f"HOHQMesh did not write a usable Abaqus mesh: {path}")

    ordered_ids = sorted(nodes)
    id_to_index = {node_id: index for index, node_id in enumerate(ordered_ids)}
    vertices = [nodes[node_id] for node_id in ordered_ids]
    remapped_faces: list[tuple[int, int, int, int]] = []
    skipped_faces = 0
    for face in faces:
        if any(node_id not in id_to_index for node_id in face):
            skipped_faces += 1
            continue
        remapped_faces.append(tuple(id_to_index[node_id] for node_id in face))
    if skipped_faces:
        logger.warning("HOHQMesh skipped %s invalid Abaqus faces from %s", skipped_faces, path)
    if not remapped_faces:
        raise HOHQMeshError(f"HOHQMesh Abaqus mesh had no valid faces after filtering: {path}")
    return vertices, remapped_faces


def _run_hohqmesh(job_dir: Path, control_path: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    executable = _hohqmesh_executable()
    command = [str(executable), "-f", control_path.name, "-sLimit", str(_env_int("HALLWAY_HOHQMESH_SUBDIVISION_LIMIT", 10, 1))]
    if _env_bool("HALLWAY_HOHQMESH_VERBOSE", False):
        command.append("-verbose")
    env = os.environ.copy()
    lib_dir = executable.parent.parent / "lib"
    if platform.system() == "Linux" and lib_dir.exists():
        env["LD_LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}".rstrip(os.pathsep)
    if platform.system() == "Darwin" and lib_dir.exists():
        env["DYLD_LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{env.get('DYLD_LIBRARY_PATH', '')}".rstrip(os.pathsep)
    return subprocess.run(
        command,
        cwd=str(job_dir),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _build_group_mesh(
    source_obj: bpy.types.Object,
    group_index: int,
    loops: list[list[tuple[float, float]]],
    plane,
    edge_length: float,
    cache_dir: Path,
    profile: _BoundaryProfile,
) -> bpy.types.Mesh:
    safe = _safe_name(f"{source_obj.name}_{group_index:02d}")
    job_dir = cache_dir / safe
    job_dir.mkdir(parents=True, exist_ok=True)
    mesh_filename = f"{safe}.inp"
    control_path = job_dir / f"{safe}.control"
    output_path = job_dir / mesh_filename
    timeout = _env_float("HALLWAY_HOHQMESH_TIMEOUT", 30.0, 1.0)
    if profile.is_complex:
        timeout = _env_float("HALLWAY_HOHQMESH_COMPLEX_TIMEOUT", max(timeout, 60.0), 1.0)

    if profile.is_complex:
        attempts: list[tuple[float | None, float | None, int | None, float | None]] = [
            (None, None, None, None),
            (60.0, 0.45, 160, None),
            (90.0, 0.65, 128, None),
            (120.0, 0.85, 96, None),
            (150.0, 1.05, 72, None),
        ]
    else:
        attempts = [
            (None, None, None, None),
            (75.0, 0.80, 96, 0.03),
            (105.0, 1.00, 64, 0.02),
            (135.0, 1.20, 48, 0.0),
        ]
    last_error: HOHQMeshError | None = None
    prepared_loops: list[list[tuple[float, float]]] | None = None
    parsed_mesh: tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]] | None = None
    elapsed = 0.0
    for attempt_index, (angle, tolerance_ratio, max_segments, bias_ratio) in enumerate(attempts):
        prepared_loops = _prepare_group_loops_for_hohqmesh(
            loops,
            edge_length,
            angle_threshold=angle,
            tolerance_ratio=tolerance_ratio,
            max_segments=max_segments,
            outward_bias_ratio=bias_ratio,
            contain_original=profile.require_containment,
        )
        if profile.require_containment and prepared_loops:
            containment_tolerance = max(edge_length * 0.04, 0.0008)
            containment_misses = _coverage_miss_count(_orient_ccw(loops[0]), prepared_loops[0], containment_tolerance)
            max_misses = _env_int("HALLWAY_HOHQMESH_MAX_CONTAINMENT_MISSES", 0, 0)
            if containment_misses > max_misses:
                last_error = HOHQMeshError(
                    f"prepared boundary missed {containment_misses} original contour vertices "
                    f"(allowed {max_misses})"
                )
                logger.warning(
                    "HOHQMesh attempt %s skipped for %s:%s: %s",
                    attempt_index + 1,
                    source_obj.name,
                    group_index,
                    last_error,
                )
                continue
        if output_path.exists():
            output_path.unlink()
        control_path.write_text(_control_text(mesh_filename, edge_length, prepared_loops), encoding="utf-8")
        started = time.monotonic()
        try:
            result = _run_hohqmesh(job_dir, control_path, timeout)
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            detail = (exc.stderr or exc.stdout or "").strip()
            last_error = HOHQMeshError(f"HOHQMesh timed out after {timeout:.1f}s: {detail[-2000:]}")
            logger.warning("HOHQMesh attempt %s failed for %s:%s: %s", attempt_index + 1, source_obj.name, group_index, last_error)
            if output_path.exists():
                output_path.unlink()
            continue
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            last_error = HOHQMeshError(f"HOHQMesh exited {result.returncode} after {elapsed:.2f}s: {detail[-2000:]}")
            logger.warning("HOHQMesh attempt %s failed for %s:%s: %s", attempt_index + 1, source_obj.name, group_index, last_error)
            if output_path.exists():
                output_path.unlink()
            continue
        if not output_path.exists():
            detail = (result.stderr or result.stdout).strip()
            last_error = HOHQMeshError(f"HOHQMesh produced no mesh after {elapsed:.2f}s: {detail[-2000:]}")
            logger.warning("HOHQMesh attempt %s failed for %s:%s: %s", attempt_index + 1, source_obj.name, group_index, last_error)
            continue
        try:
            parsed_mesh = _parse_abaqus_mesh(output_path)
        except HOHQMeshError as exc:
            last_error = exc
            logger.warning("HOHQMesh attempt %s failed for %s:%s: %s", attempt_index + 1, source_obj.name, group_index, last_error)
            if output_path.exists():
                output_path.unlink()
            continue
        break
    else:
        raise last_error or HOHQMeshError("HOHQMesh failed without diagnostic output")
    if prepared_loops is None:
        raise HOHQMeshError("HOHQMesh boundary preparation failed")
    if parsed_mesh is None:
        raise last_error or HOHQMeshError("HOHQMesh produced no parseable mesh")

    planar_vertices, faces = parsed_mesh
    max_faces = _env_int("HALLWAY_HOHQMESH_MAX_FACES", 150000, 1)
    if len(faces) > max_faces:
        raise StripRemeshUnsupported(f"HOHQMesh exceeded face cap ({max_faces}); produced {len(faces)} faces")

    vertices = [_make_coord(plane, point[0], point[1]) for point in planar_vertices]
    mesh = bpy.data.meshes.new(f"{source_obj.data.name}_hohqmesh_{group_index:02d}")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    logger.info(
        "HOHQMesh group %s:%s -> loops=%s boundary_vertices=%s verts=%s faces=%s edge_length=%.4f elapsed=%.3fs",
        source_obj.name,
        group_index,
        len(prepared_loops),
        sum(len(loop) for loop in prepared_loops),
        len(mesh.vertices),
        len(mesh.polygons),
        edge_length,
        elapsed,
    )
    return mesh


def _combine_meshes(name: str, meshes: list[bpy.types.Mesh]) -> bpy.types.Mesh:
    if len(meshes) == 1:
        mesh = meshes[0]
        mesh.name = name
        return mesh
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    for mesh in meshes:
        offset = len(vertices)
        vertices.extend(tuple(vertex.co) for vertex in mesh.vertices)
        faces.extend(tuple(offset + index for index in polygon.vertices) for polygon in mesh.polygons)
    combined = bpy.data.meshes.new(name)
    combined.from_pydata(vertices, [], faces)
    combined.update(calc_edges=True)
    for mesh in meshes:
        bpy.data.meshes.remove(mesh)
    return combined


def remesh_object(
    context: bpy.types.Context,
    source_obj: bpy.types.Object,
    *,
    scale_factor: float = 10.0,
    enabled: bool = True,
) -> bpy.types.Object | None:
    if not enabled or source_obj.type != "MESH" or len(source_obj.data.polygons) == 0:
        return None

    started_at = time.monotonic()
    transform = _source_rotation_scale_matrix(source_obj)
    coords = [transform @ vertex.co for vertex in source_obj.data.vertices]
    plane = _mesh_plane(coords)
    raw_loops = [_project_loop(coords, loop, plane) for loop in _boundary_loops(source_obj.data)]
    loops = [_clean_loop(loop) for loop in raw_loops if len(loop) >= 3]
    loops = [loop for loop in loops if len(loop) >= 3 and abs(_loop_area(loop)) > _EPS]
    if not loops:
        raise StripRemeshUnsupported("no usable boundary loops")

    normalized = _normalize_domain_loops(loops)
    base_edge_length = _target_edge_length(scale_factor)
    profile = _boundary_profile(normalized, base_edge_length)
    edge_length = base_edge_length * profile.edge_length_ratio
    groups = [
        _normalize_domain_loops(group)
        for group in _independent_loop_groups(normalized)
    ]
    logger.info(
        "HOHQMesh input %s -> groups=%s loops=%s raw_boundary_vertices=%s max_loop_vertices=%s features=%s reflex=%s complex=%s contain=%s edge_length=%.4f version=%s",
        source_obj.name,
        len(groups),
        len(normalized),
        profile.raw_vertices,
        profile.max_loop_vertices,
        profile.feature_count,
        profile.reflex_count,
        profile.is_complex,
        profile.require_containment,
        edge_length,
        _HOHQMESH_VERSION,
    )

    cache_dir = paths.ensure_cache_dir() / "hohqmesh"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meshes = [
        _build_group_mesh(source_obj, group_index, group, plane, edge_length, cache_dir, profile)
        for group_index, group in enumerate(groups)
    ]
    final_mesh = _combine_meshes(f"{source_obj.data.name}_hohqmesh", meshes)
    final_obj = bpy.data.objects.new(f"{source_obj.name}__hallway_hohqmesh", final_mesh)
    final_obj.location = source_obj.location.copy()
    try:
        inverse_transform = transform.inverted()
        final_obj["hallway_avatar_hohqmesh_source_rs_inverse"] = [float(value) for row in inverse_transform for value in row]
    except ValueError:
        pass
    for collection in list(source_obj.users_collection) or [context.scene.collection]:
        collection.objects.link(final_obj)
    logger.info(
        "HOHQMesh remeshed %s -> groups=%s verts=%s faces=%s quads=%s elapsed=%.3fs",
        source_obj.name,
        len(groups),
        len(final_mesh.vertices),
        len(final_mesh.polygons),
        sum(1 for polygon in final_mesh.polygons if polygon.loop_total == 4),
        time.monotonic() - started_at,
    )
    return final_obj
