from __future__ import annotations

import math
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from heapq import heappop, heappush
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


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    def orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return ((q[0] - p[0]) * (r[1] - p[1])) - ((q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
        return (
            min(p[0], r[0]) - _EPS <= q[0] <= max(p[0], r[0]) + _EPS
            and min(p[1], r[1]) - _EPS <= q[1] <= max(p[1], r[1]) + _EPS
            and abs(orient(p, q, r)) <= _EPS
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if (o1 * o2 < -_EPS) and (o3 * o4 < -_EPS):
        return True
    return (
        on_segment(a, c, b)
        or on_segment(a, d, b)
        or on_segment(c, a, d)
        or on_segment(c, b, d)
    )


def _loop_has_self_intersections(loop: list[tuple[float, float]]) -> bool:
    if len(loop) < 4:
        return False
    for i, a in enumerate(loop):
        b = loop[(i + 1) % len(loop)]
        for j in range(i + 1, len(loop)):
            if j == i or j == (i + 1) % len(loop) or i == (j + 1) % len(loop):
                continue
            c = loop[j]
            d = loop[(j + 1) % len(loop)]
            if _segments_intersect(a, b, c, d):
                return True
    return False


def _densify_loop_min_segments(loop: list[tuple[float, float]], min_segments: int) -> list[tuple[float, float]]:
    if len(loop) >= min_segments or len(loop) < 3:
        return loop
    result: list[tuple[float, float]] = []
    missing = min_segments - len(loop)
    edge_lengths = [
        math.hypot(loop[(index + 1) % len(loop)][0] - point[0], loop[(index + 1) % len(loop)][1] - point[1])
        for index, point in enumerate(loop)
    ]
    total = sum(edge_lengths) or 1.0
    extra_by_edge = [0 for _ in loop]
    for _ in range(missing):
        index = max(range(len(loop)), key=lambda item: (edge_lengths[item] / total) / (extra_by_edge[item] + 1))
        extra_by_edge[index] += 1
    for index, point in enumerate(loop):
        result.append(point)
        nxt = loop[(index + 1) % len(loop)]
        for step in range(1, extra_by_edge[index] + 1):
            t = step / (extra_by_edge[index] + 1)
            result.append((point[0] + (nxt[0] - point[0]) * t, point[1] + (nxt[1] - point[1]) * t))
    return _clean_loop(result)


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


def _turn_angle_from_points(
    previous: tuple[float, float],
    current: tuple[float, float],
    nxt: tuple[float, float],
) -> float:
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


def _triangle_area(
    previous: tuple[float, float],
    current: tuple[float, float],
    nxt: tuple[float, float],
) -> float:
    return abs(
        ((current[0] - previous[0]) * (nxt[1] - previous[1]))
        - ((current[1] - previous[1]) * (nxt[0] - previous[0]))
    ) * 0.5


def _edge_bbox_intersects(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    return (
        max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0])) + _EPS
        and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1])) + _EPS
    )


def _topology_safe_feature_indices(
    loop: list[tuple[float, float]],
    max_segments: int,
    angle_threshold: float,
) -> set[int]:
    forced = {0}
    reversal_threshold = _env_float("HALLWAY_HOHQMESH_REVERSAL_TURN_DEGREES", 150.0, 90.0)
    feature_budget = max(0, min(max_segments - 4, max(4, max_segments // 2)))
    scored_features = [
        (abs(_turn_angle_degrees(loop, index)), index)
        for index in range(len(loop))
        if angle_threshold <= abs(_turn_angle_degrees(loop, index)) < reversal_threshold
    ]
    scored_features.sort(reverse=True)
    for _score, index in scored_features[:feature_budget]:
        forced.add(index)

    anchor_budget = max(0, min(max_segments // 8, max_segments - len(forced) - 4))
    if anchor_budget > 0:
        stride = max(1, len(loop) // anchor_budget)
        for index in range(0, len(loop), stride):
            forced.add(index)
            if len(forced) >= max_segments - 4:
                break
    return forced


def _topology_safe_simplify_loop(
    loop: list[tuple[float, float]],
    max_segments: int,
    angle_threshold: float,
) -> list[tuple[float, float]]:
    if len(loop) <= max_segments:
        return loop
    if len(loop) < 4:
        return loop

    forced = _topology_safe_feature_indices(loop, max_segments, angle_threshold)
    n = len(loop)
    previous_by_index = {index: (index - 1) % n for index in range(n)}
    next_by_index = {index: (index + 1) % n for index in range(n)}
    active: set[int] = set(range(n))
    versions = {index: 0 for index in range(n)}
    heap: list[tuple[float, int, int]] = []

    def importance(index: int) -> float:
        if index in forced:
            return float("inf")
        previous = previous_by_index[index]
        nxt = next_by_index[index]
        area = _triangle_area(loop[previous], loop[index], loop[nxt])
        turn = abs(_turn_angle_from_points(loop[previous], loop[index], loop[nxt]))
        reversal_threshold = _env_float("HALLWAY_HOHQMESH_REVERSAL_TURN_DEGREES", 150.0, 90.0)
        if turn >= reversal_threshold:
            return area * 0.05
        return area * (1.0 + (turn / 180.0))

    def push(index: int) -> None:
        if index in active and index not in forced:
            heappush(heap, (importance(index), versions[index], index))

    def removal_keeps_simple(index: int) -> bool:
        if index in forced or len(active) <= 3:
            return False
        previous = previous_by_index[index]
        nxt = next_by_index[index]
        if previous == nxt:
            return False
        a = loop[previous]
        b = loop[nxt]
        if math.hypot(b[0] - a[0], b[1] - a[1]) <= _EPS:
            return False
        skip = {previous, index, nxt}
        for edge_start in active:
            edge_end = next_by_index[edge_start]
            if edge_start in skip or edge_end in skip:
                continue
            c = loop[edge_start]
            d = loop[edge_end]
            if _edge_bbox_intersects(a, b, c, d) and _segments_intersect(a, b, c, d):
                return False
        return True

    for index in range(n):
        push(index)

    blocked_rounds = 0
    while len(active) > max_segments and heap:
        _score, version, index = heappop(heap)
        if index not in active or version != versions[index]:
            continue
        if not removal_keeps_simple(index):
            blocked_rounds += 1
            if blocked_rounds > len(active):
                break
            versions[index] += 1
            continue
        blocked_rounds = 0
        previous = previous_by_index[index]
        nxt = next_by_index[index]
        active.remove(index)
        next_by_index[previous] = nxt
        previous_by_index[nxt] = previous
        for neighbor in (previous, nxt):
            versions[neighbor] += 1
            push(neighbor)

    start = min(active)
    result: list[tuple[float, float]] = []
    index = start
    for _ in range(len(active)):
        result.append(loop[index])
        index = next_by_index[index]
        if index == start:
            break
    return _clean_loop(result)


def _remove_pathological_reversals(
    loop: list[tuple[float, float]],
    *,
    turn_threshold: float,
) -> list[tuple[float, float]]:
    if len(loop) <= 3:
        return loop

    def removal_keeps_simple(points: list[tuple[float, float]], index: int) -> bool:
        if len(points) <= 3:
            return False
        previous = (index - 1) % len(points)
        nxt = (index + 1) % len(points)
        a = points[previous]
        b = points[nxt]
        if math.hypot(b[0] - a[0], b[1] - a[1]) <= _EPS:
            return False
        skip = {previous, index, nxt}
        for edge_start, c in enumerate(points):
            edge_end = (edge_start + 1) % len(points)
            if edge_start in skip or edge_end in skip:
                continue
            d = points[edge_end]
            if _edge_bbox_intersects(a, b, c, d) and _segments_intersect(a, b, c, d):
                return False
        return True

    result = list(loop)
    max_passes = _env_int("HALLWAY_HOHQMESH_REVERSAL_CLEANUP_PASSES", 6, 0)
    for _ in range(max_passes):
        removed = False
        candidates = sorted(
            (
                (
                    abs(_turn_angle_degrees(result, index)),
                    min(
                        math.hypot(result[index][0] - result[(index - 1) % len(result)][0], result[index][1] - result[(index - 1) % len(result)][1]),
                        math.hypot(result[(index + 1) % len(result)][0] - result[index][0], result[(index + 1) % len(result)][1] - result[index][1]),
                    ),
                    index,
                )
                for index in range(len(result))
                if abs(_turn_angle_degrees(result, index)) >= turn_threshold
            ),
            reverse=True,
        )
        if not candidates:
            break
        for _turn, _short_edge, index in candidates:
            if index >= len(result):
                continue
            if abs(_turn_angle_degrees(result, index)) < turn_threshold:
                continue
            if removal_keeps_simple(result, index):
                result.pop(index)
                removed = True
                break
        if not removed:
            break
    return _clean_loop(result)


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
    complex_vertex_threshold = _env_int("HALLWAY_HOHQMESH_COMPLEX_LOOP_VERTEX_THRESHOLD", 1500, 1)
    use_topology_safe = _env_bool(
        "HALLWAY_HOHQMESH_TOPOLOGY_SAFE_SIMPLIFY",
        len(loop) >= complex_vertex_threshold,
    )
    if use_topology_safe:
        simplified = _topology_safe_simplify_loop(loop, max_segments, angle_threshold)
        return simplified if len(simplified) >= 3 else loop

    def simplify_at(current_tolerance: float) -> list[tuple[float, float]]:
        forced = {0}
        max_forced = max(4, min(max_segments - 4, max_segments // 2))
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
    preserved = set(selected)
    window = _env_int("HALLWAY_HOHQMESH_CONTAINMENT_REPAIR_VERTEX_WINDOW", 0, 0)
    protected: set[int] = set()
    for missed_index in missed_indices:
        for offset in range(-window, window + 1):
            protected.add((missed_index + offset) % len(raw_loop))
    selected.update(protected)

    configured_max = _env_int("HALLWAY_HOHQMESH_CONTAINMENT_REPAIR_MAX_VERTICES", 1600, 8)
    hard_max = _env_int("HALLWAY_HOHQMESH_CONTAINMENT_REPAIR_HARD_MAX_VERTICES", 4096, 8)
    max_vertices = min(len(raw_loop), max(configured_max, len(preserved) + len(protected)))
    max_vertices = min(max_vertices, hard_max)
    if len(selected) > max_vertices:
        required = set(protected)
        if len(required) >= max_vertices:
            locked = sorted(required)
            stride = math.ceil(len(locked) / max_vertices)
            selected = set(locked[::stride])
        else:
            budget = max_vertices - len(required)
            optional = sorted(selected - required)
            if len(optional) > budget:
                stride = math.ceil(len(optional) / budget)
                optional = optional[::stride][:budget]
            selected = required | set(optional)
    return _clean_loop([raw_loop[index] for index in sorted(selected)])


def _containment_miss_limit(raw_loop: list[tuple[float, float]]) -> int:
    configured = os.environ.get("HALLWAY_HOHQMESH_MAX_CONTAINMENT_MISSES", "").strip()
    if configured:
        return _env_int("HALLWAY_HOHQMESH_MAX_CONTAINMENT_MISSES", 0, 0)
    miss_ratio = _env_float("HALLWAY_HOHQMESH_MAX_CONTAINMENT_MISS_RATIO", 0.0125, 0.0)
    return max(8, math.ceil(len(raw_loop) * miss_ratio))


def _ensure_outer_loop_covers_raw(
    raw_loop: list[tuple[float, float]],
    simplified_loop: list[tuple[float, float]],
    edge_length: float,
    initial_bias_ratio: float,
) -> tuple[list[tuple[float, float]], int]:
    tolerance = max(edge_length * 0.04, 0.0008)
    bias_ratio = initial_bias_ratio
    max_bias_ratio = _env_float("HALLWAY_HOHQMESH_CONTAINMENT_MAX_BIAS_RATIO", 1.25, 0.0)
    repair_loops = _env_int("HALLWAY_HOHQMESH_CONTAINMENT_REPAIR_LOOPS", 3, 0)
    base_loop = simplified_loop
    best_loop = _outward_biased_loop(base_loop, edge_length, is_outer=True, bias_ratio_override=bias_ratio)
    best_misses = _coverage_miss_count(raw_loop, best_loop, tolerance)
    best_bias_ratio = bias_ratio
    while best_misses and bias_ratio < max_bias_ratio:
        bias_ratio = min(max_bias_ratio, max(bias_ratio * 1.6, bias_ratio + 0.04))
        candidate = _outward_biased_loop(base_loop, edge_length, is_outer=True, bias_ratio_override=bias_ratio)
        misses = _coverage_miss_count(raw_loop, candidate, tolerance)
        if misses <= best_misses:
            best_loop = candidate
            best_misses = misses
            best_bias_ratio = bias_ratio
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
        for candidate_bias in (best_bias_ratio, min(max_bias_ratio, best_bias_ratio * 1.25), max_bias_ratio):
            candidate = _outward_biased_loop(base_loop, edge_length, is_outer=True, bias_ratio_override=candidate_bias)
            misses = _coverage_miss_count(raw_loop, candidate, tolerance)
            if misses <= best_misses:
                best_loop = candidate
                best_misses = misses
                best_bias_ratio = candidate_bias
            if misses == 0:
                return best_loop, best_misses
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
        biased = _densify_loop_min_segments(
            biased,
            _env_int("HALLWAY_HOHQMESH_MIN_BOUNDARY_SEGMENTS", 8, 3),
        )
        if len(loop) >= _env_int("HALLWAY_HOHQMESH_COMPLEX_LOOP_VERTEX_THRESHOLD", 1500, 1):
            biased = _remove_pathological_reversals(
                biased,
                turn_threshold=_env_float("HALLWAY_HOHQMESH_REVERSAL_TURN_DEGREES", 150.0, 90.0),
            )
        if len(biased) < 3 or abs(_loop_area(biased)) <= _EPS:
            raise StripRemeshUnsupported("simplified HOHQMesh boundary became degenerate")
        if _loop_has_self_intersections(biased):
            raise StripRemeshUnsupported("simplified HOHQMesh boundary self-intersects")
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
        timeout = _env_float("HALLWAY_HOHQMESH_COMPLEX_TIMEOUT", max(timeout, 45.0), 1.0)
    work_loops = loops
    if profile.is_complex and len(loops) > 1 and _env_bool("HALLWAY_HOHQMESH_COMPLEX_OUTER_BOUNDARY_ONLY", True):
        work_loops = [loops[0]]

    if profile.is_complex:
        if len(loops) > 1:
            attempts: list[tuple[float | None, float | None, int | None, float | None]] = [
                (145.0, 1.35, 48, 0.0),
                (120.0, 1.10, 64, 0.0),
                (95.0, 0.85, 80, 0.0),
            ]
        else:
            attempts = [
                (100.0, 0.80, 112, 0.0),
                (120.0, 1.10, 64, 0.0),
                (145.0, 1.35, 48, 0.0),
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
        attempt_edge_length = edge_length
        attempt_timeout = timeout
        if profile.is_complex and max_segments is not None:
            attempt_timeout = min(timeout, max(18.0, 20.0 + (max_segments * 0.11)))
        try:
            prepared_loops = _prepare_group_loops_for_hohqmesh(
                work_loops,
                attempt_edge_length,
                angle_threshold=angle,
                tolerance_ratio=tolerance_ratio,
                max_segments=max_segments,
                outward_bias_ratio=bias_ratio,
                contain_original=profile.require_containment and not profile.is_complex,
            )
        except StripRemeshUnsupported as exc:
            last_error = HOHQMeshError(str(exc))
            logger.warning(
                "HOHQMesh attempt %s skipped for %s:%s: %s",
                attempt_index + 1,
                source_obj.name,
                group_index,
                last_error,
            )
            continue
        if profile.require_containment and not profile.is_complex and prepared_loops:
            containment_tolerance = max(edge_length * 0.04, 0.0008)
            outer_loop = _orient_ccw(work_loops[0])
            containment_misses = _coverage_miss_count(outer_loop, prepared_loops[0], containment_tolerance)
            max_misses = _containment_miss_limit(outer_loop)
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
        control_path.write_text(_control_text(mesh_filename, attempt_edge_length, prepared_loops), encoding="utf-8")
        started = time.monotonic()
        try:
            result = _run_hohqmesh(job_dir, control_path, attempt_timeout)
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            detail = (exc.stderr or exc.stdout or "").strip()
            last_error = HOHQMeshError(f"HOHQMesh timed out after {attempt_timeout:.1f}s: {detail[-2000:]}")
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


def _group_net_area(group: list[list[tuple[float, float]]]) -> float:
    if not group:
        return 0.0
    outer_area = abs(_loop_area(group[0]))
    hole_area = sum(abs(_loop_area(loop)) for loop in group[1:])
    return max(0.0, outer_area - hole_area)


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
    if profile.is_complex:
        min_group_area = (edge_length * edge_length) * _env_float("HALLWAY_HOHQMESH_MIN_GROUP_AREA_CELLS", 0.5, 0.0)
        kept_groups = [group for group in groups if _group_net_area(group) >= min_group_area]
        skipped_groups = len(groups) - len(kept_groups)
        if skipped_groups:
            logger.info(
                "HOHQMesh skipped %s sub-cell independent groups on %s (min_area=%.8f)",
                skipped_groups,
                source_obj.name,
                min_group_area,
            )
        groups = kept_groups
        if not groups:
            raise StripRemeshUnsupported("all independent contour groups are below HOHQMesh target cell area")
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
