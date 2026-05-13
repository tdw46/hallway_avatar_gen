from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import bmesh
import bpy
import mathutils
from mathutils import Matrix

from ..utils.logging import get_logger

logger = get_logger("strip_remesh")

_EPS = 1e-8
_DEFAULT_MAX_STRIPS = 96
_DEFAULT_MAX_FACES = 150000
_DEFAULT_MAX_CONTOUR_SEGMENTS = 4096
_DEFAULT_MAX_CONTOUR_LAYERS = 96
_DEFAULT_SMOOTHING_ITERATIONS = 100
_DEFAULT_SMOOTHING_FACTOR = 0.35


class StripRemeshUnsupported(RuntimeError):
    pass


@dataclass(frozen=True)
class _Plane:
    normal_axis: int
    across_axis: int
    along_axis: int
    normal_value: float


def _target_edge_length(scale_factor: float) -> float:
    env_value = os.environ.get("HALLWAY_STRIP_REMESH_EDGE_LENGTH", "").strip()
    if env_value:
        try:
            return max(0.001, float(env_value))
        except ValueError:
            logger.warning("Ignoring invalid HALLWAY_STRIP_REMESH_EDGE_LENGTH=%r", env_value)
    return max(0.008, min(0.12, float(scale_factor) * 0.004))


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


def _source_rotation_scale_matrix(source_obj: bpy.types.Object) -> Matrix:
    if source_obj.rotation_mode == "QUATERNION":
        return Matrix.LocRotScale(None, source_obj.rotation_quaternion, source_obj.scale)
    return Matrix.LocRotScale(None, source_obj.rotation_euler, source_obj.scale)


def _mesh_plane(coords: list[mathutils.Vector]) -> _Plane:
    if len(coords) < 3:
        raise StripRemeshUnsupported("not enough vertices")
    mins = [min(coord[axis] for coord in coords) for axis in range(3)]
    maxs = [max(coord[axis] for coord in coords) for axis in range(3)]
    spans = [maxs[axis] - mins[axis] for axis in range(3)]
    normal_axis = min(range(3), key=lambda axis: spans[axis])
    planar_span = max(spans[axis] for axis in range(3) if axis != normal_axis)
    if planar_span <= _EPS:
        raise StripRemeshUnsupported("degenerate flat mesh")
    if spans[normal_axis] > max(1e-5, planar_span * 1e-4):
        raise StripRemeshUnsupported("mesh is not planar enough for strip remesh")
    axes = [axis for axis in range(3) if axis != normal_axis]
    if spans[axes[0]] >= spans[axes[1]]:
        across_axis, along_axis = axes[0], axes[1]
    else:
        across_axis, along_axis = axes[1], axes[0]
    normal_value = sum(coord[normal_axis] for coord in coords) / len(coords)
    return _Plane(normal_axis, across_axis, along_axis, normal_value)


def _boundary_loops(mesh: bpy.types.Mesh) -> list[list[int]]:
    edge_counts: dict[tuple[int, int], int] = {}
    for polygon in mesh.polygons:
        verts = list(polygon.vertices)
        for index, a in enumerate(verts):
            b = verts[(index + 1) % len(verts)]
            edge = (a, b) if a < b else (b, a)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    adjacency: dict[int, list[int]] = {}
    for (a, b), count in edge_counts.items():
        if count != 1:
            continue
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    if not adjacency:
        raise StripRemeshUnsupported("mesh has no boundary")

    loops: list[list[int]] = []
    visited: set[tuple[int, int]] = set()
    for start in sorted(adjacency):
        for second in sorted(adjacency[start]):
            key = (start, second) if start < second else (second, start)
            if key in visited:
                continue
            loop = [start]
            prev = start
            current = second
            while True:
                edge_key = (prev, current) if prev < current else (current, prev)
                if edge_key in visited:
                    break
                visited.add(edge_key)
                loop.append(current)
                neighbors = sorted(adjacency.get(current, []))
                next_candidates = [vertex for vertex in neighbors if vertex != prev]
                if not next_candidates:
                    break
                nxt = next_candidates[0]
                if nxt == start:
                    visited.add((current, nxt) if current < nxt else (nxt, current))
                    break
                prev, current = current, nxt
            if len(loop) >= 3:
                loops.append(loop)
    if not loops:
        raise StripRemeshUnsupported("could not recover boundary loops")
    return loops


def _project_loop(coords: list[mathutils.Vector], loop: list[int], plane: _Plane) -> list[tuple[float, float]]:
    points = [(float(coords[index][plane.across_axis]), float(coords[index][plane.along_axis])) for index in loop]
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not cleaned or (abs(cleaned[-1][0] - point[0]) > _EPS or abs(cleaned[-1][1] - point[1]) > _EPS):
            cleaned.append(point)
    if len(cleaned) > 1 and abs(cleaned[0][0] - cleaned[-1][0]) <= _EPS and abs(cleaned[0][1] - cleaned[-1][1]) <= _EPS:
        cleaned.pop()
    return cleaned


def _point_line_distance(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    px, py = point
    dx = bx - ax
    dy = by - ay
    denom = (dx * dx) + (dy * dy)
    if denom <= _EPS:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (((px - ax) * dx) + ((py - ay) * dy)) / denom))
    qx = ax + (dx * t)
    qy = ay + (dy * t)
    return math.hypot(px - qx, py - qy)


def _segments(loops: list[list[tuple[float, float]]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    result = []
    for loop in loops:
        for index, point in enumerate(loop):
            result.append((point, loop[(index + 1) % len(loop)]))
    return result


def _intervals_at(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    across: float,
    side: str,
) -> list[tuple[float, float]]:
    hits: list[float] = []
    for (a0, b0), (a1, b1) in segments:
        da = a1 - a0
        if abs(da) <= _EPS:
            if abs(across - a0) <= _EPS:
                hits.extend([b0, b1])
            continue
        lo = min(a0, a1)
        hi = max(a0, a1)
        if side == "left":
            inside = (lo - _EPS) <= across < (hi - _EPS)
        else:
            inside = (lo + _EPS) < across <= (hi + _EPS)
        if not inside:
            continue
        t = (across - a0) / da
        if -_EPS <= t <= 1.0 + _EPS:
            hits.append(b0 + ((b1 - b0) * t))
    hits.sort()
    unique: list[float] = []
    for value in hits:
        if not unique or abs(unique[-1] - value) > 1e-6:
            unique.append(value)
    if len(unique) % 2 == 1:
        unique = unique[:-1]
    return [(unique[index], unique[index + 1]) for index in range(0, len(unique), 2) if unique[index + 1] - unique[index] > 1e-6]


def _inside_intervals_at(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    across: float,
) -> list[tuple[float, float]]:
    return _intervals_at(segments, across, "left")


def _budgeted_grid_counts(width: float, height: float, target_step: float, max_faces: int) -> tuple[int, int]:
    aspect = max(width, target_step) / max(height, target_step)
    ideal_across = max(1, int(math.ceil(width / target_step)))
    ideal_along = max(1, int(math.ceil(height / target_step)))
    max_strips = _env_int("HALLWAY_STRIP_REMESH_MAX_STRIPS", _DEFAULT_MAX_STRIPS, 4)
    across = min(max_strips, ideal_across)
    along = ideal_along
    if across * along > max_faces:
        across = max(1, min(max_strips, int(math.floor(math.sqrt(max_faces * aspect)))))
        along = max(1, int(math.floor(max_faces / max(1, across))))
    return across, along


def _make_coord(plane: _Plane, across: float, along: float) -> tuple[float, float, float]:
    coord = [0.0, 0.0, 0.0]
    coord[plane.normal_axis] = plane.normal_value
    coord[plane.across_axis] = across
    coord[plane.along_axis] = along
    return (coord[0], coord[1], coord[2])


def _loop_area(loop: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x0, y0) in enumerate(loop):
        x1, y1 = loop[(index + 1) % len(loop)]
        area += (x0 * y1) - (x1 * y0)
    return area * 0.5


def _loop_centroid(loop: list[tuple[float, float]]) -> tuple[float, float]:
    area = _loop_area(loop)
    if abs(area) <= _EPS:
        return (
            sum(point[0] for point in loop) / len(loop),
            sum(point[1] for point in loop) / len(loop),
        )
    cx = 0.0
    cy = 0.0
    for index, (x0, y0) in enumerate(loop):
        x1, y1 = loop[(index + 1) % len(loop)]
        cross = (x0 * y1) - (x1 * y0)
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    factor = 1.0 / (6.0 * area)
    return (cx * factor, cy * factor)


def _loop_perimeter(loop: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(loop[(index + 1) % len(loop)][0] - point[0], loop[(index + 1) % len(loop)][1] - point[1])
        for index, point in enumerate(loop)
    )


def _loop_max_edge(loop: list[tuple[float, float]]) -> float:
    return max(
        math.hypot(loop[(index + 1) % len(loop)][0] - point[0], loop[(index + 1) % len(loop)][1] - point[1])
        for index, point in enumerate(loop)
    )


def _point_in_loop(point: tuple[float, float], loop: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    previous = loop[-1]
    for current in loop:
        x0, y0 = previous
        x1, y1 = current
        dy = y1 - y0
        if ((y0 > y) != (y1 > y)) and abs(dy) > _EPS and x < ((x1 - x0) * (y - y0) / dy) + x0:
            inside = not inside
        previous = current
    return inside


def _point_in_domain(point: tuple[float, float], loops: list[list[tuple[float, float]]]) -> bool:
    if not loops or not _point_in_loop(point, loops[0]):
        return False
    return not any(_point_in_loop(point, hole) for hole in loops[1:])


def _point_near_domain_boundary(point: tuple[float, float], loops: list[list[tuple[float, float]]], tolerance: float) -> bool:
    for loop in loops:
        for index, start in enumerate(loop):
            if _point_line_distance(point, start, loop[(index + 1) % len(loop)]) <= tolerance:
                return True
    return False


def _nearest_point_on_segment(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    ax, ay = a
    bx, by = b
    px, py = point
    dx = bx - ax
    dy = by - ay
    denom = (dx * dx) + (dy * dy)
    if denom <= _EPS:
        return a, math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (((px - ax) * dx) + ((py - ay) * dy)) / denom))
    projected = (ax + (dx * t), ay + (dy * t))
    return projected, math.hypot(px - projected[0], py - projected[1])


def _nearest_point_on_domain_boundary(
    point: tuple[float, float],
    loops: list[list[tuple[float, float]]],
) -> tuple[float, float]:
    best_point = point
    best_distance = float("inf")
    for loop in loops:
        for index, start in enumerate(loop):
            projected, distance = _nearest_point_on_segment(point, start, loop[(index + 1) % len(loop)])
            if distance < best_distance:
                best_point = projected
                best_distance = distance
    return best_point


def _clamp_point_to_domain(
    point: tuple[float, float],
    domain_loops: list[list[tuple[float, float]]],
    reference: tuple[float, float],
    inset: float,
) -> tuple[float, float]:
    if _point_in_domain(point, domain_loops) or _point_near_domain_boundary(point, domain_loops, inset):
        return point
    nearest = _nearest_point_on_domain_boundary(point, domain_loops)
    dx = reference[0] - nearest[0]
    dy = reference[1] - nearest[1]
    length = math.hypot(dx, dy)
    if length <= _EPS:
        return nearest
    step = max(inset, 1e-6)
    candidate = (nearest[0] + (dx / length) * step, nearest[1] + (dy / length) * step)
    for _ in range(8):
        if _point_in_domain(candidate, domain_loops) or _point_near_domain_boundary(candidate, domain_loops, step):
            return candidate
        step *= 0.5
        candidate = (nearest[0] + (dx / length) * step, nearest[1] + (dy / length) * step)
    return nearest


def _clamp_grid_to_domain(
    grid: list[list[tuple[float, float]]],
    domain_loops: list[list[tuple[float, float]]],
    inset: float,
) -> int:
    if len(grid) < 3 or len(grid[0]) < 3:
        return 0
    work = (len(grid) - 2) * (len(grid[0]) - 2) * max(1, sum(len(loop) for loop in domain_loops))
    max_work = _env_int("HALLWAY_QUADRANT_REMESH_MAX_DOMAIN_CLAMP_WORK", 2500000, 10000)
    if work > max_work and not _env_bool("HALLWAY_QUADRANT_REMESH_FORCE_DOMAIN_CLAMP", False):
        logger.debug("Skipping domain clamp for large grid: work=%s max=%s", work, max_work)
        return 0
    reference = _loop_centroid(domain_loops[0])
    changed = 0
    for row in range(1, len(grid) - 1):
        for column in range(1, len(grid[0]) - 1):
            point = grid[row][column]
            clamped = _clamp_point_to_domain(point, domain_loops, reference, inset)
            if _point_distance_2d(point, clamped) > _EPS:
                grid[row][column] = clamped
                changed += 1
    return changed


def _clean_loop(loop: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for point in loop:
        if not cleaned or _point_distance_2d(point, cleaned[-1]) > 1e-7:
            cleaned.append(point)
    if len(cleaned) > 1 and _point_distance_2d(cleaned[0], cleaned[-1]) <= 1e-7:
        cleaned.pop()
    return cleaned


def _normalize_domain_loops(loops: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    if not loops:
        return []
    indexed = [(abs(_loop_area(loop)), loop) for loop in loops]
    indexed.sort(key=lambda item: item[0], reverse=True)
    normalized: list[list[tuple[float, float]]] = []
    for index, (_, loop) in enumerate(indexed):
        area = _loop_area(loop)
        if index == 0:
            normalized.append(loop if area >= 0.0 else list(reversed(loop)))
        else:
            normalized.append(loop if area <= 0.0 else list(reversed(loop)))
    return normalized


def _independent_loop_groups(loops: list[list[tuple[float, float]]]) -> list[list[list[tuple[float, float]]]]:
    if len(loops) <= 1:
        return [loops]
    if _env_bool("HALLWAY_TREAT_ALL_LOOPS_AS_COMPONENTS", False):
        return [[loop] for loop in loops]
    areas = [abs(_loop_area(loop)) for loop in loops]
    outer_indices: list[int] = []
    for index, loop in enumerate(loops):
        point = loop[0]
        parent_indices = [
            other_index
            for other_index, other_loop in enumerate(loops)
            if other_index != index and areas[other_index] > areas[index] and _point_in_loop(point, other_loop)
        ]
        if not parent_indices:
            outer_indices.append(index)
    if len(outer_indices) <= 1:
        return [loops]

    groups: list[list[list[tuple[float, float]]]] = []
    assigned: set[int] = set()
    for outer_index in sorted(outer_indices, key=lambda item: areas[item], reverse=True):
        group = [loops[outer_index]]
        assigned.add(outer_index)
        for index, loop in enumerate(loops):
            if index in assigned or index in outer_indices:
                continue
            if _point_in_loop(loop[0], loops[outer_index]):
                group.append(loop)
                assigned.add(index)
        groups.append(group)
    for index, loop in enumerate(loops):
        if index not in assigned:
            groups.append([loop])
    return groups


def _combine_meshes(name: str, meshes: list[bpy.types.Mesh]) -> bpy.types.Mesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    weld_tolerance = _env_float("HALLWAY_DECOMPOSED_PATCH_WELD_TOLERANCE", 0.0, 0.0)
    vertex_index: dict[tuple[int, int, int], list[int]] = {}
    exact_vertex_index: dict[tuple[int, int, int], int] = {}

    def add_vertex(coord: tuple[float, float, float]) -> int:
        if weld_tolerance <= _EPS:
            if _env_bool("HALLWAY_DECOMPOSED_PATCH_EXACT_WELD", True):
                key = (round(coord[0] / 1e-9), round(coord[1] / 1e-9), round(coord[2] / 1e-9))
                existing = exact_vertex_index.get(key)
                if existing is not None:
                    return existing
                exact_vertex_index[key] = len(vertices)
            vertices.append(coord)
            return len(vertices) - 1
        key = (
            round(coord[0] / weld_tolerance),
            round(coord[1] / weld_tolerance),
            round(coord[2] / weld_tolerance),
        )
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for existing in vertex_index.get((key[0] + dx, key[1] + dy, key[2] + dz), []):
                        existing_coord = vertices[existing]
                        if math.dist(coord, existing_coord) <= weld_tolerance:
                            return existing
        vertex_index.setdefault(key, []).append(len(vertices))
        vertices.append(coord)
        return len(vertices) - 1

    for mesh in meshes:
        index_map: dict[int, int] = {}
        for vertex in mesh.vertices:
            coord = tuple(vertex.co)
            index_map[vertex.index] = add_vertex(coord)
        faces.extend(tuple(index_map[index] for index in polygon.vertices) for polygon in mesh.polygons)
    vertices, faces, removed_faces = _sanitize_pydata_faces(vertices, faces)
    combined = bpy.data.meshes.new(name)
    combined.from_pydata(vertices, [], faces)
    combined.update(calc_edges=True)
    if weld_tolerance > _EPS and _env_bool("HALLWAY_DECOMPOSED_PATCH_STITCH_TJUNCTIONS", False):
        _stitch_boundary_t_junction_quads(combined, weld_tolerance * 1.5)
    if removed_faces:
        logger.info("Removed %s degenerate/duplicate faces while combining %s", removed_faces, name)
    for mesh in meshes:
        bpy.data.meshes.remove(mesh)
    return combined


def _resample_closed_loop(loop: list[tuple[float, float]], segment_count: int) -> list[tuple[float, float]]:
    perimeter = _loop_perimeter(loop)
    if perimeter <= _EPS or segment_count < 4:
        raise StripRemeshUnsupported("contour loop is degenerate")
    samples: list[tuple[float, float]] = []
    edges: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    for index, point in enumerate(loop):
        nxt = loop[(index + 1) % len(loop)]
        length = math.hypot(nxt[0] - point[0], nxt[1] - point[1])
        if length > _EPS:
            edges.append((point, nxt, length))
    edge_index = 0
    distance_before_edge = 0.0
    for sample_index in range(segment_count):
        target = perimeter * sample_index / segment_count
        while edge_index < len(edges) - 1 and distance_before_edge + edges[edge_index][2] < target:
            distance_before_edge += edges[edge_index][2]
            edge_index += 1
        start, end, length = edges[edge_index]
        t = 0.0 if length <= _EPS else (target - distance_before_edge) / length
        samples.append((start[0] + ((end[0] - start[0]) * t), start[1] + ((end[1] - start[1]) * t)))
    return samples


def _polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(points[index + 1][0] - point[0], points[index + 1][1] - point[1]) for index, point in enumerate(points[:-1]))


def _resample_open_polyline(points: list[tuple[float, float]], segment_count: int) -> list[tuple[float, float]]:
    if len(points) < 2:
        raise StripRemeshUnsupported("not enough points for boundary segment")
    segment_count = max(1, segment_count)
    total_length = _polyline_length(points)
    if total_length <= _EPS:
        return [points[0] for _ in range(segment_count + 1)]
    result: list[tuple[float, float]] = []
    edge_index = 0
    distance_before_edge = 0.0
    for sample_index in range(segment_count + 1):
        target = total_length * sample_index / segment_count
        while edge_index < len(points) - 2:
            edge_length = math.hypot(points[edge_index + 1][0] - points[edge_index][0], points[edge_index + 1][1] - points[edge_index][1])
            if distance_before_edge + edge_length >= target:
                break
            distance_before_edge += edge_length
            edge_index += 1
        start = points[edge_index]
        end = points[edge_index + 1]
        edge_length = math.hypot(end[0] - start[0], end[1] - start[1])
        t = 0.0 if edge_length <= _EPS else (target - distance_before_edge) / edge_length
        result.append((start[0] + ((end[0] - start[0]) * t), start[1] + ((end[1] - start[1]) * t)))
    return result


def _split_long_edges_closed_loop(
    loop: list[tuple[float, float]],
    edge_length: float,
    max_segments: int,
) -> list[tuple[float, float]]:
    if not _env_bool("HALLWAY_CONTOUR_REMESH_PRESERVE_RAW_VERTICES", False):
        perimeter = _loop_perimeter(loop)
        target_segments = max(8, int(math.ceil(perimeter / max(edge_length, _EPS))))
        target_segments = min(max_segments, target_segments)
        if target_segments % 2 == 1:
            target_segments += 1
        return _resample_closed_loop(loop, min(max_segments, target_segments))

    target_segments = 0
    edge_steps: list[int] = []
    for index, point in enumerate(loop):
        nxt = loop[(index + 1) % len(loop)]
        length = math.hypot(nxt[0] - point[0], nxt[1] - point[1])
        steps = max(1, int(math.ceil(length / max(edge_length, _EPS))))
        edge_steps.append(steps)
        target_segments += steps
    if target_segments > max_segments:
        return _resample_closed_loop(loop, max_segments)
    result: list[tuple[float, float]] = []
    for index, point in enumerate(loop):
        nxt = loop[(index + 1) % len(loop)]
        steps = edge_steps[index]
        for step in range(steps):
            t = step / steps
            result.append((point[0] + ((nxt[0] - point[0]) * t), point[1] + ((nxt[1] - point[1]) * t)))
    return result


def _loop_inward_normals(loop: list[tuple[float, float]]) -> list[tuple[float, float]]:
    normals: list[tuple[float, float]] = []
    count = len(loop)
    for index, point in enumerate(loop):
        prev_point = loop[(index - 1) % count]
        next_point = loop[(index + 1) % count]
        prev_dx = point[0] - prev_point[0]
        prev_dy = point[1] - prev_point[1]
        next_dx = next_point[0] - point[0]
        next_dy = next_point[1] - point[1]
        prev_len = math.hypot(prev_dx, prev_dy)
        next_len = math.hypot(next_dx, next_dy)
        nx = 0.0
        ny = 0.0
        if prev_len > _EPS:
            nx += -prev_dy / prev_len
            ny += prev_dx / prev_len
        if next_len > _EPS:
            nx += -next_dy / next_len
            ny += next_dx / next_len
        length = math.hypot(nx, ny)
        if length <= _EPS:
            if next_len > _EPS:
                nx = -next_dy / next_len
                ny = next_dx / next_len
                length = 1.0
            else:
                nx, ny, length = 0.0, 0.0, 1.0
        normals.append((nx / length, ny / length))
    return normals


def _offset_loop(loop: list[tuple[float, float]], distance: float) -> list[tuple[float, float]]:
    shifted_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for index, point in enumerate(loop):
        nxt = loop[(index + 1) % len(loop)]
        dx = nxt[0] - point[0]
        dy = nxt[1] - point[1]
        length = math.hypot(dx, dy)
        if length <= _EPS:
            shifted_edges.append((point, nxt))
            continue
        nx = -dy / length
        ny = dx / length
        shifted_edges.append(((point[0] + nx * distance, point[1] + ny * distance), (nxt[0] + nx * distance, nxt[1] + ny * distance)))

    result: list[tuple[float, float]] = []
    for index in range(len(loop)):
        prev_start, prev_end = shifted_edges[(index - 1) % len(loop)]
        curr_start, curr_end = shifted_edges[index]
        pdx = prev_end[0] - prev_start[0]
        pdy = prev_end[1] - prev_start[1]
        cdx = curr_end[0] - curr_start[0]
        cdy = curr_end[1] - curr_start[1]
        denom = (pdx * cdy) - (pdy * cdx)
        if abs(denom) <= _EPS:
            result.append(((prev_end[0] + curr_start[0]) * 0.5, (prev_end[1] + curr_start[1]) * 0.5))
            continue
        t = (((curr_start[0] - prev_start[0]) * cdy) - ((curr_start[1] - prev_start[1]) * cdx)) / denom
        result.append((prev_start[0] + pdx * t, prev_start[1] + pdy * t))
    return result


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return ((b[0] - a[0]) * (c[1] - a[1])) - ((b[1] - a[1]) * (c[0] - a[0]))


def _segments_intersect(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> bool:
    o1 = _orientation(a0, a1, b0)
    o2 = _orientation(a0, a1, b1)
    o3 = _orientation(b0, b1, a0)
    o4 = _orientation(b0, b1, a1)
    return (o1 * o2 < -1e-10) and (o3 * o4 < -1e-10)


def _loop_has_self_intersections(loop: list[tuple[float, float]]) -> bool:
    count = len(loop)
    for index in range(count):
        a0 = loop[index]
        a1 = loop[(index + 1) % count]
        for other in range(index + 2, count):
            if other == index or (other + 1) % count == index:
                continue
            if index == 0 and other == count - 1:
                continue
            if _segments_intersect(a0, a1, loop[other], loop[(other + 1) % count]):
                return True
    return False


def _segment_visible_inside_loop(
    loop: list[tuple[float, float]],
    start_index: int,
    target_edge_index: int,
    target: tuple[float, float],
) -> bool:
    start = loop[start_index]
    if _point_distance_2d(start, target) <= _EPS:
        return False
    midpoint = ((start[0] + target[0]) * 0.5, (start[1] + target[1]) * 0.5)
    if not _point_in_loop(midpoint, loop):
        return False
    for edge_index, point in enumerate(loop):
        next_index = (edge_index + 1) % len(loop)
        if edge_index in {start_index, (start_index - 1) % len(loop), target_edge_index}:
            continue
        if next_index == start_index:
            continue
        if _segments_intersect(start, target, point, loop[next_index]):
            return False
    return True


def _split_loop_at_vertex_to_edge(
    loop: list[tuple[float, float]],
    start_index: int,
    target_edge_index: int,
    target: tuple[float, float],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    count = len(loop)
    if target_edge_index == start_index or (target_edge_index + 1) % count == start_index:
        return None
    first = [loop[start_index]]
    index = (start_index + 1) % count
    while True:
        first.append(loop[index])
        if index == target_edge_index:
            break
        index = (index + 1) % count
    first.append(target)

    second = [target]
    index = (target_edge_index + 1) % count
    while True:
        second.append(loop[index])
        if index == start_index:
            break
        index = (index + 1) % count

    first = _clean_loop(first)
    second = _clean_loop(second)
    if len(first) < 4 or len(second) < 4:
        return None
    if abs(_loop_area(first)) <= _EPS or abs(_loop_area(second)) <= _EPS:
        return None
    if _loop_area(first) < 0.0:
        first = list(reversed(first))
    if _loop_area(second) < 0.0:
        second = list(reversed(second))
    return first, second


def _split_loop_between_vertices(
    loop: list[tuple[float, float]],
    start_index: int,
    end_index: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    count = len(loop)
    if start_index == end_index:
        return None
    if (start_index + 1) % count == end_index or (end_index + 1) % count == start_index:
        return None
    first = _clean_loop(_loop_chain(loop, start_index, end_index))
    second = _clean_loop(_loop_chain(loop, end_index, start_index))
    if len(first) < 4 or len(second) < 4:
        return None
    if abs(_loop_area(first)) <= _EPS or abs(_loop_area(second)) <= _EPS:
        return None
    if _loop_area(first) < 0.0:
        first = list(reversed(first))
    if _loop_area(second) < 0.0:
        second = list(reversed(second))
    return first, second


def _split_loop_between_edge_points(
    loop: list[tuple[float, float]],
    first_edge: int,
    first_point: tuple[float, float],
    second_edge: int,
    second_point: tuple[float, float],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    count = len(loop)
    if first_edge == second_edge:
        return None

    first = [first_point]
    index = (first_edge + 1) % count
    while True:
        first.append(loop[index])
        if index == second_edge:
            break
        index = (index + 1) % count
    first.append(second_point)

    second = [second_point]
    index = (second_edge + 1) % count
    while True:
        second.append(loop[index])
        if index == first_edge:
            break
        index = (index + 1) % count
    second.append(first_point)

    first = _clean_loop(first)
    second = _clean_loop(second)
    if len(first) < 4 or len(second) < 4:
        return None
    if abs(_loop_area(first)) <= _EPS or abs(_loop_area(second)) <= _EPS:
        return None
    if _loop_area(first) < 0.0:
        first = list(reversed(first))
    if _loop_area(second) < 0.0:
        second = list(reversed(second))
    return first, second


def _loop_axis_spans(loop: list[tuple[float, float]]) -> tuple[float, float]:
    center = _loop_centroid(loop)
    major, minor = _principal_axes(loop)
    major_values = [((point[0] - center[0]) * major[0]) + ((point[1] - center[1]) * major[1]) for point in loop]
    minor_values = [((point[0] - center[0]) * minor[0]) + ((point[1] - center[1]) * minor[1]) for point in loop]
    return max(major_values) - min(major_values), max(minor_values) - min(minor_values)


def _split_elongated_loop(loop: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    if len(loop) < 8:
        return None
    center = _loop_centroid(loop)
    major, _minor = _principal_axes(loop)
    projections = [((point[0] - center[0]) * major[0]) + ((point[1] - center[1]) * major[1]) for point in loop]
    major_span = max(projections) - min(projections)
    if major_span <= _EPS:
        return None
    cuts: list[tuple[int, tuple[float, float], float]] = []
    for index, value in enumerate(projections):
        next_index = (index + 1) % len(loop)
        next_value = projections[next_index]
        if abs(value) <= major_span * 1e-6:
            cuts.append((index, loop[index], value))
            continue
        if value * next_value >= 0.0:
            continue
        t = value / (value - next_value)
        start = loop[index]
        end = loop[next_index]
        point = (start[0] + ((end[0] - start[0]) * t), start[1] + ((end[1] - start[1]) * t))
        cuts.append((index, point, 0.0))
    if len(cuts) < 2:
        return None

    best: tuple[float, tuple[list[tuple[float, float]], list[tuple[float, float]]]] | None = None
    area = abs(_loop_area(loop))
    for first_index, first in enumerate(cuts):
        for second in cuts[first_index + 1 :]:
            split = _split_loop_between_edge_points(loop, first[0], first[1], second[0], second[1])
            if split is None:
                continue
            first_area = abs(_loop_area(split[0]))
            second_area = abs(_loop_area(split[1]))
            if first_area < area * 0.15 or second_area < area * 0.15:
                continue
            balance = abs(first_area - second_area) / max(area, _EPS)
            if best is None or balance < best[0]:
                best = (balance, split)
    return best[1] if best is not None else None


def _reflex_vertex_indices(loop: list[tuple[float, float]]) -> list[int]:
    if _loop_area(loop) < 0.0:
        loop = list(reversed(loop))
    result: list[int] = []
    for index, point in enumerate(loop):
        prev_point = loop[(index - 1) % len(loop)]
        next_point = loop[(index + 1) % len(loop)]
        if _orientation(prev_point, point, next_point) < -1e-8:
            result.append(index)
    return result


def _reflex_vertex_candidates(loop: list[tuple[float, float]]) -> list[int]:
    if _loop_area(loop) < 0.0:
        loop = list(reversed(loop))
    scored: list[tuple[float, int]] = []
    for index, point in enumerate(loop):
        prev_point = loop[(index - 1) % len(loop)]
        next_point = loop[(index + 1) % len(loop)]
        ax = point[0] - prev_point[0]
        ay = point[1] - prev_point[1]
        bx = next_point[0] - point[0]
        by = next_point[1] - point[1]
        if math.hypot(ax, ay) <= _EPS or math.hypot(bx, by) <= _EPS:
            continue
        turn_cross = (ax * by) - (ay * bx)
        if turn_cross >= -1e-8:
            continue
        turn = abs(math.atan2(turn_cross, (ax * bx) + (ay * by)))
        scored.append((turn, index))
    if not scored:
        return []
    max_candidates = _env_int("HALLWAY_QUADRANT_REMESH_MAX_REFLEX_CANDIDATES", 16, 1)
    scored.sort(reverse=True)
    selected: list[int] = []
    min_gap = max(1, len(loop) // max(128, max_candidates * 4))
    for _score, index in scored:
        if any(min((index - existing) % len(loop), (existing - index) % len(loop)) < min_gap for existing in selected):
            continue
        selected.append(index)
        if len(selected) >= max_candidates:
            break
    return selected


def _sampled_loop_indices(loop: list[tuple[float, float]], max_count: int) -> list[int]:
    if len(loop) <= max_count:
        return list(range(len(loop)))
    step = len(loop) / max_count
    return sorted({min(len(loop) - 1, int(round(index * step))) for index in range(max_count)})


def _split_target_vertex_candidates(loop: list[tuple[float, float]], reflex_candidates: list[int]) -> list[int]:
    max_targets = _env_int("HALLWAY_QUADRANT_REMESH_MAX_SPLIT_TARGETS", 48, 8)
    indices = set(_sampled_loop_indices(loop, max_targets))
    indices.update(_quadrant_anchor_indices(loop))
    indices.update(reflex_candidates)
    if len(indices) > max_targets * 2:
        center = _loop_centroid(loop)
        ranked = sorted(
            indices,
            key=lambda index: _point_distance_2d(loop[index], center),
            reverse=True,
        )
        indices = set(ranked[: max_targets * 2])
    return sorted(indices)


def _split_target_edge_candidates(loop: list[tuple[float, float]]) -> list[int]:
    max_edges = _env_int("HALLWAY_QUADRANT_REMESH_MAX_SPLIT_EDGES", 48, 8)
    return _sampled_loop_indices(loop, max_edges)


def _split_once_at_reflex(loop: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    if _loop_area(loop) < 0.0:
        loop = list(reversed(loop))
    best: tuple[float, list[tuple[float, float]], list[tuple[float, float]]] | None = None
    area = abs(_loop_area(loop))
    reflex_candidates = _reflex_vertex_candidates(loop)
    if not reflex_candidates:
        return None
    vertex_candidates = _split_target_vertex_candidates(loop, reflex_candidates)
    edge_candidates = _split_target_edge_candidates(loop)
    for start_index in reflex_candidates:
        start = loop[start_index]
        for end_index in vertex_candidates:
            target = loop[end_index]
            if end_index == start_index:
                continue
            if (end_index + 1) % len(loop) == start_index or (start_index + 1) % len(loop) == end_index:
                continue
            if not _segment_visible_inside_loop(loop, start_index, -1, target):
                continue
            split = _split_loop_between_vertices(loop, start_index, end_index)
            if split is None:
                continue
            first, second = split
            first_area = abs(_loop_area(first))
            second_area = abs(_loop_area(second))
            if first_area < area * 0.05 or second_area < area * 0.05:
                continue
            length = _point_distance_2d(start, target)
            balance_penalty = abs(first_area - second_area) / max(area, _EPS)
            score = length * (1.0 + balance_penalty)
            if best is None or score < best[0]:
                best = (score, first, second)
        for edge_index in edge_candidates:
            edge_start = loop[edge_index]
            edge_end = loop[(edge_index + 1) % len(loop)]
            if edge_index in {start_index, (start_index - 1) % len(loop)}:
                continue
            ex = edge_end[0] - edge_start[0]
            ey = edge_end[1] - edge_start[1]
            edge_len_sq = (ex * ex) + (ey * ey)
            if edge_len_sq <= _EPS:
                continue
            t = (((start[0] - edge_start[0]) * ex) + ((start[1] - edge_start[1]) * ey)) / edge_len_sq
            if t <= 0.05 or t >= 0.95:
                continue
            target = (edge_start[0] + (ex * t), edge_start[1] + (ey * t))
            if not _segment_visible_inside_loop(loop, start_index, edge_index, target):
                continue
            split = _split_loop_at_vertex_to_edge(loop, start_index, edge_index, target)
            if split is None:
                continue
            first, second = split
            first_area = abs(_loop_area(first))
            second_area = abs(_loop_area(second))
            if first_area < area * 0.05 or second_area < area * 0.05:
                continue
            length = _point_distance_2d(start, target)
            balance_penalty = abs(first_area - second_area) / max(area, _EPS)
            score = length * (1.0 + balance_penalty)
            if best is None or score < best[0]:
                best = (score, first, second)
    if best is None:
        return None
    return best[1], best[2]


def _decompose_concave_loop(loop: list[tuple[float, float]], max_patches: int = 8) -> list[list[tuple[float, float]]]:
    patches = [loop if _loop_area(loop) >= 0.0 else list(reversed(loop))]
    changed = True
    while changed and len(patches) < max_patches:
        changed = False
        next_patches: list[list[tuple[float, float]]] = []
        for patch_index, patch in enumerate(patches):
            remaining_after_this = len(patches) - patch_index - 1
            if len(next_patches) + remaining_after_this + 2 > max_patches:
                next_patches.append(patch)
                continue
            has_reflex = bool(_reflex_vertex_indices(patch))
            split = _split_once_at_reflex(patch) if has_reflex else None
            if split is None:
                major_span, minor_span = _loop_axis_spans(patch)
                aspect = major_span / max(minor_span, _EPS)
                if aspect > _env_float("HALLWAY_QUADRANT_REMESH_SPLIT_ASPECT", 2.4, 1.0):
                    split = _split_elongated_loop(patch)
            if split is None:
                next_patches.append(patch)
                continue
            next_patches.extend(split)
            changed = True
        patches = next_patches
    return patches


def _valid_offset_loop(loop: list[tuple[float, float]], domain_loops: list[list[tuple[float, float]]]) -> bool:
    if len(loop) < 4 or abs(_loop_area(loop)) <= _EPS:
        return False
    if len(loop) <= 512 and _loop_has_self_intersections(loop):
        return False
    stride = max(1, len(loop) // 96)
    return all(_point_in_domain(loop[index], domain_loops) for index in range(0, len(loop), stride))


def _quad_order(points: list[tuple[float, float]], indices: list[int]) -> tuple[int, int, int, int]:
    cx = sum(points[index][0] for index in indices) / len(indices)
    cy = sum(points[index][1] for index in indices) / len(indices)
    ordered = sorted(indices, key=lambda index: math.atan2(points[index][1] - cy, points[index][0] - cx))
    return (ordered[0], ordered[1], ordered[2], ordered[3])


def _quad_faces_from_tessellation(loops: list[list[tuple[float, float]]]) -> tuple[list[tuple[float, float]], list[tuple[int, ...]]]:
    points = [point for loop in loops for point in loop]
    if len(points) < 3:
        raise StripRemeshUnsupported("not enough points for contour fill")
    polygon = [[mathutils.Vector((point[0], point[1], 0.0)) for point in loop] for loop in loops if len(loop) >= 3]
    triangles = [tuple(face) for face in mathutils.geometry.tessellate_polygon(polygon)]
    if not triangles:
        raise StripRemeshUnsupported("contour fill tessellation produced no faces")

    edge_to_triangles: dict[tuple[int, int], list[int]] = {}
    for tri_index, triangle in enumerate(triangles):
        for index, a in enumerate(triangle):
            b = triangle[(index + 1) % 3]
            edge = (a, b) if a < b else (b, a)
            edge_to_triangles.setdefault(edge, []).append(tri_index)

    def triangle_has_boundary_edge(triangle: tuple[int, int, int]) -> bool:
        for index, a in enumerate(triangle):
            b = triangle[(index + 1) % 3]
            edge = (a, b) if a < b else (b, a)
            if len(edge_to_triangles.get(edge, [])) == 1:
                return True
        return False

    used: set[int] = set()
    faces: list[tuple[int, ...]] = []
    for tri_index, triangle in enumerate(triangles):
        if tri_index in used:
            continue
        if triangle_has_boundary_edge(triangle):
            continue
        partner_index = None
        for index, a in enumerate(triangle):
            b = triangle[(index + 1) % 3]
            edge = (a, b) if a < b else (b, a)
            candidates = [candidate for candidate in edge_to_triangles.get(edge, []) if candidate != tri_index and candidate not in used]
            if not candidates:
                continue
            if triangle_has_boundary_edge(triangles[candidates[0]]):
                continue
            union = list(dict.fromkeys((*triangle, *triangles[candidates[0]])))
            if len(union) == 4:
                partner_index = candidates[0]
                faces.append(_quad_order(points, union))
                break
        if partner_index is not None:
            used.add(tri_index)
            used.add(partner_index)

    for tri_index, triangle in enumerate(triangles):
        if tri_index in used:
            continue
        a, b, c = triangle
        pa, pb, pc = points[a], points[b], points[c]
        ab = len(points)
        points.append(((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5))
        bc = len(points)
        points.append(((pb[0] + pc[0]) * 0.5, (pb[1] + pc[1]) * 0.5))
        ca = len(points)
        points.append(((pc[0] + pa[0]) * 0.5, (pc[1] + pa[1]) * 0.5))
        center = len(points)
        points.append(((pa[0] + pb[0] + pc[0]) / 3.0, (pa[1] + pb[1] + pc[1]) / 3.0))
        faces.extend(((a, ab, center, ca), (ab, b, bc, center), (center, bc, c, ca)))

    return points, faces


def _face_area_3d(vertices: list[tuple[float, float, float]], face: tuple[int, ...]) -> float:
    if len(face) < 3:
        return 0.0
    origin = vertices[face[0]]
    area = mathutils.Vector((0.0, 0.0, 0.0))
    for index in range(1, len(face) - 1):
        a = mathutils.Vector(vertices[face[index]]) - mathutils.Vector(origin)
        b = mathutils.Vector(vertices[face[index + 1]]) - mathutils.Vector(origin)
        area += a.cross(b)
    return area.length * 0.5


def _sanitize_pydata_faces(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, ...]],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]], int]:
    clean_faces: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    removed = 0
    for face in faces:
        if len(face) < 3 or len(set(face)) != len(face):
            removed += 1
            continue
        if any(index < 0 or index >= len(vertices) for index in face):
            removed += 1
            continue
        if _face_area_3d(vertices, face) <= 1e-12:
            removed += 1
            continue
        key = tuple(sorted(face))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        clean_faces.append(face)
    used = sorted({index for face in clean_faces for index in face})
    if len(used) == len(vertices):
        return vertices, clean_faces, removed
    index_map = {old_index: new_index for new_index, old_index in enumerate(used)}
    compact_vertices = [vertices[index] for index in used]
    compact_faces = [tuple(index_map[index] for index in face) for face in clean_faces]
    return compact_vertices, compact_faces, removed + (len(vertices) - len(compact_vertices))


def _angle_resample_loop(
    loop: list[tuple[float, float]],
    center: tuple[float, float],
    count: int,
) -> list[tuple[float, float]]:
    if count < 3:
        raise StripRemeshUnsupported("not enough points for angular bridge")
    angle_points = [(math.atan2(point[1] - center[1], point[0] - center[0]), point) for point in loop]
    angle_points.sort(key=lambda item: item[0])
    angles = [item[0] for item in angle_points]
    points = [item[1] for item in angle_points]
    extended_angles = angles + [angles[0] + (math.pi * 2.0)]
    extended_points = points + [points[0]]
    result: list[tuple[float, float]] = []
    for index in range(count):
        angle = -math.pi + (math.pi * 2.0 * index / count)
        while angle < extended_angles[0]:
            angle += math.pi * 2.0
        segment = 0
        while segment < len(angles) - 1 and extended_angles[segment + 1] < angle:
            segment += 1
        a0 = extended_angles[segment]
        a1 = extended_angles[segment + 1]
        p0 = extended_points[segment]
        p1 = extended_points[segment + 1]
        t = 0.0 if abs(a1 - a0) <= _EPS else (angle - a0) / (a1 - a0)
        result.append((p0[0] + ((p1[0] - p0[0]) * t), p0[1] + ((p1[1] - p0[1]) * t)))
    return result


def _bridge_two_loops_by_angle(
    outer_loop: list[tuple[float, float]],
    inner_loop: list[tuple[float, float]],
    add_vertex,
) -> list[tuple[int, int, int, int]]:
    center = _loop_centroid(inner_loop)
    count = max(len(outer_loop), len(inner_loop))
    count = min(max(count, 8), _DEFAULT_MAX_CONTOUR_SEGMENTS)
    outer = _angle_resample_loop(outer_loop, center, count)
    inner = _angle_resample_loop(inner_loop, center, count)
    faces: list[tuple[int, int, int, int]] = []
    for index in range(count):
        face = (
            add_vertex(outer[index]),
            add_vertex(outer[(index + 1) % count]),
            add_vertex(inner[(index + 1) % count]),
            add_vertex(inner[index]),
        )
        if len(set(face)) == 4:
            faces.append(face)
    return faces


def _max_face_edge_length(mesh: bpy.types.Mesh) -> float:
    max_length = 0.0
    for polygon in mesh.polygons:
        vertices = list(polygon.vertices)
        for index, a in enumerate(vertices):
            b = vertices[(index + 1) % len(vertices)]
            max_length = max(max_length, (mesh.vertices[a].co - mesh.vertices[b].co).length)
    return max_length


def _max_allowed_patch_edge(edge_length: float) -> float:
    return max(edge_length * _env_float("HALLWAY_QUADRANT_REMESH_MAX_EDGE_RATIO", 2.25, 1.0), edge_length + 1e-6)


def _nonmanifold_edge_count(mesh: bpy.types.Mesh) -> int:
    edge_counts: dict[tuple[int, int], int] = {}
    for polygon in mesh.polygons:
        vertices = list(polygon.vertices)
        for index, a in enumerate(vertices):
            b = vertices[(index + 1) % len(vertices)]
            edge = (a, b) if a < b else (b, a)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return sum(1 for count in edge_counts.values() if count > 2)


def _boundary_vertex_indices(mesh: bpy.types.Mesh) -> set[int]:
    edge_counts: dict[tuple[int, int], int] = {}
    for polygon in mesh.polygons:
        vertices = list(polygon.vertices)
        for index, a in enumerate(vertices):
            b = vertices[(index + 1) % len(vertices)]
            edge = (a, b) if a < b else (b, a)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return {index for edge, count in edge_counts.items() if count == 1 for index in edge}


def _point_segment_projection_3d(
    point: tuple[float, float, float],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    denom = (dx * dx) + (dy * dy) + (dz * dz)
    if denom <= _EPS:
        return 0.0, math.dist(point, a)
    t = (((point[0] - a[0]) * dx) + ((point[1] - a[1]) * dy) + ((point[2] - a[2]) * dz)) / denom
    t = max(0.0, min(1.0, t))
    projected = (a[0] + (dx * t), a[1] + (dy * t), a[2] + (dz * t))
    return t, math.dist(point, projected)


def _stitch_boundary_t_junction_quads(mesh: bpy.types.Mesh, tolerance: float) -> int:
    if tolerance <= _EPS or len(mesh.polygons) == 0:
        return 0
    vertices = [tuple(vertex.co) for vertex in mesh.vertices]
    faces = [tuple(polygon.vertices) for polygon in mesh.polygons]
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for index, a in enumerate(face):
            b = face[(index + 1) % len(face)]
            edge = (a, b) if a < b else (b, a)
            edge_faces.setdefault(edge, []).append(face_index)
    boundary_edges = [edge for edge, owners in edge_faces.items() if len(owners) == 1]
    if not boundary_edges:
        return 0
    boundary_vertices = sorted({index for edge in boundary_edges for index in edge})
    cell_size = max(tolerance * 4.0, 1e-6)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for vertex_index in boundary_vertices:
        coord = vertices[vertex_index]
        key = (
            int(math.floor(coord[0] / cell_size)),
            int(math.floor(coord[1] / cell_size)),
            int(math.floor(coord[2] / cell_size)),
        )
        buckets.setdefault(key, []).append(vertex_index)

    splits_by_face: dict[int, tuple[int, list[tuple[float, int]]]] = {}
    endpoint_margin = max(0.02, tolerance)
    for edge in boundary_edges:
        a, b = edge
        a_coord = vertices[a]
        b_coord = vertices[b]
        min_key = (
            int(math.floor((min(a_coord[0], b_coord[0]) - tolerance) / cell_size)),
            int(math.floor((min(a_coord[1], b_coord[1]) - tolerance) / cell_size)),
            int(math.floor((min(a_coord[2], b_coord[2]) - tolerance) / cell_size)),
        )
        max_key = (
            int(math.floor((max(a_coord[0], b_coord[0]) + tolerance) / cell_size)),
            int(math.floor((max(a_coord[1], b_coord[1]) + tolerance) / cell_size)),
            int(math.floor((max(a_coord[2], b_coord[2]) + tolerance) / cell_size)),
        )
        candidates: set[int] = set()
        for ix in range(min_key[0], max_key[0] + 1):
            for iy in range(min_key[1], max_key[1] + 1):
                for iz in range(min_key[2], max_key[2] + 1):
                    candidates.update(buckets.get((ix, iy, iz), []))
        hits: list[tuple[float, int]] = []
        for vertex_index in candidates:
            if vertex_index in edge:
                continue
            t, distance = _point_segment_projection_3d(vertices[vertex_index], a_coord, b_coord)
            if endpoint_margin < t < 1.0 - endpoint_margin and distance <= tolerance:
                hits.append((t, vertex_index))
        if not hits:
            continue
        face_index = edge_faces[edge][0]
        face = faces[face_index]
        if len(face) != 4:
            continue
        edge_start_index = None
        for index, vertex_index in enumerate(face):
            next_index = face[(index + 1) % 4]
            if {vertex_index, next_index} == set(edge):
                edge_start_index = index if vertex_index == a else (index + 1) % 4
                break
        if edge_start_index is None:
            continue
        ordered_hits = sorted(hits)
        unique_hits: list[tuple[float, int]] = []
        for t, vertex_index in ordered_hits:
            if unique_hits and abs(t - unique_hits[-1][0]) <= 1e-4:
                continue
            unique_hits.append((t, vertex_index))
        existing = splits_by_face.get(face_index)
        if existing is None or len(unique_hits) > len(existing[1]):
            splits_by_face[face_index] = (edge_start_index, unique_hits)

    if not splits_by_face:
        return 0

    new_vertices = list(vertices)
    new_faces: list[tuple[int, ...]] = []
    split_faces = 0
    for face_index, face in enumerate(faces):
        split = splits_by_face.get(face_index)
        if split is None:
            new_faces.append(face)
            continue
        edge_start_index, hits = split
        if len(face) != 4:
            new_faces.append(face)
            continue
        a = face[edge_start_index]
        b = face[(edge_start_index + 1) % 4]
        next2 = face[(edge_start_index + 2) % 4]
        prev = face[(edge_start_index - 1) % 4]
        bottom = [a] + [vertex_index for _t, vertex_index in hits] + [b]
        top = [prev]
        prev_coord = new_vertices[prev]
        next_coord = new_vertices[next2]
        for t, _vertex_index in hits:
            top.append(
                len(new_vertices)
            )
            new_vertices.append(
                (
                    prev_coord[0] + ((next_coord[0] - prev_coord[0]) * t),
                    prev_coord[1] + ((next_coord[1] - prev_coord[1]) * t),
                    prev_coord[2] + ((next_coord[2] - prev_coord[2]) * t),
                )
            )
        top.append(next2)
        for index in range(len(bottom) - 1):
            quad = (bottom[index], bottom[index + 1], top[index + 1], top[index])
            if len(set(quad)) == 4:
                new_faces.append(quad)
        split_faces += 1

    mesh.clear_geometry()
    mesh.from_pydata(new_vertices, [], new_faces)
    mesh.update(calc_edges=True)
    if split_faces:
        logger.info("Stitched %s boundary T-junction faces in %s", split_faces, mesh.name)
    return split_faces


def _filter_mesh_to_projected_domain(
    mesh: bpy.types.Mesh,
    plane: _Plane,
    domain_loops: list[list[tuple[float, float]]],
    tolerance: float,
) -> int:
    if not domain_loops or len(mesh.polygons) == 0:
        return 0
    work = len(mesh.polygons) * max(1, sum(len(loop) for loop in domain_loops))
    max_work = _env_int("HALLWAY_QUADRANT_REMESH_MAX_DOMAIN_FILTER_WORK", 2500000, 10000)
    if work > max_work and not _env_bool("HALLWAY_QUADRANT_REMESH_FORCE_DOMAIN_FILTER", False):
        logger.debug("Skipping domain face filter for large mesh: work=%s max=%s", work, max_work)
        return 0

    def inside_or_boundary(point: tuple[float, float]) -> bool:
        return _point_in_domain(point, domain_loops) or _point_near_domain_boundary(point, domain_loops, tolerance)

    vertices = [tuple(vertex.co) for vertex in mesh.vertices]
    kept_faces: list[tuple[int, ...]] = []
    removed = 0
    for polygon in mesh.polygons:
        points = [
            (float(mesh.vertices[index].co[plane.across_axis]), float(mesh.vertices[index].co[plane.along_axis]))
            for index in polygon.vertices
        ]
        centroid = (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))
        samples = [centroid]
        samples.extend(
            (
                (points[index][0] + points[(index + 1) % len(points)][0]) * 0.5,
                (points[index][1] + points[(index + 1) % len(points)][1]) * 0.5,
            )
            for index in range(len(points))
        )
        if all(inside_or_boundary(sample) for sample in samples):
            kept_faces.append(tuple(polygon.vertices))
        else:
            removed += 1
    if removed == 0:
        return 0
    used = sorted({index for face in kept_faces for index in face})
    index_map = {old_index: new_index for new_index, old_index in enumerate(used)}
    compact_vertices = [vertices[index] for index in used]
    compact_faces = [tuple(index_map[index] for index in face) for face in kept_faces if all(index in index_map for index in face)]
    mesh.clear_geometry()
    mesh.from_pydata(compact_vertices, [], compact_faces)
    mesh.update(calc_edges=True)
    return removed


def _smooth_interior_vertices(mesh: bpy.types.Mesh, iterations: int, factor: float) -> None:
    if iterations <= 0 or factor <= 0.0:
        return
    pinned = _boundary_vertex_indices(mesh)
    neighbors: dict[int, set[int]] = {vertex.index: set() for vertex in mesh.vertices}
    for polygon in mesh.polygons:
        vertices = list(polygon.vertices)
        for index, a in enumerate(vertices):
            b = vertices[(index + 1) % len(vertices)]
            neighbors[a].add(b)
            neighbors[b].add(a)
    movable = [vertex.index for vertex in mesh.vertices if vertex.index not in pinned and neighbors[vertex.index]]
    if not movable:
        return
    for _ in range(iterations):
        original = [vertex.co.copy() for vertex in mesh.vertices]
        for index in movable:
            average = mathutils.Vector((0.0, 0.0, 0.0))
            for neighbor_index in neighbors[index]:
                average += original[neighbor_index]
            average /= len(neighbors[index])
            mesh.vertices[index].co = original[index].lerp(average, factor)
    mesh.update(calc_edges=True)


def _mesh_has_projected_face_intersections(mesh: bpy.types.Mesh, plane: _Plane) -> bool:
    boxes: list[tuple[float, float, float, float, int, tuple[tuple[float, float], ...]]] = []
    for polygon in mesh.polygons:
        points = tuple((float(mesh.vertices[index].co[plane.across_axis]), float(mesh.vertices[index].co[plane.along_axis])) for index in polygon.vertices)
        if len(points) < 3:
            continue
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        boxes.append((min_x, max_x, min_y, max_y, polygon.index, points))
    for index, (min_x, max_x, min_y, max_y, poly_index, points) in enumerate(boxes):
        point_set = set(points)
        for other_min_x, other_max_x, other_min_y, other_max_y, other_poly_index, other_points in boxes[index + 1 :]:
            if max_x < other_min_x or other_max_x < min_x or max_y < other_min_y or other_max_y < min_y:
                continue
            if point_set.intersection(other_points):
                continue
            for edge_index, point in enumerate(points):
                next_point = points[(edge_index + 1) % len(points)]
                for other_edge_index, other_point in enumerate(other_points):
                    other_next = other_points[(other_edge_index + 1) % len(other_points)]
                    if _segments_intersect(point, next_point, other_point, other_next):
                        return True
    return False


def _build_inset_single_loop_mesh(
    source_obj: bpy.types.Object,
    loop: list[tuple[float, float]],
    plane: _Plane,
    edge_length: float,
    max_faces: int,
    layer_count: int,
) -> bpy.types.Mesh:
    bm = bmesh.new()
    try:
        verts = [bm.verts.new(_make_coord(plane, point[0], point[1])) for point in loop]
        bm.faces.new(verts)
        bm.normal_update()
        for _ in range(layer_count):
            bm.faces.ensure_lookup_table()
            candidates = [face for face in bm.faces if len(face.verts) != 4]
            if not candidates:
                break
            face = max(candidates, key=lambda item: abs(item.calc_area()))
            try:
                bmesh.ops.inset_region(
                    bm,
                    faces=[face],
                    thickness=edge_length,
                    depth=0.0,
                    use_even_offset=True,
                    use_boundary=True,
                    use_interpolate=True,
                )
            except Exception as exc:
                logger.debug("Inset contour layer stopped for %s: %s", source_obj.name, exc)
                break

        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        bm.faces.ensure_lookup_table()
        bm.faces.index_update()
        vertices = [tuple(vertex.co) for vertex in bm.verts]
        faces: list[tuple[int, ...]] = []
        for face in bm.faces:
            indices = tuple(vertex.index for vertex in face.verts)
            if len(indices) >= 3:
                faces.append(indices)

        if not faces:
            raise StripRemeshUnsupported("inset contour paving produced no faces")
        if len(faces) > max_faces:
            raise StripRemeshUnsupported(f"inset contour paving exceeded face cap ({max_faces})")
        mesh = bpy.data.meshes.new(f"{source_obj.data.name}_inset_paved_remesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.update(calc_edges=True)
        logger.info(
            "Inset-paved remesh %s -> boundary_vertices=%s layers=%s verts=%s faces=%s max_edge=%.4f",
            source_obj.name,
            len(loop),
            layer_count,
            len(mesh.vertices),
            len(mesh.polygons),
            _max_face_edge_length(mesh),
        )
        return mesh
    finally:
        bm.free()


def _principal_axes(loop: list[tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    center = _loop_centroid(loop)
    xx = 0.0
    xy = 0.0
    yy = 0.0
    for x, y in loop:
        dx = x - center[0]
        dy = y - center[1]
        xx += dx * dx
        xy += dx * dy
        yy += dy * dy
    angle = 0.5 * math.atan2(2.0 * xy, xx - yy)
    major = (math.cos(angle), math.sin(angle))
    minor = (-major[1], major[0])
    return major, minor


def _loop_chain(loop: list[tuple[float, float]], start: int, end: int) -> list[tuple[float, float]]:
    if start <= end:
        return loop[start : end + 1]
    return loop[start:] + loop[: end + 1]


def _fallback_quarter_indices(loop: list[tuple[float, float]]) -> list[int]:
    count = len(loop)
    return [0, count // 4, count // 2, (count * 3) // 4]


def _quadrant_anchor_indices(loop: list[tuple[float, float]]) -> list[int]:
    corner_scores: list[tuple[float, int]] = []
    for index, point in enumerate(loop):
        prev_point = loop[(index - 1) % len(loop)]
        next_point = loop[(index + 1) % len(loop)]
        ax = point[0] - prev_point[0]
        ay = point[1] - prev_point[1]
        bx = next_point[0] - point[0]
        by = next_point[1] - point[1]
        if math.hypot(ax, ay) <= _EPS or math.hypot(bx, by) <= _EPS:
            continue
        turn = abs(math.atan2((ax * by) - (ay * bx), (ax * bx) + (ay * by)))
        if turn > 0.35:
            corner_scores.append((turn, index))
    corner_scores.sort(reverse=True)
    corner_indices: list[int] = []
    for _score, index in corner_scores:
        if any(min((index - existing) % len(loop), (existing - index) % len(loop)) < 1 for existing in corner_indices):
            continue
        corner_indices.append(index)
        if len(corner_indices) == 4:
            return sorted(corner_indices)

    center = _loop_centroid(loop)
    major, minor = _principal_axes(loop)
    directions = [(-minor[0], -minor[1]), major, minor, (-major[0], -major[1])]
    raw_indices = []
    for direction in directions:
        raw_indices.append(max(range(len(loop)), key=lambda index: ((loop[index][0] - center[0]) * direction[0]) + ((loop[index][1] - center[1]) * direction[1])))
    indices = sorted(set(raw_indices))
    if len(indices) != 4:
        return _fallback_quarter_indices(loop)
    gaps = [((indices[(index + 1) % 4] - indices[index]) % len(loop)) for index in range(4)]
    if min(gaps) < max(4, len(loop) // 64):
        return _fallback_quarter_indices(loop)
    return indices


def _even_segment_count(length: float, edge_length: float, minimum: int = 4) -> int:
    count = max(minimum, int(math.ceil(length / max(edge_length, _EPS))))
    if count % 2 == 1:
        count += 1
    return count


def _quadrant_patch_counts(loop: list[tuple[float, float]], edge_length: float, max_faces: int) -> tuple[list[int], int, int]:
    anchors = _quadrant_anchor_indices(loop)
    chains = [
        _loop_chain(loop, anchors[0], anchors[1]),
        _loop_chain(loop, anchors[1], anchors[2]),
        _loop_chain(loop, anchors[2], anchors[3]),
        _loop_chain(loop, anchors[3], anchors[0]),
    ]
    if any(len(chain) < 2 for chain in chains):
        raise StripRemeshUnsupported("quadrant patch has degenerate boundary chain")
    u_count = _even_segment_count((_polyline_length(chains[0]) + _polyline_length(chains[2])) * 0.5, edge_length)
    v_count = _even_segment_count((_polyline_length(chains[1]) + _polyline_length(chains[3])) * 0.5, edge_length)
    preserve_ratio = _env_float("HALLWAY_QUADRANT_REMESH_BOUNDARY_PRESERVE_RATIO", 0.0, 0.0)
    if preserve_ratio > 0.0:
        u_count = max(u_count, int(math.ceil(max(len(chains[0]), len(chains[2])) * preserve_ratio)))
        v_count = max(v_count, int(math.ceil(max(len(chains[1]), len(chains[3])) * preserve_ratio)))
        if u_count % 2 == 1:
            u_count += 1
        if v_count % 2 == 1:
            v_count += 1
    max_count_ratio = _env_float("HALLWAY_QUADRANT_REMESH_MAX_COUNT_RATIO", 2.5, 1.0)
    if u_count > v_count * max_count_ratio:
        v_count = max(v_count, int(math.ceil(u_count / max_count_ratio)))
    elif v_count > u_count * max_count_ratio:
        u_count = max(u_count, int(math.ceil(v_count / max_count_ratio)))
    if u_count % 2 == 1:
        u_count += 1
    if v_count % 2 == 1:
        v_count += 1
    if u_count * v_count > max_faces:
        scale = math.sqrt(max_faces / max(1, u_count * v_count))
        u_count = max(4, int(math.floor(u_count * scale)))
        v_count = max(4, int(math.floor(v_count * scale)))
        if u_count % 2 == 1:
            u_count -= 1
        if v_count % 2 == 1:
            v_count -= 1
    max_patch_faces = _env_int("HALLWAY_QUADRANT_REMESH_MAX_PATCH_FACES", 2500, 64)
    max_patch_side = _env_int("HALLWAY_QUADRANT_REMESH_MAX_PATCH_SIDE", 64, 4)
    if u_count * v_count > max_patch_faces:
        scale = math.sqrt(max_patch_faces / max(1, u_count * v_count))
        u_count = max(4, int(math.floor(u_count * scale)))
        v_count = max(4, int(math.floor(v_count * scale)))
    u_count = min(u_count, max_patch_side)
    v_count = min(v_count, max_patch_side)
    if u_count % 2 == 1:
        u_count -= 1
    if v_count % 2 == 1:
        v_count -= 1
    u_count = max(4, u_count)
    v_count = max(4, v_count)
    return anchors, u_count, v_count


def _quadrant_patch_boundary(
    loop: list[tuple[float, float]],
    anchors: list[int],
    u_count: int,
    v_count: int,
) -> list[tuple[float, float]]:
    chains = [
        _loop_chain(loop, anchors[0], anchors[1]),
        _loop_chain(loop, anchors[1], anchors[2]),
        _loop_chain(loop, anchors[2], anchors[3]),
        _loop_chain(loop, anchors[3], anchors[0]),
    ]
    bottom = _resample_open_polyline(chains[0], u_count)
    right = _resample_open_polyline(chains[1], v_count)
    top_walk = _resample_open_polyline(chains[2], u_count)
    left_walk = _resample_open_polyline(chains[3], v_count)
    return bottom + right[1:] + top_walk[1:] + left_walk[1:-1]


def _quadrant_patch_grid_from_counts(
    loop: list[tuple[float, float]],
    anchors: list[int],
    u_count: int,
    v_count: int,
) -> list[list[tuple[float, float]]]:
    chains = [
        _loop_chain(loop, anchors[0], anchors[1]),
        _loop_chain(loop, anchors[1], anchors[2]),
        _loop_chain(loop, anchors[2], anchors[3]),
        _loop_chain(loop, anchors[3], anchors[0]),
    ]
    bottom = _resample_open_polyline(chains[0], u_count)
    right = _resample_open_polyline(chains[1], v_count)
    top = list(reversed(_resample_open_polyline(chains[2], u_count)))
    left = list(reversed(_resample_open_polyline(chains[3], v_count)))
    p00 = bottom[0]
    p10 = bottom[-1]
    p11 = top[-1]
    p01 = top[0]
    grid: list[list[tuple[float, float]]] = []
    for row in range(v_count + 1):
        v = row / v_count
        row_points: list[tuple[float, float]] = []
        for column in range(u_count + 1):
            u = column / u_count
            b = bottom[column]
            t = top[column]
            l = left[row]
            r = right[row]
            bilinear = (
                ((1.0 - u) * (1.0 - v) * p00[0]) + (u * (1.0 - v) * p10[0]) + ((1.0 - u) * v * p01[0]) + (u * v * p11[0]),
                ((1.0 - u) * (1.0 - v) * p00[1]) + (u * (1.0 - v) * p10[1]) + ((1.0 - u) * v * p01[1]) + (u * v * p11[1]),
            )
            row_points.append(
                (
                    ((1.0 - v) * b[0]) + (v * t[0]) + ((1.0 - u) * l[0]) + (u * r[0]) - bilinear[0],
                    ((1.0 - v) * b[1]) + (v * t[1]) + ((1.0 - u) * l[1]) + (u * r[1]) - bilinear[1],
                )
            )
        grid.append(row_points)
    return grid


def _quad_signed_area_2d(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        area += (point[0] * nxt[1]) - (nxt[0] * point[1])
    return area * 0.5


def _grid_cell_signed_area(grid: list[list[tuple[float, float]]], row: int, column: int) -> float:
    return _quad_signed_area_2d(
        [
            grid[row][column],
            grid[row][column + 1],
            grid[row + 1][column + 1],
            grid[row + 1][column],
        ]
    )


def _quad_scaled_jacobian_2d(points: list[tuple[float, float]], orientation: float) -> float:
    quality = 1.0
    for index, point in enumerate(points):
        prev_point = points[(index - 1) % len(points)]
        next_point = points[(index + 1) % len(points)]
        ax = prev_point[0] - point[0]
        ay = prev_point[1] - point[1]
        bx = next_point[0] - point[0]
        by = next_point[1] - point[1]
        denom = max(math.hypot(ax, ay) * math.hypot(bx, by), _EPS)
        quality = min(quality, (((bx * ay) - (by * ax)) * orientation) / denom)
    return quality


def _grid_cell_quality(grid: list[list[tuple[float, float]]], row: int, column: int, orientation: float) -> float:
    points = [
        grid[row][column],
        grid[row][column + 1],
        grid[row + 1][column + 1],
        grid[row + 1][column],
    ]
    return min(_grid_cell_signed_area(grid, row, column) * orientation, _quad_scaled_jacobian_2d(points, orientation))


def _point_distance_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _grid_folded_cell_count(grid: list[list[tuple[float, float]]]) -> int:
    total_area = 0.0
    for row in range(len(grid) - 1):
        for column in range(len(grid[0]) - 1):
            total_area += _grid_cell_signed_area(grid, row, column)
    orientation = 1.0 if total_area >= 0.0 else -1.0
    return sum(
        1
        for row in range(len(grid) - 1)
        for column in range(len(grid[0]) - 1)
        if _grid_cell_quality(grid, row, column, orientation) <= 1e-10
    )


def _grid_projected_crossing_count(grid: list[list[tuple[float, float]]], limit: int = 1000) -> int:
    segments: list[tuple[tuple[float, float], tuple[float, float], tuple[int, int], tuple[int, int]]] = []
    row_count = len(grid)
    column_count = len(grid[0]) if row_count else 0
    projected_segment_count = (row_count * max(0, column_count - 1)) + (max(0, row_count - 1) * column_count)
    max_segments = _env_int("HALLWAY_QUADRANT_REMESH_MAX_CROSSING_SEGMENTS", 24000, 1000)
    if projected_segment_count > max_segments and not _env_bool("HALLWAY_QUADRANT_REMESH_FORCE_CROSSING_CHECK", False):
        return 0
    for row in range(row_count):
        for column in range(column_count - 1):
            segments.append((grid[row][column], grid[row][column + 1], (row, column), (row, column + 1)))
    for row in range(row_count - 1):
        for column in range(column_count):
            segments.append((grid[row][column], grid[row + 1][column], (row, column), (row + 1, column)))
    if not segments:
        return 0
    min_x = min(min(a[0], b[0]) for a, b, _a_id, _b_id in segments)
    max_x = max(max(a[0], b[0]) for a, b, _a_id, _b_id in segments)
    min_y = min(min(a[1], b[1]) for a, b, _a_id, _b_id in segments)
    max_y = max(max(a[1], b[1]) for a, b, _a_id, _b_id in segments)
    bucket_size = max(max_x - min_x, max_y - min_y, _EPS) / max(16.0, math.sqrt(len(segments)) * 0.5)
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (a, b, _a_id, _b_id) in enumerate(segments):
        x0 = int(math.floor((min(a[0], b[0]) - min_x) / bucket_size))
        x1 = int(math.floor((max(a[0], b[0]) - min_x) / bucket_size))
        y0 = int(math.floor((min(a[1], b[1]) - min_y) / bucket_size))
        y1 = int(math.floor((max(a[1], b[1]) - min_y) / bucket_size))
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                buckets.setdefault((x, y), []).append(index)
    crossings = 0
    checked: set[tuple[int, int]] = set()
    for entries in buckets.values():
        if len(entries) < 2:
            continue
        for local_index, index in enumerate(entries):
            a0, a1, a0_id, a1_id = segments[index]
            min_ax = min(a0[0], a1[0])
            max_ax = max(a0[0], a1[0])
            min_ay = min(a0[1], a1[1])
            max_ay = max(a0[1], a1[1])
            for other_index in entries[local_index + 1 :]:
                pair = (index, other_index) if index < other_index else (other_index, index)
                if pair in checked:
                    continue
                checked.add(pair)
                b0, b1, b0_id, b1_id = segments[other_index]
                if a0_id in (b0_id, b1_id) or a1_id in (b0_id, b1_id):
                    continue
                if max_ax < min(b0[0], b1[0]) or max(b0[0], b1[0]) < min_ax:
                    continue
                if max_ay < min(b0[1], b1[1]) or max(b0[1], b1[1]) < min_ay:
                    continue
                if _segments_intersect(a0, a1, b0, b1):
                    crossings += 1
                    if crossings >= limit:
                        return crossings
    return crossings


def _grid_min_edge_ratio(
    grid: list[list[tuple[float, float]]],
    rest_grid: list[list[tuple[float, float]]],
    row: int,
    column: int,
) -> float:
    ratios: list[float] = []
    for neighbor_row, neighbor_column in ((row, column - 1), (row, column + 1), (row - 1, column), (row + 1, column)):
        if neighbor_row < 0 or neighbor_row >= len(grid) or neighbor_column < 0 or neighbor_column >= len(grid[0]):
            continue
        rest_length = _point_distance_2d(rest_grid[row][column], rest_grid[neighbor_row][neighbor_column])
        if rest_length <= _EPS:
            continue
        ratios.append(_point_distance_2d(grid[row][column], grid[neighbor_row][neighbor_column]) / rest_length)
    return min(ratios) if ratios else 1.0


def _grid_vertex_quality(
    grid: list[list[tuple[float, float]]],
    row: int,
    column: int,
    orientation: float,
    rest_grid: list[list[tuple[float, float]]] | None = None,
    min_edge_ratio: float = 0.0,
) -> float:
    row_count = len(grid)
    column_count = len(grid[0]) if row_count else 0
    quality = float("inf")
    for cell_row in (row - 1, row):
        if cell_row < 0 or cell_row >= row_count - 1:
            continue
        for cell_column in (column - 1, column):
            if cell_column < 0 or cell_column >= column_count - 1:
                continue
            quality = min(quality, _grid_cell_quality(grid, cell_row, cell_column, orientation))
    if rest_grid is not None and min_edge_ratio > 0.0:
        quality = min(quality, _grid_min_edge_ratio(grid, rest_grid, row, column) - min_edge_ratio)
    return quality if quality != float("inf") else 0.0


def _edge_tension_target(
    grid: list[list[tuple[float, float]]],
    rest_grid: list[list[tuple[float, float]]],
    row: int,
    column: int,
    base_target: tuple[float, float],
    tension: float,
) -> tuple[float, float]:
    if tension <= 0.0:
        return base_target
    point = grid[row][column]
    rest_point = rest_grid[row][column]
    force_x = 0.0
    force_y = 0.0
    for neighbor_row, neighbor_column in ((row, column - 1), (row, column + 1), (row - 1, column), (row + 1, column)):
        neighbor = grid[neighbor_row][neighbor_column]
        rest_neighbor = rest_grid[neighbor_row][neighbor_column]
        rest_dx = rest_point[0] - rest_neighbor[0]
        rest_dy = rest_point[1] - rest_neighbor[1]
        rest_length = math.hypot(rest_dx, rest_dy)
        if rest_length <= _EPS:
            continue
        dx = point[0] - neighbor[0]
        dy = point[1] - neighbor[1]
        length = math.hypot(dx, dy)
        if length <= _EPS:
            unit_x = rest_dx / rest_length
            unit_y = rest_dy / rest_length
        else:
            unit_x = dx / length
            unit_y = dy / length
        delta = max(0.0, rest_length - length)
        force_x += unit_x * delta
        force_y += unit_y * delta
    return (
        base_target[0] + (force_x * tension),
        base_target[1] + (force_y * tension),
    )


def _try_move_grid_vertex(
    grid: list[list[tuple[float, float]]],
    rest_grid: list[list[tuple[float, float]]],
    row: int,
    column: int,
    candidate: tuple[float, float],
    orientation: float,
    min_edge_ratio: float,
) -> bool:
    original = grid[row][column]
    current_quality = _grid_vertex_quality(grid, row, column, orientation, rest_grid, min_edge_ratio)
    grid[row][column] = candidate
    proposed_quality = _grid_vertex_quality(grid, row, column, orientation, rest_grid, min_edge_ratio)
    if (current_quality < 1e-10 and proposed_quality > current_quality + 1e-12) or (
        current_quality >= 1e-10 and proposed_quality > 1e-10
    ):
        return True
    grid[row][column] = original
    return False


def _repel_close_grid_vertices(
    grid: list[list[tuple[float, float]]],
    rest_grid: list[list[tuple[float, float]]],
    min_distance: float,
    iterations: int,
    factor: float,
    min_edge_ratio: float,
) -> int:
    if iterations <= 0 or min_distance <= _EPS or len(grid) < 3 or len(grid[0]) < 3:
        return 0
    total_area = 0.0
    for row in range(len(grid) - 1):
        for column in range(len(grid[0]) - 1):
            total_area += _grid_cell_signed_area(grid, row, column)
    orientation = 1.0 if total_area >= 0.0 else -1.0
    moved = 0
    cell_size = min_distance
    for _ in range(iterations):
        buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for row in range(1, len(grid) - 1):
            for column in range(1, len(grid[0]) - 1):
                point = grid[row][column]
                key = (int(math.floor(point[0] / cell_size)), int(math.floor(point[1] / cell_size)))
                buckets.setdefault(key, []).append((row, column))
        any_moved = False
        for key, entries in list(buckets.items()):
            candidates: list[tuple[int, int]] = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    candidates.extend(buckets.get((key[0] + dx, key[1] + dy), []))
            for row, column in entries:
                point = grid[row][column]
                for other_row, other_column in candidates:
                    if (other_row, other_column) <= (row, column):
                        continue
                    if abs(other_row - row) <= 1 and abs(other_column - column) <= 1:
                        continue
                    other = grid[other_row][other_column]
                    dx = point[0] - other[0]
                    dy = point[1] - other[1]
                    distance = math.hypot(dx, dy)
                    if distance >= min_distance:
                        continue
                    if distance <= _EPS:
                        rest_dx = rest_grid[row][column][0] - rest_grid[other_row][other_column][0]
                        rest_dy = rest_grid[row][column][1] - rest_grid[other_row][other_column][1]
                        rest_distance = max(math.hypot(rest_dx, rest_dy), _EPS)
                        dx = rest_dx / rest_distance
                        dy = rest_dy / rest_distance
                    else:
                        dx /= distance
                        dy /= distance
                    push = (min_distance - distance) * 0.5 * factor
                    moved_a = _try_move_grid_vertex(
                        grid,
                        rest_grid,
                        row,
                        column,
                        (point[0] + (dx * push), point[1] + (dy * push)),
                        orientation,
                        min_edge_ratio,
                    )
                    moved_b = _try_move_grid_vertex(
                        grid,
                        rest_grid,
                        other_row,
                        other_column,
                        (other[0] - (dx * push), other[1] - (dy * push)),
                        orientation,
                        min_edge_ratio,
                    )
                    if moved_a or moved_b:
                        any_moved = True
                        moved += int(moved_a) + int(moved_b)
        if not any_moved:
            break
    return moved


def _median_grid_rest_edge_length(rest_grid: list[list[tuple[float, float]]]) -> float:
    lengths: list[float] = []
    for row in range(len(rest_grid)):
        for column in range(len(rest_grid[0])):
            if column + 1 < len(rest_grid[0]):
                lengths.append(_point_distance_2d(rest_grid[row][column], rest_grid[row][column + 1]))
            if row + 1 < len(rest_grid):
                lengths.append(_point_distance_2d(rest_grid[row][column], rest_grid[row + 1][column]))
    lengths = sorted(length for length in lengths if length > _EPS)
    if not lengths:
        return 0.0
    return lengths[len(lengths) // 2]


def _smooth_structured_patch_grid(
    grid: list[list[tuple[float, float]]],
    iterations: int,
    factor: float,
) -> int:
    if iterations <= 0 or factor <= 0.0 or len(grid) < 3 or len(grid[0]) < 3:
        return 0
    total_area = 0.0
    for row in range(len(grid) - 1):
        for column in range(len(grid[0]) - 1):
            total_area += _grid_cell_signed_area(grid, row, column)
    orientation = 1.0 if total_area >= 0.0 else -1.0
    rest_grid = [[point for point in row_points] for row_points in grid]
    edge_tension = _env_float("HALLWAY_QUADRANT_REMESH_EDGE_TENSION", 0.5, 0.0)
    min_edge_ratio = _env_float("HALLWAY_QUADRANT_REMESH_MIN_EDGE_RATIO", 0.45, 0.0)
    changed = 0
    for _ in range(iterations):
        for row in range(1, len(grid) - 1):
            for column in range(1, len(grid[0]) - 1):
                original = grid[row][column]
                current_quality = _grid_vertex_quality(grid, row, column, orientation, rest_grid, min_edge_ratio)
                left = grid[row][column - 1]
                right = grid[row][column + 1]
                down = grid[row - 1][column]
                up = grid[row + 1][column]
                xi_len_sq = max((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2, _EPS)
                eta_len_sq = max((up[0] - down[0]) ** 2 + (up[1] - down[1]) ** 2, _EPS)
                # Elliptic/Winslow-flavored relaxation: weight each axis by the opposite metric
                # so stretched rows/columns relax without dragging fixed silhouette vertices.
                target = (
                    ((eta_len_sq * (left[0] + right[0])) + (xi_len_sq * (down[0] + up[0]))) / (2.0 * (xi_len_sq + eta_len_sq)),
                    ((eta_len_sq * (left[1] + right[1])) + (xi_len_sq * (down[1] + up[1]))) / (2.0 * (xi_len_sq + eta_len_sq)),
                )
                target = _edge_tension_target(grid, rest_grid, row, column, target, edge_tension)
                step = min(1.0, factor)
                accepted = False
                for _attempt in range(8):
                    candidate = (
                        original[0] + ((target[0] - original[0]) * step),
                        original[1] + ((target[1] - original[1]) * step),
                    )
                    grid[row][column] = candidate
                    proposed_quality = _grid_vertex_quality(grid, row, column, orientation, rest_grid, min_edge_ratio)
                    if (current_quality < 1e-10 and proposed_quality > current_quality + 1e-12) or (
                        current_quality >= 1e-10 and proposed_quality > 1e-10
                    ):
                        accepted = True
                        changed += 1
                        break
                    step *= 0.5
                if not accepted:
                    grid[row][column] = original
    repel_ratio = _env_float("HALLWAY_QUADRANT_REMESH_VERTEX_SEPARATION_RATIO", 0.75, 0.0)
    repel_iterations = _env_int("HALLWAY_QUADRANT_REMESH_REPEL_ITERATIONS", 8, 0)
    median_edge_length = _median_grid_rest_edge_length(rest_grid)
    changed += _repel_close_grid_vertices(
        grid,
        rest_grid,
        median_edge_length * repel_ratio,
        repel_iterations,
        min(1.0, max(0.05, factor)),
        min_edge_ratio,
    )
    return changed


def _smoothing_iterations_for_grid(u_count: int, v_count: int) -> int:
    requested = _env_int("HALLWAY_QUADRANT_REMESH_SMOOTH_ITERATIONS", _DEFAULT_SMOOTHING_ITERATIONS, 0)
    if _env_bool("HALLWAY_QUADRANT_REMESH_FORCE_SMOOTH_ITERATIONS", False):
        return requested
    cells = max(1, u_count * v_count)
    max_work = _env_int("HALLWAY_QUADRANT_REMESH_MAX_SMOOTH_WORK", 400000, 10000)
    budgeted = max(4, int(max_work / cells))
    return min(requested, budgeted)


def _build_tessellated_quad_patch_mesh(
    source_obj: bpy.types.Object,
    loop: list[tuple[float, float]],
    plane: _Plane,
    max_faces: int,
    reason: str,
) -> bpy.types.Mesh:
    points, faces = _quad_faces_from_tessellation([loop])
    if len(faces) > max_faces:
        raise StripRemeshUnsupported(f"tessellated quad patch exceeded face cap ({max_faces})")
    vertices = [_make_coord(plane, point[0], point[1]) for point in points]
    vertices, faces, removed_faces = _sanitize_pydata_faces(vertices, faces)
    mesh = bpy.data.meshes.new(f"{source_obj.data.name}_tessellated_quad_patch_remesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    nonmanifold_edges = _nonmanifold_edge_count(mesh)
    if nonmanifold_edges:
        bpy.data.meshes.remove(mesh)
        raise StripRemeshUnsupported(f"tessellated quad patch produced {nonmanifold_edges} non-manifold edges")
    logger.warning(
        "Tessellated quad patch fallback %s -> reason=%s verts=%s faces=%s removed=%s max_edge=%.4f",
        source_obj.name,
        reason,
        len(mesh.vertices),
        len(mesh.polygons),
        removed_faces,
        _max_face_edge_length(mesh),
    )
    return mesh


def _build_uniform_grid_quad_patch_mesh(
    source_obj: bpy.types.Object,
    loop: list[tuple[float, float]],
    plane: _Plane,
    edge_length: float,
    max_faces: int,
    reason: str,
) -> bpy.types.Mesh:
    if len(loop) < 4:
        raise StripRemeshUnsupported("not enough vertices for uniform patch fallback")
    if _loop_area(loop) < 0.0:
        loop = list(reversed(loop))
    center = _loop_centroid(loop)
    major, minor = _principal_axes(loop)

    def to_local(point: tuple[float, float]) -> tuple[float, float]:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return (dx * major[0]) + (dy * major[1]), (dx * minor[0]) + (dy * minor[1])

    def from_local(u: float, v: float) -> tuple[float, float]:
        return (
            center[0] + (major[0] * u) + (minor[0] * v),
            center[1] + (major[1] * u) + (minor[1] * v),
        )

    local_points = [to_local(point) for point in loop]
    min_u = min(point[0] for point in local_points)
    max_u = max(point[0] for point in local_points)
    min_v = min(point[1] for point in local_points)
    max_v = max(point[1] for point in local_points)
    width = max_u - min_u
    height = max_v - min_v
    if width <= _EPS or height <= _EPS:
        raise StripRemeshUnsupported("uniform patch fallback is degenerate")

    u_count = _even_segment_count(width, edge_length, minimum=2)
    v_count = _even_segment_count(height, edge_length, minimum=2)
    max_uniform_faces = min(max_faces, _env_int("HALLWAY_UNIFORM_PATCH_MAX_FACES", 5000, 64))
    if u_count * v_count > max_uniform_faces:
        scale = math.sqrt(max_uniform_faces / max(1, u_count * v_count))
        u_count = max(2, int(math.floor(u_count * scale)))
        v_count = max(2, int(math.floor(v_count * scale)))
    max_side = _env_int("HALLWAY_UNIFORM_PATCH_MAX_SIDE", 96, 4)
    u_count = min(max_side, max(2, u_count))
    v_count = min(max_side, max(2, v_count))
    if u_count * v_count > max_uniform_faces:
        if u_count >= v_count:
            u_count = max(2, int(max_uniform_faces // max(1, v_count)))
        else:
            v_count = max(2, int(max_uniform_faces // max(1, u_count)))
    if u_count < 2 or v_count < 2:
        raise StripRemeshUnsupported("uniform patch fallback face budget too small")

    tolerance = edge_length * _env_float("HALLWAY_UNIFORM_PATCH_BOUNDARY_TOLERANCE", 0.45, 0.0)
    clamp_outside = _env_bool("HALLWAY_UNIFORM_PATCH_CLAMP_OUTSIDE", False)
    inset = edge_length * _env_float("HALLWAY_UNIFORM_PATCH_BOUNDARY_INSET", 0.03, 0.0)

    def inside_or_boundary(point: tuple[float, float]) -> bool:
        return _point_in_domain(point, [loop]) or _point_near_domain_boundary(point, [loop], tolerance)

    def cell_covers_domain(points: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]) -> bool:
        samples = [
            ((points[0][0] + points[1][0] + points[2][0] + points[3][0]) * 0.25, (points[0][1] + points[1][1] + points[2][1] + points[3][1]) * 0.25),
            *points,
            ((points[0][0] + points[1][0]) * 0.5, (points[0][1] + points[1][1]) * 0.5),
            ((points[1][0] + points[2][0]) * 0.5, (points[1][1] + points[2][1]) * 0.5),
            ((points[2][0] + points[3][0]) * 0.5, (points[2][1] + points[3][1]) * 0.5),
            ((points[3][0] + points[0][0]) * 0.5, (points[3][1] + points[0][1]) * 0.5),
        ]
        if any(inside_or_boundary(sample) for sample in samples):
            return True
        for cell_index, point in enumerate(points):
            cell_next = points[(cell_index + 1) % len(points)]
            for loop_index, loop_point in enumerate(loop):
                if _segments_intersect(point, cell_next, loop_point, loop[(loop_index + 1) % len(loop)]):
                    return True
        return False

    vertices: list[tuple[float, float, float]] = []
    clamped = 0
    for row in range(v_count + 1):
        v = min_v + (height * row / v_count)
        for column in range(u_count + 1):
            u = min_u + (width * column / u_count)
            point = from_local(u, v)
            if clamp_outside and not inside_or_boundary(point):
                clamped_point = _clamp_point_to_domain(point, [loop], center, inset)
                if _point_distance_2d(point, clamped_point) > _EPS:
                    point = clamped_point
                    clamped += 1
            vertices.append(_make_coord(plane, point[0], point[1]))

    columns = u_count + 1
    faces: list[tuple[int, int, int, int]] = []
    for row in range(v_count):
        for column in range(u_count):
            cell_points = (
                from_local(min_u + (width * column / u_count), min_v + (height * row / v_count)),
                from_local(min_u + (width * (column + 1) / u_count), min_v + (height * row / v_count)),
                from_local(min_u + (width * (column + 1) / u_count), min_v + (height * (row + 1) / v_count)),
                from_local(min_u + (width * column / u_count), min_v + (height * (row + 1) / v_count)),
            )
            if not cell_covers_domain(cell_points):
                continue
            face = (
                (row * columns) + column,
                (row * columns) + column + 1,
                ((row + 1) * columns) + column + 1,
                ((row + 1) * columns) + column,
            )
            faces.append(face)
    vertices, faces, removed_faces = _sanitize_pydata_faces(vertices, faces)
    if not faces:
        raise StripRemeshUnsupported("uniform patch fallback produced no faces")

    mesh = bpy.data.meshes.new(f"{source_obj.data.name}_uniform_quad_patch_remesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    nonmanifold_edges = _nonmanifold_edge_count(mesh)
    if nonmanifold_edges:
        bpy.data.meshes.remove(mesh)
        raise StripRemeshUnsupported(f"uniform patch fallback produced {nonmanifold_edges} non-manifold edges")
    max_edge = _max_face_edge_length(mesh)
    if max_edge > _max_allowed_patch_edge(edge_length) * _env_float("HALLWAY_UNIFORM_PATCH_EDGE_SLACK", 1.4, 1.0):
        bpy.data.meshes.remove(mesh)
        raise StripRemeshUnsupported(f"uniform patch fallback retained max_edge={max_edge:.4f}")
    logger.warning(
        "Uniform grid patch fallback %s -> reason=%s grid=%sx%s verts=%s faces=%s clamped=%s removed=%s max_edge=%.4f",
        source_obj.name,
        reason,
        u_count,
        v_count,
        len(mesh.vertices),
        len(mesh.polygons),
        clamped,
        removed_faces,
        max_edge,
    )
    return mesh


def _build_quadrant_patch_mesh(
    source_obj: bpy.types.Object,
    loop: list[tuple[float, float]],
    plane: _Plane,
    edge_length: float,
    max_faces: int,
) -> bpy.types.Mesh:
    if len(loop) < 8:
        raise StripRemeshUnsupported("not enough silhouette vertices for quadrant patch")
    if _loop_area(loop) < 0.0:
        loop = list(reversed(loop))
    anchors, u_count, v_count = _quadrant_patch_counts(loop, edge_length, max_faces)
    grid = _quadrant_patch_grid_from_counts(loop, anchors, u_count, v_count)

    smoothing_iterations = _smoothing_iterations_for_grid(u_count, v_count)
    smoothing_factor = _env_float("HALLWAY_QUADRANT_REMESH_SMOOTH_FACTOR", _DEFAULT_SMOOTHING_FACTOR, 0.0)
    base_grid = [[point for point in row_points] for row_points in grid]
    base_folded = _grid_folded_cell_count(base_grid)
    base_crossings = _grid_projected_crossing_count(base_grid)
    smooth_moves = 0
    smoothed_folded = base_folded
    smoothed_crossings = base_crossings
    for factor_scale in (1.0, 0.5, 0.25, 0.0):
        candidate_grid = [[point for point in row_points] for row_points in base_grid]
        candidate_moves = _smooth_structured_patch_grid(candidate_grid, smoothing_iterations, min(smoothing_factor * factor_scale, 1.0))
        candidate_folded = _grid_folded_cell_count(candidate_grid)
        candidate_crossings = _grid_projected_crossing_count(candidate_grid)
        if (candidate_folded, candidate_crossings) <= (base_folded, base_crossings):
            grid = candidate_grid
            smooth_moves = candidate_moves
            smoothed_folded = candidate_folded
            smoothed_crossings = candidate_crossings
            if candidate_folded == 0 and candidate_crossings == 0:
                break
    clamped_points = _clamp_grid_to_domain(grid, [loop], edge_length * 0.08)

    vertices: list[tuple[float, float, float]] = []
    for row_points in grid:
        for point in row_points:
            vertices.append(_make_coord(plane, point[0], point[1]))

    faces: list[tuple[int, int, int, int]] = []
    columns = u_count + 1
    for row in range(v_count):
        for column in range(u_count):
            faces.append((
                (row * columns) + column,
                (row * columns) + column + 1,
                ((row + 1) * columns) + column + 1,
                ((row + 1) * columns) + column,
            ))
    mesh = bpy.data.meshes.new(f"{source_obj.data.name}_quadrant_patch_remesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    removed_outside = (
        _filter_mesh_to_projected_domain(mesh, plane, [loop], edge_length * 0.25)
        if _env_bool("HALLWAY_QUADRANT_REMESH_CLIP_OUTSIDE", False)
        else 0
    )
    if removed_outside:
        logger.info("Quadrant patch clipped %s faces outside %s", removed_outside, source_obj.name)
    if _nonmanifold_edge_count(mesh) != 0:
        bpy.data.meshes.remove(mesh)
        raise StripRemeshUnsupported("quadrant patch produced non-manifold edges")
    if _env_bool("HALLWAY_QUADRANT_REMESH_REJECT_INTERSECTIONS") and _mesh_has_projected_face_intersections(mesh, plane):
        bpy.data.meshes.remove(mesh)
        raise StripRemeshUnsupported("quadrant patch produced projected face intersections")
    max_folded = _env_int("HALLWAY_QUADRANT_REMESH_MAX_ACCEPTED_FOLDED", 16, 0)
    max_crossings = _env_int("HALLWAY_QUADRANT_REMESH_MAX_ACCEPTED_CROSSINGS", 64, 0)
    if (smoothed_folded > max_folded or smoothed_crossings > max_crossings) and not _env_bool("HALLWAY_QUADRANT_REMESH_ALLOW_BAD_PATCHES", False):
        bpy.data.meshes.remove(mesh)
        raise StripRemeshUnsupported(f"quadrant patch retained folded={smoothed_folded} crossings={smoothed_crossings}")
    max_face_edge = _max_face_edge_length(mesh)
    if max_face_edge > _max_allowed_patch_edge(edge_length):
        bpy.data.meshes.remove(mesh)
        raise StripRemeshUnsupported(f"quadrant patch retained max_edge={max_face_edge:.4f}")
    logger.info(
        "Quadrant patch remesh %s -> anchors=%s grid=%sx%s verts=%s faces=%s max_edge=%.4f structured_smoothing=%sx%.2f moves=%s clamped=%s folded=%s->%s crossings=%s->%s",
        source_obj.name,
        anchors,
        u_count,
        v_count,
        len(mesh.vertices),
        len(mesh.polygons),
        max_face_edge,
        smoothing_iterations,
        min(smoothing_factor, 1.0),
        smooth_moves,
        clamped_points,
        base_folded,
        smoothed_folded,
        base_crossings,
        smoothed_crossings,
    )
    return mesh


def _build_contour_quadrant_patch_mesh(
    source_obj: bpy.types.Object,
    loop: list[tuple[float, float]],
    plane: _Plane,
    edge_length: float,
    max_faces: int,
    *,
    allow_decompose: bool = True,
) -> bpy.types.Mesh:
    if len(loop) < 8:
        raise StripRemeshUnsupported("not enough silhouette vertices for contour-quadrant patch")
    if _loop_area(loop) < 0.0:
        loop = list(reversed(loop))
    if allow_decompose and _env_bool("HALLWAY_QUADRANT_REMESH_DECOMPOSE_CONCAVE", True):
        patches = _decompose_concave_loop(loop, _env_int("HALLWAY_QUADRANT_REMESH_MAX_PATCHES", 12, 1))
        if len(patches) > 1:
            meshes: list[bpy.types.Mesh] = []
            for patch in patches:
                try:
                    patch_mesh = _build_contour_quadrant_patch_mesh(
                        source_obj,
                        patch,
                        plane,
                        edge_length,
                        max(100, max_faces // len(patches)),
                        allow_decompose=False,
                    )
                except StripRemeshUnsupported as exc:
                    patch_budget = max(100, max_faces // len(patches))
                    try:
                        patch_mesh = _build_uniform_grid_quad_patch_mesh(
                            source_obj,
                            patch,
                            plane,
                            edge_length,
                            patch_budget,
                            str(exc),
                        )
                    except StripRemeshUnsupported as fallback_exc:
                        if not _env_bool("HALLWAY_ALLOW_TESSELLATED_PATCH_FALLBACK", False):
                            raise StripRemeshUnsupported(f"uniform patch fallback failed: {fallback_exc}") from fallback_exc
                        logger.warning("Uniform patch fallback failed for %s: %s", source_obj.name, fallback_exc)
                        patch_mesh = _build_tessellated_quad_patch_mesh(source_obj, patch, plane, patch_budget, str(exc))
                meshes.append(patch_mesh)
            combined = _combine_meshes(f"{source_obj.data.name}_decomposed_contour_quadrant_remesh", meshes)
            nonmanifold_edges = _nonmanifold_edge_count(combined)
            logger.info(
                "Decomposed contour-quadrant remesh %s -> patches=%s verts=%s faces=%s nonmanifold=%s max_edge=%.4f",
                source_obj.name,
                len(patches),
                len(combined.vertices),
                len(combined.polygons),
                nonmanifold_edges,
                _max_face_edge_length(combined),
            )
            if nonmanifold_edges and _env_bool("HALLWAY_QUADRANT_REMESH_REJECT_COMBINED_NONMANIFOLD", True):
                bpy.data.meshes.remove(combined)
                raise StripRemeshUnsupported(f"decomposed contour-quadrant produced {nonmanifold_edges} non-manifold edges")
            return combined
    anchors, u_count, v_count = _quadrant_patch_counts(loop, edge_length, max_faces)
    preflow_layers = _env_int("HALLWAY_QUADRANT_REMESH_PREFLOW_LAYERS", 4, 0)
    preflow_step = edge_length * _env_float("HALLWAY_QUADRANT_REMESH_PREFLOW_STEP", 0.85, 0.05)
    rings = [loop]
    for layer_index in range(1, preflow_layers + 1):
        candidate = _offset_loop(loop, preflow_step * layer_index)
        if not _valid_offset_loop(candidate, [loop]):
            break
        rings.append(candidate)
    while len(rings) > 1:
        test_anchors, test_u_count, test_v_count = _quadrant_patch_counts(rings[-1], edge_length, max_faces)
        test_grid = _quadrant_patch_grid_from_counts(rings[-1], test_anchors, test_u_count, test_v_count)
        if _grid_folded_cell_count(test_grid) == 0 and _grid_projected_crossing_count(test_grid) == 0:
            break
        rings.pop()
    if len(rings) <= 1:
        return _build_quadrant_patch_mesh(source_obj, loop, plane, edge_length, max_faces)

    boundaries = [_quadrant_patch_boundary(ring, anchors, u_count, v_count) for ring in rings]
    boundary_count = len(boundaries[0])
    if any(len(boundary) != boundary_count for boundary in boundaries):
        raise StripRemeshUnsupported("contour-quadrant boundaries did not align")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    ring_indices: list[list[int]] = []
    for boundary in boundaries:
        indices = []
        for point in boundary:
            indices.append(len(vertices))
            vertices.append(_make_coord(plane, point[0], point[1]))
        ring_indices.append(indices)
    for ring_index in range(len(ring_indices) - 1):
        outer = ring_indices[ring_index]
        inner = ring_indices[ring_index + 1]
        for index in range(boundary_count):
            faces.append((outer[index], outer[(index + 1) % boundary_count], inner[(index + 1) % boundary_count], inner[index]))

    grid = _quadrant_patch_grid_from_counts(rings[-1], anchors, u_count, v_count)
    smoothing_iterations = _smoothing_iterations_for_grid(u_count, v_count)
    smoothing_factor = _env_float("HALLWAY_QUADRANT_REMESH_SMOOTH_FACTOR", _DEFAULT_SMOOTHING_FACTOR, 0.0)
    base_grid = [[point for point in row_points] for row_points in grid]
    base_folded = _grid_folded_cell_count(base_grid)
    base_crossings = _grid_projected_crossing_count(base_grid)
    smooth_moves = 0
    smoothed_folded = base_folded
    smoothed_crossings = base_crossings
    for factor_scale in (1.0, 0.5, 0.25, 0.0):
        candidate_grid = [[point for point in row_points] for row_points in base_grid]
        candidate_moves = _smooth_structured_patch_grid(candidate_grid, smoothing_iterations, min(smoothing_factor * factor_scale, 1.0))
        candidate_folded = _grid_folded_cell_count(candidate_grid)
        candidate_crossings = _grid_projected_crossing_count(candidate_grid)
        if (candidate_folded, candidate_crossings) <= (base_folded, base_crossings):
            grid = candidate_grid
            smooth_moves = candidate_moves
            smoothed_folded = candidate_folded
            smoothed_crossings = candidate_crossings
            if candidate_folded == 0 and candidate_crossings == 0:
                break
    clamped_points = _clamp_grid_to_domain(grid, [rings[-1]], edge_length * 0.08)

    columns = u_count + 1
    inner_boundary = ring_indices[-1]
    grid_index: dict[tuple[int, int], int] = {}

    def boundary_vertex_index(row: int, column: int) -> int | None:
        if row == 0:
            return inner_boundary[column]
        if column == u_count:
            return inner_boundary[u_count + row]
        if row == v_count:
            return inner_boundary[u_count + v_count + (u_count - column)]
        if column == 0:
            return inner_boundary[u_count + v_count + u_count + (v_count - row)]
        return None

    for row in range(v_count + 1):
        for column in range(u_count + 1):
            existing = boundary_vertex_index(row, column)
            if existing is not None:
                grid_index[(row, column)] = existing
            else:
                point = grid[row][column]
                grid_index[(row, column)] = len(vertices)
                vertices.append(_make_coord(plane, point[0], point[1]))
    for row in range(v_count):
        for column in range(u_count):
            faces.append(
                (
                    grid_index[(row, column)],
                    grid_index[(row, column + 1)],
                    grid_index[(row + 1, column + 1)],
                    grid_index[(row + 1, column)],
                )
            )
    if len(faces) > max_faces:
        raise StripRemeshUnsupported(f"contour-quadrant patch exceeded face cap ({max_faces})")

    mesh = bpy.data.meshes.new(f"{source_obj.data.name}_contour_quadrant_patch_remesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    removed_outside = (
        _filter_mesh_to_projected_domain(mesh, plane, [loop], edge_length * 0.25)
        if _env_bool("HALLWAY_QUADRANT_REMESH_CLIP_OUTSIDE", False)
        else 0
    )
    if removed_outside:
        logger.info("Contour-quadrant clipped %s faces outside %s", removed_outside, source_obj.name)
    if _nonmanifold_edge_count(mesh) != 0:
        bpy.data.meshes.remove(mesh)
        raise StripRemeshUnsupported("contour-quadrant patch produced non-manifold edges")
    max_face_edge = _max_face_edge_length(mesh)
    if max_face_edge > _max_allowed_patch_edge(edge_length):
        bpy.data.meshes.remove(mesh)
        raise StripRemeshUnsupported(f"contour-quadrant patch retained max_edge={max_face_edge:.4f}")
    logger.info(
        "Contour-quadrant remesh %s -> anchors=%s rings=%s grid=%sx%s verts=%s faces=%s max_edge=%.4f smoothing=%sx%.2f moves=%s clamped=%s folded=%s->%s crossings=%s->%s",
        source_obj.name,
        anchors,
        len(rings),
        u_count,
        v_count,
        len(mesh.vertices),
        len(mesh.polygons),
        max_face_edge,
        smoothing_iterations,
        min(smoothing_factor, 1.0),
        smooth_moves,
        clamped_points,
        base_folded,
        smoothed_folded,
        base_crossings,
        smoothed_crossings,
    )
    return mesh


def _build_annulus_patch_mesh(
    source_obj: bpy.types.Object,
    loops: list[list[tuple[float, float]]],
    plane: _Plane,
    edge_length: float,
    max_faces: int,
) -> bpy.types.Mesh:
    if len(loops) != 2:
        raise StripRemeshUnsupported("annulus patch requires one outer loop and one hole loop")
    outer, inner = loops
    center = _loop_centroid(inner)
    angular_count = _even_segment_count(max(_loop_perimeter(outer), _loop_perimeter(inner)), edge_length, minimum=8)
    angular_count = min(angular_count, _env_int("HALLWAY_ANNULUS_REMESH_MAX_SEGMENTS", _DEFAULT_MAX_CONTOUR_SEGMENTS, 16))
    outer_samples = _angle_resample_loop(outer, center, angular_count)
    inner_samples = _angle_resample_loop(inner, center, angular_count)
    max_radial = max(_point_distance_2d(outer_samples[index], inner_samples[index]) for index in range(angular_count))
    radial_count = _even_segment_count(max_radial, edge_length, minimum=2)
    if angular_count * radial_count > max_faces:
        radial_count = max(2, int(max_faces // max(1, angular_count)))
        if radial_count % 2 == 1:
            radial_count -= 1
    if radial_count < 2:
        raise StripRemeshUnsupported("annulus patch face budget too small")

    vertices: list[tuple[float, float, float]] = []
    for row in range(radial_count + 1):
        t = row / radial_count
        for column in range(angular_count):
            outer_point = outer_samples[column]
            inner_point = inner_samples[column]
            point = (
                outer_point[0] + ((inner_point[0] - outer_point[0]) * t),
                outer_point[1] + ((inner_point[1] - outer_point[1]) * t),
            )
            vertices.append(_make_coord(plane, point[0], point[1]))

    faces: list[tuple[int, int, int, int]] = []
    for row in range(radial_count):
        for column in range(angular_count):
            nxt = (column + 1) % angular_count
            faces.append(
                (
                    (row * angular_count) + column,
                    (row * angular_count) + nxt,
                    ((row + 1) * angular_count) + nxt,
                    ((row + 1) * angular_count) + column,
                )
            )
    mesh = bpy.data.meshes.new(f"{source_obj.data.name}_annulus_patch_remesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    removed_outside = (
        _filter_mesh_to_projected_domain(mesh, plane, loops, edge_length * 0.25)
        if _env_bool("HALLWAY_QUADRANT_REMESH_CLIP_OUTSIDE", False)
        else 0
    )
    if removed_outside:
        logger.info("Annulus patch clipped %s faces outside %s", removed_outside, source_obj.name)
    if not mesh.polygons:
        bpy.data.meshes.remove(mesh)
        raise StripRemeshUnsupported("annulus patch clipping removed every face")
    if _nonmanifold_edge_count(mesh) != 0:
        bpy.data.meshes.remove(mesh)
        raise StripRemeshUnsupported("annulus patch produced non-manifold edges")
    max_face_edge = _max_face_edge_length(mesh)
    if max_face_edge > _max_allowed_patch_edge(edge_length) * 1.5:
        bpy.data.meshes.remove(mesh)
        raise StripRemeshUnsupported(f"annulus patch retained max_edge={max_face_edge:.4f}")
    logger.info(
        "Annulus patch remesh %s -> loops=2 grid=%sx%s verts=%s faces=%s max_edge=%.4f",
        source_obj.name,
        angular_count,
        radial_count,
        len(mesh.vertices),
        len(mesh.polygons),
        max_face_edge,
    )
    return mesh


def _build_paved_mesh(
    source_obj: bpy.types.Object,
    loops: list[list[tuple[float, float]]],
    plane: _Plane,
    edge_length: float,
    *,
    _forced_layers: int | None = None,
) -> bpy.types.Mesh:
    max_faces = _env_int("HALLWAY_STRIP_REMESH_MAX_FACES", _DEFAULT_MAX_FACES, 100)
    max_segments = _env_int("HALLWAY_CONTOUR_REMESH_MAX_SEGMENTS", _DEFAULT_MAX_CONTOUR_SEGMENTS, 16)
    max_layers = _env_int("HALLWAY_CONTOUR_REMESH_MAX_LAYERS", _DEFAULT_MAX_CONTOUR_LAYERS, 0)
    contour_step = edge_length * _env_float("HALLWAY_CONTOUR_REMESH_LAYER_SCALE", 0.35, 0.05)
    if _forced_layers is None:
        loop_groups = _independent_loop_groups(loops)
        if len(loop_groups) > 1:
            meshes = [_build_paved_mesh(source_obj, group, plane, edge_length) for group in loop_groups]
            combined = _combine_meshes(f"{source_obj.data.name}_paved_components_remesh", meshes)
            logger.info(
                "Component contour remesh %s -> components=%s verts=%s faces=%s max_edge=%.4f",
                source_obj.name,
                len(loop_groups),
                len(combined.vertices),
                len(combined.polygons),
                _max_face_edge_length(combined),
            )
            return combined
    domain_loops = _normalize_domain_loops(loops)
    if not domain_loops:
        raise StripRemeshUnsupported("no contour loops")
    resampled_loops: list[list[tuple[float, float]]] = []
    for loop in domain_loops:
        resampled_loops.append(_split_long_edges_closed_loop(loop, edge_length, max_segments))

    if len(resampled_loops) == 2 and _forced_layers is None:
        try:
            return _build_annulus_patch_mesh(source_obj, resampled_loops, plane, edge_length, max_faces)
        except StripRemeshUnsupported as exc:
            logger.warning("Annulus patch remesh fallback for %s: %s", source_obj.name, exc)

    all_points = [point for loop in resampled_loops for point in loop]
    width = max(point[0] for point in all_points) - min(point[0] for point in all_points)
    height = max(point[1] for point in all_points) - min(point[1] for point in all_points)
    total_segments = sum(len(loop) for loop in resampled_loops)
    target_layers = max(1, int(math.ceil(min(width, height) / max(contour_step * 2.0, _EPS))))
    layer_count = (
        max(0, _forced_layers)
        if _forced_layers is not None
        else min(max_layers, target_layers, max(0, max_faces // max(1, total_segments)))
    )
    if len(resampled_loops) == 1 and _forced_layers is None:
        try:
            return _build_contour_quadrant_patch_mesh(source_obj, resampled_loops[0], plane, edge_length, max_faces)
        except StripRemeshUnsupported as exc:
            logger.warning("Quadrant patch remesh fallback for %s: %s", source_obj.name, exc)
    if len(resampled_loops) == 1 and len(domain_loops[0]) <= 128 and _forced_layers is None:
        try:
            inset_mesh = _build_inset_single_loop_mesh(source_obj, domain_loops[0], plane, contour_step, max_faces, layer_count)
            max_allowed_inset_edge = max(_loop_max_edge(domain_loops[0]) * 1.5, edge_length * 4.0)
            if _nonmanifold_edge_count(inset_mesh) == 0:
                if _max_face_edge_length(inset_mesh) <= max_allowed_inset_edge:
                    return inset_mesh
            bpy.data.meshes.remove(inset_mesh)
        except StripRemeshUnsupported as exc:
            logger.debug("Inset contour paving unsupported for %s: %s", source_obj.name, exc)

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    vertex_index: dict[tuple[int, int, int], int] = {}

    def add_vertex(point: tuple[float, float]) -> int:
        coord = _make_coord(plane, point[0], point[1])
        key = (round(coord[0] / 1e-7), round(coord[1] / 1e-7), round(coord[2] / 1e-7))
        existing = vertex_index.get(key)
        if existing is not None:
            return existing
        vertex_index[key] = len(vertices)
        vertices.append(coord)
        return vertex_index[key]

    ring_sets: list[list[list[tuple[float, float]]]] = []
    for loop in resampled_loops:
        rings = [loop]
        for layer_index in range(1, layer_count + 1):
            candidate = _offset_loop(loop, contour_step * layer_index)
            if not _valid_offset_loop(candidate, domain_loops):
                break
            rings.append(candidate)
        ring_sets.append(rings)

    contour_face_count = 0
    for rings in ring_sets:
        clockwise = _loop_area(rings[0]) < 0.0
        for layer_index in range(len(rings) - 1):
            outer = rings[layer_index]
            inner = rings[layer_index + 1]
            count = min(len(outer), len(inner))
            for index in range(count):
                a0 = add_vertex(outer[index])
                a1 = add_vertex(outer[(index + 1) % count])
                b1 = add_vertex(inner[(index + 1) % count])
                b0 = add_vertex(inner[index])
                face = (a0, b0, b1, a1) if clockwise else (a0, a1, b1, b0)
                if len(set(face)) == 4:
                    faces.append(face)
                    contour_face_count += 1

    fill_loops = [rings[-1] for rings in ring_sets]
    if len(fill_loops) == 1 and contour_face_count > 0:
        clockwise = _loop_area(fill_loops[0]) < 0.0
        face = tuple(add_vertex(point) for point in (fill_loops[0] if clockwise else list(reversed(fill_loops[0]))))
        if len(set(face)) >= 3:
            faces.append(face)
    elif len(fill_loops) >= 2 and contour_face_count > 0:
        fill_points, fill_faces = _quad_faces_from_tessellation(fill_loops)
        for face in fill_faces:
            remapped = tuple(add_vertex(fill_points[index]) for index in face)
            if len(set(remapped)) >= 3:
                faces.append(remapped)
    else:
        try:
            fill_mesh = _build_strip_mesh(source_obj, fill_loops, plane, edge_length)
        except StripRemeshUnsupported as exc:
            fill_mesh = None
            logger.warning("Bounded strip interior fill skipped for %s: %s", source_obj.name, exc)
        if fill_mesh is not None:
            index_map: dict[int, int] = {}
            for vertex in fill_mesh.vertices:
                index_map[vertex.index] = add_vertex((float(vertex.co[plane.across_axis]), float(vertex.co[plane.along_axis])))
            for polygon in fill_mesh.polygons:
                remapped = tuple(index_map[index] for index in polygon.vertices)
                if len(set(remapped)) >= 3:
                    faces.append(remapped)
            bpy.data.meshes.remove(fill_mesh)

    if not faces:
        fill_points, fill_faces = _quad_faces_from_tessellation(fill_loops)
        for face in fill_faces:
            remapped = tuple(add_vertex(fill_points[index]) for index in face)
            if len(set(remapped)) >= 3:
                faces.append(remapped)
    if not faces:
        raise StripRemeshUnsupported("contour paving produced no faces")
    if len(faces) > max_faces:
        raise StripRemeshUnsupported(f"contour paving exceeded face cap ({max_faces})")

    mesh = bpy.data.meshes.new(f"{source_obj.data.name}_paved_remesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    nonmanifold_edges = _nonmanifold_edge_count(mesh)
    max_face_edge = _max_face_edge_length(mesh)
    if nonmanifold_edges and _forced_layers is None and layer_count > 0:
        bpy.data.meshes.remove(mesh)
        for retry_layers in range(layer_count - 1, -1, -1):
            candidate = _build_paved_mesh(source_obj, loops, plane, edge_length, _forced_layers=retry_layers)
            candidate_nonmanifold = _nonmanifold_edge_count(candidate)
            if candidate_nonmanifold == 0:
                logger.warning(
                    "Contour-paved remesh reduced contour layers for %s from %s to %s to avoid %s non-manifold edges",
                    source_obj.name,
                    layer_count,
                    retry_layers,
                    nonmanifold_edges,
                )
                return candidate
            bpy.data.meshes.remove(candidate)
        raise StripRemeshUnsupported(f"contour paving produced {nonmanifold_edges} non-manifold edges")
    logger.info(
        "Contour-paved remesh %s -> loops=%s boundary_vertices=%s layers=%s verts=%s faces=%s max_edge=%.4f",
        source_obj.name,
        len(resampled_loops),
        total_segments,
        max(len(rings) - 1 for rings in ring_sets),
        len(vertices),
        len(faces),
        max_face_edge,
    )
    return mesh


def _build_strip_mesh(
    source_obj: bpy.types.Object,
    loops: list[list[tuple[float, float]]],
    plane: _Plane,
    edge_length: float,
) -> bpy.types.Mesh:
    all_points = [point for loop in loops for point in loop]
    min_across = min(point[0] for point in all_points)
    max_across = max(point[0] for point in all_points)
    if max_across - min_across <= _EPS:
        raise StripRemeshUnsupported("degenerate strip width")

    segments = _segments(loops)
    max_faces = _env_int("HALLWAY_STRIP_REMESH_MAX_FACES", _DEFAULT_MAX_FACES, 100)
    min_along = min(point[1] for point in all_points)
    max_along = max(point[1] for point in all_points)
    across_count, along_count = _budgeted_grid_counts(max_across - min_across, max_along - min_along, edge_length, max_faces)
    levels = [min_across + ((max_across - min_across) * index / across_count) for index in range(across_count + 1)]
    row_height = (max_along - min_along) / max(1, along_count)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    vertex_index: dict[tuple[int, int], int] = {}

    def add_vertex(across: float, along: float) -> int:
        key = (round(across / 1e-7), round(along / 1e-7))
        existing = vertex_index.get(key)
        if existing is not None:
            return existing
        vertex_index[key] = len(vertices)
        vertices.append(_make_coord(plane, across, along))
        return vertex_index[key]

    for index in range(len(levels) - 1):
        left = levels[index]
        right = levels[index + 1]
        if right - left <= max(1e-7, edge_length * 0.02):
            continue
        mid = (left + right) * 0.5
        intervals = _inside_intervals_at(segments, mid)
        for inside_min, inside_max in intervals:
            if inside_max - inside_min <= 1e-6:
                continue
            first_row = max(0, int(math.floor((inside_min - min_along) / row_height)))
            last_row = min(along_count - 1, int(math.ceil((inside_max - min_along) / row_height)) - 1)
            for row in range(first_row, last_row + 1):
                row_min = min_along + (row * row_height)
                row_max = min_along + ((row + 1) * row_height)
                bottom = max(row_min, inside_min)
                top = min(row_max, inside_max)
                if top - bottom <= 1e-6:
                    continue
                face = (
                    add_vertex(left, bottom),
                    add_vertex(right, bottom),
                    add_vertex(right, top),
                    add_vertex(left, top),
                )
                if len(set(face)) == 4:
                    faces.append(face)
                    if len(faces) > max_faces:
                        raise StripRemeshUnsupported(f"strip remesh exceeded face cap ({max_faces}); strips={across_count} rows={along_count}")

    if not faces:
        raise StripRemeshUnsupported("strip remesh produced no faces")
    mesh = bpy.data.meshes.new(f"{source_obj.data.name}_strip_remesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    return mesh


def remesh_object(
    context: bpy.types.Context,
    source_obj: bpy.types.Object,
    *,
    scale_factor: float = 10.0,
    enabled: bool = True,
) -> bpy.types.Object | None:
    if not enabled or source_obj.type != "MESH" or len(source_obj.data.polygons) == 0:
        return None
    mesh = source_obj.data
    started_at = time.monotonic()
    transform = _source_rotation_scale_matrix(source_obj)
    coords = [transform @ vertex.co for vertex in mesh.vertices]
    plane = _mesh_plane(coords)
    raw_loops = [_project_loop(coords, loop, plane) for loop in _boundary_loops(mesh)]
    edge_length = _target_edge_length(scale_factor)
    loops = [loop for loop in raw_loops if len(loop) >= 3]
    if not loops:
        raise StripRemeshUnsupported("no usable boundary loops")
    logger.info(
        "Strip remesh input %s -> raw_loop_vertices=%s simplified_loop_vertices=%s edge_length=%.4f",
        source_obj.name,
        sum(len(loop) for loop in raw_loops),
        sum(len(loop) for loop in loops),
        edge_length,
    )
    strip_mesh = _build_paved_mesh(source_obj, loops, plane, edge_length)
    new_obj = bpy.data.objects.new(f"{source_obj.name}__hallway_strip_remesh", strip_mesh)
    new_obj.location = source_obj.location.copy()
    try:
        inverse_transform = transform.inverted()
        new_obj["hallway_avatar_strip_source_rs_inverse"] = [float(value) for row in inverse_transform for value in row]
    except ValueError:
        pass
    for collection in list(source_obj.users_collection) or [context.scene.collection]:
        collection.objects.link(new_obj)
    logger.info(
        "Strip remeshed %s -> loops=%s verts=%s faces=%s edge_length=%.4f elapsed=%.3fs",
        source_obj.name,
        len(loops),
        len(strip_mesh.vertices),
        len(strip_mesh.polygons),
        edge_length,
        time.monotonic() - started_at,
    )
    return new_obj
