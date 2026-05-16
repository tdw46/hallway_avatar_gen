from __future__ import annotations

from math import ceil, floor, hypot

import bpy

from .models import BonePlan, LayerPart, RigPlan
from . import seethrough_naming
from ..utils.logging import get_logger


logger = get_logger("heuristic_rigger")


def _visible_parts(parts: list[LayerPart]) -> list[LayerPart]:
    return [part for part in parts if not part.skipped]


def _union_bbox(parts: list[LayerPart], labels: set[str] | None = None) -> tuple[float, float, float, float] | None:
    selected = [part for part in parts if labels is None or part.semantic_label in labels]
    if not selected:
        return None
    return (
        min(part.alpha_bbox[0] for part in selected),
        min(part.alpha_bbox[1] for part in selected),
        max(part.alpha_bbox[2] for part in selected),
        max(part.alpha_bbox[3] for part in selected),
    )


def _pixel_to_plane(x: float, y: float, canvas_size: tuple[int, int], import_scale: float = 1.0) -> tuple[float, float, float]:
    scale = 2.0 / max(1.0, float(max(canvas_size)))
    canvas_w, canvas_h = canvas_size
    return ((x - canvas_w * 0.5) * scale * import_scale, 0.0, (canvas_h * 0.5 - y) * scale * import_scale)


def _center_of_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


NECK_TOKENS = {"neck", "neckwear"}
HEAD_TOKENS = {
    "face",
    "headwear",
    "nose",
    "mouth",
    "eyewhite",
    "eyelash",
    "eyebrow",
    "eyewear",
    "ears",
    "earwear",
}
IRIS_TOKENS = {"irides"}
BODY_UPWARD_BONES = {"root", "hips", "torso", "spine", "neck", "head", "eyes"}
DOWNWARD_BONES = {
    "front_hair",
    "back_hair",
    "leftArm",
    "rightArm",
    "leftElbow",
    "rightElbow",
    "leftHand",
    "rightHand",
    "bothArms",
    "leftLeg",
    "rightLeg",
    "leftKnee",
    "rightKnee",
    "leftFoot",
    "rightFoot",
    "bothLegs",
}
HAIR_SEGMENT_MIN_FACE_RATIO = 0.75
HAIR_SEGMENT_MAX_FACE_RATIO = 1.125
BACK_HAIR_SEGMENT_DENSITY_MULTIPLIER = 1.5
DEFAULT_BONE_COLLECTIONS = ("Body", "Face", "Hair", "Arms", "Legs", "Objects")
SILHOUETTE_SNAP_EPSILON = 1.0e-6


def _is_body_upward_bone(name: str) -> bool:
    return name in BODY_UPWARD_BONES


def _is_downward_bone(name: str) -> bool:
    return name in DOWNWARD_BONES or name.startswith("front_hair_") or name.startswith("back_hair_")


def _canonical_token(part: LayerPart) -> str:
    token, _, _ = seethrough_naming.classify_name(part.layer_name, part.layer_path)
    return token


def _stretchy_side_from_x(x_value: float, centerline_x: float) -> str:
    if x_value > centerline_x:
        return "LEFT"
    if x_value < centerline_x:
        return "RIGHT"
    return "CENTER"


def _stretchy_side(part: LayerPart, centerline_x: float) -> str:
    return _stretchy_side_from_x(part.centroid[0], centerline_x)


def _first_bbox(parts: list[LayerPart], token: str, *, side: str | None = None, centerline_x: float | None = None) -> tuple[float, float, float, float] | None:
    for part in parts:
        if _canonical_token(part) != token:
            continue
        if side is not None:
            if centerline_x is None:
                continue
            if _stretchy_side(part, centerline_x) != side:
                continue
        return tuple(float(value) for value in part.alpha_bbox)
    return None


def analyze_groups(parts: list[LayerPart]) -> dict[str, str | bool]:
    visible = _visible_parts(parts)
    if not visible:
        return {
            "head": False,
            "torso": False,
            "hips": False,
            "arms": "missing",
            "legs": "missing",
            "feet": "missing",
        }

    centerline_x = _center_of_bbox(_union_bbox(visible))[0]

    def has_token(token: str) -> bool:
        return any(_canonical_token(part) == token for part in visible)

    def split_state(base_token: str) -> str:
        relevant = [part for part in visible if _canonical_token(part) == base_token]
        if not relevant:
            return "missing"
        left = any(_stretchy_side(part, centerline_x) == "LEFT" for part in relevant)
        right = any(_stretchy_side(part, centerline_x) == "RIGHT" for part in relevant)
        if len(relevant) >= 2 and left and right:
            return "split"
        if left ^ right:
            return "partial"
        return "merged"

    return {
        "head": any(_canonical_token(part) in {"face", "front hair", "back hair", "headwear"} for part in visible),
        "torso": has_token("topwear") or has_token("neckwear"),
        "hips": has_token("bottomwear"),
        "arms": split_state("handwear"),
        "legs": split_state("legwear"),
        "feet": split_state("footwear"),
    }


def _has_token_side(parts: list[LayerPart], token: str, side: str, centerline_x: float) -> bool:
    return any(_canonical_token(part) == token and _stretchy_side(part, centerline_x) == side for part in parts)


def _estimate_keypoints(parts: list[LayerPart]) -> tuple[dict[str, tuple[float, float]], float]:
    visible = _visible_parts(parts)
    overall_bbox = _union_bbox(visible)
    centerline_x = _center_of_bbox(overall_bbox)[0]
    canvas_w = float(visible[0].canvas_size[0])
    canvas_h = float(visible[0].canvas_size[1])

    tag_bboxes: dict[str, tuple[float, float, float, float]] = {}
    for part in visible:
        token = _canonical_token(part)
        if not token or token in tag_bboxes:
            continue
        tag_bboxes[token] = tuple(float(value) for value in part.alpha_bbox)

    def get_bbox(tag: str) -> tuple[float, float, float, float] | None:
        return tag_bboxes.get(tag)

    def bbox_stats(bbox: tuple[float, float, float, float]) -> dict[str, float]:
        x0, y0, x1, y1 = bbox
        return {
            "x": x0,
            "y": y0,
            "w": max(1.0, x1 - x0),
            "h": max(1.0, y1 - y0),
            "cx": (x0 + x1) * 0.5,
            "cy": (y0 + y1) * 0.5,
            "x2": x1,
            "y2": y1,
        }

    def first_of(tags: tuple[str, ...]) -> dict[str, float] | None:
        for tag in tags:
            bbox = get_bbox(tag)
            if bbox is not None:
                return bbox_stats(bbox)
        return None

    keypoints: dict[str, tuple[float, float]] = {}

    face = first_of(("face", "front hair", "headwear"))
    if face:
        keypoints["nose"] = (face["cx"], face["cy"] + face["h"] * 0.08)
        keypoints["lEye"] = (face["cx"] + face["w"] * 0.18, face["cy"] - face["h"] * 0.05)
        keypoints["rEye"] = (face["cx"] - face["w"] * 0.18, face["cy"] - face["h"] * 0.05)
        keypoints["midEye"] = (face["cx"], face["cy"] - face["h"] * 0.05)
        keypoints["lEar"] = (face["cx"] + face["w"] * 0.45, face["cy"])
        keypoints["rEar"] = (face["cx"] - face["w"] * 0.45, face["cy"])
        keypoints["headBase"] = (face["cx"], face["y2"])

    topwear_bbox = get_bbox("topwear")
    topwear = bbox_stats(topwear_bbox) if topwear_bbox else None
    if topwear:
        keypoints["neck"] = (topwear["cx"], topwear["y"])
        keypoints["lShoulder"] = (topwear["x"] + topwear["w"] * 0.85, topwear["y"] + topwear["h"] * 0.12)
        keypoints["rShoulder"] = (topwear["x"] + topwear["w"] * 0.15, topwear["y"] + topwear["h"] * 0.12)
        keypoints["shoulderMid"] = (topwear["cx"], topwear["y"] + topwear["h"] * 0.12)
        keypoints["spine"] = (topwear["cx"], topwear["cy"])
        keypoints["waist"] = (topwear["cx"], topwear["y"] + topwear["h"] * 0.85)
    elif face:
        keypoints["neck"] = (face["cx"], face["y2"])
        keypoints["lShoulder"] = (face["cx"] - face["w"] * 0.2, face["y2"])
        keypoints["rShoulder"] = (face["cx"] + face["w"] * 0.2, face["y2"])
        keypoints["shoulderMid"] = (face["cx"], face["y2"])
        keypoints["spine"] = (face["cx"], face["y2"])
        keypoints["waist"] = (face["cx"], face["y2"])

    hand_left_bbox = _first_bbox(visible, "handwear", side="LEFT", centerline_x=centerline_x) or get_bbox("handwear")
    hand_right_bbox = _first_bbox(visible, "handwear", side="RIGHT", centerline_x=centerline_x) or get_bbox("handwear")
    hand_left = bbox_stats(hand_left_bbox) if hand_left_bbox else None
    hand_right = bbox_stats(hand_right_bbox) if hand_right_bbox else None
    if "lShoulder" in keypoints and hand_left:
        keypoints["lWrist"] = (hand_left["cx"], hand_left["y"] + hand_left["h"] * 0.1)
        keypoints["lElbow"] = (
            (keypoints["lShoulder"][0] + keypoints["lWrist"][0]) * 0.5,
            (keypoints["lShoulder"][1] + keypoints["lWrist"][1]) * 0.5,
        )
    if "rShoulder" in keypoints and hand_right:
        keypoints["rWrist"] = (hand_right["cx"], hand_right["y"] + hand_right["h"] * 0.1)
        keypoints["rElbow"] = (
            (keypoints["rShoulder"][0] + keypoints["rWrist"][0]) * 0.5,
            (keypoints["rShoulder"][1] + keypoints["rWrist"][1]) * 0.5,
        )

    bottomwear_bbox = get_bbox("bottomwear")
    bottomwear = bbox_stats(bottomwear_bbox) if bottomwear_bbox else None
    if bottomwear:
        keypoints["pelvis"] = (bottomwear["cx"], bottomwear["cy"])
        keypoints["lHip"] = (bottomwear["cx"] + bottomwear["w"] * 0.2, bottomwear["y"] + bottomwear["h"] * 0.15)
        keypoints["rHip"] = (bottomwear["cx"] - bottomwear["w"] * 0.2, bottomwear["y"] + bottomwear["h"] * 0.15)
    elif "waist" in keypoints:
        keypoints["pelvis"] = (keypoints["waist"][0], keypoints["waist"][1] + canvas_h * 0.08)
        keypoints["lHip"] = (keypoints["pelvis"][0] + canvas_w * 0.1, keypoints["pelvis"][1])
        keypoints["rHip"] = (keypoints["pelvis"][0] - canvas_w * 0.1, keypoints["pelvis"][1])

    leg_left_bbox = _first_bbox(visible, "legwear", side="LEFT", centerline_x=centerline_x) or get_bbox("legwear")
    leg_right_bbox = _first_bbox(visible, "legwear", side="RIGHT", centerline_x=centerline_x) or get_bbox("legwear")
    foot_left_bbox = _first_bbox(visible, "footwear", side="LEFT", centerline_x=centerline_x) or get_bbox("footwear")
    foot_right_bbox = _first_bbox(visible, "footwear", side="RIGHT", centerline_x=centerline_x) or get_bbox("footwear")
    leg_left = bbox_stats(leg_left_bbox) if leg_left_bbox else None
    leg_right = bbox_stats(leg_right_bbox) if leg_right_bbox else None
    foot_left = bbox_stats(foot_left_bbox) if foot_left_bbox else None
    foot_right = bbox_stats(foot_right_bbox) if foot_right_bbox else None
    if "lHip" in keypoints and leg_left:
        ankle = (foot_left["cx"], foot_left["cy"]) if foot_left else (leg_left["cx"], leg_left["y2"])
        keypoints["lAnkle"] = ankle
        keypoints["lKnee"] = ((keypoints["lHip"][0] + ankle[0]) * 0.5, (keypoints["lHip"][1] + ankle[1]) * 0.5)
    if "rHip" in keypoints and leg_right:
        ankle = (foot_right["cx"], foot_right["cy"]) if foot_right else (leg_right["cx"], leg_right["y2"])
        keypoints["rAnkle"] = ankle
        keypoints["rKnee"] = ((keypoints["rHip"][0] + ankle[0]) * 0.5, (keypoints["rHip"][1] + ankle[1]) * 0.5)

    cx = canvas_w * 0.5
    cy = canvas_h * 0.5
    keypoints.setdefault("pelvis", (cx, cy))
    keypoints.setdefault("neck", (cx, canvas_h * 0.25))
    keypoints.setdefault("headBase", (cx, canvas_h * 0.22))
    keypoints.setdefault("lShoulder", (cx + canvas_w * 0.15, canvas_h * 0.30))
    keypoints.setdefault("rShoulder", (cx - canvas_w * 0.15, canvas_h * 0.30))
    keypoints.setdefault("shoulderMid", (cx, canvas_h * 0.30))
    keypoints.setdefault("waist", (cx, canvas_h * 0.55))
    keypoints.setdefault("spine", (cx, canvas_h * 0.42))
    keypoints.setdefault("lHip", (cx + canvas_w * 0.1, canvas_h * 0.58))
    keypoints.setdefault("rHip", (cx - canvas_w * 0.1, canvas_h * 0.58))
    keypoints.setdefault("lElbow", (cx + canvas_w * 0.22, canvas_h * 0.42))
    keypoints.setdefault("rElbow", (cx - canvas_w * 0.22, canvas_h * 0.42))
    keypoints.setdefault("lWrist", (cx + canvas_w * 0.26, canvas_h * 0.55))
    keypoints.setdefault("rWrist", (cx - canvas_w * 0.26, canvas_h * 0.55))
    keypoints.setdefault("lKnee", (cx + canvas_w * 0.1, canvas_h * 0.74))
    keypoints.setdefault("rKnee", (cx - canvas_w * 0.1, canvas_h * 0.74))
    keypoints.setdefault("lAnkle", (cx + canvas_w * 0.1, canvas_h * 0.90))
    keypoints.setdefault("rAnkle", (cx - canvas_w * 0.1, canvas_h * 0.90))
    keypoints.setdefault("midEye", (cx, canvas_h * 0.18))
    return keypoints, centerline_x


def _layer_bone_for_part(part: LayerPart, groups: dict[str, str | bool], centerline_x: float) -> str:
    token = _canonical_token(part)
    if token == "front hair":
        return "front_hair"
    if token == "back hair":
        return "back_hair"
    if token in IRIS_TOKENS:
        return "eyes"
    if token in NECK_TOKENS:
        return "neck"
    if token in HEAD_TOKENS:
        return "head"
    if token == "topwear":
        return "torso"
    if token == "bottomwear":
        return "hips"
    if token == "handwear":
        side = _stretchy_side(part, centerline_x)
        if groups["arms"] == "merged":
            return "bothArms"
        if side == "LEFT":
            return "leftArm"
        if side == "RIGHT":
            return "rightArm"
    if token in {"legwear", "footwear"}:
        side = _stretchy_side(part, centerline_x)
        if groups["legs"] == "merged":
            return "bothLegs"
        if side == "LEFT":
            return "leftLeg"
        if side == "RIGHT":
            return "rightLeg"
    return "root"


def _tail_target(
    bone_name: str,
    keypoints: dict[str, tuple[float, float]],
    canvas_size: tuple[int, int],
) -> tuple[float, float]:
    x_value, y_value = keypoints[bone_name]
    default_offset = max(12.0, canvas_size[1] * 0.025)
    targets = {
        "root": keypoints.get("hips"),
        "hips": keypoints.get("spine"),
        "spine": keypoints.get("torso"),
        "torso": keypoints.get("neck"),
        "neck": keypoints.get("head"),
        "head": keypoints.get("eyes"),
        "eyes": (x_value, y_value - default_offset),
        "front_hair": keypoints.get("front_hair_tip"),
        "back_hair": keypoints.get("back_hair_tip"),
        "leftArm": keypoints.get("lElbow") or keypoints.get("lWrist"),
        "rightArm": keypoints.get("rElbow") or keypoints.get("rWrist"),
        "leftElbow": keypoints.get("lWrist"),
        "rightElbow": keypoints.get("rWrist"),
        "leftHand": (
            keypoints.get("lWrist", (x_value, y_value))[0],
            keypoints.get("lWrist", (x_value, y_value))[1] + default_offset,
        ),
        "rightHand": (
            keypoints.get("rWrist", (x_value, y_value))[0],
            keypoints.get("rWrist", (x_value, y_value))[1] + default_offset,
        ),
        "bothArms": (
            (
                keypoints.get("lWrist", keypoints["bothArms"])[0]
                + keypoints.get("rWrist", keypoints["bothArms"])[0]
            ) * 0.5,
            (
                keypoints.get("lWrist", keypoints["bothArms"])[1]
                + keypoints.get("rWrist", keypoints["bothArms"])[1]
            ) * 0.5,
        ) if "bothArms" in keypoints else None,
        "leftLeg": keypoints.get("lKnee") or keypoints.get("lAnkle"),
        "rightLeg": keypoints.get("rKnee") or keypoints.get("rAnkle"),
        "leftKnee": keypoints.get("lAnkle"),
        "rightKnee": keypoints.get("rAnkle"),
        "leftFoot": (
            keypoints.get("lAnkle", (x_value, y_value))[0],
            keypoints.get("lAnkle", (x_value, y_value))[1] + default_offset,
        ),
        "rightFoot": (
            keypoints.get("rAnkle", (x_value, y_value))[0],
            keypoints.get("rAnkle", (x_value, y_value))[1] + default_offset,
        ),
        "bothLegs": (
            (
                keypoints.get("lAnkle", keypoints["bothLegs"])[0]
                + keypoints.get("rAnkle", keypoints["bothLegs"])[0]
            ) * 0.5,
            (
                keypoints.get("lAnkle", keypoints["bothLegs"])[1]
                + keypoints.get("rAnkle", keypoints["bothLegs"])[1]
            ) * 0.5,
        ) if "bothLegs" in keypoints else None,
    }
    target = targets.get(bone_name)
    if target is None:
        return (x_value, y_value - default_offset)
    if abs(target[0] - x_value) < 1e-5 and abs(target[1] - y_value) < 1e-5:
        return (x_value, y_value - default_offset)
    return target


def _subdivide_chain(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    segments: int,
) -> list[tuple[float, float]]:
    if segments <= 0:
        return [start_xy, end_xy]
    return [
        (
            start_xy[0] + (end_xy[0] - start_xy[0]) * (index / segments),
            start_xy[1] + (end_xy[1] - start_xy[1]) * (index / segments),
        )
        for index in range(segments + 1)
    ]


def _distance_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(b[0] - a[0], b[1] - a[1])


def _bbox_for_tokens(parts: list[LayerPart], tokens: set[str]) -> tuple[float, float, float, float] | None:
    bboxes = [
        part.alpha_bbox
        for part in parts
        if _canonical_token(part) in tokens
        and part.alpha_bbox[2] > part.alpha_bbox[0]
        and part.alpha_bbox[3] > part.alpha_bbox[1]
    ]
    if not bboxes:
        return None
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _bbox_center_x(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[0] + bbox[2]) * 0.5


def _apply_bust_body_keypoints(
    keypoints: dict[str, tuple[float, float]],
    visible: list[LayerPart],
    centerline_x: float,
    canvas_size: tuple[int, int],
) -> None:
    canvas_h = float(canvas_size[1])
    min_gap = max(8.0, canvas_h * 0.015)
    head_base = keypoints.get("headBase", (centerline_x, canvas_h * 0.34))

    neck_bbox = _bbox_for_tokens(visible, NECK_TOKENS)
    topwear_bbox = _bbox_for_tokens(visible, {"topwear"})

    body_x = (
        _bbox_center_x(topwear_bbox)
        if topwear_bbox is not None
        else (_bbox_center_x(neck_bbox) if neck_bbox is not None else head_base[0])
    )
    neck_bottom_y = neck_bbox[3] if neck_bbox is not None else head_base[1] + canvas_h * 0.08
    torso_bottom_y = topwear_bbox[3] if topwear_bbox is not None else neck_bottom_y + canvas_h * 0.18

    neck_bottom_y = max(neck_bottom_y, head_base[1] + min_gap)
    neck_mid_y = head_base[1] + ((neck_bottom_y - head_base[1]) * 0.5)
    torso_bottom_y = max(torso_bottom_y, neck_bottom_y + min_gap)
    torso_length = max(torso_bottom_y - neck_mid_y, canvas_h * 0.10)
    spine_bottom_y = torso_bottom_y + torso_length
    hips_bottom_y = spine_bottom_y + torso_length

    keypoints["neck"] = (body_x, neck_mid_y)
    keypoints["torso"] = (body_x, torso_bottom_y)
    keypoints["spine"] = (body_x, spine_bottom_y)
    keypoints["hips"] = (body_x, hips_bottom_y)
    keypoints["pelvis"] = keypoints["hips"]
    keypoints["waist"] = keypoints["torso"]
    keypoints["root"] = (body_x, hips_bottom_y + torso_length * 0.5)


def _loop_area(loop: list[tuple[float, float]]) -> float:
    area = 0.0
    if len(loop) < 3:
        return area
    for index, point in enumerate(loop):
        next_point = loop[(index + 1) % len(loop)]
        area += (point[0] * next_point[1]) - (next_point[0] * point[1])
    return area * 0.5


def _point_in_loop(point: tuple[float, float], loop: list[tuple[float, float]]) -> bool:
    if len(loop) < 3:
        return False
    px, pz = point
    inside = False
    previous = loop[-1]
    for current in loop:
        x0, z0 = previous
        x1, z1 = current
        crosses = (z0 > pz) != (z1 > pz)
        if crosses:
            at_x = x0 + ((pz - z0) * (x1 - x0) / (z1 - z0))
            if px < at_x:
                inside = not inside
        previous = current
    return inside


def _point_in_silhouette(point: tuple[float, float], loops: list[list[tuple[float, float]]]) -> bool:
    return any(_point_in_loop(point, loop) for loop in loops)


def _clean_world_loop(loop: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for point in loop:
        if cleaned and abs(cleaned[-1][0] - point[0]) <= SILHOUETTE_SNAP_EPSILON and abs(cleaned[-1][1] - point[1]) <= SILHOUETTE_SNAP_EPSILON:
            continue
        cleaned.append(point)
    if len(cleaned) > 1 and abs(cleaned[0][0] - cleaned[-1][0]) <= SILHOUETTE_SNAP_EPSILON and abs(cleaned[0][1] - cleaned[-1][1]) <= SILHOUETTE_SNAP_EPSILON:
        cleaned.pop()
    return cleaned if len(cleaned) >= 3 and abs(_loop_area(cleaned)) > SILHOUETTE_SNAP_EPSILON else []


def _world_silhouette_loops(part: LayerPart) -> list[list[tuple[float, float]]]:
    if not part.imported_object_name:
        return []
    obj = bpy.data.objects.get(part.imported_object_name)
    if obj is None or obj.type != "MESH" or obj.data is None:
        return []

    edge_counts: dict[tuple[int, int], int] = {}
    for polygon in obj.data.polygons:
        vertices = list(polygon.vertices)
        for index, vertex_index in enumerate(vertices):
            next_index = vertices[(index + 1) % len(vertices)]
            edge = (vertex_index, next_index) if vertex_index < next_index else (next_index, vertex_index)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    adjacency: dict[int, list[int]] = {}
    for (vertex_a, vertex_b), count in edge_counts.items():
        if count != 1:
            continue
        adjacency.setdefault(vertex_a, []).append(vertex_b)
        adjacency.setdefault(vertex_b, []).append(vertex_a)
    if not adjacency:
        return []

    loops: list[list[tuple[float, float]]] = []
    visited: set[tuple[int, int]] = set()
    for start in sorted(adjacency):
        for second in sorted(adjacency[start]):
            edge = (start, second) if start < second else (second, start)
            if edge in visited:
                continue
            vertex_loop = [start]
            previous = start
            current = second
            while True:
                edge = (previous, current) if previous < current else (current, previous)
                if edge in visited:
                    break
                visited.add(edge)
                vertex_loop.append(current)
                neighbors = sorted(adjacency.get(current, []))
                next_candidates = [vertex for vertex in neighbors if vertex != previous]
                if not next_candidates:
                    break
                next_vertex = next_candidates[0]
                if next_vertex == start:
                    visited.add((current, next_vertex) if current < next_vertex else (next_vertex, current))
                    break
                previous, current = current, next_vertex

            if len(vertex_loop) < 3:
                continue
            world_loop = []
            for vertex_index in vertex_loop:
                world_co = obj.matrix_world @ obj.data.vertices[vertex_index].co
                world_loop.append((float(world_co.x), float(world_co.z)))
            cleaned = _clean_world_loop(world_loop)
            if cleaned:
                loops.append(cleaned)

    sorted_loops = sorted(loops, key=lambda loop: abs(_loop_area(loop)), reverse=True)
    outer_loops: list[list[tuple[float, float]]] = []
    for loop in sorted_loops:
        sample = loop[0]
        if any(_point_in_loop(sample, outer_loop) for outer_loop in outer_loops):
            continue
        outer_loops.append(loop)
    return outer_loops


def _import_scale_from_visible_parts(parts: list[LayerPart]) -> float:
    scales: list[float] = []
    for part in parts:
        if part.skipped or not part.imported_object_name:
            continue
        obj = bpy.data.objects.get(part.imported_object_name)
        if obj is None:
            continue
        scale = float(obj.get("hallway_avatar_import_scale", 1.0) or 1.0)
        if scale > 0.0:
            scales.append(scale)
    if not scales:
        return 1.0
    return sum(scales) / len(scales)


def _scanline_intervals(
    loops: list[list[tuple[float, float]]],
    fixed_value: float,
    *,
    horizontal: bool,
) -> list[tuple[float, float]]:
    hits: list[float] = []
    for loop in loops:
        for index, start in enumerate(loop):
            end = loop[(index + 1) % len(loop)]
            if horizontal:
                fixed_start, fixed_end = start[1], end[1]
                varying_start, varying_end = start[0], end[0]
            else:
                fixed_start, fixed_end = start[0], end[0]
                varying_start, varying_end = start[1], end[1]

            delta = fixed_end - fixed_start
            if abs(delta) <= SILHOUETTE_SNAP_EPSILON:
                continue
            low = min(fixed_start, fixed_end)
            high = max(fixed_start, fixed_end)
            if not (low <= fixed_value < high):
                continue
            t = (fixed_value - fixed_start) / delta
            hits.append(varying_start + ((varying_end - varying_start) * t))

    hits.sort()
    unique: list[float] = []
    for value in hits:
        if not unique or abs(unique[-1] - value) > 1.0e-5:
            unique.append(value)
    if len(unique) % 2 == 1:
        unique = unique[:-1]
    return [
        (unique[index], unique[index + 1])
        for index in range(0, len(unique), 2)
        if unique[index + 1] - unique[index] > SILHOUETTE_SNAP_EPSILON
    ]


def _nearest_value_inside_intervals(value: float, intervals: list[tuple[float, float]]) -> float | None:
    if not intervals:
        return None
    best_value: float | None = None
    best_distance: float | None = None
    for start, end in intervals:
        width = end - start
        inset = min(max(width * 0.30, 1.0e-5), width * 0.45)
        if start <= value <= end:
            candidate = (start + end) * 0.5
        elif value < start:
            candidate = start + inset
        else:
            candidate = end - inset
        distance = abs(candidate - value)
        if best_distance is None or distance < best_distance:
            best_value = candidate
            best_distance = distance
    return best_value


def _snap_world_point_to_silhouette(
    point: tuple[float, float],
    loops: list[list[tuple[float, float]]],
) -> tuple[float, float]:
    if not loops or _point_in_silhouette(point, loops):
        return point

    x_value, z_value = point
    horizontal_x = _nearest_value_inside_intervals(
        x_value,
        _scanline_intervals(loops, z_value, horizontal=True),
    )
    if horizontal_x is not None:
        candidate = (horizontal_x, z_value)
        if _point_in_silhouette(candidate, loops):
            return candidate

    vertical_z = _nearest_value_inside_intervals(
        z_value,
        _scanline_intervals(loops, x_value, horizontal=False),
    )
    if vertical_z is not None:
        candidate = (x_value, vertical_z)
        if _point_in_silhouette(candidate, loops):
            return candidate

    z_values = [point_z for loop in loops for _, point_z in loop]
    if not z_values:
        return point
    min_z = min(z_values)
    max_z = max(z_values)
    samples = 32
    best_candidate: tuple[float, float] | None = None
    best_distance: float | None = None
    for sample_index in range(samples + 1):
        scan_z = min_z + ((max_z - min_z) * (sample_index / max(1, samples)))
        scan_x = _nearest_value_inside_intervals(
            x_value,
            _scanline_intervals(loops, scan_z, horizontal=True),
        )
        if scan_x is None:
            continue
        candidate = (scan_x, scan_z)
        if not _point_in_silhouette(candidate, loops):
            continue
        distance = hypot(candidate[0] - x_value, candidate[1] - z_value)
        if best_distance is None or distance < best_distance:
            best_candidate = candidate
            best_distance = distance
    return best_candidate or point


def _snap_pixel_to_part_silhouette(
    point: tuple[float, float],
    loops: list[list[tuple[float, float]]],
    canvas_size: tuple[int, int],
    ground_offset_z: float,
    import_scale: float,
    *,
    desired_x: float | None = None,
) -> tuple[float, float]:
    if not loops:
        return point
    plane_point = _pixel_to_plane(point[0], point[1], canvas_size, import_scale)
    desired_plane_x = _pixel_to_plane(desired_x, point[1], canvas_size, import_scale)[0] if desired_x is not None else plane_point[0]
    world_point = (desired_plane_x, plane_point[2] + ground_offset_z)
    snapped = _snap_world_point_to_silhouette(world_point, loops)
    original_world_point = (plane_point[0], plane_point[2] + ground_offset_z)
    if abs(snapped[0] - original_world_point[0]) <= SILHOUETTE_SNAP_EPSILON and abs(snapped[1] - original_world_point[1]) <= SILHOUETTE_SNAP_EPSILON:
        return point
    return _plane_to_pixel(snapped[0], snapped[1] - ground_offset_z, canvas_size, import_scale)


def _snap_pixel_chain_to_part_silhouette(
    points: list[tuple[float, float]],
    part: LayerPart | None,
    canvas_size: tuple[int, int],
    ground_offset_z: float,
    import_scale: float,
    *,
    label: str,
    desired_x_values: list[float | None] | None = None,
) -> list[tuple[float, float]]:
    if part is None:
        return points
    loops = _world_silhouette_loops(part)
    if not loops:
        return points
    snapped_points = [
        _snap_pixel_to_part_silhouette(
            point,
            loops,
            canvas_size,
            ground_offset_z,
            import_scale,
            desired_x=desired_x_values[index] if desired_x_values is not None and index < len(desired_x_values) else None,
        )
        for index, point in enumerate(points)
    ]
    adjusted = sum(
        1
        for original, snapped in zip(points, snapped_points, strict=False)
        if abs(original[0] - snapped[0]) > 1.0e-4 or abs(original[1] - snapped[1]) > 1.0e-4
    )
    if adjusted:
        logger.info("Snapped %s hair rig points into mesh silhouette -> adjusted=%s/%s", label, adjusted, len(points))
    return snapped_points


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) * 0.5


def _plane_to_pixel(x: float, z: float, canvas_size: tuple[int, int], import_scale: float = 1.0) -> tuple[float, float]:
    scale = 2.0 / max(1.0, float(max(canvas_size)))
    effective_scale = scale * max(float(import_scale), 1.0e-6)
    canvas_w, canvas_h = canvas_size
    return ((x / effective_scale) + canvas_w * 0.5, canvas_h * 0.5 - (z / effective_scale))


def _hair_chain_length(total_length: float, face_bone_length: float, density_multiplier: float = 1.0) -> int:
    if total_length <= 1e-6:
        return 1

    reference_length = max(face_bone_length, 1.0)
    min_segment = reference_length * HAIR_SEGMENT_MIN_FACE_RATIO
    max_segment = reference_length * HAIR_SEGMENT_MAX_FACE_RATIO
    if min_segment <= 1e-6 or max_segment <= 1e-6:
        return 1

    min_segments = max(1, ceil(total_length / max_segment))
    max_segments = max(1, floor(total_length / min_segment))
    target_segment = (min_segment + max_segment) * 0.5
    target_segments = max(1, round(total_length / max(target_segment, 1e-6)))

    if min_segments <= max_segments:
        segments = max(min_segments, min(target_segments, max_segments))
    elif total_length < min_segment:
        segments = 1
    else:
        segments = min_segments
    return max(1, ceil(segments * max(density_multiplier, 1.0)))


def _bone_collection_name(name: str) -> str:
    if name.startswith("front_hair_") or name.startswith("back_hair_") or name in {"front_hair", "back_hair"}:
        return "Hair"
    if name in {"head", "eyes"}:
        return "Face"
    if name in {"leftArm", "rightArm", "leftElbow", "rightElbow", "leftHand", "rightHand", "bothArms"}:
        return "Arms"
    if name in {"leftLeg", "rightLeg", "leftKnee", "rightKnee", "leftFoot", "rightFoot", "bothLegs"}:
        return "Legs"
    return "Body"


def _detect_split_front_hair_strands(
    part: LayerPart,
    *,
    centerline_x: float,
    canvas_size: tuple[int, int],
    head_mid_world_z: float,
    head_tail_world_z: float,
    ground_offset_z: float = 0.0,
    import_scale: float = 1.0,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
] | None:
    if not part.imported_object_name:
        logger.info("Front hair split reject %s -> no imported object name", part.layer_name or part.layer_path)
        return None
    obj = bpy.data.objects.get(part.imported_object_name)
    if obj is None or obj.type != "MESH" or obj.data is None or len(obj.data.vertices) < 8:
        logger.info(
            "Front hair split reject %s -> invalid mesh object obj=%s type=%s verts=%s",
            part.layer_name or part.layer_path,
            getattr(obj, "name", None),
            getattr(obj, "type", None),
            len(obj.data.vertices) if obj and obj.type == "MESH" and obj.data is not None else 0,
        )
        return None

    world_points = []
    for vertex in obj.data.vertices:
        world_co = obj.matrix_world @ vertex.co
        world_points.append((world_co.x, world_co.z))
    if not world_points:
        logger.info("Front hair split reject %s -> no world points", part.layer_name or part.layer_path)
        return None

    world_centerline_x = _pixel_to_plane(centerline_x, canvas_size[1] * 0.5, canvas_size, import_scale)[0]
    world_x_values = [point[0] for point in world_points]
    world_z_values = [point[1] for point in world_points]
    width = max(world_x_values) - min(world_x_values)
    if width <= 1e-6:
        logger.info("Front hair split reject %s -> zero width", part.layer_name or part.layer_path)
        return None

    world_scale = (2.0 / max(1.0, float(max(canvas_size)))) * max(float(import_scale), 1.0e-6)
    slice_half_width = max(width * 0.10, 6.0 * world_scale)
    left_points = [point for point in world_points if point[0] < world_centerline_x - slice_half_width]
    right_points = [point for point in world_points if point[0] > world_centerline_x + slice_half_width]
    center_points = [point for point in world_points if abs(point[0] - world_centerline_x) <= slice_half_width]

    total_points = len(world_points)
    min_side_count = max(6, int(total_points * 0.18))
    min_center_count = max(3, int(total_points * 0.02))
    if len(left_points) < min_side_count or len(right_points) < min_side_count or len(center_points) < min_center_count:
        logger.info(
            "Front hair split reject %s -> insufficient point groups left=%s/%s right=%s/%s center=%s/%s width=%.3f slice_half_width=%.3f",
            part.layer_name or part.layer_path,
            len(left_points),
            min_side_count,
            len(right_points),
            min_side_count,
            len(center_points),
            min_center_count,
            width,
            slice_half_width,
        )
        return None

    center_mass_world_z = sum(point[1] for point in center_points) / len(center_points)
    left_median_z = _median([point[1] for point in left_points])
    right_median_z = _median([point[1] for point in right_points])
    left_strand_points = [point for point in left_points if point[1] <= left_median_z]
    right_strand_points = [point for point in right_points if point[1] <= right_median_z]
    left_center_world_z = sum(point[1] for point in left_strand_points) / len(left_strand_points)
    right_center_world_z = sum(point[1] for point in right_strand_points) / len(right_strand_points)
    if center_mass_world_z <= head_tail_world_z:
        logger.info(
            "Front hair split reject %s -> center COM not above head tail center_mass_world_z=%.6f head_tail_world_z=%.6f head_mid_world_z=%.6f",
            part.layer_name or part.layer_path,
            center_mass_world_z,
            head_tail_world_z,
            head_mid_world_z,
        )
        return None
    lower_head_z = min(head_mid_world_z, head_tail_world_z)
    upper_head_z = max(head_mid_world_z, head_tail_world_z)
    side_center_limit_z = lower_head_z + ((upper_head_z - lower_head_z) * 0.45)
    side_center_limit_z = max(side_center_limit_z, lower_head_z + (8.0 * world_scale))
    if left_center_world_z >= side_center_limit_z or right_center_world_z >= side_center_limit_z:
        left_low_world_z = min(point[1] for point in left_points)
        right_low_world_z = min(point[1] for point in right_points)
        logger.info(
            "Front hair split side COM high; continuing with bbox-guided split %s -> left_center_world_z=%.6f right_center_world_z=%.6f left_low_world_z=%.6f right_low_world_z=%.6f side_center_limit_z=%.6f head_mid_world_z=%.6f head_tail_world_z=%.6f",
            part.layer_name or part.layer_path,
            left_center_world_z,
            right_center_world_z,
            left_low_world_z,
            right_low_world_z,
            side_center_limit_z,
            head_mid_world_z,
            head_tail_world_z,
        )

    left_x_values = [point[0] for point in left_points]
    right_x_values = [point[0] for point in right_points]

    root_sample_count = max(6, int(total_points * 0.04))
    left_root_points = sorted(
        (point for point in world_points if point[0] <= world_centerline_x),
        key=lambda point: abs(point[0] - world_centerline_x),
    )[:root_sample_count]
    right_root_points = sorted(
        (point for point in world_points if point[0] >= world_centerline_x),
        key=lambda point: abs(point[0] - world_centerline_x),
    )[:root_sample_count]
    if len(left_root_points) < 3 or len(right_root_points) < 3:
        logger.info(
            "Front hair split reject %s -> insufficient symmetry-near root points left=%s right=%s root_sample_count=%s",
            part.layer_name or part.layer_path,
            len(left_root_points),
            len(right_root_points),
            root_sample_count,
        )
        return None
    left_root_top_points = [point for point in left_root_points if point[1] >= _median([value[1] for value in left_root_points])]
    right_root_top_points = [point for point in right_root_points if point[1] >= _median([value[1] for value in right_root_points])]
    if left_root_top_points:
        left_root_points = left_root_top_points
    if right_root_top_points:
        right_root_points = right_root_top_points

    if part.alpha_bbox[2] > part.alpha_bbox[0] and part.alpha_bbox[3] > part.alpha_bbox[1]:
        _, y0, _, y1 = part.alpha_bbox
        head_plane_z = _pixel_to_plane(centerline_x, y0 + (y1 - y0) * 0.18, canvas_size, import_scale)[2]
        tail_plane_z = _pixel_to_plane(centerline_x, y0 + (y1 - y0) * 0.92, canvas_size, import_scale)[2]
    else:
        max_world_z = max(world_z_values)
        height = max(max_world_z - min(world_z_values), 1.0e-6)
        head_plane_z = (max_world_z - height * 0.18) - ground_offset_z
        tail_plane_z = (max_world_z - height * 0.92) - ground_offset_z

    left_offset = world_centerline_x - ((min(left_x_values) + max(left_x_values)) * 0.5)
    right_offset = ((min(right_x_values) + max(right_x_values)) * 0.5) - world_centerline_x
    strand_offset = max((left_offset + right_offset) * 0.5, slice_half_width * 1.1)
    left_root_offset = sum(world_centerline_x - point[0] for point in left_root_points) / len(left_root_points)
    right_root_offset = sum(point[0] - world_centerline_x for point in right_root_points) / len(right_root_points)
    mirrored_root_offset = max((left_root_offset + right_root_offset) * 0.5, 1.0e-6)
    shared_root_world_z = (
        sum(point[1] for point in left_root_points) + sum(point[1] for point in right_root_points)
    ) / (len(left_root_points) + len(right_root_points))
    shared_root_plane_z = shared_root_world_z - ground_offset_z
    left_root = (world_centerline_x - mirrored_root_offset, shared_root_plane_z)
    right_root = (world_centerline_x + mirrored_root_offset, shared_root_plane_z)
    left_head = (world_centerline_x - strand_offset, head_plane_z)
    left_tail = (world_centerline_x - strand_offset, tail_plane_z)
    right_head = (world_centerline_x + strand_offset, head_plane_z)
    right_tail = (world_centerline_x + strand_offset, tail_plane_z)
    left_root_pixel = _plane_to_pixel(left_root[0], left_root[1], canvas_size, import_scale)
    left_head_pixel = _plane_to_pixel(left_head[0], left_head[1], canvas_size, import_scale)
    left_tail_pixel = _plane_to_pixel(left_tail[0], left_tail[1], canvas_size, import_scale)
    right_root_pixel = _plane_to_pixel(right_root[0], right_root[1], canvas_size, import_scale)
    right_head_pixel = _plane_to_pixel(right_head[0], right_head[1], canvas_size, import_scale)
    right_tail_pixel = _plane_to_pixel(right_tail[0], right_tail[1], canvas_size, import_scale)

    logger.info(
        "Front hair split accept %s -> center_mass_world_z=%.6f head_tail_world_z=%.6f left_center_world_z=%.6f right_center_world_z=%.6f head_mid_world_z=%.6f strand_offset=%.6f head_plane_z=%.6f tail_plane_z=%.6f left_root_world=(%.6f, %.6f) right_root_world=(%.6f, %.6f) counts(left=%s right=%s center=%s left_strand=%s right_strand=%s left_root=%s right_root=%s)",
        part.layer_name or part.layer_path,
        center_mass_world_z,
        head_tail_world_z,
        left_center_world_z,
        right_center_world_z,
        head_mid_world_z,
        strand_offset,
        head_plane_z,
        tail_plane_z,
        left_root[0],
        shared_root_world_z,
        right_root[0],
        shared_root_world_z,
        len(left_points),
        len(right_points),
        len(center_points),
        len(left_strand_points),
        len(right_strand_points),
        len(left_root_points),
        len(right_root_points),
    )

    return (
        left_root_pixel,
        left_head_pixel,
        left_tail_pixel,
        right_root_pixel,
        right_head_pixel,
        right_tail_pixel,
    )


def estimate_rig(parts: list[LayerPart]) -> RigPlan:
    visible = _visible_parts(parts)
    if not visible:
        return RigPlan()

    canvas_size = visible[0].canvas_size
    import_scale = _import_scale_from_visible_parts(visible)
    keypoints, centerline_x = _estimate_keypoints(visible)
    _apply_bust_body_keypoints(keypoints, visible, centerline_x, canvas_size)
    groups = analyze_groups(visible)
    has_neck = any(_canonical_token(part) in NECK_TOKENS for part in visible)
    has_head = bool(groups["head"]) or any(_canonical_token(part) in HEAD_TOKENS for part in visible)
    has_front_hair = any(_canonical_token(part) == "front hair" for part in visible)
    has_back_hair = any(_canonical_token(part) == "back hair" for part in visible)
    has_left_arm = _has_token_side(visible, "handwear", "LEFT", centerline_x)
    has_right_arm = _has_token_side(visible, "handwear", "RIGHT", centerline_x)
    has_left_leg = _has_token_side(visible, "legwear", "LEFT", centerline_x)
    has_right_leg = _has_token_side(visible, "legwear", "RIGHT", centerline_x)
    has_left_foot = _has_token_side(visible, "footwear", "LEFT", centerline_x)
    has_right_foot = _has_token_side(visible, "footwear", "RIGHT", centerline_x)

    front_hair_bbox = _first_bbox(visible, "front hair")
    back_hair_bbox = _first_bbox(visible, "back hair")

    def hair_points(bbox: tuple[float, float, float, float] | None, fallback_x: float) -> tuple[tuple[float, float], tuple[float, float]]:
        if bbox is None:
            head_xy = (fallback_x, keypoints["headBase"][1] - canvas_size[1] * 0.06)
            tail_xy = (fallback_x, head_xy[1] + canvas_size[1] * 0.12)
            return head_xy, tail_xy
        x0, y0, x1, y1 = bbox
        center_x = (x0 + x1) * 0.5
        head_y = y0 + (y1 - y0) * 0.18
        tail_y = y0 + (y1 - y0) * 0.92
        return (center_x, head_y), (center_x, tail_y)

    front_hair_head, front_hair_tail = hair_points(front_hair_bbox, keypoints["headBase"][0])
    back_hair_head, back_hair_tail = hair_points(back_hair_bbox, keypoints["headBase"][0])
    need_group = {
        "root": True,
        "hips": True,
        "torso": True,
        "spine": True,
        "neck": has_neck or has_head,
        "head": True,
        "front_hair": has_front_hair,
        "back_hair": has_back_hair,
        "eyes": any(_canonical_token(part) in IRIS_TOKENS for part in visible),
        "leftArm": True,
        "rightArm": True,
        "leftElbow": True,
        "rightElbow": True,
        "leftHand": True,
        "rightHand": True,
        "bothArms": groups["arms"] == "merged",
        "leftLeg": True,
        "rightLeg": True,
        "leftKnee": True,
        "rightKnee": True,
        "leftFoot": True,
        "rightFoot": True,
        "bothLegs": groups["legs"] == "merged",
    }
    deform_bones = {
        "root",
        "hips",
        "torso",
        "spine",
        "neck",
        "head",
        "eyes",
        "front_hair",
        "back_hair",
        "bothArms",
        "bothLegs",
    }
    if groups["arms"] != "merged":
        if has_left_arm:
            deform_bones.add("leftArm")
        if has_right_arm:
            deform_bones.add("rightArm")
    if groups["legs"] != "merged":
        if has_left_leg or has_left_foot:
            deform_bones.add("leftLeg")
        if has_right_leg or has_right_foot:
            deform_bones.add("rightLeg")
    parent_lookup = {
        "root": None,
        "hips": "root",
        "spine": "hips" if need_group["hips"] else "root",
        "torso": "spine" if need_group["spine"] else ("hips" if need_group["hips"] else "root"),
        "neck": "torso" if need_group["torso"] else ("spine" if need_group["spine"] else "root"),
        "head": "neck" if need_group["neck"] else ("torso" if need_group["torso"] else "root"),
        "front_hair": "head" if need_group["head"] else ("neck" if need_group["neck"] else "root"),
        "back_hair": "head" if need_group["head"] else ("neck" if need_group["neck"] else "root"),
        "eyes": "head" if need_group["head"] else ("neck" if need_group["neck"] else "root"),
        "leftArm": "torso" if need_group["torso"] else ("spine" if need_group["spine"] else "root"),
        "rightArm": "torso" if need_group["torso"] else ("spine" if need_group["spine"] else "root"),
        "leftElbow": "leftArm" if need_group["leftArm"] else ("torso" if need_group["torso"] else "root"),
        "rightElbow": "rightArm" if need_group["rightArm"] else ("torso" if need_group["torso"] else "root"),
        "leftHand": "leftElbow" if need_group["leftElbow"] else ("leftArm" if need_group["leftArm"] else "root"),
        "rightHand": "rightElbow" if need_group["rightElbow"] else ("rightArm" if need_group["rightArm"] else "root"),
        "bothArms": "torso" if need_group["torso"] else ("spine" if need_group["spine"] else "root"),
        "leftLeg": "hips",
        "rightLeg": "hips",
        "leftKnee": "leftLeg" if need_group["leftLeg"] else "root",
        "rightKnee": "rightLeg" if need_group["rightLeg"] else "root",
        "leftFoot": "leftKnee" if need_group["leftKnee"] else ("leftLeg" if need_group["leftLeg"] else "root"),
        "rightFoot": "rightKnee" if need_group["rightKnee"] else ("rightLeg" if need_group["rightLeg"] else "root"),
        "bothLegs": "hips",
    }
    pivot_points = {
        "root": keypoints["root"],
        "hips": keypoints["hips"],
        "spine": keypoints["spine"],
        "torso": keypoints["torso"],
        "neck": keypoints["neck"],
        "head": keypoints.get("headBase", keypoints["midEye"]),
        "front_hair": front_hair_head,
        "back_hair": back_hair_head,
        "eyes": keypoints["midEye"],
        "leftArm": keypoints["lShoulder"],
        "rightArm": keypoints["rShoulder"],
        "leftElbow": keypoints.get("lElbow", keypoints["lShoulder"]),
        "rightElbow": keypoints.get("rElbow", keypoints["rShoulder"]),
        "leftHand": keypoints.get("lWrist", keypoints.get("lElbow", keypoints["lShoulder"])),
        "rightHand": keypoints.get("rWrist", keypoints.get("rElbow", keypoints["rShoulder"])),
        "bothArms": keypoints["shoulderMid"],
        "leftLeg": keypoints["lHip"],
        "rightLeg": keypoints["rHip"],
        "leftKnee": keypoints.get("lKnee", keypoints["lHip"]),
        "rightKnee": keypoints.get("rKnee", keypoints["rHip"]),
        "leftFoot": keypoints.get("lAnkle", keypoints.get("lKnee", keypoints["lHip"])),
        "rightFoot": keypoints.get("rAnkle", keypoints.get("rKnee", keypoints["rHip"])),
        "bothLegs": keypoints["pelvis"],
        "front_hair_tip": front_hair_tail,
        "back_hair_tip": back_hair_tail,
    }
    create_order = [
        "root",
        "hips",
        "spine",
        "torso",
        "neck",
        "head",
        "eyes",
        "leftArm",
        "rightArm",
        "leftElbow",
        "rightElbow",
        "leftHand",
        "rightHand",
        "bothArms",
        "leftLeg",
        "rightLeg",
        "leftKnee",
        "rightKnee",
        "leftFoot",
        "rightFoot",
        "bothLegs",
    ]
    hair_chain_map: dict[str, tuple[str, ...]] = {}
    bones: dict[str, BonePlan] = {}

    def add_bone(
        name: str,
        head_xy: tuple[float, float],
        tail_xy: tuple[float, float],
        parent: str | None,
        *,
        connected: bool = False,
        deform: bool | None = None,
    ) -> None:
        if _is_body_upward_bone(name) and head_xy[1] < tail_xy[1]:
            head_xy, tail_xy = tail_xy, head_xy
        if _is_downward_bone(name) and head_xy[1] > tail_xy[1]:
            head_xy, tail_xy = tail_xy, head_xy
        if deform is None:
            should_deform = name in deform_bones or name.startswith("front_hair_") or name.startswith("back_hair_")
        else:
            should_deform = deform
        bones[name] = BonePlan(
            name=name,
            head=_pixel_to_plane(head_xy[0], head_xy[1], canvas_size, import_scale),
            tail=_pixel_to_plane(tail_xy[0], tail_xy[1], canvas_size, import_scale),
            parent=parent,
            connected=connected,
            deform=should_deform,
            collection_name=_bone_collection_name(name),
        )

    for bone_name in create_order:
        if not need_group[bone_name]:
            continue
        head_xy = pivot_points[bone_name]
        tail_xy = _tail_target(bone_name, pivot_points | keypoints, canvas_size)
        add_bone(bone_name, head_xy, tail_xy, parent_lookup[bone_name], connected=bone_name in {"hips", "spine", "torso", "neck", "head"})

    face_reference_length = _distance_2d(pivot_points["head"], _tail_target("head", pivot_points | keypoints, canvas_size))
    if face_reference_length <= 1e-6:
        face_reference_length = max(visible[0].canvas_size[1] * 0.08, 1.0)

    head_tail_xy = _tail_target("head", pivot_points | keypoints, canvas_size)
    ground_offset_z = 0.0
    for visible_part in visible:
        if not visible_part.imported_object_name:
            continue
        visible_obj = bpy.data.objects.get(visible_part.imported_object_name)
        if visible_obj is None:
            continue
        ground_offset_z = float(visible_obj.get("hallway_avatar_ground_offset_z", 0.0))
        break
    head_head_plane = _pixel_to_plane(pivot_points["head"][0], pivot_points["head"][1], canvas_size, import_scale)
    head_tail_plane = _pixel_to_plane(head_tail_xy[0], head_tail_xy[1], canvas_size, import_scale)
    head_mid_world_z = ((head_head_plane[2] + head_tail_plane[2]) * 0.5) + ground_offset_z
    head_tail_world_z = head_tail_plane[2] + ground_offset_z

    if need_group["front_hair"]:
        split_front_hair = next((part for part in visible if _canonical_token(part) == "front hair"), None)
        split_layout = None
        if split_front_hair is not None:
            split_layout = _detect_split_front_hair_strands(
                split_front_hair,
                centerline_x=centerline_x,
                canvas_size=canvas_size,
                head_mid_world_z=head_mid_world_z,
                head_tail_world_z=head_tail_world_z,
                ground_offset_z=ground_offset_z,
                import_scale=import_scale,
            )

        if split_layout is not None:
            left_root, left_head, left_tail, right_root, right_head, right_tail = split_layout
            front_segments = _hair_chain_length(
                (_distance_2d(left_head, left_tail) + _distance_2d(right_head, right_tail)) * 0.5,
                face_reference_length,
            )
            front_segments = max(front_segments, 6)
            left_points = _subdivide_chain(left_head, left_tail, front_segments)
            right_points = _subdivide_chain(right_head, right_tail, front_segments)
            left_full_chain = _snap_pixel_chain_to_part_silhouette(
                [left_root] + left_points,
                split_front_hair,
                canvas_size,
                ground_offset_z,
                import_scale,
                label="front_hair_left",
                desired_x_values=[left_head[0]] + [None] * len(left_points),
            )
            right_full_chain = _snap_pixel_chain_to_part_silhouette(
                [right_root] + right_points,
                split_front_hair,
                canvas_size,
                ground_offset_z,
                import_scale,
                label="front_hair_right",
                desired_x_values=[right_head[0]] + [None] * len(right_points),
            )
            left_root = left_full_chain[0]
            right_root = right_full_chain[0]
            left_points = left_full_chain[1:]
            right_points = right_full_chain[1:]
            left_head, left_tail = left_points[0], left_points[-1]
            right_head, right_tail = right_points[0], right_points[-1]
            names: list[str] = []
            left_top_name = "front_hair_left_top"
            right_top_name = "front_hair_right_top"
            add_bone(left_top_name, left_root, left_head, "head")
            add_bone(right_top_name, right_root, right_head, "head")
            names.append(left_top_name)
            parent_name = left_top_name
            for index in range(len(left_points) - 1):
                bone_name = f"front_hair_left_{index + 1:02d}"
                add_bone(bone_name, left_points[index], left_points[index + 1], parent_name, connected=True)
                names.append(bone_name)
                parent_name = bone_name
            names.append(right_top_name)
            parent_name = right_top_name
            for index in range(len(right_points) - 1):
                bone_name = f"front_hair_right_{index + 1:02d}"
                add_bone(bone_name, right_points[index], right_points[index + 1], parent_name, connected=True)
                names.append(bone_name)
                parent_name = bone_name
            hair_chain_map["front_hair"] = tuple(names)
        else:
            front_segments = _hair_chain_length(_distance_2d(front_hair_head, front_hair_tail), face_reference_length)
            points = _subdivide_chain(front_hair_head, front_hair_tail, front_segments)
            points = _snap_pixel_chain_to_part_silhouette(
                points,
                split_front_hair,
                canvas_size,
                ground_offset_z,
                import_scale,
                label="front_hair",
            )
            names: list[str] = []
            parent_name = "head"
            for index in range(front_segments):
                bone_name = f"front_hair_{index + 1:02d}"
                add_bone(bone_name, points[index], points[index + 1], parent_name, connected=index > 0)
                names.append(bone_name)
                parent_name = bone_name
            hair_chain_map["front_hair"] = tuple(names)

    if need_group["back_hair"]:
        back_hair_part = next((part for part in visible if _canonical_token(part) == "back hair"), None)
        back_segments = _hair_chain_length(
            _distance_2d(back_hair_head, back_hair_tail),
            face_reference_length,
            BACK_HAIR_SEGMENT_DENSITY_MULTIPLIER,
        )
        points = _subdivide_chain(back_hair_head, back_hair_tail, back_segments)
        points = _snap_pixel_chain_to_part_silhouette(
            points,
            back_hair_part,
            canvas_size,
            ground_offset_z,
            import_scale,
            label="back_hair",
        )
        names: list[str] = []
        parent_name = "head"
        for index in range(back_segments):
            bone_name = f"back_hair_{index + 1:02d}"
            add_bone(bone_name, points[index], points[index + 1], parent_name, connected=index > 0)
            names.append(bone_name)
            parent_name = bone_name
        hair_chain_map["back_hair"] = tuple(names)

    layer_bone_map = {
        part.layer_path: _layer_bone_for_part(part, groups, centerline_x)
        for part in visible
    }
    layer_auto_weight_bones: dict[str, tuple[str, ...]] = {}
    body_chain = tuple(name for name in ("root", "hips", "spine", "torso", "neck", "head") if name in bones)
    neck_chain = tuple(name for name in ("torso", "neck", "head") if name in bones)
    for part in visible:
        token = _canonical_token(part)
        if token == "topwear" and body_chain:
            layer_auto_weight_bones[part.layer_path] = body_chain
        elif token in NECK_TOKENS and neck_chain:
            layer_auto_weight_bones[part.layer_path] = neck_chain
        elif token == "front hair" and "front_hair" in hair_chain_map:
            layer_auto_weight_bones[part.layer_path] = hair_chain_map["front_hair"]
            layer_bone_map[part.layer_path] = hair_chain_map["front_hair"][0]
        elif token == "back hair" and "back_hair" in hair_chain_map:
            layer_auto_weight_bones[part.layer_path] = hair_chain_map["back_hair"]
            layer_bone_map[part.layer_path] = hair_chain_map["back_hair"][0]
    confidence_values = [part.confidence for part in visible if part.confidence > 0.0]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.25
    joint_pixels = {name: pivot_points[name] for name in create_order if need_group.get(name)}
    joint_pixels.update({"front_hair_tip": front_hair_tail, "back_hair_tip": back_hair_tail})
    group_states = {key: str(value) for key, value in groups.items()}
    return RigPlan(
        bones=bones,
        confidence=confidence,
        centerline_x=centerline_x,
        method="stretchy_studio_bounds",
        layer_bone_map=layer_bone_map,
        layer_auto_weight_bones=layer_auto_weight_bones,
        joint_pixels=joint_pixels,
        group_states=group_states,
        bone_collection_names=DEFAULT_BONE_COLLECTIONS,
    )
