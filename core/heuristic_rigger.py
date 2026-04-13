from __future__ import annotations

from .models import BonePlan, LayerPart, RigPlan
from . import seethrough_naming


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


def _pixel_to_plane(x: float, y: float, canvas_size: tuple[int, int]) -> tuple[float, float, float]:
    scale = 2.0 / max(1.0, float(max(canvas_size)))
    canvas_w, canvas_h = canvas_size
    return ((x - canvas_w * 0.5) * scale, 0.0, (canvas_h * 0.5 - y) * scale)


def _center_of_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


NECK_TOKENS = {"neck", "neckwear"}
HEAD_TOKENS = {
    "face",
    "front hair",
    "back hair",
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
    keypoints.setdefault("midEye", (cx, canvas_h * 0.18))
    return keypoints, centerline_x


def _layer_bone_for_part(part: LayerPart, groups: dict[str, str | bool], centerline_x: float) -> str:
    token = _canonical_token(part)
    if token in IRIS_TOKENS:
        return "eyes"
    if token in NECK_TOKENS:
        return "neck"
    if token in HEAD_TOKENS:
        return "head"
    if token == "topwear":
        return "torso"
    if token == "bottomwear":
        return "root"
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
        "root": keypoints.get("waist"),
        "torso": keypoints.get("neck"),
        "neck": keypoints.get("head"),
        "head": keypoints.get("eyes"),
        "eyes": (x_value, y_value - default_offset),
        "leftArm": keypoints.get("leftElbow") or keypoints.get("lWrist"),
        "rightArm": keypoints.get("rightElbow") or keypoints.get("rWrist"),
        "leftElbow": keypoints.get("lWrist"),
        "rightElbow": keypoints.get("rWrist"),
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
        "leftLeg": keypoints.get("leftKnee") or keypoints.get("lAnkle"),
        "rightLeg": keypoints.get("rightKnee") or keypoints.get("rAnkle"),
        "leftKnee": keypoints.get("lAnkle"),
        "rightKnee": keypoints.get("rAnkle"),
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


def estimate_rig(parts: list[LayerPart]) -> RigPlan:
    visible = _visible_parts(parts)
    if not visible:
        return RigPlan()

    canvas_size = visible[0].canvas_size
    keypoints, centerline_x = _estimate_keypoints(visible)
    groups = analyze_groups(visible)
    has_neck = any(_canonical_token(part) in NECK_TOKENS for part in visible)
    has_head = bool(groups["head"]) or any(_canonical_token(part) in HEAD_TOKENS for part in visible)
    has_left_arm = _has_token_side(visible, "handwear", "LEFT", centerline_x)
    has_right_arm = _has_token_side(visible, "handwear", "RIGHT", centerline_x)
    has_left_leg = _has_token_side(visible, "legwear", "LEFT", centerline_x)
    has_right_leg = _has_token_side(visible, "legwear", "RIGHT", centerline_x)
    need_group = {
        "root": True,
        "torso": bool(groups["torso"]) or has_neck or has_head,
        "neck": has_neck or has_head,
        "head": has_head,
        "eyes": any(_canonical_token(part) in IRIS_TOKENS for part in visible),
        "leftArm": groups["arms"] == "split" or (groups["arms"] == "partial" and has_left_arm),
        "rightArm": groups["arms"] == "split" or (groups["arms"] == "partial" and has_right_arm),
        "leftElbow": groups["arms"] == "split" or (groups["arms"] == "partial" and has_left_arm),
        "rightElbow": groups["arms"] == "split" or (groups["arms"] == "partial" and has_right_arm),
        "bothArms": groups["arms"] == "merged",
        "leftLeg": groups["legs"] == "split" or (groups["legs"] == "partial" and has_left_leg),
        "rightLeg": groups["legs"] == "split" or (groups["legs"] == "partial" and has_right_leg),
        "leftKnee": groups["legs"] == "split" or (groups["legs"] == "partial" and has_left_leg),
        "rightKnee": groups["legs"] == "split" or (groups["legs"] == "partial" and has_right_leg),
        "bothLegs": groups["legs"] == "merged",
    }
    parent_lookup = {
        "root": None,
        "torso": "root",
        "neck": "torso" if need_group["torso"] else "root",
        "head": "neck" if need_group["neck"] else ("torso" if need_group["torso"] else "root"),
        "eyes": "head" if need_group["head"] else ("neck" if need_group["neck"] else "root"),
        "leftArm": "torso" if need_group["torso"] else "root",
        "rightArm": "torso" if need_group["torso"] else "root",
        "leftElbow": "leftArm" if need_group["leftArm"] else ("torso" if need_group["torso"] else "root"),
        "rightElbow": "rightArm" if need_group["rightArm"] else ("torso" if need_group["torso"] else "root"),
        "bothArms": "torso" if need_group["torso"] else "root",
        "leftLeg": "root",
        "rightLeg": "root",
        "leftKnee": "leftLeg" if need_group["leftLeg"] else "root",
        "rightKnee": "rightLeg" if need_group["rightLeg"] else "root",
        "bothLegs": "root",
    }
    pivot_points = {
        "root": keypoints["pelvis"],
        "torso": keypoints["waist"],
        "neck": keypoints["neck"],
        "head": keypoints.get("headBase", keypoints["midEye"]),
        "eyes": keypoints["midEye"],
        "leftArm": keypoints["lShoulder"],
        "rightArm": keypoints["rShoulder"],
        "leftElbow": keypoints.get("lElbow", keypoints["lShoulder"]),
        "rightElbow": keypoints.get("rElbow", keypoints["rShoulder"]),
        "bothArms": keypoints["shoulderMid"],
        "leftLeg": keypoints["lHip"],
        "rightLeg": keypoints["rHip"],
        "leftKnee": keypoints.get("lKnee", keypoints["lHip"]),
        "rightKnee": keypoints.get("rKnee", keypoints["rHip"]),
        "bothLegs": keypoints["pelvis"],
    }
    create_order = [
        "root",
        "torso",
        "neck",
        "head",
        "eyes",
        "leftArm",
        "rightArm",
        "leftElbow",
        "rightElbow",
        "bothArms",
        "leftLeg",
        "rightLeg",
        "leftKnee",
        "rightKnee",
        "bothLegs",
    ]
    bones: dict[str, BonePlan] = {}

    def add_bone(name: str, head_xy: tuple[float, float], tail_xy: tuple[float, float], parent: str | None) -> None:
        bones[name] = BonePlan(
            name=name,
            head=_pixel_to_plane(head_xy[0], head_xy[1], canvas_size),
            tail=_pixel_to_plane(tail_xy[0], tail_xy[1], canvas_size),
            parent=parent,
        )

    for bone_name in create_order:
        if not need_group[bone_name]:
            continue
        head_xy = pivot_points[bone_name]
        tail_xy = _tail_target(bone_name, pivot_points | keypoints, canvas_size)
        add_bone(bone_name, head_xy, tail_xy, parent_lookup[bone_name])

    layer_bone_map = {
        part.layer_path: _layer_bone_for_part(part, groups, centerline_x)
        for part in visible
    }
    confidence_values = [part.confidence for part in visible if part.confidence > 0.0]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.25
    joint_pixels = {name: pivot_points[name] for name in create_order if need_group[name]}
    group_states = {key: str(value) for key, value in groups.items()}
    return RigPlan(
        bones=bones,
        confidence=confidence,
        centerline_x=centerline_x,
        method="stretchy_studio_bounds",
        layer_bone_map=layer_bone_map,
        joint_pixels=joint_pixels,
        group_states=group_states,
    )
