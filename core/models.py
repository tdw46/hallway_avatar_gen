from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LayerPart:
    source_path: str
    source_type: str
    document_path: str | None = None
    layer_path: str = ""
    layer_name: str = ""
    normalized_token: str = ""
    imported_object_name: str = ""
    temp_image_path: str | None = None
    image_size: tuple[int, int] = (0, 0)
    canvas_size: tuple[int, int] = (0, 0)
    canvas_offset: tuple[int, int] = (0, 0)
    alpha_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    local_alpha_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    centroid: tuple[float, float] = (0.0, 0.0)
    area: int = 0
    perimeter: float = 0.0
    side_guess: str = "UNKNOWN"
    semantic_label: str = "unclassified"
    parent_semantic_label: str = ""
    confidence: float = 0.0
    skipped: bool = False
    skip_reason: str = ""
    draw_index: int = 0


@dataclass
class BonePlan:
    name: str
    head: tuple[float, float, float]
    tail: tuple[float, float, float]
    parent: str | None = None
    connected: bool = False
    deform: bool = True
    collection_name: str = "Body"


@dataclass
class RigPlan:
    bones: dict[str, BonePlan] = field(default_factory=dict)
    confidence: float = 0.0
    centerline_x: float = 0.0
    method: str = ""
    layer_bone_map: dict[str, str] = field(default_factory=dict)
    layer_auto_weight_bones: dict[str, tuple[str, ...]] = field(default_factory=dict)
    joint_pixels: dict[str, tuple[float, float]] = field(default_factory=dict)
    group_states: dict[str, str] = field(default_factory=dict)
    bone_collection_names: tuple[str, ...] = field(default_factory=tuple)
