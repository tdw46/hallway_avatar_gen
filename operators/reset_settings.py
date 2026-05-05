from __future__ import annotations

import bpy
from bpy.props import EnumProperty
from bpy.types import Operator


RESET_GROUP_ITEMS = (
    ("import_options", "Import Options", ""),
    ("alpha_thresholds", "Alpha Thresholds", ""),
    ("trace_contrast", "Trace Contrast", ""),
    ("remesh_main", "Remesh Main", ""),
    ("edge_loops", "Edge Loops", ""),
    ("remesh_misc", "Remesh Misc", ""),
    ("remesh_filters", "Remesh Filters", ""),
)


class HALLWAYAVATAR_OT_reset_settings_group(Operator):
    bl_idname = "hallway_avatar.reset_settings_group"
    bl_label = "Reset Settings Group"
    bl_options = {"INTERNAL"}

    group: EnumProperty(name="Group", items=RESET_GROUP_ITEMS)

    def execute(self, context: bpy.types.Context):
        state = context.scene.hallway_avatar_state
        remesh = state.qremesh_settings

        if self.group == "import_options":
            state.ignore_hidden_layers = True
            state.ignore_empty_layers = True
            state.keep_tiny_named_parts = True
            state.min_visible_pixels = 8
            state.mesh_grid_resolution = 12
            state.replace_existing = True
            state.auto_bind_on_build = True
            state.import_facial_features = False
            state.auto_rig_on_import = True
            state.auto_setup_facial_video = False
            state.facial_video_frame_duration = 1000
            state.facial_video_start_frame = 0
            state.facial_video_frame_offset = 0
            state.facial_video_auto_refresh = True
        elif self.group == "alpha_thresholds":
            state.alpha_noise_floor = 64
            state.visible_alpha_threshold = 32
            state.auto_alpha_threshold_boost = True
        elif self.group == "trace_contrast":
            state.trace_contrast_low = 0.1
            state.trace_contrast_high = 0.9
        elif self.group == "remesh_main":
            remesh.auto_on_import = True
            remesh.target_quad_count = 3000
            remesh.unsubdivide_iterations = 2
            remesh.unsubdivide_target_count = 1400
            remesh.target_count_as_input_percentage = True
            remesh.target_edge_length = 0.02
            remesh.adaptive_size = 100.0
            remesh.adapt_quad_count = True
        elif self.group == "edge_loops":
            remesh.use_vertex_color_map = False
            remesh.use_materials = False
            remesh.use_normals_splitting = False
            remesh.autodetect_hard_edges = True
        elif self.group == "remesh_misc":
            remesh.symmetry_x = False
            remesh.symmetry_y = False
            remesh.symmetry_z = False
        elif self.group == "remesh_filters":
            remesh.remesh_front_hair = True
            remesh.remesh_back_hair = True
            remesh.remesh_face_head = False
            remesh.remesh_topwear = True
            remesh.remesh_handwear = True
            remesh.remesh_bottomwear = False
            remesh.remesh_legwear = True
            remesh.remesh_footwear = True
            remesh.remesh_tail = False
            remesh.remesh_wings = False
            remesh.remesh_objects = False
            remesh.remesh_unclassified = False
        else:
            return {"CANCELLED"}

        return {"FINISHED"}


classes = (HALLWAYAVATAR_OT_reset_settings_group,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
