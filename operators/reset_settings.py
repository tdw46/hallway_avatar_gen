from __future__ import annotations

import bpy
from bpy.props import EnumProperty
from bpy.types import Operator


RESET_GROUP_ITEMS = (
    ("import_options", "Import Options", ""),
    ("alpha_thresholds", "Alpha Thresholds", ""),
    ("trace_contrast", "Trace Contrast", ""),
    ("remesh_main", "QRemeshify Main", ""),
    ("remesh_advanced", "QRemeshify Advanced", ""),
    ("remesh_callbacks", "QRemeshify Callbacks", ""),
    ("remesh_filters", "Remesh Filters", ""),
)


class HALLWAYAVATAR_OT_reset_settings_group(Operator):
    bl_idname = "hallway_avatar.reset_settings_group"
    bl_label = "Reset Settings Group"
    bl_options = {"INTERNAL"}

    group: EnumProperty(name="Group", items=RESET_GROUP_ITEMS)

    def execute(self, context: bpy.types.Context):
        state = context.scene.hallway_avatar_state
        remesh = state.qremeshify_settings
        qw = context.scene.quadwild_props
        qr = context.scene.quadpatches_props

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
            state.auto_setup_facial_video = True
            state.setup_mouth_video_plane = False
            state.mouth_video_path = ""
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
            remesh.use_fast_planar_strips = True
            qw.debug = False
            qw.useCache = False
            qw.enableRemesh = True
            qw.enableSmoothing = True
            qw.enableSharp = True
            qw.sharpAngle = 35
            qw.symmetryX = False
            qw.symmetryY = False
            qw.symmetryZ = False
            qr.scaleFact = 10.0
        elif self.group == "remesh_advanced":
            qr.fixedChartClusters = 0
            qr.alpha = 0.005
            qr.ilpMethod = "LEASTSQUARES"
            qr.timeLimit = 200
            qr.gapLimit = 0.0
            qr.minimumGap = 0.4
            qr.isometry = True
            qr.regularityQuadrilaterals = True
            qr.regularityNonQuadrilaterals = True
            qr.regularityNonQuadrilateralsWeight = 0.9
            qr.alignSingularities = True
            qr.alignSingularitiesWeight = 0.1
            qr.repeatLosingConstraintsIterations = True
            qr.repeatLosingConstraintsQuads = False
            qr.repeatLosingConstraintsNonQuads = False
            qr.repeatLosingConstraintsAlign = True
            qr.hardParityConstraint = True
            qr.flowConfig = "SIMPLE"
            qr.satsumaConfig = "DEFAULT"
        elif self.group == "remesh_callbacks":
            qr.callbackTimeLimit = [3.00, 5.000, 10.0, 20.0, 30.0, 60.0, 90.0, 120.0]
            qr.callbackGapLimit = [0.005, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3]
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
