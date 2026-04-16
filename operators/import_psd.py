from __future__ import annotations

import os

import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper

from ..core import pipeline


class HALLWAYAVATAR_OT_import_psd(Operator, ImportHelper):
    bl_idname = "hallway_avatar.import_psd"
    bl_label = "Import PSD Avatar"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".psd"
    filter_glob: StringProperty(default="*.psd;*.PSD", options={"HIDDEN"})

    def invoke(self, context: bpy.types.Context, event):
        stored_path = (context.scene.hallway_avatar_state.source_psd_path or "").strip()
        if stored_path:
            resolved = bpy.path.abspath(stored_path)
            if resolved.lower().endswith(".psd") and os.path.isfile(resolved):
                self.filepath = resolved
                return self.execute(context)
            self.filepath = resolved

        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context):
        filepath = (self.filepath or context.scene.hallway_avatar_state.source_psd_path or "").strip()
        if not filepath:
            self.report({"ERROR"}, "Choose a PSD first.")
            return {"CANCELLED"}
        try:
            pipeline.import_psd_scene(context, filepath)
            self.report({"INFO"}, context.scene.hallway_avatar_state.last_report)
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


classes = (HALLWAYAVATAR_OT_import_psd,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
