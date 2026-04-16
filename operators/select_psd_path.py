from __future__ import annotations

import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper


class HALLWAYAVATAR_OT_select_psd_path(Operator, ImportHelper):
    bl_idname = "hallway_avatar.select_psd_path"
    bl_label = "Choose PSD"
    bl_options = {"INTERNAL"}

    filename_ext = ".psd"
    filter_glob: StringProperty(default="*.psd;*.PSD", options={"HIDDEN"})

    def invoke(self, context: bpy.types.Context, event):
        stored_path = (context.scene.hallway_avatar_state.source_psd_path or "").strip()
        if stored_path:
            self.filepath = bpy.path.abspath(stored_path)
        return super().invoke(context, event)

    def execute(self, context: bpy.types.Context):
        context.scene.hallway_avatar_state.source_psd_path = self.filepath
        return {"FINISHED"}


classes = (HALLWAYAVATAR_OT_select_psd_path,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
