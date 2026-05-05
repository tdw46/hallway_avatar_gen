from __future__ import annotations

import bpy
from bpy.types import Operator

from ..core import facial_video_preview


class HALLWAYAVATAR_OT_setup_facial_video(Operator):
    bl_idname = "hallway_avatar.setup_facial_video"
    bl_label = "Setup Facial Video Preview"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        try:
            obj = facial_video_preview.setup_from_state(context, raise_on_missing=True)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        context.scene.hallway_avatar_state.last_report = f"Configured facial video preview on {obj.name}"
        self.report({"INFO"}, context.scene.hallway_avatar_state.last_report)
        return {"FINISHED"}


classes = (HALLWAYAVATAR_OT_setup_facial_video,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
