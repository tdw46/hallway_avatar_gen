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
        mouth_plane_name = str(obj.get("hallway_avatar_mouth_video_plane_object", "") or "").strip()
        if context.scene.hallway_avatar_state.setup_mouth_video_plane:
            if mouth_plane_name:
                context.scene.hallway_avatar_state.last_report = f"Configured facial video preview on {obj.name}; mouth video plane: {mouth_plane_name}"
            else:
                context.scene.hallway_avatar_state.last_report = f"Configured facial video preview on {obj.name}; mouth video plane was not created"
        else:
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
