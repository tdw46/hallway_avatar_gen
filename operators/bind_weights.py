from __future__ import annotations

import bpy
from bpy.types import Operator

from ..core import pipeline


class HALLWAYAVATAR_OT_bind_weights(Operator):
    bl_idname = "hallway_avatar.bind_weights"
    bl_label = "Bind Weights"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        try:
            pipeline.bind_weights_scene(context)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, context.scene.hallway_avatar_state.last_report or "Weights bound.")
        return {"FINISHED"}


classes = (HALLWAYAVATAR_OT_bind_weights,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
