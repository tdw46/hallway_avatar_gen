from __future__ import annotations

import bpy
from bpy.types import Operator

from ..core import pipeline


class HALLWAYAVATAR_OT_build_armature(Operator):
    bl_idname = "hallway_avatar.build_armature"
    bl_label = "Build Armature"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        try:
            armature_obj, rig_plan = pipeline.build_armature_scene(
                context,
                bind_weights=context.scene.hallway_avatar_state.auto_bind_on_build,
            )
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Built {armature_obj.name} using {rig_plan.method or 'heuristic'} with {len(rig_plan.bones)} bones.",
        )
        return {"FINISHED"}


classes = (HALLWAYAVATAR_OT_build_armature,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
