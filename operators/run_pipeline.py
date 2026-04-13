from __future__ import annotations

import bpy
from bpy.types import Operator

from ..core import pipeline


class HALLWAYAVATAR_OT_run_pipeline(Operator):
    bl_idname = "hallway_avatar.run_pipeline"
    bl_label = "Run Full Pipeline"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context):
        filepath = context.scene.hallway_avatar_state.source_psd_path
        if not filepath:
            self.report({"ERROR"}, "Choose or import a PSD first.")
            return {"CANCELLED"}

        try:
            pipeline.import_psd_scene(context, filepath)
            armature_obj, rig_plan = pipeline.build_armature_scene(
                context,
                bind_weights=context.scene.hallway_avatar_state.auto_bind_on_build,
            )
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Imported PSD and built {armature_obj.name} using {rig_plan.method or 'heuristic'}.",
        )
        return {"FINISHED"}


classes = (HALLWAYAVATAR_OT_run_pipeline,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
