from __future__ import annotations

import bpy
from bpy.types import Operator

from ..core import pipeline


class HALLWAYAVATAR_OT_remesh_imports(Operator):
    bl_idname = "hallway_avatar.remesh_imports"
    bl_label = "Remesh Imported Layers"
    bl_options = {"REGISTER", "UNDO"}

    only_selected: bpy.props.BoolProperty(name="Only Selected", default=False)

    def execute(self, context: bpy.types.Context):
        try:
            count = pipeline.remesh_imported_scene(context, only_selected=self.only_selected)
            self.report({"INFO"}, f"Remeshed {count} imported layers.")
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


classes = (HALLWAYAVATAR_OT_remesh_imports,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
