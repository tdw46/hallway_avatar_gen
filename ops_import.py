from __future__ import annotations

import bpy

from . import runtime, scene_builder, utils


class HALLWAYAVATARGEN_OT_import_latest_result(bpy.types.Operator):
    bl_idname = "hallway_avatar_gen.import_latest_result"
    bl_label = "Import Latest Result"
    bl_description = "Import the most recent completed decomposition result into the current scene."

    def execute(self, context):
        state = utils.get_runtime_state(context)
        if not state or not state.last_output_dir:
            self.report({"ERROR"}, "No completed result has been recorded yet.")
            return {"CANCELLED"}

        collection_name = scene_builder.import_result_directory(state.last_output_dir, context=context)
        state.last_import_collection = collection_name
        self.report({"INFO"}, f"Imported {collection_name}")
        return {"FINISHED"}


class HALLWAYAVATARGEN_OT_open_output_folder(bpy.types.Operator):
    bl_idname = "hallway_avatar_gen.open_output_folder"
    bl_label = "Open Output Folder"
    bl_description = "Open the extension output folder for the latest run."

    def execute(self, context):
        state = utils.get_runtime_state(context)
        if state and state.last_output_dir:
            utils.open_directory(state.last_output_dir)
        else:
            utils.open_directory(utils.output_path(create=True))
        return {"FINISHED"}
