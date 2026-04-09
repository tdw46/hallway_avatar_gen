from __future__ import annotations

import bpy

from . import runtime, wheel_manager


class HALLWAYAVATARGEN_OT_launch_webview_ui(bpy.types.Operator):
    bl_idname = "hallway_avatar_gen.launch_webview_ui"
    bl_label = "Launch Hallway Avatar Gen"
    bl_description = "Open the native pywebview interface for Hallway Avatar Gen."

    def execute(self, context):
        installed, _status = wheel_manager.group_status(wheel_manager.get_group("ui"))
        if not installed:
            self.report({"ERROR"}, "Install the UI Essentials wheels in the add-on preferences first.")
            return {"CANCELLED"}
        runtime.ensure_webview_running(context)
        self.report({"INFO"}, "Hallway Avatar Gen UI launched.")
        return {"FINISHED"}


class HALLWAYAVATARGEN_OT_stop_webview_ui(bpy.types.Operator):
    bl_idname = "hallway_avatar_gen.stop_webview_ui"
    bl_label = "Stop Hallway Avatar Gen"
    bl_description = "Terminate the background pywebview process."

    def execute(self, context):
        runtime.stop_webview_process()
        self.report({"INFO"}, "Hallway Avatar Gen UI stopped.")
        return {"FINISHED"}
