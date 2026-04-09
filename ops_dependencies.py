from __future__ import annotations

import bpy

from . import utils, wheel_manager


class HALLWAYAVATARGEN_OT_install_dependency_group(bpy.types.Operator):
    bl_idname = "hallway_avatar_gen.install_dependency_group"
    bl_label = "Install Dependency Group"
    bl_description = "Download cached wheels and install them into the extension-local vendor folder."

    group_key: bpy.props.StringProperty()

    def execute(self, context):
        ok, message = wheel_manager.install_group_async(self.group_key)
        self.report({"INFO"} if ok else {"ERROR"}, message)
        return {"FINISHED"} if ok else {"CANCELLED"}


class HALLWAYAVATARGEN_OT_rescan_dependencies(bpy.types.Operator):
    bl_idname = "hallway_avatar_gen.rescan_dependencies"
    bl_label = "Rescan Dependencies"
    bl_description = "Refresh shared dependency detection and runtime status for this extension."

    def execute(self, context):
        wheel_manager.refresh_status_snapshot(redraw=True)
        self.report({"INFO"}, "Dependency status rescanned")
        return {"FINISHED"}


class HALLWAYAVATARGEN_OT_open_runtime_folder(bpy.types.Operator):
    bl_idname = "hallway_avatar_gen.open_runtime_folder"
    bl_label = "Open Runtime Folder"
    bl_description = "Open the user-writable runtime folder for this extension."

    def execute(self, context):
        utils.open_directory(utils.extension_user_path(create=True))
        return {"FINISHED"}
