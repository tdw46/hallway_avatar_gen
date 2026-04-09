from __future__ import annotations

import bpy


class HALLWAYAVATARGEN_MT_viewport_menu(bpy.types.Menu):
    bl_idname = "HALLWAYAVATARGEN_MT_viewport_menu"
    bl_label = "Hallway Avatar Gen"

    def draw(self, context):
        layout = self.layout
        layout.operator("hallway_avatar_gen.launch_webview_ui", icon="WORLD")
        layout.operator("hallway_avatar_gen.stop_webview_ui", icon="CANCEL")
        layout.separator()
        layout.operator("hallway_avatar_gen.import_latest_result", icon="IMAGE_DATA")
        layout.operator("hallway_avatar_gen.open_output_folder", icon="FILE_FOLDER")
        layout.operator("hallway_avatar_gen.open_runtime_folder", icon="PREFERENCES")


def draw_view3d_header(self, context):
    self.layout.menu(HALLWAYAVATARGEN_MT_viewport_menu.bl_idname)


def register():
    bpy.types.VIEW3D_MT_editor_menus.append(draw_view3d_header)


def unregister():
    bpy.types.VIEW3D_MT_editor_menus.remove(draw_view3d_header)
