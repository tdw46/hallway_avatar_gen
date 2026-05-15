from __future__ import annotations

import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper


class HALLWAYAVATAR_OT_select_facial_video_transform(Operator, ImportHelper):
    bl_idname = "hallway_avatar.select_facial_video_transform"
    bl_label = "Choose Facial Video Transform"
    bl_options = {"INTERNAL"}

    filename_ext = ".txt"
    filter_glob: StringProperty(default="*.txt;*.TXT", options={"HIDDEN"})

    def invoke(self, context: bpy.types.Context, event):
        stored_path = (context.scene.hallway_avatar_state.facial_video_transform_path or "").strip()
        if stored_path:
            self.filepath = bpy.path.abspath(stored_path)
        return super().invoke(context, event)

    def execute(self, context: bpy.types.Context):
        context.scene.hallway_avatar_state.facial_video_transform_path = self.filepath
        return {"FINISHED"}


class HALLWAYAVATAR_OT_select_facial_video_file(Operator, ImportHelper):
    bl_idname = "hallway_avatar.select_facial_video_file"
    bl_label = "Choose Facial Video"
    bl_options = {"INTERNAL"}

    filename_ext = ".mp4"
    filter_glob: StringProperty(default="*.mp4;*.mov;*.m4v;*.avi;*.MP4;*.MOV;*.M4V;*.AVI", options={"HIDDEN"})

    def invoke(self, context: bpy.types.Context, event):
        stored_path = (context.scene.hallway_avatar_state.facial_video_path or "").strip()
        if stored_path:
            self.filepath = bpy.path.abspath(stored_path)
        return super().invoke(context, event)

    def execute(self, context: bpy.types.Context):
        context.scene.hallway_avatar_state.facial_video_path = self.filepath
        return {"FINISHED"}


class HALLWAYAVATAR_OT_select_mouth_video_file(Operator, ImportHelper):
    bl_idname = "hallway_avatar.select_mouth_video_file"
    bl_label = "Choose Mouth Video"
    bl_options = {"INTERNAL"}

    filename_ext = ".mp4"
    filter_glob: StringProperty(default="*.mp4;*.mov;*.m4v;*.avi;*.MP4;*.MOV;*.M4V;*.AVI", options={"HIDDEN"})

    def invoke(self, context: bpy.types.Context, event):
        stored_path = (context.scene.hallway_avatar_state.mouth_video_path or "").strip()
        if stored_path:
            self.filepath = bpy.path.abspath(stored_path)
        return super().invoke(context, event)

    def execute(self, context: bpy.types.Context):
        context.scene.hallway_avatar_state.mouth_video_path = self.filepath
        return {"FINISHED"}


classes = (
    HALLWAYAVATAR_OT_select_facial_video_transform,
    HALLWAYAVATAR_OT_select_facial_video_file,
    HALLWAYAVATAR_OT_select_mouth_video_file,
)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
