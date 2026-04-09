from __future__ import annotations

import bpy


class HallwayAvatarGenRuntimeState(bpy.types.PropertyGroup):
    last_job_id: bpy.props.StringProperty(name="Last Job ID")
    last_output_dir: bpy.props.StringProperty(name="Last Output Directory")
    last_import_collection: bpy.props.StringProperty(name="Last Import Collection")
    webview_running: bpy.props.BoolProperty(name="Webview Running", default=False)


def register_properties() -> None:
    bpy.types.WindowManager.hallway_avatar_gen_runtime = bpy.props.PointerProperty(
        type=HallwayAvatarGenRuntimeState
    )


def unregister_properties() -> None:
    if hasattr(bpy.types.WindowManager, "hallway_avatar_gen_runtime"):
        del bpy.types.WindowManager.hallway_avatar_gen_runtime
