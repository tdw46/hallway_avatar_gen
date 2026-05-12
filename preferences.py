from __future__ import annotations

import platform
import sys

import bpy
from bpy.props import EnumProperty, StringProperty
from bpy.types import AddonPreferences

from .core import qremeshify
from .utils import env, paths

ADDON_ID = __package__ or "hallway_avatar_gen"


class HALLWAYAVATAR_Preferences(AddonPreferences):
    bl_idname = ADDON_ID

    cache_dir: StringProperty(
        name="Cache Directory",
        subtype="DIR_PATH",
        default="",
        description="Optional override for cached PSD layer exports and logs",
    )
    log_level: EnumProperty(
        name="Log Level",
        items=(
            ("DEBUG", "Debug", ""),
            ("INFO", "Info", ""),
            ("WARNING", "Warning", ""),
            ("ERROR", "Error", ""),
        ),
        default="INFO",
    )

    def resolved_cache_dir(self) -> str:
        return str(paths.resolve_cache_dir(self.cache_dir))

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        layout.prop(self, "cache_dir")
        layout.prop(self, "log_level")

        status_box = layout.box()
        status_box.label(text="Diagnostics")
        status_box.label(text=f"Blender: {bpy.app.version_string}")
        status_box.label(text=f"Python: {sys.version.split()[0]}")
        status_box.label(text=f"OS: {platform.platform()}")
        status_box.label(text=f"Add-on Root: {paths.addon_root()}")
        status_box.label(text=f"Cache: {self.resolved_cache_dir()}")
        status_box.label(text=f"Vendor Dir: {paths.vendor_dir()}")
        status_box.label(text=f"Dependency Site: {paths.dependency_site_dir(self.cache_dir)}")
        status_box.label(text=f"Wheels Dir: {paths.wheels_dir()}")
        status_box.label(text=f"Resources Dir: {paths.resources_dir()}")
        status_box.label(text=f"PSD Backend: {env.psd_backend_status(self.cache_dir)}")
        status_box.label(text=f"QRemeshify Backend: {qremeshify.runtime_status()}")
        status_box.label(text=f"QRemeshify Runtime: {paths.qremeshify_runtime_dir()}")

        psd_backend_status = env.psd_backend_status(self.cache_dir)
        if psd_backend_status != "ready":
            install_box = layout.box()
            install_box.label(text="PSD Backend")
            install_box.label(text="Installs PSD parsing plus silhouette tracing dependencies.")
            install_box.label(text="See-through generation dependencies are not required yet.")
            install_box.label(text="Use bundled wheels or vendored packages inside this extension folder.")
            install_box.operator("hallway_avatar.install_psd_backend", icon="IMPORT")

        remesh_box = layout.box()
        remesh_box.label(text="QRemeshify Remesh")
        remesh_box.label(text="Hallway uses the vendored QRemeshify runtime from this extension.")
        remesh_box.label(text="It runs QuadWild tracing and QuadPatches quadrangulation, then rebuilds the Blender layer.")


classes = (HALLWAYAVATAR_Preferences,)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
