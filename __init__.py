from __future__ import annotations

import importlib
from pathlib import Path

import bpy

from . import auto_load, utils


PACKAGE_ROOT = Path(__file__).resolve().parent


def _extension_user_path(path: str) -> Path | None:
    try:
        resolved = utils.extension_user_path(path, create=True)
    except Exception:
        resolved = None
    return Path(resolved) if resolved else None


def _bootstrap_vendor_path() -> None:
    utils.bootstrap_dependency_paths()


_bootstrap_vendor_path()

auto_load.set_modules(
    [
        "utils",
        "wheel_manager",
        "scene_builder",
        "runtime",
        "properties",
        "preferences",
        "ops_dependencies",
        "ops_import",
        "ops_webview",
        "ui_menus",
    ]
)


def register() -> None:
    importlib.invalidate_caches()
    auto_load.register()

    props_mod = auto_load.get_module("properties")
    if props_mod and hasattr(props_mod, "register_properties"):
        props_mod.register_properties()

    runtime_mod = auto_load.get_module("runtime")
    if runtime_mod and hasattr(runtime_mod, "register_runtime"):
        runtime_mod.register_runtime()

    wheel_mod = auto_load.get_module("wheel_manager")
    if wheel_mod and hasattr(wheel_mod, "schedule_startup_scan"):
        wheel_mod.schedule_startup_scan()

    ui_mod = auto_load.get_module("ui_menus")
    if ui_mod and hasattr(ui_mod, "register"):
        ui_mod.register()


def unregister() -> None:
    ui_mod = auto_load.get_module("ui_menus")
    if ui_mod and hasattr(ui_mod, "unregister"):
        ui_mod.unregister()

    runtime_mod = auto_load.get_module("runtime")
    if runtime_mod and hasattr(runtime_mod, "unregister_runtime"):
        runtime_mod.unregister_runtime()

    wheel_mod = auto_load.get_module("wheel_manager")
    if wheel_mod and hasattr(wheel_mod, "cancel_startup_scan"):
        wheel_mod.cancel_startup_scan()

    props_mod = auto_load.get_module("properties")
    if props_mod and hasattr(props_mod, "unregister_properties"):
        props_mod.unregister_properties()

    auto_load.unregister()
