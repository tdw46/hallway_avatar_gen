from __future__ import annotations

import importlib
import sys

from . import auto_load

MODULES = [
    "utils.paths",
    "utils.env",
    "utils.logging",
    "utils.blender",
    "core.models",
    "core.seethrough_naming",
    "core.psd_layer_filters",
    "core.psd_io",
    "core.alpha_mesh_adapter",
    "core.part_classifier",
    "core.heuristic_rigger",
    "core.armature_builder",
    "core.vrm_integration",
    "core.mtoon_materials",
    "core.facial_video_preview",
    "core.qremeshify_runtime.lib.data",
    "core.qremeshify_runtime.lib",
    "core.qremeshify_runtime.util.bisect",
    "core.qremeshify_runtime.util.exporter",
    "core.qremeshify_runtime.util.importer",
    "core.qremeshify_runtime.props",
    "core.qremeshify_worker",
    "core.qremeshify",
    "core.weighting",
    "core.pipeline",
    "properties",
    "preferences",
    "operators.install_dependencies",
    "operators.import_psd",
    "operators.select_psd_path",
    "operators.select_facial_video_paths",
    "operators.classify_parts",
    "operators.set_bool",
    "operators.reset_settings",
    "operators.remesh_imports",
    "operators.setup_facial_video",
    "operators.build_armature",
    "operators.bind_weights",
    "operators.run_pipeline",
    "ui.panels",
]

auto_load.set_modules(MODULES)


def register() -> None:
    env_mod = importlib.import_module(".utils.env", __package__)
    if hasattr(env_mod, "bootstrap"):
        env_mod.bootstrap()

    auto_load.register()

    props_mod = auto_load.get_module("properties")
    if props_mod and hasattr(props_mod, "register_properties"):
        props_mod.register_properties()


def unregister() -> None:
    props_mod = auto_load.get_module("properties")
    if props_mod and hasattr(props_mod, "unregister_properties"):
        try:
            props_mod.unregister_properties()
        except Exception:
            pass

    auto_load.unregister()

    try:
        package = __package__
        to_delete = [name for name in list(sys.modules.keys()) if name == package or name.startswith(package + ".")]
        for name in to_delete:
            del sys.modules[name]
        importlib.invalidate_caches()
    except Exception:
        pass
