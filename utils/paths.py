from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

ADDON_ID = "hallway_avatar_gen"


def addon_root() -> Path:
    return Path(__file__).resolve().parent.parent


def vendor_dir() -> Path:
    return addon_root() / "vendor"


def vendor_site_dir() -> Path:
    return vendor_dir() / "site-packages"


def ensure_vendor_site_dir() -> Path:
    path = vendor_site_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def wheels_dir() -> Path:
    return addon_root() / "wheels"


def resources_dir() -> Path:
    return addon_root() / "resources"


def qremeshify_runtime_dir() -> Path:
    return resources_dir() / "qremeshify" / "lib"


def voxel_skinning_runtime_dir() -> Path:
    return resources_dir() / "voxel_skinning"


def default_cache_dir() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Caches" / ADDON_ID
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / ADDON_ID
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / ADDON_ID


def resolve_cache_dir(configured_path: str = "") -> Path:
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return default_cache_dir()


def ensure_cache_dir(configured_path: str = "") -> Path:
    path = resolve_cache_dir(configured_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dependency_site_dir(configured_path: str = "") -> Path:
    return ensure_vendor_site_dir()


def import_cache_dir(configured_path: str = "") -> Path:
    path = ensure_cache_dir(configured_path) / "imports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_session_dir(source_name: str, configured_path: str = "") -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in source_name)
    path = import_cache_dir(configured_path) / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_file_path(configured_path: str = "") -> Path:
    logs_dir = ensure_cache_dir(configured_path) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{ADDON_ID}.log"
