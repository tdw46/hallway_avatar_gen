"""
Shared helpers for Hallway Avatar Gen.
"""

from __future__ import annotations

import os
import site
import subprocess
import sys
import textwrap
import tomllib
import importlib
import json
import time
from pathlib import Path

import bpy


DEPENDENCY_MARKERS = (
    "torch",
    "gradio",
    "webview",
    "diffusers",
    "transformers",
    "numpy",
)

PROBE_MODULES = ("torch", "webview", "gradio")
_PROBE_CACHE: dict[tuple[str, str], tuple[bool, str]] = {}
_SHARED_DEPENDENCY_CACHE: tuple[float, list[Path]] | None = None
_SHARED_WHEEL_CACHE: tuple[float, list[Path]] | None = None
_LEGACY_ADDON_CACHE: tuple[float, list[Path]] | None = None
_PATH_CACHE_TTL = 30.0


def wrap_text_to_panel(text: str, context, *, min_chars: int = 8, full_width: bool = False) -> str:
    try:
        width = getattr(context.region, "width", 300) or 300
        prefs = getattr(context, "preferences", None)
        view = getattr(prefs, "view", None) if prefs else None
        scale = getattr(view, "ui_scale", 1.0) if view else 1.0
        reserved = 240 if not full_width else 60
        available = max(50, width - reserved)
        px_per_char = (13.5 if not full_width else 9.5) * max(scale, 0.5)
        max_chars = max(min_chars, int(available / px_per_char))
    except Exception:
        max_chars = min_chars

    max_cap = 75 if not full_width else 260
    max_chars = max(min_chars, min(max_cap, max_chars))

    return textwrap.fill(
        text or "",
        width=max_chars,
        break_long_words=False,
        replace_whitespace=False,
        expand_tabs=False,
    )


def package_root() -> Path:
    return Path(__file__).resolve().parent


def manifest_path() -> Path:
    return package_root() / "blender_manifest.toml"


def manifest_id(default: str = "hallway_avatar_gen") -> str:
    try:
        data = tomllib.loads(manifest_path().read_text(encoding="utf-8"))
    except Exception:
        return default
    value = data.get("id")
    return value if isinstance(value, str) and value else default


def _path_has_dependency_markers(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for marker in DEPENDENCY_MARKERS:
        if (path / marker).exists():
            return True
        if any(path.glob(f"{marker}*.dist-info")):
            return True
    return False


def extension_user_path(path: str = "", *, create: bool = False) -> Path:
    try:
        resolved = bpy.utils.extension_path_user(__package__, path=path, create=create)
    except ValueError:
        base = bpy.utils.user_resource("EXTENSIONS", create=create)
        repo_module = package_root().parent.name
        pkg_idname = manifest_id()
        target = Path(base) / ".user" / repo_module / pkg_idname
        if path:
            target = target / path
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return target
    return Path(resolved)


def vendor_path(*, create: bool = False) -> Path:
    return extension_user_path("_vendor", create=create)


def wheel_cache_path(*, create: bool = False) -> Path:
    return extension_user_path("wheels/cache", create=create)


def logs_path(*, create: bool = False) -> Path:
    return extension_user_path("logs", create=create)


def runtime_path(*, create: bool = False) -> Path:
    return extension_user_path("runtime", create=create)


def job_queue_path(*, create: bool = False) -> Path:
    return extension_user_path("jobs", create=create)


def output_path(*, create: bool = False) -> Path:
    return extension_user_path("workspace/webui_output", create=create)


def hf_cache_path(*, create: bool = False) -> Path:
    return extension_user_path(".hf_cache", create=create)


def blender_python_executable() -> str:
    candidate = getattr(bpy.app, "binary_path_python", "")
    if candidate:
        return candidate
    return sys.executable


def get_addon_preferences(context=None):
    ctx = context or bpy.context
    addon = ctx.preferences.addons.get(__package__)
    return getattr(addon, "preferences", None)


def get_runtime_state(context=None):
    ctx = context or bpy.context
    return getattr(ctx.window_manager, "hallway_avatar_gen_runtime", None)


def open_directory(path: os.PathLike[str] | str) -> None:
    target = str(path)
    if sys.platform.startswith("darwin"):
        subprocess.Popen(["open", target])
        return
    if os.name == "nt":
        os.startfile(target)
        return
    subprocess.Popen(["xdg-open", target])


def ensure_sys_path(path: os.PathLike[str] | str) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def clear_dependency_caches() -> None:
    global _SHARED_DEPENDENCY_CACHE, _SHARED_WHEEL_CACHE, _LEGACY_ADDON_CACHE
    _PROBE_CACHE.clear()
    _SHARED_DEPENDENCY_CACHE = None
    _SHARED_WHEEL_CACHE = None
    _LEGACY_ADDON_CACHE = None


def _probe_module_from_path(path: Path, module_name: str) -> tuple[bool, str]:
    cache_key = (str(path), module_name)
    cached = _PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not path.exists():
        result = (False, "missing path")
        _PROBE_CACHE[cache_key] = result
        return result

    code = (
        "import importlib, json, sys; "
        f"sys.path.insert(0, {str(path)!r}); "
        "payload = {'ok': False, 'detail': ''}; "
        "try: "
        f" m = importlib.import_module({module_name!r}); "
        " payload['ok'] = True; "
        " payload['detail'] = getattr(m, '__file__', ''); "
        "except Exception as exc: "
        " payload['detail'] = f'{exc.__class__.__name__}: {exc}'; "
        "print(json.dumps(payload))"
    )
    try:
        proc = subprocess.run(
            [blender_python_executable(), "-c", code],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception as exc:
        result = (False, f"{exc.__class__.__name__}: {exc}")
        _PROBE_CACHE[cache_key] = result
        return result

    stdout = proc.stdout.strip().splitlines()
    payload_line = stdout[-1] if stdout else ""
    try:
        payload = json.loads(payload_line)
    except Exception:
        result = (False, payload_line or proc.stderr.strip() or f"exit {proc.returncode}")
        _PROBE_CACHE[cache_key] = result
        return result

    result = (bool(payload.get("ok")), str(payload.get("detail", "")))
    _PROBE_CACHE[cache_key] = result
    return result


def _sibling_extension_dirs() -> list[Path]:
    directories: list[Path] = []
    source_parent = package_root().parent
    if source_parent.exists():
        directories.extend(path for path in source_parent.iterdir() if path.is_dir())

    try:
        user_parent = extension_user_path().parent
    except Exception:
        user_parent = None
    if user_parent and user_parent.exists():
        directories.extend(path for path in user_parent.iterdir() if path.is_dir())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(directories, key=lambda item: item.name.lower()):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def _candidate_import_score(path: Path) -> tuple[int, int, str]:
    marker_count = 0
    ok_count = 0
    failures: list[str] = []
    for module_name in PROBE_MODULES:
        if (path / module_name).exists() or any(path.glob(f"{module_name}*.dist-info")):
            marker_count += 1
            ok, detail = _probe_module_from_path(path, module_name)
            if ok:
                ok_count += 1
            else:
                failures.append(f"{module_name}:{detail}")
    penalty = 0 if not failures else 100 - ok_count
    return (penalty, -ok_count, str(path))


def shared_dependency_paths(*, force_refresh: bool = False) -> list[Path]:
    global _SHARED_DEPENDENCY_CACHE

    now = time.monotonic()
    if not force_refresh and _SHARED_DEPENDENCY_CACHE and now - _SHARED_DEPENDENCY_CACHE[0] <= _PATH_CACHE_TTL:
        return list(_SHARED_DEPENDENCY_CACHE[1])

    candidates: list[Path] = []

    for path in (
        vendor_path(create=False),
        package_root() / "_vendor",
    ):
        if _path_has_dependency_markers(path):
            candidates.append(path.resolve())

    for sibling in _sibling_extension_dirs():
        for candidate in (sibling / "_vendor", sibling / "vendor"):
            if _path_has_dependency_markers(candidate):
                candidates.append(candidate.resolve())

    for site_path in [site.getusersitepackages(), *site.getsitepackages()]:
        candidate = Path(site_path)
        if _path_has_dependency_markers(candidate):
            candidates.append(candidate.resolve())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            deduped.append(path)

    result = sorted(deduped, key=_candidate_import_score)
    _SHARED_DEPENDENCY_CACHE = (now, result)
    return list(result)


def shared_wheel_cache_paths(*, force_refresh: bool = False) -> list[Path]:
    global _SHARED_WHEEL_CACHE

    now = time.monotonic()
    if not force_refresh and _SHARED_WHEEL_CACHE and now - _SHARED_WHEEL_CACHE[0] <= _PATH_CACHE_TTL:
        return list(_SHARED_WHEEL_CACHE[1])

    caches: list[Path] = []
    for sibling in _sibling_extension_dirs():
        for candidate in (sibling / "wheels" / "cache", sibling / "wheels"):
            if candidate.exists() and candidate.is_dir():
                if any(candidate.glob("*.whl")) or candidate.name == "cache":
                    caches.append(candidate.resolve())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in caches:
        if path not in seen:
            seen.add(path)
            deduped.append(path)

    _SHARED_WHEEL_CACHE = (now, deduped)
    return list(deduped)


def bootstrap_dependency_paths() -> list[Path]:
    paths = shared_dependency_paths()
    managed = {str(path) for path in paths}
    if managed:
        sys.path[:] = [entry for entry in sys.path if entry not in managed]
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    importlib.invalidate_caches()
    return paths


def legacy_hallway_addon_paths(*, force_refresh: bool = False) -> list[Path]:
    global _LEGACY_ADDON_CACHE

    now = time.monotonic()
    if not force_refresh and _LEGACY_ADDON_CACHE and now - _LEGACY_ADDON_CACHE[0] <= _PATH_CACHE_TTL:
        return list(_LEGACY_ADDON_CACHE[1])

    try:
        scripts_root = Path(bpy.utils.user_resource("SCRIPTS", create=False))
    except Exception:
        return []

    addons_root = scripts_root / "addons"
    if not addons_root.exists():
        return []

    legacy_names = {
        "Hallway-Image-Gen-Tools",
    }
    matches: list[Path] = []
    for name in legacy_names:
        candidate = addons_root / name
        if candidate.exists():
            matches.append(candidate.resolve())

    _LEGACY_ADDON_CACHE = (now, matches)
    return list(matches)
