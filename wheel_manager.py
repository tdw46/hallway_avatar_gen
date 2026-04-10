from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import utils


@dataclass(frozen=True)
class DependencyGroup:
    key: str
    label: str
    requirements_file: str
    modules: tuple[str, ...]
    description: str
    index_url: str | None = None
    supported_platforms: tuple[str, ...] = ()


@dataclass
class InstallState:
    is_running: bool = False
    group_key: str = ""
    message: str = "Idle"
    current_line: str = ""
    progress: float = 0.0
    log_path: str = ""
    log_tail: str = ""
    failure_summary: str = ""
    last_return_code: int | None = None
    started_at: float = 0.0
    _thread: threading.Thread | None = field(default=None, repr=False)


INSTALL_STATE = InstallState()
_STATE_LOCK = threading.RLock()
_RUNTIME_PROBE_CACHE: dict[tuple[str, str], tuple[float, dict[str, object]]] = {}
_MODULE_AVAILABILITY_CACHE: dict[tuple[str, str], tuple[float, bool]] = {}
_GROUP_STATUS_CACHE: dict[tuple[str, str], tuple[float, tuple[bool, str]]] = {}
_PANEL_SNAPSHOT_CACHE: dict[str, object] | None = None
_STARTUP_SCAN_REGISTERED = False
_RUNTIME_PROBE_TTL = 2.0
_STATUS_CACHE_TTL = 15.0
_STACK_GROUP_KEYS = {"apple_metal", "windows_cuda", "linux_cuda"}
_FAST_LOCAL_MARKER_GROUP_KEYS = {"inference_base"}

_SPECIAL_IMPORT_TARGETS: dict[str, str] = {
    "PIL": "PIL.Image",
}

_DIST_INFO_ALIASES: dict[str, tuple[str, ...]] = {
    "PIL": ("Pillow", "pillow"),
    "cv2": ("opencv_python", "opencv_python_headless"),
    "pillow_jxl": ("pillow_jxl_plugin",),
    "yaml": ("PyYAML", "pyyaml"),
    "sklearn": ("scikit_learn",),
    "webview": ("pywebview",),
    "pytorch_grad_cam": ("grad_cam",),
}


DEPENDENCY_GROUPS: tuple[DependencyGroup, ...] = (
    DependencyGroup(
        key="ui",
        label="UI Essentials",
        requirements_file="wheels/requirements/ui.txt",
        modules=("webview",),
        description="Installs the native pywebview shell used by the Hallway Avatar Gen window.",
    ),
    DependencyGroup(
        key="inference_base",
        label="Inference Base",
        requirements_file="wheels/requirements/inference_base.txt",
        modules=(
            "numpy",
            "cv2",
            "PIL",
            "yaml",
            "scipy",
            "sklearn",
            "einops",
            "pandas",
            "transformers",
            "diffusers",
            "huggingface_hub",
            "tokenizers",
            "accelerate",
            "safetensors",
            "kornia",
            "timm",
            "psd_tools",
            "tqdm",
        ),
        description="Core Python wheels used by the decomposition and PSD assembly pipeline.",
    ),
    DependencyGroup(
        key="apple_metal",
        label="Apple Metal Torch",
        requirements_file="wheels/requirements/apple_metal.txt",
        modules=("torch", "torchvision", "torchaudio"),
        description="Installs the Torch stack used for Apple Silicon / Metal (MPS) execution.",
        supported_platforms=("Darwin",),
    ),
    DependencyGroup(
        key="windows_cuda",
        label="Windows CUDA Torch",
        requirements_file="wheels/requirements/windows_cuda.txt",
        modules=("torch", "torchvision", "torchaudio", "bitsandbytes"),
        description="Installs the CUDA Torch stack plus bitsandbytes for NF4 inference on NVIDIA.",
        index_url="https://download.pytorch.org/whl/cu128",
        supported_platforms=("Windows",),
    ),
    DependencyGroup(
        key="linux_cuda",
        label="Linux CUDA Torch",
        requirements_file="wheels/requirements/linux_cuda.txt",
        modules=("torch", "torchvision", "torchaudio", "bitsandbytes"),
        description="Installs the CUDA Torch stack plus bitsandbytes for NF4 inference on NVIDIA.",
        index_url="https://download.pytorch.org/whl/cu128",
        supported_platforms=("Linux",),
    ),
)

REQUIREMENT_IMPORTS_BY_GROUP: dict[str, tuple[str, ...]] = {
    "ui": (
        "webview",
    ),
    "inference_base": (
        "numpy",
        "cv2",
        "PIL",
        "pillow_jxl",
        "yaml",
        "scipy",
        "sklearn",
        "skimage",
        "einops",
        "pandas",
        "transformers",
        "diffusers",
        "huggingface_hub",
        "tokenizers",
        "accelerate",
        "safetensors",
        "kornia",
        "timm",
        "pytorch_grad_cam",
        "pycocotools",
        "psd_tools",
        "tqdm",
        "colorama",
        "matplotlib",
    ),
    "apple_metal": (
        "torch",
        "torchvision",
        "torchaudio",
    ),
    "windows_cuda": (
        "torch",
        "torchvision",
        "torchaudio",
        "bitsandbytes",
    ),
    "linux_cuda": (
        "torch",
        "torchvision",
        "torchaudio",
        "bitsandbytes",
    ),
}



def dependency_groups() -> tuple[DependencyGroup, ...]:
    return DEPENDENCY_GROUPS


def get_group(group_key: str) -> DependencyGroup:
    for group in DEPENDENCY_GROUPS:
        if group.key == group_key:
            return group
    raise KeyError(group_key)


def requirements_path(group: DependencyGroup) -> Path:
    return utils.package_root() / group.requirements_file


def requirement_entries(group: DependencyGroup) -> tuple[tuple[str, str], ...]:
    import_names = REQUIREMENT_IMPORTS_BY_GROUP.get(group.key)
    if import_names is None:
        raise KeyError(group.key)

    requirement_lines = tuple(
        line.strip()
        for line in requirements_path(group).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    if len(import_names) != len(requirement_lines):
        raise ValueError(f"Requirement import mapping mismatch for {group.key}")
    return tuple(zip(import_names, requirement_lines))


def _module_import_target(name: str) -> str:
    return _SPECIAL_IMPORT_TARGETS.get(name, name)


def _local_dependency_paths() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for path in (
        *utils.vendor_overlay_paths(),
        utils.vendor_path(create=False),
        utils.package_root() / "_vendor",
    ):
        try:
            if path.exists() and path.is_dir():
                candidates.append(path.resolve())
        except Exception:
            continue

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return tuple(deduped)


def _distribution_prefixes(name: str) -> tuple[str, ...]:
    prefixes = {
        name,
        name.lower(),
        name.replace("_", "-"),
        name.replace("_", "-").lower(),
        name.replace("_", ""),
        name.replace("_", "").lower(),
    }
    prefixes.update(_DIST_INFO_ALIASES.get(name, ()))
    prefixes.update(alias.lower() for alias in _DIST_INFO_ALIASES.get(name, ()))
    return tuple(sorted(prefixes))


def _path_has_module_marker(path: Path, module_name: str) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    if module_name == "PIL":
        return (path / "PIL" / "Image.py").exists()
    if module_name == "cv2":
        return (path / "cv2").exists() or any(path.glob("cv2*.pyd"))
    if module_name == "yaml":
        return (path / "yaml" / "__init__.py").exists() or any(path.glob("_yaml*.pyd"))

    if (path / module_name).exists() or (path / f"{module_name}.py").exists():
        return True

    return any(
        any(path.glob(f"{prefix}-*.dist-info"))
        for prefix in _distribution_prefixes(module_name)
    )


def _local_marker_group_ready(group: DependencyGroup) -> bool | None:
    if group.key not in _FAST_LOCAL_MARKER_GROUP_KEYS:
        return None

    local_paths = _local_dependency_paths()
    if not local_paths:
        return None

    for module_name, _ in requirement_entries(group):
        if not any(_path_has_module_marker(path, module_name) for path in local_paths):
            return False
    return True


def _dependency_signature() -> str:
    return os.pathsep.join(str(path) for path in utils.shared_dependency_paths())


def invalidate_status_caches() -> None:
    global _PANEL_SNAPSHOT_CACHE
    with _STATE_LOCK:
        _clear_runtime_probe_cache()
        _MODULE_AVAILABILITY_CACHE.clear()
        _GROUP_STATUS_CACHE.clear()
        _PANEL_SNAPSHOT_CACHE = None
        utils.clear_dependency_caches()


def refresh_status_snapshot(*, redraw: bool = True) -> dict[str, object]:
    invalidate_status_caches()
    snapshot = preferences_snapshot(force_refresh=True)
    if redraw:
        _tag_preferences_redraw()
    return snapshot


def _probe_runtime_payload(mode: str, module_name: str) -> dict[str, object]:
    paths = [str(path) for path in utils.shared_dependency_paths()]
    cache_key = (f"{mode}:{module_name}", os.pathsep.join(paths))
    now = time.monotonic()
    with _STATE_LOCK:
        cached = _RUNTIME_PROBE_CACHE.get(cache_key)
    if cached and now - cached[0] <= _RUNTIME_PROBE_TTL:
        return cached[1]

    code = f"""
import importlib
import importlib.util
import json
import sys

paths = {paths!r}
for path in paths:
    if path in sys.path:
        sys.path.remove(path)
for path in reversed(paths):
    sys.path.insert(0, path)
importlib.invalidate_caches()
payload = {{'ok': False, 'summary': '', 'origin': '', 'error': ''}}

try:
    if {mode!r} == 'torch':
        spec = importlib.util.find_spec('torch')
        payload['origin'] = getattr(spec, 'origin', '') if spec else ''
        if spec is None:
            payload['summary'] = 'Torch not installed'
        else:
            import torch
            version = getattr(torch, '__version__', 'unknown')
            summary = f'Torch {{version}} | CPU only'
            if torch.cuda.is_available():
                summary = f'Torch {{version}} | CUDA available'
            else:
                backends = getattr(torch, 'backends', None)
                mps_backend = getattr(backends, 'mps', None)
                if mps_backend is not None:
                    is_built = getattr(mps_backend, 'is_built', lambda: True)
                    is_available = getattr(mps_backend, 'is_available', lambda: False)
                    if bool(is_built()) and bool(is_available()):
                        summary = f'Torch {{version}} | Apple Metal (MPS) available'
            payload['ok'] = True
            payload['summary'] = summary
            payload['origin'] = getattr(torch, '__file__', payload['origin'])
    elif {mode!r} == 'webview':
        spec = importlib.util.find_spec('webview')
        payload['origin'] = getattr(spec, 'origin', '') if spec else ''
        if spec is None:
            payload['summary'] = 'pywebview not installed'
        else:
            import webview
            version = getattr(webview, '__version__', 'unknown')
            payload['ok'] = True
            payload['summary'] = f'pywebview {{version}}'
            payload['origin'] = getattr(webview, '__file__', payload['origin'])
    else:
        spec = importlib.util.find_spec({module_name!r})
        payload['origin'] = getattr(spec, 'origin', '') if spec else ''
        if spec is None:
            payload['summary'] = 'not installed'
        else:
            mod = importlib.import_module({module_name!r})
            payload['ok'] = True
            payload['summary'] = 'installed'
            payload['origin'] = getattr(mod, '__file__', payload['origin'])
except Exception as exc:
    if {mode!r} == 'torch':
        payload['summary'] = f'Torch found but failed to import: {{exc.__class__.__name__}}'
    elif {mode!r} == 'webview':
        payload['summary'] = f'pywebview found but failed to import: {{exc.__class__.__name__}}'
    else:
        payload['summary'] = f'failed: {{exc.__class__.__name__}}'
    payload['error'] = f'{{exc.__class__.__name__}}: {{exc}}'

print(json.dumps(payload))
"""

    try:
        proc = subprocess.run(
            [utils.blender_python_executable(), '-c', code],
            capture_output=True,
            text=True,
            timeout=12,
        )
        stdout = proc.stdout.strip().splitlines()
        payload_line = stdout[-1] if stdout else ''
        payload = json.loads(payload_line) if payload_line else {}
    except Exception as exc:
        payload = {
            'ok': False,
            'summary': f'probe failed: {exc.__class__.__name__}',
            'origin': '',
            'error': f'{exc.__class__.__name__}: {exc}',
        }

    with _STATE_LOCK:
        _RUNTIME_PROBE_CACHE[cache_key] = (now, payload)
    return payload


def _probe_import_payloads(module_names: tuple[str, ...] | list[str]) -> dict[str, dict[str, object]]:
    requested = tuple(dict.fromkeys(module_names))
    paths = [str(path) for path in utils.shared_dependency_paths()]
    probe_targets = {name: _module_import_target(name) for name in requested}
    cache_key = (
        "imports:" + "|".join(f"{name}={probe_targets[name]}" for name in requested),
        os.pathsep.join(paths),
    )
    now = time.monotonic()
    with _STATE_LOCK:
        cached = _RUNTIME_PROBE_CACHE.get(cache_key)
    if cached and now - cached[0] <= _RUNTIME_PROBE_TTL:
        return cached[1]

    code = f"""
import importlib
import importlib.util
import json
import sys

paths = {paths!r}
probe_targets = {probe_targets!r}
for path in paths:
    if path in sys.path:
        sys.path.remove(path)
for path in reversed(paths):
    sys.path.insert(0, path)
importlib.invalidate_caches()
payload = {{}}

for public_name, target_name in probe_targets.items():
    item = {{'ok': False, 'summary': '', 'origin': '', 'error': ''}}
    try:
        spec = importlib.util.find_spec(target_name)
        item['origin'] = getattr(spec, 'origin', '') if spec else ''
        if spec is None:
            item['summary'] = 'not installed'
        else:
            mod = importlib.import_module(target_name)
            item['ok'] = True
            item['summary'] = 'installed'
            item['origin'] = getattr(mod, '__file__', item['origin'])
    except Exception as exc:
        item['summary'] = f'failed: {{exc.__class__.__name__}}'
        item['error'] = f'{{exc.__class__.__name__}}: {{exc}}'
    payload[public_name] = item

print(json.dumps(payload))
"""

    try:
        proc = subprocess.run(
            [utils.blender_python_executable(), "-c", code],
            capture_output=True,
            text=True,
            timeout=20,
        )
        stdout = proc.stdout.strip().splitlines()
        payload_line = stdout[-1] if stdout else ""
        payload = json.loads(payload_line) if payload_line else {}
    except Exception as exc:
        payload = {
            name: {
                "ok": False,
                "summary": f"probe failed: {exc.__class__.__name__}",
                "origin": "",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            for name in requested
        }

    with _STATE_LOCK:
        _RUNTIME_PROBE_CACHE[cache_key] = (now, payload)
    return payload


def _clear_runtime_probe_cache() -> None:
    with _STATE_LOCK:
        _RUNTIME_PROBE_CACHE.clear()


def _module_available(name: str) -> bool:
    signature = _dependency_signature()
    cache_key = (name, signature)
    now = time.monotonic()
    with _STATE_LOCK:
        cached = _MODULE_AVAILABILITY_CACHE.get(cache_key)
    if cached and now - cached[0] <= _STATUS_CACHE_TTL:
        return cached[1]

    payload = _probe_import_payloads((name,))
    available = bool((payload.get(name) or {}).get("ok"))

    with _STATE_LOCK:
        _MODULE_AVAILABILITY_CACHE[cache_key] = (now, available)
    return available


def missing_modules(group: DependencyGroup) -> list[str]:
    module_names = tuple(module_name for module_name, _ in requirement_entries(group))
    payload = _probe_import_payloads(module_names)
    return [module_name for module_name in module_names if not bool((payload.get(module_name) or {}).get("ok"))]


def group_status(group: DependencyGroup) -> tuple[bool, str]:
    current_platform = platform.system()

    if group.supported_platforms and current_platform not in group.supported_platforms:
        supported = ", ".join(group.supported_platforms)
        result = (False, f"This profile is for {supported}.")
    else:
        local_ready = _local_marker_group_ready(group)
        if local_ready is True:
            result = (True, "Installed")
        else:
            signature = f"{current_platform}|{_dependency_signature()}"
            cache_key = (group.key, signature)
            now = time.monotonic()
            with _STATE_LOCK:
                cached = _GROUP_STATUS_CACHE.get(cache_key)
            if cached and now - cached[0] <= _STATUS_CACHE_TTL:
                return cached[1]

            missing = missing_modules(group)
            if missing:
                result = (False, f"Missing: {', '.join(missing)}")
            else:
                result = (True, "Installed")

            with _STATE_LOCK:
                _GROUP_STATUS_CACHE[cache_key] = (now, result)
            return result
    return result


def torch_runtime_details() -> tuple[str, str]:
    payload = _probe_runtime_payload("torch", "torch")
    summary = str(payload.get("summary") or "Torch not installed")
    origin = str(payload.get("origin") or "")
    return summary, origin


def torch_device_summary() -> str:
    return torch_runtime_details()[0]


def webview_runtime_details() -> tuple[str, str]:
    payload = _probe_runtime_payload("webview", "webview")
    summary = str(payload.get("summary") or "pywebview not installed")
    origin = str(payload.get("origin") or "")
    if summary == "pywebview not installed":
        shared_wheels = [path for path in utils.shared_wheel_cache_paths() if list(path.glob("pywebview-*.whl"))]
        if shared_wheels:
            return "pywebview wheel available for install", str(shared_wheels[0])
    return summary, origin


def _torch_module():
    utils.bootstrap_dependency_paths()
    try:
        import torch
    except Exception:
        return None
    return torch


def _torch_cuda_available() -> bool:
    torch_mod = _torch_module()
    return bool(torch_mod and torch_mod.cuda.is_available())


def _torch_mps_available() -> bool:
    torch_mod = _torch_module()
    if torch_mod is None:
        return False
    backends = getattr(torch_mod, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    if mps_backend is None:
        return False
    is_built = getattr(mps_backend, "is_built", None)
    is_available = getattr(mps_backend, "is_available", None)
    built_ok = bool(is_built()) if callable(is_built) else True
    avail_ok = bool(is_available()) if callable(is_available) else False
    return built_ok and avail_ok


def _detect_nvidia_gpu() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return False, ""

    if result.returncode != 0:
        return False, ""

    first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    return bool(first_line), first_line


def _quick_host_profile() -> dict[str, object]:
    current_platform = platform.system()
    machine = platform.machine().lower()
    visible_keys = ["ui", "inference_base"]
    notes = ["Click Rescan to probe shared runtimes and confirm what is already available."]

    if current_platform == "Darwin":
        system_label = "macOS"
        backend_label = "Metal (MPS) likely" if machine in {"arm64", "aarch64"} else "CPU"
        if machine in {"arm64", "aarch64"}:
            visible_keys.append("apple_metal")
    elif current_platform == "Windows":
        system_label = "Windows"
        backend_label = "GPU not checked"
    elif current_platform == "Linux":
        system_label = "Linux"
        backend_label = "GPU not checked"
    else:
        system_label = current_platform
        backend_label = "GPU not checked"

    return {
        "platform": current_platform,
        "machine": machine,
        "system_label": system_label,
        "backend_label": backend_label,
        "visible_keys": tuple(visible_keys),
        "notes": tuple(notes),
    }


def _visible_groups_for_profile(profile: dict[str, object]) -> tuple[DependencyGroup, ...]:
    visible_keys = set(profile["visible_keys"])
    return tuple(group for group in DEPENDENCY_GROUPS if group.key in visible_keys)


def _tag_preferences_redraw() -> None:
    try:
        import bpy
    except Exception:
        return

    window_manager = getattr(bpy.context, "window_manager", None)
    if window_manager is None:
        return

    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if screen is None:
            continue
        for area in screen.areas:
            area.tag_redraw()


def _run_startup_scan() -> None:
    global _STARTUP_SCAN_REGISTERED
    _STARTUP_SCAN_REGISTERED = False
    try:
        refresh_status_snapshot(redraw=True)
    except Exception as exc:
        print(f"Hallway Avatar Gen startup scan failed: {exc}")


def schedule_startup_scan() -> None:
    global _STARTUP_SCAN_REGISTERED
    if _PANEL_SNAPSHOT_CACHE is not None or _STARTUP_SCAN_REGISTERED:
        return

    _STARTUP_SCAN_REGISTERED = True
    _run_startup_scan()


def cancel_startup_scan() -> None:
    global _STARTUP_SCAN_REGISTERED
    _STARTUP_SCAN_REGISTERED = False


def current_host_profile() -> dict[str, object]:
    current_platform = platform.system()
    machine = platform.machine().lower()
    has_cuda = _torch_cuda_available()
    has_mps = _torch_mps_available()
    has_nvidia, nvidia_name = _detect_nvidia_gpu()

    visible_keys = ["ui", "inference_base"]
    system_label = current_platform
    backend_label = "CPU"
    notes = []

    if current_platform == "Darwin":
        system_label = "macOS"
        if has_mps or machine in {"arm64", "aarch64"}:
            backend_label = "Metal (MPS)"
            visible_keys.append("apple_metal")
        else:
            notes.append("No Apple Metal wheel profile needed for this Mac.")
    elif current_platform == "Windows":
        system_label = "Windows"
        if has_cuda or has_nvidia:
            backend_label = f"CUDA ({nvidia_name})" if nvidia_name else "CUDA (NVIDIA)"
            visible_keys.append("windows_cuda")
        else:
            notes.append("No NVIDIA CUDA GPU detected on this Windows system.")
    elif current_platform == "Linux":
        system_label = "Linux"
        if has_cuda or has_nvidia:
            backend_label = f"CUDA ({nvidia_name})" if nvidia_name else "CUDA (NVIDIA)"
            visible_keys.append("linux_cuda")
        else:
            notes.append("No NVIDIA CUDA GPU detected on this Linux system.")
    else:
        notes.append("No GPU-specific wheel profile is defined for this platform.")

    return {
        "platform": current_platform,
        "machine": machine,
        "system_label": system_label,
        "backend_label": backend_label,
        "visible_keys": tuple(visible_keys),
        "notes": tuple(notes),
    }


def visible_dependency_groups() -> tuple[DependencyGroup, ...]:
    return _visible_groups_for_profile(current_host_profile())


def recommended_torch_group_key() -> str:
    profile = current_host_profile()
    for key in ("apple_metal", "windows_cuda", "linux_cuda"):
        if key in profile["visible_keys"]:
            return key
    return "apple_metal" if profile["platform"] == "Darwin" else "windows_cuda"


def preferences_snapshot(*, force_refresh: bool = False) -> dict[str, object]:
    global _PANEL_SNAPSHOT_CACHE

    with _STATE_LOCK:
        cached_snapshot = _PANEL_SNAPSHOT_CACHE

    if not force_refresh and cached_snapshot is not None:
        return cached_snapshot

    if not force_refresh:
        host_profile = _quick_host_profile()
        return {
            "legacy_addons": utils.legacy_hallway_addon_paths(),
            "torch_summary": "Not checked yet. Click Rescan.",
            "torch_origin": "",
            "webview_summary": "Not checked yet. Click Rescan.",
            "webview_origin": "",
            "shared_dependency_paths": (),
            "host_profile": host_profile,
            "groups": tuple(
                {
                    "group": group,
                    "installed": False,
                    "status_text": "Not checked yet. Click Rescan to probe shared packages.",
                }
                for group in _visible_groups_for_profile(host_profile)
            ),
            "shared_wheel_count": 0,
        }

    legacy_addons = utils.legacy_hallway_addon_paths(force_refresh=True)
    torch_summary, torch_origin = torch_runtime_details()
    webview_summary, webview_origin = webview_runtime_details()
    shared_paths = utils.shared_dependency_paths()[:3]
    host_profile = current_host_profile()
    groups = []
    for group in _visible_groups_for_profile(host_profile):
        installed, status_text = group_status(group)
        groups.append({
            "group": group,
            "installed": installed,
            "status_text": status_text,
        })

    snapshot = {
        "legacy_addons": legacy_addons,
        "torch_summary": torch_summary,
        "torch_origin": torch_origin,
        "webview_summary": webview_summary,
        "webview_origin": webview_origin,
        "shared_dependency_paths": shared_paths,
        "host_profile": host_profile,
        "groups": tuple(groups),
        "shared_wheel_count": len(utils.shared_wheel_cache_paths(force_refresh=True)),
    }
    with _STATE_LOCK:
        _PANEL_SNAPSHOT_CACHE = snapshot
    return snapshot


def install_state_snapshot() -> dict[str, object]:
    with _STATE_LOCK:
        return {
            "is_running": INSTALL_STATE.is_running,
            "group_key": INSTALL_STATE.group_key,
            "message": INSTALL_STATE.message,
            "current_line": INSTALL_STATE.current_line,
            "progress": INSTALL_STATE.progress,
            "log_path": INSTALL_STATE.log_path,
            "log_tail": INSTALL_STATE.log_tail,
            "log_lines": tuple(line for line in INSTALL_STATE.log_tail.splitlines() if line.strip()),
            "failure_summary": INSTALL_STATE.failure_summary,
            "last_return_code": INSTALL_STATE.last_return_code,
            "started_at": INSTALL_STATE.started_at,
        }


def _update_install_state(**kwargs) -> None:
    with _STATE_LOCK:
        for key, value in list(kwargs.items()):
            setattr(INSTALL_STATE, key, value)


def _pip_command_base() -> list[str]:
    return [utils.blender_python_executable(), "-m", "pip"]


def _find_links_args(paths: list[Path]) -> list[str]:
    args: list[str] = []
    for path in paths:
        args.extend(["--find-links", str(path)])
    return args


def _stream_process(command: list[str], log_file, *, cwd: Path | None = None) -> int:
    tail_lines: list[str] = []
    with subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                clean_line = line.rstrip()
                tail_lines.append(clean_line)
                if len(tail_lines) > 20:
                    tail_lines.pop(0)
                kwargs = {"log_tail": "\n".join(tail_lines)}
                if clean_line:
                    kwargs["current_line"] = clean_line
                    if clean_line.startswith("ERROR:"):
                        kwargs["failure_summary"] = clean_line
                _update_install_state(**kwargs)
        return process.wait()


def _retarget_install_command(command: list[str], target_dir: Path) -> list[str]:
    updated = list(command)
    if "--target" in updated:
        target_idx = updated.index("--target") + 1
        updated[target_idx] = str(target_dir)
    return updated


def _activate_staged_vendor(source_dir: Path, group_key: str) -> Path:
    if not source_dir.exists():
        raise RuntimeError(f"Staged install directory is missing: {source_dir}")

    overlays_root = utils.vendor_overlays_root_path(create=True)
    target_dir = overlays_root / f"{int(time.time())}-{group_key}"
    suffix = 1
    while target_dir.exists():
        suffix += 1
        target_dir = overlays_root / f"{int(time.time())}-{group_key}-{suffix}"
    shutil.move(str(source_dir), str(target_dir))
    return target_dir


def _write_filtered_requirements(group: DependencyGroup, destination_dir: Path) -> tuple[Path, list[str]]:
    entries = list(requirement_entries(group))
    if group.key in _STACK_GROUP_KEYS:
        selected_entries = entries
    else:
        payload = _probe_import_payloads(tuple(module_name for module_name, _ in entries))
        selected_entries = [
            (module_name, requirement_line)
            for module_name, requirement_line in entries
            if not bool((payload.get(module_name) or {}).get("ok"))
        ]
    if not selected_entries:
        raise RuntimeError(f"{group.label} does not have any missing requirements")

    fd, temp_name = tempfile.mkstemp(
        prefix=f"{group.key}-missing-",
        suffix=".txt",
        dir=str(destination_dir),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for _, requirement_line in selected_entries:
                handle.write(f"{requirement_line}\n")
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path, [module_name for module_name, _ in selected_entries]


def _remove_path(target: Path, failures: list[str] | None = None) -> None:
    if not target.exists():
        return
    try:
        if target.is_dir() and not target.is_symlink():
            for child in target.iterdir():
                _remove_path(child, failures)
            target.rmdir()
        else:
            target.unlink()
    except (PermissionError, OSError) as exc:
        if failures is not None:
            failures.append(f"{target}: {exc}")


def _purge_group_install_targets(group: DependencyGroup, vendor_dir: Path) -> list[str]:
    if group.key not in _STACK_GROUP_KEYS:
        return []

    prefixes = {"torch", "torchvision", "torchaudio", "functorch", "torchgen", "bitsandbytes"}
    failures: list[str] = []
    for prefix in prefixes:
        _remove_path(vendor_dir / prefix, failures)
        _remove_path(vendor_dir / f"{prefix}.libs", failures)
        for match in vendor_dir.glob(f"{prefix}-*.dist-info"):
            _remove_path(match, failures)
        for match in vendor_dir.glob(f"{prefix}-*.data"):
            _remove_path(match, failures)
    return failures


def _install_group_worker(group_key: str) -> None:
    group = get_group(group_key)
    vendor_dir = utils.vendor_path(create=True)
    wheel_cache = utils.wheel_cache_path(create=True)
    logs_dir = utils.logs_path(create=True)
    log_path = logs_dir / f"{group_key}-{int(time.time())}.log"
    shared_caches = utils.shared_wheel_cache_paths()

    _update_install_state(
        is_running=True,
        group_key=group_key,
        message=f"Installing {group.label}",
        current_line="Preparing install",
        progress=0.05,
        log_path=str(log_path),
        log_tail="",
        failure_summary="",
        started_at=time.time(),
        last_return_code=None,
    )

    install_from_cache_command = _pip_command_base() + [
        "install",
        "--upgrade",
        "--target",
        str(vendor_dir),
        "--no-index",
        "-r",
        str(requirements_path(group)),
    ] + _find_links_args([wheel_cache, *shared_caches])
    download_command = _pip_command_base() + [
        "download",
        "--dest",
        str(wheel_cache),
        "--only-binary=:all:",
        "-r",
        str(requirements_path(group)),
    ] + _find_links_args(shared_caches)
    install_command = _pip_command_base() + [
        "install",
        "--upgrade",
        "--target",
        str(vendor_dir),
        "--no-index",
        "-r",
        str(requirements_path(group)),
    ] + _find_links_args([wheel_cache, *shared_caches])
    if group.index_url:
        download_command.extend(["--extra-index-url", group.index_url])

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Installing dependency group: {group.label}\n")
        log_file.write(f"Requirements: {requirements_path(group)}\n\n")

        try:
            filtered_requirements_path, missing_imports = _write_filtered_requirements(group, logs_dir)
        except RuntimeError:
            _update_install_state(
                is_running=False,
                message=f"{group.label} is already available",
                last_return_code=0,
            )
            return

        log_file.write(f"Filtered requirements: {filtered_requirements_path}\n")
        log_file.write(f"Missing imports: {', '.join(missing_imports)}\n\n")

        staging_root = Path(tempfile.mkdtemp(prefix=f"{group.key}-stage-", dir=str(logs_dir)))
        staged_vendor_dir = staging_root / "_vendor"
        staged_vendor_dir.mkdir(parents=True, exist_ok=True)

        _update_install_state(message=f"{group.label}: preparing installer", current_line="Bootstrapping pip", progress=0.12)
        ensurepip_code = _stream_process(_pip_command_base()[:2] + ["ensurepip", "--upgrade"], log_file)
        if ensurepip_code != 0:
            filtered_requirements_path.unlink(missing_ok=True)
            shutil.rmtree(staging_root, ignore_errors=True)
            _update_install_state(
                is_running=False,
                message="ensurepip failed",
                current_line="ensurepip failed",
                progress=0.0,
                failure_summary="ensurepip failed",
                last_return_code=ensurepip_code,
            )
            return

        install_from_cache_command[install_from_cache_command.index(str(requirements_path(group)))] = str(filtered_requirements_path)
        download_command[download_command.index(str(requirements_path(group)))] = str(filtered_requirements_path)
        install_command[install_command.index(str(requirements_path(group)))] = str(filtered_requirements_path)
        staged_install_from_cache_command = _retarget_install_command(install_from_cache_command, staged_vendor_dir)
        staged_install_command = _retarget_install_command(install_command, staged_vendor_dir)

        _update_install_state(message=f"{group.label}: checking shared wheel caches", current_line="Checking shared wheel caches", progress=0.35)
        try:
            rc = _stream_process(staged_install_from_cache_command, log_file)
            if rc != 0:
                _update_install_state(message=f"{group.label}: downloading missing wheels", current_line="Downloading missing wheels", progress=0.65)
                rc = _stream_process(download_command, log_file)
                if rc == 0:
                    _update_install_state(message=f"{group.label}: installing downloaded wheels", current_line="Installing downloaded wheels", progress=0.88)
                    rc = _stream_process(staged_install_command, log_file)
            if rc == 0:
                _update_install_state(message=f"{group.label}: activating staged wheels", current_line="Activating staged dependency overlay", progress=0.94)
                try:
                    activated_path = _activate_staged_vendor(staged_vendor_dir, group.key)
                    log_file.write(f"Activated dependency overlay: {activated_path}\n")
                except Exception as exc:
                    rc = 1
                    sync_error = f"Staged activation failed: {exc.__class__.__name__}: {exc}"
                    log_file.write(f"{sync_error}\n")
                    _update_install_state(
                        current_line="Activating staged dependency overlay failed",
                        failure_summary=sync_error,
                    )
        finally:
            filtered_requirements_path.unlink(missing_ok=True)
            shutil.rmtree(staging_root, ignore_errors=True)

    message = f"Installed missing packages for {group.label}" if rc == 0 else f"Failed installing {group.label}"
    refresh_error = ""
    try:
        refresh_status_snapshot(redraw=True)
    except Exception as exc:
        refresh_error = f" Status refresh failed: {exc.__class__.__name__}."
    with _STATE_LOCK:
        failure_summary = INSTALL_STATE.failure_summary or INSTALL_STATE.current_line or message
    _update_install_state(
        is_running=False,
        message=f"{message}{refresh_error}",
        current_line=message,
        progress=1.0 if rc == 0 else 0.0,
        failure_summary=failure_summary if rc != 0 else "",
        last_return_code=rc,
    )


def install_group_async(group_key: str) -> tuple[bool, str]:
    if INSTALL_STATE.is_running:
        return False, f"Already installing {INSTALL_STATE.group_key}"

    group = get_group(group_key)
    installed, status_text = group_status(group)
    if installed:
        return False, f"{group.label} is already available ({status_text})"
    current_platform = platform.system()
    if group.supported_platforms and current_platform not in group.supported_platforms:
        supported = ", ".join(group.supported_platforms)
        return False, f"{group.label} is only supported on {supported}"

    worker = threading.Thread(target=_install_group_worker, args=(group_key,), daemon=True)
    INSTALL_STATE._thread = worker
    worker.start()
    return True, f"Started installing {group.label}"
