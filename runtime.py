from __future__ import annotations

import json
import os
import subprocess
import threading
from collections import deque
from pathlib import Path

import bpy

from . import scene_builder, utils, wheel_manager


WEBVIEW_PROCESS: subprocess.Popen | None = None
WEBVIEW_LOG_HANDLE = None
TIMER_REGISTERED = False
LOG_TIMER_REGISTERED = False
WEBVIEW_LOG_QUEUE: deque[str] = deque()
WEBVIEW_LOG_LOCK = threading.Lock()


def _state(context=None):
    return utils.get_runtime_state(context)


def _log_webview_line(line: str) -> None:
    clean = line.rstrip()
    if not clean:
        return
    with WEBVIEW_LOG_LOCK:
        WEBVIEW_LOG_QUEUE.append(clean)


def _flush_webview_log_queue(*, limit: int = 40) -> None:
    drained: list[str] = []
    with WEBVIEW_LOG_LOCK:
        while WEBVIEW_LOG_QUEUE and len(drained) < limit:
            drained.append(WEBVIEW_LOG_QUEUE.popleft())
    for clean in drained:
        print(f"Hallway Avatar Gen UI> {clean}", flush=True)


def _close_webview_log_handle() -> None:
    global WEBVIEW_LOG_HANDLE
    handle = WEBVIEW_LOG_HANDLE
    WEBVIEW_LOG_HANDLE = None
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


def _stream_webview_output(stream, log_handle) -> None:
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            try:
                log_handle.write(line)
                log_handle.flush()
            except Exception:
                pass
            _log_webview_line(line)
    finally:
        try:
            stream.close()
        except Exception:
            pass
        _close_webview_log_handle()


def _mark_processed(job_file: Path) -> None:
    processed_dir = job_file.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    job_file.rename(processed_dir / job_file.name)


def _poll_job_queue() -> float:
    queue_dir = utils.job_queue_path(create=True)
    prefs = utils.get_addon_preferences()
    state = _state()

    for job_file in sorted(queue_dir.glob("*.json")):
        try:
            payload = json.loads(job_file.read_text(encoding="utf-8"))
        except Exception:
            _mark_processed(job_file)
            continue

        if not payload.get("status") == "completed":
            _mark_processed(job_file)
            continue

        result_dir = payload.get("layer_dir") or payload.get("result_dir")
        if state:
            state.last_job_id = payload.get("job_id", "")
            state.last_output_dir = result_dir or ""

        if prefs and prefs.auto_import_results and result_dir:
            try:
                collection_name = scene_builder.import_result_directory(result_dir)
            except Exception as exc:
                print(f"Hallway Avatar Gen import failed: {exc}")
            else:
                if state:
                    state.last_import_collection = collection_name

        _mark_processed(job_file)

    if WEBVIEW_PROCESS and WEBVIEW_PROCESS.poll() is not None:
        state = _state()
        if state:
            state.webview_running = False

    return 2.0


def _poll_webview_log_queue() -> float:
    _flush_webview_log_queue()

    if WEBVIEW_PROCESS and WEBVIEW_PROCESS.poll() is not None:
        state = _state()
        if state:
            state.webview_running = False

    if WEBVIEW_PROCESS is None and not WEBVIEW_LOG_QUEUE:
        return 0.5
    return 0.25


def register_runtime() -> None:
    global TIMER_REGISTERED, LOG_TIMER_REGISTERED
    if not TIMER_REGISTERED:
        bpy.app.timers.register(_poll_job_queue, first_interval=2.0, persistent=True)
        TIMER_REGISTERED = True
    if not LOG_TIMER_REGISTERED:
        bpy.app.timers.register(_poll_webview_log_queue, first_interval=0.25, persistent=True)
        LOG_TIMER_REGISTERED = True


def stop_webview_process() -> None:
    global WEBVIEW_PROCESS
    if WEBVIEW_PROCESS and WEBVIEW_PROCESS.poll() is None:
        WEBVIEW_PROCESS.terminate()
    WEBVIEW_PROCESS = None
    _close_webview_log_handle()
    state = _state()
    if state:
        state.webview_running = False


def unregister_runtime() -> None:
    global TIMER_REGISTERED, LOG_TIMER_REGISTERED
    if TIMER_REGISTERED:
        try:
            bpy.app.timers.unregister(_poll_job_queue)
        except ValueError:
            pass
        TIMER_REGISTERED = False
    if LOG_TIMER_REGISTERED:
        try:
            bpy.app.timers.unregister(_poll_webview_log_queue)
        except ValueError:
            pass
        LOG_TIMER_REGISTERED = False
    stop_webview_process()


def _resolved_default_device(preferred: str) -> str:
    requested = (preferred or 'auto').strip().lower()
    if requested and requested != 'auto':
        return requested

    profile = wheel_manager.current_host_profile()
    visible = set(profile.get('visible_keys', ()))
    if 'windows_cuda' in visible or 'linux_cuda' in visible:
        return 'cuda'
    if 'apple_metal' in visible:
        return 'mps'
    return 'cpu'


def ensure_webview_running(context=None) -> subprocess.Popen:
    global WEBVIEW_PROCESS, WEBVIEW_LOG_HANDLE
    state = _state(context)
    if WEBVIEW_PROCESS and WEBVIEW_PROCESS.poll() is None:
        if state:
            state.webview_running = True
        return WEBVIEW_PROCESS

    prefs = utils.get_addon_preferences(context)
    log_dir = utils.logs_path(create=True)
    log_path = log_dir / "webview.log"
    log_file = log_path.open("w", encoding="utf-8")
    WEBVIEW_LOG_HANDLE = log_file

    vendor_dir = utils.vendor_path(create=True)
    shared_dependency_paths = utils.shared_dependency_paths()
    env = os.environ.copy()
    pythonpath_parts = [str(vendor_dir), str(utils.package_root())]
    pythonpath_parts.extend(str(path) for path in shared_dependency_paths if not path == vendor_dir)
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["HAG_VENDOR_DIR"] = str(vendor_dir)
    env["HAG_SHARED_DEP_PATHS"] = os.pathsep.join(str(path) for path in shared_dependency_paths)
    env["HAG_RUNTIME_DIR"] = str(utils.runtime_path(create=True))
    env["HAG_JOB_QUEUE_DIR"] = str(utils.job_queue_path(create=True))
    env["HAG_OUTPUT_DIR"] = str(utils.output_path(create=True))
    env["HAG_HF_HOME"] = str(utils.hf_cache_path(create=True))
    env["HAG_DEFAULT_DEVICE"] = _resolved_default_device(getattr(prefs, "default_device", "auto"))
    env["HAG_DEFAULT_QUANT_MODE"] = getattr(prefs, "default_quant_mode", "auto")
    env["HAG_DEFAULT_RESOLUTION"] = str(getattr(prefs, "default_resolution", 1024))
    env["PYTHONUNBUFFERED"] = "1"

    WEBVIEW_PROCESS = subprocess.Popen(
        [utils.blender_python_executable(), str(utils.package_root() / "tools" / "blender_webview.py")],
        cwd=str(utils.package_root()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if WEBVIEW_PROCESS.stdout is not None:
        threading.Thread(
            target=_stream_webview_output,
            args=(WEBVIEW_PROCESS.stdout, log_file),
            daemon=True,
        ).start()
    if state:
        state.webview_running = True
    return WEBVIEW_PROCESS
