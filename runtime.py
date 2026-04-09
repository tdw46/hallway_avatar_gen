from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import bpy

from . import scene_builder, utils


WEBVIEW_PROCESS: subprocess.Popen | None = None
TIMER_REGISTERED = False


def _state(context=None):
    return utils.get_runtime_state(context)


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

        if payload.get("status") != "completed":
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


def register_runtime() -> None:
    global TIMER_REGISTERED
    if TIMER_REGISTERED:
        return
    bpy.app.timers.register(_poll_job_queue, first_interval=2.0, persistent=True)
    TIMER_REGISTERED = True


def stop_webview_process() -> None:
    global WEBVIEW_PROCESS
    if WEBVIEW_PROCESS and WEBVIEW_PROCESS.poll() is None:
        WEBVIEW_PROCESS.terminate()
    WEBVIEW_PROCESS = None
    state = _state()
    if state:
        state.webview_running = False


def unregister_runtime() -> None:
    global TIMER_REGISTERED
    if TIMER_REGISTERED:
        try:
            bpy.app.timers.unregister(_poll_job_queue)
        except ValueError:
            pass
        TIMER_REGISTERED = False
    stop_webview_process()


def ensure_webview_running(context=None) -> subprocess.Popen:
    global WEBVIEW_PROCESS
    state = _state(context)
    if WEBVIEW_PROCESS and WEBVIEW_PROCESS.poll() is None:
        if state:
            state.webview_running = True
        return WEBVIEW_PROCESS

    prefs = utils.get_addon_preferences(context)
    log_dir = utils.logs_path(create=True)
    log_path = log_dir / "webview.log"
    log_file = log_path.open("a", encoding="utf-8")

    vendor_dir = utils.vendor_path(create=True)
    env = os.environ.copy()
    pythonpath_parts = [str(vendor_dir), str(utils.package_root())]
    pythonpath_parts.extend(str(path) for path in utils.shared_dependency_paths() if path != vendor_dir)
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["HAG_VENDOR_DIR"] = str(vendor_dir)
    env["HAG_RUNTIME_DIR"] = str(utils.runtime_path(create=True))
    env["HAG_JOB_QUEUE_DIR"] = str(utils.job_queue_path(create=True))
    env["HAG_OUTPUT_DIR"] = str(utils.output_path(create=True))
    env["HAG_HF_HOME"] = str(utils.hf_cache_path(create=True))
    env["HAG_DEFAULT_DEVICE"] = getattr(prefs, "default_device", "auto")
    env["HAG_DEFAULT_QUANT_MODE"] = getattr(prefs, "default_quant_mode", "auto")
    env["HAG_DEFAULT_RESOLUTION"] = str(getattr(prefs, "default_resolution", 1024))

    WEBVIEW_PROCESS = subprocess.Popen(
        [utils.blender_python_executable(), str(utils.package_root() / "tools" / "blender_webview.py")],
        cwd=str(utils.package_root()),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if state:
        state.webview_running = True
    return WEBVIEW_PROCESS
