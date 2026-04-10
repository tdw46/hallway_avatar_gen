from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from PIL import Image


SEETHROUGH_ROOT = Path(__file__).resolve().parent.parent
COMMON_ROOT = SEETHROUGH_ROOT / "common"
INFERENCE_ROOT = SEETHROUGH_ROOT / "inference"
SCRIPT_PATH = INFERENCE_ROOT / "scripts" / "inference_psd_quantized.py"
HF_CACHE_DIR = Path(os.environ.get("HAG_HF_HOME", SEETHROUGH_ROOT / ".hf_cache"))
OUTPUT_BASE = Path(os.environ.get("HAG_OUTPUT_DIR", SEETHROUGH_ROOT / "workspace" / "webui_output"))
JOB_QUEUE_DIR = Path(os.environ.get("HAG_JOB_QUEUE_DIR", ""))

SKIP_TAGS = {"src_img", "src_head", "reconstruction"}

LAYER_ORDER = [
    "front hair",
    "back hair",
    "head",
    "neck",
    "neckwear",
    "topwear",
    "handwear",
    "bottomwear",
    "legwear",
    "footwear",
    "tail",
    "wings",
    "objects",
    "headwear",
    "face",
    "irides",
    "eyebrow",
    "eyewhite",
    "eyelash",
    "eyewear",
    "ears",
    "earwear",
    "nose",
    "mouth",
]

STAGE_MARKERS = [
    ("Quantized inference:", "Preparing inference"),
    ("Building LayerDiff", "Building LayerDiff pipeline"),
    ("[NF4 fix]", "Applying NF4 compatibility fix"),
    ("Running LayerDiff", "Running layer decomposition"),
    ("LayerDiff3D done", "Layer decomposition complete"),
    ("Building Marigold", "Building depth pipeline"),
    ("Running Marigold", "Estimating depth"),
    ("Marigold done", "Depth estimation complete"),
    ("Running PSD assembly", "Assembling PSD and per-layer crops"),
    ("PSD assembly done", "PSD assembly complete"),
]


def _webui_log(level: str, message: str) -> None:
    print(f"Hallway Avatar Gen webui {level.upper()}: {message}", flush=True)


def _safe_error_tail(path: Path, *, limit: int = 1500) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[-limit:]
    except Exception as exc:
        _webui_log("warning", f"Failed reading log tail from {path}: {exc}")
        return ""


def _tag_sort_key(tag: str) -> int:
    try:
        return LAYER_ORDER.index(tag)
    except ValueError:
        return len(LAYER_ORDER)


def _job_payload(run_id: str, *, status: str, save_dir: Path, layer_dir: Path, error: str = "") -> None:
    if not str(JOB_QUEUE_DIR):
        return
    JOB_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": run_id,
        "status": status,
        "save_dir": str(save_dir),
        "layer_dir": str(layer_dir),
        "error": error,
    }
    (JOB_QUEUE_DIR / f"{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def collect_layers(output_dir: str | os.PathLike[str]) -> list[tuple[Path, str]]:
    output_path = Path(output_dir)
    if not output_path.is_dir():
        return []
    layers: list[tuple[Path, str]] = []
    for file_path in output_path.iterdir():
        if file_path.suffix.lower() != ".png":
            continue
        tag = file_path.stem
        if tag.endswith("_depth") or tag in SKIP_TAGS:
            continue
        layers.append((file_path, tag))
    layers.sort(key=lambda item: _tag_sort_key(item[1]))
    return layers


def parse_log_status(log_path: str | os.PathLike[str]) -> str:
    path = Path(log_path)
    if not path.exists():
        return "Initializing models"

    try:
        size = path.stat().st_size
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(max(0, size - 6000))
            tail = handle.read()
    except Exception:
        return "Initializing models"

    tail = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", tail)
    current_stage = "Initializing models"
    progress_line = ""
    progress_pattern = re.compile(r"(\d+)%\|([^|]+)\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]")

    for part in re.split(r"[\r\n]+", tail):
        part = part.strip()
        if not part:
            continue

        for keyword, label in STAGE_MARKERS:
            if keyword in part:
                current_stage = label
                progress_line = ""

        match = progress_pattern.search(part)
        if match:
            pct, bar, cur, total, timing = match.groups()
            progress_line = f"{pct}% |{bar.strip()}| {cur}/{total} [{timing}]"

    return f"{current_stage}\n{progress_line}" if progress_line else current_stage


def log_update_note(log_path: str | os.PathLike[str]) -> str:
    path = Path(log_path)
    if not path.exists():
        return ""
    try:
        age_seconds = max(0, int(time.time() - path.stat().st_mtime))
    except Exception:
        return ""
    if age_seconds < 2:
        return "Log updating now"
    if age_seconds < 15:
        return f"Last log update: {age_seconds}s ago"
    return f"Waiting on next log line: {age_seconds}s since last update"


def open_output_folder(output_path: str | os.PathLike[str]) -> None:
    target = Path(output_path) if output_path else OUTPUT_BASE
    target.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("darwin"):
        subprocess.Popen(["open", str(target)])
    elif os.name == "nt":
        os.startfile(str(target))
    else:
        subprocess.Popen(["xdg-open", str(target)])


def _resolve_quant_mode(mode_str: str) -> str:
    requested = (mode_str or "auto").strip().lower()
    if requested in {"nf4", "none", "auto"}:
        return requested
    return os.environ.get("HAG_DEFAULT_QUANT_MODE", "auto")


def _safe_image_stem(path_str: str) -> str:
    stem = Path(path_str).stem
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in stem) or "image"


def _convert_image_with_external_tool(source_path: Path, target_path: Path) -> bool:
    commands = []
    if sys.platform.startswith("darwin"):
        commands.append(["sips", "-s", "format", "png", str(source_path), "--out", str(target_path)])
    commands.extend(
        [
            ["magick", str(source_path), str(target_path)],
            ["dwebp", str(source_path), "-o", str(target_path)],
        ]
    )

    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True)
        except FileNotFoundError:
            continue
        except Exception as exc:
            _webui_log("warning", f"Image conversion command failed to launch ({command[0]}): {exc}")
            continue

        if result.returncode == 0 and target_path.exists():
            return True

        detail = (result.stderr or result.stdout or f"exit {result.returncode}").strip()
        if detail:
            _webui_log("warning", f"Image conversion command failed ({command[0]}): {detail[:400]}")

    return False


def _decode_data_url(image_value: str) -> bytes:
    header, _, encoded = image_value.partition(",")
    if not encoded or ";base64" not in header:
        raise ValueError("Unsupported image payload.")
    return base64.b64decode(encoded)


def _coerce_image_payload(image_input):
    if isinstance(image_input, (bytes, bytearray)):
        if not image_input:
            raise ValueError("Uploaded image is empty.")
        return {"kind": "bytes", "name": "uploaded image", "bytes": bytes(image_input)}

    if isinstance(image_input, dict):
        return image_input

    raw_value = str(image_input or "").strip()
    if not raw_value:
        return None

    if raw_value.startswith("{"):
        try:
            payload = json.loads(raw_value)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            kind = str(payload.get("kind") or payload.get("type") or "").lower()
            if kind == "path":
                return {"kind": "path", "path": str(payload.get("value") or payload.get("path") or "")}
            if kind in {"data_url", "data-url"}:
                return {
                    "kind": "bytes",
                    "name": str(payload.get("name") or "uploaded image"),
                    "bytes": _decode_data_url(str(payload.get("value") or payload.get("data") or "")),
                }

    if raw_value.startswith("data:"):
        return {"kind": "bytes", "name": "uploaded image", "bytes": _decode_data_url(raw_value)}

    return {"kind": "path", "path": raw_value}


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _path_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _load_preview_image(payload: dict[str, object], target_path: Path) -> tuple[Image.Image, str]:
    payload_kind = payload.get("kind")
    target_written = False
    temp_source_path = None

    if payload_kind == "path":
        source_path = Path(str(payload.get("path") or ""))
        source_label = source_path.name or "selected image"
        if not source_path.exists():
            raise FileNotFoundError(f"Selected image was not found: {source_label}")
        if source_path.stat().st_size <= 0:
            raise ValueError(f"Selected image is empty: {source_label}")
        try:
            preview_image = Image.open(source_path).convert("RGBA")
        except Exception:
            if not _convert_image_with_external_tool(source_path, target_path):
                raise
            preview_image = Image.open(target_path).convert("RGBA")
            target_written = True
    else:
        source_bytes = bytes(payload.get("bytes") or b"")
        source_label = str(payload.get("name") or "uploaded image")
        if not source_bytes:
            raise ValueError("Uploaded image is empty.")
        try:
            preview_image = Image.open(io.BytesIO(source_bytes)).convert("RGBA")
        except Exception:
            suffix = Path(source_label).suffix or ".img"
            temp_source_path = target_path.with_suffix(suffix)
            temp_source_path.write_bytes(source_bytes)
            if not _convert_image_with_external_tool(temp_source_path, target_path):
                raise
            preview_image = Image.open(target_path).convert("RGBA")
            target_written = True
        finally:
            if temp_source_path and temp_source_path.exists():
                try:
                    temp_source_path.unlink()
                except Exception:
                    pass

    if not target_written:
        preview_image.save(target_path, format="PNG")
    return preview_image, source_label


def prepare_input_image_payload(image_input) -> dict[str, object]:
    payload = _coerce_image_payload(image_input)
    if not payload:
        return {
            "selected_image_path": "",
            "selected_name": "No image selected.",
            "input_preview_url": "",
            "status_text": "Drag an image in or click to browse.",
            "error": "",
        }

    target_path = None
    source_label = "uploaded image"
    try:
        prepared_dir = OUTPUT_BASE / "_prepared_inputs"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        stem_seed = str(payload.get("path") or payload.get("name") or source_label)
        target_path = prepared_dir / f"{_safe_image_stem(stem_seed)}_{int(time.time() * 1000)}.png"
        preview_image, source_label = _load_preview_image(payload, target_path)
        preview_url = _image_to_data_url(preview_image)
    except Exception as exc:
        err_text = traceback.format_exc()
        _webui_log("error", f"Failed preparing input image {source_label}: {exc}\n{err_text}")
        raise RuntimeError(f"Unable to load image '{source_label}'. {exc}") from exc

    return {
        "selected_image_path": str(target_path),
        "selected_name": source_label,
        "input_preview_url": preview_url,
        "status_text": f"Ready: {source_label}\nConverted to PNG for compatibility.",
        "error": "",
    }


def _split_path_entries(raw_value: str) -> list[str]:
    if not raw_value:
        return []
    return [entry for entry in raw_value.split(os.pathsep) if entry]


def _dedupe_paths(paths: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in paths:
        normalized = os.path.normcase(os.path.normpath(entry))
        if not entry or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(entry)
    return deduped


def _current_overlay_paths_from_env() -> list[str]:
    vendor_raw = os.environ.get("HAG_VENDOR_DIR", "").strip()
    if not vendor_raw:
        return []

    overlays_root = Path(vendor_raw).parent / "_vendor_overlays"
    if not overlays_root.exists() or not overlays_root.is_dir():
        return []

    overlays: list[Path] = [child for child in overlays_root.iterdir() if child.is_dir()]
    overlays.sort(key=lambda item: (item.stat().st_mtime, item.name.lower()), reverse=True)
    return [str(path) for path in overlays]


def _subprocess_pythonpath() -> str:
    vendor_dir = os.environ.get("HAG_VENDOR_DIR", "").strip()
    shared_paths = _split_path_entries(os.environ.get("HAG_SHARED_DEP_PATHS", ""))
    existing = _split_path_entries(os.environ.get("PYTHONPATH", ""))
    parts = [
        *_current_overlay_paths_from_env(),
        vendor_dir,
        *shared_paths,
        str(COMMON_ROOT),
        str(INFERENCE_ROOT),
        str(SEETHROUGH_ROOT),
        *existing,
    ]
    return os.pathsep.join(_dedupe_paths(parts))


def _summarize_process_error(error_text: str) -> str:
    lines = [line.strip() for line in error_text.splitlines() if line.strip()]
    for prefix in (
        "ModuleNotFoundError:",
        "ImportError:",
        "RuntimeError:",
        "ValueError:",
        "TypeError:",
        "AssertionError:",
        "OSError:",
    ):
        for line in reversed(lines):
            if line.startswith(prefix):
                return line
    return lines[-1] if lines else "See Blender console for details."


def _gallery_payload(layer_dir: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for layer_path, label in collect_layers(layer_dir):
        try:
            image_url = _path_to_data_url(layer_path)
        except Exception as exc:
            _webui_log("warning", f"Failed loading gallery image {layer_path}: {exc}")
            continue
        items.append({"label": label, "image_url": image_url})
    return items


class HallwayWebApp:
    def __init__(self) -> None:
        self.window = None
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self.state: dict[str, object] = {
            "selected_image_path": "",
            "selected_name": "No image selected.",
            "input_preview_url": "",
            "gallery": [],
            "status_text": "Drag an image in or click to browse.",
            "error": "",
            "is_running": False,
            "output_dir": str(OUTPUT_BASE),
            "controls": {
                "device": os.environ.get("HAG_DEFAULT_DEVICE", "auto"),
                "quant_mode": os.environ.get("HAG_DEFAULT_QUANT_MODE", "auto"),
                "resolution": int(os.environ.get("HAG_DEFAULT_RESOLUTION", "1024")),
                "seed": 42,
                "tblr_split": True,
            },
        }

    def bind_window(self, window) -> None:
        self.window = window

    def on_console(self, payload):
        try:
            level = str((payload or {}).get("level") or "log").lower()
            args = (payload or {}).get("args") or []
            message = " ".join(str(item) for item in args).strip() or json.dumps(payload)
            if message == "Method not implemented.":
                return
            print(f"WebUI console[{level}]: {message}", flush=True)
        except Exception as error:
            print(f"WebUI console bridge error: {error} | payload={payload}", flush=True)

    def _snapshot(self) -> dict[str, object]:
        with self._lock:
            return json.loads(json.dumps(self.state))

    def get_state(self) -> dict[str, object]:
        return self._snapshot()

    def choose_image_file(self) -> str:
        window = self.window
        if window is None:
            try:
                import webview
                window = webview.windows[0]
            except Exception:
                window = None
        if window is None:
            return ""

        try:
            import webview
            result = window.create_file_dialog(
                webview.OPEN_DIALOG,
                allow_multiple=False,
                file_types=("Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.gif;*.tif;*.tiff)",),
            )
        except Exception as error:
            _webui_log("error", f"File dialog error: {error}")
            return ""

        if not result:
            return ""
        selected = result[0] if isinstance(result, (list, tuple)) else result
        return str(selected)

    def pick_image(self) -> dict[str, object]:
        selected = self.choose_image_file()
        if not selected:
            _webui_log("warning", "Input image picker closed without selection.")
            return self._snapshot()
        return self.set_image_from_payload({"kind": "path", "path": selected})

    def set_dropped_image(self, name: str, data_url: str) -> dict[str, object]:
        return self.set_image_from_payload({"kind": "data_url", "name": name, "value": data_url})

    def set_image_from_payload(self, payload: dict[str, object]) -> dict[str, object]:
        try:
            prepared = prepare_input_image_payload(payload)
        except Exception as exc:
            with self._lock:
                self.state["error"] = str(exc)
                self.state["status_text"] = str(exc)
            return self._snapshot()

        with self._lock:
            self.state.update(prepared)
            self.state["gallery"] = []
            self.state["output_dir"] = str(OUTPUT_BASE)
        return self._snapshot()

    def clear_image(self) -> dict[str, object]:
        with self._lock:
            self.state.update(
                {
                    "selected_image_path": "",
                    "selected_name": "No image selected.",
                    "input_preview_url": "",
                    "gallery": [],
                    "status_text": "Drag an image in or click to browse.",
                    "error": "",
                    "output_dir": str(OUTPUT_BASE),
                }
            )
        return self._snapshot()

    def set_controls(self, updates: dict[str, object] | None = None) -> dict[str, object]:
        updates = updates or {}
        with self._lock:
            controls = dict(self.state.get("controls") or {})
            if "device" in updates:
                controls["device"] = str(updates.get("device") or controls.get("device") or "auto")
            if "quant_mode" in updates:
                controls["quant_mode"] = str(updates.get("quant_mode") or controls.get("quant_mode") or "auto")
            if "resolution" in updates:
                resolution = int(updates.get("resolution") or controls.get("resolution") or 1024)
                controls["resolution"] = max(512, min(2048, round(resolution / 64) * 64))
            if "seed" in updates:
                controls["seed"] = int(updates.get("seed") or controls.get("seed") or 42)
            if "tblr_split" in updates:
                controls["tblr_split"] = bool(updates.get("tblr_split"))
            self.state["controls"] = controls
        return self._snapshot()

    def start_inference(self, options: dict[str, object] | None = None) -> dict[str, object]:
        options = options or {}
        with self._lock:
            if self.state.get("is_running"):
                self.state["error"] = "Inference is already running."
                return self._snapshot()
            image_path = str(self.state.get("selected_image_path") or "")
            if not image_path:
                self.state["error"] = "Choose an image first."
                self.state["status_text"] = "Choose an image first."
                return self._snapshot()

            controls = dict(self.state.get("controls") or {})
            controls.update(
                {
                    "device": str(options.get("device") or controls.get("device") or "auto"),
                    "quant_mode": str(options.get("quant_mode") or controls.get("quant_mode") or "auto"),
                    "resolution": int(options.get("resolution") or controls.get("resolution") or 1024),
                    "seed": int(options.get("seed") or controls.get("seed") or 42),
                    "tblr_split": bool(options.get("tblr_split", controls.get("tblr_split", True))),
                }
            )
            self.state["controls"] = controls
            self.state["is_running"] = True
            self.state["error"] = ""
            self.state["gallery"] = []
            self.state["status_text"] = "Starting inference"

        self._worker = threading.Thread(target=self._run_inference_worker, daemon=True)
        self._worker.start()
        return self._snapshot()

    def _set_state(self, **updates) -> None:
        with self._lock:
            self.state.update(updates)

    def _run_inference_worker(self) -> None:
        run_id = ""
        save_dir = OUTPUT_BASE
        layer_dir = OUTPUT_BASE
        log_path = OUTPUT_BASE / "webui.log"
        try:
            with self._lock:
                image_path = str(self.state.get("selected_image_path") or "")
                controls = dict(self.state.get("controls") or {})

            seed_val = int(controls.get("seed", 42))
            resolution = int(controls.get("resolution", 1024))
            resolution = max(512, min(2048, round(resolution / 64) * 64))
            quant_mode = _resolve_quant_mode(str(controls.get("quant_mode") or "auto"))
            device = str(controls.get("device") or "auto")
            tblr_split = bool(controls.get("tblr_split", True))
            img_stem = Path(image_path).stem

            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in img_stem) or "image"
            run_id = f"{safe_name}_{int(time.time())}"
            save_dir = OUTPUT_BASE / run_id
            save_dir.mkdir(parents=True, exist_ok=True)

            input_path = save_dir / f"{safe_name}.png"
            Image.open(image_path).convert("RGBA").save(str(input_path))

            layer_dir = save_dir / safe_name
            log_path = save_dir / "webui.log"
            _job_payload(run_id, status="running", save_dir=save_dir, layer_dir=layer_dir)

            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--srcp",
                str(input_path),
                "--save_to_psd",
                "--save_dir",
                str(save_dir),
                "--seed",
                str(seed_val),
                "--resolution",
                str(resolution),
                "--quant_mode",
                quant_mode,
                "--device",
                device,
                "--no_group_offload",
            ]
            if not tblr_split:
                command.append("--no_tblr_split")

            env = dict(os.environ)
            env["HF_HOME"] = str(HF_CACHE_DIR)
            env["PYTHONPATH"] = _subprocess_pythonpath()
            env["PYTHONUNBUFFERED"] = "1"
            env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            start_time = time.time()

            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=str(SEETHROUGH_ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

                while process.poll() is None:
                    time.sleep(2)
                    elapsed = time.time() - start_time
                    elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
                    log_status = parse_log_status(log_path)
                    update_note = log_update_note(log_path)
                    live_gallery = _gallery_payload(layer_dir)
                    layer_count = len(live_gallery)
                    status_text = f"{log_status}\nElapsed: {elapsed_str}"
                    if update_note:
                        status_text += f"\n{update_note}"
                    if layer_count:
                        status_text += f"\nLayers ready: {layer_count}"
                    self._set_state(
                        status_text=status_text,
                        output_dir=str(save_dir),
                        error="",
                        gallery=live_gallery,
                    )

            if process.returncode != 0:
                err_tail = _safe_error_tail(log_path)
                live_gallery = _gallery_payload(layer_dir)
                _webui_log("error", f"Inference failed for {input_path}.\n{err_tail}")
                _job_payload(run_id, status="failed", save_dir=save_dir, layer_dir=layer_dir, error=err_tail)
                self._set_state(
                    is_running=False,
                    error=_summarize_process_error(err_tail),
                    status_text=f"Inference failed.\n{_summarize_process_error(err_tail)}",
                    output_dir=str(save_dir),
                    gallery=live_gallery,
                )
                return

            stats_path = layer_dir / "stats.json"
            stats_data = {}
            if stats_path.exists():
                try:
                    stats_data = json.loads(stats_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    _webui_log("warning", f"Failed reading stats.json from {stats_path}: {exc}")

            gallery = _gallery_payload(layer_dir)
            elapsed = time.time() - start_time
            peak_note = ""
            peak_gb = stats_data.get("peak_vram_gb")
            if peak_gb:
                peak_note = f" | Peak memory: {peak_gb:.2f} GB"

            status = (
                f"Completed in {int(elapsed // 60)}m {int(elapsed % 60)}s\n"
                f"Device: {stats_data.get('device', device)} | Quant: {stats_data.get('quant_mode', quant_mode)}"
                f"{peak_note}\nOutput: {save_dir}"
            )
            _job_payload(run_id, status="completed", save_dir=save_dir, layer_dir=layer_dir)
            self._set_state(
                is_running=False,
                error="",
                gallery=gallery,
                status_text=status,
                output_dir=str(save_dir),
            )
        except Exception as exc:
            err_text = traceback.format_exc()
            _webui_log("error", f"Unhandled webui exception: {exc}\n{err_text}")
            if run_id:
                _job_payload(run_id, status="failed", save_dir=save_dir, layer_dir=layer_dir, error=err_text[-1500:])
            self._set_state(
                is_running=False,
                error=str(exc),
                status_text=f"Web UI error:\n{exc}",
                output_dir=str(save_dir),
            )

    def open_output_folder(self) -> dict[str, object]:
        with self._lock:
            output_dir = str(self.state.get("output_dir") or OUTPUT_BASE)
        open_output_folder(output_dir)
        return self._snapshot()


APP_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hallway Avatar Gen</title>
  <style>
    :root {
      --bg: #141420;
      --panel: #2d2d3a;
      --panel-strong: #3a3a4d;
      --panel-soft: #1e1e28;
      --ink: #f6f6f3;
      --muted: rgba(246, 246, 243, 0.72);
      --border: rgba(255,255,255,0.08);
      --accent: #e8985a;
      --accent-strong: #f0a050;
      --danger: #ff5f5f;
      --shadow: 0 18px 46px rgba(0,0,0,0.22);
      --radius: 20px;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      min-height: 100%;
      font-family: Inter, system-ui, sans-serif;
      background: radial-gradient(circle at top, #232335 0%, var(--bg) 56%);
      color: var(--ink);
    }
    body { padding: 18px; }
    .app {
      max-width: 1480px;
      margin: 0 auto;
    }
    .hero {
      text-align: center;
      margin-bottom: 16px;
    }
    .hero h1 {
      margin: 0;
      font-size: 2.45rem;
      line-height: 1.02;
    }
    .hero p {
      margin: 10px auto 0;
      max-width: 800px;
      color: var(--muted);
      font-size: 0.98rem;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(340px, 420px) minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    .left-stack,
    .right-stack {
      display: grid;
      gap: 16px;
      align-items: start;
    }
    .panel {
      background: linear-gradient(180deg, rgba(74,74,94,0.98) 0%, rgba(45,45,58,0.98) 100%);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 16px;
    }
    .section-title {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      padding: 0.45rem 0.82rem;
      border-radius: 0.8rem;
      background: var(--accent);
      color: white;
      font-weight: 700;
      font-size: 1.02rem;
      line-height: 1;
    }
    .dropzone {
      position: relative;
      margin-top: 12px;
      min-height: 110px;
      border-radius: 18px;
      border: 1.5px dashed rgba(232,152,90,0.38);
      background: rgba(20,20,32,0.18);
      padding: 14px 16px;
      cursor: pointer;
      transition: transform .16s ease, border-color .16s ease, background .16s ease;
      overflow: hidden;
    }
    .dropzone:hover,
    .dropzone.dragover {
      transform: translateY(-1px);
      border-color: rgba(240,160,80,0.92);
      background: rgba(240,160,80,0.08);
    }
    .dropzone h3 {
      margin: 0 0 6px;
      font-size: 1rem;
    }
    .dropzone p {
      margin: 0;
      color: var(--muted);
      line-height: 1.38;
      font-size: 0.92rem;
      max-width: calc(100% - 108px);
    }
    .dropzone button {
      position: absolute;
      top: 14px;
      right: 14px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.04);
      color: var(--ink);
      border-radius: 999px;
      padding: 9px 14px;
      font-weight: 600;
      cursor: pointer;
    }
    .dropzone button:hover { background: rgba(255,255,255,0.08); }
    .file-line {
      margin-top: 10px;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255,255,255,0.04);
      color: var(--ink);
      min-height: 42px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .file-line .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .file-line .ghost {
      border: none;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 0.95rem;
    }
    .preview {
      margin-top: 10px;
      width: 100%;
      aspect-ratio: 4 / 5;
      border-radius: 18px;
      background: linear-gradient(45deg, #ddd 25%, transparent 25%),
                  linear-gradient(-45deg, #ddd 25%, transparent 25%),
                  linear-gradient(45deg, transparent 75%, #ddd 75%),
                  linear-gradient(-45deg, transparent 75%, #ddd 75%);
      background-size: 16px 16px;
      background-position: 0 0, 0 8px, 8px -8px, -8px 0;
      background-color: #e8e8ed;
      display: grid;
      place-items: center;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .preview img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: none;
    }
    .preview.empty::after {
      content: 'Preview';
      color: rgba(20,20,32,0.48);
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .controls {
      display: grid;
      gap: 12px;
    }
    .control-card {
      border-radius: 16px;
      background: rgba(20,20,32,0.16);
      border: 1px solid rgba(255,255,255,0.06);
      padding: 12px;
    }
    .control-card label {
      display: inline-flex;
      align-items: center;
      margin-bottom: 8px;
      padding: 0.32rem 0.62rem;
      border-radius: 0.72rem;
      background: var(--accent);
      color: #fff;
      font-weight: 700;
      font-size: 0.9rem;
    }
    .toolbar-panel {
      display: grid;
      gap: 12px;
    }
    .compact-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      align-items: start;
    }
    .compact-span {
      grid-column: 1 / -1;
    }
    .radio-row, .inline-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .pill {
      flex: 1 1 0;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      color: var(--ink);
      border-radius: 12px;
      padding: 10px 12px;
      cursor: pointer;
      font-weight: 600;
      text-align: center;
      font-size: 0.96rem;
    }
    .pill.active {
      background: rgba(232,152,90,0.16);
      border-color: rgba(232,152,90,0.75);
    }
    select, input[type="number"], input[type="range"] {
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.06);
      color: var(--ink);
      padding: 10px 12px;
      font: inherit;
    }
    input[type="checkbox"] { transform: scale(1.15); }
    .resolution-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
    }
    .resolution-chip {
      min-width: 68px;
      text-align: center;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
      font-weight: 700;
    }
    .seed-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
    }
    .check-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
      color: var(--ink);
      font-size: 0.92rem;
      font-weight: 600;
      white-space: nowrap;
    }
    .primary {
      width: 100%;
      border: none;
      border-radius: 18px;
      padding: 18px 20px;
      background: var(--accent);
      color: white;
      font-size: 1.18rem;
      font-weight: 800;
      cursor: pointer;
      transition: transform .16s ease, filter .16s ease;
    }
    .primary:hover { transform: translateY(-1px); filter: brightness(1.03); }
    .primary:disabled { opacity: 0.55; cursor: wait; transform: none; }
    .secondary {
      width: 100%;
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px;
      padding: 12px 16px;
      background: rgba(255,255,255,0.05);
      color: var(--ink);
      font-weight: 600;
      cursor: pointer;
    }
    .status-card { min-height: 180px; }
    .status-card pre {
      margin: 14px 0 0;
      white-space: pre-wrap;
      font-family: Consolas, 'Courier New', monospace;
      font-size: 0.92rem;
      line-height: 1.48;
      color: var(--ink);
    }
    .alert {
      margin-top: 12px;
      border: 1px solid rgba(255,95,95,0.45);
      background: rgba(255,95,95,0.08);
      color: #ffd2d2;
      padding: 12px 14px;
      border-radius: 14px;
      display: none;
    }
    .gallery-grid {
      margin-top: 16px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
    }
    .gallery-item {
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.03);
    }
    .gallery-item img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      display: block;
      background: #e8e8ed;
    }
    .gallery-item .caption {
      padding: 10px 12px;
      font-size: 0.92rem;
      color: var(--muted);
      text-transform: capitalize;
    }
    .empty-state {
      margin-top: 14px;
      color: var(--muted);
      padding: 18px;
      border-radius: 16px;
      background: rgba(255,255,255,0.03);
      border: 1px dashed rgba(255,255,255,0.08);
    }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      body { padding: 18px; }
      .hero h1 { font-size: 2.4rem; }
      .compact-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="hero">
      <h1>Hallway Avatar Gen</h1>
      <p>Run the See-through layer decomposition pipeline in a Blender-friendly native window.</p>
    </div>

    <div class="layout">
      <div class="left-stack">
        <section class="panel">
          <div class="section-title">Input Image</div>
          <div id="dropzone" class="dropzone" tabindex="0">
            <button id="browse-btn" type="button">Browse</button>
            <h3>Drag and drop an image here</h3>
            <p>Or click anywhere in this card to open the file dialog. Supports PNG, JPG, WEBP, BMP, GIF, and TIFF.</p>
            <input id="browser-file-input" type="file" accept="image/png,image/jpeg,image/webp,image/bmp,image/gif,image/tiff,.png,.jpg,.jpeg,.webp,.bmp,.gif,.tif,.tiff" hidden />
          </div>
          <div class="file-line">
            <div id="selected-file" class="name">No image selected.</div>
            <button id="clear-btn" class="ghost" type="button">Clear</button>
          </div>
          <div id="preview" class="preview empty"><img id="preview-img" alt="Input preview" /></div>
        </section>
      </div>

      <div class="right-stack">
        <section class="panel toolbar-panel">
          <div class="compact-grid">
            <div class="control-card compact-span">
              <label>Inference Mode</label>
              <div class="radio-row">
                <button class="pill active" data-mode="auto" type="button">Auto</button>
                <button class="pill" data-mode="nf4" type="button">NF4 4-bit</button>
                <button class="pill" data-mode="none" type="button">Full Precision</button>
              </div>
            </div>

            <div class="control-card">
              <label>Device</label>
              <select id="device-select">
                <option value="auto">Auto</option>
                <option value="cuda">CUDA</option>
                <option value="mps">Apple Metal</option>
                <option value="cpu">CPU</option>
              </select>
            </div>

            <div class="control-card">
              <label>Resolution</label>
              <div class="resolution-row">
                <input id="resolution-input" type="range" min="512" max="2048" step="64" value="1024" />
                <div id="resolution-value" class="resolution-chip">1024</div>
              </div>
            </div>

            <div class="control-card compact-span">
              <div class="seed-row">
                <div>
                  <label>Seed</label>
                  <input id="seed-input" type="number" min="0" step="1" value="42" />
                </div>
                <label class="check-chip">
                  <input id="tblr-split" type="checkbox" checked /> Split Left / Right
                </label>
              </div>
            </div>
          </div>

          <button id="run-btn" class="primary" type="button">Decompose</button>
        </section>

        <section class="panel status-card">
          <div class="section-title">Status</div>
          <div id="error-box" class="alert"></div>
          <pre id="status-text">Waiting for UI bridge…</pre>
        </section>

        <section class="panel">
          <div class="section-title">Layer Preview</div>
          <div id="gallery-empty" class="empty-state">No generated layers yet.</div>
          <div id="gallery-grid" class="gallery-grid"></div>
          <div style="height:14px"></div>
          <button id="open-folder-btn" class="secondary" type="button">Open Output Folder</button>
        </section>
      </div>
    </div>
  </div>

  <script>
    (() => {
      const refs = {
        dropzone: document.getElementById('dropzone'),
        browseBtn: document.getElementById('browse-btn'),
        browserInput: document.getElementById('browser-file-input'),
        clearBtn: document.getElementById('clear-btn'),
        selectedFile: document.getElementById('selected-file'),
        preview: document.getElementById('preview'),
        previewImg: document.getElementById('preview-img'),
        statusText: document.getElementById('status-text'),
        errorBox: document.getElementById('error-box'),
        galleryEmpty: document.getElementById('gallery-empty'),
        galleryGrid: document.getElementById('gallery-grid'),
        runBtn: document.getElementById('run-btn'),
        openFolderBtn: document.getElementById('open-folder-btn'),
        deviceSelect: document.getElementById('device-select'),
        resolutionInput: document.getElementById('resolution-input'),
        resolutionValue: document.getElementById('resolution-value'),
        seedInput: document.getElementById('seed-input'),
        tblrSplit: document.getElementById('tblr-split'),
        modeButtons: Array.from(document.querySelectorAll('[data-mode]')),
      };

      let selectedMode = 'auto';
      let currentState = null;
      let pollTimer = null;
      let apiPromise = null;
      let controlsDirty = false;
      let controlsDirtyTimer = null;

      function getApi() {
        if (window.pywebview && window.pywebview.api) {
          return Promise.resolve(window.pywebview.api);
        }
        if (!apiPromise) {
          apiPromise = new Promise((resolve) => {
            window.addEventListener('pywebviewready', () => resolve(window.pywebview.api), { once: true });
          });
        }
        return apiPromise;
      }

      async function callApi(method, ...args) {
        const api = await getApi();
        return api[method](...args);
      }

      function updateModeButtons() {
        refs.modeButtons.forEach((button) => {
          button.classList.toggle('active', button.dataset.mode === selectedMode);
        });
      }

      function renderGallery(items) {
        refs.galleryGrid.innerHTML = '';
        if (!items || !items.length) {
          refs.galleryEmpty.style.display = 'block';
          return;
        }
        refs.galleryEmpty.style.display = 'none';
        for (const item of items) {
          const card = document.createElement('div');
          card.className = 'gallery-item';
          const img = document.createElement('img');
          img.src = item.image_url;
          img.alt = item.label;
          const caption = document.createElement('div');
          caption.className = 'caption';
          caption.textContent = item.label;
          card.appendChild(img);
          card.appendChild(caption);
          refs.galleryGrid.appendChild(card);
        }
      }

      function renderState(state) {
        currentState = state || {};
        const controls = currentState.controls || {};
        refs.selectedFile.textContent = currentState.selected_name || 'No image selected.';
        refs.statusText.textContent = currentState.status_text || 'Drag an image in or click to browse.';
        refs.errorBox.textContent = currentState.error || '';
        refs.errorBox.style.display = currentState.error ? 'block' : 'none';
        const previewUrl = currentState.input_preview_url || '';
        if (previewUrl) {
          refs.previewImg.src = previewUrl;
          refs.previewImg.style.display = 'block';
          refs.preview.classList.remove('empty');
        } else {
          refs.previewImg.removeAttribute('src');
          refs.previewImg.style.display = 'none';
          refs.preview.classList.add('empty');
        }
        if (!controlsDirty || currentState.is_running) {
          selectedMode = controls.quant_mode || selectedMode || 'auto';
          updateModeButtons();
          refs.deviceSelect.value = controls.device || 'auto';
          refs.resolutionInput.value = String(controls.resolution || 1024);
          refs.resolutionValue.textContent = String(controls.resolution || 1024);
          refs.seedInput.value = String(controls.seed || 42);
          refs.tblrSplit.checked = Boolean(controls.tblr_split);
        }
        refs.runBtn.disabled = Boolean(currentState.is_running);
        refs.runBtn.textContent = currentState.is_running ? 'Decomposing…' : 'Decompose';
        if (currentState.is_running) {
          controlsDirty = false;
          if (controlsDirtyTimer) {
            window.clearTimeout(controlsDirtyTimer);
            controlsDirtyTimer = null;
          }
        }
        renderGallery(currentState.gallery || []);
      }

      async function refreshState() {
        try {
          renderState(await callApi('get_state'));
        } catch (error) {
          console.error('refresh-state-failed', error && (error.message || error));
        }
      }

      async function pickImage() {
        console.warn('ui-pick-image');
        try {
          renderState(await callApi('pick_image'));
        } catch (error) {
          console.error('ui-pick-image-failed', error && (error.message || error));
        }
      }

      async function clearImage() {
        renderState(await callApi('clear_image'));
      }

      async function sendDroppedFile(file) {
        if (!file) {
          return;
        }
        console.warn('ui-drop-file', file.name || 'unknown', file.size || 0);
        const reader = new FileReader();
        reader.onload = async () => {
          try {
            renderState(await callApi('set_dropped_image', file.name || 'uploaded image', String(reader.result || '')));
          } catch (error) {
            console.error('ui-drop-file-failed', error && (error.message || error));
          }
        };
        reader.onerror = () => {
          console.error('ui-drop-file-read-error', file.name || 'unknown');
        };
        reader.readAsDataURL(file);
      }

      async function startInference() {
        const payload = {
          quant_mode: selectedMode,
          device: refs.deviceSelect.value,
          resolution: Number(refs.resolutionInput.value || 1024),
          seed: Number(refs.seedInput.value || 42),
          tblr_split: refs.tblrSplit.checked,
        };
        console.warn('ui-start-inference', JSON.stringify(payload));
        try {
          renderState(await callApi('start_inference', payload));
        } catch (error) {
          console.error('ui-start-inference-failed', error && (error.message || error));
        }
      }

      async function openOutputFolder() {
        try {
          await callApi('open_output_folder');
        } catch (error) {
          console.error('ui-open-output-folder-failed', error && (error.message || error));
        }
      }

      function markControlsDirty() {
        controlsDirty = true;
        if (controlsDirtyTimer) {
          window.clearTimeout(controlsDirtyTimer);
        }
        controlsDirtyTimer = window.setTimeout(() => {
          controlsDirty = false;
          controlsDirtyTimer = null;
        }, 1200);
      }

      async function syncControls() {
        markControlsDirty();
        try {
          await callApi('set_controls', {
            quant_mode: selectedMode,
            device: refs.deviceSelect.value,
            resolution: Number(refs.resolutionInput.value || 1024),
            seed: Number(refs.seedInput.value || 42),
            tblr_split: refs.tblrSplit.checked,
          });
        } catch (error) {
          console.error('ui-sync-controls-failed', error && (error.message || error));
        }
      }

      refs.dropzone.addEventListener('click', (event) => {
        if (event.target === refs.browserInput) {
          return;
        }
        pickImage();
      });
      refs.dropzone.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          pickImage();
        }
      });
      refs.browseBtn.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        pickImage();
      });
      refs.browserInput.addEventListener('change', () => {
        const file = refs.browserInput.files && refs.browserInput.files[0];
        if (file) {
          sendDroppedFile(file);
        }
        refs.browserInput.value = '';
      });
      refs.clearBtn.addEventListener('click', (event) => {
        event.preventDefault();
        clearImage();
      });
      refs.runBtn.addEventListener('click', startInference);
      refs.openFolderBtn.addEventListener('click', openOutputFolder);
      refs.resolutionInput.addEventListener('input', () => {
        refs.resolutionValue.textContent = refs.resolutionInput.value;
        syncControls();
      });
      refs.deviceSelect.addEventListener('change', syncControls);
      refs.seedInput.addEventListener('input', syncControls);
      refs.tblrSplit.addEventListener('change', syncControls);
      refs.modeButtons.forEach((button) => {
        button.addEventListener('click', () => {
          selectedMode = button.dataset.mode || 'auto';
          updateModeButtons();
          syncControls();
        });
      });
      ['dragenter', 'dragover'].forEach((eventName) => {
        refs.dropzone.addEventListener(eventName, (event) => {
          event.preventDefault();
          event.stopPropagation();
          refs.dropzone.classList.add('dragover');
        });
      });
      ['dragleave', 'dragend'].forEach((eventName) => {
        refs.dropzone.addEventListener(eventName, (event) => {
          event.preventDefault();
          event.stopPropagation();
          refs.dropzone.classList.remove('dragover');
        });
      });
      refs.dropzone.addEventListener('drop', (event) => {
        event.preventDefault();
        event.stopPropagation();
        refs.dropzone.classList.remove('dragover');
        const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
        if (file) {
          sendDroppedFile(file);
        }
      });

      async function bootUiBridge() {
        console.warn('ui-ready');
        await refreshState();
        if (pollTimer) {
          window.clearInterval(pollTimer);
        }
        pollTimer = window.setInterval(refreshState, 1000);
      }

      if (window.pywebview && window.pywebview.api) {
        bootUiBridge();
      } else {
        window.addEventListener('pywebviewready', bootUiBridge, { once: true });
        const readyPoll = window.setInterval(() => {
          if (window.pywebview && window.pywebview.api) {
            window.clearInterval(readyPoll);
            bootUiBridge();
          }
        }, 150);
        window.setTimeout(() => window.clearInterval(readyPoll), 5000);
      }
    })();
  </script>
</body>
</html>
"""
