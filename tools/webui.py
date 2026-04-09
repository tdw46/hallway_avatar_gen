"""Hallway Avatar Gen web UI.

This keeps the original Gradio-based workflow, but is now designed to be hosted
inside a native pywebview shell launched from Blender.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
from PIL import Image


SEETHROUGH_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = SEETHROUGH_ROOT / "inference" / "scripts" / "inference_psd_quantized.py"
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


def collect_layers(output_dir: str | os.PathLike[str]) -> list[tuple[str, str]]:
    output_path = Path(output_dir)
    if not output_path.is_dir():
        return []
    layers: list[tuple[str, str]] = []
    for file_path in output_path.iterdir():
        if file_path.suffix.lower() != ".png":
            continue
        tag = file_path.stem
        if tag.endswith("_depth") or tag in SKIP_TAGS:
            continue
        layers.append((str(file_path), tag))
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
    for keyword, label in STAGE_MARKERS:
        if keyword in tail:
            current_stage = label

    progress_line = ""
    for part in reversed(tail.split("\r")):
        part = part.strip()
        if not part:
            continue
        match = re.search(r"(\d+)%\|([^|]+)\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]", part)
        if match:
            pct, bar, cur, total, timing = match.groups()
            progress_line = f"{pct}% |{bar.strip()}| {cur}/{total} [{timing}]"
            break

    return f"{current_stage}\n{progress_line}" if progress_line else current_stage


def open_output_folder(output_path: str) -> None:
    target = Path(output_path) if output_path else OUTPUT_BASE
    target.mkdir(parents=True, exist_ok=True)
    if sys.platform.startswith("darwin"):
        subprocess.Popen(["open", str(target)])
    elif os.name == "nt":
        os.startfile(str(target))
    else:
        subprocess.Popen(["xdg-open", str(target)])


def _resolve_quant_mode(mode_str: str) -> str:
    if mode_str.startswith("NF4"):
        return "nf4"
    if mode_str.startswith("Full"):
        return "none"
    return os.environ.get("HAG_DEFAULT_QUANT_MODE", "auto")


def run_inference(image_path, mode_str, device, resolution, seed_val, tblr_split):
    if image_path is None:
        raise gr.Error("Upload an image first.")

    seed_val = int(seed_val)
    resolution = int(resolution)
    resolution = max(512, min(2048, round(resolution / 64) * 64))
    quant_mode = _resolve_quant_mode(mode_str)
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

    yield [], str(save_dir), "Starting inference"

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
            layers = collect_layers(layer_dir)
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
            log_status = parse_log_status(log_path)
            status_text = f"{log_status}\nElapsed: {elapsed_str}"
            if layers:
                status_text += f"\nLayers ready: {len(layers)}"
            yield layers, str(save_dir), status_text

    if process.returncode != 0:
        err_tail = log_path.read_text(encoding="utf-8", errors="ignore")[-1000:] if log_path.exists() else ""
        _job_payload(run_id, status="failed", save_dir=save_dir, layer_dir=layer_dir, error=err_tail)
        raise gr.Error(f"Inference failed.\n{err_tail}")

    gallery = collect_layers(layer_dir)
    elapsed = time.time() - start_time
    stats_path = layer_dir / "stats.json"
    stats_data = {}
    if stats_path.exists():
        try:
            stats_data = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            stats_data = {}

    peak_note = ""
    peak_gb = stats_data.get("peak_vram_gb")
    if peak_gb:
        peak_note = f" | Peak memory: {peak_gb:.2f} GB"

    _job_payload(run_id, status="completed", save_dir=save_dir, layer_dir=layer_dir)

    status = (
        f"Completed in {int(elapsed // 60)}m {int(elapsed % 60)}s\n"
        f"Device: {stats_data.get('device', device)} | Quant: {stats_data.get('quant_mode', quant_mode)}"
        f"{peak_note}\nOutput: {save_dir}"
    )
    yield gallery, str(save_dir), status


CUSTOM_CSS = """
.gallery-item img,
div[data-testid="image"] img {
    background-image:
        linear-gradient(45deg, #ddd 25%, transparent 25%),
        linear-gradient(-45deg, #ddd 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #ddd 75%),
        linear-gradient(-45deg, transparent 75%, #ddd 75%);
    background-size: 16px 16px;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0;
    background-color: #e8e8ed;
}
.header-text { text-align: center; padding: 0.5rem 0; }
#status-box textarea {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 0.85rem;
    line-height: 1.4;
}
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#fef7f0",
        c100="#fde8d4",
        c200="#fbd0a8",
        c300="#f5b47a",
        c400="#f0a050",
        c500="#e8985a",
        c600="#d88a4e",
        c700="#c07838",
        c800="#a0632e",
        c900="#7a4c24",
        c950="#5a3818",
    ),
    secondary_hue="orange",
    neutral_hue=gr.themes.Color(
        c50="#f9f9f7",
        c100="#f3f3f0",
        c200="#ededea",
        c300="#d8d8d5",
        c400="#b0b0ad",
        c500="#9090a0",
        c600="#5c5c72",
        c700="#4a4a5e",
        c800="#2d2d3a",
        c900="#1e1e28",
        c950="#141420",
    ),
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=["Consolas", "Fira Code", "monospace"],
)


def build_demo():
    with gr.Blocks(title="Hallway Avatar Gen") as demo:
        output_path_state = gr.State(value="")

        gr.Markdown(
            "# Hallway Avatar Gen\n"
            "Run the See-through layer decomposition pipeline in a Blender-friendly native window.",
            elem_classes=["header-text"],
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                input_image = gr.Image(type="filepath", label="Input Image", height=350)
                mode = gr.Radio(
                    choices=[
                        "Auto (recommended)",
                        "NF4 4-bit (CUDA only)",
                        "Full precision",
                    ],
                    value="Auto (recommended)",
                    label="Inference Mode",
                )
                device = gr.Dropdown(
                    choices=["auto", "cuda", "mps", "cpu"],
                    value=os.environ.get("HAG_DEFAULT_DEVICE", "auto"),
                    label="Device",
                )
                resolution = gr.Slider(
                    minimum=512,
                    maximum=2048,
                    step=64,
                    value=int(os.environ.get("HAG_DEFAULT_RESOLUTION", "1024")),
                    label="Resolution",
                )
                with gr.Row():
                    seed = gr.Number(value=42, label="Seed", precision=0, minimum=0)
                    tblr_split = gr.Checkbox(value=True, label="Split Left / Right")

            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Layer Preview",
                    columns=4,
                    height="auto",
                    object_fit="contain",
                )
                run_btn = gr.Button("Decompose", variant="primary", size="lg")
                status = gr.Textbox(label="Status", interactive=False, lines=4, elem_id="status-box")
                open_folder_btn = gr.Button("Open Output Folder", size="sm")

        run_btn.click(
            fn=run_inference,
            inputs=[input_image, mode, device, resolution, seed, tblr_split],
            outputs=[gallery, output_path_state, status],
        )
        open_folder_btn.click(fn=open_output_folder, inputs=[output_path_state])

    return demo


if __name__ == "__main__":
    build_demo().queue().launch(
        inbrowser=bool(os.environ.get("HAG_OPEN_BROWSER")),
        server_name="127.0.0.1",
        css=CUSTOM_CSS,
        theme=theme,
    )
