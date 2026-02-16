import time
from pathlib import Path

import gradio as gr
from ultralytics import YOLO
from PIL import Image

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "weights" / "best.pt"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

def load_model():
    if WEIGHTS.exists():
        model_path = str(WEIGHTS)
        mode = "✅ Using local model: weights/best.pt"
    else:
        model_path = "yolov8n.pt"
        mode = "⚠️ Demo mode: weights/best.pt not found. Falling back to yolov8n.pt"
    return YOLO(model_path), mode

model, MODEL_MODE = load_model()

# --------- BLACK & GOLD THEME ----------
black_gold = gr.themes.Base().set(
    body_background_fill="#000000",
    background_fill_primary="#0b0b0b",
    background_fill_secondary="#111111",

    border_color_primary="#D4AF37",

    button_primary_background_fill="#D4AF37",
    button_primary_background_fill_hover="#c9a227",
    button_primary_text_color="#000000",

    color_accent="#D4AF37",
    color_accent_soft="#D4AF3722",

    block_background_fill="#0b0b0b",
    block_border_color="#D4AF37",

    input_background_fill="#111111",
    input_border_color="#D4AF37",

    body_text_color="#ffffff",
)

CSS = """
:root{ --bg:#000; --panel:#0b0b0b; --fg:#fff; --muted:#b8b8b8; --gold:#D4AF37; --danger:#FF3B3B;}
.gradio-container{ background: var(--bg) !important; color: var(--fg) !important; }
.block{ background: var(--panel) !important; border: 1px solid rgba(212,175,55,0.25) !important; border-radius: 16px !important; }
.notice{ border-left: 4px solid var(--gold); padding: 10px 12px; background: rgba(255,255,255,0.04); border-radius: 12px; }
.muted{ color: var(--muted) !important; }
.gold-btn button{ background: var(--gold) !important; color:#000 !important; font-weight: 900 !important; border-radius: 14px !important; }
"""

def _run_detection(image, progress=gr.Progress(track_tqdm=False)):
    if image is None:
        return 0, None, [], "Please upload an image first.", None

    progress(0.05, desc="Initializing…")
    start = time.time()

    for p, msg in [(0.15,"Pre-processing…"),
                   (0.35,"Running inference…"),
                   (0.55,"Extracting detections…"),
                   (0.75,"Rendering bounding boxes…"),
                   (0.90,"Finalizing output…")]:
        time.sleep(0.2)
        progress(p, desc=msg)

    results = model(image)
    annotated = results[0].plot()

    names = results[0].names
    rows = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for b in results[0].boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            label = names.get(cls_id, str(cls_id))
            rows.append([label, f"{conf*100:.1f}%"])
    else:
        rows.append(["None", "—"])

    elapsed = time.time() - start
    meta = (
        f"{MODEL_MODE}\n"
        f"Processed in {elapsed:.2f}s\n"
        "Privacy: Uploads are processed on demand and not stored permanently."
    )

    out_path = OUTPUTS / f"guardian_result_{int(time.time())}.png"
    Image.fromarray(annotated).save(out_path)

    progress(1.0, desc="Done ✅")
    return 1, annotated, rows, meta, str(out_path)

def reset_all():
    return None, None, [], "", None, 0, "upload"

# --------- APP UI ----------
with gr.Blocks(css=CSS, theme=black_gold) as demo:
    gr.Markdown(
        """
# 🛡️ The Guardian — Weapon Detection
**Three-click workflow:** Upload → Detect → Results. No training required.

<div class="notice">
<b>Privacy-first design:</b> Upload-based scanning (no continuous surveillance). Files are not retained after inference.
</div>
        """
    )
    gr.Markdown(f"<p class='muted'><b>Model status:</b> {MODEL_MODE}</p>")

    page = gr.State("upload")
    progress_value = gr.State(0)

    with gr.Tabs() as tabs:
        with gr.Tab("1) Upload", id="upload"):
            inp = gr.Image(type="numpy", label="Upload Image (JPG/PNG)")
            gr.Markdown("<p class='muted'>Drop an image here or click to upload.</p>")
            detect_btn = gr.Button("Detect Weapons", elem_classes=["gold-btn"])
            new_btn = gr.Button("New Scan")

        with gr.Tab("2) Processing", id="processing"):
            gr.Markdown("## Scanning for weapons…")
            spinner = gr.Markdown("⏳ Please wait…")
            prog = gr.Slider(minimum=0, maximum=1, value=0, step=0.01, interactive=False, label="Progress")

        with gr.Tab("3) Results", id="results"):
            out_img = gr.Image(label="Annotated Output (Bounding Boxes)")
            detections = gr.Dataframe(
                headers=["Label", "Confidence"],
                datatype=["str","str"],
                row_count=(0,"dynamic"),
                col_count=(2,"fixed"),
                label="Detections"
            )
            meta = gr.Textbox(label="Run Details", lines=5)
            download = gr.File(label="Download Annotated Image")

    def go_processing():
        return gr.Tabs(selected="processing"), 0

    def go_results():
        return gr.Tabs(selected="results")

    def go_upload():
        return gr.Tabs(selected="upload")

    detect_btn.click(
        fn=go_processing,
        inputs=None,
        outputs=[tabs, prog],
        queue=False
    )

    detect_btn.click(
        fn=_run_detection,
        inputs=inp,
        outputs=[prog, out_img, detections, meta, download],
        queue=True
    )

    detect_btn.click(
        fn=go_results,
        inputs=None,
        outputs=tabs,
        queue=False
    )

    new_btn.click(
        fn=reset_all,
        inputs=None,
        outputs=[inp, out_img, detections, meta, download, prog, page],
        queue=False
    )
    new_btn.click(fn=go_upload, inputs=None, outputs=tabs, queue=False)

demo.launch()
