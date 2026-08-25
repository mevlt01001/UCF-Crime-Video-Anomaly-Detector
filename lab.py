"""Yerel smoke-test. Çalıştır: python3 lab.py"""
import os
import json
import time
from pathlib import Path

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ["no_proxy"] = os.environ.get("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import gradio as gr
import numpy as np
from gradio_client import utils as _gcu

from utils.llm import LLM_Manager
from utils.env import env_get

ROOT = Path(__file__).resolve().parent

# Gradio + pydantic: additionalProperties=true/false (bool) API şemasını kırıyor.
_orig_schema_type = _gcu._json_schema_to_python_type


def _safe_schema_type(schema, defs):
    if isinstance(schema, bool):
        return "Any"
    return _orig_schema_type(schema, defs)


_gcu._json_schema_to_python_type = _safe_schema_type

_llm = None
_vlm = None
_vlm_cls = None
OUT_DIR = Path("_stuff/lab_runs")


def llm():
    global _llm
    if _llm is None:
        _llm = LLM_Manager()
    return _llm


def vlm():
    global _vlm, _vlm_cls
    import importlib
    import utils.vlm as vlm_mod

    importlib.reload(vlm_mod)
    if _vlm is None or _vlm_cls is not vlm_mod.VLM_Manager:
        _vlm = vlm_mod.VLM_Manager()
        _vlm_cls = vlm_mod.VLM_Manager
    return _vlm


def _timed(fn):
    t0 = time.perf_counter()
    try:
        out = fn()
    except Exception as e:
        out = f"[HATA] {e}"
    ms = (time.perf_counter() - t0) * 1000
    return f"{out}\n\n---\nsüre: {ms:.0f} ms"


def _path(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    return getattr(obj, "name", None) or str(obj)


def run_llm(prompt, keep_history):
    if not (prompt or "").strip():
        return "Prompt boş."
    m = llm()
    if not keep_history:
        m.clear_history()
    return _timed(lambda: m.run(prompt.strip()))


def _frames_from_upload(path, max_frames):
    if not path:
        return None
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        img = cv2.imread(path)
        cap.release()
        if img is None:
            return None
        return np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

    count = max(1, min(int(max_frames), total))
    idxs = np.linspace(0, total - 1, count, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames) if frames else None


def run_vlm(prompt, image, video, max_frames, keep_history):
    path = _path(video) or _path(image)
    if not (prompt or "").strip() and path is None:
        return "Prompt veya medya gerekli."
    m = vlm()
    if not keep_history:
        m.reset_context()
    frames = _frames_from_upload(path, max_frames)

    def _call():
        return m.run((prompt or "").strip(), frames=frames)

    extra = "" if frames is None else f"\nframes: {frames.shape}"
    return _timed(_call) + extra


def run_analyzer(video, threshold, clip_size, overlap, fps):
    path = _path(video)
    if not path:
        return "Video yok.", None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = env_get("ANALYZER_FC_CHECKPOINT")
    ckpt_path = str((ROOT / ckpt).resolve()) if ckpt else None
    backbone = env_get("ANALYZER_BACKBONE", "s3d") or "s3d"
    clip_size = int(clip_size)
    overlap = int(overlap)
    fps = int(fps)
    if overlap >= clip_size:
        return "overlap, clip_size'dan küçük olmalı.", None

    def _call():
        import sys

        for name in (
            "utils.visualization_tools",
            "utils.video_process",
            "utils.fc_model",
            "utils.video_analyzer_model",
        ):
            sys.modules.pop(name, None)

        from utils.video_analyzer_model import Video_Analyzer, pick_device

        if ckpt_path and not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Checkpoint yok: {ckpt_path}")

        device = pick_device()
        model = Video_Analyzer(
            backbone,
            clip_size=clip_size,
            overlap=overlap,
            fc_layer_checkpoint=ckpt_path,
        )
        model.to(device)
        segments = model.analyze(
            video_path=path,
            width=224,
            height=224,
            fps=fps,
            batch_size=4,
            threshold=float(threshold),
            save_graph=True,
            save_clips=False,
            save_dir=str(OUT_DIR),
        )
        return {
            "device": str(device),
            "backbone": backbone,
            "checkpoint": ckpt_path,
            "clip_size": clip_size,
            "overlap": overlap,
            "stride": clip_size - overlap,
            "fps": fps,
            "segments": segments,
        }

    t0 = time.perf_counter()
    try:
        payload = _call()
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        text = f"[HATA] {e}"
    ms = (time.perf_counter() - t0) * 1000
    graph = None
    candidate = OUT_DIR / Path(path).stem / "segmentation_graph.png"
    if candidate.exists():
        graph = str(candidate)
    return f"{text}\n\n---\nsüre: {ms:.0f} ms", graph


with gr.Blocks(title="lab") as demo:
    gr.Markdown("Yerel test. Key `.env` içinde. Çıktı: `_stuff/lab_runs/`")
    with gr.Tab("LLM"):
        p = gr.Textbox(label="prompt", lines=3)
        h = gr.Checkbox(label="geçmişi tut", value=False)
        o = gr.Textbox(label="çıktı", lines=8)
        gr.Button("çalıştır").click(run_llm, [p, h], o)
    with gr.Tab("VLM"):
        vp = gr.Textbox(label="prompt", lines=3)
        image = gr.Image(label="görüntü (opsiyonel)", type="filepath")
        video = gr.Video(label="video (opsiyonel)")
        n = gr.Slider(1, 32, value=8, step=1, label="max frame")
        vh = gr.Checkbox(label="geçmişi tut", value=False)
        vo = gr.Textbox(label="çıktı", lines=10)
        gr.Button("çalıştır").click(run_vlm, [vp, image, video, n, vh], vo)
    with gr.Tab("Analyzer"):
        gr.Markdown("S3D + `Checkpoint/best_loss_fold_3.pt`. CUDA varsa GPU, yoksa MPS/CPU. Klip kaydetmez.")
        av = gr.Video(label="video")
        with gr.Row():
            clip_in = gr.Slider(4, 32, value=16, step=1, label="clip_size")
            overlap_in = gr.Slider(0, 24, value=8, step=1, label="overlap")
            fps_in = gr.Slider(5, 30, value=30, step=1, label="fps")
        th = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="threshold")
        ao = gr.Textbox(label="segmentler", lines=12)
        img = gr.Image(label="grafik", type="filepath")
        gr.Button("çalıştır").click(
            run_analyzer, [av, th, clip_in, overlap_in, fps_in], [ao, img]
        )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False,
        inbrowser=True,
    )
