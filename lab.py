"""Yerel smoke-test. venv: source .venv/bin/activate && python lab.py"""
import os
import json
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ["no_proxy"] = os.environ.get("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import gradio as gr
import numpy as np
from gradio_client import utils as _gcu

from utils.llm import LLM_Manager
from utils.env import env_first, env_int

ROOT = Path(__file__).resolve().parent

# Gradio 4 schema bool bug. Gradio 6'da bu helper yok/değişmiş olabilir.
if hasattr(_gcu, "_json_schema_to_python_type"):
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
        return None, None
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if total <= 0:
        img = cv2.imread(path)
        cap.release()
        if img is None:
            return None, None
        return np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0), None

    count = max(1, min(int(max_frames), total))
    idxs = np.linspace(0, total - 1, count, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    source_duration = total / source_fps if source_fps > 0 else None
    return (np.stack(frames) if frames else None), source_duration


def run_vlm(prompt, image, video, max_frames, keep_history):
    path = _path(video) or _path(image)
    if not (prompt or "").strip() and path is None:
        return "Prompt veya medya gerekli."
    m = vlm()
    if not keep_history:
        m.reset_context()
    frames, source_duration = _frames_from_upload(path, max_frames)

    def _call():
        return m.run(
            (prompt or "").strip(),
            frames=frames,
            source_duration=source_duration,
        )

    extra = "" if frames is None else f"\nframes: {frames.shape}"
    return _timed(_call) + extra


def _resolve_analyzer_checkpoint() -> Optional[str]:
    for candidate in (
        env_first("AS_FC_CHECKPOINT", "ANALYZER_FC_CHECKPOINT"),
        "Checkpoint/best_loss_fold_3.pt",
    ):
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return str(path)
    return None


def run_analyzer(video, threshold):
    path = _path(video)
    if not path:
        return "Video yok.", None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clip_size = env_int("AS_CLIP_SIZE", 16)
    overlap = env_int("AS_OVERLAP", 8)
    fps = env_int("AS_FPS", 30)
    width = env_int("AS_WIDTH", 224)
    height = env_int("AS_HEIGHT", 224)
    batch = env_int("AS_BATCH", 4)
    backbone = env_first("AS_MODEL_NAME", "ANALYZER_BACKBONE") or "s3d"
    ckpt_path = _resolve_analyzer_checkpoint()
    if overlap >= clip_size:
        return "overlap, clip_size'dan küçük olmalı (.env).", None

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
            width=width,
            height=height,
            fps=fps,
            batch_size=batch,
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
            "width": width,
            "height": height,
            "batch": batch,
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


def _tool_calls_line(msg) -> str:
    calls = getattr(msg, "tool_calls", None) or []
    if not calls:
        return ""
    parts = []
    for call in calls:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "?")
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        parts.append(f"{name}({args})")
    return " | ".join(parts)


def _content_preview(msg, limit: int = 600) -> str:
    content = getattr(msg, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    content = content.strip()
    if len(content) > limit:
        return content[:limit] + "…"
    return content


def run_agent(message, history, lc_messages, video):
    from langchain_core.messages import AIMessage, HumanMessage

    message = (message or "").strip()
    history = list(history or [])
    lc_messages = list(lc_messages or [])
    if not message:
        return history, "", lc_messages, "Mesaj boş."

    path = _path(video)
    content = message if not path else f"{message}\n\n[Hedef video: {path}]"
    # Bu state sadece temiz ana sohbet hafızasıdır. Planner/Executor/Tool mesajları
    # graph'ın tek görevlik `messages` alanında kalır ve sonraki tura taşınmaz.
    conversation_messages = lc_messages + [HumanMessage(content=content)]

    from utils.agents import video_agent_app

    initial_state = {
        "user_query": message,
        "video_path": path or "",
        "video_paths": [path] if path else [],
        "image_paths": [],
        "plan": "",
        "conversation_messages": conversation_messages,
        "messages": [],
        "feedback": "",
        "review_route": "",
        "final_answer": "",
        "tool_rounds": 0,
        "review_loops": 0,
    }

    traces = []
    final_answer = ""
    t0 = time.perf_counter()
    try:
        for event in video_agent_app.stream(initial_state, {"recursion_limit": 40}):
            for node_name, update in event.items():
                line = f"[{node_name}]"
                if node_name == "planner" and update.get("plan"):
                    line += f"\n{update['plan']}"
                elif node_name == "executor":
                    msgs = update.get("messages") or []
                    last = msgs[-1] if msgs else None
                    calls = _tool_calls_line(last) if last else ""
                    if calls:
                        line += f" tool: {calls}"
                    else:
                        preview = _content_preview(last) if last else ""
                        if preview:
                            line += f"\n{preview}"
                elif node_name == "tools":
                    msgs = update.get("messages") or []
                    for tool_msg in msgs:
                        preview = _content_preview(tool_msg, 800)
                        line += f"\n{preview}"
                elif node_name == "reviewer":
                    if update.get("feedback"):
                        line += f" feedback: {update['feedback']}"
                    if update.get("final_answer"):
                        final_answer = update["final_answer"]
                        line += f"\n{final_answer}"
                traces.append(line)
    except Exception as e:
        final_answer = f"[HATA] {e}"
        traces.append(final_answer)

    ms = (time.perf_counter() - t0) * 1000
    if not final_answer:
        final_answer = "(nihai cevap yok — trace'e bak)"
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": final_answer})
    lc_messages = conversation_messages + [AIMessage(content=final_answer)]
    trace_text = "\n\n".join(traces) + f"\n\n---\nsüre: {ms:.0f} ms"
    return history, "", lc_messages, trace_text


def clear_agent():
    return [], "", [], "Sohbet sıfırlandı."


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
        vlm_max_frames = env_int("VLM_MAX_FRAMES", 128)
        n = gr.Slider(1, vlm_max_frames, value=vlm_max_frames, step=1, label="VLM max frame")
        vh = gr.Checkbox(label="geçmişi tut", value=False)
        vo = gr.Textbox(label="çıktı", lines=10)
        gr.Button("çalıştır").click(run_vlm, [vp, image, video, n, vh], vo)
    with gr.Tab("Analyzer"):
        gr.Markdown(
            "clip_size / overlap / fps / batch / çözünürlük `.env` (`AS_*`). "
            "CUDA varsa GPU, yoksa MPS/CPU. Klip kaydetmez."
        )
        av = gr.Video(label="video")
        th = gr.Slider(0.05, 0.9, value=0.3, step=0.05, label="threshold")
        ao = gr.Textbox(label="segmentler", lines=12)
        img = gr.Image(label="grafik", type="filepath")
        gr.Button("çalıştır").click(run_analyzer, [av, th], [ao, img])
    with gr.Tab("Agent"):
        gr.Markdown(
            "LangGraph smoke: planner → executor → tool → reviewer. "
            "Video yükle, yola mesajda gerek yok; state `video_path` olarak gider. "
            "Analyzer tool ilk çağrıda modeli yükler (yavaş)."
        )
        agent_video = gr.Video(label="hedef video (opsiyonel)")
        agent_chat = gr.Chatbot(label="sohbet", height=360)
        agent_msg = gr.Textbox(
            label="mesaj",
            placeholder="merhaba  |  video kaç saniye  |  anomalileri bul  |  12-18. saniyede ne oluyor",
            lines=2,
        )
        agent_lc = gr.State([])
        with gr.Row():
            agent_send = gr.Button("gönder", variant="primary")
            agent_clear = gr.Button("sohbeti sil")
        agent_trace = gr.Textbox(label="node trace", lines=14)
        gr.Examples(
            examples=[
                ["merhaba"],
                ["bu videonun süresini ve fps değerini söyle"],
                ["bu videoda anormal bir durum var mı, varsa kaçıncı saniyelerde?"],
                ["12. saniye ile 18. saniye arasında ne oluyor?"],
            ],
            inputs=[agent_msg],
            label="örnek input",
        )
        agent_send.click(
            run_agent,
            [agent_msg, agent_chat, agent_lc, agent_video],
            [agent_chat, agent_msg, agent_lc, agent_trace],
        )
        agent_msg.submit(
            run_agent,
            [agent_msg, agent_chat, agent_lc, agent_video],
            [agent_chat, agent_msg, agent_lc, agent_trace],
        )
        agent_clear.click(
            clear_agent,
            outputs=[agent_chat, agent_msg, agent_lc, agent_trace],
        )

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
    )
