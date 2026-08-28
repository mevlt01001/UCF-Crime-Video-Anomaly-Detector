"""Gradio lab sekmelerinin API karşılıkları; lab.py'ye bağımlı değildir."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.env import env_first, env_int

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "_stuff" / "lab_runs"

class ModelContext:
    """Owned by one session; callers serialize access with the session lock."""
    def __init__(self):
        self.llm = None
        self.vlm = None

    def get_llm(self):
        if self.llm is None:
            from utils.llm import LLM_Manager
            self.llm = LLM_Manager()
        return self.llm

    def get_vlm(self):
        if self.vlm is None:
            from utils.vlm import VLM_Manager
            self.vlm = VLM_Manager()
        return self.vlm


def _timed(fn) -> str:
    started = time.perf_counter()
    out = fn()
    ms = (time.perf_counter() - started) * 1000
    return f"{out}\n\n---\nsüre: {ms:.0f} ms"


def run_llm(prompt: str, keep_history: bool, context: ModelContext) -> str:
    if not (prompt or "").strip():
        raise ValueError("Prompt boş.")
    manager = context.get_llm()
    if not keep_history:
        manager.clear_history()
    return _timed(lambda: manager.run(prompt.strip(), raise_on_error=True))


def _frames_from_upload(path: Optional[str], max_frames: int):
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


def run_vlm(prompt: str, media_path: Optional[str], max_frames: int, keep_history: bool, context: ModelContext) -> str:
    if not (prompt or "").strip() and not media_path:
        raise ValueError("Prompt veya medya gerekli.")
    manager = context.get_vlm()
    if not keep_history:
        manager.reset_context()
    frames, source_duration = _frames_from_upload(media_path, max_frames)

    def _call():
        return manager.run((prompt or "").strip(), frames=frames, source_duration=source_duration, raise_on_error=True)

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


# Agent tool (`run_abnormal_event_segmenter`) ile aynı sabit eşik.
ANALYZER_THRESHOLD = 0.3


def run_analyzer(video_path: str) -> tuple[str, Optional[str]]:
    if not video_path:
        raise ValueError("Video yok.")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    clip_size = env_int("AS_CLIP_SIZE", 16)
    overlap = env_int("AS_OVERLAP", 8)
    fps = env_int("AS_FPS", 30)
    width = env_int("AS_WIDTH", 224)
    height = env_int("AS_HEIGHT", 224)
    batch = env_int("AS_BATCH", 4)
    backbone = env_first("AS_MODEL_NAME", "ANALYZER_BACKBONE") or "s3d"
    ckpt_path = _resolve_analyzer_checkpoint()
    if overlap >= clip_size:
        raise ValueError("overlap, clip_size'dan küçük olmalı (.env).")

    def _call():
        if not ckpt_path or not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Checkpoint yok: {ckpt_path}")

        from utils.video_analyzer_model import Video_Analyzer, pick_device

        device = pick_device()
        model = Video_Analyzer(
            backbone,
            clip_size=clip_size,
            overlap=overlap,
            fc_layer_checkpoint=ckpt_path,
        )
        model.to(device)
        segments = model.analyze(
            video_path=video_path,
            width=width,
            height=height,
            fps=fps,
            batch_size=batch,
            threshold=ANALYZER_THRESHOLD,
            save_graph=True,
            save_clips=False,
            save_dir=str(RUNS_DIR),
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
            "threshold": ANALYZER_THRESHOLD,
            "segments": segments,
        }

    started = time.perf_counter()
    payload = _call()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    ms = (time.perf_counter() - started) * 1000
    graph = None
    candidate = RUNS_DIR / Path(video_path).stem / "segmentation_graph.png"
    if candidate.exists():
        graph = str(candidate.resolve())
    return f"{text}\n\n---\nsüre: {ms:.0f} ms", graph
