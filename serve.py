from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gradio as gr
from fastapi.staticfiles import StaticFiles

from ui_backend import create_app

ROOT = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = ROOT / "frontend" / "dist"

app = create_app()

try:
    import lab

    app = gr.mount_gradio_app(app, lab.demo, path="/gradio")
except Exception as exc:
    # Keep web API/UI alive even if local Gradio environment is incompatible.
    print(f"[warn] Gradio fallback mount skipped: {type(exc).__name__}: {exc}")

if FRONTEND_DIST_DIR.is_dir():
    # New UI is default while Gradio remains available at /gradio.
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="frontend")
