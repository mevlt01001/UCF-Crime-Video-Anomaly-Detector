from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage

from ui_backend.contracts import (
    CancelRequest,
    CancelResponse,
    ChatRequest,
    JobStartResponse,
    LLMRequest,
    VLMRequest,
    ReportRequest,
    SessionRequest,
    SessionResponse,
)
from ui_backend.lab_runs import ModelContext, run_analyzer, run_llm, run_vlm
from utils.agents import video_agent_app
from utils.reporting import REPORT_TASK
from utils.usage_tracking import JobUsageTracker, reset_current_tracker, set_current_tracker

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "_stuff" / "lab_runs"
UPLOADS_DIR = RUNS_DIR / "uploads"
REPORTS_DIR = RUNS_DIR / "reports"
FRONTEND_DIST_DIR = ROOT / "frontend" / "dist"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SessionState:
    session_id: str
    lc_messages: list[Any] = field(default_factory=list)
    chat_history: list[dict[str, str]] = field(default_factory=list)
    active_video_path: Optional[str] = None
    operation_lock: Any = field(default_factory=Lock)
    models: ModelContext = field(default_factory=ModelContext)


@dataclass
class JobState:
    job_id: str
    session_id: str
    mode: str
    queue: Queue[dict[str, Any]] = field(default_factory=Queue)
    cancel_event: Event = field(default_factory=Event)
    done_event: Event = field(default_factory=Event)
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


_sessions: dict[str, SessionState] = {}
_jobs: dict[str, JobState] = {}
_sessions_lock = Lock()
_jobs_lock = Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sse_payload(payload: dict[str, Any]) -> bytes:
    return f"event: message\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _tool_calls_line(msg: Any) -> str:
    calls = getattr(msg, "tool_calls", None) or []
    if not calls:
        return ""
    parts = []
    for call in calls:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "?")
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        parts.append(f"{name}({args})")
    return " | ".join(parts)


def _content_preview(msg: Any, limit: int = 600) -> str:
    content = getattr(msg, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    content = content.strip()
    if len(content) > limit:
        return content[:limit] + "…"
    return content


def _path_to_media_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    try:
        rel = p.resolve().relative_to(RUNS_DIR.resolve())
    except Exception:
        return None
    return f"/media/{rel.as_posix()}"


def _safe_name(filename: str) -> str:
    return "".join(ch for ch in Path(filename).name if ch.isalnum() or ch in {"-", "_", "."}) or "upload.mp4"


def _get_or_create_session(session_id: Optional[str]) -> SessionState:
    resolved = (session_id or str(uuid.uuid4())).strip()
    with _sessions_lock:
        state = _sessions.get(resolved)
        if state is None:
            state = SessionState(session_id=resolved)
            _sessions[resolved] = state
        return state


def _get_session_or_404(session_id: str) -> SessionState:
    with _sessions_lock:
        state = _sessions.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        return state


@contextmanager
def _operation(session: SessionState):
    if not session.operation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Bu oturumda işlem sürüyor; bitmesini bekleyin.")
    try:
        yield
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        session.operation_lock.release()


def _cancelled(job: JobState) -> bool:
    if job.cancel_event.is_set():
        _emit(job, {"type": "job_cancelled", "job_id": job.job_id})
        return True
    return False


def _emit(job: JobState, payload: dict[str, Any]) -> None:
    payload.setdefault("timestamp_ms", _now_ms())
    job.queue.put(payload)


def _emit_usage(job: JobState, tracker: JobUsageTracker) -> None:
    payload = tracker.snapshot().as_payload()
    payload["type"] = "usage_update"
    payload["job_id"] = job.job_id
    _emit(job, payload)


def _normalize_node_update(node_name: str, update: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    summary = f"[{node_name}]"
    details: dict[str, Any] = {}

    if node_name == "planner":
        raw_plan = update.get("plan") or ""
        details["plan_raw"] = raw_plan
        try:
            details["plan"] = json.loads(raw_plan) if raw_plan else None
        except (ValueError, TypeError):
            details["plan"] = None
        if raw_plan:
            summary += " plan produced"
    elif node_name == "executor":
        msgs = update.get("messages") or []
        last = msgs[-1] if msgs else None
        calls = _tool_calls_line(last) if last else ""
        if calls:
            details["tool_calls"] = calls
            summary += f" tool call: {calls}"
        else:
            preview = _content_preview(last) if last else ""
            details["preview"] = preview
            if preview:
                summary += f" {preview[:180]}"
    elif node_name == "tools":
        previews = []
        for tool_msg in update.get("messages") or []:
            preview = _content_preview(tool_msg, 800)
            parsed = None
            if isinstance(getattr(tool_msg, "content", None), str):
                try:
                    parsed = json.loads(tool_msg.content)
                except (ValueError, TypeError):
                    parsed = None
            previews.append(
                {
                    "name": getattr(tool_msg, "name", "") or "",
                    "tool_call_id": getattr(tool_msg, "tool_call_id", "") or "",
                    "preview": preview,
                    "parsed": parsed,
                }
            )
        details["tool_results"] = previews
        summary += f" tool results: {len(previews)}"
    elif node_name == "reviewer":
        feedback = update.get("feedback") or ""
        final_answer = update.get("final_answer") or ""
        details["feedback"] = feedback
        details["final_answer"] = final_answer
        if final_answer:
            details["answer_approved"] = update.get("answer_approved") is True
            summary += " final answer approved" if details["answer_approved"] else " finished without approval"
        elif feedback:
            summary += f" feedback: {feedback[:160]}"
    else:
        details["raw"] = update
        summary += " updated"

    return summary, details


def _build_initial_state(mode: str, session: SessionState, message: str = "") -> dict[str, Any]:
    video_path = session.active_video_path or ""
    if mode == "report":
        conversation = [HumanMessage(content=REPORT_TASK)]
        user_query = REPORT_TASK
    else:
        content = message if not video_path else f"{message}\n\n[Hedef video: {video_path}]"
        conversation = list(session.lc_messages) + [HumanMessage(content=content)]
        user_query = message
    return {
        "output_mode": mode,
        "report": None,
        "user_query": user_query,
        "video_path": video_path,
        "video_paths": [video_path] if video_path else [],
        "image_paths": [],
        "plan": "",
        "conversation_messages": conversation,
        "messages": [],
        "feedback": "",
        "review_route": "",
        "final_answer": "",
        "tool_rounds": 0,
        "review_loops": 0,
    }


def _run_job(job: JobState, message: str = "") -> None:
    _emit(job, {"type": "job_started", "job_id": job.job_id, "session_id": job.session_id, "mode": job.mode})
    tracker = JobUsageTracker(on_update=lambda _snap: _emit_usage(job, tracker))
    tracker_token = set_current_tracker(tracker)
    _emit_usage(job, tracker)
    try:
        session = _get_session_or_404(job.session_id)
        if not session.active_video_path:
            raise ValueError("Önce video yükleyin.")
        if job.mode == "chat" and not message.strip():
            raise ValueError("Mesaj boş.")

        if _cancelled(job):
            return
        if job.mode == "analyzer":
            text, graph_path = run_analyzer(session.active_video_path)
            if not _cancelled(job):
                _emit(job, {"type": "analyzer_final", "job_id": job.job_id,
                            "output": text, "graph_url": _path_to_media_url(graph_path)})
            return

        state = _build_initial_state(job.mode, session, message)
        final_answer = ""
        report_data = None

        stream_config = {
            "recursion_limit": 40,
            "callbacks": [tracker.callback_handler],
        }
        usage_before_node = tracker.snapshot()
        for event in video_agent_app.stream(state, stream_config):
            if _cancelled(job):
                return
            for node_name, update in event.items():
                summary, details = _normalize_node_update(node_name, update)
                usage_after_node = tracker.snapshot()
                _emit(
                    job,
                    {
                        "type": "node_update",
                        "job_id": job.job_id,
                        "node": node_name,
                        "summary": summary,
                        "details": details,
                        "node_usage": usage_after_node.delta_since(usage_before_node),
                    },
                )
                usage_before_node = usage_after_node
                _emit_usage(job, tracker)
                if node_name == "reviewer" and update.get("final_answer"):
                    final_answer = update["final_answer"]
                if update.get("report") is not None:
                    report_data = update["report"]

        if _cancelled(job):
            return
        if job.mode == "chat":
            if not final_answer:
                final_answer = "(nihai cevap yok — trace'e bak)"
            with _sessions_lock:
                session.chat_history.append({"role": "user", "content": message})
                session.chat_history.append({"role": "assistant", "content": final_answer})
                convo = state["conversation_messages"] + [AIMessage(content=final_answer)]
                session.lc_messages = convo
                history = list(session.chat_history)
            _emit(
                job,
                {
                    "type": "chat_final",
                    "job_id": job.job_id,
                    "assistant_message": final_answer,
                    "chat_history": history,
                },
            )
        else:
            if report_data is None:
                _emit(job, {"type": "job_error", "job_id": job.job_id, "message": "Doğrulanmış rapor üretilemedi."})
            else:
                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    suffix=".json",
                    prefix="olay_raporu_",
                    dir=REPORTS_DIR,
                    delete=False,
                ) as handle:
                    json.dump(report_data, handle, ensure_ascii=False, indent=2, allow_nan=False)
                    report_path = handle.name
                _emit(
                    job,
                    {
                        "type": "report_final",
                        "job_id": job.job_id,
                        "report": report_data,
                        "download_path": report_path,
                        "download_url": _path_to_media_url(report_path),
                    },
                )
    except Exception as exc:
        _emit(job, {"type": "job_error", "job_id": job.job_id, "message": f"{type(exc).__name__}: {exc}"})
    finally:
        _emit_usage(job, tracker)
        reset_current_tracker(tracker_token)
        _get_session_or_404(job.session_id).operation_lock.release()
        job.done_event.set()
        _emit(job, {"type": "done", "job_id": job.job_id})


def _start_job(mode: str, session_id: str, message: str = "") -> JobState:
    if mode not in {"chat", "report", "analyzer"}:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")
    session = _get_or_create_session(session_id)
    if not session.operation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Bu oturumda zaten çalışan bir iş var.")
    if not session.active_video_path:
        session.operation_lock.release()
        raise HTTPException(status_code=400, detail="Önce video yükleyin.")
    job = JobState(job_id=str(uuid.uuid4()), session_id=session_id, mode=mode)
    with _jobs_lock:
        _jobs[job.job_id] = job
    thread = Thread(target=_run_job, args=(job, message), daemon=True)
    try:
        thread.start()
    except Exception:
        session.operation_lock.release()
        with _jobs_lock:
            _jobs.pop(job.job_id, None)
        raise
    return job


def create_app() -> FastAPI:
    app = FastAPI(title="neokortex API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:8000", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/media", StaticFiles(directory=str(RUNS_DIR)), name="media")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "gradio_fallback": "/gradio", "timestamp_ms": _now_ms()}

    @app.post("/api/sessions", response_model=SessionResponse)
    def create_session(payload: SessionRequest) -> SessionResponse:
        session = _get_or_create_session(payload.session_id)
        return SessionResponse(
            session_id=session.session_id,
            active_video_path=session.active_video_path,
            active_video_url=_path_to_media_url(session.active_video_path),
        )

    @app.post("/api/videos")
    async def upload_video(
        session_id: str = Form(...),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        session = _get_or_create_session(session_id)
        with _operation(session):
            if session.active_video_path:
                raise HTTPException(status_code=409, detail="Video değiştirmek için yeni sohbet başlatın.")
            target_dir = UPLOADS_DIR / session.session_id
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{uuid.uuid4()}_{_safe_name(file.filename or 'upload.mp4')}"
            target_path = target_dir / filename

            size_bytes = 0
            with target_path.open("wb") as handle:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    handle.write(chunk)

            if size_bytes <= 0:
                target_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="Boş dosya yüklenemez.")

            with _sessions_lock:
                session.active_video_path = str(target_path.resolve())

            return {
                "session_id": session.session_id,
                "video_path": session.active_video_path,
                "video_url": _path_to_media_url(session.active_video_path),
                "size_bytes": size_bytes,
            }

    @app.post("/api/chat", response_model=JobStartResponse)
    def start_chat(payload: ChatRequest) -> JobStartResponse:
        job = _start_job("chat", payload.session_id, payload.message.strip())
        return JobStartResponse(job_id=job.job_id, mode="chat")

    @app.post("/api/report", response_model=JobStartResponse)
    def start_report(payload: ReportRequest) -> JobStartResponse:
        job = _start_job("report", payload.session_id)
        return JobStartResponse(job_id=job.job_id, mode="report")

    @app.post("/api/sessions/{session_id}/clear")
    def clear_session(session_id: str) -> dict[str, Any]:
        session = _get_or_create_session(session_id)
        with _operation(session):
            with _sessions_lock:
                session.lc_messages = []
                session.chat_history = []
                session.active_video_path = None
            session.models = ModelContext()
        return {
            "ok": True,
            "session_id": session.session_id,
            "active_video_path": None,
            "active_video_url": None,
        }

    @app.post("/api/llm")
    def run_llm_tab(payload: LLMRequest) -> dict[str, Any]:
        session = _get_or_create_session(payload.session_id)
        with _operation(session):
            output = run_llm(payload.prompt, payload.keep_history, session.models)
        return {"ok": True, "output": output}

    @app.post("/api/vlm")
    def run_vlm_tab(payload: VLMRequest) -> dict[str, Any]:
        session = _get_or_create_session(payload.session_id)
        with _operation(session):
            if not session.active_video_path:
                raise HTTPException(status_code=400, detail="VLM için soldan video yükleyin.")
            output = run_vlm(payload.prompt, session.active_video_path, payload.max_frames, payload.keep_history, session.models)
        return {"ok": True, "output": output}

    @app.post("/api/analyzer")
    def run_analyzer_tab(session_id: str = Form(...)) -> dict[str, Any]:
        session = _get_or_create_session(session_id)
        with _operation(session):
            video_path = session.active_video_path
            if not video_path:
                raise HTTPException(status_code=400, detail="Analyzer için soldan video yükleyin.")
            text, graph_path = run_analyzer(video_path)
            return {
                "ok": True,
                "output": text,
                "graph_path": graph_path,
                "graph_url": _path_to_media_url(graph_path),
            }

    @app.post("/api/jobs/analyzer", response_model=JobStartResponse)
    def start_analyzer(payload: ReportRequest) -> JobStartResponse:
        job = _start_job("analyzer", payload.session_id)
        return JobStartResponse(job_id=job.job_id, mode="analyzer")

    @app.post("/api/jobs/{job_id}/cancel", response_model=CancelResponse)
    def cancel_job(job_id: str, payload: CancelRequest) -> CancelResponse:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None or job.session_id != payload.session_id:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        job.cancel_event.set()
        return CancelResponse(ok=True)

    @app.get("/api/stream/{job_id}")
    def stream_job(job_id: str, session_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None or job.session_id != session_id:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        def event_generator():
            last_heartbeat = _now_ms()
            while True:
                try:
                    payload = job.queue.get(timeout=1.0)
                    yield _sse_payload(payload)
                    if payload.get("type") == "done":
                        break
                except Empty:
                    now = _now_ms()
                    if now - last_heartbeat >= 15000:
                        last_heartbeat = now
                        yield _sse_payload({"type": "heartbeat", "job_id": job.job_id, "timestamp_ms": now})
                    if job.done_event.is_set() and job.queue.empty():
                        yield _sse_payload({"type": "done", "job_id": job.job_id, "timestamp_ms": _now_ms()})
                        break

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    if FRONTEND_DIST_DIR.is_dir():
        app.mount("/app", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="app")

    return app
