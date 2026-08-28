from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SessionRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, min_length=1)


class SessionResponse(BaseModel):
    session_id: str
    active_video_path: Optional[str] = None
    active_video_url: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ReportRequest(BaseModel):
    session_id: str = Field(min_length=1)


class JobStartResponse(BaseModel):
    job_id: str
    mode: Literal["chat", "report", "analyzer"]


class CancelRequest(BaseModel):
    session_id: str = Field(min_length=1)


class CancelResponse(BaseModel):
    ok: bool


class LLMRequest(BaseModel):
    session_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    keep_history: bool = True


class VLMRequest(BaseModel):
    session_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    keep_history: bool = True
    max_frames: int = 128
