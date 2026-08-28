"""Job-scope Evren token usage tracking for live UI metrics."""
from __future__ import annotations

import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

_current_tracker: ContextVar[Optional["JobUsageTracker"]] = ContextVar("job_usage_tracker", default=None)


@dataclass(frozen=True)
class UsageSnapshot:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tokens_per_sec: float | None
    complete: bool
    api_duration_sec: float

    def as_payload(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens if self.complete else None,
            "tokens_per_sec": self.tokens_per_sec,
            "complete": self.complete,
            "api_duration_sec": round(self.api_duration_sec, 4),
        }

    def delta_since(self, previous: "UsageSnapshot") -> dict[str, int]:
        return {
            "input_tokens": max(0, self.input_tokens - previous.input_tokens),
            "output_tokens": max(0, self.output_tokens - previous.output_tokens),
            "total_tokens": max(0, self.total_tokens - previous.total_tokens),
        }


def _usage_fields(usage: Any) -> tuple[Any, Any, Any]:
    """Evren/OpenAI usage nesnesinden token alanlarını çıkarır."""
    if usage is None:
        return None, None, None
    if isinstance(usage, dict):
        data = usage
    elif hasattr(usage, "model_dump"):
        data = usage.model_dump()
    else:
        data = usage
    if isinstance(data, dict):
        return (
            data.get("input_tokens") or data.get("prompt_tokens"),
            data.get("output_tokens") or data.get("completion_tokens"),
            data.get("total_tokens"),
        )
    return (
        getattr(data, "input_tokens", None) or getattr(data, "prompt_tokens", None),
        getattr(data, "output_tokens", None) or getattr(data, "completion_tokens", None),
        getattr(data, "total_tokens", None),
    )


def _extract_llm_usage(response: LLMResult) -> Any:
    for generation_list in response.generations:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            if isinstance(message, AIMessage) and message.usage_metadata:
                return message.usage_metadata
    if response.llm_output:
        return response.llm_output.get("token_usage") or response.llm_output.get("usage")
    return None


def _normalize_usage(
    *,
    input_tokens: Any = None,
    output_tokens: Any = None,
    total_tokens: Any = None,
) -> tuple[int, int, int, bool]:
    try:
        inp = int(input_tokens or 0)
        out = int(output_tokens or 0)
        if total_tokens is None:
            total = inp + out
        else:
            total = int(total_tokens)
    except (TypeError, ValueError):
        return 0, 0, 0, False
    if inp < 0 or out < 0 or total < 0:
        return 0, 0, 0, False
    if inp == 0 and out == 0 and total == 0:
        return 0, 0, 0, False
    return inp, out, total, True


class JobUsageTracker:
    def __init__(self, on_update: Callable[[UsageSnapshot], None] | None = None) -> None:
        self._lock = Lock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._api_duration_sec = 0.0
        self._complete = True
        self._on_update = on_update
        self.callback_handler = _JobUsageCallbackHandler(self)

    def record_usage(
        self,
        *,
        input_tokens: Any = None,
        output_tokens: Any = None,
        total_tokens: Any = None,
        duration_sec: float,
        source_complete: bool = True,
    ) -> UsageSnapshot:
        inp, out, total, parsed = _normalize_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        with self._lock:
            if parsed:
                self._input_tokens += inp
                self._output_tokens += out
                self._total_tokens += total
            if not source_complete or not parsed:
                self._complete = False
            if duration_sec > 0:
                self._api_duration_sec += float(duration_sec)
            snapshot = self._snapshot_locked()
        if self._on_update is not None:
            self._on_update(snapshot)
        return snapshot

    def record_openai_usage(self, usage: Any, duration_sec: float) -> UsageSnapshot:
        if usage is None:
            return self.record_usage(duration_sec=duration_sec, source_complete=False)
        inp, out, total = _usage_fields(usage)
        return self.record_usage(
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            duration_sec=duration_sec,
            source_complete=True,
        )

    def snapshot(self) -> UsageSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> UsageSnapshot:
        tokens_per_sec = None
        if self._complete and self._api_duration_sec > 0 and self._output_tokens > 0:
            tokens_per_sec = self._output_tokens / self._api_duration_sec
        return UsageSnapshot(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            total_tokens=self._total_tokens,
            tokens_per_sec=tokens_per_sec,
            complete=self._complete,
            api_duration_sec=self._api_duration_sec,
        )


class _JobUsageCallbackHandler(BaseCallbackHandler):
    def __init__(self, tracker: JobUsageTracker) -> None:
        super().__init__()
        self._tracker = tracker
        self._starts: dict[str, float] = {}

    def _mark_start(self, run_id: Any) -> None:
        self._starts[str(run_id)] = time.perf_counter()

    def _mark_end(self, response: LLMResult, run_id: Any) -> None:
        started = self._starts.pop(str(run_id), None)
        duration = time.perf_counter() - started if started is not None else 0.0
        usage = _extract_llm_usage(response)
        if usage is None:
            self._tracker.record_usage(duration_sec=duration, source_complete=False)
            return
        inp, out, total = _usage_fields(usage)
        self._tracker.record_usage(
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            duration_sec=duration,
            source_complete=True,
        )

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: Any, **kwargs: Any) -> None:
        self._mark_start(run_id)

    def on_llm_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        self._mark_end(response, run_id)

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list], *, run_id: Any, **kwargs: Any) -> None:
        self._mark_start(run_id)

    def on_chat_model_end(self, response: LLMResult, *, run_id: Any, **kwargs: Any) -> None:
        self._mark_end(response, run_id)


def get_current_tracker() -> JobUsageTracker | None:
    return _current_tracker.get()


def set_current_tracker(tracker: JobUsageTracker | None) -> Token:
    return _current_tracker.set(tracker)


def reset_current_tracker(token: Token) -> None:
    _current_tracker.reset(token)


def invoke_config() -> dict[str, Any]:
    tracker = get_current_tracker()
    if tracker is None:
        return {}
    return {"callbacks": [tracker.callback_handler]}
