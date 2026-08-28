# New UI API Contract

This document maps existing `lab.py` behavior to HTTP endpoints used by the new UI.
Core model and tool logic remains in `utils/*`.

## Transport

- Base path: `/api`
- Event stream: Server-Sent Events (SSE), UTF-8 JSON payloads
- Session model: client-owned `session_id` (UUID string)

## Session

### `POST /api/sessions`

Create or restore a session identity.

Request body:

```json
{
  "session_id": "optional-existing-id"
}
```

Response:

```json
{
  "session_id": "uuid",
  "active_video_path": "/abs/path/or/null",
  "active_video_url": "/media/uploads/.../video.mp4 or null"
}
```

## Video

### `POST /api/videos`

Upload a target video and bind it to a session.

Form fields:
- `session_id` (required)
- `file` (required)

Response:

```json
{
  "session_id": "uuid",
  "video_path": "/absolute/path/to/uploaded/file.mp4",
  "video_url": "/media/uploads/<session_id>/<filename>.mp4"
}
```

## Chat Job (Agent mode)

### `POST /api/chat`

Starts one async graph run equivalent to `run_agent(...)`.

Request:

```json
{
  "session_id": "uuid",
  "message": "user message"
}
```

Response:

```json
{
  "job_id": "uuid",
  "mode": "chat"
}
```

## Report Job (Report mode)

### `POST /api/report`

Starts one async graph run equivalent to `run_video_report(...)`.

Request:

```json
{
  "session_id": "uuid"
}
```

Response:

```json
{
  "job_id": "uuid",
  "mode": "report"
}
```

## Live Stream

### `GET /api/stream/{job_id}?session_id=<id>`

SSE event channel for a running job.

Event data schema (all messages include `type` and `timestamp_ms`):

- `job_started`: mode + session
- `node_update`: one LangGraph node update
  - `node`: `planner|executor|tools|tool_limit|reviewer`
  - `summary`: short human-readable line
  - `details`: node payload for UI cards
- `chat_final`: final approved assistant text + full chat history
- `report_final`: validated report JSON + downloadable file URL
- `job_cancelled`: cancellation acknowledged
- `job_error`: unhandled exception info
- `done`: terminal marker
- `heartbeat`: keep-alive marker

## Cancel

### `POST /api/jobs/{job_id}/cancel`

Signals cancellation for future node boundaries.

Request:

```json
{
  "session_id": "uuid"
}
```

Response:

```json
{
  "ok": true
}
```

## Compatibility rules

- Tool response envelope remains unchanged: `{ok,data,warnings,error}`.
- Same graph entrypoint is used: `utils.agents.video_agent_app`.
- Gradio remains available at `/gradio` until cutover is complete.

## Session isolation and job lifecycle

- The React UI creates a fresh UUID for each page load/tab and each “Yeni sohbet”.
  It does not restore hidden model context from localStorage.
- One video per session. A second upload returns 409; start a new conversation to
  choose another video. Upload does not erase prior text-only chat.
- LLM/VLM managers belong to a session; clearing an idle session resets both.
- Only one operation may run per session (including uploads, clear, LLM/VLM,
  Analyzer, chat and reports). Conflicting operations return HTTP 409.
- Text-only chat is allowed without a video. Reports/Analyzer require a video.
- `POST /api/jobs/analyzer` takes `{ "session_id": "uuid" }` and returns
  `{ "job_id": "uuid", "mode": "analyzer" }`. Subscribe via the existing SSE route.
  Success emits `analyzer_final` with `output` and `graph_url`, then `done`.
  Failure emits `job_error`, then `done`, never a success event.
- The synchronous `/api/analyzer` endpoint remains available. Model exceptions
  return HTTP 500, not `ok: true`; this also applies to LLM/VLM calls.
- Cancellation is cooperative: an in-flight model/tool call is not interrupted.
  Its completion is awaited, then its result is discarded and `job_cancelled` /
  `done` are emitted. The session stays busy until that point. New conversation
  and upload controls remain disabled while work or cancellation is pending.
