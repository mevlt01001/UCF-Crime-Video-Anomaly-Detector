# Parallel Validation Checklist (Gradio + New UI)

Use this checklist while both interfaces are enabled.

## Environment

- [ ] `.venv` active (`python --version` is 3.10+)
- [ ] `python run_web.py` is running
- [ ] New UI opens at `http://127.0.0.1:8000/`
- [ ] Gradio fallback opens at `http://127.0.0.1:8000/gradio`

## Core behavior parity

Run each prompt once in new UI and once in Gradio Agent:

- [ ] Greeting without tools (no unnecessary analysis)
- [ ] Metadata request (duration, fps, resolution)
- [ ] Full anomaly + explanation flow
- [ ] Specific valid time range explanation
- [ ] Out-of-range time request handled safely
- [ ] Clip save request returns real file path
- [ ] Unsupported capability request (audio transcript etc.) returns clear limitation

Reference scenarios:

- `AGENT_RESPONSE_FLOW_TEST_SCENARIOS.md`
- `AGENT_TOOL_REFACTOR_TEST_SCENARIOS.md`
- `SAVE_VIDEO_SEGMENT_TEST_SCENARIOS.md`
- `OBJECT_TRACKING_TEST_SCENARIOS.md`

## New UI specific checks

- [ ] Live process cards stream per node (`planner`, `executor`, `tools`, `reviewer`)
- [ ] Cancel button stops queued progress for long jobs
- [ ] Report mode shows validated JSON and download link
- [ ] Session keeps active video between requests
- [ ] Tool warnings/errors appear in stream details

## Exit gate before removing Gradio

- [ ] No semantic regressions in critical user asks
- [ ] Stable SSE stream across repeated runs
- [ ] Team confirms fallback is no longer needed
