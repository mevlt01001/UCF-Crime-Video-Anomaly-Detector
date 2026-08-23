from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolResult:
    status: str
    data: list[dict] = field(default_factory=list)
    ask_user: str | None = None
