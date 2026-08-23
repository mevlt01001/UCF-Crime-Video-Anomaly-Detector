from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Camera:
    id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    location: str = ""
    view: str = ""
    description: str = ""
    overlaps_with: list[str] = field(default_factory=list)
    video_path: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> Camera:
        return cls(
            id=str(raw["id"]),
            name=str(raw.get("name", "")),
            aliases=list(raw.get("aliases") or []),
            location=str(raw.get("location", "")),
            view=str(raw.get("view", "")),
            description=str(raw.get("description", "")),
            overlaps_with=list(raw.get("overlaps_with") or []),
            video_path=str(raw.get("video_path", "")),
        )

    def to_result(self, reason: str = "") -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "reason": reason,
            "video_path": self.video_path,
        }

    def to_dict(self) -> dict:
        return asdict(self)
