from __future__ import annotations

import json
from pathlib import Path

from .models import Camera

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = ROOT / "config" / "cameras.json"


class CameraRepository:
    """Kamera kataloğu. v1 JSON okur; imza Postgres'e taşınabilir kalır."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_CATALOG_PATH
        self._cameras: list[Camera] = []
        self.reload()

    def reload(self) -> None:
        with self.path.open(encoding="utf-8") as f:
            payload = json.load(f)
        raw_list = payload.get("cameras", payload if isinstance(payload, list) else [])
        self._cameras = [Camera.from_dict(item) for item in raw_list]

    def get_all(self) -> list[Camera]:
        return list(self._cameras)

    def get_by_id(self, camera_id: str) -> Camera | None:
        if not camera_id:
            return None
        needle = camera_id.strip().casefold()
        for camera in self._cameras:
            if camera.id.casefold() == needle:
                return camera
        return None

    def filter_location(self, location: str) -> list[Camera]:
        needle = _fold(location)
        if not needle:
            return []
        return [camera for camera in self._cameras if needle in _fold(camera.location)]


def _fold(text: str) -> str:
    return (text or "").casefold().strip()
