"""Categorized, local, repeat-safe clip archive. No semantic classifier."""
import hashlib
import json
import re
import shutil
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

from utils.video_export import export_video

ArchiveCategory = Literal["hirsizlik", "soygun", "kavga_saldiri", "trafik_kazasi",
                          "is_kazasi", "diger", "belirsiz"]
CATEGORIES = ("hirsizlik", "soygun", "kavga_saldiri", "trafik_kazasi", "is_kazasi", "diger", "belirsiz")
ROOT = Path(__file__).resolve().parents[1] / "_stuff/lab_runs/actions/archive"
_INCIDENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
_LOCK = Lock()


class ArchiveError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.data = code, {}


def _signature(path):
    stat = path.stat()
    return [str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _sanitize_incident_id(incident_id: str) -> str:
    if not isinstance(incident_id, str):
        raise ArchiveError("INVALID_INCIDENT_ID", "olay_id geçersiz.")
    cleaned = incident_id.strip()
    if not cleaned or not _INCIDENT_ID.fullmatch(cleaned):
        raise ArchiveError("INVALID_INCIDENT_ID", "olay_id 1–80 karakter; harf/rakam ile başlamalı; yalnız A-Za-z0-9_-.")
    return cleaned


def _category_root(category: str) -> Path:
    parent = ROOT / category
    parent.mkdir(parents=True, exist_ok=True)
    if parent.resolve().parent != ROOT.resolve():
        raise ArchiveError("INVALID_ARCHIVE_PATH", "Kategori klasörü arşiv kökü dışında.")
    return parent


def _clip_parent(category: str, incident_id: Optional[str], source_key: str) -> Path:
    parent = _category_root(category)
    if incident_id is None:
        return parent
    # Incident names are local to a source video, not globally unique.
    # Legacy category/incident folders remain untouched.
    parent = parent / source_key
    parent.mkdir(parents=True, exist_ok=True)
    if parent.resolve().parent != _category_root(category).resolve():
        raise ArchiveError("INVALID_ARCHIVE_PATH", "Video klasörü kategori dışında.")
    folder = parent / incident_id
    folder.mkdir(parents=True, exist_ok=True)
    if folder.resolve().parent != parent.resolve():
        raise ArchiveError("INVALID_ARCHIVE_PATH", "Olay klasörü kategori dışında.")
    return folder


def _update_incident_manifest(incident_dir: Path, incident_id: str, category: str,
                              video_path: str, clip_data: dict) -> str:
    manifest = incident_dir / "incident.json"
    entry = {
        "start_sec": clip_data["saved_range"]["start_sec"],
        "end_sec": clip_data["saved_range"]["end_sec"],
        "output_path": clip_data["output_path"],
        "metadata_path": clip_data["metadata_path"],
        "clip_sha256": clip_data["clip_sha256"],
        "explanation": clip_data["explanation"],
        "cache_hit": clip_data.get("cache_hit", False),
    }
    if manifest.is_file():
        try:
            stored = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            stored = {}
    else:
        stored = {}
    clips = [item for item in stored.get("clips") or []
             if item.get("output_path") != entry["output_path"]]
    clips.append(entry)
    clips.sort(key=lambda item: (item.get("start_sec", 0), item.get("end_sec", 0)))
    payload = {
        "incident_id": incident_id,
        "category": category,
        "video_path": video_path,
        "clips": clips,
    }
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return str(manifest)


def archive_clip(video_path, start, end, category, explanation, incident_id: Optional[str] = None):
    from filelock import FileLock

    if category not in CATEGORIES:
        raise ArchiveError("INVALID_CATEGORY", "Desteklenmeyen arşiv kategorisi.")
    if not isinstance(explanation, str) or not 0 < len(explanation.strip()) <= 2000:
        raise ArchiveError("INVALID_ARCHIVE_DESCRIPTION", "1–2000 karakterlik kategori gerekçesi gerekli.")
    cleaned_incident = _sanitize_incident_id(incident_id) if incident_id is not None else None
    source = Path(video_path).resolve()
    signature = _signature(source)
    source_key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
    identity = {"source_signature": signature, "start_sec": start, "end_sec": end,
                "category": category, "revision": 1}
    if cleaned_incident is not None:
        identity["incident_id"] = cleaned_incident
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    ROOT.mkdir(parents=True, exist_ok=True)
    with _LOCK, FileLock(str(ROOT / ".archive.lock"), timeout=310):
        if _signature(source) != signature:
            raise ArchiveError("SOURCE_CHANGED", "Kaynak video değişti; yeniden deneyin.")
        parent = _clip_parent(category, cleaned_incident, source_key)
        folder = parent / key
        if folder.resolve().parent != parent.resolve():
            raise ArchiveError("INVALID_ARCHIVE_PATH", "Klip klasörü hedef dışında.")
        for other in CATEGORIES:
            if other == category:
                continue
            other_key = hashlib.sha256(json.dumps({**identity, "category": other}, sort_keys=True,
                                                  allow_nan=False).encode()).hexdigest()
            if (ROOT / other / other_key).exists():
                raise ArchiveError("ARCHIVE_CATEGORY_CONFLICT", f"Bu kesit zaten {other} kategorisinde; ikinci kategoriye kopyalanmadı.")
            if cleaned_incident is not None and (ROOT / other / cleaned_incident / other_key).exists():
                raise ArchiveError("ARCHIVE_CATEGORY_CONFLICT", f"Bu kesit zaten {other} kategorisinde; ikinci kategoriye kopyalanmadı.")
            if cleaned_incident is not None and (ROOT / other / source_key / cleaned_incident / other_key).exists():
                raise ArchiveError("ARCHIVE_CATEGORY_CONFLICT", f"Bu kesit zaten {other} kategorisinde; ikinci kategoriye kopyalanmadı.")
        target, manifest = folder / "clip.mp4", folder / "metadata.json"
        if folder.exists():
            try:
                if manifest.stat().st_size > 16384 or manifest.resolve().parent != folder.resolve():
                    raise ValueError("Geçersiz arşiv metadata dosyası.")
                stored = json.loads(manifest.read_text(encoding="utf-8"))
                valid = (stored["identity"] == identity and target.is_file()
                         and target.resolve().parent == folder.resolve()
                         and stored["clip_sha256"] == _digest(target)
                         and stored["output_path"] == str(target)
                         and stored["metadata_path"] == str(manifest))
            except (OSError, ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                raise ArchiveError("ARCHIVE_CONFLICT", "Mevcut arşiv kaydı eksik/değişmiş; üzerine yazılmadı. Kaydı inceleyin.")
            data = {**stored, "cache_hit": True}
        else:
            folder.mkdir()
            try:
                export_video(str(source), target, start, end, exact=True)
                if _signature(source) != signature:
                    raise ArchiveError("SOURCE_CHANGED", "Arşivleme sırasında kaynak video değişti.")
                data = {"video_path": video_path, "category": category,
                        "explanation": explanation.strip(), "saved_range": {"start_sec": start, "end_sec": end},
                        "output_path": str(target), "metadata_path": str(manifest),
                        "output_size_bytes": target.stat().st_size, "clip_sha256": _digest(target),
                        "identity": identity, "cache_hit": False}
                if cleaned_incident is not None:
                    data["incident_id"] = cleaned_incident
                manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            except Exception:
                shutil.rmtree(folder)
                raise
        if cleaned_incident is not None:
            data["incident_path"] = _update_incident_manifest(parent, cleaned_incident, category, video_path, data)
            data.setdefault("incident_id", cleaned_incident)
        return data
