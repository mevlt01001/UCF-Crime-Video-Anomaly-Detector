"""Categorized, local, repeat-safe clip archive. No semantic classifier."""
import hashlib
import json
import shutil
from pathlib import Path
from threading import Lock
from typing import Literal

from utils.video_export import export_video

ArchiveCategory = Literal["hirsizlik", "soygun", "kavga_saldiri", "trafik_kazasi",
                          "is_kazasi", "diger", "belirsiz"]
CATEGORIES = ("hirsizlik", "soygun", "kavga_saldiri", "trafik_kazasi", "is_kazasi", "diger", "belirsiz")
ROOT = Path(__file__).resolve().parents[1] / "_stuff/lab_runs/actions/archive"
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


def archive_clip(video_path, start, end, category, explanation):
    from filelock import FileLock

    if category not in CATEGORIES:
        raise ArchiveError("INVALID_CATEGORY", "Desteklenmeyen arşiv kategorisi.")
    if not isinstance(explanation, str) or not 0 < len(explanation.strip()) <= 2000:
        raise ArchiveError("INVALID_ARCHIVE_DESCRIPTION", "1–2000 karakterlik kategori gerekçesi gerekli.")
    source = Path(video_path).resolve()
    signature = _signature(source)
    identity = {"source_signature": signature, "start_sec": start, "end_sec": end,
                "category": category, "revision": 1}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    ROOT.mkdir(parents=True, exist_ok=True)
    with _LOCK, FileLock(str(ROOT / ".archive.lock"), timeout=310):
        if _signature(source) != signature:
            raise ArchiveError("SOURCE_CHANGED", "Kaynak video değişti; yeniden deneyin.")
        parent = ROOT / category
        parent.mkdir(exist_ok=True)
        if parent.resolve().parent != ROOT.resolve():
            raise ArchiveError("INVALID_ARCHIVE_PATH", "Kategori klasörü arşiv kökü dışında.")
        folder = parent / key
        if folder.resolve().parent != parent.resolve():
            raise ArchiveError("INVALID_ARCHIVE_PATH", "Klip klasörü kategori dışında.")
        for other in CATEGORIES:
            if other == category:
                continue
            other_key = hashlib.sha256(json.dumps({**identity, "category": other}, sort_keys=True,
                                                  allow_nan=False).encode()).hexdigest()
            if (ROOT / other / other_key).exists():
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
            return {**stored, "cache_hit": True}
        folder.mkdir()
        try:
            export_video(str(source), target, start, end, exact=True)
            if _signature(source) != signature:
                raise ArchiveError("SOURCE_CHANGED", "Arşivleme sırasında kaynak video değişti.")
            data = {"video_path": video_path, "category": category,
                    "explanation": explanation.strip(), "saved_range": {"start_sec": start, "end_sec": end},
                    "output_path": str(target), "metadata_path": str(manifest),
                    "output_size_bytes": target.stat().st_size, "clip_sha256": _digest(target),
                    "identity": identity}
            manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            return {**data, "cache_hit": False}
        except Exception:
            # Only this call's newly created directory; existing archives are never removed.
            shutil.rmtree(folder)
            raise
