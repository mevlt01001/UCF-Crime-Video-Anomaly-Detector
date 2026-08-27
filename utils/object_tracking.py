"""Local YOLO detections and per-job ByteTrack state; no agent or API client."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from fractions import Fraction
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

from utils.env import env_float, env_get, env_int

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "_stuff/lab_runs/actions/objects"
_MODEL_LOCK = Lock()
_model = None
_model_key = None
_REVISION = 3
_PREVIEW_LIMIT = 100


class TrackingError(ValueError):
    def __init__(self, code: str, message: str, data: dict | None = None):
        super().__init__(message)
        self.code, self.data = code, data or {}


def _signature(path: Path) -> tuple:
    stat = path.stat()
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _settings() -> dict:
    path = Path(env_get("OBJECT_MODEL_PATH", "_stuff/models/yolo11s.pt"))
    if not path.is_absolute():
        path = ROOT / path
    try:
        settings = {
            "model_path": str(path.resolve()),
            "imgsz": env_int("OBJECT_IMAGE_SIZE", 640),
            "confidence": env_float("OBJECT_CONFIDENCE", 0.25),
            "max_detections": env_int("OBJECT_MAX_DETECTIONS", 100),
            "max_frames": env_int("OBJECT_MAX_FRAMES", 18000),
            "timeout_sec": env_float("OBJECT_TIMEOUT_SEC", 900),
            "device": env_get("OBJECT_DEVICE", "auto"),
        }
        if (settings["imgsz"] <= 0 or settings["imgsz"] % 32
                or settings["max_detections"] <= 0 or settings["max_frames"] <= 0
                or not 0.1 <= settings["confidence"] <= 1
                or not math.isfinite(settings["timeout_sec"]) or settings["timeout_sec"] <= 0
                or settings["device"] not in {"auto", "cpu", "mps", "cuda"}):
            raise ValueError("Geçersiz OBJECT_* ayarı.")
    except (ValueError, TypeError) as exc:
        raise TrackingError("INVALID_OBJECT_CONFIG", str(exc)) from exc
    return settings


def _device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise TrackingError("DEVICE_UNAVAILABLE", "CUDA bu ortamda kullanılamıyor.")
        return "cuda:0"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise TrackingError("DEVICE_UNAVAILABLE", "MPS bu ortamda kullanılamıyor.")
    return requested


def _load_model(path: Path, device: str):
    global _model, _model_key
    # No implicit runtime installation, network inference or model download.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "_stuff/ultralytics"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    key = (_signature(path), device)
    if key != _model_key or _model is None:
        # Commit both fields only after a successful load/device transfer.
        # A failed candidate must not invalidate the previously working model.
        candidate = YOLO(str(path), task="detect")
        if candidate.task != "detect":
            raise TrackingError("INVALID_MODEL", "Bir nesne tespit modeli gerekli.")
        candidate.to(device)
        _model, _model_key = candidate, key
    return _model


def _class_ids(classes: list[str] | None, names: dict) -> list[int] | None:
    if classes is None:
        return None
    lookup = {value.casefold(): int(key) for key, value in names.items()}
    invalid = [name for name in classes if name.strip().casefold() not in lookup]
    if not classes or invalid:
        raise TrackingError("UNSUPPORTED_OBJECT_CLASS", "Model sınıf adı desteklenmiyor.",
                            {"unsupported_classes": invalid, "supported_classes": list(names.values())})
    return sorted({lookup[name.strip().casefold()] for name in classes})


def _lower_frame(reader, timestamp: float) -> int:
    """First frame whose source timestamp is >= timestamp (end is exclusive)."""
    left, right = 0, len(reader)
    while left < right:
        middle = (left + right) // 2
        if float(reader.get_frame_timestamp(middle)[0]) < timestamp - 1e-6:
            left = middle + 1
        else:
            right = middle
    return left


class _Intervals:
    """Stream complete intervals to disk; retain only active intervals + preview."""
    def __init__(self, handle):
        self.handle, self.active = handle, {}
        self.class_preview, self.track_preview = [], []
        self.count = 0

    @property
    def preview(self):
        # Many fragmented tracks must not displace the requested class ranges.
        return (self.class_preview + self.track_preview)[:_PREVIEW_LIMIT]

    def update(self, visible: dict, start: float, end: float):
        for key in list(self.active):
            if key not in visible:
                self._close(key)
        for key, info in visible.items():
            if key not in self.active:
                self.active[key] = {**info, "start_sec": start, "end_sec": end}
            else:
                self.active[key]["end_sec"] = end

    def _close(self, key):
        interval = self.active.pop(key)
        if self.count:
            self.handle.write(",\n")
        json.dump(interval, self.handle, ensure_ascii=False, allow_nan=False)
        self.count += 1
        preview = self.class_preview if interval["kind"] == "class" else self.track_preview
        if len(preview) < _PREVIEW_LIMIT:
            preview.append(interval)

    def finish(self):
        for key in list(self.active):
            self._close(key)


def _cached(index: Path) -> dict | None:
    try:
        record = json.loads(index.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or not record.get("files"):
            return None
        if not isinstance(record.get("result", {}).get("data"), dict):
            return None
        for entry in record["files"]:
            path = Path(entry["path"])
            if not path.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
                return None
            if path.stat().st_size != entry["size"] or _digest(path) != entry["sha256"]:
                return None
        return record["result"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _encode_video(raw: Path, destination: Path, source: Path,
                  start: float, duration: float, timeout: float):
    import cv2

    if timeout <= 0:
        raise TrackingError("TRACKING_TIMEOUT", "Nesne analizi süre sınırını aştı.")
    command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(raw),
               "-ss", str(start), "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
               "-t", str(duration), "-c:v", "copy",
               "-c:a", "aac", "-movflags", "+faststart",
               "-map_metadata", "-1", str(destination)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=max(0.1, timeout))
    except subprocess.TimeoutExpired as exc:
        raise TrackingError("TRACKING_TIMEOUT", "Kutulu video kodlaması süre sınırını aştı.") from exc
    except subprocess.CalledProcessError as exc:
        raise TrackingError("FFMPEG_ERROR", (exc.stderr or "Kodlama başarısız.")[-500:]) from exc
    cap = cv2.VideoCapture(str(destination))
    try:
        if not destination.is_file() or destination.stat().st_size == 0 or not cap.read()[0]:
            raise TrackingError("INVALID_OUTPUT", "Kutulu MP4 doğrulanamadı.")
        output_fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(output_fps) or output_fps <= 0:
            raise TrackingError("INVALID_OUTPUT", "Kutulu video FPS bilgisi geçersiz.")
        actual_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / output_fps
        if not math.isfinite(actual_duration) or abs(actual_duration - duration) > max(0.1, 2 / output_fps):
            raise TrackingError("INVALID_OUTPUT", "Kutulu video süresi beklenen süreyle uyuşmuyor.")
    finally:
        cap.release()


def _draw_detection(frame, box, label, identity, score):
    import cv2

    x1, y1, x2, y2 = map(int, box)
    color = tuple(80 + ((identity or 0) * factor) % 176 for factor in (53, 97, 131))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    caption = f"{label} #{identity if identity is not None else '?'} {score:.2f}"
    scale = max(0.35, min(0.65, frame.shape[1] / 1200))
    (width, height), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    left = max(0, min(x1, frame.shape[1] - width - 4))
    top = max(0, y1 - height - baseline - 6)
    cv2.rectangle(frame, (left, top), (left + width + 4, top + height + baseline + 4), (25, 25, 25), -1)
    cv2.putText(frame, caption, (left + 2, top + height + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _process(model, reader, start, end, classes, settings, folder, deadline):
    import cv2
    from ultralytics.trackers.byte_tracker import BYTETracker

    first, stop = _lower_frame(reader, start), _lower_frame(reader, end)
    count = stop - first
    if count <= 0:
        raise TrackingError("NO_FRAMES", "İstenen aralıkta görüntü karesi yok.")
    if count > settings["max_frames"]:
        raise TrackingError("FRAME_LIMIT_EXCEEDED", "Aralık kare sınırını aşıyor; daha kısa bir aralık seçin.",
                            {"frame_count": count, "max_frames": settings["max_frames"]})
    first_time = max(start, float(reader.get_frame_timestamp(first)[0]))
    final_time = min(end, float(reader.get_frame_timestamp(stop - 1)[1]))
    duration = final_time - first_time
    if not math.isfinite(duration) or duration <= 0:
        raise TrackingError("INVALID_VIDEO", "Kaynak kare zamanları geçersiz.")
    fps = count / duration
    tracker = BYTETracker(SimpleNamespace(track_high_thresh=settings["confidence"],
                          track_low_thresh=0.1, new_track_thresh=settings["confidence"],
                          track_buffer=30, match_thresh=0.8, fuse_score=True), frame_rate=fps)
    tracker.reset()
    names = model.names
    saturated = False
    detections_count = 0
    details, intervals_path = folder / "frames.json", folder / "intervals.json"
    with details.open("w", encoding="utf-8") as frames, intervals_path.open("w", encoding="utf-8") as interval_file:
        frames.write("[\n")
        interval_file.write("[\n")
        intervals = _Intervals(interval_file)
        previous = None
        for position in range(first, stop):
            if time.monotonic() > deadline:
                raise TrackingError("TRACKING_TIMEOUT", "Nesne analizi süre sınırını aştı; tamamlandı sayılmadı.")
            timestamp, frame_end = map(float, reader.get_frame_timestamp(position))
            if not math.isfinite(timestamp) or not math.isfinite(frame_end) or frame_end <= timestamp:
                raise TrackingError("INVALID_VIDEO", "Geçersiz kare zaman damgası.")
            if previous is not None:
                if timestamp <= previous:
                    raise TrackingError("INVALID_VIDEO", "Kare zamanları artan sırada değil.")
            previous = timestamp
            frame = cv2.cvtColor(reader[position].asnumpy(), cv2.COLOR_RGB2BGR)
            result = model.predict(frame, conf=0.1, iou=0.7, imgsz=settings["imgsz"],
                                   classes=classes, max_det=settings["max_detections"],
                                   device=settings["device"], verbose=False, save=False)[0]
            boxes = result.boxes.cpu().numpy()
            tracks = tracker.update(boxes, frame)
            # ByteTrack indexes its high/low-confidence subsets separately.
            # Map high-confidence rows back to original detection indices;
            # a low-confidence row must not steal another box's identity.
            high_indices = [i for i, score in enumerate(boxes.conf) if score >= settings["confidence"]]
            identities = {high_indices[int(row[-1])]: int(row[4]) for row in tracks
                          if row[5] >= settings["confidence"]}
            saturated |= len(boxes) >= settings["max_detections"]
            detections, visible = [], {}
            for index, (box, score, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                if float(score) < settings["confidence"]:
                    continue
                label, identity = names[int(cls)], identities.get(index)
                xyxy = [round(float(value), 2) for value in box]
                detections.append({"class": label, "confidence": round(float(score), 4),
                                   "track_id": identity, "xyxy": xyxy})
                visible[("class", label)] = {"kind": "class", "class": label, "track_id": None}
                if identity is not None:
                    visible[("track", identity, label)] = {"kind": "track", "class": label, "track_id": identity}
            detections_count += len(detections)
            intervals.update(visible, max(start, timestamp), min(end, frame_end))
            if position > first:
                frames.write(",\n")
            json.dump({"frame_index": position, "source_sec": timestamp,
                       "detections": detections}, frames, ensure_ascii=False, allow_nan=False)
        intervals.finish()
        frames.write("\n]")
        interval_file.write("\n]")
    warnings = [{"code": "TRACK_ID_SCOPE", "message": "Takip kimliği yalnız bu analiz içindir; örtüşme/kaybolma ID değişimine neden olabilir."}]
    if intervals.count > _PREVIEW_LIMIT:
        warnings.append({"code": "SUMMARY_TRUNCATED", "message": "Sınıf aralıkları öncelikli en çok 100 aralık özetlendi; tamamı intervals.json dosyasındadır."})
    if saturated:
        warnings.append({"code": "DETECTION_LIMIT_REACHED", "message": "Bazı karelerde nesne sayısı sınırına ulaşıldı; ek nesneler kaçırılmış olabilir."})
    files = [details, intervals_path]
    return {"processed_frame_count": count, "detection_count": detections_count,
            "interval_count": intervals.count, "intervals": intervals.preview,
            "intervals_truncated": intervals.count > _PREVIEW_LIMIT,
            "sampled_range": {"start_sec": first_time, "end_sec": final_time},
            "output_fps": None}, warnings, files


def _frame_records(path: Path):
    """Read our canonical one-record-per-line JSON array without loading it all."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text and text not in {"[", "]"}:
                yield json.loads(text.rstrip(","))


def _render(source: Path, data: dict, folder: Path, deadline: float) -> Path:
    """Draw cached detections; encode each frame at its source presentation time."""
    import av
    import cv2
    from decord import VideoReader, cpu

    raw = folder / "timed.mp4"
    destination = folder / "annotated.mp4"
    start, end = data["sampled_range"]["start_sec"], data["sampled_range"]["end_sec"]
    time_base = Fraction(1, 1_000_000)
    durations = {}
    expected_pts = []
    reader = VideoReader(str(source), ctx=cpu(0), num_threads=1)
    try:
        with av.open(str(raw), mode="w") as container:
            # Nominal rate is only a codec hint; explicit PTS/duration determine playback.
            nominal_rate = Fraction(data["processed_frame_count"] / (end - start)).limit_denominator(100_000)
            stream = container.add_stream("libx264", rate=nominal_rate)
            stream.time_base = time_base
            stream.codec_context.time_base = time_base
            stream.codec_context.max_b_frames = 0
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "veryfast", "crf": "20", "tune": "zerolatency"}

            def mux(packets):
                for packet in packets:
                    ticks = round(float(packet.pts * packet.time_base / time_base))
                    if ticks not in durations:
                        raise TrackingError("INVALID_OUTPUT", "Kodlayıcı kare zamanını değiştirdi.")
                    packet.duration = round(durations.pop(ticks) * time_base / packet.time_base)
                    container.mux(packet)

            records = iter(_frame_records(Path(data["details_path"])))
            current = next(records, None)
            while current is not None:
                if time.monotonic() > deadline:
                    raise TrackingError("TRACKING_TIMEOUT", "Kutulu video çizimi süre sınırını aştı.")
                following = next(records, None)
                source_time = max(start, current["source_sec"])
                next_time = following["source_sec"] if following is not None else end
                pts = round((source_time - start) / time_base)
                stop_pts = round((next_time - start) / time_base)
                if stop_pts <= pts or (expected_pts and pts <= expected_pts[-1]):
                    raise TrackingError("INVALID_VIDEO", "Kare zamanları kodlama için geçersiz.")
                frame = cv2.cvtColor(reader[current["frame_index"]].asnumpy(), cv2.COLOR_RGB2BGR)
                for detection in current["detections"]:
                    _draw_detection(frame, detection["xyxy"], detection["class"],
                                    detection["track_id"], detection["confidence"])
                height, width = frame.shape[:2]
                if height % 2 or width % 2:
                    frame = cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                if not expected_pts:
                    stream.height, stream.width = frame.shape[:2]
                video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                video_frame.pts, video_frame.time_base = pts, time_base
                durations[pts] = stop_pts - pts
                expected_pts.append(pts)
                mux(stream.encode(video_frame))
                current = following
            mux(stream.encode())
            if durations or len(expected_pts) != data["processed_frame_count"]:
                raise TrackingError("INVALID_OUTPUT", "Kodlanan kare sayısı tespit kaydıyla uyuşmuyor.")
        _encode_video(raw, destination, source, start, end - start, deadline - time.monotonic())
        # Verify every output timestamp, not merely overall duration/readability.
        actual_pts = []
        with av.open(str(destination)) as encoded:
            video = encoded.streams.video[0]
            if video.duration is None or abs(float(video.duration * video.time_base) - (end - start)) > 0.002:
                raise TrackingError("INVALID_OUTPUT", "Kutulu videonun son kare süresi korunamadı.")
            for packet in encoded.demux(video):
                if packet.pts is not None:
                    actual_pts.append(round(float(packet.pts * packet.time_base / time_base)))
        if len(actual_pts) != len(expected_pts) or any(abs(a - b) > 1000 for a, b in zip(actual_pts, expected_pts)):
            raise TrackingError("INVALID_OUTPUT", "Kutulu videoda kaynak kare zamanları korunamadı.")
        raw.unlink()
        return destination
    finally:
        del reader


def _publish(folder: Path, index: Path, data: dict, warnings: list,
             artifacts: dict[str, str], dependencies=()) -> tuple[dict, list]:
    """Publish a detection or render result, with independent integrity checks."""
    destination = OUTPUT_ROOT / f"{index.stem[:12]}-{uuid.uuid4().hex[:12]}"
    data = {**data, **{key: str(destination / name) for key, name in artifacts.items()}}
    entries = [{"path": str(destination / name), "size": (folder / name).stat().st_size,
                "sha256": _digest(folder / name)} for name in artifacts.values()]
    entries.extend({"path": str(path), "size": path.stat().st_size, "sha256": _digest(path)} for path in dependencies)
    record = {"result": {"data": data, "warnings": warnings}, "files": entries}
    serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
    (folder / "manifest.json").write_text(serialized, encoding="utf-8")
    (folder / "cache-index.json").write_text(serialized, encoding="utf-8")
    folder.rename(destination)
    try:
        os.replace(destination / "cache-index.json", index)
    except OSError:
        warnings = warnings + [{"code": "CACHE_INDEX_ERROR", "message": "Çıktılar tamamlandı ancak tekrar kullanım kaydı yazılamadı."}]
    return data, warnings


def track_objects(video_path: str, start: float, end: float,
                  classes: list[str] | None, annotate: bool) -> tuple[dict, list]:
    """Validated tool inputs enter here; all readers and tracking state are job-local."""
    settings = _settings()
    source, weights = Path(video_path).resolve(), Path(settings["model_path"])
    if not weights.is_file():
        raise TrackingError("OBJECT_MODEL_NOT_FOUND", "YOLO11 ağırlığı yok; README'deki nesne tespiti kurulumunu tamamlayın.",
                            {"model_path": str(weights)})
    try:
        version = importlib.metadata.version("ultralytics")
        importlib.metadata.version("lap")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TrackingError("OBJECT_DEPENDENCY_MISSING", "requirements-objects.txt paketlerini kurun.") from exc
    if annotate and not shutil.which("ffmpeg"):
        raise TrackingError("FFMPEG_NOT_FOUND", "Kutulu MP4 için FFmpeg gerekli; yalnız tespit için render_video=false kullanın.")
    from filelock import FileLock, Timeout
    from decord import VideoReader, cpu

    settings["device"] = _device(settings["device"])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not _MODEL_LOCK.acquire(timeout=settings["timeout_sec"]):
        raise TrackingError("TRACKING_BUSY", "Önceki nesne analizi devam ediyor.")
    reader = None
    try:
        with FileLock(str(OUTPUT_ROOT / ".tracking.lock"), timeout=settings["timeout_sec"]):
            deadline = time.monotonic() + settings["timeout_sec"]
            source_signature, weights_signature = _signature(source), _signature(weights)
            requested_classes = sorted({name.strip().casefold() for name in classes}) if classes is not None else None
            key = hashlib.sha256(json.dumps([_REVISION, source_signature, weights_signature, version,
                                            settings, start, end, requested_classes], sort_keys=True).encode()).hexdigest()
            index = OUTPUT_ROOT / f"{key}.json"
            cached = _cached(index)

            def check_inputs():
                if time.monotonic() > deadline:
                    raise TrackingError("TRACKING_TIMEOUT", "İşlem süre sınırını aştı; tamamlandı sayılmadı.")
                if _signature(source) != source_signature or _signature(weights) != weights_signature:
                    raise TrackingError("INPUT_CHANGED", "Video veya model analiz sırasında değişti; sonuç kullanılmadı.")

            detection_cache_hit = cached is not None
            if cached is None:
                model = _load_model(weights, settings["device"])
                selected = _class_ids(requested_classes, model.names)
                reader = VideoReader(str(source), ctx=cpu(0), num_threads=1)
                with tempfile.TemporaryDirectory(prefix=".working-", dir=OUTPUT_ROOT) as temporary:
                    folder = Path(temporary)
                    data, warnings, _ = _process(model, reader, start, end, selected, settings, folder, deadline)
                    del reader
                    reader = None
                    check_inputs()
                    data.update({"video_path": str(source), "effective_range": {"start_sec": start, "end_sec": end},
                                 "model": weights.name, "model_version": version,
                                 "tracker": "ByteTrack", "device": settings["device"],
                                 "settings": settings, "classes": [model.names[i] for i in selected] if selected is not None else list(model.names.values()),
                                 "annotated_video_path": None, "cache_hit": False, "detection_cache_hit": False})
                    data, warnings = _publish(folder, index, data, warnings,
                                              {"details_path": "frames.json", "intervals_path": "intervals.json"})
            else:
                data, warnings = cached["data"], cached["warnings"]
            check_inputs()
            if not annotate:
                return {**data, "cache_hit": detection_cache_hit, "detection_cache_hit": detection_cache_hit}, warnings

            render_index = OUTPUT_ROOT / f"{key}-render.json"
            rendered = _cached(render_index)
            if rendered is not None and rendered["data"].get("details_path") == data["details_path"]:
                check_inputs()
                return {**rendered["data"], "cache_hit": True, "detection_cache_hit": True}, rendered["warnings"]
            with tempfile.TemporaryDirectory(prefix=".working-render-", dir=OUTPUT_ROOT) as temporary:
                folder = Path(temporary)
                _render(source, data, folder, deadline)
                check_inputs()
                interval = data["sampled_range"]
                result = {**data, "cache_hit": False, "detection_cache_hit": detection_cache_hit,
                          "output_timing": "source_timestamps",
                          "output_fps": data["processed_frame_count"] / (interval["end_sec"] - interval["start_sec"])}
                return _publish(folder, render_index, result, warnings, {"annotated_video_path": "annotated.mp4"},
                                dependencies=[Path(data["details_path"]), Path(data["intervals_path"])])
    except Timeout as exc:
        raise TrackingError("TRACKING_BUSY", "Başka bir süreç nesne analizi yapıyor.") from exc
    finally:
        if reader is not None:
            del reader
        _MODEL_LOCK.release()
