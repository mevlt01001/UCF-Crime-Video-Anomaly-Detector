"""Local plate-region detection only: no OCR, identity lookup or network calls."""
from __future__ import annotations

import json
import hashlib
import math
import shutil
import tempfile
import time
from pathlib import Path
from threading import Lock

from utils.env import env_float, env_get, env_int

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "_stuff/lab_runs/actions/plates"
_LOCK = Lock()
_session = None
_session_key = None


class PlateError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.data = code, {}


def _signature(path):
    stat = path.stat()
    return (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _load_model(path):
    global _session, _session_key
    import onnxruntime as ort

    key = _signature(path)
    if _session is None or key != _session_key:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        candidate = ort.InferenceSession(str(path), sess_options=options,
                                        providers=["CPUExecutionProvider"])
        inputs = candidate.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 384, 384]:
            raise PlateError("INVALID_PLATE_MODEL", "384px end2end plaka ONNX modeli gerekli.")
        _session, _session_key = candidate, key
    return _session


def _detect(session, rgb, threshold):
    """YOLOv9 end2end: RGB float NCHW; output [batch,x1,y1,x2,y2,class,score]."""
    import cv2
    import numpy as np

    height, width = rgb.shape[:2]
    ratio = min(384 / height, 384 / width)
    resized = (round(width * ratio), round(height * ratio))
    dx, dy = (384 - resized[0]) / 2, (384 - resized[1]) / 2
    padded = cv2.copyMakeBorder(
        cv2.resize(rgb, resized), round(dy - 0.1), round(dy + 0.1),
        round(dx - 0.1), round(dx + 0.1), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    rows = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    if rows.ndim != 2 or rows.shape[1] != 7 or not np.isfinite(rows).all():
        raise PlateError("INVALID_PLATE_OUTPUT", "Model çıktısı beklenen Nx7 sözleşmesine uymuyor.")
    detections = []
    for batch, x1, y1, x2, y2, category, score in rows:
        if batch != 0 or category != 0 or not 0 <= score <= 1:
            raise PlateError("INVALID_PLATE_OUTPUT", "Geçersiz plaka sınıfı veya güven skoru.")
        if score < threshold:
            continue
        x1 = max(0, min(width, math.floor((float(x1) - dx) / ratio)))
        y1 = max(0, min(height, math.floor((float(y1) - dy) / ratio)))
        x2 = max(0, min(width, math.ceil((float(x2) - dx) / ratio)))
        y2 = max(0, min(height, math.ceil((float(y2) - dy) / ratio)))
        if x2 > x1 and y2 > y1:
            detections.append({"bbox_xyxy": [x1, y1, x2, y2], "confidence": float(score)})
    return detections


def _lower_frame(reader, second):
    left, right = 0, len(reader)
    while left < right:
        middle = (left + right) // 2
        # Decord timestamps are float32; normalize sub-microsecond noise.
        if round(float(reader.get_frame_timestamp(middle)[0]), 6) < second:
            left = middle + 1
        else:
            right = middle
    return left


def extract_plate_crops(video_path: str, start: float, end: float):
    """Process every frame in [start,end), save original-resolution PNG regions."""
    import cv2
    from decord import VideoReader, cpu

    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise PlateError("INVALID_TIME_RANGE", "Geçerli bir başlangıç/bitiş aralığı gerekli.")
    path = Path(env_get("PLATE_MODEL_PATH", "_stuff/models/yolo-v9-t-384-license-plates-end2end.onnx"))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise PlateError("PLATE_MODEL_MISSING", "Plaka modeli yok; README kurulum adımlarını uygulayın.")
    threshold = env_float("PLATE_CONFIDENCE", 0.25)
    max_frames = env_int("PLATE_MAX_FRAMES", 1800)
    max_crops = env_int("PLATE_MAX_CROPS", 500)
    timeout = env_float("PLATE_TIMEOUT_SEC", 300)
    if not 0 < threshold <= 1 or max_frames <= 0 or max_crops <= 0 or not math.isfinite(timeout) or timeout <= 0:
        raise PlateError("INVALID_PLATE_CONFIG", "PLATE_* ayarları geçersiz.")
    with _LOCK:
        began = time.monotonic()
        source = Path(video_path).resolve()
        signature, model_signature = _signature(source), _signature(path)
        reader = VideoReader(str(source), ctx=cpu(0), num_threads=1)
        output = None
        try:
            first, stop = _lower_frame(reader, start), _lower_frame(reader, end)
            if first == stop:
                raise PlateError("NO_FRAMES", "İstenen aralıkta kare başlangıcı bulunamadı.")
            if stop - first > max_frames:
                raise PlateError("PLATE_FRAME_LIMIT", "Kare sınırı aşıldı; daha kısa aralık seçin.")
            session = _load_model(path)
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix="plates-", dir=OUTPUT_ROOT))
            crops = []
            for index in range(first, stop):
                if time.monotonic() - began > timeout:
                    raise PlateError("PLATE_TIMEOUT", "Plaka tespiti süre sınırını aştı.")
                rgb = reader[index].asnumpy()
                timestamp = round(float(reader.get_frame_timestamp(index)[0]), 6)
                for detection in _detect(session, rgb, threshold):
                    if len(crops) >= max_crops:
                        raise PlateError("PLATE_CROP_LIMIT", "Kırpım sınırı aşıldı; daha kısa aralık seçin.")
                    x1, y1, x2, y2 = detection["bbox_xyxy"]
                    target = output / f"frame_{index:08d}_plate_{len(crops):04d}.png"
                    crop = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
                    if not cv2.imwrite(str(target), crop) or target.stat().st_size == 0:
                        raise PlateError("PLATE_WRITE_ERROR", "Plaka görüntüsü kaydedilemedi.")
                    crops.append({**detection, "frame_index": index, "source_sec": timestamp,
                                  "crop_path": str(target), "width": x2 - x1, "height": y2 - y1,
                                  "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            if time.monotonic() - began > timeout:
                raise PlateError("PLATE_TIMEOUT", "Plaka tespiti süre sınırını aştı.")
            if signature != _signature(source) or model_signature != _signature(path):
                raise PlateError("SOURCE_CHANGED", "Video veya model işlem sırasında değişti.")
            manifest = output / "crops.json"
            data = {"video_path": str(source), "effective_range": {"start_sec": start, "end_sec": end},
                    "processed_frame_count": stop - first, "crop_count": len(crops),
                    "model": path.name, "provider": "CPUExecutionProvider", "confidence_threshold": threshold,
                    "ocr_performed": False, "details_path": str(manifest), "crops": crops}
            manifest.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
            warnings = [{"code": "DETECTION_ONLY", "message": "Plaka adaylarıdır; metin okunmadı. Aynı plaka farklı karelerde tekrarlanabilir."}]
            if not crops:
                warnings.append({"code": "NO_PLATE_DETECTED", "message": "Plaka adayı bulunamadı; bu, plaka bulunmadığını kanıtlamaz."})
            if len(crops) > 30:
                warnings.append({"code": "PREVIEW_TRUNCATED", "message": "İlk 30 kırpım gösterildi; tümü details_path dosyasında."})
            return {**data, "crops": crops[:30], "crops_truncated": len(crops) > 30}, warnings
        except Exception:
            if output is not None:
                shutil.rmtree(output)
            raise
        finally:
            del reader
