"""Read existing plate crops locally; never rerun detection or alter originals."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import tempfile
import time
from pathlib import Path
from threading import Lock

from utils.env import env_float, env_get, env_int
from utils.plate_detection import OUTPUT_ROOT as CROP_ROOT, ROOT, PlateError

OUTPUT_ROOT = ROOT / "_stuff/lab_runs/actions/plate_ocr"
_LOCK = Lock()
_model = None
_model_key = None
_CONTRACT = {
    "max_plate_slots": 10,
    "alphabet": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_",
    "pad_char": "_",
    "img_height": 64,
    "img_width": 128,
    "keep_aspect_ratio": False,
    "interpolation": "linear",
    "image_color_mode": "rgb",
}


def _signature(path):
    stat = path.stat()
    return (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _settings():
    try:
        threshold = env_float("PLATE_OCR_MIN_CONFIDENCE", 0.8)
        limit = env_int("PLATE_OCR_MAX_CROPS", 500)
        timeout = env_float("PLATE_OCR_TIMEOUT_SEC", 120)
        if not 0 < threshold <= 1 or not 0 < limit <= 5000 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Geçersiz eşik, kırpım sınırı veya süre.")
    except (ValueError, TypeError) as exc:
        raise PlateError("INVALID_OCR_CONFIG", str(exc)) from exc
    return threshold, limit, timeout


def _load_model():
    global _model, _model_key
    import onnxruntime as ort
    import yaml

    paths = []
    for name, default in (
        ("PLATE_OCR_MODEL_PATH", "_stuff/models/cct_xs_v2_global.onnx"),
        ("PLATE_OCR_CONFIG_PATH", "_stuff/models/cct_xs_v2_global_plate_config.yaml"),
    ):
        path = Path(env_get(name, default))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise PlateError("OCR_MODEL_MISSING", "OCR modeli/yapılandırması yok; README kurulumunu uygulayın.")
        paths.append(path.resolve())
    key = tuple(_signature(path) for path in paths)
    if _model is None or _model_key != key:
        if paths[1].stat().st_size > 65536:
            raise PlateError("INVALID_OCR_MODEL", "OCR yapılandırması beklenenden büyük.")
        config = yaml.safe_load(paths[1].read_text(encoding="utf-8"))
        if not isinstance(config, dict) or any(config.get(k) != v for k, v in _CONTRACT.items()):
            raise PlateError("INVALID_OCR_MODEL", "CCT XS v2 global modeline uygun yapılandırma gerekli.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(paths[0]), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = {item.name: item for item in session.get_outputs()}
        if (len(inputs) != 1 or inputs[0].type != "tensor(uint8)"
                or len(inputs[0].shape) != 4 or inputs[0].shape[1:] != [64, 128, 3]
                or (isinstance(inputs[0].shape[0], int) and inputs[0].shape[0] != 1)
                or "plate" not in outputs):
            raise PlateError("INVALID_OCR_MODEL", "OCR ONNX giriş/çıkış sözleşmesi uyumsuz.")
        if key != tuple(_signature(path) for path in paths):
            raise PlateError("OCR_SOURCE_CHANGED", "OCR modeli yüklenirken değişti.")
        # Replace the working model only after the entire candidate is validated.
        _model = (session, inputs[0].name, paths)
        _model_key = key
    return _model


def _decode(raw, threshold):
    import numpy as np

    scores = np.asarray(raw)
    if (scores.shape not in {(1, 370), (1, 10, 37)} or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()):
        raise PlateError("INVALID_OCR_OUTPUT", "OCR çıktısı geçerli karakter olasılıkları değil.")
    scores = scores.reshape(10, 37)
    if not np.allclose(scores.sum(axis=1), 1, atol=1e-3):
        raise PlateError("INVALID_OCR_OUTPUT", "OCR çıktısındaki olasılıklar normalize değil.")
    ids = scores.argmax(axis=1)
    text = "".join(_CONTRACT["alphabet"][index] for index in ids).rstrip("_")
    probabilities = scores.max(axis=1)
    minimum = float(probabilities.min())
    if not text or "_" in text:
        status = "unreadable"
    else:
        status = "read" if minimum >= threshold else "uncertain"
    return {"status": status, "text": text if status == "read" else None,
            "candidate_text": text or None, "min_slot_confidence": minimum,
            "mean_character_confidence": float(probabilities[:len(text)].mean()) if text else None,
            "slot_confidences": [float(value) for value in probabilities]}


def _recognize(session, input_name, bgr, threshold):
    import cv2
    import numpy as np

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Matching official config: NHWC uint8, linear stretch, normalization in ONNX.
    tensor = np.ascontiguousarray(cv2.resize(rgb, (128, 64), interpolation=cv2.INTER_LINEAR)[None])
    return _decode(session.run(["plate"], {input_name: tensor})[0], threshold)


def _load_manifest(manifest_path, limit):
    path = Path(manifest_path).resolve()
    if (path.name != "crops.json" or path.parent.parent != CROP_ROOT.resolve()
            or not path.parent.name.startswith("plates-")):
        raise PlateError("INVALID_CROP_MANIFEST", "Plaka tespit tool'unun details_path çıktısı gerekli.")
    if not path.is_file():
        raise PlateError("CROP_MANIFEST_MISSING", "Plaka kırpım kaydı bulunamadı.")
    signature = _signature(path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise PlateError("INVALID_CROP_MANIFEST", "Kırpım kaydı 2 MB sınırını aşıyor.")
    try:
        with path.open("rb") as handle:
            payload = handle.read(2 * 1024 * 1024 + 1)
        if len(payload) > 2 * 1024 * 1024:
            raise ValueError("Kırpım kaydı 2 MB sınırını aşıyor.")
        data = json.loads(payload)
        crops = data["crops"]
        start, end = data["effective_range"]["start_sec"], data["effective_range"]["end_sec"]
        if (not isinstance(crops, list) or type(data["crop_count"]) is not int
                or data["crop_count"] != len(crops) or data.get("ocr_performed") is not False
                or not isinstance(data["video_path"], str)
                or not all(type(t) in (int, float) and math.isfinite(t) for t in (start, end))
                or not 0 <= start < end):
            raise ValueError("Kırpım kaydı alanları uyumsuz.")
        if len(crops) > limit:
            raise PlateError("OCR_CROP_LIMIT", "OCR kırpım sınırı aşıldı; daha kısa aralıkla tespit yapın.")
        seen = set()
        for crop in crops:
            target = Path(crop["crop_path"]).resolve()
            if target.parent != path.parent or target.suffix.lower() != ".png" or target in seen:
                raise ValueError("Kırpım yolu kayıt klasörü dışında veya tekrar ediyor.")
            seen.add(target)
            timestamp = crop["source_sec"]
            box = crop["bbox_xyxy"]
            if (type(timestamp) not in (int, float) or not math.isfinite(timestamp) or not start <= timestamp < end
                    or type(crop["frame_index"]) is not int or crop["frame_index"] < 0
                    or type(crop["width"]) is not int or type(crop["height"]) is not int
                    or not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                    or not 0 <= box[0] < box[2] or not 0 <= box[1] < box[3]
                    or crop["width"] != box[2] - box[0] or crop["height"] != box[3] - box[1]
                    or type(crop["confidence"]) not in (int, float) or not 0 <= crop["confidence"] <= 1):
                raise ValueError("Kırpım zaman/koordinat/boyut/güven alanları geçersiz.")
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        if isinstance(exc, PlateError):
            raise
        raise PlateError("INVALID_CROP_MANIFEST", f"Kırpım kaydı okunamadı: {exc}") from exc
    if signature != _signature(path):
        raise PlateError("OCR_SOURCE_CHANGED", "Kırpım kaydı okunurken değişti.")
    return path, signature, data


def read_plate_crops(manifest_path: str):
    import cv2
    import numpy as np

    threshold, limit, timeout = _settings()
    with _LOCK:
        began = time.monotonic()
        manifest, signature, source = _load_manifest(manifest_path, limit)
        crops = source["crops"]
        results, snapshots = [], []
        missing_hash = False
        model_paths, model_signature = [], ()
        if crops:
            session, input_name, model_paths = _load_model()
            model_signature = tuple(_signature(path) for path in model_paths)
        for crop in crops:
            if time.monotonic() - began > timeout:
                raise PlateError("OCR_TIMEOUT", "OCR süre sınırını aştı.")
            path = Path(crop["crop_path"]).resolve()
            if path.parent != manifest.parent:
                raise PlateError("INVALID_CROP_MANIFEST", "Kırpım yolu işlem sırasında klasör dışına yönlendirildi.")
            if not path.is_file():
                raise PlateError("OCR_CROP_MISSING", "Kayıttaki bir plaka kırpımı bulunamadı.")
            before = _signature(path)
            if path.stat().st_size > 16 * 1024 * 1024 or crop["width"] * crop["height"] > 16_000_000:
                raise PlateError("OCR_CROP_TOO_LARGE", "Plaka kırpımı boyut sınırını aşıyor.")
            # Do not let a replaced PNG claim a tiny size in the manifest while
            # allocating a huge decoded image. Header checks precede imdecode.
            with path.open("rb") as handle:
                blob = handle.read(16 * 1024 * 1024 + 1)
            if (len(blob) > 16 * 1024 * 1024 or len(blob) < 24
                    or blob[:8] != b"\x89PNG\r\n\x1a\n" or blob[12:16] != b"IHDR"
                    or struct.unpack(">II", blob[16:24]) != (crop["width"], crop["height"])):
                raise PlateError("OCR_INVALID_CROP", "PNG başlığı/boyutu kırpım kaydıyla uyumsuz.")
            digest = hashlib.sha256(blob).hexdigest()
            expected = crop.get("sha256")
            if expected is not None and expected != digest:
                raise PlateError("OCR_CROP_CHANGED", "Plaka kırpımı tespitten sonra değiştirilmiş.")
            missing_hash |= expected is None
            image = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (crop["height"], crop["width"]):
                raise PlateError("OCR_INVALID_CROP", "Plaka kırpımı bozuk veya boyutları kayda uymuyor.")
            result = _recognize(session, input_name, image, threshold)
            if _signature(path) != before:
                raise PlateError("OCR_SOURCE_CHANGED", "Plaka kırpımı okunurken değişti.")
            snapshots.append((path, before))
            results.append({"crop_path": str(path), "source_sec": crop["source_sec"],
                            "frame_index": crop["frame_index"], "bbox_xyxy": crop["bbox_xyxy"],
                            "detection_confidence": crop["confidence"], "crop_sha256": digest, **result})
        if time.monotonic() - began > timeout:
            raise PlateError("OCR_TIMEOUT", "OCR süre sınırını aştı.")
        if (signature != _signature(manifest) or any(_signature(p) != s for p, s in snapshots)
                or model_signature != tuple(_signature(path) for path in model_paths)):
            raise PlateError("OCR_SOURCE_CHANGED", "OCR sırasında kaynak veya model değişti.")
        warnings = [{"code": "OCR_REQUIRES_VERIFICATION", "message": "OCR model tahminidir; yüksek güven doğruluk garantisi değildir. Kırpımı gözle doğrulayın."}]
        if missing_hash:
            warnings.append({"code": "LEGACY_CROPS_UNVERIFIED", "message": "Eski kırpım kaydında hash yok; tespitten beri değişmediği doğrulanamadı."})
        if not crops:
            warnings.append({"code": "NO_CROPS", "message": "Okunacak plaka kırpımı yok; OCR çalıştırılmadı."})
        uncertain = sum(r["status"] != "read" for r in results)
        if uncertain:
            warnings.append({"code": "UNCERTAIN_OCR", "message": f"{uncertain} kırpım belirsiz/okunamaz; candidate_text kesin plaka değildir."})
        if len(results) > 30:
            warnings.append({"code": "PREVIEW_TRUNCATED", "message": "İlk 30 okuma gösterildi; tümü details_path dosyasında."})
        data = {"video_path": source["video_path"], "effective_range": source["effective_range"],
                "source_manifest_path": str(manifest), "model": "cct-xs-v2-global-model",
                "model_path": str(model_paths[0]) if model_paths else None,
                "model_config_path": str(model_paths[1]) if model_paths else None,
                "provider": "CPUExecutionProvider", "min_confidence_threshold": threshold,
                "ocr_performed": bool(crops), "processed_crop_count": len(results),
                "read_count": len(results) - uncertain, "uncertain_or_unreadable_count": uncertain,
                "results": results, "warnings": warnings}
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="ocr-", dir=OUTPUT_ROOT))
        try:
            target = output / "readings.json"
            data["details_path"] = str(target)
            target.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        except Exception:
            shutil.rmtree(output)
            raise
        return {**data, "results": results[:30], "results_truncated": len(results) > 30}, warnings
