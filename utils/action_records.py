"""Deterministic action ledger from this task's paired tool calls/results."""
import json
import os

from langchain_core.messages import AIMessage, ToolMessage

ACTION_TOOLS = frozenset({"save_video_segment", "archive_anomaly_clip", "detect_and_track_objects",
                          "detect_license_plate_regions", "read_license_plate_crops"})


def _same_path(left, right):
    return isinstance(left, str) and bool(left) and isinstance(right, str) and bool(right) and (
        os.path.realpath(left) == os.path.realpath(right))


def _text(value):
    return " ".join(str(value).split())[:1200]


def _summary(name, data):
    if name in {"save_video_segment", "archive_anomaly_clip"}:
        interval = data.get("saved_range") or {}
        category = f"kategori={_text(data.get('category'))}; " if name == "archive_anomaly_clip" else ""
        reused = "Mevcut kayıt kullanıldı" if data.get("cache_hit") else "Klip kaydedildi"
        return (f"{reused}; {category}{interval.get('start_sec')}–{interval.get('end_sec')} sn; "
                f"dosya={_text(data.get('output_path'))}")
    if name == "detect_and_track_objects":
        summary = f"Nesne takibi tamamlandı; karelerdeki toplam kutu={data.get('detection_count')}; tekil nesne sayısı değildir"
        if data.get("annotated_video_path"):
            summary += f"; kutulu video={_text(data['annotated_video_path'])}"
        return summary
    if name == "detect_license_plate_regions":
        return f"Plaka bölgesi taraması tamamlandı; kırpım={data.get('crop_count')}; kayıt={_text(data.get('details_path'))}"
    if not data.get("ocr_performed"):
        return "Okunacak kırpım bulunmadı; OCR çalıştırılmadı"
    return (f"OCR tamamlandı; işlenen kırpım={data.get('processed_crop_count')}, "
            f"eşik üstü okuma={data.get('read_count')}, belirsiz/okunamaz={data.get('uncertain_or_unreadable_count')}; "
            f"sonuçlar model tahminidir; kayıt={_text(data.get('details_path'))}")


def action_records(messages, video_path):
    """Ignore unrelated videos, unpaired messages and never-executed pending calls."""
    calls, manifests, records, seen = {}, set(), [], set()
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls[call["id"]] = call
            continue
        if not isinstance(message, ToolMessage) or message.tool_call_id in seen:
            continue
        call = calls.get(message.tool_call_id)
        if not call or call["name"] not in ACTION_TOOLS or message.name not in (None, call["name"]):
            continue
        args = call.get("args") or {}
        name = call["name"]
        relevant = _same_path(args.get("video_path"), video_path)
        if name == "read_license_plate_crops":
            relevant = any(_same_path(args.get("crops_manifest_path"), p) for p in manifests)
        if not relevant:
            continue
        seen.add(message.tool_call_id)
        try:
            result = json.loads(message.content)
        except (ValueError, TypeError):
            result = {}
        if not isinstance(result, dict):
            result = {}
        data = result.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        success = result.get("ok") is True and _same_path(data.get("video_path"), video_path)
        if success and name == "detect_license_plate_regions" and isinstance(data.get("details_path"), str):
            manifests.add(data["details_path"])
        if success:
            label, summary = "BASARILI", _summary(name, data)
        else:
            error = result.get("error") or {}
            error_code = error.get("code") if isinstance(error, dict) else None
            label, summary = "BASARISIZ", f"İşlem tamamlanmadı: {_text(error_code or 'çalıştırılmadı/geçerli sonuç yok')}"
        records.append(f"[{label}] {name} ({_text(message.tool_call_id)}): {summary}")
    return records


def action_instructions(messages, video_path):
    records = action_records(messages, video_path)
    return ("\nBu görevin gerçek tool sonuçlarından üretilmiş eylem kayıtları:\n"
            + json.dumps(records, ensure_ascii=False)
            + "\nNihai eylemler listesine bu kayıtların tümünü aynen al. Ek kayıt yalnız "
              "[ONERI] ile başlayan, henüz uygulanmamış bir öneri olabilir. Başarı/başarısızlık uydurma.")
