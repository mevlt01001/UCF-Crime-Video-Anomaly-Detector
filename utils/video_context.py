"""Per-job technical context, not a tool result or persistent chat memory."""
import json
import math


def prepare_video_context(video_path):
    if not video_path:
        return {"status": "no_video", "video_path": None, "metadata": None, "error": None}
    try:
        # Reuse the file-signature-aware metadata cache; no model/API call.
        from utils.tools import _get_video_metadata

        raw = _get_video_metadata(video_path)
        metadata = {key: raw[key] for key in ("duration_sec", "fps", "frame_count", "width", "height")}
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in metadata.values()):
            raise ValueError("Invalid video metadata")
        return {"status": "ready", "video_path": video_path, "metadata": metadata, "error": None}
    except Exception as exc:
        # A bad video must not prevent unrelated conversation. Tools still validate.
        return {"status": "unavailable", "video_path": video_path, "metadata": None,
                "error": {"code": getattr(exc, "code", type(exc).__name__),
                          "message": "Video teknik bilgileri okunamadı; süre bilinmiyor."}}


def video_context_prompt(context, video_path):
    if not context or context.get("video_path") != (video_path or None):
        return ""
    return (
        "\n\nGörev başlangıcında kodun hazırladığı hedef video teknik bilgisi:\n"
        + json.dumps(context, ensure_ascii=False, allow_nan=False)
        + "\nBu yalnız teknik bilgidir; görsel analiz/olay kanıtı veya yapılmış eylem değildir. "
          "Bilgi okunamamışsa süreyi sıfır veya tahmini bir sayı kabul etme. "
          "Video gerektirmeyen sohbet devam edebilir; video işlemleri kendi güncel doğrulamalarını yapar."
    )
