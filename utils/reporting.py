"""Rapor modunun çıktı sözleşmesi; sohbet veya model istemcisi içermez."""
import json
import math

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from utils.action_records import action_records


class ReportEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    saniye: float = Field(ge=0, allow_inf_nan=False)
    aciklama: str = Field(min_length=1)


class VideoReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    ozet: str = Field(min_length=1)
    olaylar: list[ReportEvent]
    risk_seviyesi: Literal["dusuk", "orta", "yuksek"]
    eylemler: list[str]


REPORT_TASK = "Videodaki anormal aralıkları bul, bu aralıkları görsel olarak incele ve kanıta dayalı olay raporu oluştur."

REPORT_RULES = """
Bu görev bağımsız video raporudur; kullanıcıyla takip sohbeti yoktur.
Anormal aralıkları tespit et; bulunan aralıkların tümünü görsel olarak incele.
Executor'ın nihai raporu yalnız verilen şemaya uygun JSON olmalıdır; Markdown kullanma.
Planner ve reviewer kendi plan/denetim şemalarını kullanmaya devam eder.
ozet: İncelenen kapsamı, önemli bulguları, belirsizlikleri ve kısa risk gerekçesini yaz.
olaylar: Yalnız görsel kanıtla desteklenen olayları yaz. saniye kaynak videoya aittir;
kesin olay anı bilinmiyorsa incelenen kesitin başlangıcını kullan ve açıklamada
bunun kesit başlangıcı olduğunu, kesitin aralığını belirt. Zaman uydurma.
Risk anomali skoru değildir. dusuk: incelenen kesitlerde belirgin zarar tehdidi yok;
orta: potansiyel tehlike var ama yakın/ciddi zarar açık değil;
yuksek: aktif saldırı veya yakın/ciddi zarar tehlikesini destekleyen bulgular var.
Birden fazla olayda en yüksek kanıtla desteklenen risk seviyesini seç.
Anomali bulunmaması tüm videonun güvenli olduğunu kanıtlamaz; kapsamı açıkla.
Yetersiz görüntüyü veya başarısız analizi düşük risk diye sunma. Gerekli analiz
tamamlanamıyorsa bunu belirt; tamamlanmış rapor uydurma.
Görsel bulgulara uygun ve yararlı yerel eylemleri araç kataloğundan seç;
her aracı sırf mevcut diye çalıştırma. Kişi/araç kimliği veya dışa aktarım yapma.
eylemler bir metin listesidir: gerçek kayıtlar [BASARILI] / [BASARISIZ],
henüz uygulanmamış öneriler [ONERI] ile başlar. Kodun verdiği gerçek kayıtları
aynen koru; gerçekleşmeyen işlemi yapılmış gibi yazma. Öneri geleceğe yöneliktir,
kendisi işlem çalıştırmaz. Ek eylem yoksa boş liste geçerlidir. Eylem hatası,
zorunlu görsel analiz tamamlandıysa tek başına raporu engellemez; hatayı açıkla.
"""


def report_instructions() -> str:
    return REPORT_RULES + "\nJSON şeması:\n" + json.dumps(VideoReport.model_json_schema(), ensure_ascii=False)


def _unwrap_report_json(answer: str) -> str:
    """Yalnız bilinen sunum sarmallarını kaldırır; JSON içeriğini onarmaz.

    Açıklamalar arasından rastgele süslü parantez aramak yerine tek, açıkça
    sınırlandırılmış JSON kod bloğunu kabul eder. Birden fazla blok/nesne veya
    yarım kalmış çıktı hata olarak kalır.
    """
    text = answer.strip()
    if text.startswith(("{", "[")):
        return text

    lines = text.split("\n")
    fences = [index for index, line in enumerate(lines) if line.strip().startswith("```")]
    if fences:
        if (
            len(fences) != 2
            or lines[fences[0]].strip().lower() not in ("```", "```json")
            or lines[fences[1]].strip() != "```"
        ):
            raise ValueError("Rapor tek ve tamamlanmış bir JSON kod bloğu içermeli.")
        start, end = fences
        outside = "\n".join(lines[:start] + lines[end + 1:])
        if any(char in outside for char in "{}[]"):
            raise ValueError("JSON kod bloğu dışında başka bir JSON yapısı bulunamaz.")
        return "\n".join(lines[start + 1:end]).strip()

    # Bazı model yanıtları kod çiti olmadan yalnız 'json' dil etiketini ekler.
    if lines and lines[0].strip().lower() == "json":
        return "\n".join(lines[1:]).strip()
    return text


def validate_report(answer: str, messages, video_path: str) -> dict:
    """Biçim, kaynak süre ve görsel kapsam kontrolü; anlam denetimi reviewer'dadır."""
    from langchain_core.messages import ToolMessage

    report = VideoReport.model_validate_json(_unwrap_report_json(answer))
    if not report.ozet.strip() or any(not event.aciklama.strip() for event in report.olaylar):
        raise ValueError("Özet ve olay açıklamaları boş olamaz.")
    expected_actions = action_records(messages, video_path)
    actual_actions = []
    for entry in report.eylemler:
        if entry.startswith("[ONERI] ") and 0 < len(entry[8:].strip()) <= 2000:
            continue
        if entry.startswith(("[BASARILI] ", "[BASARISIZ] ")):
            actual_actions.append(entry)
        else:
            raise ValueError("Eylem yalnız doğrulanmış kayıt veya [ONERI] açıklaması olabilir.")
    if actual_actions != expected_actions:
        raise ValueError("Başarı/başarısızlık eylemleri gerçek tool kayıtlarıyla birebir ve aynı sırada eşleşmeli.")
    segments = None
    duration = None
    ranges = []
    for message in messages:
        if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            continue
        try:
            result = json.loads(message.content)
        except (ValueError, TypeError):
            continue
        if not isinstance(result, dict) or result.get("ok") is not True:
            continue
        data = result.get("data") or {}
        if data.get("video_path") != video_path:
            continue
        if "segments" in data and data.get("analysis_scope") == "full_video":
            segments = data["segments"]
            duration = (data.get("video") or {}).get("duration_sec")
        visual_text = data.get("vlm_response")
        if isinstance(visual_text, str) and visual_text.startswith("[video_url-mp4 "):
            visual_text = visual_text.partition("\n")[2]
        if isinstance(visual_text, str) and visual_text.strip() and "[VLM HATA]:" not in visual_text:
            interval = data.get("effective_range")
            if interval:
                ranges.append((interval["start_sec"], interval["end_sec"]))
    if segments is None or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Hedef video için başarılı tam-video segmentasyon sonucu ve geçerli süre gerekli.")
    for segment in segments:
        start, end = segment["start_time"], min(segment["end_time"], duration)
        cursor = start
        for left, right in sorted(ranges):
            if left <= cursor + 0.001:
                cursor = max(cursor, right)
        if cursor < end - 0.001:
            raise ValueError(f"{start}–{end} aralığının görsel incelemesi tamamlanmadı.")
    for event in report.olaylar:
        if event.saniye >= duration:
            raise ValueError("Olay zamanı kaynak video süresi dışında.")
        if not any(left <= event.saniye <= right for left, right in ranges):
            raise ValueError("Olay zamanı görsel olarak incelenen aralıkların dışında.")
    return report.model_dump()
